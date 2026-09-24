"""CPU contract for the opt-in EAGLE-3 signed W4A4 draft format."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

QMAX = 7


def pack_signed_nibbles(codes: np.ndarray) -> np.ndarray:
    """Pack signed [-7, 7] codes, even K in the low nibble, odd K in the high."""
    if codes.ndim < 1 or codes.shape[-1] == 0 or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("W4A4 codes must be a nonempty integer array")
    if np.any((codes < -QMAX) | (codes > QMAX)):
        raise ValueError("W4A4 codes must be in [-7, 7]")
    logical_k = codes.shape[-1]
    packed = np.zeros((*codes.shape[:-1], (logical_k + 1) // 2), dtype=np.uint8)
    packed |= (codes[..., 0::2] & 0x0f).astype(np.uint8)
    if logical_k > 1:
        packed[..., :logical_k // 2] |= ((codes[..., 1::2] & 0x0f) << 4).astype(np.uint8)
    return packed.view(np.int8)


def unpack_signed_nibbles(packed: np.ndarray, logical_k: int) -> np.ndarray:
    """Decode packed I8 storage and reject forbidden -8 codes or nonzero padding."""
    if packed.dtype != np.int8 or packed.ndim < 1 or logical_k < 1:
        raise ValueError("W4A4 packed data must be nonempty I8 with positive logical K")
    if packed.shape[-1] != (logical_k + 1) // 2:
        raise ValueError("W4A4 packed width disagrees with logical K")
    raw = packed.view(np.uint8)
    if logical_k % 2 and np.any(raw[..., -1] & 0xf0):
        raise ValueError("W4A4 high padding nibble must be zero")
    nibbles = np.empty((*packed.shape[:-1], logical_k), dtype=np.uint8)
    nibbles[..., 0::2] = raw & 0x0f
    if logical_k > 1:
        nibbles[..., 1::2] = raw[..., :logical_k // 2] >> 4
    if np.any(nibbles == 8):
        raise ValueError("W4A4 signed -8 code is forbidden")
    return np.where(nibbles >= 8, nibbles.astype(np.int16) - 16, nibbles).astype(np.int8)


def _quantize_last_dim(values: Tensor) -> tuple[Tensor, Tensor]:
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("W4A4 values must have a nonempty last dimension")
    if values.device.type != "cpu":
        raise ValueError("W4A4 reference quantization is CPU only")
    source = values.float()
    if not torch.isfinite(source).all():
        raise ValueError("W4A4 values contain nonfinite values")
    scales = source.abs().amax(dim=-1, keepdim=True) / QMAX
    divisor = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.round(source / divisor).clamp(-QMAX, QMAX)
    codes = torch.where(scales == 0, torch.zeros_like(codes), codes)
    return codes.to(torch.int8), scales


def quantize_w4a4_weights(
    weight: Tensor, rows_per_chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Return packed I8 [rows, ceil(K/2)] and F32 [rows] absmax/7 scales."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or min(weight.shape) == 0:
        raise ValueError("W4A4 EAGLE weights must be a nonempty 2D BF16 tensor")
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    rows, logical_k = weight.shape
    packed = np.empty((rows, (logical_k + 1) // 2), dtype=np.int8)
    scales = np.empty(rows, dtype=np.float32)
    for first in range(0, rows, rows_per_chunk):
        codes, row_scales = _quantize_last_dim(weight[first:first + rows_per_chunk].cpu())
        packed[first:first + len(codes)] = pack_signed_nibbles(codes.numpy())
        scales[first:first + len(codes)] = row_scales.squeeze(-1).numpy()
    return packed, scales


def quantize_w4a4_activations(values: Tensor) -> tuple[Tensor, Tensor]:
    """Return packed signed nibbles and one F32 absmax/7 scale per token."""
    codes, scales = _quantize_last_dim(values)
    return torch.from_numpy(pack_signed_nibbles(codes.numpy())), scales


def w4a4_reference_linear(
    values: Tensor,
    weight_packed: Tensor,
    weight_scales: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Exact I32 CPU dot, then F32 row scale, token scale, optional bias."""
    if any(x.device.type != "cpu" for x in (values, weight_packed, weight_scales)):
        raise ValueError("W4A4 reference linear is CPU only")
    if weight_packed.ndim != 2 or weight_packed.dtype != torch.int8:
        raise ValueError("weight_packed must be a 2D I8 tensor")
    rows, packed_k = weight_packed.shape
    if (values.ndim < 1 or values.shape[-1] < 1 or packed_k != (values.shape[-1] + 1) // 2
            or weight_scales.shape != (rows,) or weight_scales.dtype != torch.float32):
        raise ValueError("W4A4 operand dimensions or scales disagree")
    logical_k = values.shape[-1]
    if logical_k * QMAX * QMAX > torch.iinfo(torch.int32).max:
        raise ValueError("W4A4 reduction exceeds signed I32 accumulator range")
    if bias is not None and (bias.device.type != "cpu" or bias.shape != (rows,)):
        raise ValueError("bias must be a CPU vector with one value per output row")
    weight_codes = torch.from_numpy(unpack_signed_nibbles(weight_packed.numpy(), logical_k))
    activation_packed, activation_scales = quantize_w4a4_activations(values)
    activation_codes = torch.from_numpy(unpack_signed_nibbles(activation_packed.numpy(), logical_k))
    dots = torch.matmul(activation_codes.to(torch.int32), weight_codes.to(torch.int32).T)
    output = dots.to(torch.float32) * weight_scales
    output = output * activation_scales
    if bias is not None:
        output = output + bias.float()
    return output.to(values.dtype)
