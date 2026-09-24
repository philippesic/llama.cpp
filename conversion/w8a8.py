"""CPU contract for the opt-in EAGLE-3 signed W8A8 draft format."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

QMAX = 127


def _quantize_last_dim(values: Tensor) -> tuple[Tensor, Tensor]:
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("W8A8 values must have a nonempty last dimension")
    if values.device.type != "cpu":
        raise ValueError("W8A8 reference quantization is CPU only")
    source = values.float()
    if not torch.isfinite(source).all():
        raise ValueError("W8A8 values contain nonfinite values")
    scales = source.abs().amax(dim=-1, keepdim=True) / QMAX
    divisor = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.round(source / divisor).clamp(-QMAX, QMAX)
    codes = torch.where(scales == 0, torch.zeros_like(codes), codes)
    return codes.to(torch.int8), scales


def quantize_w8a8_weights(
    weight: Tensor, rows_per_chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous signed I8 [rows, K] and F32 [rows] absmax scales.

    The source BF16 weight is converted to F32 before scale and code generation,
    matching the project's PyTorch W8A8 acceptance simulation.
    """
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or min(weight.shape) == 0:
        raise ValueError("W8A8 EAGLE weights must be a nonempty 2D BF16 tensor")
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    rows, k = weight.shape
    codes = np.empty((rows, k), dtype=np.int8)
    scales = np.empty(rows, dtype=np.float32)
    for first in range(0, rows, rows_per_chunk):
        q, d = _quantize_last_dim(weight[first : first + rows_per_chunk].cpu())
        codes[first : first + len(q)] = q.numpy()
        scales[first : first + len(q)] = d.squeeze(-1).numpy()
    return codes, scales


def quantize_w8a8_activations(values: Tensor) -> tuple[Tensor, Tensor]:
    """Return signed I8 codes and one F32 scale per input token/vector."""
    return _quantize_last_dim(values)


def w8a8_reference_linear(
    values: Tensor,
    weight_codes: Tensor,
    weight_scales: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Integer-dot CPU oracle, with the simulation's F32 scale and output order."""
    if (
        values.device.type != "cpu"
        or weight_codes.device.type != "cpu"
        or weight_scales.device.type != "cpu"
    ):
        raise ValueError("W8A8 reference linear is CPU only")
    if weight_codes.ndim != 2 or weight_codes.dtype != torch.int8:
        raise ValueError("weight_codes must be a 2D I8 tensor")
    rows, k = weight_codes.shape
    if (
        values.ndim < 1
        or values.shape[-1] != k
        or weight_scales.shape != (rows,)
        or weight_scales.dtype != torch.float32
    ):
        raise ValueError("W8A8 operand dimensions or scales disagree")
    if k * QMAX * QMAX > torch.iinfo(torch.int32).max:
        raise ValueError("W8A8 reduction exceeds signed I32 accumulator range")
    if bias is not None and (bias.device.type != "cpu" or bias.shape != (rows,)):
        raise ValueError("bias must be a CPU vector with one value per output row")

    input_codes, input_scales = quantize_w8a8_activations(values)
    dots = torch.matmul(input_codes.to(torch.int32), weight_codes.to(torch.int32).T)
    output = dots.to(torch.float32) * weight_scales
    output = output * input_scales
    if bias is not None:
        output = output + bias.float()
    return output.to(values.dtype)
