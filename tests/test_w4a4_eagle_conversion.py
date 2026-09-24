"""CPU numerical and GGUF storage checks for packed EAGLE W4A4 export."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf

from conversion.llama import eagle3_w1a1_candidates
from conversion.w4a4 import (
    pack_signed_nibbles,
    quantize_w4a4_activations,
    quantize_w4a4_weights,
    unpack_signed_nibbles,
    w4a4_reference_linear,
)


class TestW4A4EagleConversion(unittest.TestCase):
    def test_signed_nibbles_odd_k_and_forbidden_codes(self):
        codes = np.array([[-7, -1, 0, 7, 2], [7, 0, -2, -7, 0]], dtype=np.int8)
        packed = pack_signed_nibbles(codes)
        self.assertEqual(packed.dtype, np.int8)
        self.assertEqual(packed.shape, (2, 3))
        self.assertEqual(packed[0].view(np.uint8).tolist(), [0xf9, 0x70, 0x02])
        np.testing.assert_array_equal(unpack_signed_nibbles(packed, 5), codes)
        with self.assertRaisesRegex(ValueError, "[-]7, 7"):
            pack_signed_nibbles(np.array([[-8]], dtype=np.int8))
        bad_padding = packed.copy()
        bad_padding[0, -1] = np.int8(0x12)
        with self.assertRaisesRegex(ValueError, "padding"):
            unpack_signed_nibbles(bad_padding, 5)
        with self.assertRaisesRegex(ValueError, "[-]8"):
            unpack_signed_nibbles(np.array([[8]], dtype=np.int8), 2)
        with self.assertRaisesRegex(ValueError, "logical K"):
            unpack_signed_nibbles(packed, 7)

    def test_nine_names_and_packed_gguf_roundtrip(self):
        candidates = eagle3_w1a1_candidates(
            SimpleNamespace(block_count=1, hf_arch="LlamaForCausalLM")
        )
        self.assertEqual(len(candidates), 9)
        self.assertEqual(
            {group for group, _ in candidates.values()}, {"fusion", "attention", "ffn", "head"}
        )
        weight = torch.tensor([[0, -2, 2, 7, 1], [0, 0, 0, 0, 0]], dtype=torch.bfloat16)
        packed, scales = quantize_w4a4_weights(weight, rows_per_chunk=1)
        self.assertEqual(scales.tolist(), [1.0, 0.0])
        np.testing.assert_array_equal(
            unpack_signed_nibbles(packed, 5), np.array([[0, -2, 2, 7, 1], [0] * 5])
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w4a4.gguf"
            writer = gguf.GGUFWriter(path, "eagle3")
            writer.add_uint32("eagle3.w4a4.version", 1)
            writer.add_array("eagle3.w4a4.tensors", sorted(x for _, x in candidates.values()))
            writer.add_string("eagle3.w4a4.nibble_order", "even_low_odd_high")
            for _, name in candidates.values():
                base = name.removesuffix(".weight")
                writer.add_uint32(f"eagle3.w4a4.tensor.{name.replace('.', '_')}.logical_k", 5)
                writer.add_tensor(
                    f"{base}.w4a4_packed", packed, raw_dtype=gguf.GGMLQuantizationType.I8
                )
                writer.add_tensor(
                    f"{base}.w4a4_scale", scales, raw_dtype=gguf.GGMLQuantizationType.F32
                )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            reader = gguf.GGUFReader(path)
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(len(tensors), 18)
            self.assertEqual(reader.get_field("eagle3.w4a4.version").contents(), 1)
            for _, name in candidates.values():
                base = name.removesuffix(".weight")
                self.assertEqual(reader.get_field(f"eagle3.w4a4.tensor.{name.replace('.', '_')}.logical_k").contents(), 5)
                self.assertEqual(tensors[f"{base}.w4a4_packed"].tensor_type, gguf.GGMLQuantizationType.I8)
                self.assertEqual(tensors[f"{base}.w4a4_packed"].shape.tolist(), [3, 2])
                np.testing.assert_array_equal(tensors[f"{base}.w4a4_packed"].data, packed)
                np.testing.assert_array_equal(tensors[f"{base}.w4a4_scale"].data, scales)

    def test_ties_zero_vectors_and_integer_reference(self):
        weight = torch.tensor(
            [[0.5, 1.5, 2.5, 7], [-0.5, -1.5, -2.5, -7]], dtype=torch.bfloat16
        )
        packed, scales = quantize_w4a4_weights(weight, rows_per_chunk=1)
        self.assertEqual(unpack_signed_nibbles(packed, 4).tolist(), [[0, 2, 2, 7], [0, -2, -2, -7]])
        self.assertEqual(scales.tolist(), [1.0, 1.0])

        values = torch.tensor([[0.5, 1.5, 2.5, 7], [0, 0, 0, 0]], dtype=torch.bfloat16)
        input_packed, input_scales = quantize_w4a4_activations(values)
        self.assertEqual(unpack_signed_nibbles(input_packed.numpy(), 4).tolist(), [[0, 2, 2, 7], [0] * 4])
        self.assertEqual(input_scales.tolist(), [[1.0], [0.0]])
        got = w4a4_reference_linear(values, torch.from_numpy(packed), torch.from_numpy(scales))
        self.assertEqual(got.tolist(), [[57.0, -57.0], [0.0, 0.0]])

    def test_integer_oracle_matches_prior_float_code_simulation(self):
        weight = torch.tensor(
            [[1, -0.75, 0, 0.25, 0.5], [-0.5, 0.5, -0.25, 0.75, 1]],
            dtype=torch.bfloat16,
        )
        values = torch.tensor(
            [[0.25, -0.5, 0.75, 1, 0], [-1, 0.5, -0.75, 0, 0.25]],
            dtype=torch.bfloat16,
        )
        packed, scales = quantize_w4a4_weights(weight)
        actual = w4a4_reference_linear(values, torch.from_numpy(packed), torch.from_numpy(scales))
        weight_scale = weight.float().abs().amax(dim=-1, keepdim=True) / 7
        input_scale = values.float().abs().amax(dim=-1, keepdim=True) / 7
        weight_codes = torch.round(weight.float() / weight_scale).clamp(-7, 7)
        input_codes = torch.round(values.float() / input_scale).clamp(-7, 7)
        simulated = (
            F.linear(input_codes, weight_codes) * weight_scale.squeeze(-1) * input_scale
        ).to(values.dtype)
        torch.testing.assert_close(actual, simulated, rtol=0, atol=0)

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "BF16"):
            quantize_w4a4_weights(torch.ones((2, 3), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            quantize_w4a4_weights(torch.tensor([[float("nan")]], dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "positive"):
            quantize_w4a4_weights(torch.zeros((1, 4), dtype=torch.bfloat16), rows_per_chunk=0)


if __name__ == "__main__":
    unittest.main()
