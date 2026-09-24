"""CPU numerical and GGUF storage checks for the opt-in EAGLE W8A8 export."""

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

import gguf  # noqa: E402
from conversion.llama import eagle3_w1a1_candidates  # noqa: E402
from conversion.w8a8 import (  # noqa: E402
    quantize_w8a8_activations,
    quantize_w8a8_weights,
    w8a8_reference_linear,
)


class TestW8A8EagleConversion(unittest.TestCase):
    def test_all_nine_candidate_names_and_signed_i8_gguf_round_trip(self):
        candidates = eagle3_w1a1_candidates(
            SimpleNamespace(block_count=1, hf_arch="LlamaForCausalLM")
        )
        self.assertEqual(len(candidates), 9)
        self.assertEqual(
            {group for group, _ in candidates.values()}, {"fusion", "attention", "ffn", "head"}
        )
        weight = torch.tensor([[0, -2, 2, 127], [0, 0, 0, 0]], dtype=torch.bfloat16)
        codes, scales = quantize_w8a8_weights(weight, rows_per_chunk=1)
        self.assertEqual(codes.dtype, np.int8)
        self.assertEqual(scales.dtype, np.float32)
        self.assertEqual(codes.tolist(), [[0, -2, 2, 127], [0, 0, 0, 0]])
        self.assertEqual(scales.tolist(), [1.0, 0.0])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w8a8.gguf"
            writer = gguf.GGUFWriter(path, "eagle3")
            writer.add_uint32("eagle3.w8a8.version", 1)
            writer.add_array(
                "eagle3.w8a8.tensors", sorted(target for _, target in candidates.values())
            )
            for _, target in candidates.values():
                base = target.removesuffix(".weight")
                writer.add_tensor(
                    f"{base}.w8a8_codes", codes, raw_dtype=gguf.GGMLQuantizationType.I8
                )
                writer.add_tensor(
                    f"{base}.w8a8_scale", scales, raw_dtype=gguf.GGMLQuantizationType.F32
                )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            reader = gguf.GGUFReader(path)
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(len(tensors), 18)
            for _, target in candidates.values():
                base = target.removesuffix(".weight")
                actual_codes = tensors[f"{base}.w8a8_codes"]
                self.assertEqual(actual_codes.tensor_type, gguf.GGMLQuantizationType.I8)
                self.assertEqual(actual_codes.shape.tolist(), [4, 2])
                np.testing.assert_array_equal(actual_codes.data, codes)
                np.testing.assert_array_equal(tensors[f"{base}.w8a8_scale"].data, scales)
            self.assertEqual(reader.get_field("eagle3.w8a8.version").contents(), 1)

    def test_ties_zero_vectors_and_integer_reference(self):
        weight = torch.tensor(
            [[0.5, 1.5, 2.5, 127], [-0.5, -1.5, -2.5, -127]], dtype=torch.bfloat16
        )
        codes, scales = quantize_w8a8_weights(weight, rows_per_chunk=1)
        self.assertEqual(codes.tolist(), [[0, 2, 2, 127], [0, -2, -2, -127]])
        self.assertEqual(scales.tolist(), [1.0, 1.0])

        values = torch.tensor([[0.5, 1.5, 2.5, 127], [0, 0, 0, 0]], dtype=torch.bfloat16)
        input_codes, input_scales = quantize_w8a8_activations(values)
        self.assertEqual(input_codes.tolist(), [[0, 2, 2, 127], [0, 0, 0, 0]])
        self.assertEqual(input_scales.tolist(), [[1.0], [0.0]])
        got = w8a8_reference_linear(values, torch.from_numpy(codes), torch.from_numpy(scales))
        expected_dot = 2 * 2 + 2 * 2 + 127 * 127
        self.assertEqual(
            got[0].tolist(),
            [
                float(torch.tensor(expected_dot, dtype=torch.bfloat16)),
                float(torch.tensor(-expected_dot, dtype=torch.bfloat16)),
            ],
        )
        self.assertEqual(got[1].tolist(), [0.0, 0.0])

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "BF16"):
            quantize_w8a8_weights(torch.ones((2, 3), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            quantize_w8a8_weights(torch.tensor([[float("nan")]], dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "positive"):
            quantize_w8a8_weights(torch.zeros((1, 4), dtype=torch.bfloat16), rows_per_chunk=0)

    def test_integer_oracle_agrees_with_previous_float_code_simulation(self):
        weight = torch.tensor(
            [
                [1.0, -0.75, 0.0, 0.25, 0.5],
                [-0.5, 0.5, -0.25, 0.75, 1.0],
            ],
            dtype=torch.bfloat16,
        )
        values = torch.tensor(
            [
                [0.25, -0.5, 0.75, 1.0, 0.0],
                [-1.0, 0.5, -0.75, 0.0, 0.25],
            ],
            dtype=torch.bfloat16,
        )
        codes, scales = quantize_w8a8_weights(weight)
        actual = w8a8_reference_linear(values, torch.from_numpy(codes), torch.from_numpy(scales))

        weight_scale = weight.float().abs().amax(dim=-1, keepdim=True) / 127
        input_scale = values.float().abs().amax(dim=-1, keepdim=True) / 127
        weight_float_codes = torch.round(weight.float() / weight_scale).clamp(-127, 127)
        input_float_codes = torch.round(values.float() / input_scale).clamp(-127, 127)
        simulated = (
            F.linear(input_float_codes, weight_float_codes) * weight_scale.squeeze(-1) * input_scale
        ).to(values.dtype)
        torch.testing.assert_close(actual, simulated, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
