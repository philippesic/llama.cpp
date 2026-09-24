"""Focused checks for the EAGLE-3 packed head GGUF contract."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402
from conversion.llama import eagle3_w1a1_candidates, pack_w1a1_head  # noqa: E402


class TestPackedEagleHead(unittest.TestCase):
    def test_full_drafter_groups_cover_nine_eligible_linears(self):
        candidates = eagle3_w1a1_candidates(SimpleNamespace(block_count=1, hf_arch="LlamaForCausalLM"))
        grouped = {}
        for group, name in candidates.values():
            grouped.setdefault(group, set()).add(name)
        self.assertEqual(len(candidates), 9)
        self.assertEqual(len(grouped["fusion"]), 1)
        self.assertEqual(len(grouped["attention"]), 4)
        self.assertEqual(len(grouped["ffn"]), 3)
        self.assertEqual(len(grouped["head"]), 1)
        self.assertEqual(set(grouped), {"fusion", "attention", "ffn", "head"})
        base_model_candidates = eagle3_w1a1_candidates(SimpleNamespace(block_count=1, hf_arch="LlamaModel"))
        self.assertEqual(len(base_model_candidates), 9)
        self.assertIn("layers.0.self_attn.q_proj.weight", base_model_candidates)

    def test_signed_zero_tail_scales_and_gguf_round_trip(self):
        weight = torch.tensor([
            [0.0, -0.0, -2.0, 3.0] + [-1.0] * 29,
            [0.0] * 33,
        ], dtype=torch.bfloat16)
        # PyTorch tensor construction preserves the negative-zero sign bit.
        self.assertTrue(torch.signbit(weight[0, 1]))
        packed, scales = pack_w1a1_head(weight, rows_per_chunk=1)
        self.assertEqual(packed.shape, (2, 2))
        self.assertEqual(int(packed[0, 0]) & 15, 0b1011)
        self.assertEqual(int(packed[0, 1]) & 1, 0)
        self.assertEqual(int(packed[0, 1]) & ~1, 0)
        self.assertEqual(packed[1].view(np.uint32).tolist(), [0xFFFFFFFF, 1])
        np.testing.assert_array_equal(scales, weight.float().abs().mean(1).numpy())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.gguf"
            writer = gguf.GGUFWriter(path, "eagle3")
            writer.add_uint32("eagle3.w1a1_head.version", 1)
            writer.add_uint32("eagle3.w1a1_head.logical_k", 33)
            writer.add_tensor("output.w1a1_packed", packed, raw_dtype=gguf.GGMLQuantizationType.I32)
            writer.add_tensor("output.w1a1_scale", scales, raw_dtype=gguf.GGMLQuantizationType.F32)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            reader = gguf.GGUFReader(path)
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(tensors["output.w1a1_packed"].tensor_type, gguf.GGMLQuantizationType.I32)
            self.assertEqual(tensors["output.w1a1_packed"].shape.tolist(), [2, 2])
            np.testing.assert_array_equal(tensors["output.w1a1_packed"].data, packed)
            np.testing.assert_array_equal(tensors["output.w1a1_scale"].data, scales)
            self.assertEqual(reader.get_field("eagle3.w1a1_head.logical_k").contents(), 33)

    def test_rejects_invalid_source(self):
        with self.assertRaisesRegex(ValueError, "BF16"):
            pack_w1a1_head(torch.ones((2, 3), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            pack_w1a1_head(torch.tensor([[float("nan")]], dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
