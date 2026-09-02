import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaOp,
    _fp8_sparse_padded_heads,
)
from rtp_llm.models_py.modules.hybrid.mla_attention import _infer_gated_mla_type


class Hy4AttentionSinkTest(unittest.TestCase):
    def test_fp8_head_envelope_keeps_64_heads_on_blackwell_decode(self):
        self.assertEqual(_fp8_sparse_padded_heads(8), 64)
        self.assertEqual(_fp8_sparse_padded_heads(64), 64)
        self.assertEqual(_fp8_sparse_padded_heads(65), 128)
        self.assertEqual(_fp8_sparse_padded_heads(128), 128)

    def test_gate_layout_inference_accepts_bf16_and_fp8_orientations(self):
        for shape in ((32, 24), (24, 32)):
            gate = torch.empty(shape)
            self.assertEqual(_infer_gated_mla_type(gate, 3, 8), "elementwise")

    def test_query_and_sink_use_identical_noop_head_padding(self):
        op = SparseMlaOp.__new__(SparseMlaOp)
        op.num_heads = 64
        query = torch.randn(2, 3, 8)
        sink = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)

        padded_query, padded_sink, actual_heads = op._pad_query_and_sink(query, sink)

        self.assertEqual(actual_heads, 3)
        self.assertEqual(tuple(padded_query.shape), (2, 64, 8))
        self.assertEqual(tuple(padded_sink.shape), (64,))
        torch.testing.assert_close(padded_query[:, :3], query)
        torch.testing.assert_close(padded_sink[:3], sink)
        self.assertTrue(torch.isneginf(padded_sink[3:]).all())
        self.assertEqual(torch.count_nonzero(padded_query[:, 3:]).item(), 0)

    def test_bf16_prefill_can_override_fp8_head_envelope(self):
        op = SparseMlaOp.__new__(SparseMlaOp)
        op.num_heads = 64
        query = torch.randn(2, 64, 8)
        sink = torch.randn(64, dtype=torch.float32)

        padded_query, padded_sink, actual_heads = op._pad_query_and_sink(
            query, sink, kernel_heads=128
        )

        self.assertEqual(actual_heads, 64)
        self.assertEqual(tuple(padded_query.shape), (2, 128, 8))
        self.assertEqual(tuple(padded_sink.shape), (128,))
        self.assertTrue(torch.isneginf(padded_sink[64:]).all())
        self.assertEqual(torch.count_nonzero(padded_query[:, 64:]).item(), 0)

    def test_sink_head_count_must_match_unpadded_query(self):
        op = SparseMlaOp.__new__(SparseMlaOp)
        op.num_heads = 64
        with self.assertRaisesRegex(ValueError, "expected 3"):
            op._pad_query_and_sink(torch.randn(2, 3, 8), torch.zeros(4))


if __name__ == "__main__":
    unittest.main()
