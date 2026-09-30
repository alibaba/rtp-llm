"""Bitmap prefill preserves sparse selection, including empty and repeated rows."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.sparse_fmha import (
    sparse_prefill_attn,
    sparse_prefill_attn_torch_reference,
)
from rtp_llm.models_py.modules.qwen4_exp.sparse_prefill_bitmap import try_bitmap_prefill


class SparsePrefillBitmapTest(unittest.TestCase):
    def inputs(self, batch=2, heads=6, length=131):
        torch.manual_seed(71)
        q = torch.randn(batch, heads, length, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(
            batch, heads // 3, length, 256, device="cuda", dtype=torch.bfloat16
        )
        v = torch.randn_like(k)
        # An unordered unique subset, including selections beyond the query
        # position: the kernel must follow the supplied mask, not add causality.
        selected = torch.rand(batch, length, length, device="cuda").argsort(-1).int()
        selected[..., length // 2 :] = -1
        selected = torch.cat(
            (
                selected,
                torch.full(
                    (batch, length, 256 - length), -1, device="cuda", dtype=torch.int32
                ),
            ),
            -1,
        )
        selected[:, 0] = -1
        return q, k, v, selected

    def test_ragged_queries_multiple_kv_heads_match_torch(self):
        q, k, v, selected = self.inputs()
        candidate = try_bitmap_prefill(q, k, v, selected)
        self.assertIsNotNone(candidate)
        expected = sparse_prefill_attn_torch_reference(q, k, v, selected)
        torch.testing.assert_close(candidate, expected, atol=0.0078125, rtol=0.008)
        self.assertTrue((candidate[:, :, 0] == 0).all())

    def test_actual_long_qsa_selection_matches_grouped_kernel(self):
        length = 4096
        torch.manual_seed(72)
        q = torch.randn(1, 3, length, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 1, length, 256, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        pos = torch.arange(length, device="cuda")
        complete = (pos + 1) // 4
        scores = torch.rand(length, length // 4, device="cuda")
        scores.masked_fill_(
            torch.arange(length // 4, device="cuda")[None] >= complete[:, None],
            float("-inf"),
        )
        values, blocks = scores.topk(512, dim=-1)
        tokens = blocks[..., None] * 4 + torch.arange(4, device="cuda")
        tokens = torch.where(torch.isfinite(values)[..., None], tokens, -1)
        tail = complete[:, None] * 4 + torch.arange(3, device="cuda")
        tail = torch.where(tail <= pos[:, None], tail, -1)
        selected = torch.cat((tokens.flatten(1), tail), -1)[None].int()
        with patch.dict(
            "os.environ",
            {
                "RTP_LLM_QWEN4_SPARSE_PREFILL_BITMAP": "0",
                "RTP_LLM_QWEN4_SPARSE_PREFILL_GROUPED": "1",
            },
        ):
            expected = sparse_prefill_attn(q, k, v, selected)
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_SPARSE_PREFILL_BITMAP": "1"}):
            candidate = sparse_prefill_attn(q, k, v, selected)
        torch.testing.assert_close(candidate, expected, atol=0.0078125, rtol=0.008)

    def test_repeated_indices_keep_multiplicity_via_fallback(self):
        q, k, v, selected = self.inputs()
        selected[:, 1, 1] = selected[:, 1, 0]
        self.assertIsNone(try_bitmap_prefill(q, k, v, selected))
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_SPARSE_PREFILL_BITMAP": "1"}):
            candidate = sparse_prefill_attn(q, k, v, selected)
        expected = sparse_prefill_attn_torch_reference(q, k, v, selected)
        torch.testing.assert_close(candidate, expected, atol=0.0078125, rtol=0.008)

    def test_layout_backend_and_capture_fallback(self):
        q, k, v, selected = self.inputs()
        # Valid last-dimension stride, unsupported bitmap layout.
        strided = torch.empty((*q.shape[:-1], 512), device=q.device, dtype=q.dtype)[
            ..., :256
        ]
        strided.copy_(q)
        self.assertIsNone(try_bitmap_prefill(strided, k, v, selected))
        with patch("torch.version.hip", "hip"):
            self.assertIsNone(try_bitmap_prefill(q, k, v, selected))
        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
            self.assertIsNone(try_bitmap_prefill(q, k, v, selected))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = try_bitmap_prefill(q, k, v, selected)
        self.assertIsNone(result)

    def test_invalid_selection_is_rejected_before_bitmap(self):
        q, k, v, selected = self.inputs()
        selected[0, 0, 0] = k.shape[2]
        snapshot = k.clone()
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_SPARSE_PREFILL_BITMAP": "1"}):
            with self.assertRaisesRegex(ValueError, "outside KV sequence"):
                sparse_prefill_attn(q, k, v, selected)
        self.assertTrue(torch.equal(k, snapshot))


class SparsePrefillWordsTest(SparsePrefillBitmapTest):
    def setUp(self):
        setting = patch.dict("os.environ", {"RTP_LLM_QWEN4_SPARSE_PREFILL_WORDS": "1"})
        setting.start()
        self.addCleanup(setting.stop)

    def test_word_boundaries_and_repeated_bits(self):
        q, k, v, selected = self.inputs()
        selected.fill_(-1)
        selected[..., :6] = torch.tensor(
            [0, 31, 32, 63, 64, 130], device="cuda", dtype=torch.int32
        )
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_SPARSE_PREFILL_WORDS": "0"}):
            expected = try_bitmap_prefill(q, k, v, selected)
        candidate = try_bitmap_prefill(q, k, v, selected)
        self.assertIsNotNone(candidate)
        torch.testing.assert_close(candidate, expected, atol=0, rtol=0)
        # This six-key fixture exposes BF16 intermediate rounding in the Torch
        # gather reference. Check the unchanged attention math in FP64 as well.
        ids = selected[0, 0, :6].long()
        oracle = torch.empty_like(q, dtype=torch.float64)
        for batch in range(q.shape[0]):
            for head in range(q.shape[1]):
                keys = k[batch, head // 3, ids].double()
                values = v[batch, head // 3, ids].double()
                probabilities = torch.softmax(q[batch, head].double() @ keys.T / 16, -1)
                oracle[batch, head] = probabilities @ values
        torch.testing.assert_close(
            candidate.double(), oracle, atol=0.0078125, rtol=0.008
        )
        selected[..., 200] = 31
        self.assertIsNone(try_bitmap_prefill(q, k, v, selected))


if __name__ == "__main__":
    unittest.main()
