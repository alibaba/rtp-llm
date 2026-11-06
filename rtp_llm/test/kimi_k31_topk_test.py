"""Constructed-score checks; these do not assert real-model accuracy."""

import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.base.cuda.bf16_indexer_op import Bf16DraftIndexerOp
from rtp_llm.models_py.modules.dsv4.indexer_topk import (
    TorchIndexerTopKBackend,
    native_page_rr_selection_supported,
    page_rr_unordered_topk_enabled,
)


class KimiK31PageRRTopKPolicyTest(unittest.TestCase):
    def setUp(self):
        page_rr_unordered_topk_enabled.cache_clear()
        self.addCleanup(page_rr_unordered_topk_enabled.cache_clear)

    def test_default_preserves_dense_order_when_a_chunk_can_exceed_k(self):
        with patch.dict(os.environ):
            os.environ.pop("DSA_PAGE_RR_TOPK_ALLOW_UNORDERED", None)
            geometry = dict(shards=4, page=128, start=1024)
            self.assertTrue(native_page_rr_selection_supported(512, 2048, **geometry))
            self.assertFalse(native_page_rr_selection_supported(640, 2048, **geometry))

    def test_opt_in_allows_long_selection_on_supported_geometry(self):
        with patch.dict(os.environ, {"DSA_PAGE_RR_TOPK_ALLOW_UNORDERED": "1"}):
            self.assertTrue(native_page_rr_selection_supported(262144, 2048, shards=4, page=1024, start=0))
            self.assertFalse(native_page_rr_selection_supported(262145, 2048, shards=4, page=1024, start=0))


class KimiK31TopKTest(unittest.TestCase):
    def setUp(self):
        self.op = Bf16DraftIndexerOp(SimpleNamespace(
            indexer_use_fp8=False, index_rope_dim=0,
            index_head_dim=128, index_topk=2048,
        ), device="cpu")

    def test_short_equal_and_long_history_with_causal_mask(self):
        for length in (17, 2048, 4097):
            with self.subTest(length=length):
                positions = torch.arange(length, dtype=torch.int32)
                score = positions.float().repeat(2, 1)
                score[0, 5:] = -float("inf")
                result = self.op._select(score, positions)
                self.assertEqual(result.shape, (2, 2048))
                self.assertEqual(result[0, :5].tolist(), [4, 3, 2, 1, 0])
                self.assertTrue((result[0, 5:] == -1).all())
                count = min(length, 2048)
                self.assertEqual(result[1, :count].tolist(),
                                 list(range(length - 1, length - 1 - count, -1)))
                self.assertTrue((result[1, count:] == -1).all())

    def test_owner_order_does_not_change_ties(self):
        positions = torch.tensor([1024, 0, 2048, 1025, 1], dtype=torch.int32)
        score = torch.ones(1, 5)
        result = self.op._select(score, positions)
        self.assertEqual(result[0, :5].tolist(), [0, 1, 1024, 1025, 2048])

    def test_empty_and_invalid_queries_have_no_selected_positions(self):
        for length in (0, 32):
            result = self.op._select(torch.full((2, length), -float("inf")),
                                     torch.arange(length, dtype=torch.int32))
            self.assertTrue((result == -1).all())

    def test_stable_sort_preserves_mask_ties_and_offsets(self):
        rows, width = 257, 2053
        score = (torch.arange(rows * width).reshape(rows, width) % 7).float()
        lengths = torch.arange(rows, dtype=torch.int32) % width
        lengths[128:130] = torch.tensor([2048, width])
        positions = torch.arange(width)[None, :]
        masked = score.masked_fill(positions >= lengths[:, None], -float("inf"))
        expected = masked.argsort(dim=-1, descending=True, stable=True)[:, :2048]
        expected = torch.where(expected < lengths[:, None], expected + 31, -1).int()
        actual = TorchIndexerTopKBackend(stable_ties=True).select(
            score, 2048, lengths=lengths, offset=31)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
