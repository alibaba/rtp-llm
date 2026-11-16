"""CPU contracts for the SM90 FP8 page table and DeepGEMM score wrapper."""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_fp8_indexer_cache as cache
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_fp8_indexer_score as score


class _FakeLaunch:
    def __init__(self, run):
        self.run = run

    def __getitem__(self, grid):
        return self.run


class FP8DeepGEMMContractTest(unittest.TestCase):
    def test_ratio_pages_and_missing_page_sentinel(self):
        table = torch.tensor([[-1, 1, 2], [0, 3, -1]], dtype=torch.int32)

        def expand(src, dst, **kwargs):
            raw = src.clamp_min(0)
            dst[:, 0::2] = raw * 2
            dst[:, 1::2] = torch.where(raw > 0, raw * 2 + 1, 0)

        def sanitize(src, dst, **kwargs):
            dst.copy_(src.clamp_min(0))

        ratio1 = torch.empty((4, 128, 132), dtype=torch.uint8)
        ratio2 = torch.empty((4, 64, 132), dtype=torch.uint8)
        self.assertEqual(cache.dg_pages(ratio1).shape, (8, 64, 132))
        self.assertEqual(cache.dg_pages(ratio1).data_ptr(), ratio1.data_ptr())
        self.assertEqual(cache.dg_pages(ratio2).shape, (4, 64, 132))
        with patch.object(cache, "_expand_table_kernel", _FakeLaunch(expand)), patch.object(
            cache, "_sanitize_table_kernel", _FakeLaunch(sanitize)
        ):
            self.assertEqual(
                cache.dg_block_table(table, ratio1).tolist(),
                [[0, 0, 2, 3, 4, 5], [0, 0, 6, 7, 0, 0]],
            )
            self.assertEqual(
                cache.dg_block_table(table, ratio2).tolist(),
                [[0, 1, 2], [0, 3, 0]],
            )

    def test_odd_q_uses_pair_max_for_schedule_and_compacts_only_ids(self):
        q = torch.zeros((2, 3, 32, 128), dtype=torch.bfloat16)
        weights = torch.ones((2, 3, 32), dtype=torch.float32)
        positions = torch.arange(6, dtype=torch.int64).view(2, 3)
        lengths = torch.tensor([[130, 32, 65], [2, -1, 70]], dtype=torch.int32)
        pages = torch.zeros((5, 64, 132), dtype=torch.uint8)
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        calls = {}
        fake_dg = ModuleType("deep_gemm")

        def metadata(context_lens, block_size, num_sms):
            calls["metadata_lengths"] = context_lens.clone()
            self.assertEqual((block_size, num_sms), (64, 132))
            return "metadata"

        def logits(q_arg, k_arg, w_arg, context_lens, block_table, meta, width):
            calls["q_shape"] = q_arg[0].shape
            calls["k_shape"] = k_arg.shape
            calls["weights_shape"] = w_arg.shape
            calls["table"] = block_table.clone()
            calls["score_lengths"] = context_lens.clone()
            self.assertIsNone(q_arg[1])
            self.assertEqual((meta, width), ("metadata", 128))
            return torch.zeros((8, width), dtype=torch.float32)

        fake_dg.get_paged_mqa_logits_metadata = metadata
        fake_dg.fp8_fp4_paged_mqa_logits = logits
        q_fp8 = torch.zeros(q.shape, dtype=torch.float8_e4m3fn)
        with patch.dict(sys.modules, {"deep_gemm": fake_dg}), patch.object(
            score, "indexer_q_rope_fp8_quant_fold_indexed",
            return_value=(q_fp8, weights),
        ), patch.object(score, "_get_num_sms", return_value=132):
            result = score.score_paged_fp8_indexer(
                q, weights, torch.empty(0), positions, pages, table, lengths,
                max_ctx_len=128, rope_head_dim=64,
            )

        self.assertEqual(calls["q_shape"], (4, 2, 32, 128))
        self.assertEqual(calls["k_shape"], (5, 64, 1, 132))
        self.assertEqual(calls["weights_shape"], (8, 32))
        self.assertEqual(calls["table"].tolist(), [[1, 2], [1, 2], [3, 4], [3, 4]])
        self.assertEqual(calls["metadata_lengths"].tolist(), [[128, 128], [65, 65], [2, 2], [70, 70]])
        self.assertTrue(torch.equal(calls["metadata_lengths"], calls["score_lengths"]))
        self.assertEqual(result.lengths.tolist(), [128, 32, 65, 0, 2, 0, 70, 0])
        selected = torch.arange(8 * 512, dtype=torch.int32).view(8, 512)
        self.assertEqual(
            result.compact_selected(selected)[:, 0].tolist(),
            [0, 512, 1024, 2048, 2560, 3072],
        )


if __name__ == "__main__":
    unittest.main()
