"""Contracts for the SM90 FP8 page table, DeepGEMM, and decode selection."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
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


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "SM90 GPU required",
)
class FP8DeepGEMMDecodeTest(unittest.TestCase):
    def test_eager_table_capacity_and_short_causal_selection(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_global
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
        from rtp_llm.models_py.modules.dsv41.kv_cache_utils import INDEXER_KV

        device = torch.device("cuda")
        max_seq_len = 2048
        tokens_per_block = 128
        topk = 512
        q = torch.ones((2, 1, 32, 128), dtype=torch.bfloat16, device=device)
        weights = torch.ones((2, 32), dtype=torch.float32, device=device)
        freqs = torch.ones((max_seq_len, 32), dtype=torch.complex64, device=device)
        output_ids = torch.arange(topk, dtype=torch.int32, device=device)
        for ratio, widths in ((1, (1, 4, 8, 16)), (2, (1, 8, 9, 16))):
            for table_width in widths:
                with self.subTest(ratio=ratio, table_width=table_width):
                    raw_span = min(max_seq_len, table_width * tokens_per_block)
                    expected_capacity = raw_span // ratio
                    positions = torch.tensor([0, raw_span - 1], device=device)
                    table = (
                        torch.arange(
                            1, table_width + 1, dtype=torch.int32, device=device
                        )
                        .expand(2, -1)
                        .contiguous()
                    )
                    entries = tokens_per_block // ratio
                    pool = torch.empty(
                        (table_width + 1, entries, 132),
                        dtype=torch.uint8,
                        device=device,
                    )
                    # All K elements are one before the per-key scale. Make the
                    # positive dot products increase strictly with logical ID.
                    pages = cache.dg_pages(pool).flatten(1)
                    pages[:, : 64 * 128].fill_(0x38)
                    scales = pages[:, 64 * 128 :].view(torch.float32)
                    scales.copy_(
                        torch.arange(
                            1, scales.numel() + 1, dtype=torch.float32, device=device
                        ).view_as(scales)
                    )
                    attn = SimpleNamespace(
                        _shared_attention={},
                        _source_pool=lambda tag: pool,
                        _cp_ctx=None,
                        _rope_max_seq_len=max_seq_len,
                        _block_tables_by_type={INDEXER_KV: table},
                        _kv_cache=SimpleNamespace(
                            group_tags=(INDEXER_KV,),
                            get_kernel_seq_size_per_block=lambda tag: tokens_per_block,
                        ),
                        _indexer_mode="fp8_dg",
                        compress_ratio=ratio,
                        is_index_source=True,
                        layer_id=1,
                        kv_source_layer_id=1,
                        index_n_heads=32,
                        index_head_dim=128,
                        index_topk=topk,
                        v41_config={"candidate_source_layer_id": 1},
                        rope_head_dim=64,
                        freqs_cis=freqs,
                    )
                    x = torch.zeros((2, 1, 1), dtype=torch.bfloat16, device=device)
                    with patch.object(
                        _v41_decode_global, "try_produce_global", return_value=True
                    ):
                        AttentionV41FP8._produce_global_decode(
                            attn,
                            x,
                            positions,
                            torch.arange(2, device=device),
                            positions,
                        )
                    keys = attn._shared_attention["global"][1]
                    self.assertEqual(keys.capacity, expected_capacity)
                    visible = (positions + 1) // ratio
                    logical_ids = output_ids + (visible[:, None] - topk).clamp_min(0)
                    reference = torch.where(
                        logical_ids < visible[:, None], logical_ids, -1
                    ).int()
                    actual = AttentionV41FP8._select_indices_decode(
                        attn, x, None, positions, prepared_query=(q, weights, None)
                    )
                    short = visible <= topk
                    torch.testing.assert_close(
                        actual[short], reference[short], rtol=0, atol=0
                    )
                    # Kernel output order is unspecified for nontrivial TopK.
                    torch.testing.assert_close(
                        actual[~short].sort(-1).values,
                        reference[~short],
                        rtol=0,
                        atol=0,
                    )

                    self.assertEqual(keys.block_table.shape[1] * 64, keys.capacity)
                    logits = score.score_paged_fp8_indexer(
                        q,
                        weights.view(2, 1, 32),
                        freqs,
                        positions,
                        keys.pool,
                        keys.block_table,
                        visible[:, None],
                        max_ctx_len=keys.capacity,
                        rope_head_dim=64,
                    )
                    torch.testing.assert_close(
                        logits.lengths, visible.int(), rtol=0, atol=0
                    )
                    self.assertEqual(logits.logits.shape, (2, keys.capacity))
                    expected_scales = torch.arange(
                        entries + 1,
                        entries + keys.capacity + 1,
                        dtype=torch.float32,
                        device=device,
                    )
                    torch.testing.assert_close(
                        logits.logits[1],
                        expected_scales * (32 * 128) ** 0.5,
                        rtol=1e-5,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
