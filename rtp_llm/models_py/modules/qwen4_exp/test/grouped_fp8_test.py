import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp import grouped_sparse_paged_fmha as grouped
from rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha import sparse_paged_gqa_attn
from rtp_llm.models_py.modules.qwen4_exp.test.sparse_paged_fmha_test import _reference


class GroupedFp8Test(unittest.TestCase):
    def fixture(self, heads, kv_heads, queries):
        torch.manual_seed(763)
        q = torch.randn(8, heads, queries, 256, device="cuda", dtype=torch.bfloat16)
        scale = torch.rand(35, 2, kv_heads, 128, device="cuda") * 0.8 + 0.2
        source = torch.randn(35, 2, kv_heads, 128, 256, device="cuda") * 0.3
        cache = (source / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        table = torch.stack(
            [
                torch.arange(1, 33, device="cuda", dtype=torch.int32).roll(i)
                for i in range(8)
            ]
        )
        lengths = torch.full((8, queries), 3000, device="cuda", dtype=torch.int32)
        lengths[2] = 97
        lengths[5] = 0
        selected = torch.arange(2051, device="cuda", dtype=torch.int32).repeat(
            8, queries, 1
        )
        selected = torch.where(selected < lengths.unsqueeze(-1), selected, -1)
        return q, cache, scale, table, lengths, selected

    def run_grouped(
        self, q, cache, scale, table, lengths, selected, kv_heads, **kwargs
    ):
        return sparse_paged_gqa_attn(
            q,
            cache,
            table,
            lengths,
            selected,
            page_size=128,
            kv_head_num=kv_heads,
            kv_scale=scale,
            **kwargs
        )

    def test_tp2_tp4_tp8_and_two_kv_heads_with_scaled_padded_cache(self):
        for heads, kv_heads in [(3, 1), (6, 1), (12, 1), (24, 2)]:
            for queries in (1, 4):
                with self.subTest(heads=heads, kv_heads=kv_heads, queries=queries):
                    q, cache, scale, table, lengths, selected = self.fixture(
                        heads, kv_heads, queries
                    )
                    packed = torch.empty(
                        35, cache[0].numel() + 37, device="cuda", dtype=cache.dtype
                    )
                    packed[:, : cache[0].numel()].copy_(cache.flatten(1))
                    packed_scale = torch.full(
                        (35, scale[0].numel() + 13), float("nan"), device="cuda"
                    )
                    packed_scale[:, : scale[0].numel()].copy_(scale.flatten(1))
                    expected = _reference(
                        q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
                    )
                    with patch.dict(
                        os.environ,
                        {
                            "RTP_LLM_QWEN4_PAGED_ATTENTION_ONLINE": "1",
                            "RTP_LLM_QWEN4_PAGED_ATTENTION_SPLITS": "8",
                            "RTP_LLM_QWEN4_GROUPED_SPARSE_DECODE": "1",
                            "RTP_LLM_QWEN4_GROUPED_SPARSE_FP8": "1",
                        },
                    ), patch.object(
                        grouped,
                        "grouped_sparse_paged_split",
                        wraps=grouped.grouped_sparse_paged_split,
                    ) as called:
                        got = self.run_grouped(
                            q, packed, packed_scale, table, lengths, selected, kv_heads
                        )
                        self.assertEqual(called.call_count, 1)
                    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)
                    self.assertTrue(
                        torch.isnan(packed_scale[:, scale[0].numel() :]).all()
                    )

    def test_graph_updates_fp8_values_scales_pages_and_zero_rows(self):
        q, cache, scale, table, lengths, selected = self.fixture(12, 1, 4)
        with patch.dict(
            os.environ,
            {
                "RTP_LLM_QWEN4_PAGED_ATTENTION_ONLINE": "1",
                "RTP_LLM_QWEN4_PAGED_ATTENTION_SPLITS": "8",
                "RTP_LLM_QWEN4_GROUPED_SPARSE_DECODE": "1",
                "RTP_LLM_QWEN4_GROUPED_SPARSE_FP8": "1",
            },
        ):
            for _ in range(2):
                self.run_grouped(q, cache, scale, table, lengths, selected, 1)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = self.run_grouped(
                    q, cache, scale, table, lengths, selected, 1, graph_capture=True
                )
            for factor in (0.5, 2.0):
                q.mul_(0.75)
                scale.mul_(factor)
                cache.copy_((torch.randn_like(cache.float()) * 0.3).to(cache.dtype))
                table.copy_(table.flip(1))
                graph.replay()
                torch.cuda.synchronize()
                expected = _reference(
                    q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
                )
                torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)
            lengths.zero_()
            graph.replay()
            torch.testing.assert_close(got, torch.zeros_like(got))

    def test_small_batches_and_intermediate_query_lengths(self):
        for batch in range(1, 9):
            queries = 1 + batch % 4
            with self.subTest(batch=batch, queries=queries):
                q, cache, scale, table, lengths, selected = self.fixture(12, 1, queries)
                q, table, lengths, selected = (
                    x[:batch].contiguous() for x in (q, table, lengths, selected)
                )
                with patch.dict(
                    os.environ,
                    {
                        "RTP_LLM_QWEN4_PAGED_ATTENTION_ONLINE": "1",
                        "RTP_LLM_QWEN4_PAGED_ATTENTION_SPLITS": "8",
                        "RTP_LLM_QWEN4_GROUPED_SPARSE_DECODE": "1",
                        "RTP_LLM_QWEN4_GROUPED_SPARSE_FP8": "1",
                    },
                ), patch.object(
                    grouped,
                    "grouped_sparse_paged_split",
                    wraps=grouped.grouped_sparse_paged_split,
                ) as called:
                    got = self.run_grouped(q, cache, scale, table, lengths, selected, 1)
                    self.assertEqual(called.call_count, 1)
                torch.testing.assert_close(
                    got,
                    _reference(
                        q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
                    ),
                    atol=2e-2,
                    rtol=2e-2,
                )

    def test_two_kv_graph_live_selected_tail_and_invalid_pages(self):
        q, cache, scale, table, lengths, selected = self.fixture(24, 2, 3)
        q, table, lengths, selected = (
            x[:7].contiguous() for x in (q, table, lengths, selected)
        )
        selected.fill_(-1)
        selected[..., -3:] = torch.tensor(
            [90, 91, 92], device="cuda", dtype=torch.int32
        )
        selected.masked_fill_(selected >= lengths.unsqueeze(-1), -1)
        with patch.dict(
            os.environ,
            {
                "RTP_LLM_QWEN4_PAGED_ATTENTION_ONLINE": "1",
                "RTP_LLM_QWEN4_PAGED_ATTENTION_SPLITS": "8",
                "RTP_LLM_QWEN4_GROUPED_SPARSE_DECODE": "1",
                "RTP_LLM_QWEN4_GROUPED_SPARSE_FP8": "1",
            },
        ):
            for _ in range(2):
                self.run_grouped(q, cache, scale, table, lengths, selected, 2)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = self.run_grouped(
                    q, cache, scale, table, lengths, selected, 2, graph_capture=True
                )
            for tail in ([0, 1, 2], [90, 91, 92]):
                selected[..., -3:] = torch.tensor(
                    tail, device="cuda", dtype=torch.int32
                )
                scale.mul_(0.75)
                table.copy_(table.flip(1))
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    got,
                    _reference(
                        q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
                    ),
                    atol=2e-2,
                    rtol=2e-2,
                )
            selected.fill_(-1)
            graph.replay()
            torch.testing.assert_close(got, torch.zeros_like(got))
            selected[..., -3:] = torch.tensor(
                [0, 1, 2], device="cuda", dtype=torch.int32
            )
            table.fill_(cache.shape[0])
            graph.replay()
            torch.testing.assert_close(got, torch.zeros_like(got))

    def test_enabled_flag_keeps_unprofitable_small_shapes_on_fallback(self):
        for batch, heads, queries in [(1, 12, 1), (2, 12, 1), (3, 12, 1), (1, 6, 4)]:
            with self.subTest(batch=batch, heads=heads, queries=queries):
                q, cache, scale, table, lengths, selected = self.fixture(
                    heads, 1, queries
                )
                q, table, lengths, selected = (
                    x[:batch].contiguous() for x in (q, table, lengths, selected)
                )
                with patch.dict(
                    os.environ,
                    {
                        "RTP_LLM_QWEN4_PAGED_ATTENTION_ONLINE": "1",
                        "RTP_LLM_QWEN4_PAGED_ATTENTION_SPLITS": "8",
                        "RTP_LLM_QWEN4_GROUPED_SPARSE_DECODE": "1",
                        "RTP_LLM_QWEN4_GROUPED_SPARSE_FP8": "1",
                    },
                ), patch.object(
                    grouped,
                    "grouped_sparse_paged_split",
                    wraps=grouped.grouped_sparse_paged_split,
                ) as called:
                    got = self.run_grouped(q, cache, scale, table, lengths, selected, 1)
                    self.assertEqual(called.call_count, 0)
                torch.testing.assert_close(
                    got,
                    _reference(
                        q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
                    ),
                    atol=2e-2,
                    rtol=2e-2,
                )

    def test_flag_off_keeps_fp8_online_fallback(self):
        q, cache, scale, table, lengths, selected = self.fixture(12, 1, 1)
        with patch.dict(
            os.environ,
            {
                "RTP_LLM_QWEN4_PAGED_ATTENTION_ONLINE": "1",
                "RTP_LLM_QWEN4_PAGED_ATTENTION_SPLITS": "8",
                "RTP_LLM_QWEN4_GROUPED_SPARSE_DECODE": "1",
                "RTP_LLM_QWEN4_GROUPED_SPARSE_FP8": "0",
            },
        ), patch.object(
            grouped,
            "grouped_sparse_paged_split",
            wraps=grouped.grouped_sparse_paged_split,
        ) as called:
            got = self.run_grouped(q, cache, scale, table, lengths, selected, 1)
            self.assertEqual(called.call_count, 0)
        torch.testing.assert_close(
            got,
            _reference(
                q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
            ),
            atol=2e-2,
            rtol=2e-2,
        )


if __name__ == "__main__":
    unittest.main()
