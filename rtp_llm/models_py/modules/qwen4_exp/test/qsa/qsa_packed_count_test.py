import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_topk_triton import (
    compact_qsa_selection,
    try_select_qsa_tokens,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
    sparse_paged_gqa_attn,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class QsaPackedCountTest(unittest.TestCase):
    def test_exact_stable_selection_nonfinite_tail_and_graph(self):
        for rows, columns in ((4, 1024), (16, 2048), (7, 513)):
            torch.manual_seed(75100 + columns)
            scores = torch.randn(rows, columns, device="cuda")
            lengths = torch.empty(rows, device="cuda", dtype=torch.int32)

            def refresh(step):
                lengths.copy_(
                    (torch.arange(rows, device="cuda", dtype=torch.int32) * 137 + step)
                    % (columns * 4 + 4)
                )
                scores.copy_(torch.randn_like(scores))
                scores[0] = float("nan")
                scores[1] = float("-inf")
                scores[2, :20] = float("inf")
                scores[3] = 0
                scores.masked_fill_(
                    torch.arange(columns, device="cuda")[None, :]
                    >= lengths[:, None] // 4,
                    float("-inf"),
                )

            refresh(3)

            def call(packed):
                return try_select_qsa_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    scores_are_masked=True,
                    packed_count=packed,
                )

            call(True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                packed = call(True)
            for step in (0, 1, 2, 3, 4091):
                refresh(step)
                graph.replay()
                legacy = call(False)
                torch.cuda.synchronize()
                for a, b in zip(packed.cpu(), legacy.cpu()):
                    expected = b[b >= 0]
                    self.assertEqual(int(a[-1]), len(expected))
                    self.assertTrue(torch.equal(a[: len(expected)], expected))
                    self.assertTrue((a[len(expected) : -1] == -1).all())

    def test_core_dynamic_count_scale_page_zero_and_reordered_rows(self):
        for batch, queries in ((1, 4), (4, 4)):
            torch.manual_seed(7593 + batch)
            q = torch.randn(
                batch, 12, queries, 256, device="cuda", dtype=torch.bfloat16
            )
            cache = (torch.randn(35, 2, 1, 128, 256, device="cuda") * 0.3).to(
                torch.float8_e4m3fn
            )
            scale = torch.rand(35, 2, 1, 128, device="cuda") + 0.2
            table = (
                torch.arange(1, 34, device="cuda", dtype=torch.int32)
                .expand(batch, -1)
                .contiguous()
            )
            lengths = torch.full(
                (batch, queries), 4096, device="cuda", dtype=torch.int32
            )
            selected = torch.full(
                (batch, queries, 2051), -1, device="cuda", dtype=torch.int32
            )
            packed = compact_qsa_selection(selected)

            def run(indices, count, capture):
                return sparse_paged_gqa_attn(
                    q,
                    cache,
                    table,
                    lengths,
                    indices,
                    page_size=128,
                    kv_scale=scale,
                    selected_has_count=count,
                    graph_capture=capture,
                )

            with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
                run(packed, True, True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = run(packed, True, True)
            for count in (0, 3, 127, 511, 2048, 9):
                selected.fill_(-1)
                if count:
                    selected[..., :count] = torch.randperm(4096, device="cuda")[:count]
                    selected.copy_(selected.flip(0))
                packed.copy_(compact_qsa_selection(selected))
                graph.replay()
                # Existing compensated-BF16 numerical contract remains intact.
                expected = run(selected, False, False)
                torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)
                table[:, 0] = 0
                with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
                    expected = run(selected, False, True)
                    graph.replay()
                    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)
                table[:, 0] = 1
            packed[..., -1] = 2052
            with self.assertRaisesRegex(ValueError, "packed QSA"):
                run(packed, True, False)

    def test_fp16_pv_against_independent_fp64_and_large_value_fallback(self):
        torch.manual_seed(77991)
        q = torch.randn(1, 12, 4, 256, device="cuda", dtype=torch.bfloat16)
        cache = (torch.randn(18, 2, 1, 128, 256, device="cuda") * 0.2).to(
            torch.float8_e4m3fn
        )
        scale = torch.rand(18, 2, 1, 128, device="cuda") + 0.3
        table = torch.arange(1, 18, device="cuda", dtype=torch.int32)[None, :]
        lengths = torch.full((1, 4), 2048, device="cuda", dtype=torch.int32)
        selected = torch.full((1, 4, 2051), -1, device="cuda", dtype=torch.int32)
        selected[:, :, :2048] = torch.randperm(2048, device="cuda")
        packed = compact_qsa_selection(selected)
        for multiplier in (0.01, 1.0, 16.0, 1000000.0):
            sf = scale * multiplier
            actual = sparse_paged_gqa_attn(
                q,
                cache,
                table,
                lengths,
                packed,
                page_size=128,
                kv_scale=sf,
                selected_has_count=True,
                fp16_pv=True,
            )
            expected = torch.empty_like(q)
            for query in range(4):
                ids = selected[0, query, :2048].long()
                pages = table[0, ids // 128]
                slot = ids % 128
                k = (
                    (cache[pages, 0, 0, slot].float() * sf[pages, 0, 0, slot, None])
                    .bfloat16()
                    .double()
                )
                v = (
                    (cache[pages, 1, 0, slot].float() * sf[pages, 1, 0, slot, None])
                    .bfloat16()
                    .double()
                )
                score = q[0, :, query].double() @ k.T / 16
                expected[0, :, query] = (score.softmax(-1) @ v).bfloat16()
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            if multiplier == 1000000.0:
                legacy = sparse_paged_gqa_attn(
                    q,
                    cache,
                    table,
                    lengths,
                    packed,
                    page_size=128,
                    kv_scale=sf,
                    selected_has_count=True,
                )
                torch.testing.assert_close(actual, legacy, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
