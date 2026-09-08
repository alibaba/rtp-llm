"""GLM53 real mHC token independence and compressed Top-k semantics."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed.sequence_parallel import (
    shard_tokens,
    token_shard_layout,
)
from rtp_llm.models_py.modules.dsv4.fp8.indexer import _run_prefill_topk
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit
from rtp_llm.models_py.modules.indexer_grouping import (
    append_incomplete_tail_indices,
    expand_indexer_group_indices,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class MhcTopkTest(unittest.TestCase):
    @torch.inference_mode()
    def test_actual_mhc_is_token_independent(self):
        torch.manual_seed(41)
        dim, hc = 4096, 4
        fn = torch.randn(24, hc * dim, device="cuda", dtype=torch.float32) * 0.02
        unit = TileLangHCUnit(
            fn,
            torch.zeros(24, device="cuda"),
            torch.ones(3, device="cuda"),
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        for tokens in (1, 7, 8, 9, 255, 1025):
            residual = torch.randn(tokens, hc, dim, device="cuda", dtype=torch.bfloat16)
            sublayer = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
            full_pre, full_post, full_comb = unit.pre(residual)
            expected = unit.post(sublayer, residual.clone(), full_post, full_comb)
            # Different token counts may select different mHC GEMM tiles;
            # preserve the BF16 numerical contract, not an unsupported bitwise guarantee.
            result = []
            for rank in range(8):
                layout = token_shard_layout(tokens, 8, rank)
                local_residual = shard_tokens(residual, layout).clone()
                pre, post, comb = unit.pre(local_residual)
                valid = layout.local_valid_tokens
                expected_pre = shard_tokens(full_pre, layout)
                torch.testing.assert_close(
                    pre[:valid], expected_pre[:valid], atol=1e-2, rtol=1e-2
                )
                actual = unit.post(
                    shard_tokens(sublayer, layout), local_residual, post, comb
                )
                result.append(actual)
            actual_full = torch.cat(result)[:tokens]
            print(
                f"mHC tokens={tokens} max_abs={(actual_full.float()-expected.float()).abs().max().item()}",
                flush=True,
            )
            torch.testing.assert_close(actual_full, expected, atol=1e-2, rtol=1e-2)

    @patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"})
    def test_prefill_v3_preserves_ragged_ties_tail_and_canonicalization(self):
        torch.manual_seed(3)
        width = 66048
        # Separate request K windows, no-valid-K rows, and partial tail groups.
        starts = torch.tensor(
            [0, 0, 128, 32768, 65536], dtype=torch.int32, device="cuda"
        )
        lengths = torch.tensor(
            [0, 1, 700, 32768, 512], dtype=torch.int32, device="cuda"
        )
        scores = torch.randint(0, 10, (5, width), device="cuda").float()
        output = torch.full((5, 512), -7, dtype=torch.int32, device="cuda")
        _run_prefill_topk(
            scores,
            starts,
            starts + lengths,
            output,
            512,
            4,
            backend="topk_v3_tie_break",
        )
        for row in range(5):
            start, length = starts[row].item(), lengths[row].item()
            selected = torch.argsort(
                scores[row, start : start + length], descending=True, stable=True
            )[:512]
            expected = torch.full((512,), -1, device="cuda", dtype=torch.int32)
            expected[: selected.numel()] = selected.sort().values.int()
            torch.testing.assert_close(output[row], expected, atol=0, rtol=0)
        raw_lengths = lengths * 4 + torch.tensor(
            [1, 2, 3, 0, 3], device="cuda", dtype=torch.int32
        )
        expanded = expand_indexer_group_indices(
            output, 4, raw_sequence_lengths=raw_lengths
        )
        combined = append_incomplete_tail_indices(expanded, raw_lengths, 4)
        self.assertEqual(tuple(combined.shape), (5, 2051))
        for row in range(5):
            tail = raw_lengths[row].item() % 4
            self.assertEqual(
                combined[row, -3:][:tail].tolist(),
                list(range(lengths[row].item() * 4, raw_lengths[row].item())),
            )
            self.assertTrue(torch.all(combined[row, -3:][tail:] == -1))

    def test_decode_v3_graph_padded_lengths(self):
        scores = torch.rand(48, 32832, device="cuda")
        lengths = torch.full((48,), 32768, dtype=torch.int32, device="cuda")
        output = torch.empty(48, 512, device="cuda", dtype=torch.int32)
        workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
        rtp_llm_ops.topk_v3(scores, lengths, output, workspace, 512, 32832)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rtp_llm_ops.topk_v3(scores, lengths, output, workspace, 512, 32832)
        lengths[7:] = 0
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.all(output[7:] == -1))
        expected = (
            scores[:7, :32768].topk(512, dim=-1).indices.sort(dim=-1).values.int()
        )
        torch.testing.assert_close(
            output[:7].sort(dim=-1).values, expected, atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
