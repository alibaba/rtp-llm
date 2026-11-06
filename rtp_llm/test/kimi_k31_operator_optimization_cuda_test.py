"""Contracts for K3.1's bounded exact selector, strided norm and cache writer."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.indexer_topk import (
    NativeExactIndexerTopKBackend,
    TorchIndexerTopKBackend,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
    fused_q_kv_rmsnorm,
    grouped_q_kv_rmsnorm,
)
from rtp_llm.ops import compute_ops


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class KimiK31OperatorOptimizationTest(unittest.TestCase):
    def assert_candidate_sets_equal(self, actual, expected):
        # Compare exact members and their associated scores, without requiring
        # an order that sparse attention does not consume.
        actual_order = actual[1].argsort(dim=-1)
        expected_order = expected[1].argsort(dim=-1)
        for lhs, rhs in zip(actual, expected):
            torch.testing.assert_close(
                lhs.gather(-1, actual_order), rhs.gather(-1, expected_order),
                rtol=0, atol=0,
            )

    def test_exact_topk_boundary_ties_and_graph(self):
        torch.manual_seed(3101)
        old, new = TorchIndexerTopKBackend(), NativeExactIndexerTopKBackend()
        positions = torch.randperm(8193, device="cuda").to(torch.int32)
        score = torch.randn((7, 8193), device="cuda").floor()
        score[0].zero_()
        score[1].fill_(-torch.inf)
        score[2, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf], device="cuda")
        score[3, :8000] = -torch.inf
        score[4, :4096] = 1e9
        positions[::31] = -1
        reference = old.select_candidates(score.clone(), positions, 2048)
        output = tuple(torch.empty_like(t) for t in reference)
        with patch("torch.topk", side_effect=AssertionError("redundant candidate TopK")):
            actual = new.select_candidates(score.clone(), positions, 2048, out=output)
        for lhs, rhs in zip(actual, output):
            self.assertIs(lhs, rhs)
        self.assert_candidate_sets_equal(actual, reference)
        graph_score = score.clone()
        new.select_candidates(graph_score, positions, 2048)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_result = new.select_candidates(graph_score, positions, 2048)
        graph_score.copy_(score + 1)
        graph.replay()
        torch.cuda.synchronize()
        replay_reference = old.select_candidates(graph_score, positions, 2048)
        self.assert_candidate_sets_equal(replay_result, replay_reference)
        # A running merge sees only 2K candidates, retaining the same tie rule.
        second_positions = torch.where(positions >= 0, positions + 10000, positions)
        second = new.select_candidates(score.clone(), second_positions, 2048)
        running = tuple(t.clone() for t in actual)
        reference_running = tuple(t.clone() for t in reference)
        new.merge_candidates(running, second)
        old.merge_candidates(reference_running, second)
        self.assert_candidate_sets_equal(running, reference_running)

    def test_page_rr_topk_matches_dense_bounds_ties_and_graph(self):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        torch.manual_seed(712)
        device = "cuda"
        k, rows = 512, 8
        for shards, page, width in ((1, 128, 1024), (4, 128, 2048),
                                    (8, 1024, 2048), (4, 1024, 262144)):
            with self.subTest(shards=shards, page=page, width=width):
                rank, start, capacity = shards - 1, page * 3, shards * width
                first = start * shards
                dense = torch.randn((rows, capacity), device=device).round_()
                dense[:, :4] = torch.tensor(
                    [torch.nan, torch.inf, -torch.inf, -0.0], device=device
                )
                # The logical columns alternate owners one page at a time.
                received = (dense.view(rows, width // page, shards, page)
                            .permute(2, 0, 1, 3).contiguous().view(rows * shards, width))
                positions = torch.arange(first, first + capacity, dtype=torch.int32, device=device)
                query = torch.full((rows * shards - 1,), -1, dtype=torch.int32, device=device)
                visible = query[rank * rows:]
                lengths = torch.tensor([0, 1, k - 1, k, k + 1, capacity - 1, capacity],
                                       dtype=torch.int32, device=device)
                visible.copy_(first + lengths - 1)
                old_scores = torch.empty((rows, k), device=device)
                old_pos = torch.empty((rows, k), dtype=torch.int32, device=device)
                new_scores, new_pos = torch.empty_like(old_scores), torch.empty_like(old_pos)

                def select_direct():
                    rtp_llm_ops.dsa_select_page_rr_candidates(
                        received, query, new_scores, new_pos, k, shards, rank, page, start
                    )

                def compare():
                    ends = torch.cat((lengths, lengths.new_zeros(1)))
                    rtp_llm_ops.dsa_select_topk_candidates(
                        dense, positions, None, ends, old_scores, old_pos, k
                    )
                    short = ends <= k
                    self.assertTrue(torch.equal(old_pos[short], new_pos[short]))
                    self.assertTrue(torch.equal(old_scores.view(torch.int32)[short],
                                                new_scores.view(torch.int32)[short]))
                    for values, output in ((old_scores, old_pos), (new_scores, new_pos)):
                        order = output.argsort(dim=1)
                        pairs = (output.gather(1, order), values.view(torch.int32).gather(1, order))
                        if output is old_pos:
                            expected = pairs
                        else:
                            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, pairs)))

                select_direct()
                compare()
                if shards == 8:
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        select_direct()
                    lengths.copy_(lengths.roll(1))
                    visible.copy_(first + lengths - 1)
                    graph.replay()
                    compare()

    def test_grouped_norm_strides_padding_and_graph(self):
        torch.manual_seed(3101)
        qw = torch.rand((1536,), device="cuda", dtype=torch.bfloat16) + 0.5
        kw = torch.rand((512,), device="cuda", dtype=torch.bfloat16) + 0.5
        for rows in (0, 1, 127, 128, 129, 8192):
            storage = torch.randn((rows, 2112), device="cuda", dtype=torch.bfloat16)
            q, kv = storage[:, :1536], storage[:, 1536:2048]
            actual = grouped_q_kv_rmsnorm(q, kv, qw, kw, 1e-5)
            for y, x, weight in zip(actual, (q, kv), (qw, kw)):
                reference = (
                    x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
                    * weight.float()
                ).bfloat16()
                if rows:
                    ulp = (y.view(torch.int16).int() - reference.view(torch.int16).int()).abs()
                    self.assertLessEqual(int(ulp.max()), 1)
            if rows == 129:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay_result = grouped_q_kv_rmsnorm(q, kv, qw, kw, 1e-5)
                storage.add_(0.25)
                graph.replay()
                torch.cuda.synchronize()
                eager_result = grouped_q_kv_rmsnorm(q, kv, qw, kw, 1e-5)
                for lhs, rhs in zip(replay_result, eager_result):
                    self.assertTrue(torch.equal(lhs, rhs))
            # Keep legacy Decode and non-K3 dimensions unchanged.
            if rows < 128:
                legacy = fused_q_kv_rmsnorm(q, kv, qw, kw, 1e-5)
                for lhs, rhs in zip(actual, legacy):
                    self.assertTrue(torch.equal(lhs, rhs))

    def test_cache_writer_page_strides_slot_zero_and_graph(self):
        scale = torch.ones((), device="cuda", dtype=torch.float32)
        for rows in (0, 1, 127, 128, 129, 2049):
            for entry_padding in (0, 8, 9):
                storage = torch.randn((rows, 2112), device="cuda", dtype=torch.bfloat16)
                latent, suffix = storage[:, 1536:2048], storage[:, 2048:]
                pages, entry = (rows + 255) // 128 + 1, 576 + entry_padding
                cache = torch.empty_strided(
                    (pages, 128, 576), (128 * entry + 8, entry, 1),
                    device="cuda", dtype=torch.bfloat16,
                ).fill_(-3)
                reference = cache.clone()
                slots = torch.randperm(pages * 128, device="cuda")[:rows].long()
                slots[1::7] = -1
                if rows:
                    slots[slots == 0] = -1
                    slots[0] = 0
                valid = slots >= 0
                selected = slots[valid]
                reference[selected // 128, selected % 128] = torch.cat(
                    (latent[valid], suffix[valid]), dim=-1
                )
                compute_ops.concat_and_cache_mla(latent, suffix, cache, slots, "auto", scale)
                self.assertTrue(torch.equal(cache, reference))
                if rows == 129 and entry_padding == 8:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        compute_ops.concat_and_cache_mla(latent, suffix, cache, slots, "auto", scale)
                    storage.add_(1)
                    reference.fill_(-3)
                    reference[selected // 128, selected % 128] = torch.cat(
                        (latent[valid], suffix[valid]), dim=-1
                    )
                    cache.fill_(-3)
                    graph.replay()
                    self.assertTrue(torch.equal(cache, reference))


if __name__ == "__main__":
    unittest.main()
