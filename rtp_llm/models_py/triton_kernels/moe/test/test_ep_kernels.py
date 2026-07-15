"""Unit tests for ep_kernels.recompute_topk_ids_sum_expert_count.

Pins the -1 sentinel contract that PureDpRouter relies on when padding
short batches before allgather:
- input slots equal to -1 must NOT be counted in expert_count
- output adjusted_topk_ids must keep -1 unchanged in those slots
- out-of-range expert ids collapse to -1 the same way

Run with bazel:
    bazel test //rtp_llm/models_py/triton_kernels/moe/test:test_ep_kernels --config=cuda12_9
"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    recompute_topk_ids_sum_expert_count,
)


class TestRecomputeTopkIdsSentinel(unittest.TestCase):
    """-1 sentinel contract for recompute_topk_ids_sum_expert_count."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for triton kernel test")
        self.device = torch.device("cuda")

    def _ref_count(
        self, topk_ids: torch.Tensor, start: int, num_local: int
    ) -> torch.Tensor:
        """Reference: count tokens per local expert, ignoring -1 slots."""
        adj = topk_ids - start
        valid = (topk_ids != -1) & (adj >= 0) & (adj < num_local)
        flat = adj[valid].to(torch.int32)
        out = torch.zeros(num_local, dtype=torch.int32, device=topk_ids.device)
        if flat.numel() > 0:
            out.scatter_add_(
                0, flat.to(torch.int64), torch.ones_like(flat, dtype=torch.int32)
            )
        return out

    def test_sentinel_rows_not_counted(self) -> None:
        """A row of all -1 contributes nothing to expert_count."""
        # 4 tokens, topk=2; rank 0 owns experts [0, 4)
        start, num_local = 0, 4
        topk_ids = torch.tensor(
            [
                [0, 1],
                [-1, -1],  # padded row
                [2, 3],
                [-1, -1],  # padded row
            ],
            dtype=torch.int32,
            device=self.device,
        )

        adjusted, count = recompute_topk_ids_sum_expert_count(
            topk_ids, start, num_local
        )

        # adjusted: -1 rows preserved, others remapped (start=0 → identity)
        expected_adj = topk_ids.clone()
        self.assertTrue(torch.equal(adjusted, expected_adj))

        # count: each of experts [0,1,2,3] gets exactly 1, padded rows ignored
        expected_count = torch.tensor([1, 1, 1, 1], dtype=torch.int32, device=self.device)
        self.assertTrue(torch.equal(count, expected_count))

    def test_mixed_sentinel_and_out_of_range(self) -> None:
        """-1 sentinels and out-of-range ids both collapse to -1, neither counted."""
        # rank 1 owns experts [4, 8) — adjusted by start=4, num_local=4
        start, num_local = 4, 4
        topk_ids = torch.tensor(
            [
                [4, 5],   # both local → adjusted (0, 1)
                [0, 6],   # 0 is remote (out of range) → -1; 6 → adjusted 2
                [-1, 7],  # sentinel + adjusted 3
                [-1, -1], # padded row
            ],
            dtype=torch.int32,
            device=self.device,
        )

        adjusted, count = recompute_topk_ids_sum_expert_count(
            topk_ids, start, num_local
        )

        # adjusted: locals remapped, remotes/sentinels → -1
        expected_adj = torch.tensor(
            [
                [0, 1],
                [-1, 2],
                [-1, 3],
                [-1, -1],
            ],
            dtype=torch.int32,
            device=self.device,
        )
        self.assertTrue(
            torch.equal(adjusted, expected_adj),
            f"adjusted mismatch:\n  got={adjusted}\n  want={expected_adj}",
        )

        # count: experts 0,1,2,3 each get exactly 1; out-of-range and -1 contribute 0
        expected_count = torch.tensor([1, 1, 1, 1], dtype=torch.int32, device=self.device)
        self.assertTrue(
            torch.equal(count, expected_count),
            f"count mismatch:\n  got={count}\n  want={expected_count}",
        )

    def test_all_sentinel_input(self) -> None:
        """All-sentinel input → adjusted preserves -1, count is all zero."""
        start, num_local = 0, 8
        topk_ids = torch.full(
            (5, 4), -1, dtype=torch.int32, device=self.device
        )

        adjusted, count = recompute_topk_ids_sum_expert_count(
            topk_ids, start, num_local
        )

        self.assertTrue(torch.equal(adjusted, topk_ids))
        self.assertTrue(
            torch.equal(
                count,
                torch.zeros(num_local, dtype=torch.int32, device=self.device),
            )
        )

    def test_matches_reference_random(self) -> None:
        """Random input with sentinels: kernel matches CPU reference."""
        torch.manual_seed(0)
        num_tokens, topk = 64, 8
        num_total_experts, num_local = 32, 8
        start = 16  # rank 2 owns [16, 24)

        topk_ids = torch.randint(
            0, num_total_experts, (num_tokens, topk), dtype=torch.int32, device=self.device
        )
        # Inject -1 sentinels into random positions
        sentinel_mask = torch.rand(num_tokens, topk, device=self.device) < 0.2
        topk_ids[sentinel_mask] = -1

        adjusted, count = recompute_topk_ids_sum_expert_count(
            topk_ids, start, num_local
        )
        ref_count = self._ref_count(topk_ids, start, num_local)

        # Sentinel rows in input must remain -1 in output
        self.assertTrue(
            (adjusted[sentinel_mask] == -1).all(),
            "sentinel slots must stay -1 in adjusted output",
        )
        # Count matches reference (which excludes both -1 and out-of-range)
        self.assertTrue(
            torch.equal(count, ref_count),
            f"count mismatch:\n  got={count}\n  want={ref_count}",
        )


if __name__ == "__main__":
    unittest.main()
