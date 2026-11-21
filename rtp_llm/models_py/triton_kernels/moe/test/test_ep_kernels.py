"""Unit tests for ep_kernels.recompute_topk_ids_sum_expert_count.

Pins the -1 sentinel contract that PureDpRouter relies on when padding
short batches before allgather:
- input slots equal to -1 must NOT be counted in expert_count
- output adjusted_topk_ids must keep -1 unchanged in those slots
- out-of-range expert ids collapse to -1 the same way

Run with bazel:
    bazel test //rtp_llm/models_py/triton_kernels/moe/test:test_ep_kernels --config=cuda13
"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    ep_scatter_v2,
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


@unittest.skipUnless(torch.cuda.is_available(), "requires a Triton GPU")
class TestEPRoutedWidths(unittest.TestCase):
    def test_weighted_gather_and_tail_guards(self):
        ids = torch.tensor([[0, 1], [1, -1], [0, 1], [-1, -1]], device="cuda")
        rows = torch.tensor([[2, 1], [0, 3], [-1, 99], [0, 2]], device="cuda")
        weights = torch.tensor(
            [[0.25, 0.75], [-0.5, 1], [1, 2], [3, 4]], device="cuda"
        )
        for width in (0, 128, 384, 512, 513, 1024):
            with self.subTest(width=width):
                src = torch.arange(3 * width, device="cuda").reshape(3, width)
                src = (src % 31 - 15).bfloat16()
                # Deliberately give every row a canary beyond its logical end.
                storage = torch.full(
                    (4, width + 512), -512, dtype=torch.bfloat16, device="cuda"
                )
                dst = storage[:, :width]
                ep_gather(src, ids, weights, rows, dst)
                expected = torch.zeros_like(dst)
                expected[0] = src[2].float() * 0.25 + src[1].float() * 0.75
                expected[1] = src[0].float() * -0.5
                torch.testing.assert_close(dst, expected, rtol=0, atol=0)
                self.assertTrue((storage[:, width:] == -512).all().item())

    def test_scatter_copies_complete_tail_group(self):
        ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32, device="cuda")
        counts = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
        for width in (128, 512, 513):
            data = torch.arange(4 * width, device="cuda").reshape(4, width)
            data = (data % 13 - 6).to(torch.float8_e4m3fn)
            for packed in (False, True):
                groups = (width + 127) // 128
                scale_size = (groups + 3) // 4 if packed else groups
                scales = torch.arange(4 * scale_size, device="cuda").reshape(
                    4, scale_size
                )
                scales = scales.to(torch.int32 if packed else torch.float32) + 1
                for masked_layout in (False, True):
                    with self.subTest(
                        width=width, packed=packed, masked_layout=masked_layout
                    ):
                        index = torch.full_like(ids, -1)
                        starts = torch.zeros_like(counts)
                        out = torch.empty((256, width), dtype=data.dtype, device="cuda")
                        out_scales = torch.zeros(
                            (256, scale_size), dtype=scales.dtype, device="cuda"
                        )
                        if masked_layout:
                            ep_scatter_v2(
                                data, scales, ids, 128, starts, out,
                                out_scales.view(2, 128, scale_size), index,
                                scale_ue8m0=packed,
                            )
                        else:
                            m_indices = torch.full(
                                (256,), -1, dtype=torch.int32, device="cuda"
                            )
                            ep_scatter(
                                data, scales, ids, counts, starts, out, out_scales,
                                m_indices, index, scale_ue8m0=packed,
                            )
                        rows = index.flatten().long()
                        torch.testing.assert_close(
                            out.view(torch.uint8)[rows], data.view(torch.uint8),
                            rtol=0, atol=0,
                        )
                        torch.testing.assert_close(
                            out_scales[rows], scales, rtol=0, atol=0
                        )


if __name__ == "__main__":
    unittest.main()
