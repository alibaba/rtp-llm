"""Unit tests for ep_kernels.recompute_topk_ids_sum_expert_count and ep_gather.

recompute_topk_ids_sum_expert_count pins the -1 sentinel contract that
PureDpRouter relies on when padding short batches before allgather:
- input slots equal to -1 must NOT be counted in expert_count
- output adjusted_topk_ids must keep -1 unchanged in those slots
- out-of-range expert ids collapse to -1 the same way

ep_gather pins the shared-expert accumulate contract used by
GenericMoeLayer/DeepGemmHybridExecutor to remove the separate
``experts_output + shared_expert_output`` bf16 add:
- ``ep_gather(..., accumulate_input=shared)`` must be bitwise identical to
  ``ep_gather(...)`` followed by the aten add (same two RTNE roundings)
- the fold must actually change the output (not compile out)
- strides/dtypes/shape edge cases must hold

Run with bazel:
    bazel test //rtp_llm/models_py/triton_kernels/moe/test:test_ep_kernels --config=cuda13
"""

import ctypes
import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
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


class TestRecomputeTopkIdsSelfZero(unittest.TestCase):
    """Single-program self-zero path of recompute_topk_ids_sum_expert_count.

    num_total <= BLOCK_SIZE launches a single program, so the kernel zeroes
    the expert_count cells itself (torch.empty workspace) instead of a
    separate torch.zeros fill kernel.  This is the decode CUDA-graph hot
    path (one fill per MoE layer per step).  These tests pin:
    - the self-zeroed counters match the reference for grid == 1 shapes,
      including count values that are 0 and a workspace backed by poisoned
      (0xFF) allocator memory;
    - the grid == 1 / grid > 1 boundary follows num_total vs BLOCK_SIZE;
    - the fill-kernel launch actually disappears (CUDA-graph capture node
      census, the established CUPTI-free method in this tree) while the
      multi-program path keeps exactly one fill.
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for triton kernel test")
        self.device = torch.device("cuda")

    def _ref_count(
        self, topk_ids: torch.Tensor, start: int, num_local: int
    ) -> torch.Tensor:
        adj = topk_ids - start
        valid = (topk_ids != -1) & (adj >= 0) & (adj < num_local)
        flat = adj[valid].to(torch.int32)
        out = torch.zeros(num_local, dtype=torch.int32, device=topk_ids.device)
        if flat.numel() > 0:
            out.scatter_add_(
                0, flat.to(torch.int64), torch.ones_like(flat, dtype=torch.int32)
            )
        return out

    def _block_size(self, topk: int) -> int:
        import triton

        return triton.next_power_of_2(min(256, 1024 // max(topk, 1)))

    def test_grid_one_matches_reference(self) -> None:
        """grid == 1 shapes: self-zeroed counters equal the reference."""
        torch.manual_seed(7)
        cases = [
            # (num_tokens, topk, num_local, start, num_total_experts)
            (1, 4, 32, 32, 64),   # Xing4-style decode step: num_total 4
            (3, 4, 32, 32, 64),   # MTP draft q_len 3
            (64, 4, 32, 0, 32),   # num_total 256 == BLOCK: boundary
            (16, 8, 8, 0, 64),    # topk 8 -> BLOCK 128
            (1, 1, 5, 0, 5),
            (10, 2, 257, 512, 1024),  # non-pow2 expert count
        ]
        for num_tokens, topk, num_local, start, num_total_experts in cases:
            block = self._block_size(topk)
            num_total = num_tokens * topk
            self.assertLessEqual(
                num_total, block, "case must launch a single program"
            )
            topk_ids = torch.randint(
                -1, num_total_experts, (num_tokens, topk),
                dtype=torch.int32, device=self.device,
            )
            topk_ids[torch.rand_like(topk_ids.float()) < 0.25] = -1
            adjusted, count = recompute_topk_ids_sum_expert_count(
                topk_ids, start, num_local
            )
            ref_count = self._ref_count(topk_ids, start, num_local)
            self.assertTrue(
                torch.equal(count, ref_count),
                f"count mismatch for tokens={num_tokens} topk={topk} "
                f"local={num_local}:\n  got={count}\n  want={ref_count}",
            )
            # counts stay dense: every cell is written (zero or accumulated)
            self.assertEqual(count.numel(), num_local)

    def test_grid_one_poisoned_workspace(self) -> None:
        """0xFF-backed workspace memory must not leak into the counters."""
        num_tokens, topk, num_local, start = 32, 4, 64, 64
        topk_ids = torch.full(
            (num_tokens, topk), -1, dtype=torch.int32, device=self.device
        )
        # Only expert (start + 0) is routed: every other cell must come back
        # as an exact zero written by the in-kernel zeroing phase.
        topk_ids[0, 0] = start
        # Poison the caching allocator so the torch.empty workspace reuses
        # all-ones memory.
        poison = torch.full((num_local,), 0x7F7F7F7F, dtype=torch.int32,
                            device=self.device)
        del poison
        adjusted, count = recompute_topk_ids_sum_expert_count(
            topk_ids, start, num_local
        )
        expected = torch.zeros(num_local, dtype=torch.int32, device=self.device)
        expected[0] = 1
        self.assertTrue(torch.equal(count, expected))
        # Only slot (0, 0) is routed: it remaps to local expert 0, every
        # other slot keeps the -1 sentinel.
        expected_adj = torch.full_like(topk_ids, -1)
        expected_adj[0, 0] = 0
        self.assertTrue(torch.equal(adjusted, expected_adj))

    def test_grid_boundary_follows_block_size(self) -> None:
        """num_total == BLOCK -> single program; BLOCK + 1 -> two programs."""
        topk = 4
        block = self._block_size(topk)
        start, num_local = 0, 8

        at_boundary = torch.randint(
            0, num_local, (block // topk, topk), dtype=torch.int32,
            device=self.device,
        )
        _, count = recompute_topk_ids_sum_expert_count(
            at_boundary, start, num_local
        )
        self.assertTrue(torch.equal(count, self._ref_count(at_boundary, start, num_local)))

        beyond = torch.randint(
            0, num_local, (block // topk + 1, topk), dtype=torch.int32,
            device=self.device,
        )
        _, count = recompute_topk_ids_sum_expert_count(
            beyond, start, num_local
        )
        self.assertTrue(torch.equal(count, self._ref_count(beyond, start, num_local)))

    def test_fill_kernel_launch_count(self) -> None:
        """Launch census: grid == 1 drops the fill; grid > 1 keeps it.

        CUPTI activity tracing is unavailable in the CUDA13 dev containers,
        so launches are counted by CUDA-graph capture (the established
        method in this tree): every launch becomes exactly one graph node,
        enumerated with cuGraphGetNodes/cuGraphNodeGetType.  Node type 0 is
        CUgraphNodeTypeKernel; torch.zeros lowers to one FillFunctor kernel.
        """

        def graph_kernel_nodes(fn) -> int:
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                fn()
            torch.cuda.current_stream().wait_stream(side)
            with torch.cuda.graph(graph):
                fn()
            handle = graph.raw_cuda_graph()
            cuda = ctypes.CDLL("libcuda.so.1")
            count = ctypes.c_size_t(0)
            assert cuda.cuGraphGetNodes(
                ctypes.c_void_p(handle), None, ctypes.byref(count)
            ) == 0
            nodes = (ctypes.c_void_p * count.value)()
            assert cuda.cuGraphGetNodes(
                ctypes.c_void_p(handle), nodes, ctypes.byref(count)
            ) == 0
            kernels = 0
            for i in range(count.value):
                node_type = ctypes.c_int(0)
                assert cuda.cuGraphNodeGetType(
                    ctypes.c_void_p(nodes[i]), ctypes.byref(node_type)
                ) == 0
                self.assertEqual(node_type.value, 0, "expected only kernel nodes")
                kernels += 1
            return kernels

        topk = 4
        block = self._block_size(topk)
        num_local, start = 32, 0
        ids_single = torch.randint(
            0, num_local, (block // topk, topk), dtype=torch.int32,
            device=self.device,
        )
        ids_multi = torch.randint(
            0, num_local, (block * 8 // topk, topk), dtype=torch.int32,
            device=self.device,
        )
        # num_total == BLOCK -> exactly the recompute kernel, no fill.
        self.assertEqual(
            graph_kernel_nodes(lambda: recompute_topk_ids_sum_expert_count(
                ids_single, start, num_local)),
            1,
            "single-program path must launch only the recompute kernel",
        )
        # num_total == 8 * BLOCK -> pre-fill kernel + recompute kernel.
        self.assertEqual(
            graph_kernel_nodes(lambda: recompute_topk_ids_sum_expert_count(
                ids_multi, start, num_local)),
            2,
            "multi-program path must keep the torch.zeros pre-fill",
        )


class TestEpGatherAccumulate(unittest.TestCase):
    """Bitwise equivalence of ep_gather's shared-expert accumulate fold."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for triton kernel test")
        self.device = torch.device("cuda")

    def _make_case(
        self,
        num_tokens: int,
        topk: int,
        hidden: int,
        num_src: int,
        *,
        seed: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        ids_dtype: torch.dtype = torch.int64,
        pad_probability: float = 0.2,
    ):
        """Build gather inputs plus a torch fp32-order reference."""

        generator = torch.Generator(device="cpu").manual_seed(seed)
        input_tensor = (
            (torch.randn(num_src, hidden, generator=generator) * 0.5)
            .to(self.device)
            .to(dtype)
        )
        topk_ids = (
            torch.randint(0, num_src, (num_tokens, topk), generator=generator)
            .to(ids_dtype)
            .to(self.device)
        )
        topk_weights = torch.rand(
            (num_tokens, topk), generator=generator, dtype=torch.float32
        ).to(self.device)
        # -1 id slots are skipped by the kernel (sentinel padding).
        invalid = (
            torch.rand(num_tokens, topk, generator=generator) < pad_probability
        ).to(self.device)
        topk_ids = torch.where(invalid, torch.full_like(topk_ids, -1), topk_ids)
        # input_index: -1 slots stay skipped (source row bounds check).
        input_index = (
            torch.randint(0, num_src, (num_tokens, topk), generator=generator)
            .to(ids_dtype)
            .to(self.device)
        )
        input_index = torch.where(
            invalid, torch.full_like(input_index, -1), input_index
        )
        shared = (
            (torch.randn(num_tokens, hidden, generator=generator) * 0.5)
            .to(self.device)
            .to(dtype)
        )
        # fp32 reference of the gather accumulation, in topk order.
        ref_fp32 = torch.zeros(num_tokens, hidden, device=self.device, dtype=torch.float32)
        for k in range(topk):
            valid = topk_ids[:, k] >= 0
            rows = input_index[valid, k]
            weights = topk_weights[valid, k]
            ref_fp32[valid] += (
                input_tensor[rows].to(torch.float32) * weights.unsqueeze(1)
            )
        return input_tensor, topk_ids, topk_weights, input_index, shared, ref_fp32

    def _run_gather(self, case, output, accumulate=None):
        input_tensor, topk_ids, topk_weights, input_index, _, _ = case
        ep_gather(
            input_tensor,
            topk_ids,
            topk_weights,
            input_index,
            output,
            accumulate_input=accumulate,
        )
        return output

    def test_accumulate_is_bitwise_identical_to_separate_add(self) -> None:
        """Folded store == unfused gather output + shared, rounding for rounding.

        The claim of the shared-expert fold is exact: the kernel rounds its
        fp32 accumulator to the output dtype, widens both operands to fp32,
        adds, and rounds once more — the same two round-to-nearest-even steps
        the separate kernels perform.
        """
        shapes = (
            (1, 1, 512, 3),  # single token, single expert
            (7, 4, 512, 33),  # padding sentinels in play
            (64, 8, 1024, 512),
            (1500, 4, 1024, 4096),  # grid-stride loop over > 1024 tokens
            (3, 4, 2048, 17),  # several hidden blocks
        )
        for num_tokens, topk, hidden, num_src in shapes:
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(
                    num_tokens=num_tokens,
                    topk=topk,
                    hidden=hidden,
                    dtype=dtype,
                ):
                    case = self._make_case(
                        num_tokens, topk, hidden, num_src, seed=1234, dtype=dtype
                    )
                    shared = case[4]
                    base = torch.empty(
                        num_tokens, hidden, device=self.device, dtype=dtype
                    )
                    fused = torch.empty_like(base)
                    self._run_gather(case, base)
                    self._run_gather(case, fused, accumulate=shared)

                    expected = base + shared
                    self.assertTrue(
                        torch.equal(fused, expected),
                        "folded ep_gather output must be bitwise identical "
                        "to the unfused gather + aten add",
                    )
                    # The fold must actually engage: with nonzero shared
                    # values the fused output must differ from the base.
                    self.assertFalse(
                        torch.equal(fused, base),
                        "accumulate_input was not applied by the kernel",
                    )
                    # Wiring sanity: the unfused output matches an
                    # independent fp32 reference (the separate-add claim is
                    # only meaningful when the gather itself is correct).
                    ref_bf16 = case[5].to(dtype)
                    self.assertTrue(
                        torch.allclose(
                            base.float(), ref_bf16.float(), rtol=0.02, atol=0.02
                        ),
                        "gather output deviates from the fp32 reference",
                    )

    def test_accumulate_with_int32_ids_and_full_padding_row(self) -> None:
        """int32 routing ids plus an all-sentinel token row ( routed sum 0 )."""
        num_tokens, topk, hidden, num_src = 9, 4, 512, 25
        case = self._make_case(
            num_tokens, topk, hidden, num_src, seed=7, ids_dtype=torch.int32
        )
        _, topk_ids, _, input_index, shared, _ = case
        topk_ids[3, :] = -1
        input_index[3, :] = -1
        base = torch.empty(num_tokens, hidden, device=self.device, dtype=torch.bfloat16)
        fused = torch.empty_like(base)
        self._run_gather(case, base)
        self._run_gather(case, fused, accumulate=shared)

        self.assertTrue(
            torch.equal(base[3], torch.zeros_like(base[3])),
            "all-sentinel row must gather to zero",
        )
        self.assertTrue(
            torch.equal(fused, base + shared),
            "int32 ids + padding row must stay bitwise identical",
        )
        # A zero routed sum folds to exactly the shared values.
        self.assertTrue(torch.equal(fused[3], shared[3]))

    def test_zero_shared_is_identity(self) -> None:
        """A zero accumulate input must reproduce the unfused output."""
        case = self._make_case(11, 4, 1024, 40, seed=99)
        num_tokens, hidden = 11, 1024
        zeros = torch.zeros(
            num_tokens, hidden, device=self.device, dtype=torch.bfloat16
        )
        base = torch.empty(num_tokens, hidden, device=self.device, dtype=torch.bfloat16)
        fused = torch.empty_like(base)
        self._run_gather(case, base)
        self._run_gather(case, fused, accumulate=zeros)
        self.assertTrue(torch.equal(fused, base))

    def test_non_contiguous_shared_row_stride(self) -> None:
        """A shared tensor sliced from a wider buffer (row stride > hidden)."""
        num_tokens, topk, hidden, num_src = 13, 4, 512, 29
        case = self._make_case(num_tokens, topk, hidden, num_src, seed=55)
        wide = torch.randn(
            num_tokens, hidden * 2, device=self.device, dtype=torch.bfloat16
        )
        shared = wide[:, :hidden]
        self.assertNotEqual(shared.stride(0), hidden)

        base = torch.empty(num_tokens, hidden, device=self.device, dtype=torch.bfloat16)
        fused = torch.empty_like(base)
        self._run_gather(case, base)
        self._run_gather(case, fused, accumulate=shared)
        self.assertTrue(torch.equal(fused, base + shared))

    def test_accumulate_without_request_matches_legacy_call(self) -> None:
        """The default call (no accumulate) must be unchanged behavior."""
        case = self._make_case(5, 2, 512, 19, seed=31)
        num_tokens, hidden = 5, 512
        legacy = torch.empty(num_tokens, hidden, device=self.device, dtype=torch.bfloat16)
        explicit = torch.empty_like(legacy)
        input_tensor, topk_ids, topk_weights, input_index, _, _ = case
        ep_gather(input_tensor, topk_ids, topk_weights, input_index, legacy)
        ep_gather(
            input_tensor,
            topk_ids,
            topk_weights,
            input_index,
            explicit,
            accumulate_input=None,
        )
        self.assertTrue(torch.equal(legacy, explicit))


if __name__ == "__main__":
    unittest.main()
