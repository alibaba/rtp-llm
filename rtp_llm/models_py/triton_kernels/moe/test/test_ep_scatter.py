"""Unit tests for ep_scatter / _fwd_kernel_ep_scatter_1 correctness.

Targets the fix for a cross-warp synchronization bug in _fwd_kernel_ep_scatter_1
where a vector store to expert_start_loc followed by a scalar load could read
stale data when Triton omits the inter-warp barrier (num_experts=256, num_warps=8).

The test verifies:
  1. m_indices is filled with correct expert ids at correct positions
  2. expert_start_loc holds the correct exclusive prefix sum
  3. Runs in a tight loop (stress) to increase race-window hit probability

Run with bazel:
    bazel test //rtp_llm/models_py/triton_kernels/moe/test:test_ep_scatter --config=cuda12
"""

import math
import unittest

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    _fwd_kernel_ep_scatter_1,
    ep_scatter,
)


def align_up(n: int, alignment: int = 128) -> int:
    return int(math.ceil(n / alignment)) * alignment


def reference_scatter_1(
    num_recv_tokens_per_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for _fwd_kernel_ep_scatter_1 outputs."""
    counts = num_recv_tokens_per_expert.cpu().tolist()
    num_experts = len(counts)
    all_tokens = sum(counts)

    expert_start_loc = torch.zeros(num_experts, dtype=torch.int32)
    running = 0
    for i in range(num_experts):
        expert_start_loc[i] = running
        running += counts[i]

    m_indices = torch.full((all_tokens,), -1, dtype=torch.int32)
    for i in range(num_experts):
        start = expert_start_loc[i].item()
        end = start + counts[i]
        m_indices[start:end] = i

    return expert_start_loc, m_indices


@triton.jit
def _old_kernel_output_start(
    num_recv_tokens_per_expert,
    expert_start_loc,
    result_buf,
    num_experts: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offset = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens = tl.load(
        num_recv_tokens_per_expert + offset,
        mask=offset < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens) - tokens
    tl.store(expert_start_loc + offset, cumsum, mask=offset < num_experts)
    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    tl.store(result_buf + cur_expert, cur_expert_start)


class TestEpScatter1Correctness(unittest.TestCase):
    """Verify _fwd_kernel_ep_scatter_1 produces correct m_indices and expert_start_loc."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")

    def _run_kernel(
        self, num_recv_tokens_per_expert: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_experts = num_recv_tokens_per_expert.shape[0]
        all_tokens = int(num_recv_tokens_per_expert.sum().item())
        expert_start_loc = torch.empty(
            num_experts, dtype=torch.int32, device=self.device
        )
        m_indices = torch.full((all_tokens,), -1, dtype=torch.int32, device=self.device)
        BLOCK_E = 128
        _fwd_kernel_ep_scatter_1[(num_experts,)](
            num_recv_tokens_per_expert,
            expert_start_loc,
            m_indices,
            m_indices.numel(),
            num_experts=num_experts,
            num_warps=8,
            BLOCK_E=BLOCK_E,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
            ALIGN_M=1,
        )
        torch.cuda.synchronize()
        return expert_start_loc, m_indices

    def _check(
        self,
        num_experts: int,
        token_counts: list[int],
        label: str = "",
    ) -> None:
        counts_gpu = torch.tensor(token_counts, dtype=torch.int32, device=self.device)
        ref_start, ref_m = reference_scatter_1(counts_gpu)

        got_start, got_m = self._run_kernel(counts_gpu)

        got_start_cpu = got_start.cpu()
        self.assertTrue(
            torch.equal(got_start_cpu, ref_start),
            f"[{label}] expert_start_loc mismatch:\n"
            f"  got={got_start_cpu.tolist()[:16]}...\n"
            f"  ref={ref_start.tolist()[:16]}...",
        )
        got_m_cpu = got_m.cpu()
        m_equal = torch.equal(got_m_cpu, ref_m)
        if not m_equal:
            diff_idx = (got_m_cpu != ref_m).nonzero(as_tuple=True)[0][0].item()
        else:
            diff_idx = "N/A"
        self.assertTrue(
            m_equal,
            f"[{label}] m_indices mismatch (first diff at {diff_idx})",
        )

    def test_small_4_experts(self) -> None:
        self._check(4, [128, 256, 128, 0], "4experts")

    def test_64_experts_uniform(self) -> None:
        self._check(64, [128] * 64, "64experts_uniform")

    def test_256_experts_uniform(self) -> None:
        """Exact config from the crash: 256 experts, 8 warps, BLOCK_EXPERT_NUM=256."""
        self._check(256, [128] * 256, "256experts_uniform")

    def test_256_experts_mixed(self) -> None:
        """256 experts with varying token counts (some zero)."""
        torch.manual_seed(42)
        raw = torch.randint(0, 20, (256,)).tolist()
        aligned = [align_up(x, 128) for x in raw]
        self._check(256, aligned, "256experts_mixed")

    def test_256_experts_sparse(self) -> None:
        """256 experts, most with 0 tokens, a few with large counts."""
        counts = [0] * 256
        for i in [0, 7, 42, 100, 200, 255]:
            counts[i] = 128 * (i % 5 + 1)
        self._check(256, counts, "256experts_sparse")

    def test_256_experts_stress(self) -> None:
        """Repeat 256-expert scatter 200 times to probe race window."""
        counts_gpu = torch.tensor([128] * 256, dtype=torch.int32, device=self.device)
        ref_start, ref_m = reference_scatter_1(counts_gpu)

        for iteration in range(200):
            got_start, got_m = self._run_kernel(counts_gpu)
            if not torch.equal(got_start.cpu(), ref_start):
                self.fail(f"expert_start_loc mismatch at iteration {iteration}")
            if not torch.equal(got_m.cpu(), ref_m):
                self.fail(f"m_indices mismatch at iteration {iteration}")

    def test_aligned_expert_layout(self) -> None:
        """Each expert starts at ALIGN_M while padding remains sentinel-filled."""
        counts_gpu = torch.tensor([3, 5, 2], dtype=torch.int32, device=self.device)
        alignment = 4
        all_tokens = 16
        expert_start_loc = torch.empty(3, dtype=torch.int32, device=self.device)
        m_indices = torch.full((all_tokens,), -1, dtype=torch.int32, device=self.device)

        _fwd_kernel_ep_scatter_1[(3,)](
            counts_gpu,
            expert_start_loc,
            m_indices,
            m_indices.numel(),
            num_experts=3,
            num_warps=4,
            BLOCK_E=128,
            BLOCK_EXPERT_NUM=4,
            ALIGN_M=alignment,
        )
        torch.cuda.synchronize()

        self.assertEqual(expert_start_loc.cpu().tolist(), [0, 4, 12])
        self.assertEqual(
            m_indices.cpu().tolist(),
            [0, 0, 0, -1, 1, 1, 1, 1, 1, -1, -1, -1, 2, 2, -1, -1],
        )

    def test_metadata_counts_cannot_write_past_workspace(self) -> None:
        """Layout follows valid top-k assignments, not inconsistent metadata."""
        alignment = 4
        capacity = 12
        guard_rows = 8
        hidden_size = 128
        sentinel = 37.0
        recv_x = (
            torch.arange(3 * hidden_size, device=self.device, dtype=torch.float32)
            .reshape(3, hidden_size)
            .to(torch.float8_e4m3fn)
        )
        recv_x_scale = torch.ones((3, 1), device=self.device, dtype=torch.float32)
        recv_topk = torch.full((3, 1), 2, device=self.device, dtype=torch.int32)
        inconsistent_counts = torch.tensor(
            [5, 5, 5], device=self.device, dtype=torch.int32
        )
        expert_start_loc = torch.empty(3, device=self.device, dtype=torch.int32)
        output_storage = torch.full(
            (capacity + guard_rows, hidden_size),
            sentinel,
            device=self.device,
            dtype=torch.float32,
        )
        scale_storage = torch.full(
            (capacity + guard_rows, 1),
            sentinel,
            device=self.device,
            dtype=torch.float32,
        )
        m_indices = torch.full((capacity,), -1, device=self.device, dtype=torch.int32)
        output_index = torch.full_like(recv_topk, -1)

        ep_scatter(
            recv_x,
            recv_x_scale,
            recv_topk,
            inconsistent_counts,
            expert_start_loc,
            output_storage[:capacity],
            scale_storage[:capacity],
            m_indices,
            output_index,
            align_m=alignment,
            derive_counts_from_topk=True,
        )
        torch.cuda.synchronize()

        self.assertEqual(output_index.cpu().flatten().sort().values.tolist(), [0, 1, 2])
        self.assertTrue(torch.all(output_storage[capacity:] == sentinel).item())
        self.assertTrue(torch.all(scale_storage[capacity:] == sentinel).item())

    def test_topk_multi_expert_invalid_ids_and_metadata(self) -> None:
        """Top-k layout ignores invalid ids and does not trust metadata counts."""
        alignment = 4
        capacity = 12
        guard_rows = 4
        hidden_size = 128
        sentinel = 37.0
        recv_x = torch.stack(
            [
                torch.full(
                    (hidden_size,),
                    token_id + 1,
                    device=self.device,
                    dtype=torch.float32,
                )
                for token_id in range(4)
            ]
        ).to(torch.float8_e4m3fn)
        recv_x_scale = torch.arange(
            1, 5, device=self.device, dtype=torch.float32
        ).reshape(4, 1)
        recv_topk = torch.tensor(
            [[0, 1], [1, -1], [2, 9], [0, 2]],
            device=self.device,
            dtype=torch.int32,
        )
        inconsistent_counts = torch.tensor(
            [100, 100, 100], device=self.device, dtype=torch.int32
        )
        expert_start_loc = torch.empty(3, device=self.device, dtype=torch.int32)
        output_storage = torch.full(
            (capacity + guard_rows, hidden_size),
            sentinel,
            device=self.device,
            dtype=torch.float32,
        )
        scale_storage = torch.full(
            (capacity + guard_rows, 1),
            sentinel,
            device=self.device,
            dtype=torch.float32,
        )
        m_indices = torch.full((capacity,), -1, device=self.device, dtype=torch.int32)
        output_index = torch.full_like(recv_topk, -1)

        ep_scatter(
            recv_x,
            recv_x_scale,
            recv_topk,
            inconsistent_counts,
            expert_start_loc,
            output_storage[:capacity],
            scale_storage[:capacity],
            m_indices,
            output_index,
            align_m=alignment,
            derive_counts_from_topk=True,
        )
        torch.cuda.synchronize()

        self.assertEqual(expert_start_loc.cpu().tolist(), [2, 6, 10])
        self.assertEqual(
            m_indices.cpu().tolist(),
            [0, 0, -1, -1, 1, 1, -1, -1, 2, 2, -1, -1],
        )
        output_index_cpu = output_index.cpu()
        self.assertEqual(output_index_cpu[1, 1].item(), -1)
        self.assertEqual(output_index_cpu[2, 1].item(), -1)
        expected_destinations = {0: {0, 1}, 1: {4, 5}, 2: {8, 9}}
        observed_destinations = {0: set(), 1: set(), 2: set()}
        for token_id in range(recv_topk.shape[0]):
            for topk_id in range(recv_topk.shape[1]):
                expert_id = recv_topk[token_id, topk_id].item()
                if expert_id not in expected_destinations:
                    continue
                destination = output_index_cpu[token_id, topk_id].item()
                observed_destinations[expert_id].add(destination)
                self.assertTrue(
                    torch.equal(output_storage[destination], recv_x[token_id].float())
                )
                self.assertEqual(
                    scale_storage[destination].item(), recv_x_scale[token_id].item()
                )
        self.assertEqual(observed_destinations, expected_destinations)
        self.assertTrue(torch.all(output_storage[capacity:] == sentinel).item())
        self.assertTrue(torch.all(scale_storage[capacity:] == sentinel).item())


class TestEpScatter1PoisonRegression(unittest.TestCase):
    """Poison-fill regression test for the cross-warp store-load race.

    Pre-fills expert_start_loc with poison, then checks whether the kernel
    reads the correct prefix sum or the stale poison value.

    Old kernel (global memory load): reads poison cross-warp -> mismatch.
    Fixed kernel (register-local read): immune to poison -> always correct.
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")

    def _compute_reference(self, counts: list[int]) -> torch.Tensor:
        ref = torch.zeros(len(counts), dtype=torch.int32)
        running = 0
        for i, c in enumerate(counts):
            ref[i] = running
            running += c
        return ref

    def test_old_kernel_reads_poison(self) -> None:
        """Diagnostic: check if old kernel (global memory load) reads stale
        poison values. This is NOT a CI gate — if Triton compiler is updated
        to insert bar.sync, 0 mismatches is expected and acceptable."""
        num_experts = 256
        counts = [128] * num_experts
        counts_gpu = torch.tensor(counts, dtype=torch.int32, device=self.device)
        ref_start = self._compute_reference(counts)

        total_mismatches = 0
        for _ in range(100):
            expert_start_loc = torch.full(
                (num_experts,), 0x7FFFFFFF, dtype=torch.int32, device=self.device
            )
            result_buf = torch.full(
                (num_experts,), -1, dtype=torch.int32, device=self.device
            )
            _old_kernel_output_start[(num_experts,)](
                counts_gpu,
                expert_start_loc,
                result_buf,
                num_experts=num_experts,
                num_warps=8,
                BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
            )
            torch.cuda.synchronize()
            got = result_buf.cpu()
            total_mismatches += int((got != ref_start).sum().item())

        print(
            f"Old kernel poison diagnostic: {total_mismatches} mismatches in 100 rounds"
        )

    def test_new_kernel_immune_to_poison(self) -> None:
        """Production kernel (fixed) + poison fill: expert_start_loc and
        m_indices must always match the golden reference."""
        num_experts = 256
        counts = [1 + (i % 17) for i in range(num_experts)]
        align_m = 128
        counts_gpu = torch.tensor(counts, dtype=torch.int32, device=self.device)
        aligned_counts = [
            ((count + align_m - 1) // align_m) * align_m for count in counts
        ]
        all_tokens = sum(aligned_counts)
        ref_start = torch.zeros(num_experts, dtype=torch.int32)
        ref_m = torch.full((all_tokens,), -1, dtype=torch.int32)
        running = 0
        for expert_id, count in enumerate(counts):
            ref_start[expert_id] = running
            ref_m[running : running + count] = expert_id
            running += aligned_counts[expert_id]

        for round_i in range(100):
            expert_start_loc = torch.full(
                (num_experts,), 0x7FFFFFFF, dtype=torch.int32, device=self.device
            )
            m_indices = torch.full(
                (all_tokens,), -1, dtype=torch.int32, device=self.device
            )
            _fwd_kernel_ep_scatter_1[(num_experts,)](
                counts_gpu,
                expert_start_loc,
                m_indices,
                m_indices.numel(),
                num_experts=num_experts,
                num_warps=8,
                BLOCK_E=128,
                BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
                ALIGN_M=align_m,
            )
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(expert_start_loc.cpu(), ref_start),
                f"Round {round_i}: expert_start_loc mismatch with golden",
            )
            self.assertTrue(
                torch.equal(m_indices.cpu(), ref_m),
                f"Round {round_i}: m_indices mismatch with golden. "
                f"Production kernel may have been reverted to the buggy "
                f"global-memory-load pattern.",
            )


if __name__ == "__main__":
    unittest.main()
