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
        mask=offset < num_experts, other=0,
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
        m_indices = torch.full(
            (all_tokens,), -1, dtype=torch.int32, device=self.device
        )
        BLOCK_E = 128
        _fwd_kernel_ep_scatter_1[(num_experts,)](
            num_recv_tokens_per_expert,
            expert_start_loc,
            m_indices,
            num_experts=num_experts,
            num_warps=8,
            BLOCK_E=BLOCK_E,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        )
        torch.cuda.synchronize()
        return expert_start_loc, m_indices

    def _check(
        self,
        num_experts: int,
        token_counts: list[int],
        label: str = "",
    ) -> None:
        counts_gpu = torch.tensor(
            token_counts, dtype=torch.int32, device=self.device
        )
        ref_start, ref_m = reference_scatter_1(counts_gpu)

        got_start, got_m = self._run_kernel(counts_gpu)

        self.assertTrue(
            torch.equal(got_start.cpu(), ref_start),
            f"[{label}] expert_start_loc mismatch:\n  got={got_start.cpu().tolist()[:16]}...\n  ref={ref_start.tolist()[:16]}...",
        )
        self.assertTrue(
            torch.equal(got_m.cpu(), ref_m),
            f"[{label}] m_indices mismatch (first diff at {(got_m.cpu() != ref_m).nonzero(as_tuple=True)[0][0].item() if not torch.equal(got_m.cpu(), ref_m) else 'N/A'})",
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
        counts_gpu = torch.tensor(
            [128] * 256, dtype=torch.int32, device=self.device
        )
        ref_start, ref_m = reference_scatter_1(counts_gpu)

        for iteration in range(200):
            got_start, got_m = self._run_kernel(counts_gpu)
            if not torch.equal(got_start.cpu(), ref_start):
                self.fail(
                    f"expert_start_loc mismatch at iteration {iteration}"
                )
            if not torch.equal(got_m.cpu(), ref_m):
                self.fail(f"m_indices mismatch at iteration {iteration}")


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
                counts_gpu, expert_start_loc, result_buf,
                num_experts=num_experts, num_warps=8,
                BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
            )
            torch.cuda.synchronize()
            got = result_buf.cpu()
            total_mismatches += int((got != ref_start).sum().item())

        print(f"Old kernel poison diagnostic: {total_mismatches} mismatches in 100 rounds")

    def test_new_kernel_immune_to_poison(self) -> None:
        """Production kernel (fixed) + poison fill: expert_start_loc and
        m_indices must always match the golden reference."""
        num_experts = 256
        counts = [128] * num_experts
        counts_gpu = torch.tensor(counts, dtype=torch.int32, device=self.device)
        ref_start, ref_m = reference_scatter_1(counts_gpu)
        all_tokens = sum(counts)

        for round_i in range(100):
            expert_start_loc = torch.full(
                (num_experts,), 0x7FFFFFFF, dtype=torch.int32, device=self.device
            )
            m_indices = torch.full(
                (all_tokens,), -1, dtype=torch.int32, device=self.device
            )
            _fwd_kernel_ep_scatter_1[(num_experts,)](
                counts_gpu, expert_start_loc, m_indices,
                num_experts=num_experts, num_warps=8,
                BLOCK_E=128,
                BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
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
