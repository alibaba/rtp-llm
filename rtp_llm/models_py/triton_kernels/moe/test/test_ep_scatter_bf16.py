"""Unit tests for bf16 EP scatter kernels.

ep_scatter_v2_bf16: flat [M, K] → 3D [E, alignment, K], for masked deepgemm (decode).
ep_scatter_bf16:    flat [M, K] → flat expert-sorted [all_tokens, K], for contiguous deepgemm (prefill).

Both scatter kernels assign output slots with a non-deterministic tl.atomic_add, so the
row ORDER within an expert is not fixed across runs. All correctness checks here are
therefore order-independent: they follow output_index (the authoritative token->slot map
the gather stage uses) rather than assuming a token-sequential layout.

ep_gather's BLOCK_D is 512, so any test that gathers must use hidden_size % 512 == 0.

Run with:
    python -m pytest github-opensource/rtp_llm/models_py/triton_kernels/moe/test/test_ep_scatter_bf16.py -v
"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter_bf16,
    ep_scatter_v2_bf16,
)
from rtp_llm.models_py.utils.math import align


class TestEpScatterV2Bf16(unittest.TestCase):
    """Tests for ep_scatter_v2_bf16 — flat [M, K] → 3D [E, alignment, K] (masked)."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for triton kernel test")
        self.device = torch.device("cuda")

    @staticmethod
    def _distinct_topk(token_num, topk, num_experts, device):
        """One row per token with topk *distinct* experts (keeps per-expert count <= token_num)."""
        ids = torch.empty((token_num, topk), dtype=torch.int32, device=device)
        for t in range(token_num):
            ids[t] = torch.randperm(num_experts, device=device)[:topk].to(torch.int32)
        return ids

    def _verify_v2(
        self, recv_x, recv_topk, num_experts, alignment, output_flat, output_index
    ):
        """Order-independent check for the 3D masked layout via output_index.

        Each valid (token, topk) must map to a slot inside its expert's
        [e * alignment, (e + 1) * alignment) block, carry the right row content, and
        no slot may be reused.
        """
        seen: set[int] = set()
        topk = recv_topk.shape[1]
        for t in range(recv_topk.shape[0]):
            for k in range(topk):
                expert = int(recv_topk[t, k].item())
                if expert < 0 or expert >= num_experts:
                    continue
                slot = int(output_index[t, k].item())
                lo, hi = expert * alignment, (expert + 1) * alignment
                self.assertTrue(
                    lo <= slot < hi,
                    f"token {t} topk {k} -> expert {expert}: slot {slot} outside [{lo}, {hi})",
                )
                self.assertNotIn(
                    slot, seen, f"slot {slot} assigned to more than one token"
                )
                seen.add(slot)
                self.assertTrue(
                    torch.equal(output_flat[slot], recv_x[t]),
                    f"row content mismatch at slot {slot} (token {t})",
                )

    def _run_scatter_v2(self, recv_x, recv_topk, num_experts, alignment):
        hidden_size = recv_x.shape[1]
        output_tensor = torch.zeros(
            num_experts * alignment,
            hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        output_index = torch.empty_like(recv_topk)
        expert_start_loc = torch.empty(
            num_experts, dtype=torch.int32, device=self.device
        )
        ep_scatter_v2_bf16(
            recv_x, recv_topk, alignment, expert_start_loc, output_tensor, output_index
        )
        return output_tensor, output_index

    def test_basic_scatter(self) -> None:
        """Basic scatter: each token assigned to one expert."""
        num_experts = 4
        hidden_size = 256
        token_num = 8
        alignment = align(token_num, 128)

        recv_x = torch.randn(
            token_num, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        recv_topk = torch.tensor(
            [[0], [1], [2], [3], [0], [1], [2], [3]],
            dtype=torch.int32,
            device=self.device,
        )

        output_tensor, output_index = self._run_scatter_v2(
            recv_x, recv_topk, num_experts, alignment
        )
        self._verify_v2(
            recv_x, recv_topk, num_experts, alignment, output_tensor, output_index
        )

    def test_topk_scatter(self) -> None:
        """Scatter with topk=2: each token placed in two expert slots."""
        num_experts = 4
        hidden_size = 128
        token_num = 4
        alignment = align(token_num, 128)

        recv_x = torch.randn(
            token_num, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        recv_topk = torch.tensor(
            [[0, 1], [2, 3], [0, 2], [1, 3]],
            dtype=torch.int32,
            device=self.device,
        )

        output_tensor, output_index = self._run_scatter_v2(
            recv_x, recv_topk, num_experts, alignment
        )
        self._verify_v2(
            recv_x, recv_topk, num_experts, alignment, output_tensor, output_index
        )

    def test_negative_expert_ids_skipped(self) -> None:
        """Expert id = -1 should be skipped (not scattered)."""
        num_experts = 4
        hidden_size = 128
        token_num = 4
        alignment = align(token_num, 128)

        recv_x = torch.randn(
            token_num, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        recv_topk = torch.tensor(
            [[0, -1], [-1, 2], [1, 3], [-1, -1]],
            dtype=torch.int32,
            device=self.device,
        )

        output_tensor, output_index = self._run_scatter_v2(
            recv_x, recv_topk, num_experts, alignment
        )
        # Each expert here receives exactly one valid token, so slot 0 of each block is
        # deterministic regardless of atomic ordering.
        output_3d = output_tensor.view(num_experts, alignment, hidden_size)
        self.assertTrue(torch.equal(output_3d[0, 0], recv_x[0]))
        self.assertTrue(torch.equal(output_3d[2, 0], recv_x[1]))
        self.assertTrue(torch.equal(output_3d[1, 0], recv_x[2]))
        self.assertTrue(torch.equal(output_3d[3, 0], recv_x[2]))
        self._verify_v2(
            recv_x, recv_topk, num_experts, alignment, output_tensor, output_index
        )

    def test_scatter_gather_roundtrip(self) -> None:
        """Scatter then gather should recover original (weighted) values."""
        num_experts = 4
        hidden_size = 512  # ep_gather BLOCK_D = 512
        token_num = 16
        topk = 2
        alignment = align(token_num, 128)

        torch.manual_seed(42)
        recv_x = torch.randn(
            token_num, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        recv_topk = self._distinct_topk(token_num, topk, num_experts, self.device)
        topk_weights = (
            torch.ones(token_num, topk, dtype=torch.float32, device=self.device) / topk
        )

        output_tensor, output_index = self._run_scatter_v2(
            recv_x, recv_topk, num_experts, alignment
        )

        # Gather back (identity transform: no GEMM, just scatter → gather)
        gather_out = torch.zeros_like(recv_x)
        ep_gather(output_tensor, recv_topk, topk_weights, output_index, gather_out)

        # Each token was scattered topk times with weight 1/topk, so gather sum = original
        self.assertTrue(
            torch.allclose(gather_out.float(), recv_x.float(), atol=1e-2),
            f"roundtrip mismatch: max diff = {(gather_out.float() - recv_x.float()).abs().max().item()}",
        )

    def test_large_random(self) -> None:
        """Stress test with larger dimensions."""
        num_experts = 8
        hidden_size = 2048
        token_num = 256
        topk = 4
        alignment = align(token_num, 128)  # 256 >= max per-expert count

        torch.manual_seed(123)
        recv_x = torch.randn(
            token_num, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        # distinct experts per token => per-expert count <= token_num <= alignment
        recv_topk = self._distinct_topk(token_num, topk, num_experts, self.device)

        output_tensor, output_index = self._run_scatter_v2(
            recv_x, recv_topk, num_experts, alignment
        )
        self._verify_v2(
            recv_x, recv_topk, num_experts, alignment, output_tensor, output_index
        )


class TestEpScatterBf16Contiguous(unittest.TestCase):
    """Tests for ep_scatter_bf16 — flat [M, K] → expert-sorted flat [all_tokens, K]."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")

    @staticmethod
    def _aligned_counts_from_topk(recv_topk, num_experts, alignment):
        """Per-expert capacity from the actual routing histogram, aligned up.

        Mirrors the production allocation in DeepGemmBf16HybridExecutor.execute_contiguous:
        the scatter kernel assigns slots with an unchecked atomic_add, so the caller MUST
        size output_tensor to cover every expert's real (aligned) token count. Hardcoding a
        fixed per-expert capacity smaller than the real load causes out-of-bounds writes.
        """
        valid = recv_topk[recv_topk >= 0].to(torch.int64).flatten()
        counts = torch.bincount(valid, minlength=num_experts)
        return [align(int(c), alignment) for c in counts.tolist()]

    def _verify_contiguous(
        self,
        recv_x,
        recv_topk,
        num_experts,
        aligned_counts,
        output,
        m_indices,
        output_index,
    ):
        """Order-independent correctness check via output_index.

        The atomic_add slot assignment order is non-deterministic, but every valid
        (token, topk) assignment must land in its expert's [start, start + aligned_count)
        region, with the right row content and m_indices label, and no slot may be reused.
        m_indices must be correct on OCCUPIED rows because the downstream grouped GEMM uses
        it to pick each row's expert.
        """
        start_loc = []
        acc = 0
        for count in aligned_counts:
            start_loc.append(acc)
            acc += count

        seen: set[int] = set()
        topk = recv_topk.shape[1]
        for t in range(recv_topk.shape[0]):
            for k in range(topk):
                expert = int(recv_topk[t, k].item())
                if expert < 0:
                    continue
                slot = int(output_index[t, k].item())
                lo, hi = start_loc[expert], start_loc[expert] + aligned_counts[expert]
                self.assertTrue(
                    lo <= slot < hi,
                    f"token {t} topk {k} -> expert {expert}: slot {slot} outside [{lo}, {hi})",
                )
                self.assertNotIn(
                    slot, seen, f"slot {slot} assigned to more than one token"
                )
                seen.add(slot)
                self.assertTrue(
                    torch.equal(output[slot], recv_x[t]),
                    f"row content mismatch at slot {slot} (token {t})",
                )
                self.assertEqual(
                    int(m_indices[slot].item()),
                    expert,
                    f"m_indices[{slot}] != expert {expert}",
                )

    def _run_scatter(self, recv_x, recv_topk, num_experts, aligned_tokens_per_expert):
        """Run ep_scatter_bf16 and return (output_tensor, m_indices, output_index)."""
        all_tokens = sum(aligned_tokens_per_expert)
        hidden_size = recv_x.shape[1]
        num_recv_tokens_per_expert_gpu = torch.tensor(
            aligned_tokens_per_expert, dtype=torch.int32, device=self.device
        )
        expert_start_loc = torch.empty(
            num_experts, dtype=torch.int32, device=self.device
        )
        output_tensor = torch.zeros(
            all_tokens, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        m_indices = torch.zeros(all_tokens, dtype=torch.int32, device=self.device)
        output_index = torch.empty_like(recv_topk)
        ep_scatter_bf16(
            recv_x,
            recv_topk,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            output_tensor,
            m_indices,
            output_index,
        )
        return output_tensor, m_indices, output_index

    def test_basic_contiguous_scatter(self) -> None:
        """Each expert gets exactly one token; verify output ordering."""
        num_experts = 4
        hidden_size = 128
        alignment = 128
        recv_x = torch.randn(4, hidden_size, dtype=torch.bfloat16, device=self.device)
        recv_topk = torch.tensor(
            [[0], [1], [2], [3]], dtype=torch.int32, device=self.device
        )
        aligned_counts = [alignment] * num_experts  # 1 token padded to 128

        output, m_indices, output_index = self._run_scatter(
            recv_x, recv_topk, num_experts, aligned_counts
        )
        # Expert 0 token should appear in slot 0..127, expert 1 in 128..255, etc.
        self.assertEqual(output.shape[0], alignment * num_experts)
        # m_indices for slot 0 should be expert 0
        self.assertEqual(m_indices[0].item(), 0)
        self.assertEqual(m_indices[alignment].item(), 1)
        self._verify_contiguous(
            recv_x,
            recv_topk,
            num_experts,
            aligned_counts,
            output,
            m_indices,
            output_index,
        )

    def test_negative_expert_id_skipped(self) -> None:
        """expert_id = -1 slots should not be scattered."""
        num_experts = 2
        hidden_size = 64
        alignment = 128
        recv_x = torch.randn(4, hidden_size, dtype=torch.bfloat16, device=self.device)
        recv_topk = torch.tensor(
            [[0], [-1], [1], [-1]], dtype=torch.int32, device=self.device
        )
        aligned_counts = [alignment, alignment]

        output, m_indices, output_index = self._run_scatter(
            recv_x, recv_topk, num_experts, aligned_counts
        )
        # Expert 0 gets token 0, expert 1 gets token 2
        self.assertTrue(torch.equal(output[0], recv_x[0]))
        self.assertTrue(torch.equal(output[alignment], recv_x[2]))
        self._verify_contiguous(
            recv_x,
            recv_topk,
            num_experts,
            aligned_counts,
            output,
            m_indices,
            output_index,
        )

    def test_scatter_gather_roundtrip(self) -> None:
        """Scatter then gather with uniform weights recovers original."""
        num_experts = 4
        hidden_size = 512  # ep_gather BLOCK_D = 512
        token_num = 16
        topk = 2
        alignment = 128
        torch.manual_seed(7)
        recv_x = torch.randn(
            token_num, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        recv_topk = torch.randint(
            0, num_experts, (token_num, topk), dtype=torch.int32, device=self.device
        )
        topk_weights = (
            torch.ones(token_num, topk, dtype=torch.float32, device=self.device) / topk
        )
        aligned_counts = self._aligned_counts_from_topk(
            recv_topk, num_experts, alignment
        )

        output, _, output_index = self._run_scatter(
            recv_x, recv_topk, num_experts, aligned_counts
        )
        gather_out = torch.zeros_like(recv_x)
        ep_gather(output, recv_topk, topk_weights, output_index, gather_out)
        self.assertTrue(
            torch.allclose(gather_out.float(), recv_x.float(), atol=1e-2),
            f"roundtrip max diff={( gather_out.float() - recv_x.float()).abs().max():.4f}",
        )

    def test_large_random(self) -> None:
        """Stress: large token count, multiple experts."""
        num_experts = 8
        hidden_size = 1024
        token_num = 512
        topk = 4
        alignment = 128
        torch.manual_seed(99)
        recv_x = torch.randn(
            token_num, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        recv_topk = torch.randint(
            0, num_experts, (token_num, topk), dtype=torch.int32, device=self.device
        )
        # token_num * topk = 2048 assignments over 8 experts (~256 each); a fixed
        # [alignment] * num_experts = 1024-slot buffer would be ~2x too small and the
        # unchecked atomic_add in the kernel would write out of bounds. Size capacity
        # from the real per-expert histogram, exactly as the executor does.
        aligned_counts = self._aligned_counts_from_topk(
            recv_topk, num_experts, alignment
        )

        output, m_indices, output_index = self._run_scatter(
            recv_x, recv_topk, num_experts, aligned_counts
        )
        self.assertEqual(output.shape, (sum(aligned_counts), hidden_size))
        self.assertTrue(m_indices.min().item() >= 0)
        self.assertTrue(m_indices.max().item() < num_experts)
        self._verify_contiguous(
            recv_x,
            recv_topk,
            num_experts,
            aligned_counts,
            output,
            m_indices,
            output_index,
        )


if __name__ == "__main__":
    unittest.main()
