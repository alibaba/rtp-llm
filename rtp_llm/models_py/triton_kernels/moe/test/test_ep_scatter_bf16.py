"""Unit tests for bf16 EP scatter kernels.

ep_scatter_v2_bf16: flat [M, K] → 3D [E, alignment, K], for masked deepgemm (decode).
ep_scatter_bf16:    flat [M, K] → flat expert-sorted [all_tokens, K], for contiguous deepgemm (prefill).

Run with:
    python -m pytest github-opensource/rtp_llm/models_py/triton_kernels/moe/test/test_ep_scatter_bf16.py -v
"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_scatter_v2_bf16,
    ep_scatter_bf16,
    ep_gather,
)
from rtp_llm.models_py.utils.math import align


class TestEpScatterV2Bf16(unittest.TestCase):
    """Tests for ep_scatter_v2_bf16 kernel."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for triton kernel test")
        self.device = torch.device("cuda")

    def _reference_scatter(
        self,
        recv_x: torch.Tensor,
        recv_topk: torch.Tensor,
        num_experts: int,
        alignment: int,
    ):
        """CPU reference: scatter tokens to [E, alignment, K] based on topk assignments."""
        token_num, hidden_size = recv_x.shape
        topk_num = recv_topk.shape[1]
        output = torch.zeros(
            (num_experts, alignment, hidden_size),
            dtype=recv_x.dtype,
            device=recv_x.device,
        )
        output_index = torch.full_like(recv_topk, -1)
        expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=recv_x.device)

        for token_idx in range(token_num):
            for topk_idx in range(topk_num):
                expert_id = recv_topk[token_idx, topk_idx].item()
                if 0 <= expert_id < num_experts:
                    slot = expert_counts[expert_id].item()
                    output[expert_id, slot, :] = recv_x[token_idx, :]
                    output_index[token_idx, topk_idx] = expert_id * alignment + slot
                    expert_counts[expert_id] += 1

        return output, output_index

    def test_basic_scatter(self) -> None:
        """Basic scatter: each token assigned to one expert."""
        num_experts = 4
        hidden_size = 256
        token_num = 8
        topk = 1
        alignment = align(token_num, 128)

        recv_x = torch.randn(token_num, hidden_size, dtype=torch.bfloat16, device=self.device)
        recv_topk = torch.tensor(
            [[0], [1], [2], [3], [0], [1], [2], [3]],
            dtype=torch.int32,
            device=self.device,
        )

        output_tensor = torch.zeros(
            num_experts * alignment, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        output_index = torch.empty_like(recv_topk)
        expert_start_loc = torch.empty(num_experts, dtype=torch.int32, device=self.device)

        ep_scatter_v2_bf16(recv_x, recv_topk, alignment, expert_start_loc, output_tensor, output_index)

        ref_output, ref_index = self._reference_scatter(recv_x, recv_topk, num_experts, alignment)
        ref_output_flat = ref_output.view(num_experts * alignment, hidden_size)

        self.assertTrue(
            torch.equal(output_tensor, ref_output_flat),
            f"scatter output mismatch",
        )
        self.assertTrue(
            torch.equal(output_index, ref_index),
            f"index mismatch:\n  got={output_index}\n  want={ref_index}",
        )

    def test_topk_scatter(self) -> None:
        """Scatter with topk=2: each token placed in two expert slots."""
        num_experts = 4
        hidden_size = 128
        token_num = 4
        topk = 2
        alignment = align(token_num, 128)

        recv_x = torch.randn(token_num, hidden_size, dtype=torch.bfloat16, device=self.device)
        recv_topk = torch.tensor(
            [[0, 1], [2, 3], [0, 2], [1, 3]],
            dtype=torch.int32,
            device=self.device,
        )

        output_tensor = torch.zeros(
            num_experts * alignment, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        output_index = torch.empty_like(recv_topk)
        expert_start_loc = torch.empty(num_experts, dtype=torch.int32, device=self.device)

        ep_scatter_v2_bf16(recv_x, recv_topk, alignment, expert_start_loc, output_tensor, output_index)

        ref_output, ref_index = self._reference_scatter(recv_x, recv_topk, num_experts, alignment)
        ref_output_flat = ref_output.view(num_experts * alignment, hidden_size)

        self.assertTrue(
            torch.equal(output_tensor, ref_output_flat),
            "scatter output mismatch for topk=2",
        )
        self.assertTrue(
            torch.equal(output_index, ref_index),
            f"index mismatch for topk=2",
        )

    def test_negative_expert_ids_skipped(self) -> None:
        """Expert id = -1 should be skipped (not scattered)."""
        num_experts = 4
        hidden_size = 128
        token_num = 4
        topk = 2
        alignment = align(token_num, 128)

        recv_x = torch.randn(token_num, hidden_size, dtype=torch.bfloat16, device=self.device)
        recv_topk = torch.tensor(
            [[0, -1], [-1, 2], [1, 3], [-1, -1]],
            dtype=torch.int32,
            device=self.device,
        )

        output_tensor = torch.zeros(
            num_experts * alignment, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        output_index = torch.empty_like(recv_topk)
        expert_start_loc = torch.empty(num_experts, dtype=torch.int32, device=self.device)

        ep_scatter_v2_bf16(recv_x, recv_topk, alignment, expert_start_loc, output_tensor, output_index)

        # Expert 0 should have token 0, expert 1 should have token 2,
        # expert 2 should have token 1, expert 3 should have token 2
        output_3d = output_tensor.view(num_experts, alignment, hidden_size)
        self.assertTrue(torch.equal(output_3d[0, 0], recv_x[0]))
        self.assertTrue(torch.equal(output_3d[2, 0], recv_x[1]))
        self.assertTrue(torch.equal(output_3d[1, 0], recv_x[2]))
        self.assertTrue(torch.equal(output_3d[3, 0], recv_x[2]))

    def test_scatter_gather_roundtrip(self) -> None:
        """Scatter then gather should recover original (weighted) values."""
        num_experts = 4
        hidden_size = 256
        token_num = 16
        topk = 2
        alignment = align(token_num, 128)

        torch.manual_seed(42)
        recv_x = torch.randn(token_num, hidden_size, dtype=torch.bfloat16, device=self.device)

        # Random expert assignments
        recv_topk = torch.randint(0, num_experts, (token_num, topk), dtype=torch.int32, device=self.device)
        topk_weights = torch.ones(token_num, topk, dtype=torch.float32, device=self.device) / topk

        output_tensor = torch.zeros(
            num_experts * alignment, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        output_index = torch.empty_like(recv_topk)
        expert_start_loc = torch.empty(num_experts, dtype=torch.int32, device=self.device)

        ep_scatter_v2_bf16(recv_x, recv_topk, alignment, expert_start_loc, output_tensor, output_index)

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
        alignment = align(token_num, 128)

        torch.manual_seed(123)
        recv_x = torch.randn(token_num, hidden_size, dtype=torch.bfloat16, device=self.device)
        recv_topk = torch.randint(0, num_experts, (token_num, topk), dtype=torch.int32, device=self.device)

        output_tensor = torch.zeros(
            num_experts * alignment, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        output_index = torch.empty_like(recv_topk)
        expert_start_loc = torch.empty(num_experts, dtype=torch.int32, device=self.device)

        ep_scatter_v2_bf16(recv_x, recv_topk, alignment, expert_start_loc, output_tensor, output_index)

        ref_output, ref_index = self._reference_scatter(recv_x, recv_topk, num_experts, alignment)
        ref_output_flat = ref_output.view(num_experts * alignment, hidden_size)

        self.assertTrue(
            torch.equal(output_tensor, ref_output_flat),
            f"large random scatter mismatch",
        )


class TestEpScatterBf16Contiguous(unittest.TestCase):
    """Tests for ep_scatter_bf16 — flat [M, K] → expert-sorted flat [all_tokens, K]."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")

    def _run_scatter(self, recv_x, recv_topk, num_experts, aligned_tokens_per_expert):
        """Run ep_scatter_bf16 and return (output_tensor, m_indices, output_index)."""
        all_tokens = sum(aligned_tokens_per_expert)
        hidden_size = recv_x.shape[1]
        num_recv_tokens_per_expert_gpu = torch.tensor(
            aligned_tokens_per_expert, dtype=torch.int32, device=self.device
        )
        expert_start_loc = torch.empty(num_experts, dtype=torch.int32, device=self.device)
        output_tensor = torch.zeros(all_tokens, hidden_size, dtype=torch.bfloat16, device=self.device)
        m_indices = torch.zeros(all_tokens, dtype=torch.int32, device=self.device)
        output_index = torch.empty_like(recv_topk)
        ep_scatter_bf16(
            recv_x, recv_topk,
            num_recv_tokens_per_expert_gpu, expert_start_loc,
            output_tensor, m_indices, output_index,
        )
        return output_tensor, m_indices, output_index

    def test_basic_contiguous_scatter(self) -> None:
        """Each expert gets exactly one token; verify output ordering."""
        num_experts = 4
        hidden_size = 128
        alignment = 128
        recv_x = torch.randn(4, hidden_size, dtype=torch.bfloat16, device=self.device)
        recv_topk = torch.tensor([[0], [1], [2], [3]], dtype=torch.int32, device=self.device)
        aligned_counts = [alignment] * num_experts  # 1 token padded to 128

        output, m_indices, output_index = self._run_scatter(
            recv_x, recv_topk, num_experts, aligned_counts
        )
        # Expert 0 token should appear in slot 0..127, expert 1 in 128..255, etc.
        self.assertEqual(output.shape[0], alignment * num_experts)
        # m_indices for slot 0 should be expert 0
        self.assertEqual(m_indices[0].item(), 0)
        self.assertEqual(m_indices[alignment].item(), 1)

    def test_negative_expert_id_skipped(self) -> None:
        """expert_id = -1 slots should not be scattered."""
        num_experts = 2
        hidden_size = 64
        alignment = 128
        recv_x = torch.randn(4, hidden_size, dtype=torch.bfloat16, device=self.device)
        recv_topk = torch.tensor([[0], [-1], [1], [-1]], dtype=torch.int32, device=self.device)
        aligned_counts = [alignment, alignment]

        output, m_indices, output_index = self._run_scatter(
            recv_x, recv_topk, num_experts, aligned_counts
        )
        # Expert 0 gets token 0, expert 1 gets token 2
        self.assertTrue(torch.equal(output[0], recv_x[0]))
        self.assertTrue(torch.equal(output[alignment], recv_x[2]))

    def test_scatter_gather_roundtrip(self) -> None:
        """Scatter then gather with uniform weights recovers original."""
        num_experts = 4
        hidden_size = 256
        token_num = 16
        topk = 2
        alignment = 128
        torch.manual_seed(7)
        recv_x = torch.randn(token_num, hidden_size, dtype=torch.bfloat16, device=self.device)
        recv_topk = torch.randint(0, num_experts, (token_num, topk), dtype=torch.int32, device=self.device)
        topk_weights = torch.ones(token_num, topk, dtype=torch.float32, device=self.device) / topk
        aligned_counts = [alignment] * num_experts

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
        recv_x = torch.randn(token_num, hidden_size, dtype=torch.bfloat16, device=self.device)
        recv_topk = torch.randint(0, num_experts, (token_num, topk), dtype=torch.int32, device=self.device)
        aligned_counts = [alignment] * num_experts

        output, m_indices, _ = self._run_scatter(
            recv_x, recv_topk, num_experts, aligned_counts
        )
        self.assertEqual(output.shape, (alignment * num_experts, hidden_size))
        self.assertTrue(m_indices.min().item() >= 0)
        self.assertTrue(m_indices.max().item() < num_experts)


if __name__ == "__main__":
    unittest.main()
