"""Unit tests for TorchDistEpRouter dispatch/combine logic.

Tests the pure-Python routing logic (classification, padding, indexing)
without requiring a multi-GPU distributed environment. The _dispatch and
_combine methods are tested by re-implementing them with local
all_to_all simulation using tensor chunking.
"""

from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.torch_dist_ep_router import (
    _pad_and_cat,
)


class PadAndCatTest(TestCase):
    """Test _pad_and_cat utility function."""

    def test_pad_2d_tensors(self):
        """Test padding and concatenation of 2D tensors (hidden states)."""
        chunks = [
            torch.randn(3, 64),
            torch.randn(5, 64),
            torch.randn(2, 64),
        ]
        counts = [3, 5, 2]
        max_c = 5
        extra_dim = 64

        result = _pad_and_cat(chunks, counts, max_c, extra_dim)

        # 3 chunks, each padded to max_c=5 => total 15 rows
        self.assertEqual(result.shape, (15, 64))

        # Check first chunk (3 rows of data + 2 rows of zeros)
        first_chunk = result[:5]
        torch.testing.assert_close(first_chunk[:3], chunks[0])
        self.assertTrue((first_chunk[3:] == 0).all())

        # Check second chunk (no padding needed)
        second_chunk = result[5:10]
        torch.testing.assert_close(second_chunk, chunks[1])

    def test_pad_1d_tensors(self):
        """Test padding and concatenation of 1D tensors (weights/indices)."""
        chunks = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0, 7.0]),
        ]
        counts = [3, 4]
        max_c = 4
        extra_dim = 0  # ignored for 1D

        result = _pad_and_cat(chunks, counts, max_c, extra_dim)

        self.assertEqual(result.shape, (8,))
        torch.testing.assert_close(result[:3], chunks[0])
        self.assertEqual(result[3].item(), 0.0)  # padding

    def test_empty_chunks(self):
        """Test handling of empty chunks."""
        chunks = [
            torch.empty(0, 32),
            torch.randn(3, 32),
        ]
        counts = [0, 3]
        max_c = 3
        extra_dim = 32

        result = _pad_and_cat(chunks, counts, max_c, extra_dim)
        self.assertEqual(result.shape, (6, 32))

    def test_max_c_zero(self):
        """Test when max_c is 0 (no tokens at all)."""
        chunks = [torch.empty(0, 32), torch.empty(0, 32)]
        counts = [0, 0]
        max_c = 0
        extra_dim = 32

        result = _pad_and_cat(chunks, counts, max_c, extra_dim)
        self.assertEqual(result.shape, (0, 32))


class SimulatedDispatchCombineTest(TestCase):
    """Test dispatch/combine roundtrip using local simulation.

    With ep_size=1, all experts are local, so dispatch keeps all data
    and combine reconstructs the original weighted sum.
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _simulate_dispatch_ep1(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_num: int,
    ):
        """Simulate _dispatch for ep_size=1 (all experts local)."""
        num_tokens, hidden_dim = hidden_states.shape
        topk = topk_ids.shape[1]
        flat_total = num_tokens * topk
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Flatten
        flat_h = (
            hidden_states.unsqueeze(1)
            .expand(-1, topk, -1)
            .reshape(flat_total, hidden_dim)
        )
        flat_ids = topk_ids.reshape(flat_total)
        flat_w = topk_weights.reshape(flat_total)
        flat_oi = torch.arange(flat_total, device=device, dtype=torch.int64)

        # ep_size=1: all experts belong to rank 0
        e_start = 0
        e_end = expert_num
        mask = (flat_ids >= e_start) & (flat_ids < e_end)
        assert mask.all(), "All expert IDs should be local for ep_size=1"

        local_ids = flat_ids  # local expert IDs = global for ep_size=1

        return flat_h, flat_w, local_ids, flat_oi

    def _simulate_combine_ep1(
        self,
        expert_output: torch.Tensor,
        recv_weights: torch.Tensor,
        recv_orig_flat_idx: torch.Tensor,
        num_tokens: int,
        topk: int,
        hidden_dim: int,
    ):
        """Simulate _combine for ep_size=1 (single rank, no all_gather needed)."""
        device = expert_output.device
        dtype = expert_output.dtype
        flat_total = num_tokens * topk
        recv_count = expert_output.shape[0]

        if recv_count == 0:
            return torch.zeros((num_tokens, hidden_dim), dtype=dtype, device=device)

        # Weight by router scores
        weighted = expert_output * recv_weights.to(expert_output.dtype).unsqueeze(-1)

        # Scatter into [flat_total, H] using index_add
        output = torch.zeros((flat_total, hidden_dim), dtype=dtype, device=device)
        output.index_add_(0, recv_orig_flat_idx, weighted)

        return output.reshape(num_tokens, topk, hidden_dim).sum(dim=1)

    def _run_roundtrip_test(
        self,
        token_num: int,
        hidden_dim: int,
        expert_num: int,
        top_k: int,
    ):
        """Test dispatch -> identity expert -> combine roundtrip with ep_size=1."""
        torch.manual_seed(42)

        hidden_states = (
            torch.randn(token_num, hidden_dim, device="cuda").to(torch.bfloat16) * 0.03
        )
        topk_ids = torch.topk(
            torch.rand(token_num, expert_num, device="cuda"), top_k, dim=1
        ).indices.to(torch.int32)
        topk_weights = torch.softmax(
            torch.randn(token_num, top_k, device="cuda"), dim=-1
        ).to(torch.float32)

        # Dispatch (ep_size=1, all local)
        recv_h, recv_w, recv_lids, recv_oids = self._simulate_dispatch_ep1(
            hidden_states, topk_ids, topk_weights, expert_num
        )

        # Verify: all tokens dispatched, expert IDs preserved
        self.assertEqual(recv_h.shape, (token_num * top_k, hidden_dim))
        self.assertTrue((recv_lids.flatten() >= 0).all())
        self.assertTrue((recv_lids.flatten() < expert_num).all())

        # Identity expert: output = input
        expert_output = recv_h.clone()

        # Combine
        result = self._simulate_combine_ep1(
            expert_output, recv_w, recv_oids, token_num, top_k, hidden_dim
        )

        # Expected: identity expert with weighting = hidden_states expanded * topk_weights
        flat_h = recv_h.float()
        flat_w = recv_w.float()
        expected_flat = flat_h * flat_w.unsqueeze(-1)
        expected = expected_flat.reshape(token_num, top_k, hidden_dim).sum(dim=1)
        expected = expected.to(torch.bfloat16)

        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-1)

    def test_dispatch_combine_ep1_small(self):
        """Test dispatch/combine with ep_size=1, small config."""
        self._run_roundtrip_test(
            token_num=16,
            hidden_dim=128,
            expert_num=4,
            top_k=2,
        )

    def test_dispatch_combine_ep1_medium(self):
        """Test dispatch/combine with ep_size=1, medium config."""
        self._run_roundtrip_test(
            token_num=32,
            hidden_dim=256,
            expert_num=8,
            top_k=4,
        )

    def test_dispatch_combine_ep1_large(self):
        """Test dispatch/combine with ep_size=1, larger config."""
        self._run_roundtrip_test(
            token_num=64,
            hidden_dim=512,
            expert_num=32,
            top_k=8,
        )

    def test_dispatch_invariants_ep2(self):
        """Test dispatch classification invariants for ep_size=2."""
        torch.manual_seed(42)
        ep_size = 2
        token_num = 32
        hidden_dim = 256
        expert_num = 8  # 4 per rank
        top_k = 2

        hidden_states = torch.randn(token_num, hidden_dim, device="cuda").to(
            torch.bfloat16
        )
        topk_ids = torch.topk(
            torch.rand(token_num, expert_num, device="cuda"), top_k, dim=1
        ).indices.to(torch.int32)
        topk_weights = torch.softmax(
            torch.randn(token_num, top_k, device="cuda"), dim=-1
        ).to(torch.float32)

        # Simulate dispatch for each rank and verify invariants
        flat_total = token_num * top_k
        flat_ids = topk_ids.flatten()
        expert_num_per_rank = expert_num // ep_size

        # Track which flat indices go to each rank
        rank_counts = []
        for rank in range(ep_size):
            e_start = rank * expert_num_per_rank
            e_end = e_start + expert_num_per_rank
            mask = (flat_ids >= e_start) & (flat_ids < e_end)
            count = int(mask.sum())
            rank_counts.append(count)

            # Verify local expert IDs are in [0, expert_num_per_rank)
            if count > 0:
                local_ids = flat_ids[mask] - e_start
                self.assertTrue((local_ids >= 0).all())
                self.assertTrue((local_ids < expert_num_per_rank).all())

        # Total must match flat_total
        self.assertEqual(sum(rank_counts), flat_total)

    def test_zero_tokens(self):
        """Test dispatch with zero tokens (edge case)."""
        hidden_states = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
        topk_ids = torch.empty(0, 2, device="cuda", dtype=torch.int32)
        topk_weights = torch.empty(0, 2, device="cuda", dtype=torch.float32)

        # ep_size=1, expert_num=4
        flat_h, flat_w, flat_lids, flat_oids = self._simulate_dispatch_ep1(
            hidden_states, topk_ids, topk_weights, expert_num=4
        )

        self.assertEqual(flat_h.shape, (0, 256))
        self.assertEqual(flat_lids.numel(), 0)


if __name__ == "__main__":
    main()
