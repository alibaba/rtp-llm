"""
Unit tests for PCPAll2AllAttnOp (all-to-all with point-to-point send/recv).

Unlike all-gather, all-to-all uses point-to-point send/recv. The mocks inject
microsecond-level sleeps to better simulate real communication latency and
exercise the stream / event synchronization logic.
"""

import contextlib
import math
import time
import unittest
from typing import List
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.alltoall_cp_impl import (
    PCPAll2AllAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    build_cp_attn_inputs,
    compute_rank_positions,
    extract_kv_from_paged_cache,
    fill_prefix_into_kv_cache,
    make_configs,
    make_kv_cache,
    reference_causal_attention,
    reference_prefill_with_prefix,
    zigzag_positions_for_rank,
)

_A2A_MODULE = (
    "rtp_llm.models_py.modules.factory.attention."
    "cuda_cp_impl.prefill_mha.alltoall_cp_impl"
)

_COMM_DELAY_US = 50  # microseconds injected per mock collective call


def _sleep_us(us: int = _COMM_DELAY_US) -> None:
    """Spin-sleep for *us* microseconds (more precise than time.sleep for <1ms)."""
    end = time.perf_counter() + us * 1e-6
    while time.perf_counter() < end:
        pass


class TestPCPAll2AllAttnOp(unittest.TestCase):
    """Tests for PCPAll2AllAttnOp with cp_size=4."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def _assert_close(self, actual, expected, rtol=1e-2, atol=1e-2):
        af, ef = actual.float(), expected.float()
        diff = (af - ef).abs()
        max_diff, mean_diff = diff.max().item(), diff.mean().item()
        self.assertTrue(
            torch.allclose(af, ef, rtol=rtol, atol=atol),
            f"Output mismatch: max_diff={max_diff}, mean_diff={mean_diff}",
        )

    @staticmethod
    def _build_shuffle_indices(
        new_lengths: List[int],
        cp_size: int,
        rank: int,
        prefix_lengths: List[int] | None = None,
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        if prefix_lengths is None:
            prefix_lengths = [0] * len(new_lengths)
        indices: List[int] = []
        for new_len, pl in zip(new_lengths, prefix_lengths):
            positions = zigzag_positions_for_rank(new_len, cp_size, rank)
            indices.extend(p + pl for p in positions)
        return torch.tensor(indices, dtype=torch.int32, device=device)

    # ---- mock builders ----

    @staticmethod
    def _make_mocks(all_shuffle, all_kv_buffers):
        """Return (mock_all_gather, mock_send, mock_recv) with μs delays."""

        def mock_all_gather(tensor, group=None):
            _sleep_us()
            return torch.cat(all_shuffle, dim=0)

        def mock_send(tensor, dst, group=None):
            _sleep_us()

        def mock_recv(tensor, src, group=None):
            _sleep_us()
            tensor.copy_(all_kv_buffers[src])

        return mock_all_gather, mock_send, mock_recv

    def _run_with_mocks(
        self, attn_cfg, par_cfg, attn_inputs, qkv, kv_cache, all_shuffle, all_kv_buffers
    ):
        mock_ag, mock_send, mock_recv = self._make_mocks(
            all_shuffle,
            all_kv_buffers,
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(f"{_A2A_MODULE}.all_gather", side_effect=mock_ag))
            stack.enter_context(patch(f"{_A2A_MODULE}.send", side_effect=mock_send))
            stack.enter_context(patch(f"{_A2A_MODULE}.recv", side_effect=mock_recv))
            stack.enter_context(
                patch(
                    f"{_A2A_MODULE}.get_user_buffers_communicator",
                    return_value=None,
                )
            )
            op = PCPAll2AllAttnOp(attn_cfg, attn_inputs, par_cfg)
            params = op.prepare(attn_inputs)
            return op.forward(qkv, kv_cache, params)

    # ---- no-prefix driver ----

    def run_no_prefix(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        cp_size: int = 4,
        cp_rank: int = 0,
        head_num: int = 8,
        kv_head_num: int = 2,
        head_dim: int = 64,
        tokens_per_block: int = 16,
    ):
        assert all(sl % cp_size == 0 for sl in sequence_lengths)
        cp_chunk_lengths = [sl // cp_size for sl in sequence_lengths]
        assert all(cl % 2 == 0 for cl in cp_chunk_lengths)

        attn_cfg, par_cfg = make_configs(
            head_num=head_num,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )

        total_tokens = sum(sequence_lengths)
        q_full = torch.randn(
            total_tokens,
            head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k_full = torch.randn(
            total_tokens,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        v_full = torch.randn(
            total_tokens,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        cu_full = [0]
        for sl in sequence_lengths:
            cu_full.append(cu_full[-1] + sl)
        ref_output = reference_causal_attention(q_full, k_full, v_full, cu_full)

        all_rank_pos = compute_rank_positions(sequence_lengths, cp_size)
        all_kv_buffers = {}
        for r in range(cp_size):
            r_idx = torch.tensor(all_rank_pos[r], device=self.device)
            k_r = k_full[r_idx].reshape(-1, kv_head_num * head_dim)
            v_r = v_full[r_idx].reshape(-1, kv_head_num * head_dim)
            all_kv_buffers[r] = torch.cat([k_r, v_r], dim=0)

        rank_idx = torch.tensor(all_rank_pos[cp_rank], device=self.device)
        qkv = torch.cat(
            [
                q_full[rank_idx].reshape(-1, head_num * head_dim),
                k_full[rank_idx].reshape(-1, kv_head_num * head_dim),
                v_full[rank_idx].reshape(-1, kv_head_num * head_dim),
            ],
            dim=-1,
        )

        all_shuffle = [
            self._build_shuffle_indices(
                sequence_lengths,
                cp_size,
                r,
                device=self.device,
            )
            for r in range(cp_size)
        ]

        attn_inputs = build_cp_attn_inputs(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            tokens_per_block,
            device=self.device,
        )
        attn_inputs.context_parallel_info.prefill_shuffle_indices = all_shuffle[
            cp_rank
        ].cpu()
        total_blocks = sum(math.ceil(s / tokens_per_block) for s in sequence_lengths)
        kv_cache = make_kv_cache(
            total_blocks,
            kv_head_num,
            tokens_per_block,
            head_dim,
            device=self.device,
        )

        output = self._run_with_mocks(
            attn_cfg,
            par_cfg,
            attn_inputs,
            qkv,
            kv_cache,
            all_shuffle,
            all_kv_buffers,
        )
        self._assert_close(output, ref_output[rank_idx])

        cache_k, cache_v = extract_kv_from_paged_cache(
            kv_cache, sequence_lengths, tokens_per_block
        )
        self.assertTrue(
            torch.equal(cache_k, k_full),
            f"KV cache K mismatch: max_diff="
            f"{(cache_k.float() - k_full.float()).abs().max().item()}",
        )
        self.assertTrue(
            torch.equal(cache_v, v_full),
            f"KV cache V mismatch: max_diff="
            f"{(cache_v.float() - v_full.float()).abs().max().item()}",
        )

    # ---- with-prefix driver ----

    def run_with_prefix(
        self,
        batch_size: int,
        new_lengths: List[int],
        prefix_lengths: List[int],
        cp_size: int = 4,
        cp_rank: int = 0,
        head_num: int = 8,
        kv_head_num: int = 2,
        head_dim: int = 64,
        tokens_per_block: int = 16,
    ):
        sequence_lengths = [p + n for p, n in zip(prefix_lengths, new_lengths)]
        cp_chunk_lengths = [n // cp_size for n in new_lengths]
        assert all(cl % 2 == 0 for cl in cp_chunk_lengths)
        assert all(pl % tokens_per_block == 0 for pl in prefix_lengths)

        attn_cfg, par_cfg = make_configs(
            head_num=head_num,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )

        total_prefix = sum(prefix_lengths)
        total_new = sum(new_lengths)
        prefix_k = torch.randn(
            total_prefix,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        prefix_v = torch.randn(
            total_prefix,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        new_q = torch.randn(
            total_new,
            head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        new_k = torch.randn(
            total_new,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        new_v = torch.randn(
            total_new,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        ref_output = reference_prefill_with_prefix(
            new_q,
            prefix_k,
            prefix_v,
            new_k,
            new_v,
            new_lengths,
            prefix_lengths,
        )

        all_rank_pos = compute_rank_positions(new_lengths, cp_size)
        all_kv_buffers = {}
        for r in range(cp_size):
            r_idx = torch.tensor(all_rank_pos[r], device=self.device)
            k_r = new_k[r_idx].reshape(-1, kv_head_num * head_dim)
            v_r = new_v[r_idx].reshape(-1, kv_head_num * head_dim)
            all_kv_buffers[r] = torch.cat([k_r, v_r], dim=0)

        rank_idx = torch.tensor(all_rank_pos[cp_rank], device=self.device)
        qkv = torch.cat(
            [
                new_q[rank_idx].reshape(-1, head_num * head_dim),
                new_k[rank_idx].reshape(-1, kv_head_num * head_dim),
                new_v[rank_idx].reshape(-1, kv_head_num * head_dim),
            ],
            dim=-1,
        )

        all_shuffle = [
            self._build_shuffle_indices(
                new_lengths,
                cp_size,
                r,
                prefix_lengths=prefix_lengths,
                device=self.device,
            )
            for r in range(cp_size)
        ]

        attn_inputs = build_cp_attn_inputs(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            tokens_per_block,
            prefix_lengths=prefix_lengths,
            device=self.device,
        )
        attn_inputs.context_parallel_info.prefill_shuffle_indices = all_shuffle[
            cp_rank
        ].cpu()
        total_blocks = sum(math.ceil(s / tokens_per_block) for s in sequence_lengths)
        kv_cache = make_kv_cache(
            total_blocks,
            kv_head_num,
            tokens_per_block,
            head_dim,
            device=self.device,
        )
        fill_prefix_into_kv_cache(
            kv_cache,
            prefix_k,
            prefix_v,
            prefix_lengths,
            sequence_lengths,
            tokens_per_block,
        )

        output = self._run_with_mocks(
            attn_cfg,
            par_cfg,
            attn_inputs,
            qkv,
            kv_cache,
            all_shuffle,
            all_kv_buffers,
        )
        self._assert_close(output, ref_output[rank_idx])

        cache_k, cache_v = extract_kv_from_paged_cache(
            kv_cache, sequence_lengths, tokens_per_block
        )
        expected_k_parts: List[torch.Tensor] = []
        expected_v_parts: List[torch.Tensor] = []
        pk_off, nk_off = 0, 0
        for pfx_len, new_len in zip(prefix_lengths, new_lengths):
            expected_k_parts.append(prefix_k[pk_off : pk_off + pfx_len])
            expected_k_parts.append(new_k[nk_off : nk_off + new_len])
            expected_v_parts.append(prefix_v[pk_off : pk_off + pfx_len])
            expected_v_parts.append(new_v[nk_off : nk_off + new_len])
            pk_off += pfx_len
            nk_off += new_len
        expected_cache_k = torch.cat(expected_k_parts, dim=0)
        expected_cache_v = torch.cat(expected_v_parts, dim=0)
        self.assertTrue(
            torch.equal(cache_k, expected_cache_k),
            f"KV cache K mismatch: max_diff="
            f"{(cache_k.float() - expected_cache_k.float()).abs().max().item()}",
        )
        self.assertTrue(
            torch.equal(cache_v, expected_cache_v),
            f"KV cache V mismatch: max_diff="
            f"{(cache_v.float() - expected_cache_v.float()).abs().max().item()}",
        )

    # ==================================================================
    # Case 1: No-prefix cp_size=4 (all ranks)
    # ==================================================================

    def test_no_prefix_cp4_rank0(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[64], cp_rank=0)

    def test_no_prefix_cp4_rank1(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[64], cp_rank=1)

    def test_no_prefix_cp4_rank2(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[64], cp_rank=2)

    def test_no_prefix_cp4_rank3(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[64], cp_rank=3)

    def test_no_prefix_multi_batch_cp4(self):
        self.run_no_prefix(
            batch_size=2,
            sequence_lengths=[64, 128],
            cp_rank=0,
        )

    # ==================================================================
    # Case 2: With-prefix cp_size=4
    # ==================================================================

    def test_prefix_cp4_rank0(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[64],
            prefix_lengths=[64],
            cp_size=4,
            cp_rank=0,
            tokens_per_block=16,
        )

    def test_prefix_cp4_rank2(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[64],
            prefix_lengths=[64],
            cp_size=4,
            cp_rank=2,
            tokens_per_block=16,
        )

    def test_prefix_multi_batch_cp4(self):
        self.run_with_prefix(
            batch_size=2,
            new_lengths=[64, 128],
            prefix_lengths=[64, 128],
            cp_size=4,
            cp_rank=1,
            tokens_per_block=16,
        )

    # ==================================================================
    # Case 3: Irregular seq_len (non-power-of-2, partial pages)
    # ==================================================================

    def test_no_prefix_irregular_seqlen(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[24], cp_rank=0)

    def test_no_prefix_irregular_multi_batch(self):
        self.run_no_prefix(
            batch_size=2,
            sequence_lengths=[24, 40],
            cp_rank=1,
        )

    def test_prefix_irregular_seqlen(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[24],
            prefix_lengths=[48],
            cp_size=4,
            cp_rank=0,
            tokens_per_block=16,
        )

    def test_prefix_irregular_multi_batch(self):
        self.run_with_prefix(
            batch_size=2,
            new_lengths=[24, 40],
            prefix_lengths=[48, 32],
            cp_size=4,
            cp_rank=2,
            tokens_per_block=16,
        )


if __name__ == "__main__":
    unittest.main()
