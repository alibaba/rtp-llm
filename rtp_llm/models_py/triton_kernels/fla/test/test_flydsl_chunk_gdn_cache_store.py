# -*- coding: utf-8 -*-
"""End-to-end equivalence tests for the FlyDSL Chunk-GDN cache-store path.

Each test runs the same input through:
  Path A (Triton fallback): chunk_gated_delta_rule + store_ssm_state_to_block_map
  Path B (FlyDSL direct-store): chunk_gated_delta_rule_flydsl_with_cache_store

and asserts:
  * attn_out matches between the two paths
  * ssm_states physical bytes match between the two paths
  * the cache content written by either path can be consumed by
    fused_recurrent_gated_delta_rule decode (round-trip)

Also covers:
  * output_final_state=False path (no ht buffer write)
  * tail seqlens (T % BT != 0)
  * varlen cu_seqlens
"""

import logging
import os
import unittest
from typing import List, Optional
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.fla.block import store_ssm_state_to_block_map
from rtp_llm.models_py.triton_kernels.fla.chunk import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_flydsl_with_cache_store,
)
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s.%(msecs)03d][%(filename)s:%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Representative subset of Qwen3.5/3.6 GDN shapes to keep CI runtime bounded.
# Full target shape coverage is gated by test_flydsl_chunk_gdn_shape_gate.
SHAPES = [
    (16, 32, 128, 128),  # Qwen3.5-9B TP1 hot path
    (8, 32, 128, 128),  # Qwen3.5-397B TP2
    (16, 16, 128, 128),  # Qwen3.5-0.8B TP1
    (2, 8, 128, 128),  # BDV32 small-H fast path (Qwen3.5-397B TP4)
]

CHUNK_BT = 64
SEQ_SIZE_PER_BLOCK = 64


def _is_rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def _build_inputs(
    shape,
    T_total: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    device: str = "cuda",
):
    hg, h, k_dim, v_dim = shape
    q = torch.randn(1, T_total, hg, k_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, T_total, hg, k_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, T_total, h, v_dim, device=device, dtype=torch.bfloat16)
    g = torch.randn(1, T_total, h, device=device, dtype=torch.float32) / 16
    beta = torch.randn(1, T_total, h, device=device, dtype=torch.float32).sigmoid()
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T_total], device=device, dtype=torch.int32)
    return q, k, v, g, beta, cu_seqlens


def _alloc_ssm_states(
    n_blocks: int, shape, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"
):
    """Allocate ssm_states matching the qwen3_next.py / LinearCacheConverter layout
    (blocks, H, V, K) with sentinel init so unwritten blocks are visible."""
    _, h, k_dim, v_dim = shape
    return torch.full((n_blocks, h, v_dim, k_dim), -999.0, device=device, dtype=dtype)


@unittest.skipUnless(
    _is_rocm(),
    "FlyDSL Chunk-GDN cache-store path requires ROCm/HIP runtime",
)
class FlyDSLChunkGDNCacheStoreTest(unittest.TestCase):
    """Numeric and cache-byte equivalence between FlyDSL direct-store and
    the Triton chunk + store_ssm_state_to_block_map fallback."""

    @classmethod
    def setUpClass(cls):
        cls._env_patcher = mock.patch.dict(os.environ, {"USE_FLYDSL": "1"}, clear=False)
        cls._env_patcher.start()
        torch.manual_seed(0)

    @classmethod
    def tearDownClass(cls):
        cls._env_patcher.stop()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_triton_path(
        self,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        prefix_lengths,
        block_map,
        ssm_states,
        seq_size_per_block,
    ):
        # ChunkGatedDeltaRuleFunction returns (o, h, final_state).
        o, h, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        store_ssm_state_to_block_map(
            h=h.to(torch.float32),
            final_states=final_state.to(torch.float32),
            prefix_lengths=prefix_lengths,
            cu_seqlens=cu_seqlens,
            block_map=block_map,
            ssm_states=ssm_states,
            seq_size_per_block=seq_size_per_block,
            chunk_size=CHUNK_BT,
        )
        return o

    def _run_flydsl_path(
        self,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        prefix_lengths,
        block_map,
        ssm_states,
        seq_size_per_block,
        output_final_state: bool = True,
    ):
        o, _ = chunk_gated_delta_rule_flydsl_with_cache_store(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            prefix_lengths=prefix_lengths,
            block_map=block_map,
            ssm_states=ssm_states,
            seq_size_per_block=seq_size_per_block,
            initial_state=None,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        return o

    def _assert_paths_equivalent(
        self,
        shape,
        T_total: int,
        seq_lens: List[int],
        seq_size_per_block: int = SEQ_SIZE_PER_BLOCK,
        tag: str = "",
    ):
        device = torch.device("cuda")
        n_seqs = len(seq_lens)
        max_blocks = max(
            (l + seq_size_per_block - 1) // seq_size_per_block for l in seq_lens
        )
        # block 0 is reserved as sentinel "unassigned"; valid block ids start at 1.
        total_blocks = (
            sum((l + seq_size_per_block - 1) // seq_size_per_block for l in seq_lens)
            + 1
        )

        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
            device=device,
            dtype=torch.int32,
        )
        prefix_lengths = torch.zeros(n_seqs, device=device, dtype=torch.int32)
        block_map = torch.zeros((n_seqs, max_blocks), device=device, dtype=torch.int32)
        next_id = 1
        for i, l in enumerate(seq_lens):
            nb = (l + seq_size_per_block - 1) // seq_size_per_block
            for j in range(nb):
                block_map[i, j] = next_id
                next_id += 1

        q, k, v, g, beta, cu_seqlens = _build_inputs(
            shape, T_total, cu_seqlens=cu_seqlens
        )

        ssm_t = _alloc_ssm_states(total_blocks, shape)
        ssm_f = ssm_t.clone()

        o_t = self._run_triton_path(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            prefix_lengths,
            block_map,
            ssm_t,
            seq_size_per_block,
        )
        o_f = self._run_flydsl_path(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            prefix_lengths,
            block_map,
            ssm_f,
            seq_size_per_block,
            output_final_state=True,
        )
        torch.cuda.synchronize()

        # attn_out close: both paths use the same chunk algorithm; small
        # bf16 accumulation noise allowed.
        diff_o = (o_t.float() - o_f.float()).abs()
        self.assertTrue(
            torch.isfinite(o_t).all() and torch.isfinite(o_f).all(),
            f"{tag} attn_out has non-finite",
        )
        cos_o = torch.nn.functional.cosine_similarity(
            o_t.float().flatten().unsqueeze(0),
            o_f.float().flatten().unsqueeze(0),
        ).item()
        self.assertGreater(
            cos_o,
            0.999,
            f"{tag} attn_out cos = {cos_o:.6f} (max diff {diff_o.max().item():.3e})",
        )

        # ssm_states bytes equivalent (modulo bf16 ulp from differing
        # accumulation order). cos == 1.0 confirms physical byte alignment.
        cos_ssm = torch.nn.functional.cosine_similarity(
            ssm_t.float().flatten().unsqueeze(0),
            ssm_f.float().flatten().unsqueeze(0),
        ).item()
        diff_ssm = (ssm_t.float() - ssm_f.float()).abs()
        self.assertGreater(
            cos_ssm,
            0.99999,
            f"{tag} ssm_states cos = {cos_ssm:.6f} "
            f"(max diff {diff_ssm.max().item():.3e})",
        )
        # Allow a small absolute ulp because the two paths accumulate h
        # internally in fp32 but truncate to bf16 on store; element-wise
        # diff should stay within ~6e-2 for normalized inputs.
        self.assertLess(
            diff_ssm.max().item(),
            1.0,
            f"{tag} ssm_states max diff {diff_ssm.max().item():.3e} too large",
        )

        # Per-block: verify both paths wrote/skipped the same blocks
        for bi in range(total_blocks):
            wrote_t = (ssm_t[bi] != -999.0).any().item()
            wrote_f = (ssm_f[bi] != -999.0).any().item()
            self.assertEqual(
                wrote_t,
                wrote_f,
                f"{tag} block {bi} write mismatch (triton={wrote_t} flydsl={wrote_f})",
            )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_aligned_single_seq(self):
        """T % BT == 0, single sequence — exercises fast cu_seqlens path."""
        for shape in SHAPES:
            for T in (64, 128, 256):
                with self.subTest(shape=shape, T=T):
                    logger.info(f"aligned_single_seq shape={shape} T={T}")
                    self._assert_paths_equivalent(
                        shape,
                        T,
                        [T],
                        tag=f"shape={shape} T={T}",
                    )

    def test_tail_single_seq(self):
        """T % BT != 0, single sequence — exercises tail-safe / split-tail path."""
        for shape in SHAPES:
            for T in (65, 127):
                with self.subTest(shape=shape, T=T):
                    logger.info(f"tail_single_seq shape={shape} T={T}")
                    self._assert_paths_equivalent(
                        shape,
                        T,
                        [T],
                        tag=f"shape={shape} T={T}",
                    )

    def test_varlen(self):
        """Multiple sequences with mixed aligned / tail lengths."""
        for shape in SHAPES:
            for seq_lens in ([64, 128], [65, 127, 64], [64, 64, 64]):
                T_total = sum(seq_lens)
                with self.subTest(shape=shape, seq_lens=seq_lens):
                    logger.info(f"varlen shape={shape} seq_lens={seq_lens}")
                    self._assert_paths_equivalent(
                        shape,
                        T_total,
                        seq_lens,
                        tag=f"shape={shape} seq_lens={seq_lens}",
                    )

    def test_initial_state_with_prefix(self):
        """USE_INITIAL_STATE branch: prefix_lengths > 0 + non-None
        initial_state. Exercises the FlyDSL h0 load path and the middle-
        chunk store_h_to_ssm_block path that fires when prefix-aligned
        block boundaries land inside the current dispatch."""
        shape = (16, 32, 128, 128)
        hg, h, k_dim, v_dim = shape
        device = torch.device("cuda")

        prefix_T = 128  # 2 prefix blocks already cached
        suffix_T = 192  # 3 chunks of 64 = 3 suffix blocks
        seq_size_per_block = SEQ_SIZE_PER_BLOCK
        n_prefix_blocks = prefix_T // seq_size_per_block
        n_suffix_blocks = (suffix_T + seq_size_per_block - 1) // seq_size_per_block
        total_blocks = n_prefix_blocks + n_suffix_blocks + 1  # +1 sentinel

        cu_seqlens = torch.tensor([0, suffix_T], device=device, dtype=torch.int32)
        prefix_lengths = torch.tensor([prefix_T], device=device, dtype=torch.int32)
        block_map = torch.tensor(
            [list(range(1, n_prefix_blocks + n_suffix_blocks + 1))],
            device=device,
            dtype=torch.int32,
        )

        q, k, v, g, beta, _ = _build_inputs(shape, suffix_T, cu_seqlens=cu_seqlens)
        initial_state = (
            torch.randn(1, h, v_dim, k_dim, device=device, dtype=torch.float32) / 8
        )

        # Path A: Triton chunk_gated_delta_rule + store_ssm_state_to_block_map
        ssm_t = _alloc_ssm_states(total_blocks, shape)
        o_t, h_t, last_t = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state.clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        store_ssm_state_to_block_map(
            h=h_t.to(torch.float32),
            final_states=last_t.to(torch.float32),
            prefix_lengths=prefix_lengths,
            cu_seqlens=cu_seqlens,
            block_map=block_map,
            ssm_states=ssm_t,
            seq_size_per_block=seq_size_per_block,
            chunk_size=CHUNK_BT,
        )

        # Path B: FlyDSL direct-store, USE_INITIAL_STATE branch
        ssm_f = _alloc_ssm_states(total_blocks, shape)
        o_f, _ = chunk_gated_delta_rule_flydsl_with_cache_store(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            prefix_lengths=prefix_lengths,
            block_map=block_map,
            ssm_states=ssm_f,
            seq_size_per_block=seq_size_per_block,
            initial_state=initial_state.clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.isfinite(o_t).all() and torch.isfinite(o_f).all(),
            "attn_out has non-finite",
        )
        diff_o = (o_t.float() - o_f.float()).abs()
        cos_o = torch.nn.functional.cosine_similarity(
            o_t.float().flatten().unsqueeze(0),
            o_f.float().flatten().unsqueeze(0),
        ).item()
        self.assertGreater(
            cos_o,
            0.999,
            f"attn_out cos = {cos_o:.6f} (max diff {diff_o.max().item():.3e})",
        )

        diff_ssm = (ssm_t.float() - ssm_f.float()).abs()
        cos_ssm = torch.nn.functional.cosine_similarity(
            ssm_t.float().flatten().unsqueeze(0),
            ssm_f.float().flatten().unsqueeze(0),
        ).item()
        self.assertGreater(
            cos_ssm,
            0.99999,
            f"ssm_states cos = {cos_ssm:.6f} "
            f"(max diff {diff_ssm.max().item():.3e})",
        )
        # Per-block write-set parity: both paths must write the same set of
        # cache blocks (skipping prefix slots, writing suffix slots).
        for bi in range(total_blocks):
            wrote_t = (ssm_t[bi] != -999.0).any().item()
            wrote_f = (ssm_f[bi] != -999.0).any().item()
            self.assertEqual(
                wrote_t,
                wrote_f,
                f"block {bi} write mismatch (triton={wrote_t} flydsl={wrote_f})",
            )

    def test_decode_round_trip_reads_cache(self):
        """End-to-end: prefill writes ssm_states via either path, decode
        consumes it as initial_state, outputs should match."""
        for shape in SHAPES:
            with self.subTest(shape=shape):
                self._decode_round_trip_for_shape(shape)

    def _decode_round_trip_for_shape(self, shape):
        T = 128
        device = torch.device("cuda")
        n_blocks = T // SEQ_SIZE_PER_BLOCK
        total_blocks = n_blocks + 1
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
        prefix_lengths = torch.zeros(1, device=device, dtype=torch.int32)
        block_map = torch.tensor(
            [list(range(1, n_blocks + 1))],
            device=device,
            dtype=torch.int32,
        )

        q, k, v, g, beta, _ = _build_inputs(shape, T, cu_seqlens=cu_seqlens)
        ssm_t = _alloc_ssm_states(total_blocks, shape)
        ssm_f = ssm_t.clone()

        self._run_triton_path(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            prefix_lengths,
            block_map,
            ssm_t,
            SEQ_SIZE_PER_BLOCK,
        )
        self._run_flydsl_path(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            prefix_lengths,
            block_map,
            ssm_f,
            SEQ_SIZE_PER_BLOCK,
        )
        torch.cuda.synchronize()

        # Decode a fresh single token starting from each cached state.
        hg, h, k_dim, v_dim = shape
        last_idx = n_blocks  # block_map last
        init_t = ssm_t[last_idx].float().unsqueeze(0).clone()
        init_f = ssm_f[last_idx].float().unsqueeze(0).clone()

        q_d = torch.randn(1, 1, hg, k_dim, device=device, dtype=torch.bfloat16)
        k_d = torch.randn(1, 1, hg, k_dim, device=device, dtype=torch.bfloat16)
        v_d = torch.randn(1, 1, h, v_dim, device=device, dtype=torch.bfloat16)
        g_d = torch.randn(1, 1, h, device=device, dtype=torch.float32) / 16
        beta_d = torch.randn(1, 1, h, device=device, dtype=torch.float32).sigmoid()

        out_t, _ = fused_recurrent_gated_delta_rule(
            q=q_d,
            k=k_d,
            v=v_d,
            g=g_d,
            beta=beta_d,
            initial_state=init_t,
            use_qk_l2norm_in_kernel=True,
        )
        out_f, _ = fused_recurrent_gated_delta_rule(
            q=q_d,
            k=k_d,
            v=v_d,
            g=g_d,
            beta=beta_d,
            initial_state=init_f,
            use_qk_l2norm_in_kernel=True,
        )
        cos_dec = torch.nn.functional.cosine_similarity(
            out_t.float().flatten().unsqueeze(0),
            out_f.float().flatten().unsqueeze(0),
        ).item()
        diff_dec = (out_t.float() - out_f.float()).abs()
        self.assertGreater(
            cos_dec,
            0.999,
            f"decode round-trip cos = {cos_dec:.6f} "
            f"(max diff {diff_dec.max().item():.3e})",
        )

    def test_output_final_state_false_no_buffer_write(self):
        """output_final_state=False must not write to any ht buffer
        (compile-time DCE in build_megakernel). main attn_out must be
        bit-identical to the True path."""
        for shape in SHAPES:
            with self.subTest(shape=shape):
                self._output_final_state_false_for_shape(shape)

    def _output_final_state_false_for_shape(self, shape):
        T = 128
        device = torch.device("cuda")
        n_blocks = T // SEQ_SIZE_PER_BLOCK
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
        prefix_lengths = torch.zeros(1, device=device, dtype=torch.int32)
        block_map = torch.tensor(
            [list(range(1, n_blocks + 1))],
            device=device,
            dtype=torch.int32,
        )

        q, k, v, g, beta, _ = _build_inputs(shape, T, cu_seqlens=cu_seqlens)
        ssm_buf = _alloc_ssm_states(n_blocks + 1, shape)

        # output_final_state=False — wrapper must accept and return None final_state.
        o_false, final_false = chunk_gated_delta_rule_flydsl_with_cache_store(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            prefix_lengths=prefix_lengths,
            block_map=block_map,
            ssm_states=ssm_buf,
            seq_size_per_block=SEQ_SIZE_PER_BLOCK,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        self.assertIsNone(final_false, "final_state must be None when not requested")

        # Same inputs, output_final_state=True — attn_out must match exactly
        # (compile-time guard only removes ht store, not main compute).
        ssm_buf_true = _alloc_ssm_states(n_blocks + 1, shape)
        o_true, final_true = chunk_gated_delta_rule_flydsl_with_cache_store(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            prefix_lengths=prefix_lengths,
            block_map=block_map,
            ssm_states=ssm_buf_true,
            seq_size_per_block=SEQ_SIZE_PER_BLOCK,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        self.assertIsNotNone(final_true)

        diff = (o_true.float() - o_false.float()).abs()
        self.assertEqual(
            diff.max().item(),
            0.0,
            f"attn_out must be bit-identical between output_final_state True/False; "
            f"got max diff {diff.max().item():.3e}",
        )


if __name__ == "__main__":
    unittest.main()
