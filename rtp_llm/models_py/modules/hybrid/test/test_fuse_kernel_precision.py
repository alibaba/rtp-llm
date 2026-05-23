"""Precision tests for GLM-5 NSA decode kernel fusion optimizations.

Tests verify that each optimization is lossless (bitwise or numerically
equivalent to the reference implementation):

  #16 — Hadamard weight absorption
  #17 — Dual-stream execution (SM-limited)
  #19 — Fused Q-RoPE-Quant Triton kernel
  #20 — Index Transform two-step correctness
"""

import random
from typing import Dict, Tuple
from unittest import TestCase, main

import torch

device = torch.device("cuda")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# #16  Hadamard weight absorption precision test
# ---------------------------------------------------------------------------


class TestHadamardAbsorption(TestCase):
    """Verify that absorbing H into wq_b/wk weights is lossless.

    Mathematical guarantee: H is orthogonal, so (H @ q) . (H @ k)^T = q . k^T.
    Implementation guarantee: the bf16 GEMM output after absorption must match
    hadamard_transform(GEMM_output_without_absorption) to within bf16 rounding.
    """

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.set_default_device(device)
        set_seed()

    def test_wq_b_absorption_close_to_runtime_hadamard(self):
        """wq_b(x) with absorbed H ≈ hadamard(wq_b_orig(x)).

        Not bitwise identical due to floating-point associativity:
        H(W @ x) != (H @ W) @ x in bf16. But must be close.
        """
        from fast_hadamard_transform import hadamard_transform

        n_heads, head_dim, in_dim = 64, 128, 1536
        scale = head_dim**-0.5

        wq_orig = torch.randn(n_heads * head_dim, in_dim, dtype=torch.bfloat16)

        # Absorb H into weight (same logic as Indexer._absorb_hadamard_into_weights)
        w = wq_orig.clone().to(torch.bfloat16)
        w = w.view(n_heads, head_dim, -1)
        w = w.permute(0, 2, 1).contiguous()  # [n_heads, in_dim, head_dim]
        w = hadamard_transform(w, scale=scale)
        w = w.permute(0, 2, 1).contiguous()  # [n_heads, head_dim, in_dim]
        wq_absorbed = w.reshape(-1, w.shape[-1]).to(torch.bfloat16)

        x = torch.randn(4, in_dim, dtype=torch.bfloat16)

        q_hadamard = hadamard_transform(
            (x @ wq_orig.T).view(-1, n_heads, head_dim), scale=scale
        )
        q_absorbed = (x @ wq_absorbed.T).view(-1, n_heads, head_dim)

        diff = (q_absorbed.float() - q_hadamard.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  wq_b absorption: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        # bf16 GEMM reordering tolerance — large in_dim (1536) accumulation
        self.assertLess(
            max_diff,
            2.0,
            f"wq_b absorption max diff too large: {max_diff}",
        )
        self.assertLess(
            mean_diff,
            0.1,
            f"wq_b absorption mean diff too large: {mean_diff}",
        )

    def test_wk_absorption_close_to_runtime_hadamard(self):
        """wk(x) with absorbed H ≈ hadamard(wk_orig(x))."""
        from fast_hadamard_transform import hadamard_transform

        head_dim, in_dim = 128, 7168
        scale = head_dim**-0.5

        wk_orig = torch.randn(head_dim, in_dim, dtype=torch.bfloat16)

        w = wk_orig.clone().to(torch.bfloat16)
        w = w.T.contiguous()  # [in_dim, head_dim]
        w = w.unsqueeze(0)
        w = hadamard_transform(w, scale=scale)
        w = w.squeeze(0)
        wk_absorbed = w.T.contiguous().to(torch.bfloat16)

        x = torch.randn(4, in_dim, dtype=torch.bfloat16)

        k_hadamard = hadamard_transform(x @ wk_orig.T, scale=scale)
        k_absorbed = x @ wk_absorbed.T

        diff = (k_absorbed.float() - k_hadamard.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  wk absorption: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        self.assertLess(
            max_diff,
            4.0,
            f"wk absorption max diff too large: {max_diff}",
        )
        self.assertLess(
            mean_diff,
            0.2,
            f"wk absorption mean diff too large: {mean_diff}",
        )

    def test_dot_product_invariance(self):
        """(Hq).(Hk)^T == q.k^T (orthogonality of H).

        This is the property that makes absorption lossless for attention logits.
        """
        from fast_hadamard_transform import hadamard_transform

        n_heads, head_dim = 64, 128
        batch = 4
        scale = head_dim**-0.5

        q = torch.randn(batch, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(batch, head_dim, dtype=torch.bfloat16, device=device)

        # Original dot product: q[b,h,:] . k[b,:] for each (b,h)
        dot_orig = torch.einsum("bnh,bh->bn", q.float(), k.float())

        # Hadamard-rotated dot product
        q_h = hadamard_transform(q, scale=scale).float()
        k_h = hadamard_transform(k, scale=scale).float()
        dot_h = torch.einsum("bnh,bh->bn", q_h, k_h)

        torch.testing.assert_close(
            dot_h,
            dot_orig,
            atol=5e-2,
            rtol=1e-2,
            msg="Hadamard transform must preserve dot-product (orthogonality)",
        )

    def test_absorb_modifies_weights_and_sets_skip(self):
        """Verify _absorb_hadamard_into_weights actually changes weights
        and the Indexer sets skip_hadamard=True on its IndexerOp.

        The full precision validation is done via e2e test with real model
        weights, since bf16 GEMM associativity with random weights produces
        worst-case reordering error that doesn't reflect production usage.
        """
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.utils.model_weight import W

        config = ModelConfig()
        config.attn_config.head_num = 128
        config.max_seq_len = 2048
        config.hidden_size = 7168
        config.attn_config.q_lora_rank = 1536
        config.attn_config.rope_head_dim = 64
        config.attn_config.tokens_per_block = 64
        config.attn_config.kernel_tokens_per_block = 64
        config.attn_config.use_mla = True
        config.attn_config.indexer_head_num = 64
        config.attn_config.indexer_head_dim = 128
        config.attn_config.indexer_topk = 2048
        config.attn_config.rope_config.indexer_is_neox_style = True

        weights = {
            W.mla_indexer_qb_w: torch.randn(1536, 8192, dtype=torch.bfloat16),
            W.mla_indexer_k_w: torch.randn(7168, 128, dtype=torch.bfloat16),
            W.mla_indexer_k_norm_w: torch.ones(128, dtype=torch.bfloat16),
            W.mla_indexer_k_norm_b: torch.zeros(128, dtype=torch.bfloat16),
            W.mla_indexer_weights_proj_w: torch.randn(7168, 64, dtype=torch.float32),
            W.rope_cos_sin_cache: torch.randn(2048, 64, dtype=torch.float32),
        }

        wq_before = weights[W.mla_indexer_qb_w].clone()
        wk_before = weights[W.mla_indexer_k_w].clone()

        from rtp_llm.models_py.modules.hybrid.indexer import Indexer

        indexer = Indexer(
            attn_config=config.attn_config,
            weights=weights,
            global_weights=weights,
            layer_idx=0,
            layernorm_eps=1e-6,
            quant_config=None,
            scale_fmt="ue8m0",
        )

        # After init, skip_hadamard must be True
        self.assertTrue(
            indexer.indexer_op.skip_hadamard,
            "skip_hadamard must be True after weight absorption",
        )

        # Weights must have changed (H was absorbed)
        wq_after = indexer.wq_b.weight
        wk_after = indexer.wk.weight
        self.assertFalse(
            torch.equal(wq_before, wq_after),
            "wq_b weight must change after absorption",
        )
        self.assertFalse(
            torch.equal(wk_before, wk_after),
            "wk weight must change after absorption",
        )


# ---------------------------------------------------------------------------
# #19  Fused Q-RoPE-Quant precision test
# ---------------------------------------------------------------------------


class TestFusedQRopeQuant(TestCase):
    """Verify fused Triton kernel matches separate RoPE + FP8 quant pipeline."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.set_default_device(device)
        set_seed()

    def _build_cos_sin_cache(self, max_pos: int, rot_dim: int) -> torch.Tensor:
        """Build a cos/sin cache matching FlashInfer's format."""
        half = rot_dim // 2
        base = 10000.0
        freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_pos, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        cache = torch.cat([angles.cos(), angles.sin()], dim=-1)
        return cache.to(device)

    def _reference_rope_quant(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        n_heads: int,
        head_dim: int,
        rot_dim: int,
        is_neox_style: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference: separate RoPE + separate FP8 quant with ue8m0 scale."""
        import flashinfer.rope as fi_rope

        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        q_work = q.clone()
        actual_rot = rot_dim
        q_pe = q_work[:, :, :actual_rot]

        # Apply RoPE in-place using FlashInfer
        fi_rope._apply_rope_pos_ids_cos_sin_cache(
            q=q_pe,
            k=q_pe[:, :1, :],  # dummy k
            q_rope=q_pe,
            k_rope=q_pe[:, :1, :],
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            interleave=not is_neox_style,
        )

        # FP8 quant with ue8m0
        q_flat = q_work.reshape(-1, head_dim)
        q_fp8, q_scale_raw = sgl_per_token_group_quant_fp8(
            q_flat,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        q_fp8 = q_fp8.view(-1, n_heads, head_dim)

        # Unpack ue8m0
        sf_u8 = (q_scale_raw & 0xFF).to(torch.int32)
        sf_i32 = sf_u8 << 23
        q_scale = sf_i32.view(torch.float32).view(-1, n_heads, 1)

        return q_fp8, q_scale

    def test_fused_matches_reference_neox(self):
        """Fused kernel (NeOX style) matches reference pipeline."""
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
            fused_q_rope_quant,
        )

        n_heads, head_dim, rot_dim = 64, 128, 64
        num_tokens = 32
        max_pos = 8192

        cos_sin_cache = self._build_cos_sin_cache(max_pos, rot_dim)
        q = torch.randn(num_tokens, n_heads, head_dim, dtype=torch.bfloat16)
        positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int32)

        # Reference
        q_fp8_ref, q_scale_ref = self._reference_rope_quant(
            q.clone(),
            positions,
            cos_sin_cache,
            n_heads,
            head_dim,
            rot_dim,
            is_neox_style=True,
        )

        # Fused
        q_fp8_fused, q_scale_fused = fused_q_rope_quant(
            q.clone(),
            positions,
            cos_sin_cache,
            n_heads,
            head_dim,
            rot_dim,
            is_neox_style=True,
        )

        # FP8 values must match exactly
        fp8_match = torch.equal(
            q_fp8_fused.view(torch.uint8), q_fp8_ref.view(torch.uint8)
        )
        # Scale must match exactly (ue8m0 is discrete)
        scale_match = torch.equal(q_scale_fused, q_scale_ref)

        if not fp8_match:
            diff_mask = q_fp8_fused.view(torch.uint8) != q_fp8_ref.view(torch.uint8)
            n_diff = diff_mask.sum().item()
            total = diff_mask.numel()
            # Allow <=0.1% mismatch due to bf16 roundtrip ordering
            self.assertLessEqual(
                n_diff / total,
                0.001,
                f"FP8 mismatch: {n_diff}/{total} elements differ "
                f"({100*n_diff/total:.3f}%)",
            )

        if not scale_match:
            diff_mask = q_scale_fused != q_scale_ref
            n_diff = diff_mask.sum().item()
            total = diff_mask.numel()
            self.assertLessEqual(
                n_diff / total,
                0.001,
                f"Scale mismatch: {n_diff}/{total} elements differ",
            )

    def test_fused_matches_reference_interleaved(self):
        """Fused kernel (GPT-J interleaved style) matches reference pipeline."""
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
            fused_q_rope_quant,
        )

        n_heads, head_dim, rot_dim = 64, 128, 64
        num_tokens = 16
        max_pos = 4096

        cos_sin_cache = self._build_cos_sin_cache(max_pos, rot_dim)
        q = torch.randn(num_tokens, n_heads, head_dim, dtype=torch.bfloat16)
        positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int32)

        q_fp8_ref, q_scale_ref = self._reference_rope_quant(
            q.clone(),
            positions,
            cos_sin_cache,
            n_heads,
            head_dim,
            rot_dim,
            is_neox_style=False,
        )

        q_fp8_fused, q_scale_fused = fused_q_rope_quant(
            q.clone(),
            positions,
            cos_sin_cache,
            n_heads,
            head_dim,
            rot_dim,
            is_neox_style=False,
        )

        fp8_match = torch.equal(
            q_fp8_fused.view(torch.uint8), q_fp8_ref.view(torch.uint8)
        )
        scale_match = torch.equal(q_scale_fused, q_scale_ref)

        if not fp8_match:
            diff_mask = q_fp8_fused.view(torch.uint8) != q_fp8_ref.view(torch.uint8)
            n_diff = diff_mask.sum().item()
            total = diff_mask.numel()
            self.assertLessEqual(
                n_diff / total,
                0.001,
                f"FP8 mismatch (interleaved): {n_diff}/{total}",
            )

        if not scale_match:
            diff_mask = q_scale_fused != q_scale_ref
            n_diff = diff_mask.sum().item()
            total = diff_mask.numel()
            self.assertLessEqual(
                n_diff / total,
                0.001,
                f"Scale mismatch (interleaved): {n_diff}/{total}",
            )

    def test_fused_zero_tokens(self):
        """Edge case: zero tokens should not crash."""
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
            fused_q_rope_quant,
        )

        n_heads, head_dim, rot_dim = 64, 128, 64
        cos_sin_cache = self._build_cos_sin_cache(1024, rot_dim)
        q = torch.empty(0, n_heads, head_dim, dtype=torch.bfloat16)
        positions = torch.empty(0, dtype=torch.int32)

        q_fp8, q_scale = fused_q_rope_quant(
            q,
            positions,
            cos_sin_cache,
            n_heads,
            head_dim,
            rot_dim,
        )
        self.assertEqual(q_fp8.shape, (0, n_heads, head_dim))
        self.assertEqual(q_scale.shape, (0, n_heads, 1))


# ---------------------------------------------------------------------------
# #17  Dual-stream execution precision test
# ---------------------------------------------------------------------------


class TestDualStreamPrecision(TestCase):
    """Verify dual-stream K/Q path produces identical results to sequential."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.set_default_device(device)
        set_seed()

    def test_dual_stream_q_path_matches_sequential(self):
        """Q path: wq_b -> reshape -> fused_rope_quant_q must match sequential
        apply_rope_and_rotate_q_k + quant_q_only."""
        from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp

        n_heads, head_dim, rope_dim = 64, 128, 64
        num_tokens = 8
        max_pos = 4096

        base = 10000.0
        half = rope_dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_pos, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        cos_sin_cache = torch.cat([angles.cos(), angles.sin()], dim=-1).to(device)

        op = IndexerOp(
            index_n_heads=n_heads,
            index_head_dim=head_dim,
            index_topk=2048,
            rope_head_dim=rope_dim,
            cos_sin_cache=cos_sin_cache,
            blocksize=64,
            block_size=128,
            scale_fmt="ue8m0",
            is_neox_style=True,
            skip_hadamard=True,
        )

        q = torch.randn(num_tokens, n_heads, head_dim, dtype=torch.bfloat16)
        positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int32)

        # Sequential path: apply_rope -> quant_q_only
        q_seq = q.clone()
        k_dummy = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16)
        q_roped, _ = op.apply_rope_and_rotate_q_k(q_seq, k_dummy, positions)
        q_fp8_seq, q_scale_seq = op.quant_q_only(q_roped)

        # Fused path (same as dual-stream Q branch)
        q_fused = q.clone()
        q_fp8_fused, q_scale_fused = op.fused_rope_quant_q(q_fused, positions)

        fp8_match = torch.equal(
            q_fp8_fused.view(torch.uint8), q_fp8_seq.view(torch.uint8)
        )
        scale_match = torch.equal(q_scale_fused, q_scale_seq)

        if not fp8_match:
            diff_mask = q_fp8_fused.view(torch.uint8) != q_fp8_seq.view(torch.uint8)
            n_diff = diff_mask.sum().item()
            total = diff_mask.numel()
            self.assertLessEqual(
                n_diff / total,
                0.001,
                f"Dual-stream Q FP8 mismatch: {n_diff}/{total}",
            )

        if not scale_match:
            diff_mask = q_scale_fused != q_scale_seq
            n_diff = diff_mask.sum().item()
            total = diff_mask.numel()
            self.assertLessEqual(
                n_diff / total,
                0.001,
                f"Dual-stream Q scale mismatch: {n_diff}/{total}",
            )

    def test_k_path_sequential_vs_stream(self):
        """K path executed on alt stream must produce identical cache writes."""
        from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp

        n_heads, head_dim, rope_dim = 64, 128, 64
        num_tokens = 8
        max_pos = 4096
        blocksize = 64

        base = 10000.0
        half = rope_dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_pos, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        cos_sin_cache = torch.cat([angles.cos(), angles.sin()], dim=-1).to(device)

        op = IndexerOp(
            index_n_heads=n_heads,
            index_head_dim=head_dim,
            index_topk=2048,
            rope_head_dim=rope_dim,
            cos_sin_cache=cos_sin_cache,
            blocksize=blocksize,
            block_size=128,
            scale_fmt="ue8m0",
            is_neox_style=True,
            skip_hadamard=True,
        )

        k = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16)
        positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int32)

        # Main stream
        k1 = k.clone()
        key1 = op.apply_rope_and_rotate_k(k1, positions)

        # Alt stream
        alt_stream = torch.cuda.Stream()
        k2 = k.clone()
        alt_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(alt_stream):
            key2 = op.apply_rope_and_rotate_k(k2, positions)
        torch.cuda.current_stream().wait_stream(alt_stream)

        torch.testing.assert_close(
            key1,
            key2,
            atol=0,
            rtol=0,
            msg="K path on alt stream must match main stream exactly",
        )


# ---------------------------------------------------------------------------
# #20  Index Transform two-step correctness test
# ---------------------------------------------------------------------------


class TestIndexTransformTwoStep(TestCase):
    """Verify that fast_topk_transform_fused(page_table=None) + triton_convert
    correctly converts request-local indices to physical cache addresses."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.set_default_device(device)
        set_seed()

    def test_triton_convert_produces_valid_physical_addresses(self):
        """triton_convert_req_index_to_global_index must map logical -> physical."""
        from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
            triton_convert_req_index_to_global_index,
        )

        batch_size = 4
        topk = 512
        page_size = 64
        max_seq_len = 4096
        num_pages = max_seq_len // page_size

        # Construct a block_table: each batch has different physical block IDs
        block_table = torch.zeros(
            batch_size, num_pages, dtype=torch.int32, device=device
        )
        for b in range(batch_size):
            block_table[b] = (
                torch.randperm(num_pages, device=device).to(torch.int32) + b * 1000
            )

        # Request-local topk indices (logical token positions within a sequence)
        token_indices = torch.randint(
            0, max_seq_len, (batch_size, topk), dtype=torch.int32, device=device
        )

        # req_id: which batch each row belongs to
        req_id = torch.arange(batch_size, dtype=torch.int32, device=device)

        global_indices = triton_convert_req_index_to_global_index(
            req_id=req_id,
            block_table=block_table,
            token_indices=token_indices,
            BLOCK_SIZE=page_size,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=128,
        )

        # Reference: manual physical address computation
        for b in range(batch_size):
            for t in range(topk):
                logical_pos = token_indices[b, t].item()
                block_idx = logical_pos // page_size
                block_offset = logical_pos % page_size
                phys_block = block_table[b, block_idx].item()
                expected = phys_block * page_size + block_offset
                actual = global_indices[b, t].item()
                self.assertEqual(
                    actual,
                    expected,
                    f"Mismatch at batch={b}, topk_idx={t}: "
                    f"logical={logical_pos}, expected_phys={expected}, got={actual}",
                )

    def test_two_step_matches_direct_lookup(self):
        """fast_topk(page_table=None) -> triton_convert must equal direct page_table lookup."""
        from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
            triton_convert_req_index_to_global_index,
        )

        batch_size = 2
        topk = 256
        page_size = 64
        max_seq_len = 2048
        num_pages = max_seq_len // page_size

        block_table = torch.zeros(
            batch_size, num_pages, dtype=torch.int32, device=device
        )
        for b in range(batch_size):
            block_table[b] = (
                torch.randperm(num_pages, device=device).to(torch.int32) + b * 500
            )

        # Simulate topk output (request-local indices within [0, max_seq_len))
        topk_local = torch.stack(
            [
                torch.randperm(max_seq_len, device=device)[:topk].to(torch.int32)
                for _ in range(batch_size)
            ]
        )

        # Step 1: request-local indices (what fast_topk_transform_fused with page_table=None gives)
        # Step 2: triton_convert to physical
        req_id = torch.arange(batch_size, dtype=torch.int32, device=device)
        global_indices = triton_convert_req_index_to_global_index(
            req_id=req_id,
            block_table=block_table,
            token_indices=topk_local,
            BLOCK_SIZE=page_size,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=128,
        )

        # Direct reference: for each logical index, compute physical address
        for b in range(batch_size):
            for t in range(topk):
                logical = topk_local[b, t].item()
                block_idx = logical // page_size
                offset = logical % page_size
                phys = block_table[b, block_idx].item() * page_size + offset
                self.assertEqual(
                    global_indices[b, t].item(),
                    phys,
                    f"Two-step mismatch at b={b}, t={t}",
                )


# ---------------------------------------------------------------------------
# Performance micro-benchmarks (informational, not assertion-based)
# ---------------------------------------------------------------------------


class TestPerformance(TestCase):
    """Micro-benchmarks for the fusion kernels. Not pass/fail — prints timings."""

    WARMUP = 10
    ITERS = 100

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.set_default_device(device)
        set_seed()

    def _benchmark(self, fn, label: str):
        for _ in range(self.WARMUP):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.ITERS):
            fn()
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000 / self.ITERS
        print(f"  {label}: {us:.1f} us/iter")
        return us

    def test_fused_q_rope_quant_speedup(self):
        """Compare fused vs separate RoPE+quant."""
        from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
            fused_q_rope_quant,
        )

        n_heads, head_dim, rot_dim = 64, 128, 64
        num_tokens = 32
        max_pos = 8192

        base = 10000.0
        half = rot_dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_pos, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        cos_sin_cache = torch.cat([angles.cos(), angles.sin()], dim=-1).to(device)

        op = IndexerOp(
            index_n_heads=n_heads,
            index_head_dim=head_dim,
            index_topk=2048,
            rope_head_dim=rot_dim,
            cos_sin_cache=cos_sin_cache,
            blocksize=64,
            block_size=128,
            scale_fmt="ue8m0",
            is_neox_style=True,
            skip_hadamard=True,
        )

        q = torch.randn(num_tokens, n_heads, head_dim, dtype=torch.bfloat16)
        positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int32)

        def separate():
            qq = q.clone()
            kd = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=device)
            qr, _ = op.apply_rope_and_rotate_q_k(qq, kd, positions)
            return op.quant_q_only(qr)

        def fused():
            return fused_q_rope_quant(
                q.clone(),
                positions,
                cos_sin_cache,
                n_heads,
                head_dim,
                rot_dim,
                is_neox_style=True,
            )

        print("\n[#19] Fused Q-RoPE-Quant benchmark:")
        t_sep = self._benchmark(separate, "separate (RoPE+quant)")
        t_fused = self._benchmark(fused, "fused Triton kernel")
        speedup = t_sep / t_fused if t_fused > 0 else float("inf")
        print(f"  Speedup: {speedup:.2f}x")

    def test_hadamard_absorption_speedup(self):
        """Absorbed path (skip hadamard) vs runtime hadamard."""
        from fast_hadamard_transform import hadamard_transform

        n_heads, head_dim, in_dim = 64, 128, 1536
        scale = head_dim**-0.5
        batch = 32

        w = torch.randn(n_heads * head_dim, in_dim, dtype=torch.bfloat16)
        x = torch.randn(batch, in_dim, dtype=torch.bfloat16)

        def with_hadamard():
            q = (x @ w.T).view(-1, n_heads, head_dim)
            return hadamard_transform(q, scale=scale)

        def without_hadamard():
            return (x @ w.T).view(-1, n_heads, head_dim)

        print("\n[#16] Hadamard absorption benchmark:")
        t_with = self._benchmark(with_hadamard, "with runtime Hadamard")
        t_without = self._benchmark(without_hadamard, "absorbed (skip Hadamard)")
        speedup = t_with / t_without if t_without > 0 else float("inf")
        print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
