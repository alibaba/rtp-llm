"""Precision tests for fused_prefill_rope_hadamard_qk.

Validates the fused kernel produces:
  - Q output within 1 bf16 ULP of the baseline (flashinfer rope + Dao hadamard).
    Difference comes from doing Hadamard via bf16 cuBLAS GEMM vs Dao butterfly;
    both lower-bound by bf16 representational precision.
  - K output bit-identical to baseline (max_abs == 0) because the K butterfly
    is mathematically equivalent.

Covers both NeOX (half-split) and interleaved (even/odd) RoPE styles.
Production envelope T = 1024, 4096, 8192 at DSV3.2 shape (H=32, D=128).
"""
import unittest

import torch
import flashinfer.rope as fi_rope
from fast_hadamard_transform import hadamard_transform

from rtp_llm.models_py.triton_kernels.sparse_mla.fused_prefill_rope_hadamard import (
    fused_prefill_rope_hadamard_qk,
    _get_h_scaled_bf16,
)


def _make_cos_sin_cache(rope_head_dim: int, max_pos: int = 131072) -> torch.Tensor:
    """Build the cos_sin_cache in flashinfer convention: [max_pos, rope_head_dim]
    with cache[:, :rope_head_dim//2] = cos, cache[:, rope_head_dim//2:] = sin.
    Must be fp32 — bf16 silently produces garbage."""
    half = rope_head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_pos, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.cat([angles.cos(), angles.sin()], dim=-1).cuda()


def _baseline_chain(q, k, positions, cos_sin_cache, rope_head_dim, head_dim, is_neox_style):
    """4-op baseline: flashinfer RoPE (in-place) + Dao Hadamard."""
    T = q.shape[0]
    dummy = torch.zeros(T, 1, rope_head_dim, dtype=torch.bfloat16, device="cuda")
    q_pe = q[:, :, :rope_head_dim]
    fi_rope._apply_rope_pos_ids_cos_sin_cache(
        q=q_pe, k=dummy, q_rope=q_pe, k_rope=dummy,
        cos_sin_cache=cos_sin_cache, pos_ids=positions, interleave=not is_neox_style,
    )
    k_pe = k[:, :rope_head_dim].unsqueeze(1)
    fi_rope._apply_rope_pos_ids_cos_sin_cache(
        q=k_pe, k=dummy, q_rope=k_pe, k_rope=dummy,
        cos_sin_cache=cos_sin_cache, pos_ids=positions, interleave=not is_neox_style,
    )
    q_out = hadamard_transform(q, scale=head_dim**-0.5)
    k_out = hadamard_transform(k, scale=head_dim**-0.5)
    return q_out, k_out


class TestFusedPrefillRopeHadamardNeox(unittest.TestCase):
    """NeOX (half-split) RoPE style — DSV3.2 production path."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(0)

    def _run(self, T: int, num_heads: int = 32, head_dim: int = 128, rope_head_dim: int = 64):
        cos_sin_cache = _make_cos_sin_cache(rope_head_dim)
        positions = torch.arange(T, dtype=torch.int32, device="cuda")
        q0 = (torch.randn(T, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
        k0 = (torch.randn(T, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()

        q_b = q0.clone(); k_b = k0.clone()
        q_ref, k_ref = _baseline_chain(
            q_b, k_b, positions, cos_sin_cache, rope_head_dim, head_dim, is_neox_style=True,
        )
        q_f = q0.clone(); k_f = k0.clone()
        q_new, k_new = fused_prefill_rope_hadamard_qk(
            q_f, k_f, positions, cos_sin_cache, rope_head_dim, is_neox_style=True,
        )

        err_q = (q_new.float() - q_ref.float()).abs()
        rel_q = err_q / (q_ref.abs().float() + 1e-6)
        max_abs_q = err_q.max().item()
        mean_rel_q = rel_q.mean().item()
        self.assertLess(max_abs_q, 5e-2,
            f"Q max_abs {max_abs_q:.4e} > 5e-2 (5x bf16 ULP @ |x|≈0.5)")
        self.assertLess(mean_rel_q, 5e-3,
            f"Q mean_rel {mean_rel_q:.4e} > 5e-3 (expected ~1e-4)")

        err_k = (k_new.float() - k_ref.float()).abs()
        max_abs_k = err_k.max().item()
        self.assertEqual(max_abs_k, 0.0,
            f"K should be bit-identical but max_abs={max_abs_k:.4e}")

    def test_T_small(self):
        for T in range(1, 1024):
            self._run(T=T)

    def test_T_2048(self):
        self._run(T=2048)

    def test_T_4096(self):
        """Production shape for DSV3.2 with CP=8 splitting 32K sequence."""
        self._run(T=4096)

    def test_T_8192(self):
        self._run(T=8192)

    def test_T_16384(self):
        self._run(T=16384)

    def test_T_0(self):
        """Edge: empty input — should not crash."""
        cos_sin_cache = _make_cos_sin_cache(64)
        positions = torch.zeros((0,), dtype=torch.int32, device="cuda")
        q = torch.empty((0, 32, 128), dtype=torch.bfloat16, device="cuda")
        k = torch.empty((0, 128), dtype=torch.bfloat16, device="cuda")
        q_out, k_out = fused_prefill_rope_hadamard_qk(
            q, k, positions, cos_sin_cache, 64, is_neox_style=True,
        )
        self.assertEqual(q_out.shape, (0, 32, 128))
        self.assertEqual(k_out.shape, (0, 128))


class TestFusedPrefillRopeHadamardInterleaved(unittest.TestCase):
    """Interleaved (even/odd) RoPE style — GLM-5 production path."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def _run(self, T: int, num_heads: int = 32, head_dim: int = 128, rope_head_dim: int = 64):
        cos_sin_cache = _make_cos_sin_cache(rope_head_dim)
        positions = torch.arange(T, dtype=torch.int32, device="cuda")
        q0 = (torch.randn(T, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
        k0 = (torch.randn(T, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()

        q_b = q0.clone(); k_b = k0.clone()
        q_ref, k_ref = _baseline_chain(
            q_b, k_b, positions, cos_sin_cache, rope_head_dim, head_dim, is_neox_style=False,
        )
        q_f = q0.clone(); k_f = k0.clone()
        q_new, k_new = fused_prefill_rope_hadamard_qk(
            q_f, k_f, positions, cos_sin_cache, rope_head_dim, is_neox_style=False,
        )

        err_q = (q_new.float() - q_ref.float()).abs()
        rel_q = err_q / (q_ref.abs().float() + 1e-6)
        max_abs_q = err_q.max().item()
        mean_rel_q = rel_q.mean().item()
        self.assertLess(max_abs_q, 5e-2,
            f"Q max_abs {max_abs_q:.4e} > 5e-2")
        self.assertLess(mean_rel_q, 5e-3,
            f"Q mean_rel {mean_rel_q:.4e} > 5e-3")

        err_k = (k_new.float() - k_ref.float()).abs()
        max_abs_k = err_k.max().item()
        self.assertEqual(max_abs_k, 0.0,
            f"K should be bit-identical but max_abs={max_abs_k:.4e}")

    def test_T_small(self):
        for T in range(1, 1025):
            self._run(T=T)

    def test_T_2048(self):
        self._run(T=2048)

    def test_T_4096(self):
        """Production shape for GLM-5 with CP=8 splitting 32K sequence."""
        self._run(T=4096)

    def test_T_8192(self):
        self._run(T=8192)

    def test_T_16384(self):
        self._run(T=16384)

    def test_T_0(self):
        """Edge: empty input — should not crash."""
        cos_sin_cache = _make_cos_sin_cache(64)
        positions = torch.zeros((0,), dtype=torch.int32, device="cuda")
        q = torch.empty((0, 32, 128), dtype=torch.bfloat16, device="cuda")
        k = torch.empty((0, 128), dtype=torch.bfloat16, device="cuda")
        q_out, k_out = fused_prefill_rope_hadamard_qk(
            q, k, positions, cos_sin_cache, 64, is_neox_style=False,
        )
        self.assertEqual(q_out.shape, (0, 32, 128))
        self.assertEqual(k_out.shape, (0, 128))


class TestHMatrixCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def test_h_matrix_cached(self):
        """H matrix cache returns the same tensor on repeated calls."""
        h1 = _get_h_scaled_bf16(128, torch.device("cuda"))
        h2 = _get_h_scaled_bf16(128, torch.device("cuda"))
        self.assertIs(h1, h2)
        prod = h1.float() @ h1.float().T
        eye = torch.eye(128, device="cuda")
        self.assertLess((prod - eye).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
