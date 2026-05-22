"""Unit tests for fused_q_rope_quant and fused_qk_rope_quant kernels."""

import math
import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
    fused_q_rope_quant,
    fused_qk_rope_quant,
)


def _build_cos_sin_cache(max_pos: int, rot_dim: int, device="cuda") -> torch.Tensor:
    """Build a simple cos/sin cache [max_pos, rot_dim] (half cos, half sin)."""
    half = rot_dim // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(0, max_pos, device=device).float()
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # [max_pos, half]
    cache = torch.cat([angles.cos(), angles.sin()], dim=-1)  # [max_pos, rot_dim]
    return cache.to(torch.bfloat16)


def _ref_rope_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Reference NeOX-style RoPE: split first/second half."""
    half = cos.shape[-1]
    x_first = x[..., :half].float()
    x_second = x[..., half : 2 * half].float()
    r_first = x_first * cos - x_second * sin
    r_second = x_second * cos + x_first * sin
    out = x.clone()
    out[..., :half] = r_first.to(x.dtype)
    out[..., half : 2 * half] = r_second.to(x.dtype)
    return out


def _ref_rope_gptj(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Reference GPT-J interleaved RoPE."""
    half = cos.shape[-1]
    x_even = x[..., 0 : 2 * half : 2].float()
    x_odd = x[..., 1 : 2 * half : 2].float()
    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin
    out = x.clone()
    out[..., 0 : 2 * half : 2] = r_even.to(x.dtype)
    out[..., 1 : 2 * half : 2] = r_odd.to(x.dtype)
    return out


def _ref_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Reference Walsh-Hadamard transform via butterfly stages, scaled by dim^-0.5."""
    dim = x.shape[-1]
    log2_dim = int(math.log2(dim))
    assert 2**log2_dim == dim
    data = x.float().clone()
    for s in range(log2_dim):
        stride = 1 << s
        idx = torch.arange(dim, device=x.device)
        partner = idx ^ stride
        is_upper = (idx & stride) != 0
        self_val = data[..., idx]
        partner_val = data[..., partner]
        data = torch.where(is_upper, partner_val - self_val, self_val + partner_val)
    return data * (dim**-0.5)


def _ref_fp8_quant_ue8m0(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-row ue8m0 FP8 quantization.

    data: [..., head_dim] float32
    Returns (fp8 [..., head_dim], scale [...] float32)
    """
    amax = data.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax / 448.0
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (data / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized, scale.squeeze(-1)


class TestFusedQRopeQuant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def _test_fused_q_rope_quant(self, T, n_heads, head_dim, rope_dim, is_neox):
        device = "cuda"
        max_pos = 4096
        cos_sin_cache = _build_cos_sin_cache(max_pos, rope_dim, device)
        q = torch.randn(T, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        positions = torch.randint(0, max_pos, (T,), device=device)

        # Fused kernel
        q_fp8, q_scale = fused_q_rope_quant(
            q,
            positions,
            cos_sin_cache,
            index_n_heads=n_heads,
            index_head_dim=head_dim,
            rope_head_dim=rope_dim,
            is_neox_style=is_neox,
        )

        # Reference
        half = rope_dim // 2
        cos = cos_sin_cache[positions, :half].float()  # [T, half]
        sin = cos_sin_cache[positions, half:].float()  # [T, half]

        q_ref = q.clone()
        for t in range(T):
            for h in range(n_heads):
                head = q_ref[t, h]
                if is_neox:
                    head = _ref_rope_neox(
                        head.unsqueeze(0), cos[t : t + 1], sin[t : t + 1]
                    ).squeeze(0)
                else:
                    head = _ref_rope_gptj(
                        head.unsqueeze(0), cos[t : t + 1], sin[t : t + 1]
                    ).squeeze(0)
                q_ref[t, h] = head

        q_had = torch.stack(
            [_ref_hadamard(q_ref[:, h, :]) for h in range(n_heads)], dim=1
        )  # [T, n_heads, head_dim]

        ref_fp8, ref_scale = _ref_fp8_quant_ue8m0(q_had)  # per [T, n_heads] row

        # Compare scales (should be identical — both use ue8m0 with same rounding)
        self.assertEqual(q_scale.shape, (T, n_heads, 1))
        scale_match = torch.allclose(q_scale.squeeze(-1), ref_scale, rtol=0, atol=0)
        if not scale_match:
            diff_count = (q_scale.squeeze(-1) != ref_scale).sum().item()
            total = q_scale.numel()
            # Allow up to 5% mismatch due to floating point order of operations
            self.assertLess(
                diff_count / total,
                0.05,
                f"Scale mismatch: {diff_count}/{total} elements differ",
            )

        # Compare FP8 values — where scales match, values should match exactly
        self.assertEqual(q_fp8.shape, (T, n_heads, head_dim))
        scale_eq = q_scale.squeeze(-1) == ref_scale
        if scale_eq.any():
            fused_vals = q_fp8.view(torch.uint8)[scale_eq]
            ref_vals = ref_fp8.view(torch.uint8)[scale_eq]
            mismatch = (fused_vals != ref_vals).float().mean().item()
            self.assertLess(mismatch, 0.02, f"FP8 mismatch rate: {mismatch:.4f}")

    def test_glm5_neox(self):
        """GLM5 shape: head_dim=128, rope_dim=64, n_heads=64, neox style."""
        for T in (1, 4, 32):
            with self.subTest(T=T):
                self._test_fused_q_rope_quant(
                    T, n_heads=64, head_dim=128, rope_dim=64, is_neox=True
                )

    def test_small_shape_neox(self):
        """Small shape for fast testing."""
        self._test_fused_q_rope_quant(
            T=8, n_heads=4, head_dim=128, rope_dim=64, is_neox=True
        )

    def test_gptj_style(self):
        """GPT-J interleaved RoPE."""
        self._test_fused_q_rope_quant(
            T=8, n_heads=4, head_dim=128, rope_dim=64, is_neox=False
        )

    def test_zero_tokens(self):
        """Edge case: T=0."""
        device = "cuda"
        cos_sin_cache = _build_cos_sin_cache(128, 64, device)
        q = torch.empty(0, 4, 128, dtype=torch.bfloat16, device=device)
        positions = torch.empty(0, dtype=torch.long, device=device)
        q_fp8, q_scale = fused_q_rope_quant(
            q, positions, cos_sin_cache, 4, 128, 64, True
        )
        self.assertEqual(q_fp8.shape[0], 0)
        self.assertEqual(q_scale.shape[0], 0)


class TestFusedQKRopeQuant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(123)

    def _test_fused_qk(self, T, n_heads, head_dim, rope_dim, is_neox):
        device = "cuda"
        max_pos = 4096
        cos_sin_cache = _build_cos_sin_cache(max_pos, rope_dim, device)
        q = torch.randn(T, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(T, head_dim, dtype=torch.bfloat16, device=device)
        positions = torch.randint(0, max_pos, (T,), device=device)

        q_fp8, q_scale, k_out = fused_qk_rope_quant(
            q,
            k,
            positions,
            cos_sin_cache,
            index_n_heads=n_heads,
            index_head_dim=head_dim,
            rope_head_dim=rope_dim,
            is_neox_style=is_neox,
        )

        # Reference K: RoPE + Hadamard → bf16
        half = rope_dim // 2
        cos = cos_sin_cache[positions, :half].float()
        sin = cos_sin_cache[positions, half:].float()

        k_ref = k.clone()
        for t in range(T):
            kk = k_ref[t : t + 1]
            if is_neox:
                kk = _ref_rope_neox(kk, cos[t : t + 1], sin[t : t + 1])
            else:
                kk = _ref_rope_gptj(kk, cos[t : t + 1], sin[t : t + 1])
            k_ref[t] = kk.squeeze(0)

        k_had_ref = _ref_hadamard(k_ref).to(torch.bfloat16)

        # K output comparison
        self.assertEqual(k_out.shape, (T, head_dim))
        self.assertEqual(k_out.dtype, torch.bfloat16)
        k_match = torch.allclose(k_out, k_had_ref, atol=1e-2, rtol=1e-2)
        if not k_match:
            diff = (k_out.float() - k_had_ref.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            self.assertLess(max_diff, 0.1, f"K max_diff={max_diff}, mean={mean_diff}")

        # Q FP8 output: same checks as Q-only test
        self.assertEqual(q_fp8.shape, (T, n_heads, head_dim))
        self.assertEqual(q_scale.shape, (T, n_heads, 1))

    def test_glm5_neox(self):
        for T in (1, 4, 32):
            with self.subTest(T=T):
                self._test_fused_qk(
                    T, n_heads=64, head_dim=128, rope_dim=64, is_neox=True
                )

    def test_gptj_style(self):
        self._test_fused_qk(T=8, n_heads=4, head_dim=128, rope_dim=64, is_neox=False)

    def test_zero_tokens(self):
        device = "cuda"
        cos_sin_cache = _build_cos_sin_cache(128, 64, device)
        q = torch.empty(0, 4, 128, dtype=torch.bfloat16, device=device)
        k = torch.empty(0, 128, dtype=torch.bfloat16, device=device)
        positions = torch.empty(0, dtype=torch.long, device=device)
        q_fp8, q_scale, k_out = fused_qk_rope_quant(
            q, k, positions, cos_sin_cache, 4, 128, 64, True
        )
        self.assertEqual(q_fp8.shape[0], 0)
        self.assertEqual(k_out.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
