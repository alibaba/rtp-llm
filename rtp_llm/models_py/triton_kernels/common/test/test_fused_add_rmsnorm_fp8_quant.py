"""Test fused add-residual + RMSNorm + per-token-group FP8 quantization.

Compares the fused triton kernel against a 3-step reference:
  1. residual += hidden_states
  2. normed = rmsnorm(residual, weight, eps)
  3. (fp8, scale) = sgl_per_token_group_quant_fp8(normed)

Run:
    python -m pytest rtp_llm/models_py/triton_kernels/common/test/test_fused_add_rmsnorm_fp8_quant.py -v -s
"""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
    fused_add_rmsnorm_fp8_quant,
    fused_add_rmsnorm_fp8_quant_with_bf16_output,
)


def _ref_unfused(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    scale_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference 3-step implementation.

    Returns (fp8, scale, residual_updated).
    """
    residual_new = residual + hidden
    # RMSNorm in fp32
    r_fp32 = residual_new.float()
    variance = r_fp32.pow(2).mean(dim=-1, keepdim=True)
    normed = r_fp32 * torch.rsqrt(variance + eps)
    normed = (normed * weight.float()).to(hidden.dtype)
    # per-token-group fp8 quant
    fp8, scale = sgl_per_token_group_quant_fp8(
        normed.contiguous(),
        group_size=group_size,
        eps=1e-10,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    return fp8, scale, residual_new.to(hidden.dtype)


def _dequantize(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 128,
    scale_ue8m0: bool = True,
) -> torch.Tensor:
    T, H = fp8.shape
    n_groups = H // group_size
    scales_f = torch.empty((T, n_groups), dtype=torch.float32, device=fp8.device)
    if scale_ue8m0:
        scale_int = scale.to(torch.int32)
        for g in range(n_groups):
            packed_idx = g // 4
            byte_idx = g % 4
            shift = byte_idx * 8
            exp_byte = (scale_int[:, packed_idx] >> shift) & 0xFF
            f32_bits = (exp_byte << 23).to(torch.int32)
            scales_f[:, g] = f32_bits.view(torch.float32)
    else:
        scales_f = scale.float()
    scales_expanded = (
        scales_f.unsqueeze(-1).expand(T, n_groups, group_size).reshape(T, H)
    )
    return fp8.to(torch.float32) * scales_expanded


class TestFusedAddRmsNormFp8Quant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def _run(self, T: int, H: int, scale_ue8m0: bool, eps: float = 1e-6):
        device = "cuda"
        hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device)
        residual_fused = torch.randn(T, H, dtype=torch.bfloat16, device=device)
        residual_ref = residual_fused.clone()
        weight = torch.randn(H, dtype=torch.bfloat16, device=device)

        # Reference
        ref_fp8, ref_scale, ref_residual = _ref_unfused(
            hidden.clone(), residual_ref, weight, eps, 128, scale_ue8m0
        )

        # Fused
        fused_fp8, fused_scale = fused_add_rmsnorm_fp8_quant(
            hidden.clone(), residual_fused, weight, eps, 128, scale_ue8m0
        )

        # Check residual updated correctly
        res_diff = (residual_fused.float() - ref_residual.float()).abs().max().item()
        self.assertLess(res_diff, 1e-3, f"residual mismatch: {res_diff}")

        # Dequantize and compare
        ref_dq = _dequantize(ref_fp8, ref_scale, 128, scale_ue8m0)
        fused_dq = _dequantize(fused_fp8, fused_scale, 128, scale_ue8m0)

        per_group_absmax = (
            torch.maximum(ref_dq.abs(), fused_dq.abs())
            .reshape(T, H // 128, 128)
            .amax(dim=-1, keepdim=True)
            .expand(T, H // 128, 128)
            .reshape(T, H)
        )
        ulp_bound = 4.0 * per_group_absmax / 127.0
        delta = (fused_dq - ref_dq).abs()
        violations = (delta > ulp_bound + 1e-6).float()
        violation_pct = violations.mean().item() * 100.0
        # Small tensors (few groups) have higher relative violation rates
        # because a single scale-boundary case affects a larger % of elements.
        max_pct = 1.0
        self.assertLess(
            violation_pct,
            max_pct,
            f"T={T} H={H} ue8m0={scale_ue8m0}: {violation_pct:.3f}% > 2 ULP",
        )

    def test_fp32_scale(self):
        cases = [(1, 256), (1, 384), (8, 384), (32, 768), (256, 1024)]
        for T, H in cases:
            with self.subTest(T=T, H=H, scale_ue8m0=False):
                self._run(T, H, scale_ue8m0=False)

    def test_ue8m0_scale(self):
        cases = [(1, 512), (8, 512), (32, 1024), (256, 2048)]
        for T, H in cases:
            with self.subTest(T=T, H=H, scale_ue8m0=True):
                self._run(T, H, scale_ue8m0=True)

    def test_non_pow2_H(self):
        """Non-power-of-2 H (GLM5=6144, DSV3=7168) uses singlepass with padding."""
        cases = [(1, 3072), (1, 5120), (4, 6144), (32, 6144), (1, 7168)]
        for T, H in cases:
            for ue8m0 in [False, True]:
                with self.subTest(T=T, H=H, scale_ue8m0=ue8m0):
                    self._run(T, H, scale_ue8m0=ue8m0)

    def test_baseline_fallback_large_H(self):
        """H > MAX_INREG_H forces baseline fallback path."""
        cases = [(1, 8192), (4, 8192)]
        for T, H in cases:
            for ue8m0 in [False, True]:
                with self.subTest(T=T, H=H, scale_ue8m0=ue8m0):
                    self._run(T, H, scale_ue8m0=ue8m0)


class TestFusedAddRmsNormFp8QuantDualOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def _run(self, T: int, H: int, scale_ue8m0: bool, eps: float = 1e-6):
        device = "cuda"
        hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device)
        residual_fused = torch.randn(T, H, dtype=torch.bfloat16, device=device)
        residual_ref = residual_fused.clone()
        weight = torch.randn(H, dtype=torch.bfloat16, device=device)

        # Reference
        ref_fp8, ref_scale, ref_residual = _ref_unfused(
            hidden.clone(), residual_ref, weight, eps, 128, scale_ue8m0
        )
        # Reference bf16 normed
        r_fp32 = ref_residual.float()
        variance = r_fp32.pow(2).mean(dim=-1, keepdim=True)
        ref_bf16_normed = (r_fp32 * torch.rsqrt(variance + eps) * weight.float()).to(
            torch.bfloat16
        )

        # Fused dual-output
        fused_bf16, fused_fp8, fused_scale = (
            fused_add_rmsnorm_fp8_quant_with_bf16_output(
                hidden.clone(), residual_fused, weight, eps, 128, scale_ue8m0
            )
        )

        # Check residual updated correctly
        res_diff = (residual_fused.float() - ref_residual.float()).abs().max().item()
        self.assertLess(res_diff, 1e-3, f"residual mismatch: {res_diff}")

        # Check bf16 normed output (relative tolerance — bf16 has limited precision)
        ref_abs = ref_bf16_normed.float().abs().clamp(min=1e-6)
        bf16_rel = (
            ((fused_bf16.float() - ref_bf16_normed.float()).abs() / ref_abs)
            .max()
            .item()
        )
        self.assertLess(bf16_rel, 1e-2, f"bf16 normed relative mismatch: {bf16_rel}")

        # Dequantize and compare fp8 output
        ref_dq = _dequantize(ref_fp8, ref_scale, 128, scale_ue8m0)
        fused_dq = _dequantize(fused_fp8, fused_scale, 128, scale_ue8m0)

        per_group_absmax = (
            torch.maximum(ref_dq.abs(), fused_dq.abs())
            .reshape(T, H // 128, 128)
            .amax(dim=-1, keepdim=True)
            .expand(T, H // 128, 128)
            .reshape(T, H)
        )
        ulp_bound = 4.0 * per_group_absmax / 127.0
        delta = (fused_dq - ref_dq).abs()
        violations = (delta > ulp_bound + 1e-6).float()
        violation_pct = violations.mean().item() * 100.0
        max_pct = 1.0
        self.assertLess(
            violation_pct,
            max_pct,
            f"T={T} H={H} ue8m0={scale_ue8m0}: {violation_pct:.3f}% > 2 ULP",
        )

    def test_fp32_scale(self):
        cases = [(1, 256), (1, 384), (8, 384), (32, 768), (256, 1024)]
        for T, H in cases:
            with self.subTest(T=T, H=H, scale_ue8m0=False):
                self._run(T, H, scale_ue8m0=False)

    def test_ue8m0_scale(self):
        cases = [(1, 512), (8, 512), (32, 1024), (256, 2048)]
        for T, H in cases:
            with self.subTest(T=T, H=H, scale_ue8m0=True):
                self._run(T, H, scale_ue8m0=True)

    def test_non_pow2_H(self):
        """Non-power-of-2 H (GLM5=6144, DSV3=7168) uses singlepass with padding."""
        cases = [(1, 3072), (4, 6144), (32, 6144), (1, 7168)]
        for T, H in cases:
            for ue8m0 in [False, True]:
                with self.subTest(T=T, H=H, scale_ue8m0=ue8m0):
                    self._run(T, H, scale_ue8m0=ue8m0)

    def test_baseline_fallback_large_H(self):
        """H > MAX_INREG_H forces baseline fallback path."""
        cases = [(1, 8192), (4, 8192)]
        for T, H in cases:
            for ue8m0 in [False, True]:
                with self.subTest(T=T, H=H, scale_ue8m0=ue8m0):
                    self._run(T, H, scale_ue8m0=ue8m0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
