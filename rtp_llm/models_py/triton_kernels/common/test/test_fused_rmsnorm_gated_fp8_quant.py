"""Test fused RmsNormGated + per-token-group FP8 quantization.

Compares the fused triton kernel against a 3-step reference:
  1. normed = rmsnorm(x, weight, eps)
  2. gated  = normed * z * sigmoid(z)
  3. (fp8, scale) = per_token_group_fp8_quant(gated.reshape(T, -1))

Run:
    python -m pytest rtp_llm/models_py/triton_kernels/common/test/test_fused_rmsnorm_gated_fp8_quant.py -v -s
"""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.common.fused_rmsnorm_gated_fp8_quant import (
    fused_rmsnorm_gated_fp8_quant,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated


def _ref_unfused(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_heads: int,
    quant_group_size: int,
    scale_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: RmsNormGated + reshape + fp8 quant."""
    norm = RmsNormGated(weight, eps=eps, group_size=weight.shape[0])
    normed = norm(x, gate)
    M, N = normed.shape
    T = M // num_heads
    flat = normed.reshape(T, num_heads * N).contiguous()
    fp8, scale = sgl_per_token_group_quant_fp8(
        flat,
        group_size=quant_group_size,
        eps=1e-10,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    return fp8, scale


def _dequantize(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 128,
    scale_ue8m0: bool = False,
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


class TestFusedRmsNormGatedFp8Quant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def _run(
        self,
        T: int,
        num_heads: int,
        head_v_dim: int,
        scale_ue8m0: bool,
        eps: float = 1e-6,
    ):
        device = "cuda"
        M = T * num_heads
        x = torch.randn(M, head_v_dim, dtype=torch.bfloat16, device=device)
        gate = torch.randn(M, head_v_dim, dtype=torch.bfloat16, device=device)
        weight = torch.randn(head_v_dim, dtype=torch.bfloat16, device=device)

        ref_fp8, ref_scale = _ref_unfused(
            x.clone(), gate.clone(), weight, eps, num_heads, 128, scale_ue8m0
        )

        fused_fp8, fused_scale = fused_rmsnorm_gated_fp8_quant(
            x.clone(), gate.clone(), weight, eps, num_heads, 128, scale_ue8m0
        )

        self.assertEqual(fused_fp8.shape, ref_fp8.shape)

        ref_dq = _dequantize(ref_fp8, ref_scale, 128, scale_ue8m0)
        fused_dq = _dequantize(fused_fp8, fused_scale, 128, scale_ue8m0)

        H = num_heads * head_v_dim
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
            f"T={T} heads={num_heads} hdim={head_v_dim} ue8m0={scale_ue8m0}: "
            f"{violation_pct:.3f}% > threshold",
        )

    def test_fp32_scale_hdim128(self):
        cases = [(1, 16), (8, 16), (32, 8), (256, 4)]
        for T, num_heads in cases:
            with self.subTest(T=T, num_heads=num_heads, head_v_dim=128):
                self._run(T, num_heads, 128, scale_ue8m0=False)

    def test_fp32_scale_hdim256(self):
        cases = [(1, 8), (8, 8), (32, 4)]
        for T, num_heads in cases:
            with self.subTest(T=T, num_heads=num_heads, head_v_dim=256):
                self._run(T, num_heads, 256, scale_ue8m0=False)

    def test_ue8m0_scale_hdim128(self):
        cases = [(1, 16), (8, 16), (32, 8)]
        for T, num_heads in cases:
            with self.subTest(T=T, num_heads=num_heads, head_v_dim=128):
                self._run(T, num_heads, 128, scale_ue8m0=True)

    def test_ue8m0_scale_hdim256(self):
        cases = [(1, 8), (8, 8), (32, 4)]
        for T, num_heads in cases:
            with self.subTest(T=T, num_heads=num_heads, head_v_dim=256):
                self._run(T, num_heads, 256, scale_ue8m0=True)

    def test_large_M_fallback(self):
        """M > DECODE_M_THRESHOLD (1024) forces the baseline fallback path."""
        cases = [(256, 16, 128), (128, 16, 256)]
        for T, num_heads, hdim in cases:
            for ue8m0 in [False, True]:
                with self.subTest(
                    T=T, num_heads=num_heads, head_v_dim=hdim, ue8m0=ue8m0
                ):
                    self._run(T, num_heads, hdim, scale_ue8m0=ue8m0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
