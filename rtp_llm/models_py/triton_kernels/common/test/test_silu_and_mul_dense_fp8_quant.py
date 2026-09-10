"""Unit tests for the dense SiLU-and-mul + per-token-group fp8 quant kernel.

Verifies that the fused triton kernel produces results bitwise compatible
with the unfused reference path (silu_and_mul → sgl_per_token_group_quant_fp8).
The reference uses the same UE8M0 column-major TMA-aligned scale layout that
deepgemm consumes — so the comparison is meaningful for end-to-end use.

Run:
    python -m pytest rtp_llm/models_py/triton_kernels/common/test/test_silu_and_mul_dense_fp8_quant.py -v -s
"""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul_mxfp8_quant_tiled_fwd,
    silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
)

# MiniMax-M3 SwiGLU-OAI params.
_M3_ALPHA = 1.702
_M3_LIMIT = 7.0
_MX_GROUP = 32


def _ref_unfused(
    input_2d: torch.Tensor, group_size: int = 128, scale_ue8m0: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: silu computed in fp32 (matching the fused kernel's path),
    cast to dtype, multiplied by up in dtype, then sgl per-token-group fp8
    quantization with the matching scale layout (UE8M0 packed or fp32).

    Layout: [gate | up] — first half is gate (silu applied), second half is up.
    This matches the C++ silu_and_mul convention and the fused triton kernel.
    """
    H = input_2d.shape[-1] // 2
    gate = input_2d[..., :H]
    up = input_2d[..., H:]
    gate_fp32 = gate.float()
    silu_fp32 = gate_fp32 / (1.0 + torch.exp(-gate_fp32))
    silu_bf = silu_fp32.to(input_2d.dtype)
    activated = (up * silu_bf).contiguous()
    fp8, scale = sgl_per_token_group_quant_fp8(
        activated,
        group_size=group_size,
        eps=1e-10,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    return fp8, scale


def _ref_swiglu_oai_mxfp8_tiled(
    x: torch.Tensor,
    alpha: float,
    limit: float,
    group: int = _MX_GROUP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch reference matching silu_and_mul_mxfp8_quant_tiled_fwd.

    Layout [gate | up]. SwiGLU-OAI: gate*sigmoid(gate*alpha)*(up+1) with a
    one-sided gate clamp (upper only) and two-sided up clamp, then a bf16
    round-trip (production activation dtype), then MXFP8 1x32 quant with a
    power-of-two fp32 scale (row-major [T, H//group]). Mirrors the kernel step
    for step so differences reduce to per-group pow2-boundary rounding.
    """
    T = x.shape[0]
    H = x.shape[-1] // 2
    gate = x[..., :H].float()
    up = x[..., H:].float()
    gate = torch.minimum(gate, torch.tensor(limit, device=x.device))
    up = up.clamp(-limit, limit)
    activated = gate * torch.sigmoid(gate * alpha) * (up + 1.0)
    # bf16 round-trip to match the kernel's precision.
    activated = activated.to(torch.bfloat16).to(torch.float32)

    ng = H // group
    act_g = activated.view(T, ng, group)
    absmax = act_g.abs().amax(dim=-1).clamp_min(1e-20)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    fp8_min = -fp8_max
    s0 = absmax / fp8_max
    output_s = torch.exp2(torch.ceil(torch.log2(s0.abs())))  # [T, ng]
    q = (act_g / output_s.unsqueeze(-1)).clamp(fp8_min, fp8_max)
    fp8 = q.to(torch.float8_e4m3fn).view(T, H)
    return fp8, output_s


def _dequant_mxfp8_fp32(
    fp8: torch.Tensor, scale_fp32: torch.Tensor, group: int = _MX_GROUP
) -> torch.Tensor:
    """Dequantize (e4m3 [T, H], fp32 scale [T, H//group]) → fp32."""
    T, H = fp8.shape
    ng = H // group
    return (
        fp8.to(torch.float32).view(T, ng, group) * scale_fp32.float().unsqueeze(-1)
    ).view(T, H)


class TestSiluAndMulDenseFp8Quant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(0)

    @staticmethod
    def _dequantize(
        fp8: torch.Tensor,
        scale: torch.Tensor,
        group_size: int = 128,
        scale_ue8m0: bool = True,
    ) -> torch.Tensor:
        """Dequantize fp8 + per-group scale back to fp32.

        UE8M0 path: scale is int32 packed (4 groups per int32, byte 0 = group 0).
        fp32 path:  scale is float32 unpacked, one entry per group.
        Both layouts are column-major in K, .stride() reflects that.
        """
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
        scales_f_expanded = (
            scales_f.unsqueeze(-1).expand(T, n_groups, group_size).reshape(T, H)
        )
        return fp8.to(torch.float32) * scales_f_expanded

    def _run_case(self, T: int, H: int, dtype=torch.bfloat16, scale_ue8m0: bool = True):
        # input layout is [up | gate], shape (T, 2H)
        x = (torch.randn(T, 2 * H, device="cuda") * 1.5).to(dtype).contiguous()

        # Reference: matched silu order (silu in fp32 → cast bf16 → mul up)
        # then sgl per-token-group fp8 quant with the matching scale layout.
        ref_fp8, ref_scale = _ref_unfused(
            x.clone(), group_size=128, scale_ue8m0=scale_ue8m0
        )

        # Fused path
        fp8_out, scale_out = silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
            x,
            quant_group_size=128,
            scale_ue8m0=scale_ue8m0,
        )

        ref_dq = self._dequantize(ref_fp8, ref_scale, scale_ue8m0=scale_ue8m0)
        fused_dq = self._dequantize(fp8_out, scale_out, scale_ue8m0=scale_ue8m0)

        # The two pipelines compute identical bf16 `up * silu_bf16` values
        # (we matched the silu order in `_ref_unfused`). They can disagree
        # only on per-group UE8M0 absmax rounding when absmax sits exactly
        # on a power-of-2 boundary — affecting at most a tiny minority of
        # groups. Per element, the disagreement is bounded by 1 fp8 ULP at
        # the per-group scale (absmax_pow2 / 127, where absmax_pow2 is at
        # most 2x the natural absmax).
        per_group_absmax = (
            torch.maximum(ref_dq.abs(), fused_dq.abs())
            .reshape(T, H // 128, 128)
            .amax(dim=-1, keepdim=True)
            .expand(T, H // 128, 128)
            .reshape(T, H)
        )
        # Allow 2 fp8 ULPs per element (= 4 * absmax / 127), generous to
        # cover both paths picking different powers of 2.
        ulp_bound = 4.0 * per_group_absmax / 127.0
        delta = (fused_dq - ref_dq).abs()
        # Most elements should be bitwise equal; allow up to 0.5% to differ
        # by up to 2 ULPs (scale-boundary cases).
        violations = (delta > ulp_bound + 1e-6).float()
        violation_pct = violations.mean().item() * 100.0
        max_delta = delta.max().item()
        self.assertLess(
            violation_pct,
            0.5,
            f"T={T} H={H}: {violation_pct:.3f}% > 2 ULP, max_delta={max_delta}",
        )

    def test_grid_ue8m0(self):
        # H must be divisible by 128*4 = 512 for the UE8M0 packing to work.
        cases = [
            (1, 512),
            (8, 512),
            (32, 1024),
            (256, 2048),
            (1024, 1024),
            (2048, 2048),
        ]
        for T, H in cases:
            with self.subTest(T=T, H=H, scale_ue8m0=True):
                self._run_case(T, H, torch.bfloat16, scale_ue8m0=True)

    def test_grid_fp32_scale(self):
        # H only needs to be divisible by 128 for the fp32-scale path.
        cases = [
            (1, 256),
            (1, 384),
            (8, 384),
            (32, 768),
            (256, 1024),
            (1024, 1024),
        ]
        for T, H in cases:
            with self.subTest(T=T, H=H, scale_ue8m0=False):
                self._run_case(T, H, torch.bfloat16, scale_ue8m0=False)

    def _run_tiled_case(self, T: int, H: int, dtype=torch.bfloat16):
        """Verify silu_and_mul_mxfp8_quant_tiled_fwd against a torch reference."""
        # input layout is [gate | up], shape (T, 2H)
        x = (torch.randn(T, 2 * H, device="cuda") * 1.5).to(dtype).contiguous()

        fp8_out, scale_out = silu_and_mul_mxfp8_quant_tiled_fwd(
            x,
            quant_group_size=_MX_GROUP,
            gemm1_alpha=_M3_ALPHA,
            gemm1_clamp_limit=_M3_LIMIT,
        )
        # Contract: fp8 [T, H] e4m3 + fp32 row-major scale [T, H//32].
        self.assertEqual(tuple(fp8_out.shape), (T, H))
        self.assertEqual(fp8_out.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(scale_out.shape), (T, H // _MX_GROUP))
        self.assertEqual(scale_out.dtype, torch.float32)

        ref_fp8, ref_scale = _ref_swiglu_oai_mxfp8_tiled(
            x.clone(), _M3_ALPHA, _M3_LIMIT
        )

        ref_dq = _dequant_mxfp8_fp32(ref_fp8, ref_scale)
        out_dq = _dequant_mxfp8_fp32(fp8_out, scale_out)

        # Kernel and reference share identical math; disagreement is bounded by
        # per-group UE8M0 absmax rounding at pow2 boundaries. Allow up to 2 fp8
        # ULPs (= 4 * absmax / 127) on a tiny minority of elements.
        per_group_absmax = (
            torch.maximum(ref_dq.abs(), out_dq.abs())
            .reshape(T, H // _MX_GROUP, _MX_GROUP)
            .amax(dim=-1, keepdim=True)
            .expand(T, H // _MX_GROUP, _MX_GROUP)
            .reshape(T, H)
        )
        ulp_bound = 4.0 * per_group_absmax / 127.0
        delta = (out_dq - ref_dq).abs()
        violations = (delta > ulp_bound + 1e-6).float()
        violation_pct = violations.mean().item() * 100.0
        max_delta = delta.max().item()
        self.assertLess(
            violation_pct,
            0.5,
            f"T={T} H={H}: {violation_pct:.3f}% > 2 ULP, max_delta={max_delta}",
        )

    def test_tiled_mxfp8_swiglu_oai(self):
        # H must be divisible by 32 (MX block). Cover the T >= 1024 prefill
        # regime that the production dispatch routes to the tiled kernel, plus a
        # couple of small-T shapes to exercise the kernel independent of the
        # dispatch threshold. Shapes kept modest to fit a shared card.
        cases = [
            (64, 256),
            (1024, 256),
            (1024, 768),
            (1024, 3072),   # MiniMax-M3 shared expert H_out
            (2048, 512),
            (4096, 3072),
        ]
        for T, H in cases:
            with self.subTest(T=T, H=H):
                self._run_tiled_case(T, H, torch.bfloat16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
