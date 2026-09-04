"""Precision tests for fused_strided_rmsnorm (F1a/F1b/F2).

The Triton fast path runs for all shapes that satisfy basic correctness
constraints (H<=8192, last-dim contig, H%group_size==0); no workload gate.
"""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (
    mxfp8_quant_act_packed_fused,
)
from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
    fused_strided_rmsnorm,
    fused_strided_rmsnorm_per_token_fp8_quant,
    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
)


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.nn.functional.rms_norm(
        x.contiguous().float(), [x.shape[-1]], weight.float(), eps
    ).to(torch.bfloat16)


def _ue8m0_unpack(scale_int32: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Unpack UE8M0 packed int32 scales to fp32 scales."""
    T, num_packed = scale_int32.shape
    out = torch.empty(T, num_groups, dtype=torch.float32, device=scale_int32.device)
    for g in range(num_groups):
        packed_idx = g // 4
        byte_idx = g % 4
        shift = byte_idx * 8
        exp_byte = (scale_int32[:, packed_idx].to(torch.int32) >> shift) & 0xFF
        f32_bits = (exp_byte << 23).to(torch.int32)
        out[:, g] = f32_bits.view(torch.float32)
    return out


def _dequantize(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    scale_ue8m0: bool,
) -> torch.Tensor:
    T, H = fp8.shape
    n_groups = H // group_size
    if scale_ue8m0:
        scales_f = _ue8m0_unpack(scale, n_groups)
    else:
        scales_f = scale.float()
    scales_expanded = (
        scales_f.unsqueeze(-1).expand(T, n_groups, group_size).reshape(T, H)
    )
    return fp8.float() * scales_expanded


class TestFusedStridedRMSNorm(unittest.TestCase):
    """F1a / F2 — bf16 output strided RMSNorm."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def _run(self, T: int, total_dim: int, slice_offset: int, slice_dim: int):
        device = "cuda"
        # Build a strided slice from a larger contiguous tensor (mimics
        # torch.split output in mla_attention).
        big = torch.randn(T, total_dim, dtype=torch.bfloat16, device=device)
        x = big[:, slice_offset : slice_offset + slice_dim]
        weight = torch.randn(slice_dim, dtype=torch.bfloat16, device=device)
        ref = _ref_rmsnorm(x, weight, 1e-6)
        out = fused_strided_rmsnorm(x, weight, 1e-6)
        max_abs = (out.float() - ref.float()).abs().max().item()
        self.assertLess(max_abs, 5e-3, f"T={T} H={slice_dim}: max_abs={max_abs}")
        # Also verify bit-exact when input is contiguous (no rounding diff)
        if slice_offset == 0 and slice_dim == total_dim:
            self.assertTrue(torch.equal(out, ref), "Should be bit-exact when contig")

    def test_cm_kv_path(self):
        """F2: compressed_kv slice, H=512."""
        for T in (1, 8, 32, 256):
            with self.subTest(T=T):
                self._run(
                    T, total_dim=1536 + 512 + 64, slice_offset=1536, slice_dim=512
                )

    def test_q_lora_path_bf16(self):
        """F1a: q_lora slice (when q_b_proj is bf16), H=1536."""
        for T in (1, 8, 32, 256):
            with self.subTest(T=T):
                self._run(T, total_dim=1536 + 512 + 64, slice_offset=0, slice_dim=1536)

    def test_non_pow2_h(self):
        """Non power-of-2 H still works via masking."""
        for T, H in ((1, 384), (8, 768), (32, 1536)):
            with self.subTest(T=T, H=H):
                self._run(T, total_dim=H + 100, slice_offset=20, slice_dim=H)


class TestFusedStridedRMSNormFp8Quant(unittest.TestCase):
    """F1b — fp8 quant single-output (no bf16)."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def _run(self, T: int, H: int, slice_offset: int, scale_ue8m0: bool):
        device = "cuda"
        big = torch.randn(T, H + slice_offset + 80, dtype=torch.bfloat16, device=device)
        x = big[:, slice_offset : slice_offset + H]
        weight = torch.randn(H, dtype=torch.bfloat16, device=device)
        # Reference: bf16 RMSNorm then sgl per-token-group fp8 quant
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        normed_ref = _ref_rmsnorm(x, weight, 1e-6)
        ref_fp8, ref_scale = sgl_per_token_group_quant_fp8(
            normed_ref,
            group_size=128,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
        )
        fused_fp8, fused_scale = fused_strided_rmsnorm_per_token_fp8_quant(
            x, weight, 1e-6, 128, scale_ue8m0
        )
        ref_dq = _dequantize(ref_fp8, ref_scale, 128, scale_ue8m0)
        fused_dq = _dequantize(fused_fp8, fused_scale, 128, scale_ue8m0)
        per_grp = (
            torch.maximum(ref_dq.abs(), fused_dq.abs())
            .reshape(T, H // 128, 128)
            .amax(-1, keepdim=True)
            .expand(T, H // 128, 128)
            .reshape(T, H)
        )
        delta = (fused_dq - ref_dq).abs()
        viol = (delta > 4.0 * per_grp / 127.0 + 1e-6).float().mean().item() * 100
        self.assertLess(viol, 1.0, f"T={T} H={H} ue8m0={scale_ue8m0}: viol={viol:.3f}%")

    def test_q_lora_fp8_fp32_scale(self):
        """F1b: q_lora H=1536 with fp32 scale (12 groups)."""
        for T in (1, 8, 32, 256):
            with self.subTest(T=T):
                self._run(T, H=1536, slice_offset=0, scale_ue8m0=False)

    def test_q_lora_fp8_ue8m0_scale(self):
        """F1b: q_lora H=1536 with UE8M0 scale (12 groups, % 4 == 0)."""
        for T in (1, 8, 32, 256):
            with self.subTest(T=T):
                self._run(T, H=1536, slice_offset=0, scale_ue8m0=True)


class TestFusedStridedRMSNormFp8QuantDualOutput(unittest.TestCase):
    """F1b — bf16 + fp8 dual output."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def test_dual_output(self):
        device = "cuda"
        for T, H, ue8 in [
            (1, 1536, False),
            (8, 1536, False),
            (32, 1536, False),
            (8, 1536, True),
            (32, 1536, True),
        ]:
            with self.subTest(T=T, H=H, ue8m0=ue8):
                big = torch.randn(T, H + 600, dtype=torch.bfloat16, device=device)
                x = big[:, :H]
                weight = torch.randn(H, dtype=torch.bfloat16, device=device)
                ref_bf16 = _ref_rmsnorm(x, weight, 1e-6)
                bf16_out, fp8_out, scale = (
                    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
                        x, weight, 1e-6, 128, ue8
                    )
                )
                bf16_diff = (bf16_out.float() - ref_bf16.float()).abs().max().item()
                self.assertLess(bf16_diff, 5e-3, f"bf16 mismatch: {bf16_diff}")


class TestFusedStridedRMSNormMxfp8(unittest.TestCase):
    """HY4 q_a RMSNorm must preserve the current MXFP8 contract exactly."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def _make_inputs(self, tokens: int, hidden_size: int):
        # Use a non-zero slice offset to exercise the production torch.split
        # layout rather than accidentally testing only a contiguous tensor.
        big = torch.randn(
            tokens,
            hidden_size + 640,
            dtype=torch.bfloat16,
            device="cuda",
        )
        x = big[:, 128 : 128 + hidden_size]
        weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
        return x, weight

    def _reference(self, x: torch.Tensor, weight: torch.Tensor):
        # The standalone MXFP8 test proves this quantizer is bitwise equal to
        # FlashInfer.  Using it here isolates the RMSNorm+quant fusion without
        # requiring FlashInfer's optional CuTe DSL package in this test target.
        normed = fused_strided_rmsnorm(x, weight, 1e-6)
        quantized, scale = mxfp8_quant_act_packed_fused(normed)
        return normed, quantized, scale

    def test_single_output_is_bitwise_equal(self):
        for tokens in (1, 4, 16):
            with self.subTest(tokens=tokens):
                x, weight = self._make_inputs(tokens, 2048)
                _, ref_fp8, ref_scale = self._reference(x, weight)
                fp8, scale = fused_strided_rmsnorm_per_token_fp8_quant(
                    x,
                    weight,
                    1e-6,
                    group_size=32,
                    scale_ue8m0=True,
                    mxfp8_semantics=True,
                )
                self.assertTrue(
                    torch.equal(fp8.view(torch.uint8), ref_fp8.view(torch.uint8))
                )
                self.assertTrue(torch.equal(scale, ref_scale))

    def test_dual_output_is_bitwise_equal(self):
        for tokens in (1, 4, 16):
            with self.subTest(tokens=tokens):
                x, weight = self._make_inputs(tokens, 2048)
                ref_bf16, ref_fp8, ref_scale = self._reference(x, weight)
                bf16, fp8, scale = (
                    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
                        x,
                        weight,
                        1e-6,
                        group_size=32,
                        scale_ue8m0=True,
                        mxfp8_semantics=True,
                    )
                )
                self.assertTrue(torch.equal(bf16, ref_bf16))
                self.assertTrue(
                    torch.equal(fp8.view(torch.uint8), ref_fp8.view(torch.uint8))
                )
                self.assertTrue(torch.equal(scale, ref_scale))


if __name__ == "__main__":
    unittest.main()
