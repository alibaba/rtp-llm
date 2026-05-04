"""Equivalence test for ``_silu_mul_fp8_quant_triton.silu_mul_fp8_quant_packed``.

Compares the fused Triton kernel against the legacy 5-step torch chain it
replaces (clamp + silu + mul + bf16-cast + sgl_per_token_group_quant_fp8).
Tolerance: bit-equivalent on FP8 quantized output (UE8M0 quantization is
deterministic given the same fp32→bf16-rounded input), tight on the packed
UE8M0 scale.

Runs on any CUDA device with Triton — no DeepGEMM dependency. Verified
locally before the SM100 smoke validation.
"""

from __future__ import annotations

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SiluMulFp8QuantPackedTest(unittest.TestCase):
    """Compare the fused kernel vs. the legacy reference implementation."""

    @classmethod
    def setUpClass(cls):
        # Direct file-path imports — bypass ``rtp_llm.models_py.modules``
        # package init (which pulls in fastapi/aiohttp/rtp_kernel/etc.).
        # The kernel under test depends only on torch + triton.
        import importlib.util
        import os

        here = os.path.dirname(os.path.abspath(__file__))
        kernel_path = os.path.join(
            here, "..", "moe", "_silu_mul_fp8_quant_triton.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_silu_mul_fp8_quant_triton", kernel_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls._fused = staticmethod(module.silu_mul_fp8_quant_packed)

    @staticmethod
    def _ref(
        gate_up: torch.Tensor,
        clamp_limit: float,
    ):
        """Pure-torch reference for fused silu+clamp+mul + per-token-group
        FP8 quant + UE8M0 packed scale.

        Mirrors the legacy 5-step chain (lines 514-534 of
        ``moe/strategies/grouped_fp4.py``) that we're replacing in Phase 2
        optimization 3, AND the layout that
        ``sgl_per_token_group_quant_fp8(..., column_major_scales=True,
        scale_tma_aligned=True, scale_ue8m0=True)`` produces (so the fused
        kernel's output layout matches DeepGEMM's contiguous-GEMM expectation).

        UE8M0 scale = ``2^ceil(log2(absmax/fp8_max))`` per (row, group),
        packed 4-per-int32 along K, then transposed to column-major
        ``[M, num_packed_groups]`` with M padded to multiple of 4.
        """
        import torch.nn.functional as F

        N = gate_up.size(-1)
        inter = N // 2
        group_size = 128
        # Step 1-3: legacy silu+clamp+mul → bf16 round
        gate = gate_up[:, :inter].float()
        up = gate_up[:, inter:].float()
        if clamp_limit > 0:
            up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
            gate = torch.clamp(gate, max=clamp_limit)
        hidden_f32 = F.silu(gate) * up
        hidden_bf16 = hidden_f32.to(torch.bfloat16)
        # Re-cast to fp32 for quantization (legacy passes bf16 to the
        # quantization kernel which then upcasts internally — same effect
        # as a single bf16 round + fp32 quant arithmetic).
        y = hidden_bf16.to(torch.float32)

        M = y.size(0)
        num_groups = inter // group_size

        # Step 4: per-(row, group) absmax → UE8M0 exponent
        finfo = torch.finfo(torch.float8_e4m3fn)
        fp8_max = float(finfo.max)
        fp8_min = float(finfo.min)
        y_grouped = y.view(M, num_groups, group_size)
        absmax = y_grouped.abs().amax(dim=-1).clamp_min(1e-10)
        scale_raw = absmax / fp8_max
        exponent = torch.ceil(torch.log2(scale_raw))
        scale = torch.exp2(exponent)
        # Step 5: quantize
        y_q = torch.clamp(y_grouped / scale.unsqueeze(-1), fp8_min, fp8_max)
        out_q = y_q.to(torch.float8_e4m3fn).view(M, inter)

        # Step 6: pack 4 UE8M0 exponents per int32, column-major TMA-aligned M
        exp_biased = (exponent + 127.0).clamp(0, 255).to(torch.int32)  # [M, num_groups]
        num_packed_groups = (num_groups + 3) // 4
        # Pad num_groups up to multiple of 4 with zero-byte placeholders
        if num_groups % 4 != 0:
            pad = 4 - (num_groups % 4)
            exp_biased = torch.cat(
                [exp_biased, torch.zeros((M, pad), dtype=torch.int32, device=exp_biased.device)],
                dim=1,
            )
        # Reshape into [M, num_packed_groups, 4] and pack via shifts
        exp_packed = exp_biased.view(M, num_packed_groups, 4)
        out_scale_rowmajor = (
            exp_packed[:, :, 0]
            | (exp_packed[:, :, 1] << 8)
            | (exp_packed[:, :, 2] << 16)
            | (exp_packed[:, :, 3] << 24)
        )  # [M, num_packed_groups] int32
        # Match the production layout: column-major TMA-aligned M
        tma_aligned_M = ((M + 3) // 4) * 4
        out_scale_colmajor = torch.zeros(
            (num_packed_groups, tma_aligned_M),
            dtype=torch.int32, device=out_scale_rowmajor.device,
        ).T[:M, :]
        out_scale_colmajor.copy_(out_scale_rowmajor)

        return out_q, out_scale_colmajor

    def _check(self, M: int, inter: int, clamp_limit: float, seed: int = 0):
        torch.manual_seed(seed)
        device = "cuda:0"
        gate_up = torch.randn(M, 2 * inter, dtype=torch.bfloat16, device=device) * 4.0

        ref_q, ref_s = self._ref(gate_up, clamp_limit)
        fused_q, fused_s = self._fused(gate_up, clamp_limit=clamp_limit, group_size=128)

        # Output shape contract
        self.assertEqual(fused_q.shape, (M, inter))
        self.assertEqual(fused_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(fused_s.dtype, torch.int32)

        # FP8 values: convert to fp32 for comparison (FP8 equality after
        # cast is the bit-identical check we want; small UE8M0 quantization
        # noise is OK but should be near-zero given the bf16 round-through).
        ref_q_f32 = ref_q.to(torch.float32)
        fused_q_f32 = fused_q.to(torch.float32)
        # Allow a few representable-FP8-step differences from independent
        # rounding paths (UE8M0 is deterministic; ULP drift here would point
        # to a kernel bug).
        max_abs = (ref_q_f32 - fused_q_f32).abs().max().item()
        self.assertLess(
            max_abs, 1e-2,
            f"FP8 output diff too large (M={M}, inter={inter}, clamp={clamp_limit}): "
            f"max_abs={max_abs}",
        )

        # Scale: should be exactly equal (UE8M0 packing is deterministic).
        # Compare element-wise on int32 — the layouts must match
        # (column-major TMA-aligned).
        self.assertEqual(fused_s.shape, ref_s.shape, "scale shape mismatch")
        scale_diff = (fused_s.cpu() != ref_s.cpu()).sum().item()
        self.assertEqual(
            scale_diff, 0,
            f"UE8M0 scale mismatch (M={M}, inter={inter}, clamp={clamp_limit}): "
            f"{scale_diff} elements differ out of {fused_s.numel()}",
        )

    # --- Realistic V4-Flash shapes -----------------------------------------

    def test_v4_flash_no_clamp_small(self):
        # 1k tokens × inter=2048 (V4 small batch decode-ish)
        self._check(M=1024, inter=2048, clamp_limit=0.0)

    def test_v4_flash_with_clamp_small(self):
        self._check(M=1024, inter=2048, clamp_limit=10.0)

    def test_v4_flash_with_clamp_medium(self):
        # 4k tokens × inter=2048 (typical prefill)
        self._check(M=4096, inter=2048, clamp_limit=10.0)

    def test_v4_flash_with_clamp_padded(self):
        # M not a multiple of 4 — exercises tma_aligned_M padding
        self._check(M=4099, inter=2048, clamp_limit=10.0)

    def test_v4_flash_with_clamp_large(self):
        # 16k tokens (max prefill)
        self._check(M=16384, inter=2048, clamp_limit=10.0)

    def test_smaller_inter(self):
        # inter=512: 4 groups of 128
        self._check(M=512, inter=512, clamp_limit=5.0)


if __name__ == "__main__":
    unittest.main()
