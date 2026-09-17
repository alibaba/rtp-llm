"""Equivalence test for fused ``attn * sigmoid(gate)`` + UE8M0 FP8 quant.

Compares ``sigmoid_mul_fp8_quant`` against the old three-op path::

    y = attn * torch.sigmoid(gate)
    fp8, scale = sgl_per_token_group_quant_fp8(
        y, 128, eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )

FP8 must stay within 1 ULP; packed UE8M0 scale layout must match.
"""

from __future__ import annotations

import os
import unittest

import torch


def _fp8_ulp_diff(ref: torch.Tensor, got: torch.Tensor) -> int:
    ref_u = ref.view(torch.uint8).to(torch.int32)
    got_u = got.view(torch.uint8).to(torch.int32)
    if ref_u.numel() == 0:
        return 0
    return int((ref_u - got_u).abs().max().item())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SigmoidMulFp8QuantTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"

    @classmethod
    def setUpClass(cls):
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.sigmoid_mul_fp8_quant import (
            is_supported,
            maybe_sigmoid_mul_fp8_quant,
            sigmoid_mul_fp8_quant,
        )

        cls._fused = staticmethod(sigmoid_mul_fp8_quant)
        cls._maybe = staticmethod(maybe_sigmoid_mul_fp8_quant)
        cls._is_supported = staticmethod(is_supported)
        cls._sgl = None
        try:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
                sgl_per_token_group_quant_fp8,
            )

            cls._sgl = staticmethod(sgl_per_token_group_quant_fp8)
        except Exception:
            cls._sgl = None

    @staticmethod
    def _torch_ue8m0_quant(y: torch.Tensor):
        """sgl_per_token_group_quant_fp8 UE8M0 layout, torch-only fallback."""
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fp8_scale import (
            make_ue8m0_scale_like,
        )

        group_size = 128
        m, n = y.shape
        y_f = y.to(torch.float32)
        num_groups = n // group_size
        finfo = torch.finfo(torch.float8_e4m3fn)
        if m == 0:
            out_q = torch.empty((m, n), device=y.device, dtype=torch.float8_e4m3fn)
            return out_q, make_ue8m0_scale_like(y.shape, device=y.device)
        grouped = y_f.view(m, num_groups, group_size)
        absmax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
        exponent = torch.ceil(torch.log2(absmax / float(finfo.max)))
        scale = torch.exp2(exponent)
        out_q = (
            torch.clamp(
                grouped / scale.unsqueeze(-1), float(finfo.min), float(finfo.max)
            )
            .to(torch.float8_e4m3fn)
            .view(m, n)
        )
        exp_biased = (exponent + 127.0).clamp(0, 255).to(torch.int32)
        num_packed = (num_groups + 3) // 4
        if num_groups % 4 != 0:
            exp_biased = torch.nn.functional.pad(
                exp_biased, (0, 4 - num_groups % 4)
            )
        packed = exp_biased.view(m, num_packed, 4)
        rowmajor = (
            packed[:, :, 0]
            | (packed[:, :, 1] << 8)
            | (packed[:, :, 2] << 16)
            | (packed[:, :, 3] << 24)
        )
        out_s = make_ue8m0_scale_like(y.shape, device=y.device)
        out_s.copy_(rowmajor)
        return out_q, out_s

    @staticmethod
    def _ref(attn: torch.Tensor, gate: torch.Tensor):
        y = attn * torch.sigmoid(gate)
        if SigmoidMulFp8QuantTest._sgl is not None:
            return SigmoidMulFp8QuantTest._sgl(
                y,
                128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
        return SigmoidMulFp8QuantTest._torch_ue8m0_quant(y)

    def _check(
        self,
        attn: torch.Tensor,
        gate: torch.Tensor,
        *,
        msg: str,
    ):
        ref_q, ref_s = self._ref(attn, gate)
        got_q, got_s = self._fused(attn, gate)

        self.assertEqual(got_q.shape, attn.shape, msg)
        self.assertEqual(got_q.dtype, torch.float8_e4m3fn, msg)
        self.assertEqual(got_s.dtype, torch.int32, msg)
        self.assertEqual(tuple(got_s.shape), tuple(ref_s.shape), f"{msg}: scale shape")
        self.assertEqual(got_s.stride(), ref_s.stride(), f"{msg}: scale stride")

        ulp = _fp8_ulp_diff(ref_q, got_q)
        self.assertLessEqual(ulp, 1, f"{msg}: FP8 ULP={ulp}")

        if got_s.numel() == 0:
            return
        # N=128 has 1 group / 4 packed bytes; sgl may leave the unused
        # bytes untouched (empty alloc). Compare only live groups.
        groups = attn.shape[-1] // 128
        live_bytes = (groups + 3) // 4 * 4
        if groups % 4 == 0:
            scale_mismatch = int((got_s != ref_s).sum().item())
            self.assertEqual(
                scale_mismatch,
                0,
                f"{msg}: UE8M0 scale mismatch {scale_mismatch}/{got_s.numel()}",
            )
        else:
            mask = (1 << (8 * (groups % 4))) - 1
            ref_live = ref_s.to(torch.int32) & mask
            got_live = got_s.to(torch.int32) & mask
            scale_mismatch = int((got_live != ref_live).sum().item())
            self.assertEqual(
                scale_mismatch,
                0,
                f"{msg}: live UE8M0 scale mismatch {scale_mismatch}/{got_s.numel()} "
                f"(compared low {groups % 4} bytes; {live_bytes=})",
            )

    def _rand(self, m: int, n: int, scale: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
        attn = torch.randn(m, n, dtype=torch.bfloat16, device="cuda") * scale
        gate = torch.randn(m, n, dtype=torch.bfloat16, device="cuda") * scale
        return attn.contiguous(), gate.contiguous()

    def test_typical_m(self):
        for m in (1, 8, 32, 128, 256):
            attn, gate = self._rand(m, 8192)
            self._check(attn, gate, msg=f"M={m} N=8192")

    def test_m0(self):
        attn = torch.empty(0, 8192, dtype=torch.bfloat16, device="cuda")
        gate = torch.empty(0, 8192, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(self._is_supported(attn, gate))
        self._check(attn, gate, msg="M=0 N=8192")

    def test_m3_tma_padding(self):
        attn, gate = self._rand(3, 8192)
        self._check(attn, gate, msg="M=3 N=8192")

    def test_n128(self):
        attn, gate = self._rand(16, 128)
        self._check(attn, gate, msg="M=16 N=128")

    def test_gate_zero(self):
        m, n = 32, 8192
        attn = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        gate = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
        self._check(attn, gate, msg="gate=0")

    def test_gate_large(self):
        m, n = 8, 8192
        attn = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        for value in (20.0, -20.0):
            gate = torch.full((m, n), value, dtype=torch.bfloat16, device="cuda")
            self._check(attn, gate, msg=f"gate={value}")

    def test_unsupported_returns_none(self):
        attn, gate = self._rand(4, 8192)
        self.assertIsNotNone(self._maybe(attn, gate))

        cpu = attn.cpu()
        self.assertIsNone(self._maybe(cpu, gate.cpu()))

        mismatched = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(self._maybe(attn, mismatched))

        fp32 = attn.float()
        self.assertIsNone(self._maybe(fp32, gate.float()))

        n127 = torch.randn(4, 127, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(self._maybe(n127, n127))

        non_contig = attn[:, :4096]
        self.assertFalse(non_contig.is_contiguous())
        self.assertIsNone(self._maybe(non_contig, non_contig))

        os.environ["RTP_QWEN35_DECODE_FUSION"] = "0"
        try:
            self.assertFalse(self._is_supported(attn, gate))
            self.assertIsNone(self._maybe(attn, gate))
        finally:
            os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"


if __name__ == "__main__":
    unittest.main()
