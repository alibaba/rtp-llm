"""Equivalence test for fused RmsNormGated + UE8M0 FP8 quant.

Compares ``rmsnorm_gated_fp8_quant`` against the old two-op path::

    y = RmsNormGated(weight, group_size=128, eps=1e-6, activation="silu")(x, z)
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


def _used_scale_bytes(scale: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Unpack little-endian UE8M0 bytes for the first ``num_groups`` groups."""
    if scale.numel() == 0 or num_groups == 0:
        return torch.empty(0, dtype=torch.int32, device=scale.device)
    packs = scale.to(torch.int32)
    m = packs.shape[0]
    out = torch.empty((m, num_groups), dtype=torch.int32, device=scale.device)
    for group_id in range(num_groups):
        pack = group_id // 4
        idx = group_id % 4
        out[:, group_id] = (packs[:, pack] >> (idx * 8)) & 0xFF
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RmsNormGatedFp8QuantTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"

    @classmethod
    def setUpClass(cls):
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
            RmsNormGated,
        )
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.rmsnorm_gated_fp8_quant import (
            is_supported,
            maybe_rmsnorm_gated_fp8_quant,
            rmsnorm_gated_fp8_quant,
        )

        cls._RmsNormGated = RmsNormGated
        cls._fused = staticmethod(rmsnorm_gated_fp8_quant)
        cls._maybe = staticmethod(maybe_rmsnorm_gated_fp8_quant)
        cls._is_supported = staticmethod(is_supported)
        try:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
                sgl_per_token_group_quant_fp8,
            )

            cls._sgl = staticmethod(sgl_per_token_group_quant_fp8)
        except Exception:
            from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fp8_scale import (
                make_ue8m0_scale_like,
            )
            from rtp_llm.ops.compute_ops import per_token_group_quant_fp8

            def _sgl_fallback(
                hidden: torch.Tensor,
                group_size: int,
                eps: float = 1e-4,
                **_kwargs,
            ):
                finfo = torch.finfo(torch.float8_e4m3fn)
                fp8 = torch.empty(
                    hidden.shape, device=hidden.device, dtype=torch.float8_e4m3fn
                )
                scale = make_ue8m0_scale_like(
                    hidden.shape, device=hidden.device, group_size=group_size
                )
                if (hidden.shape[-1] // group_size) % 4 != 0:
                    scale.zero_()
                if hidden.shape[0] > 0:
                    per_token_group_quant_fp8(
                        hidden,
                        fp8,
                        scale,
                        group_size,
                        eps,
                        float(finfo.min),
                        float(finfo.max),
                        True,
                    )
                return fp8, scale

            cls._sgl = staticmethod(_sgl_fallback)

    def _ref(self, x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor):
        # Production RmsNormGated uses group_size=head_v_dim=128 for both
        # shared weight [128] and a full-width weight [N].
        norm = self._RmsNormGated(weight, group_size=128, eps=1e-6, activation="silu")
        y = norm(x, z)
        q, s = self._sgl(
            y,
            128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        return y, q, s

    def _check(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        *,
        msg: str,
    ):
        ref_y, ref_q, ref_s = self._ref(x, z, weight)
        got_y, got_q, got_s = self._fused(x, z, weight)

        self.assertEqual(got_y.shape, x.shape, msg)
        self.assertEqual(got_y.dtype, torch.bfloat16, msg)
        self.assertEqual(got_q.shape, x.shape, msg)
        self.assertEqual(got_q.dtype, torch.float8_e4m3fn, msg)
        self.assertEqual(got_s.dtype, torch.int32, msg)
        self.assertEqual(tuple(got_s.shape), tuple(ref_s.shape), f"{msg}: scale shape")
        self.assertEqual(got_s.stride(), ref_s.stride(), f"{msg}: scale stride")

        if x.numel() > 0:
            y_abs = (ref_y.float() - got_y.float()).abs().max().item()
            y_ref_max = ref_y.float().abs().max().item()
            y_tol = max(2e-2, 2.5e-3 * max(y_ref_max, 1.0))
            self.assertLessEqual(
                y_abs,
                y_tol,
                f"{msg}: gated RMS y max_abs={y_abs} tol={y_tol} ref_amax={y_ref_max}",
            )

        ulp = _fp8_ulp_diff(ref_q, got_q)
        self.assertLessEqual(ulp, 1, f"{msg}: FP8 ULP={ulp}")

        num_groups = x.shape[-1] // 128
        if got_s.numel() == 0 or num_groups == 0:
            return
        ref_bytes = _used_scale_bytes(ref_s, num_groups)
        got_bytes = _used_scale_bytes(got_s, num_groups)
        scale_mismatch = int((ref_bytes != got_bytes).sum().item())
        self.assertEqual(
            scale_mismatch,
            0,
            f"{msg}: UE8M0 scale mismatch {scale_mismatch}/{ref_bytes.numel()}",
        )

    def _shared_w(self, group_size: int = 128) -> torch.Tensor:
        return (
            (torch.randn(group_size, device="cuda") * 0.1 + 1.0)
            .abs()
            .to(torch.bfloat16)
            .contiguous()
        )

    def _rand(
        self, m: int, n: int, *, shared: bool = True, scale: float = 2.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = (torch.randn(m, n, dtype=torch.bfloat16, device="cuda") * scale).contiguous()
        z = (torch.randn(m, n, dtype=torch.bfloat16, device="cuda") * scale).contiguous()
        if shared:
            weight = self._shared_w(128)
        else:
            weight = (
                (torch.randn(n, device="cuda") * 0.1 + 1.0)
                .abs()
                .to(torch.bfloat16)
                .contiguous()
            )
        return x, z, weight

    def test_typical_m(self):
        for m in (1, 8, 32, 128, 256):
            x, z, w = self._rand(m, 8192)
            self._check(x, z, w, msg=f"M={m} N=8192 shared")

    def test_m0(self):
        x = torch.empty(0, 8192, dtype=torch.bfloat16, device="cuda")
        z = torch.empty(0, 8192, dtype=torch.bfloat16, device="cuda")
        w = self._shared_w()
        self.assertTrue(self._is_supported(x, z, w))
        self._check(x, z, w, msg="M=0 N=8192")

    def test_m3_tma_padding(self):
        x, z, w = self._rand(3, 8192)
        self._check(x, z, w, msg="M=3 N=8192")

    def test_n128_one_group(self):
        x, z, w = self._rand(16, 128)
        self._check(x, z, w, msg="M=16 N=128")

    def test_n256_two_groups(self):
        x, z, w = self._rand(16, 256)
        self._check(x, z, w, msg="M=16 N=256")

    def test_non_shared_weight(self):
        x, z, w = self._rand(8, 8192, shared=False)
        self.assertEqual(tuple(w.shape), (8192,))
        self._check(x, z, w, msg="non-shared weight [N]")

    def test_z_zero(self):
        x, z, w = self._rand(32, 8192)
        z = torch.zeros_like(z)
        self._check(x, z, w, msg="z=0")

    def test_z_large(self):
        x, _, w = self._rand(8, 8192)
        for value in (20.0, -20.0):
            z = torch.full_like(x, value)
            self._check(x, z, w, msg=f"z={value}")

    def test_unsupported_returns_none(self):
        x, z, w = self._rand(4, 8192)
        self.assertIsNotNone(self._maybe(x, z, w))

        self.assertIsNone(self._maybe(x.cpu(), z.cpu(), w.cpu()))

        mismatched = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(self._maybe(x, mismatched, w))

        self.assertIsNone(self._maybe(x.float(), z.float(), w.float()))

        n127 = torch.randn(4, 127, dtype=torch.bfloat16, device="cuda")
        w127 = self._shared_w()
        self.assertIsNone(self._maybe(n127, n127, w127))

        non_contig = x[:, :4096]
        self.assertFalse(non_contig.is_contiguous())
        self.assertIsNone(self._maybe(non_contig, non_contig, w))

        self.assertIsNone(self._maybe(x, z, w, activation="sigmoid"))
        self.assertIsNone(self._maybe(x, z, w, group_size=64))
        self.assertIsNone(self._maybe(x, z, w, bias=w))

        os.environ["RTP_QWEN35_DECODE_FUSION"] = "0"
        try:
            self.assertFalse(self._is_supported(x, z, w))
            self.assertIsNone(self._maybe(x, z, w))
        finally:
            os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"


if __name__ == "__main__":
    unittest.main()
