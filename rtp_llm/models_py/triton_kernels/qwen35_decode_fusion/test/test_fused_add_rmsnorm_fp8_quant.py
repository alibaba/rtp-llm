"""Correctness tests for fused AddRMSNorm + FP8 UE8M0 quant (group G).

Compares against ``rtp_llm_ops.fused_add_rmsnorm`` +
``sgl_per_token_group_quant_fp8``.
"""

from __future__ import annotations

import os
import unittest

import torch

from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fused_add_rmsnorm_fp8_quant import (
    fused_add_rmsnorm_fp8_quant,
    is_supported,
    maybe_fused_add_rmsnorm_fp8_quant,
)

EPS = 1.0e-6
GROUP_SIZE = 128
CLAMP_EPS = 1.0e-4
NORM_ATOL = 2.0e-2
NORM_RTOL = 1.6e-2


def _sgl_per_token_group_quant_fp8(
    hidden: torch.Tensor, group_size: int, clamp_eps: float
):
    """Old-path quant. Prefer ``sgl_per_token_group_quant_fp8``; fall back to
    the same CUDA op it calls so the test does not require ``rtp_kernel``.
    """
    try:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        return sgl_per_token_group_quant_fp8(
            hidden,
            group_size,
            eps=clamp_eps,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
    except ModuleNotFoundError:
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fp8_scale import (
            make_ue8m0_scale_like,
        )
        from rtp_llm.ops.compute_ops import per_token_group_quant_fp8

        finfo = torch.finfo(torch.float8_e4m3fn)
        fp8 = torch.empty(hidden.shape, device=hidden.device, dtype=torch.float8_e4m3fn)
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
                clamp_eps,
                float(finfo.min),
                float(finfo.max),
                True,
            )
        return fp8, scale


def _old_path(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = EPS,
    group_size: int = GROUP_SIZE,
    clamp_eps: float = CLAMP_EPS,
):
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    hidden_out = hidden.clone()
    residual_out = residual.clone()
    if hidden_out.shape[0] > 0:
        stream_id = torch.cuda.current_stream().cuda_stream
        rtp_llm_ops.fused_add_rmsnorm(hidden_out, residual_out, weight, eps, stream_id)
    fp8, scale = _sgl_per_token_group_quant_fp8(hidden_out, group_size, clamp_eps)
    return hidden_out, residual_out, fp8, scale


def _make_inputs(
    m: int,
    n: int = 4096,
    *,
    residual_mode: str = "rand",
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    hidden = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    if residual_mode == "zeros":
        residual = torch.zeros(m, n, dtype=torch.bfloat16, device=device)
    else:
        residual = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    weight = torch.randn(n, dtype=torch.bfloat16, device=device)
    return hidden, residual, weight


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedAddRmsNormFp8QuantTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"

    def _assert_fp8_ulp(self, ref_q: torch.Tensor, got_q: torch.Tensor) -> None:
        self.assertEqual(got_q.shape, ref_q.shape)
        self.assertEqual(got_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(ref_q.dtype, torch.float8_e4m3fn)
        if ref_q.numel() == 0:
            return
        ref_bytes = ref_q.reshape(-1).view(torch.uint8)
        got_bytes = got_q.reshape(-1).view(torch.uint8)
        delta = (ref_bytes.to(torch.int16) - got_bytes.to(torch.int16)).abs()
        self.assertLessEqual(int(delta.max().item()), 1)

    def _assert_scale_layout(self, ref_s: torch.Tensor, got_s: torch.Tensor) -> None:
        self.assertEqual(got_s.shape, ref_s.shape)
        self.assertEqual(got_s.dtype, torch.int32)
        self.assertEqual(ref_s.dtype, torch.int32)
        self.assertEqual(got_s.stride(), ref_s.stride())
        if ref_s.numel() == 0:
            return
        torch.testing.assert_close(got_s, ref_s, rtol=0, atol=0)

    def _assert_match(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = EPS,
        group_size: int = GROUP_SIZE,
    ) -> None:
        ref_norm, ref_res, ref_q, ref_s = _old_path(
            hidden, residual, weight, eps=eps, group_size=group_size
        )
        hidden_in = hidden.clone()
        residual_in = residual.clone()
        got = maybe_fused_add_rmsnorm_fp8_quant(
            hidden_in, residual_in, weight, eps=eps, group_size=group_size
        )
        self.assertIsNotNone(got)
        got_norm, got_res, got_q, got_s = got

        torch.testing.assert_close(got_res, ref_res, rtol=0, atol=0)
        torch.testing.assert_close(got_norm, ref_norm, rtol=NORM_RTOL, atol=NORM_ATOL)
        self._assert_fp8_ulp(ref_q, got_q)
        self._assert_scale_layout(ref_s, got_s)
        self.assertEqual(got_norm.data_ptr(), hidden_in.data_ptr())
        self.assertEqual(got_res.data_ptr(), residual_in.data_ptr())

    def test_typical_m_matches_old_path(self) -> None:
        for m in (1, 8, 32, 128, 256):
            with self.subTest(m=m):
                hidden, residual, weight = _make_inputs(m, 4096)
                self._assert_match(hidden, residual, weight)

    def test_empty_m(self) -> None:
        hidden, residual, weight = _make_inputs(0, 4096)
        ref_norm, ref_res, ref_q, ref_s = _old_path(hidden, residual, weight)
        got = maybe_fused_add_rmsnorm_fp8_quant(hidden, residual, weight)
        self.assertIsNotNone(got)
        got_norm, got_res, got_q, got_s = got
        self.assertEqual(tuple(got_norm.shape), (0, 4096))
        self.assertEqual(tuple(got_res.shape), (0, 4096))
        self.assertEqual(tuple(got_q.shape), (0, 4096))
        self.assertEqual(got_q.dtype, torch.float8_e4m3fn)
        self._assert_scale_layout(ref_s, got_s)
        self.assertEqual(ref_norm.shape, got_norm.shape)
        self.assertEqual(ref_q.shape, got_q.shape)

    def test_unaligned_m(self) -> None:
        hidden, residual, weight = _make_inputs(3, 4096)
        self._assert_match(hidden, residual, weight)

    def test_n_128(self) -> None:
        hidden, residual, weight = _make_inputs(8, 128)
        self._assert_match(hidden, residual, weight)

    def test_residual_zeros_first_layer(self) -> None:
        hidden, residual, weight = _make_inputs(32, 4096, residual_mode="zeros")
        self._assert_match(hidden, residual, weight)

    def test_noncontiguous_rejected(self) -> None:
        hidden, residual, weight = _make_inputs(8, 4096)
        wide = torch.randn(8, 4104, dtype=torch.bfloat16, device=hidden.device)
        hidden_nc = wide[:, :4096]
        self.assertFalse(hidden_nc.is_contiguous())
        self.assertFalse(is_supported(hidden_nc, residual, weight))
        self.assertIsNone(maybe_fused_add_rmsnorm_fp8_quant(hidden_nc, residual, weight))

        residual_wide = torch.randn(8, 4104, dtype=torch.bfloat16, device=hidden.device)
        residual_nc = residual_wide[:, :4096]
        self.assertIsNone(maybe_fused_add_rmsnorm_fp8_quant(hidden, residual_nc, weight))

    def test_unsupported_dtype_returns_none(self) -> None:
        hidden, residual, weight = _make_inputs(4, 4096)
        hidden_f32 = hidden.float()
        self.assertFalse(is_supported(hidden_f32, residual, weight))
        self.assertIsNone(maybe_fused_add_rmsnorm_fp8_quant(hidden_f32, residual, weight))

    def test_env_opt_in(self) -> None:
        hidden, residual, weight = _make_inputs(4, 4096)
        os.environ.pop("RTP_QWEN35_DECODE_FUSION", None)
        self.assertFalse(is_supported(hidden, residual, weight))
        self.assertIsNone(maybe_fused_add_rmsnorm_fp8_quant(hidden, residual, weight))
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"
        self.assertTrue(is_supported(hidden, residual, weight))
        got = maybe_fused_add_rmsnorm_fp8_quant(
            hidden.clone(), residual.clone(), weight
        )
        self.assertIsNotNone(got)
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "0"
        self.assertIsNone(maybe_fused_add_rmsnorm_fp8_quant(hidden, residual, weight))

    def test_fused_raises_when_unsupported(self) -> None:
        hidden, residual, weight = _make_inputs(4, 4096)
        with self.assertRaises(ValueError):
            fused_add_rmsnorm_fp8_quant(hidden.float(), residual, weight)


if __name__ == "__main__":
    unittest.main()
