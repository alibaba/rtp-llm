"""Numerical gates for BS 0..16 and DSpark gamma 3..5 on SM90."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import _sm90_quant_scale_fused as fused


def assert_bytes(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    if not actual.numel():
        return
    test.assertTrue(
        torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        ),
        f"byte mismatch: {actual.shape} {actual.dtype}",
    )


def packed_reference(x):
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    q, packed = sgl_per_token_group_quant_fp8(
        x.contiguous(),
        32,
        eps=torch.finfo(torch.float32).tiny,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return q, fused.unpack_ue8m0_to_fp32(packed, mn_major=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeBatchNumerics(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 required")
        torch.manual_seed(20261105)
        self.inference = torch.inference_mode()
        self.inference.__enter__()

    def tearDown(self):
        if hasattr(self, "inference"):
            self.inference.__exit__(None, None, None)

    def test_quant_all_target_and_draft_row_counts(self):
        rows = sorted({b * g for b in range(17) for g in (3, 4, 5, 6)})
        for policy in ("auto", "legacy", "v2"):
            with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=policy):
                for m in rows:
                    for k in (1024, 2048, 2304, 5120, 8192):
                        with self.subTest(policy=policy, m=m, k=k):
                            # A view exercises the new CTA's masked final groups.
                            x = torch.randn(
                                (m, k + 32), device="cuda", dtype=torch.bfloat16
                            )[:, :k]
                            actual = fused.quantize_g32(x, mn_major=True)
                            expected = packed_reference(x)
                            for a, e in zip(actual, expected):
                                assert_bytes(self, a, e)
                            self.assertEqual(
                                actual[1].stride(), (1, fused.mn_major_row_stride(m))
                            )

    def test_every_finite_bf16_magnitude_at_new_tile_size(self):
        # Covers every exponent/mantissa, both signs, subnormals and signed zero.
        bits = torch.arange(65536, device="cuda", dtype=torch.int32)
        values = bits.to(torch.int16).view(torch.bfloat16)
        values = values[torch.isfinite(values)]
        for policy in ("auto", "legacy", "v2"):
            with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=policy):
                for start in range(0, values.numel(), 96 * 256):
                    groups = torch.zeros(96 * 256, device="cuda", dtype=torch.bfloat16)
                    part = values[start : start + groups.numel()]
                    groups[: part.numel()].copy_(part)
                    x = groups[:, None].expand(-1, 32).contiguous().view(96, 8192)
                    for a, e in zip(
                        fused.quantize_g32(x, mn_major=True), packed_reference(x)
                    ):
                        assert_bytes(self, a, e)

    def test_dense_gemm_unchanged_at_dispatch_boundary(self):
        from rtp_llm.models_py.modules.dsv41.sm90_deepgemm_dense import (
            Sm90DeepGEMMMXFP8Linear,
        )

        for n, k in ((512, 5120), (5120, 8192), (2048, 1024)):
            weight = (torch.randn((n, k), device="cuda") * 0.01).to(torch.float8_e4m3fn)
            scale = torch.ones((n // 32, k // 32), device="cuda").to(
                torch.float8_e8m0fnu
            )
            linear = Sm90DeepGEMMMXFP8Linear(weight, scale)
            for m in (0, 16, 17, 32, 48, 64, 65, 80, 90, 96):
                x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                expected = linear.forward_quantized(*packed_reference(x))
                assert_bytes(self, linear.forward_bf16(x), expected)
            with self.assertRaises(ValueError):
                linear.forward_bf16(
                    torch.empty((97, k), device="cuda", dtype=torch.bfloat16)
                )

    def test_grouped_woa_with_request_frequency_broadcast(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_output_projection import (
            Sm90GroupedOutputProjection,
        )

        for groups in (1, 8):
            n, k = 1024, 4096
            weight = (torch.randn((groups, n, k), device="cuda") * 0.01).to(
                torch.float8_e4m3fn
            )
            scale = torch.ones((groups, n // 32, k // 32), device="cuda").to(
                torch.float8_e8m0fnu
            )
            linear = Sm90GroupedOutputProjection(weight, scale)
            for batch in (1, 4, 5, 8, 9, 11, 13, 16):
                for steps in (3, 4, 5, 6):
                    with self.subTest(groups=groups, batch=batch, steps=steps):
                        x = torch.randn(
                            (batch, steps, groups * 8, 512),
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        phase = torch.randn((batch, 32), device="cuda")
                        frequencies = torch.polar(torch.ones_like(phase), phase)
                        before = linear.forward_quantized(
                            *linear.quantize(x, frequencies)
                        )
                        assert_bytes(self, linear.forward(x, frequencies), before)

    def test_mhc_tile_preserves_every_output_and_deferred_add(self):
        from rtp_llm.models_py.modules.dsv41.hc import mhc_sm90_vllm_two_kernel as mhc

        plain, combine = (
            mhc._mhc_fused_post_projection,
            mhc._mhc_fused_post_projection_combine,
        )
        for seed in range(5):
            torch.manual_seed(seed)
            weight = mhc.prepare_tf32_weight(
                torch.randn((24, 20480), device="cuda") / 20480**0.5
            )
            for m in (32, 33, 40, 48, 60, 64):
                x = torch.randn((m, 5120), device="cuda", dtype=torch.bfloat16)
                shared = torch.randn_like(x)
                args = (
                    x,
                    torch.randn((m, 4, 5120), device="cuda", dtype=torch.bfloat16),
                    torch.rand((m, 4), device="cuda") * 2,
                    torch.softmax(torch.randn((m, 4, 4), device="cuda"), -1),
                    weight,
                    torch.rand((m, 4), device="cuda"),
                    torch.ones(3, device="cuda") * 0.5,
                    torch.zeros(24, device="cuda"),
                    torch.ones(5120, device="cuda", dtype=torch.bfloat16),
                )
                for deferred in (False, True):
                    kw = dict(
                        hc_norm_eps=1e-6,
                        pre_eps=1e-6,
                        sinkhorn_eps=1e-6,
                        sinkhorn_repeat=20,
                        rmsnorm_eps=1e-20,
                        combine=(x, shared) if deferred else None,
                    )
                    actual = mhc.run(*args, **kw)
                    with patch.object(
                        mhc, "_mhc_fused_post_projection", lambda _: plain(6)
                    ), patch.object(
                        mhc, "_mhc_fused_post_projection_combine", lambda _: combine(6)
                    ):
                        expected = mhc.run(*args, **kw)
                    for a, e in zip(actual, expected):
                        assert_bytes(self, a, e)


if __name__ == "__main__":
    unittest.main()
