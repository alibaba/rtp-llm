"""Real SM90 V4.1 delayed-mHC TileLang path against independent references.

The CPU delayed-unit reference and tolerances match the existing
test_v41_delayed_hc_cuda_graph contract. This test deliberately exercises the
production CUDA BF16 path, not the CPU fallback on a CUDA-shaped fixture.
"""

import importlib
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.hc import tilelang_impl, v41_prenorm
from rtp_llm.models_py.modules.dsv41.hc.delayed import (
    DelayedHCHead,
    DelayedHCUnit,
    _tile_ops,
)

_DIM = 5120
_HC = 4
_NORM_EPS = 1e-20
_HC_EPS = 1e-6


def _unit(weights, device):
    fn, base, scale = weights
    return DelayedHCUnit(
        fn.to(device),
        base.to(device),
        scale.to(device),
        dim=_DIM,
        hc_mult=_HC,
        hc_sinkhorn_iters=20,
        norm_eps=_NORM_EPS,
        hc_eps=_HC_EPS,
    )


def _weights(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(24, _HC * _DIM, generator=generator) * 0.003,
        torch.randn(24, generator=generator) * 0.1,
        torch.tensor([0.2, 0.4, 0.3]),
    )


class Sm90DelayedHCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError("SM90 delayed-mHC test requires a CUDA SM90 GPU")

    @torch.inference_mode()
    def test_vendored_prenorm_actual_shape_and_tf32_reference(self):
        ops = _tile_ops()
        fn_cpu, _, _ = _weights(9141)
        fn_gpu = fn_cpu.cuda()
        # Independent implementation of the existing biased/truncating TF32
        # weight contract. BF16 residuals are exactly representable in TF32.
        rounded_fn = ((fn_cpu.view(torch.int32) + 0x1000) & -8192).view(torch.float32)
        for rows in (1, 3):
            with self.subTest(rows=rows):
                generator = torch.Generator(device="cpu").manual_seed(7000 + rows)
                x_cpu = torch.randn(
                    1, rows, _HC, _DIM, generator=generator, dtype=torch.bfloat16
                )
                x_gpu = x_cpu.cuda()
                # Keep explicit coverage of the original TileLang fallback
                # now that SM90 can also select DeepGEMM small-token prenorm.
                with (
                    patch.object(v41_prenorm, "is_supported", return_value=False),
                    patch.object(
                        v41_prenorm,
                        "_has_prenorm_gemm",
                        side_effect=AssertionError("DeepGEMM probe in forced fallback"),
                    ) as dg_probe,
                ):
                    self.assertIsNone(v41_prenorm.prenorm(x_gpu, fn_gpu, _NORM_EPS))
                    dg_probe.assert_not_called()
                actual = ops.mhc_pre_norm_fn(x_gpu, fn_gpu, None, _NORM_EPS, n_splits=1)
                flat = x_cpu.flatten(-2).float()
                expected = F.linear(flat, rounded_fn) * torch.rsqrt(
                    flat.square().mean(dim=-1, keepdim=True) + _NORM_EPS
                )
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(tuple(actual.shape), (1, rows, 24))
                self.assertTrue(torch.isfinite(actual).all().item())
                # Existing prenorm component tolerance; no relaxation for SM90.
                torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=1e-4)

    @torch.inference_mode()
    def test_delayed_pre_post_and_final_head_follow_production_tilelang_path(self):
        ops = _tile_ops()
        weights = [_weights(8100), _weights(8101)]
        cpu_units = [_unit(weight, "cpu") for weight in weights]
        gpu_units = [_unit(weight, "cuda") for weight in weights]
        cpu_units[1].set_previous(cpu_units[0])
        gpu_units[1].set_previous(gpu_units[0])
        cpu_head, gpu_head = DelayedHCHead(cpu_units[-1]), DelayedHCHead(gpu_units[-1])
        for rows in (1, 3):
            for batched in (False, True):
                with self.subTest(rows=rows, batched=batched):
                    generator = torch.Generator(device="cpu").manual_seed(9000 + rows)
                    shape = (1, rows, _HC, _DIM) if batched else (rows, _HC, _DIM)
                    residual_cpu = torch.randn(
                        shape, generator=generator, dtype=torch.bfloat16
                    )
                    residual_gpu = residual_cpu.cuda()
                    with (
                        # The upstream big-pre eligibility check probes the
                        # configured GEMM backend before its SM100-only gate.
                        # Select TileLang explicitly for this fallback fixture;
                        # keep the independent no-DeepGEMM assertion below.
                        patch.dict(
                            "os.environ",
                            {"DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single"},
                        ),
                        patch.object(v41_prenorm, "is_supported", return_value=False),
                        patch.object(
                            ops, "mhc_pre_norm_fn", wraps=ops.mhc_pre_norm_fn
                        ) as norm_call,
                        patch.object(
                            tilelang_impl,
                            "tk_mhc_post",
                            wraps=tilelang_impl.tk_mhc_post,
                        ) as post_call,
                        patch.object(
                            v41_prenorm,
                            "_has_prenorm_gemm",
                            side_effect=AssertionError(
                                "DeepGEMM probe in forced fallback"
                            ),
                        ) as dg_probe,
                    ):
                        for index, (cpu_unit, gpu_unit) in enumerate(
                            zip(cpu_units, gpu_units)
                        ):
                            expected_y, expected_post, expected_comb = cpu_unit.pre(
                                residual_cpu
                            )
                            actual_y, actual_post, actual_comb = gpu_unit.pre(
                                residual_gpu
                            )
                            self.assertEqual(actual_y.dtype, torch.bfloat16)
                            self.assertEqual(actual_post.dtype, torch.float32)
                            self.assertEqual(actual_comb.dtype, torch.float32)
                            if index == 0:
                                torch.testing.assert_close(
                                    actual_y, residual_gpu[..., 0, :], rtol=0, atol=0
                                )
                            for actual, expected in (
                                (actual_y, expected_y),
                                (gpu_unit.pre_mix_out, cpu_unit.pre_mix_out),
                                (actual_post, expected_post),
                                (actual_comb, expected_comb),
                            ):
                                torch.testing.assert_close(
                                    actual.cpu(), expected, rtol=0.02, atol=0.03125
                                )
                            storage = residual_gpu.data_ptr()
                            residual_cpu = cpu_unit.post(
                                expected_y * 0.25,
                                residual_cpu,
                                expected_post,
                                expected_comb,
                            )
                            residual_gpu = gpu_unit.post(
                                actual_y * 0.25, residual_gpu, actual_post, actual_comb
                            )
                            # Actual TileLang HC post reuses the production residual buffer.
                            self.assertEqual(residual_gpu.data_ptr(), storage)
                            torch.testing.assert_close(
                                residual_gpu.cpu(),
                                residual_cpu,
                                rtol=0.02,
                                atol=0.03125,
                            )
                        actual_head, expected_head = gpu_head.head(
                            residual_gpu
                        ), cpu_head.head(residual_cpu)
                        torch.testing.assert_close(
                            actual_head.cpu(), expected_head, rtol=0.02, atol=0.03125
                        )
                        self.assertEqual(tuple(actual_head.shape), (*shape[:-2], _DIM))
                        self.assertEqual(norm_call.call_count, 2)
                        self.assertEqual(post_call.call_count, 2)
                        dg_probe.assert_not_called()


def _tf32_rne_host(weight):
    """Independent host oracle for the native tensor-map conversion."""
    bits = weight.detach().cpu().contiguous().view(torch.int32).to(torch.int64)
    bits = bits & 0xFFFFFFFF
    special = (bits & 0x7F800000) == 0x7F800000
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000
    return torch.where(special, bits, rounded).to(torch.int32).view(torch.float32)


def _split_sinkhorn(mixes, base, scale, eps, iters):
    pre = torch.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    comb = (mixes[..., 8:24] * scale[2] + base[8:24]).reshape(*mixes.shape[:-1], 4, 4)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _big_fuse_oracle(
    residual, fn, base, scale, previous_pre, weight, rms_eps, eps, iters
):
    """FP64 reference for the whole fused pre chain, including the readout."""
    tokens = residual.shape[1]
    flat = residual.detach().cpu().reshape(tokens, -1).double()
    mixes = (flat @ weight.double().T) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre, post, comb = _split_sinkhorn(
        mixes.reshape(1, tokens, 24), base.double(), scale.double(), eps, iters
    )
    y = (
        residual.detach().cpu().double()
        * previous_pre.detach().cpu().double().unsqueeze(-1)
    ).sum(dim=-2)
    # The kernels emit FP32 coefficients and a BF16 readout.
    return pre.float(), post.float(), comb.float(), y.to(torch.bfloat16)


class Sm90BigPreRoutingTest(unittest.TestCase):
    """Pure dispatch contracts; no GPU is touched."""

    def test_sm90_row_window_and_backend_routing(self):
        from rtp_llm.models_py.modules.dsv41.hc.delayed import (
            _big_pre_backend,
            _sm90_big_pre_rows,
        )

        sm90, sm100 = (9, 0), (10, 0)
        self.assertFalse(_sm90_big_pre_rows(64, sm90))
        self.assertTrue(_sm90_big_pre_rows(65, sm90))
        self.assertTrue(_sm90_big_pre_rows(512, sm90))
        self.assertFalse(_sm90_big_pre_rows(513, sm90))
        self.assertFalse(_sm90_big_pre_rows(128, sm100))
        with patch.dict(os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": ""}):
            # DeepGEMM covers the whole fused decode window when installed.
            self.assertEqual(_big_pre_backend(sm90, True), "deepgemm")
            # Without the DeepGEMM primitive split-K takes over.
            self.assertEqual(_big_pre_backend(sm90, False), "tilelang_splitk")
            # SM100 keeps the established contract.
            self.assertEqual(_big_pre_backend(sm100, True), "deepgemm")
            self.assertIsNone(_big_pre_backend(sm100, False))
        with patch.dict(
            os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single"}
        ):
            self.assertIsNone(_big_pre_backend(sm90, False))
            self.assertEqual(_big_pre_backend(sm100, True), "deepgemm")
        with patch.dict(
            os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_splitk"}
        ):
            self.assertEqual(_big_pre_backend(sm90, False), "tilelang_splitk")
            self.assertEqual(_big_pre_backend(sm100, True), "deepgemm")


class Sm90BigFuseSplitKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError("SM90 split-K big-fuse test requires a CUDA SM90 GPU")

    @torch.inference_mode()
    def test_splitk_big_fuse_matches_fp64_oracle_on_unaligned_rows(self):
        ops = _tile_ops()
        fn_cpu, base_cpu, scale_cpu = _weights(4242)
        weight = _tf32_rne_host(fn_cpu)
        fn_gpu, base_gpu, scale_gpu = (
            fn_cpu.cuda(),
            base_cpu.cuda(),
            scale_cpu.cuda(),
        )
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        norm = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.norm_fn"
        )
        for rows in (65, 80, 92, 96, 128, 264, 352, 512):
            with self.subTest(rows=rows):
                # Unaligned M exercises the tail-predicated 32-token tile.
                generator = torch.Generator(device="cpu").manual_seed(5100 + rows)
                residual_cpu = torch.randn(
                    1, rows, _HC, _DIM, generator=generator, dtype=torch.bfloat16
                )
                previous_cpu = torch.rand(1, rows, _HC, generator=generator)
                splits = norm._tilelang_split_plan(rows, _HC * _DIM, num_sms)
                self.assertGreater(splits, 1)
                self.assertEqual((_HC * _DIM) % (splits * 256), 0)
                actual = ops.mhc_pre_big_fuse(
                    residual_cpu.cuda(),
                    fn_gpu,
                    scale_gpu,
                    base_gpu,
                    rms_eps=_NORM_EPS,
                    mhc_pre_eps=_HC_EPS,
                    mhc_sinkhorn_eps=_HC_EPS,
                    mhc_post_mult_value=2.0,
                    sinkhorn_repeat=20,
                    previous_pre_mix=previous_cpu.cuda(),
                    gemm_backend="tilelang_splitk",
                )
                expected = _big_fuse_oracle(
                    residual_cpu,
                    fn_cpu,
                    base_cpu,
                    scale_cpu,
                    previous_cpu,
                    weight,
                    _NORM_EPS,
                    _HC_EPS,
                    20,
                )
                for name, got, ref in zip(
                    ("pre", "post", "comb"), actual[:3], expected[:3]
                ):
                    self.assertTrue(torch.isfinite(got).all(), name)
                    torch.testing.assert_close(
                        got.cpu().reshape(ref.shape),
                        ref,
                        rtol=2e-4,
                        atol=1e-5,
                        msg=name,
                    )
                torch.testing.assert_close(
                    actual[3].cpu(), expected[3], rtol=2e-2, atol=3.2e-2
                )

    @torch.inference_mode()
    def test_splitk_halfway_weights_use_rne_not_legacy_bias(self):
        """A tie fixture pins the weight contract to RN-even TF32 values.

        Every weight is an exact TF32 halfway value with an even retained
        mantissa, so the legacy ``+0x1000`` bias rounds it up while RN-even
        keeps it. The projection then differs by one ULP per element, which the
        soft-saturated post mix turns into a measurable coefficient shift.
        """
        ops = _tile_ops()
        rows = 96
        value = (
            torch.tensor([0x3B001000], dtype=torch.int64)
            .to(torch.int32)
            .view(torch.float32)
            .item()
        )
        fn_cpu = torch.full((24, _HC * _DIM), value, dtype=torch.float32)
        base_cpu = torch.zeros(24)
        # Put the post logit at exactly zero for the RN-even projection.
        base_cpu[4:8] = -16.0
        scale_cpu = torch.tensor([0.2, 0.4, 0.3])
        residual_cpu = torch.ones(1, rows, _HC, _DIM, dtype=torch.bfloat16)
        previous_cpu = torch.full((1, rows, _HC), 0.25)
        actual = ops.mhc_pre_big_fuse(
            residual_cpu.cuda(),
            fn_cpu.cuda(),
            scale_cpu.cuda(),
            base_cpu.cuda(),
            rms_eps=_NORM_EPS,
            mhc_pre_eps=_HC_EPS,
            mhc_sinkhorn_eps=_HC_EPS,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=20,
            previous_pre_mix=previous_cpu.cuda(),
            gemm_backend="tilelang_splitk",
        )
        expected = _big_fuse_oracle(
            residual_cpu,
            fn_cpu,
            base_cpu,
            scale_cpu,
            previous_cpu,
            _tf32_rne_host(fn_cpu),
            _NORM_EPS,
            _HC_EPS,
            20,
        )
        biased = ((fn_cpu.view(torch.int32) + 0x1000) & -8192).view(torch.float32)
        legacy = _big_fuse_oracle(
            residual_cpu,
            fn_cpu,
            base_cpu,
            scale_cpu,
            previous_cpu,
            biased,
            _NORM_EPS,
            _HC_EPS,
            20,
        )
        actual_post = actual[1].cpu().reshape(expected[1].shape)
        torch.testing.assert_close(actual_post, expected[1], rtol=0, atol=1e-6)
        self.assertGreater(
            (actual_post - legacy[1]).abs().max().item(), 1e-3
        )

    @torch.inference_mode()
    def test_delayed_unit_routes_big_pre_on_sm90_above_64_rows(self):
        ops = _tile_ops()
        weights = [_weights(8300), _weights(8301)]
        from rtp_llm.models_py.modules.dsv41.hc import v41_prenorm

        for rows, deepgemm, backend in (
            (80, True, "deepgemm"),
            (264, True, "deepgemm"),
            (264, False, "tilelang_splitk"),
        ):
            with self.subTest(rows=rows, deepgemm=deepgemm):
                units = [_unit(weight, "cuda") for weight in weights]
                units[1].set_previous(units[0])
                units[0].pre_mix_out = torch.rand(1, rows, _HC, device="cuda")
                generator = torch.Generator(device="cpu").manual_seed(6400 + rows)
                residual = torch.randn(
                    1, rows, _HC, _DIM, generator=generator, dtype=torch.bfloat16
                ).cuda()
                with patch.dict(
                    os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": ""}
                ), patch.object(
                    v41_prenorm,
                    "is_supported",
                    side_effect=lambda r, f, *, max_tokens=None, _ok=deepgemm: _ok,
                ), patch.object(
                    ops, "mhc_pre_big_fuse", wraps=ops.mhc_pre_big_fuse
                ) as fused:
                    y, post, comb = units[1].pre(residual)
                self.assertEqual(fused.call_args.kwargs["gemm_backend"], backend)
                self.assertEqual(tuple(y.shape), (1, rows, _DIM))
                # The fused chain must reproduce the established split kernels.
                mixes = ops.mhc_pre_norm_fn(
                    residual, units[1].fn, None, _NORM_EPS, n_splits=1
                )
                pre, expected_post, expected_comb = ops.mhc_pre_split_mixes(
                    mixes, units[1].scale, units[1].base, _HC, 2.0, _HC_EPS
                )
                expected_comb = ops.sinkhorn_normalize(
                    expected_comb, repeat=20, eps=_HC_EPS
                )
                expected_y = ops.mhc_pre_apply_mix(
                    residual, units[0].pre_mix_out.unsqueeze(-1)
                )
                torch.testing.assert_close(
                    units[1].pre_mix_out, pre.reshape(1, rows, _HC), rtol=0, atol=1e-6
                )
                torch.testing.assert_close(post, expected_post, rtol=2e-4, atol=1e-5)
                torch.testing.assert_close(comb, expected_comb, rtol=2e-4, atol=1e-5)
                torch.testing.assert_close(y, expected_y, rtol=2e-2, atol=3.2e-2)


if __name__ == "__main__":
    unittest.main()
