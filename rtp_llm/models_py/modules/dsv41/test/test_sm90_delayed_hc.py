"""Real SM90 V4.1 delayed-mHC TileLang path against independent references.

The CPU delayed-unit reference and tolerances match the existing
test_v41_delayed_hc_cuda_graph contract. This test deliberately exercises the
production CUDA BF16 path, not the CPU fallback on a CUDA-shaped fixture.
"""

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


if __name__ == "__main__":
    unittest.main()
