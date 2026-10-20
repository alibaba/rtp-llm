"""Opt-in SM120 GEMMs: dispatch, quantization, padding, and graph replay."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4 import utils
from rtp_llm.models_py.modules.dsv4.moe import shared_expert
from rtp_llm.models_py.modules.dsv4.moe.strategies import grouped_fp4
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
)
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.utils.model_weight import W


def quantized_reference(layer, x):
    q, packed = sgl_per_token_group_quant_fp8(
        x,
        group_size=128,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    shifts = torch.arange(4, device=x.device) * 8
    exponents = ((packed.long().unsqueeze(-1) >> shifts) & 255).reshape(x.shape[0], -1)
    scale = torch.exp2(exponents[:, : x.shape[1] // 128].double() - 127)
    a = q.double() * scale.double().repeat_interleave(128, dim=1)
    b = layer.weight.double() * layer.weight_scales.double().repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)
    return (a @ b.t()).to(torch.bfloat16)


class BackendSelectionTest(unittest.TestCase):
    def test_explicit_deepgemm_probe_fails_closed(self):
        import deep_gemm

        with (
            patch.dict(os.environ, DSV4_MOE_FP4_BACKEND="deepgemm"),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            patch.object(grouped_fp4, "is_sm120", return_value=True),
            patch.object(
                grouped_fp4,
                "_has_sm120_flashinfer_grouped_fp4_kernels",
                return_value=False,
            ),
        ):
            self.assertTrue(grouped_fp4._has_fp8_fp4_grouped_kernel())
            with patch.object(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous", None):
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    grouped_fp4._has_fp8_fp4_grouped_kernel()

    def test_selectors_reject_typos(self):
        for name in (
            "DSV4_SM120_FP8_LINEAR_BACKEND",
            "DSV4_SM120_SHARED_EXPERT_BACKEND",
        ):
            with patch.dict(os.environ, {name: "typo"}):
                with self.assertRaises(ValueError):
                    utils._sm120_linear_backend(name)
        with patch.dict(os.environ, DSV4_MOE_FP4_BACKEND="typo"):
            with self.assertRaises(ValueError):
                grouped_fp4._sm120_fp4_backend()


@unittest.skipUnless(torch.cuda.is_available() and is_sm120(), "SM120 required")
class DeepGemmAlternativesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.env = patch.dict(
            os.environ,
            DSV4_SM120_FP8_LINEAR_BACKEND="default",
            DSV4_SM120_SHARED_EXPERT_BACKEND="default",
            DSV4_MOE_FP4_BACKEND="flashinfer",
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def linear_weights(self, n=256, k=512):
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.exp2(
            torch.randint(-4, 1, (n // 128, k // 128), device="cuda").float()
        ).to(torch.float8_e8m0fnu)
        return weight, scale

    def assert_bf16_error(self, actual, expected):
        self.assertTrue(torch.isfinite(actual).all().item())
        relative = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm().clamp_min(1e-10)
        self.assertLess(relative.item(), 0.004)
        torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.016)

    def capture_and_replay(self, fn, mutate):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = fn()
        torch.cuda.current_stream().wait_stream(stream)
        for mutation in mutate if isinstance(mutate, tuple) else (mutate,):
            mutation()
            expected = fn()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)

    def test_dense_quantization_output_and_graph(self):
        w, s = self.linear_weights()
        bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        linear = CudaFp8VllmBlockwiseLinear(w, s, bias=bias, backend="deepgemm")
        for rows in (0, 1, 3, 17, 127, 257):
            x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
            output = torch.full(
                (rows, 256), float("nan"), device="cuda", dtype=torch.bfloat16
            )
            self.assertEqual(linear(x, out=output).data_ptr(), output.data_ptr())
            if rows:
                expected = quantized_reference(linear, x).add_(bias)
                self.assert_bf16_error(output, expected)
        self.capture_and_replay(lambda: linear(x), lambda: x.neg_())
        with self.assertRaises(ValueError):
            linear(x[:, ::2])
        with self.assertRaises(ValueError):
            linear(x, out=torch.empty(257, 256, device="cuda"))
        with self.assertRaises(ValueError):
            CudaFp8VllmBlockwiseLinear(w, s.float(), backend="deepgemm")

    def test_default_and_independent_shared_selection(self):
        w, s = self.linear_weights()
        default = utils._v4_fp8_linear(w, s)
        baseline = CudaFp8VllmBlockwiseLinear(w, s)
        x = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(default(x), baseline(x), rtol=0, atol=0)
        w13, s13 = self.linear_weights(512, 512)
        w2, s2 = self.linear_weights(512, 256)
        weights = dict(w13_w=w13, w13_s=s13, w2_w=w2, w2_s=s2)
        with patch.dict(os.environ, DSV4_SM120_FP8_LINEAR_BACKEND="deepgemm"):
            dense = utils._v4_fp8_linear(w, s)
            shared = shared_expert.W13SharedExpert(512, 256, weights)
            self.assertEqual(dense.backend, "deepgemm")
            self.assertEqual(shared.w13.backend, "cutlass")
        with patch.dict(os.environ, DSV4_SM120_SHARED_EXPERT_BACKEND="deepgemm"):
            dense = utils._v4_fp8_linear(w, s)
            shared = shared_expert.W13SharedExpert(512, 256, weights, swiglu_limit=10)
            self.assertEqual(dense.backend, "cutlass")
            self.assertEqual(shared.w13.backend, "deepgemm")
            self.assertEqual(shared.w2.backend, "deepgemm")
        with patch.object(shared_expert, "_SM120_FUSED_ENABLED", True):
            self.assertFalse(shared_expert.FusedSharedExpertFastPath.can_run(shared, x))
        from rtp_llm.models_py.modules.dsv4.moe.expert import require_silu_mul_split

        gate, up = quantized_reference(shared.w13, x).float().chunk(2, dim=-1)
        hidden = require_silu_mul_split()(
            gate.contiguous(), up.contiguous(), clamp_limit=10
        )
        expected = quantized_reference(shared.w2, hidden.to(x.dtype))
        self.assert_bf16_error(shared(x), expected)
        self.capture_and_replay(lambda: shared(x), lambda: x.mul_(0.5))

    def routed(self):
        e, d, inter = 4, 512, 256
        cfg = MoeCfg(0, d, inter, e, 2, 10.0, 1, 0, e, 0, e, 1024)
        weights = {}
        for part, n, k in ((1, inter, d), (2, d, inter), (3, inter, d)):
            weights[getattr(W, f"v4_routed_w{part}_w")] = torch.randint(
                -128, 128, (e, n, k // 2), device="cuda", dtype=torch.int8
            )
            weights[getattr(W, f"v4_routed_w{part}_s")] = torch.full(
                (e, n, k // 32), 0.03125, device="cuda"
            ).to(torch.float8_e8m0fnu)
        fi = grouped_fp4.GroupedFP4Strategy(cfg)
        fi.setup_weights(dict(weights))
        with patch.dict(os.environ, DSV4_MOE_FP4_BACKEND="deepgemm"):
            dg = grouped_fp4.GroupedFP4Strategy(cfg)
            dg.setup_weights(dict(weights))
        return fi, dg

    def test_routed_same_quantization_and_dynamic_graph(self):
        import flashinfer

        fi, dg = self.routed()
        for rows in (0, 1, 17, 257):
            x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
            ids = torch.randint(0, 4, (rows, 2), device="cuda")
            weights = torch.rand(rows, 2, device="cuda")
            weights[::3, 1] = 0
            ids[::5, 0] = -1
            with patch.object(grouped_fp4, "_MOE_TILE_N", 64):
                expected = fi._forward_sm120_device_meta(x, weights, ids)
            actual = dg(x, weights, ids)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if rows:
                q, s = flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=False)
                s = s.reshape(rows, 16).view(torch.uint8)
                torch.testing.assert_close(
                    dg.forward_sm120_eager(q, weights, ids, s), expected, rtol=0, atol=0
                )
                real_gemm = dg._sm120_deepgemm

                def poisoned(a, b, out, *args, **kwargs):
                    out.fill_(float("nan"))
                    return real_gemm(a, b, out, *args, **kwargs)

                with patch.object(dg, "_sm120_deepgemm", poisoned):
                    torch.testing.assert_close(
                        dg(x, weights, ids), expected, rtol=0, atol=0
                    )
        self.capture_and_replay(
            lambda: dg.forward_sm120_eager(q, weights, ids, s),
            (
                lambda: (weights.mul_(0.7), ids.copy_(3 - ids.clamp_min(0))),
                lambda: (weights.zero_(), ids.fill_(-1)),
            ),
        )
        self.assertEqual(dg(x, weights, ids).count_nonzero().item(), 0)
        with self.assertRaises(ValueError):
            dg.forward_sm120_eager(q, weights, ids, s[:, :-1])


if __name__ == "__main__":
    unittest.main()
