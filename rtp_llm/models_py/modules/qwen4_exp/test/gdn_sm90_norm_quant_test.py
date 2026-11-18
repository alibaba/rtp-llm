"""SM90 FP32 scale, rounding, dispatch and concurrent-lifetime regressions."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_quant_gluon import (
    _launch_norm_quant,
    norm_quant,
)
from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton import (
    Qwen4PrefillGatedNorm,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

MODULE = "rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton.norm_quant"


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "requires SM90",
)
class GdnSm90NormQuantTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261117)

    def assert_bytes_equal(self, actual, expected):
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def build(self, dtype=torch.bfloat16):
        weight = (torch.randn(2560, 3072, device="cuda") * 0.02).to(torch.float8_e4m3fn)
        scales = torch.full((24, 20), 0.02, device="cuda", dtype=torch.float32)
        projection = CudaFp8GEMMLinear(
            weight.reshape(3072, 2560),
            weight_scales=scales,
            quant_config=Fp8BlockWiseQuantConfig(is_quanted=True),
        )
        norm = Qwen4PrefillGatedNorm(
            RmsNormGated(
                torch.randn(128, device="cuda", dtype=dtype),
                group_size=128,
                activation="sigmoid",
            ),
            role="PDFUSION",
        )
        return norm, projection

    @torch.inference_mode()
    def test_decode_fusion_preserves_legacy_fp8_divisions(self):
        norm, projection = self.build()
        for rows in (16, 17, 32, 63, 64):
            with self.subTest(rows=rows), patch.dict(
                os.environ, DSV4_FP8_QUANT_KERNEL="legacy"
            ):
                x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)
                gate = torch.randn(rows, 8192, device="cuda", dtype=torch.bfloat16)[
                    :, -3072:
                ]
                expected = projection(norm(x, gate))
                with patch(MODULE, wraps=_launch_norm_quant) as fused:
                    actual = norm.forward_project(x, gate, projection, decode=True)
                self.assertEqual(fused.call_count, 1)
                self.assert_bytes_equal(expected, actual)

    @torch.inference_mode()
    def test_decode_fallbacks_preserve_projection_and_hooks(self):
        norm, projection = self.build()
        prefill = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16)
        prefill_gate = torch.randn_like(prefill)
        self.assertTrue(norm._supports_prefill(prefill, prefill_gate))
        with patch(MODULE, side_effect=AssertionError("SM90 Prefill fused")):
            self.assert_bytes_equal(
                norm.forward_project(prefill, prefill_gate, projection),
                projection(norm(prefill, prefill_gate)),
            )
        x = torch.randn(32, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL="v2"), patch(
            MODULE, side_effect=AssertionError("v2 mode bypassed")
        ):
            self.assert_bytes_equal(
                norm.forward_project(x, gate, projection, decode=True),
                projection(norm(x, gate)),
            )
        calls = []
        hook = norm.register_forward_hook(lambda *args: calls.append(1))
        with patch(MODULE, side_effect=AssertionError("hook bypassed")):
            norm.forward_project(x, gate, projection, decode=True)
        hook.remove()
        self.assertEqual(calls, [1])

    @torch.inference_mode()
    def test_graph_and_two_streams_keep_independent_outputs(self):
        norm, projection = self.build(torch.float32)
        x = torch.randn(32, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(32, 8192, device="cuda", dtype=torch.bfloat16)[:, -3072:]
        for _ in range(3):
            norm.forward_project(x, gate, projection, decode=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = norm.forward_project(x, gate, projection, decode=True)
        for gain in (0.01, 1.0, 100.0):
            x.normal_(0, gain)
            gate.normal_()
            norm.weight.normal_()
            graph.replay()
            self.assert_bytes_equal(actual, projection(norm(x, gate)))
        graph.reset()
        values = [(torch.randn_like(x), torch.randn_like(gate)) for _ in range(2)]
        streams = [torch.cuda.Stream() for _ in values]
        outputs = []
        for stream, (value, gating) in zip(streams, values):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs.append(
                    norm.forward_project(value, gating, projection, decode=True)
                )
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for output, (value, gating) in zip(outputs, values):
            self.assert_bytes_equal(output, projection(norm(value, gating)))

    @torch.inference_mode()
    def test_fp8_input_projection_overlap_replays(self):
        from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
        from rtp_llm.models_py.modules.qwen4_exp.gdn_decode_fp8 import try_project_pair

        qkvz = CudaFp8GEMMLinear(
            torch.randn(8192, 2560, device="cuda")
            .to(torch.float8_e4m3fn)
            .reshape(2560, 8192),
            weight_scales=torch.full((20, 64), 0.02, device="cuda"),
            quant_config=Fp8BlockWiseQuantConfig(is_quanted=True),
        )
        ba = Fp8Linear(
            torch.randn(48, 2560, device="cuda").to(torch.float8_e4m3fn),
            torch.full((1,), 0.02, device="cuda"),
        )
        ba_stream = torch.cuda.Stream()
        native_ba = ba._run_quantized
        streams = []

        def run_ba(*args):
            streams.append(torch.cuda.current_stream())
            return native_ba(*args)

        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL="legacy"):
            for rows in (16, 32, 64):
                with self.subTest(rows=rows):
                    x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                    try_project_pair(x, qkvz, ba)
                    graph = torch.cuda.CUDAGraph()
                    with patch.object(ba, "_run_quantized", side_effect=run_ba):
                        with torch.cuda.graph(graph):
                            actual = try_project_pair(x, qkvz, ba, ba_stream=ba_stream)
                    self.assertEqual(streams.pop(), ba_stream)
                    for _ in range(3):
                        x.normal_()
                        expected = try_project_pair(x, qkvz, ba)
                        graph.replay()
                        for value, reference in zip(actual, expected):
                            self.assert_bytes_equal(value, reference)
                    graph.reset()

    @torch.inference_mode()
    def test_ue8m0_branch_and_nonfinite_values_retain_quantizer_semantics(self):
        norm, _ = self.build()
        x = torch.randn(2053, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        for value in (0.0, float("inf"), float("-inf"), float("nan")):
            x[0, 17] = value
            actual = norm_quant(x, gate, norm.weight, norm.eps)
            expected = sgl_per_token_group_quant_fp8(
                norm(x, gate),
                128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            for output, reference in zip(actual, expected):
                self.assert_bytes_equal(output, reference)

    @torch.inference_mode()
    def test_lengths_and_scale_strides_reuse_compilation(self):
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_quant_gluon import (
            _gdn_norm_quant,
        )

        norm, _ = self.build()
        hashes = []
        for rows in (2048, 2053, 4800, 5200):
            x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn(rows, 8192, device="cuda", dtype=torch.bfloat16)[
                :, -3072:
            ]
            q, scales = norm_quant(x, gate, norm.weight, norm.eps)
            kernel = _gdn_norm_quant.warmup(
                x,
                gate,
                norm.weight,
                q,
                scales,
                M=rows,
                N=3072,
                ZX=gate.stride(0),
                EPS=norm.eps,
                GROUPS=24,
                SS=scales.stride(1),
                GH=8,
                WARPS=1,
                UE8M0=True,
                num_warps=1,
                grid=((rows * 24 + 7) // 8,),
            )
            hashes.append(kernel.hash)
        self.assertEqual(len(set(hashes)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
