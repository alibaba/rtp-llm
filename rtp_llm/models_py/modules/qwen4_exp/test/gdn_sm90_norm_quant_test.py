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
        with patch.dict(os.environ, RTP_QWEN4_SM90_GDN_NORM_QUANT="1"):
            norm = Qwen4PrefillGatedNorm(
                RmsNormGated(
                    torch.randn(128, device="cuda", dtype=dtype),
                    group_size=128,
                    activation="sigmoid",
                ),
                role="PREFILL",
            )
        return norm, projection

    @torch.inference_mode()
    def test_fp8_bytes_and_fp32_scales_match_native(self):
        for dtype in (torch.bfloat16, torch.float32):
            norm, projection = self.build(dtype)
            for rows in (2048, 2053, 4800, 5200, 8192, 32768):
                x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)
                gate = torch.randn(rows, 8192, device="cuda", dtype=torch.bfloat16)[
                    :, -3072:
                ]
                for gain in (0.0, 1e-10, 1.0, 100.0):
                    x.normal_().mul_(gain)
                    actual = norm_quant(
                        x, gate, norm.weight, norm.eps, warps=2, scale_ue8m0=False
                    )
                    expected = projection._deepgemm_linear.quantize_input(norm(x, gate))
                    for value, reference in zip(actual, expected):
                        self.assert_bytes_equal(value, reference)

    @torch.inference_mode()
    def test_factory_dispatch_and_complete_projection(self):
        norm, projection = self.build()
        for rows in (2048, 2053, 4800, 5200, 8192, 32768):
            x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn_like(x)
            expected = projection(norm(x, gate))
            with patch(MODULE, wraps=norm_quant) as fused:
                actual = norm.forward_project(x, gate, projection)
            self.assertEqual(fused.call_count, 1)
            self.assert_bytes_equal(actual, expected)

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
    def test_projection_ranges_tails_and_explicit_output(self):
        _, projection = self.build()
        projection.flashinfer_m_ranges = ((32, 128), (5121, 5376))
        for rows in (31, 32, 33, 64, 65, 127, 128, 129, 5120, 5121, 5249, 5376, 5377):
            value = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)
            selected = rows < 32 or 32 <= rows <= 128 or 5121 <= rows <= 5376
            self.assertEqual(projection._should_use_flashinfer(value), selected)
            expected = projection._deepgemm_linear(value)
            self.assert_bytes_equal(projection(value), expected)
            output = torch.empty_like(expected)
            actual = projection(value, out=output)
            self.assertEqual(actual.data_ptr(), output.data_ptr())
            self.assert_bytes_equal(actual, expected)
        projection.flashinfer_m_ranges = ()
        value = torch.randn(65, 3072, device="cuda", dtype=torch.bfloat16)
        self.assertFalse(projection._should_use_flashinfer(value))
        self.assertFalse(projection._should_use_flashinfer(value.float()))

    @torch.inference_mode()
    def test_default_legacy_hooks_and_backend_selection_fallback(self):
        norm, projection = self.build()
        x = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        norm.sm90_norm_quant = False
        with patch(MODULE, side_effect=AssertionError("disabled candidate executed")):
            self.assert_bytes_equal(
                norm.forward_project(x, gate, projection), projection(norm(x, gate))
            )
        norm.sm90_norm_quant = True
        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL="legacy"), patch(
            MODULE, side_effect=AssertionError("legacy mode bypassed")
        ):
            self.assert_bytes_equal(
                norm.forward_project(x, gate, projection), projection(norm(x, gate))
            )
        calls = []
        hook = norm.register_forward_hook(lambda *args: calls.append(1))
        with patch(MODULE, side_effect=AssertionError("hook bypassed")):
            norm.forward_project(x, gate, projection)
        hook.remove()
        self.assertEqual(calls, [1])
        projection.flashinfer_m_ranges = ((2048, 2048),)
        expected = projection(norm(x, gate))
        with patch(MODULE, wraps=norm_quant) as fused:
            actual = norm.forward_project(x, gate, projection)
        self.assertEqual(fused.call_count, 1)
        self.assert_bytes_equal(actual, expected)

    @torch.inference_mode()
    def test_graph_and_two_streams_keep_independent_outputs(self):
        norm, projection = self.build(torch.float32)
        projection.flashinfer_m_ranges = ((2053, 2053),)
        x = torch.randn(2053, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(2053, 8192, device="cuda", dtype=torch.bfloat16)[:, -3072:]
        for _ in range(3):
            norm.forward_project(x, gate, projection)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = norm.forward_project(x, gate, projection)
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
                outputs.append(norm.forward_project(value, gating, projection))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for output, (value, gating) in zip(outputs, values):
            self.assert_bytes_equal(output, projection(norm(value, gating)))

    @torch.inference_mode()
    def test_ue8m0_branch_and_nonfinite_values_retain_quantizer_semantics(self):
        norm, _ = self.build()
        x = torch.randn(2053, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        for value in (0.0, float("inf"), float("-inf"), float("nan")):
            x[0, 17] = value
            for packed in (False, True):
                actual = norm_quant(
                    x,
                    gate,
                    norm.weight,
                    norm.eps,
                    warps=1 if packed else 2,
                    scale_ue8m0=packed,
                )
                expected = sgl_per_token_group_quant_fp8(
                    norm(x, gate),
                    128,
                    eps=1e-4,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=packed,
                )
                for output, reference in zip(actual, expected):
                    self.assert_bytes_equal(output, reference)

    @torch.inference_mode()
    def test_gdn_static_projection_dispatch_and_multistep_state(self):
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
        from rtp_llm.models_py.modules.qwen4_exp.test.gdn_composition_test import (
            GdnDecodeConvGateSM90Test,
        )
        from rtp_llm.utils.model_weight import W

        helper = GdnDecodeConvGateSM90Test
        template, _ = helper.make_pair(tp=2)
        weights = dict(template.weights)
        for name, scale_name, columns, inner in (
            (W.linear_attn_qkvz_w, W.linear_attn_qkvz_s, 8192, 2560),
            (W.linear_attn_out_w, W.linear_attn_out_s, 2560, 3072),
        ):
            value = torch.randn(columns, inner, device="cuda").to(torch.float8_e4m3fn)
            weights[name] = value.reshape(inner, columns)
            weights[scale_name] = torch.full(
                (inner // 128, columns // 128), 0.02, device="cuda"
            )
        layers = []
        for enabled in ("0", "1"):
            with patch.dict(os.environ, RTP_QWEN4_SM90_GDN_FP8_PROJECTION=enabled):
                layer = Qwen3NextGatedDeltaNet(
                    template.linear_attn_config,
                    template.parallelism_config,
                    dict(weights),
                    1e-6,
                    Fp8BlockWiseQuantConfig(is_quanted=True),
                    norm_activation="sigmoid",
                ).eval()
            layer.tp_all_reduce = lambda value, group: value
            layers.append(layer)
        self.assertEqual(layers[0].out_proj.flashinfer_m_ranges, ())
        self.assertEqual(
            layers[1].out_proj.flashinfer_m_ranges, ((32, 128), (5121, 5376))
        )
        for batch, steps in (
            (32, 1),
            (33, 1),
            (64, 1),
            (65, 1),
            (128, 1),
            (8, 4),
            (16, 4),
            (31, 4),
        ):
            hidden, initial, inputs, metadata = helper.make_decode_case(
                layers[0], batch, tokens=steps
            )
            initial = initial.repeat(2, 1)
            inputs.kv_cache_kernel_block_id_device = torch.arange(
                1, batch * (steps + 4) + 1, device="cuda", dtype=torch.int32
            ).view(batch, steps + 4)
            self.assertFalse(layers[0].in_proj_qkvz._should_use_flashinfer(hidden))
            self.assertEqual(
                layers[1].in_proj_qkvz._should_use_flashinfer(hidden),
                32 <= hidden.shape[0] <= 64,
            )
            states = [initial.clone(), initial.clone()]
            for _ in range(16):
                reference = helper.run_decode(
                    layers[0], hidden, states[0], inputs, metadata
                )
                actual = helper.run_decode(
                    layers[1], hidden, states[1], inputs, metadata
                )
                self.assert_bytes_equal(actual, reference)
                self.assert_bytes_equal(states[1], states[0])
                hidden.mul_(1.01)
                inputs.sequence_lengths_plus_1_device.add_(steps)
            inputs.is_cuda_graph = True
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = helper.run_decode(
                    layers[1], hidden, states[1], inputs, metadata
                )
            for _ in range(3):
                hidden.normal_()
                states[0].copy_(initial)
                states[1].copy_(initial)
                reference = helper.run_decode(
                    layers[0], hidden, states[0], inputs, metadata
                )
                graph.replay()
                self.assert_bytes_equal(actual, reference)
                self.assert_bytes_equal(states[1], states[0])
            graph.reset()

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
            q, scales = norm_quant(
                x, gate, norm.weight, norm.eps, warps=2, scale_ue8m0=False
            )
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
                WARPS=2,
                UE8M0=False,
                LEGACY=False,
                num_warps=2,
                grid=((rows * 24 + 7) // 8,),
            )
            hashes.append(kernel.hash)
        self.assertEqual(len(set(hashes)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
