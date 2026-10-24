import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.kernels.cuda.fp8_kernel import quant_weight_ue8m0_packed
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_quant_gluon import norm_quant
from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton import (
    Qwen4PrefillGatedNorm,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GdnNormQuantTest(unittest.TestCase):
    def build(
        self, width=3072, role="PREFILL", weight_dtype=torch.bfloat16, factory=False
    ):
        torch.manual_seed(580 + width)
        weight, scale = quant_weight_ue8m0_packed(
            torch.randn(2560, width, device="cuda") * 0.02
        )
        if factory:
            projection = LinearFactory.create_linear(
                weight=weight,
                weight_scales=scale,
                bias=None,
                quant_config=Fp8BlockWiseQuantConfig(is_quanted=True),
            )
            self.assertIs(type(projection), CudaFp8GEMMLinear)
        else:
            projection = CudaFp8DeepGEMMLinear(weight, weight_scales=scale)
        norm = Qwen4PrefillGatedNorm(
            RmsNormGated(
                torch.randn(128, device="cuda", dtype=weight_dtype),
                group_size=128,
                activation="sigmoid",
            ),
            role=role,
        )
        return norm, projection

    def test_complete_projection_and_packed_scales_exact_across_domain(self):
        for width in (1536, 3072):
            norm, projection = self.build(width)
            for rows in (2048, 2053, 4093, 4096, 8191, 8192):
                x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                gate = torch.randn(
                    rows, width * 8 // 3, device="cuda", dtype=torch.bfloat16
                )[:, -width:]
                q, s = norm_quant(x, gate, norm.weight, norm.eps)
                refq, refs = projection.quantize_input(norm(x, gate))
                self.assertTrue(
                    torch.equal(q.view(torch.uint8), refq.view(torch.uint8))
                )
                self.assertTrue(torch.equal(s, refs))
                self.assertTrue(
                    torch.equal(
                        norm.forward_project(x, gate, projection),
                        projection(norm(x, gate)),
                    )
                )

    def test_graph_replay_reads_live_values_gate_and_gamma(self):
        norm, projection = self.build(weight_dtype=torch.float32)
        x = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(2048, 8192, device="cuda", dtype=torch.bfloat16)[:, -3072:]
        for _ in range(3):
            norm.forward_project(x, gate, projection)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = norm.forward_project(x, gate, projection)
        for gain in (0.0, 1.0e-6, 0.01, 1.0, 100.0):
            x.normal_(0, gain)
            gate.normal_()
            norm.weight.normal_()
            graph.replay()
            self.assertTrue(torch.equal(actual, projection(norm(x, gate))))
        graph.reset()

    def test_nonfinite_quantization_matches_native_v2(self):
        norm, projection = self.build()
        x = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        for field in (x, gate, norm.weight):
            saved = field.clone()
            for value in (float("inf"), float("-inf"), float("nan")):
                field.reshape(-1)[17] = value
                q, s = norm_quant(x, gate, norm.weight, norm.eps)
                refq, refs = projection.quantize_input(norm(x, gate))
                self.assertTrue(
                    torch.equal(q.view(torch.uint8), refq.view(torch.uint8))
                )
                self.assertTrue(torch.equal(s, refs))
            field.copy_(saved)

    def test_decode_legacy_small_and_hooks_preserve_original_path(self):
        norm, projection = self.build()
        x = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        module = (
            "rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton.norm_quant"
        )
        for role, quant_mode, rows in (
            ("DECODE", "auto", 2048),
            ("PREFILL", "legacy", 2048),
            ("PREFILL", "auto", 16),
        ):
            norm.role = role
            with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=quant_mode), patch(
                module, side_effect=AssertionError("unexpected fused path")
            ):
                self.assertTrue(
                    torch.equal(
                        norm.forward_project(x[:rows], gate[:rows], projection),
                        projection(norm(x[:rows], gate[:rows])),
                    )
                )
        norm.role = "PREFILL"
        calls = []
        hook = norm.register_forward_hook(lambda *args: calls.append("norm"))
        with patch(module, side_effect=AssertionError("hook must be honored")):
            norm.forward_project(x, gate, projection)
        hook.remove()
        self.assertEqual(calls, ["norm"])

    def test_production_factory_dispatcher_enters_fusion_with_exact_output(self):
        module = (
            "rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton.norm_quant"
        )
        for width, rows in ((1536, 4093), (1536, 4096), (3072, 2048), (3072, 4096)):
            norm, projection = self.build(width, factory=True)
            x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn(
                rows, width * 8 // 3, device="cuda", dtype=torch.bfloat16
            )[:, -width:]
            expected = projection(norm(x, gate))
            with patch(module, wraps=norm_quant) as fused:
                actual = norm.forward_project(x, gate, projection)
            self.assertEqual(fused.call_count, 1)
            self.assertTrue(torch.equal(actual, expected))

    def test_dispatcher_graph_replay_preserves_live_inputs(self):
        norm, projection = self.build(factory=True)
        x = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(2048, 8192, device="cuda", dtype=torch.bfloat16)[:, -3072:]
        for _ in range(3):
            norm.forward_project(x, gate, projection)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = norm.forward_project(x, gate, projection)
        for gain in (0.01, 0.5, 2.0):
            x.normal_(0, gain)
            gate.normal_()
            norm.weight.normal_()
            graph.replay()
            self.assertTrue(torch.equal(actual, projection(norm(x, gate))))
        graph.reset()

    def test_dispatcher_and_backend_hooks_and_overrides_preserve_fallback(self):
        norm, projection = self.build(factory=True)
        x = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        module = (
            "rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton.norm_quant"
        )
        for owner in (projection, projection._deepgemm_linear):
            calls = []
            hook = owner.register_forward_hook(lambda *args: calls.append(1))
            with patch(module, side_effect=AssertionError("hook must run")):
                norm.forward_project(x, gate, projection)
            hook.remove()
            self.assertEqual(calls, [1])
            with patch.object(owner, "forward", wraps=owner.forward) as forward, patch(
                module, side_effect=AssertionError("override must run")
            ):
                norm.forward_project(x, gate, projection)
                self.assertEqual(forward.call_count, 1)
        # Preserve backend selection even if its threshold changes in a future
        # platform implementation. The model-local fusion must not force it.
        with patch.object(
            projection, "_should_use_flashinfer", return_value=True
        ), patch.object(projection, "_flashinfer_linear", torch.nn.Identity()), patch(
            module, side_effect=AssertionError("selected backend must run")
        ):
            self.assertTrue(
                torch.equal(norm.forward_project(x, gate, projection), norm(x, gate))
            )


if __name__ == "__main__":
    unittest.main()
