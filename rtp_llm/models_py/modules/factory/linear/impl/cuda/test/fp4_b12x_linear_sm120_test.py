import importlib
import os
import unittest
from importlib.metadata import distribution
from pathlib import Path
from unittest.mock import patch

import torch
from flashinfer import fp4_quantize, mm_fp4
from rtp_llm.models_py.kernels.cuda import fp4_b12x
from rtp_llm.models_py.modules.factory.linear.impl.cuda import fp4_linear


class B12xLinearSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This target is routed only to CUDA13 SM120 workers. A missing runtime
        # or a silently disabled plan must fail rather than count as coverage.
        if not torch.cuda.is_available():
            raise RuntimeError("b12x regression requires a CUDA13 SM120 worker")
        if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
            raise RuntimeError("b12x regression requires SM120 or SM121")
        cls.n = max(
            8192, 128 * torch.cuda.get_device_properties(0).multi_processor_count
        )
        cls.k = 1024

    def test_cuda13_cute_payload_is_consistent(self):
        from cutlass.base_dsl.version_info import CUDA_VERSION

        self.assertGreaterEqual(CUDA_VERSION.major, 13)
        root = (
            distribution("nvidia-cutlass-dsl-libs-cu13")
            .locate_file("nvidia_cutlass_dsl/python_packages")
            .resolve()
        )
        for name in (
            "cutlass",
            "cutlass._mlir.dialects._cute_ops_gen",
            "cutlass._mlir.dialects._cute_nvgpu_ops_gen",
            "cutlass._mlir._mlir_libs._cutlass_ir",
        ):
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertTrue(Path(module.__file__).resolve().is_relative_to(root))

    def make_layer(self, *, backend="b12x", n=None, bias=True):
        n = self.n if n is None else n
        weight = torch.randn(n, self.k, device="cuda", dtype=torch.bfloat16) * 0.1
        inv_weight_scale = torch.tensor(64.0, device="cuda")
        packed, scales = fp4_quantize(weight, inv_weight_scale)
        with patch.dict(os.environ, {"RTP_LLM_FP4_GEMM_BACKEND": backend}):
            return fp4_linear.CudaFp4GEMMLinear(
                packed,
                scales,
                torch.tensor(1 / 32, device="cuda"),
                torch.randn(n, device="cuda", dtype=torch.bfloat16) if bias else None,
                weight_scale_2=1 / inv_weight_scale,
            )

    def reference(self, layer, x):
        a, sa = fp4_quantize(x, layer.input_scale_inv)
        y = mm_fp4(
            a,
            layer.weight.T,
            sa,
            layer.weight_scales.T,
            layer.alpha,
            x.dtype,
            backend=layer.backend,
        )
        return y if layer.bias is None else y + layer.bias.to(y.dtype)

    def test_real_plan_and_fallback_boundaries(self):
        layer = self.make_layer()
        plan = layer._b12x_min_m_plan
        self.assertIsNotNone(
            plan, "FlashInfer ABI drift disabled the tested optimization"
        )
        for dtype in (torch.bfloat16, torch.float16):
            for m in (1, 3, plan.min_m - 1, plan.min_m, plan.min_m + 1, 127, 128):
                with self.subTest(dtype=dtype, m=m):
                    x = torch.randn(m, self.k, device="cuda", dtype=dtype) * 0.1
                    with patch.object(fp4_linear, "mm_fp4", wraps=mm_fp4) as public:
                        actual = layer(x)
                    self.assertEqual(
                        public.call_count,
                        int(dtype != torch.bfloat16 or m >= plan.min_m),
                    )
                    torch.testing.assert_close(
                        actual, self.reference(layer, x), rtol=0, atol=0
                    )

    def test_invalid_k_is_rejected_before_quantization(self):
        layer = self.make_layer()
        for shape in ((1, self.k + 128), (1, self.k - 128), (self.k,), (1, 1, self.k)):
            with (
                self.subTest(shape=shape),
                patch.object(fp4_linear, "fp4_quantize") as quant,
            ):
                with self.assertRaisesRegex(ValueError, "expects"):
                    layer(torch.empty(shape, device="cuda", dtype=torch.bfloat16))
                quant.assert_not_called()

    @torch.inference_mode()
    def test_cache_reuses_plan_without_binding_weights_or_alpha(self):
        first, second = self.make_layer(), self.make_layer()
        self.assertIsNotNone(first._b12x_min_m_plan)
        self.assertIs(first._b12x_min_m_plan, second._b12x_min_m_plan)
        second.alpha.mul_(2)
        x = torch.randn(3, self.k, device="cuda", dtype=torch.bfloat16) * 0.1
        for layer in (first, second):
            torch.testing.assert_close(
                layer(x), self.reference(layer, x), rtol=0, atol=0
            )

    def test_unsupported_plan_and_default_backend_use_public_path(self):
        with (
            patch.object(
                fp4_b12x, "_build_plan", side_effect=ValueError("unsupported ABI")
            ),
            patch.dict(fp4_b12x._PLANS, clear=True),
        ):
            layer = self.make_layer()
            self.assertIsNone(layer._b12x_min_m_plan)
            x = torch.randn(3, self.k, device="cuda", dtype=torch.bfloat16) * 0.1
            with patch.object(fp4_linear, "mm_fp4", wraps=mm_fp4) as public:
                torch.testing.assert_close(
                    layer(x), self.reference(layer, x), rtol=0, atol=0
                )
            self.assertGreater(public.call_count, 0)
        with patch.object(fp4_linear, "create_b12x_min_m_plan") as factory:
            default = self.make_layer(backend="cutlass")
            factory.assert_not_called()
        torch.testing.assert_close(
            default(x), self.reference(default, x), rtol=0, atol=0
        )
        self.assertIsNone(self.make_layer(n=128)._b12x_min_m_plan)

    @torch.inference_mode()
    def test_cuda_graph_replays_current_input_and_alpha(self):
        layer = self.make_layer()
        self.assertIsNotNone(layer._b12x_min_m_plan)
        x = torch.randn(3, self.k, device="cuda", dtype=torch.bfloat16) * 0.1
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            layer(x)
            layer(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = layer(x)
        for _ in range(3):
            x.normal_(std=0.1)
            layer.alpha.mul_(0.75)
            graph.replay()
            torch.testing.assert_close(actual, self.reference(layer, x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
