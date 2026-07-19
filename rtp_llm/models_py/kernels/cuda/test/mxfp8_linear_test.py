import os
from unittest import SkipTest, TestCase, main

# DeepGEMM invokes NVCC with paths relative to HOME. Bazel does not provide
# HOME, so use its per-test writable directory before importing CUDA modules.
_TEST_HOME = os.environ.get("TEST_TMPDIR", "/tmp/rtp_llm_mxfp8_linear_test")
_TORCH_KERNEL_CACHE = os.path.join(_TEST_HOME, "torch")
os.makedirs(_TORCH_KERNEL_CACHE, exist_ok=True)
os.environ.setdefault("HOME", _TEST_HOME)
os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", _TORCH_KERNEL_CACHE)

import torch

from rtp_llm.test.utils.numeric_util import calc_diff


class Mxfp8LinearTest(TestCase):
    M_VALUES = (1, 2, 6, 96)
    N = 512
    K = 512

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        try:
            import deep_gemm  # noqa: F401
            import flashinfer  # noqa: F401
        except Exception as error:
            raise SkipTest(f"flashinfer/deep_gemm unavailable: {error}")

        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (
            mxfp8_quant_act_eager,
            mxfp8_quant_act_packed,
        )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import (
            CudaMxfp8Linear,
        )

        torch.manual_seed(20260713)
        cls.weight_bf16 = torch.randn(cls.N, cls.K, device="cuda", dtype=torch.bfloat16)
        cls.weight, cls.weight_scale = mxfp8_quant_act_packed(cls.weight_bf16)
        external_weight, external_weight_scale = mxfp8_quant_act_eager(cls.weight_bf16)
        cls.external_scale_linear = CudaMxfp8Linear(
            external_weight, external_weight_scale
        )

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_linear

        return mxfp8_linear(x, self.weight, self.weight_scale)

    def test_batched_shapes(self) -> None:
        for m_value in self.M_VALUES:
            with self.subTest(m=m_value):
                x = torch.randn(m_value, self.K, device="cuda", dtype=torch.bfloat16)
                actual = self._run(x)
                expected = torch.matmul(x, self.weight_bf16.t())
                self.assertLess(calc_diff(actual, expected), 1e-3, f"M={m_value}")

    def test_external_scale_batched_shapes(self) -> None:
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_eager

        for m_value in self.M_VALUES:
            with self.subTest(m=m_value):
                x = torch.randn(m_value, self.K, device="cuda", dtype=torch.bfloat16)
                input_fp8, input_scale = mxfp8_quant_act_eager(x)
                actual = self.external_scale_linear(input_fp8, input_scales=input_scale)
                expected = torch.matmul(x, self.weight_bf16.t())
                self.assertLess(calc_diff(actual, expected), 1e-3, f"M={m_value}")

    def test_external_scale_batched_shape_cuda_graph(self) -> None:
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_eager

        x = torch.randn(self.M_VALUES[-1], self.K, device="cuda", dtype=torch.bfloat16)
        input_fp8, input_scale = mxfp8_quant_act_eager(x)
        eager_output = self.external_scale_linear(input_fp8, input_scales=input_scale)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = self.external_scale_linear(
                input_fp8, input_scales=input_scale
            )

        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)

    def test_batched_shape_cuda_graph(self) -> None:
        x = torch.randn(self.M_VALUES[-1], self.K, device="cuda", dtype=torch.bfloat16)

        eager_output = self._run(x)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = self._run(x)

        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)


if __name__ == "__main__":
    main()
