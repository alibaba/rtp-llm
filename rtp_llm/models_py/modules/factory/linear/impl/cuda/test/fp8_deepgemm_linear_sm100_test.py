import unittest

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.test.fp8_deepgemm_linear_test import (
    CudaFp8DeepGEMMLinearTestBase,
)
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.test.utils.numeric_util import calc_diff


class CudaFp8DeepGEMMLinearSM100Test(CudaFp8DeepGEMMLinearTestBase, unittest.TestCase):
    def test_sm100(self):
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())

    def test_k3_skip_head_mid_matches_regular_gemm_and_preserves_gap(self):
        head_splits = (128, 64, 128)
        left, middle, right = head_splits
        tokens = 257
        self.K = 512
        self.N = 96 // 8 * (left + right)
        self.scale_K = (self.K + 127) // 128
        self.scale_N = (self.N + 127) // 128
        self.weight = torch.randn(
            self.K, self.N, dtype=torch.bfloat16, device=self.device
        ).to(torch.float8_e4m3fn)
        self.weight_scales = torch.rand(
            self.scale_K,
            self.scale_N,
            dtype=torch.float32,
            device=self.device,
        )
        linear = self._create_cuda_fp8_deepgemm_linear()
        inputs = torch.randn(tokens, self.K, dtype=torch.bfloat16, device=self.device)
        values, scales = linear.quantize_input(inputs)
        scale_wire = torch.empty(
            ((self.K + 511) // 512, (tokens + 3) // 4 * 4),
            dtype=torch.int32,
            device=self.device,
        )
        scale_wire[:, :tokens].copy_(scales.T)
        activation = QuantizedActivation(values, scale_wire)
        heads = self.N // (left + right)
        output = torch.full(
            (tokens, heads * sum(head_splits)),
            11.0,
            dtype=torch.bfloat16,
            device=self.device,
        )

        expected = linear(activation).view(tokens, heads, -1)
        actual = linear.forward_skip_head_mid(
            activation, head_splits, output=output
        ).view(tokens, heads, -1)

        self.assertLess(calc_diff(actual[..., :left], expected[..., :left]), 1e-5)
        self.assertLess(calc_diff(actual[..., -right:], expected[..., left:]), 1e-5)
        self.assertTrue(torch.all(actual[..., left : left + middle] == 11.0))


if __name__ == "__main__":
    unittest.main()
