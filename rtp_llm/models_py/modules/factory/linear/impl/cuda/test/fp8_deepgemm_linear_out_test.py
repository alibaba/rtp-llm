import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda import (
    fp8_deepgemm_linear as deepgemm_linear_mod,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)


def _make_linear_for_forward(bias: torch.Tensor | None = None):
    linear = CudaFp8DeepGEMMLinear.__new__(CudaFp8DeepGEMMLinear)
    torch.nn.Module.__init__(linear)
    linear.K = 4
    linear.N = 3
    linear.weight = torch.empty(linear.N, linear.K, dtype=torch.bfloat16)
    linear.weight_scales = torch.empty(1, 1, dtype=torch.float32)
    linear.bias = bias
    linear.scale_ue8m0 = False
    linear.cached_scales = None
    linear.cached_scales_max_len = 0
    return linear


class CudaFp8DeepGEMMLinearOutContractTest(unittest.TestCase):
    def test_forward_passes_out_buffer_to_gemm_and_returns_it(self):
        bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        linear = _make_linear_for_forward(bias=bias)
        input_tensor = torch.ones(2, linear.K, dtype=torch.bfloat16)
        out = torch.empty(2, linear.N, dtype=torch.bfloat16)
        gemm_output = torch.arange(6, dtype=torch.float32).view(2, 3).to(torch.bfloat16)
        gemm_calls = []

        def fake_quant(x, **kwargs):
            return x, torch.ones(x.shape[0], 1, dtype=torch.float32)

        def fake_fp8_gemm_nt(input_pair, weight_pair, output, c=None, **kwargs):
            gemm_calls.append((input_pair, weight_pair, output, c, kwargs))
            output.copy_(gemm_output)

        with patch.object(
            deepgemm_linear_mod,
            "sgl_per_token_group_quant_fp8",
            side_effect=fake_quant,
        ), patch.object(
            deepgemm_linear_mod,
            "fp8_gemm_nt",
            side_effect=fake_fp8_gemm_nt,
        ):
            returned = linear(input_tensor, out=out)

        self.assertIs(returned, out)
        self.assertEqual(len(gemm_calls), 1)
        self.assertIs(gemm_calls[0][2], out)
        self.assertIsNone(gemm_calls[0][3])
        self.assertTrue(gemm_calls[0][4]["disable_ue8m0_cast"])
        self.assertTrue(torch.equal(out, gemm_output + bias))

    def test_forward_rejects_invalid_out_buffer(self):
        linear = _make_linear_for_forward()
        input_tensor = torch.ones(2, linear.K, dtype=torch.bfloat16)

        with self.assertRaisesRegex(ValueError, "Output tensor shape"):
            linear(input_tensor, out=torch.empty(2, linear.N + 1, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "Output tensor dtype"):
            linear(input_tensor, out=torch.empty(2, linear.N, dtype=torch.float32))

        non_contiguous = torch.empty(linear.N, 2, dtype=torch.bfloat16).t()
        self.assertEqual(tuple(non_contiguous.shape), (2, linear.N))
        self.assertFalse(non_contiguous.is_contiguous())
        with self.assertRaisesRegex(ValueError, "Output tensor must be contiguous"):
            linear(input_tensor, out=non_contiguous)


if __name__ == "__main__":
    unittest.main()
