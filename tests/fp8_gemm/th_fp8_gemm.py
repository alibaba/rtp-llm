import os
import unittest

import torch


def random_tensor(shape, dtype, device, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)


class TestFp8Gemm(unittest.TestCase):
    FP8_E4M3_MAX = 448.0
    FP8_E4M3_MIN = -352.0

    def setUp(self) -> None:
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/libth_transformer.so"
        )
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )
        self.fp8_gemm = torch.ops.fp8_gemm_ops.fp8_gemm
        self.fp8_quant_gemm = torch.ops.fp8_gemm_ops.fp8_quant_gemm
        torch.manual_seed(734876213)

    @staticmethod
    def quantize_weight_to_fp8(weight: torch.Tensor):
        max_abs_value = weight.abs().max()
        scaling_factor = max_abs_value / TestFp8Gemm.FP8_E4M3_MAX
        min_scaling_factor = 1.0 / (TestFp8Gemm.FP8_E4M3_MAX * 512.0)
        scaling_factor = max(min_scaling_factor, scaling_factor)

        # 量化操作
        quantized_weight = (
            (weight / scaling_factor).to(torch.float8_e4m3fn).view(torch.int8)
        )
        return quantized_weight, scaling_factor

    @staticmethod
    def dequantize_weight_from_fp8(
        quantized_weight: torch.Tensor, scaling_factor: float
    ):
        # 反量化操作
        dequantized_weight = (
            quantized_weight.view(torch.float8_e4m3fn).to(torch.float32)
            * scaling_factor
        ).to(torch.float32)

        return dequantized_weight

    def gt_fp8_gem(self, mat1_quant, mat2_quant, scale_a_torch, scale_b_torch):
        mat_res = torch.mm(
            mat1_quant.view(torch.float8_e4m3fn).to(torch.float32),
            mat2_quant.view(torch.float8_e4m3fn).to(torch.float32),
        )
        return mat_res * (scale_a_torch * scale_b_torch)

    def fp8_gemm_helper(self, m, n, k):
        shape1 = (m, k)
        mat1 = torch.rand(shape1, dtype=torch.float16).contiguous().cuda()

        shape2 = (k, n)
        mat2 = torch.rand(shape2, dtype=torch.float16).contiguous().cuda()

        mat1_quant, mat1_scale = self.quantize_weight_to_fp8(mat1)
        mat2_quant, mat2_scale = self.quantize_weight_to_fp8(mat2)
        mat2_quant = mat2_quant.contiguous().cuda()

        scale_a_torch = mat1_scale.to(dtype=torch.float32).cuda()
        scale_b_torch = mat2_scale.to(dtype=torch.float32).cuda()

        ref = self.gt_fp8_gem(mat1_quant, mat2_quant, scale_a_torch, scale_b_torch)

        A = mat1_quant.contiguous().cuda()
        B = mat2_quant.view(torch.float8_e4m3fn).t().contiguous().cuda()

        output = self.fp8_quant_gemm(A, B, scale_a_torch, scale_b_torch)
        msg = (
            f"fp8 quant gemm Failed on m={m}, n={n}, k={k}, output={output}, ref={ref}"
        )
        torch.testing.assert_close(
            ref, output, rtol=0.001, atol=0.001, msg=msg, check_dtype=False
        )

        A = mat1.contiguous().cuda()
        output = self.fp8_gemm(A, B, scale_a_torch, scale_b_torch)
        msg = f"fp8 gemm Failed on m={m}, n={n}, k={k}, output={output}, ref={ref}"
        torch.testing.assert_close(
            ref, output, rtol=0.01, atol=0.04, msg=msg, check_dtype=False
        )

    def test_matmul(self):
        bs_list = [1]
        inseq_list = [1, 3, 48]
        hidden_size_list = [16, 48, 4352]

        for bs in bs_list:
            for inseq in inseq_list:
                for hidden_size in hidden_size_list:
                    self.fp8_gemm_helper(bs * inseq, 3 * hidden_size, hidden_size)
                    self.fp8_gemm_helper(bs * inseq, 4 * hidden_size, hidden_size)


if __name__ == "__main__":
    unittest.main()
