import os
import unittest

import torch


def random_tensor(shape, dtype, device, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)


class FastGELUActivation(object):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    @staticmethod
    def impl(input: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input))
            )
        )


class TestInt8Gemm(unittest.TestCase):
    def setUp(self) -> None:
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/libth_transformer.so"
        )
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )
        self.int8_gemm = torch.ops.int8_gemm_ops.int8_gemm
        self.gemm_config_select = torch.ops.int8_gemm_ops.int8_gemm_config_select

        torch.manual_seed(734876213)

    def gt_matmul_smooth_quant(self, mat1, mat2, scale_a_, scale_b_, bias):
        # Convert to int32 for PyTorch GT Matmul with accumulation in int32.
        mat1 = mat1.to(dtype=torch.int32)
        # Transpose the second matrix to support the native PyTorch format
        mat2 = mat2.cuda().transpose(0, 1).to(dtype=torch.int32)
        # Do matmul
        ref = torch.matmul(mat1.cpu(), mat2.cpu())

        m = 1
        for ii in range(len(mat1.shape) - 1):
            m *= mat1.shape[ii]
        n = mat2.shape[1]

        # Prepare per element scaling
        scale_a = scale_a_.expand((m, 1))
        scale_b = scale_b_.expand((1, n))
        scaling = torch.matmul(scale_a.cuda(), scale_b.cuda()).reshape(ref.shape)
        # Scale output and cast to right type
        ref = ref.cuda() * scaling.cuda()
        if bias is not None:
            ref = ref + bias
        # ref = FastGELUActivation.impl(ref)

        return ref

    def int8_gemm_helper(self, m, n, k, has_bias=False):
        # Init operands for multiplication in int32
        shape1 = (m, k)
        mat1 = torch.randint(-128, 128, shape1, dtype=torch.int8).cuda()
        shape2 = (n, k)
        mat2 = torch.randint(-128, 128, shape2, dtype=torch.int8).cuda()

        # Init scales in fp32
        shape_scale_a = (m, 1)
        scale_a_torch = torch.ones(shape_scale_a, dtype=torch.float32) * 1e-2
        scale_a_torch *= torch.randint(1, 10, shape_scale_a, dtype=torch.float32)
        scale_a_torch = scale_a_torch.cuda()

        shape_scale_b = (1, n)
        scale_b_torch = torch.ones(shape_scale_b, dtype=torch.float32) * 1e-2
        scale_b_torch *= torch.randint(1, 10, shape_scale_b, dtype=torch.float32)
        scale_b_torch = scale_b_torch.cuda()

        if has_bias:
            bias = 2 * torch.rand([n]) - 1
            bias = bias.half().cuda()
        else:
            bias = None

        output = self.int8_gemm(mat1, mat2, scale_b_torch, scale_a_torch, bias)

        ref = self.gt_matmul_smooth_quant(
            mat1, mat2, scale_a_torch, scale_b_torch, bias
        )
        msg = "int8 gemm Failed on m={}, n={}, k={}".format(m, n, k)
        torch.testing.assert_close(
            ref, output, rtol=0.001, atol=0.001, msg=msg, check_dtype=False
        )

    def test_matmul(self):
        bs_list = [1]
        inseq_list = [1]
        hidden_size_list = [4096]
        for bs in bs_list:
            for inseq in inseq_list:
                for hidden_size in hidden_size_list:
                    self.int8_gemm_helper(bs * inseq, 3 * hidden_size, hidden_size)
                    self.int8_gemm_helper(
                        bs * inseq, 3 * hidden_size, hidden_size, True
                    )
                    self.int8_gemm_helper(bs * inseq, 4 * hidden_size, hidden_size)
                    self.int8_gemm_helper(
                        bs * inseq, 4 * hidden_size, hidden_size, True
                    )


if __name__ == "__main__":
    unittest.main()
