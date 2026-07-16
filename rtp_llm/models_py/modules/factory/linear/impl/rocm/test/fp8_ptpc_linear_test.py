import unittest
from unittest import SkipTest

import torch
import torch.nn.functional as F
from aiter import dtypes

from rtp_llm.models_py.utils.arch import is_hip

try:
    import aiter

    AITER_AVAILABLE = True
except ImportError:
    AITER_AVAILABLE = False


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def shuffle_weight(x: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    x_type = x.dtype
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size()
    BN = IN
    assert x.shape[-2] % BN == 0
    assert x.shape[-1] % BK == 0
    x_ = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous().view(*x.shape)
    return x_.view(x_type)


class RocmFp8PTPCLinearTest(unittest.TestCase):
    """PTPC fp8 linear单元测试，kernel weight 必须 shuffle。"""

    def setUp(self):
        if not is_hip():
            raise SkipTest("Test requires ROCm/HIP backend!")
        if not AITER_AVAILABLE:
            raise SkipTest("aiter required for RocmFp8PTPCLinear!")
        self.device = "cuda"
        self.hidden_size = 256
        self.output_size = 512
        self.batch_size = 32
        self.fp8_dtype = torch.float8_e4m3fnuz
        self.input_fp32 = torch.randn(
            self.batch_size, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )  # [B, K]
        self.weight_fp32 = torch.randn(
            self.output_size, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )  # [N, K]
        self.bias = torch.randn(
            self.output_size, dtype=torch.bfloat16, device=self.device
        )  # [N]

    def test_ptpc_fp8_forward(self):
        from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            RocmFp8PTPCLinear,
        )

        weight_q, weight_scales = rocm_per_token_quant_fp8(
            self.weight_fp32
        )  # weight_q: [N, K], weight_scales: [N, 1]

        weight_shuffle = shuffle_weight(weight_q, layout=(16, 16))  # [N, K]

        weight_for_init = weight_shuffle.T.contiguous()  # [K, N]

        weight_scales_for_init = weight_scales.T.contiguous()  # [1, N]

        ptpc_linear = RocmFp8PTPCLinear(
            weight=weight_for_init,  # [K, N]
            weight_scales=weight_scales_for_init,  # [1, N]
            bias=None,
        )

        ptpc_output = ptpc_linear(self.input_fp32)

        ref_input_bf16 = self.input_fp32

        quantization_eps = 1e-10
        ref_input_fp8, ref_input_scales = rocm_per_token_quant_fp8(
            ref_input_bf16,
            eps=quantization_eps,
        )
        ref_input_scales = ref_input_scales.to(torch.float32)

        ref_output = aiter.gemm_a8w8_bpreshuffle(
            ref_input_fp8,  # A_quant_tensor
            ptpc_linear.weight,  # W_kernel_tensor (使用 RocmFp8PTPCLinear 内部的 weight)
            ref_input_scales,  # A_quant_scale_tensor (M, 1)
            ptpc_linear.weight_scales,  # W_scale_tensor (使用 RocmFp8PTPCLinear 内部的 weight_scales)
            None,  # bias
            ref_input_bf16.dtype,  # output dtype (与 RocmFp8PTPCLinear.forward() 相同)
        )
        ref = ref_output

        # 5. 对比结果
        max_diff = (ptpc_output - ref).abs().max().item()
        mean_diff = (ptpc_output - ref).abs().mean().item()
        print(f"nobias max_diff: {max_diff:.6f}, mean_diff: {mean_diff:.6f}")
        self.assertLess(max_diff, 1e-5)
        self.assertLess(mean_diff, 1e-5)
        self.assertEqual(ptpc_output.shape, (self.batch_size, self.output_size))
        self.assertFalse(torch.isnan(ptpc_output).any())
        self.assertFalse(torch.isinf(ptpc_output).any())


if __name__ == "__main__":
    unittest.main()
