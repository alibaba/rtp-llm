"""sm12x-only numerical sanity tests for CudaFp8VllmBlockwiseLinear.

Quantizes a BF16 weight with per_block_cast_to_fp8 (block 128x128), runs
the kernel and compares against a fp32 reference matmul (+ optional bias).
Catches regressions in the three M-tier dispatch branches
(swap_ab / pingpong / default) and bias epilogue fusion.
"""

import unittest

import torch

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
)
from rtp_llm.models_py.utils.arch import is_sm12x
from rtp_llm.test.utils.numeric_util import calc_diff, per_block_cast_to_fp8


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm12x(),
    "CudaFp8VllmBlockwiseLinear requires sm_120 (consumer Blackwell)",
)
class CudaFp8VllmBlockwiseLinearNumericalTest(unittest.TestCase):

    K = 256
    N = 256
    # Boundary values around dispatch_blockwise_sm120 thresholds:
    #   M<=64 or M%4!=0 -> swap_ab
    #   64<M<=256 + M%4==0 -> pingpong
    #   M>256 + M%4==0 -> default
    test_batch_sizes = [1, 7, 31, 32, 64, 65, 128, 256, 257, 512]

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = "cuda"
        self.weight_bf16 = (
            torch.randn(self.N, self.K, dtype=torch.bfloat16, device=self.device) * 0.05
        )
        weight_fp8, weight_scales = per_block_cast_to_fp8(
            self.weight_bf16, use_ue8m0=False
        )
        scale_K = (self.K + 127) // 128
        scale_N = (self.N + 127) // 128
        self.weight_fp8 = weight_fp8.reshape(self.K, self.N)
        self.weight_scales = weight_scales.reshape(scale_K, scale_N)
        self.quant_config = init_quant_config("FP8_PER_BLOCK")

    def _run(self, M: int, with_bias: bool):
        bias = (
            torch.randn(self.N, dtype=torch.bfloat16, device=self.device) * 0.01
            if with_bias
            else None
        )
        linear = CudaFp8VllmBlockwiseLinear(
            weight=self.weight_fp8,
            weight_scales=self.weight_scales,
            bias=bias,
            quant_config=self.quant_config,
        )
        x = torch.randn(M, self.K, dtype=torch.bfloat16, device=self.device) * 0.1
        out = linear(x)
        ref = x.float() @ self.weight_bf16.float().t()
        if bias is not None:
            ref = ref + bias.float()
        ref = ref.to(torch.bfloat16)
        diff = calc_diff(out, ref)
        self.assertLess(diff, 0.0011, f"M={M} with_bias={with_bias} diff={diff}")
        self.assertEqual(out.shape, (M, self.N))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_no_bias_all_dispatch_tiers(self):
        for M in self.test_batch_sizes:
            with self.subTest(M=M):
                self._run(M, with_bias=False)

    def test_with_bias_epilogue_fusion(self):
        for M in [1, 33, 128, 257]:
            with self.subTest(M=M):
                self._run(M, with_bias=True)


if __name__ == "__main__":
    unittest.main()
