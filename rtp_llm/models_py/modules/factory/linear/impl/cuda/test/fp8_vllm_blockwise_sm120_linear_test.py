"""sm12x-only numerical sanity tests for CudaFp8VllmBlockwiseLinear.

Quantizes a BF16 weight with per_block_cast_to_fp8 (block 128x128), runs
the kernel and compares against a fp32 reference matmul (+ optional bias).
Catches regressions in the three M-tier dispatch branches
(swap_ab / pingpong / default) and bias epilogue fusion.
"""

import unittest

import torch

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
    cutlass_scaled_mm_blockwise_sm120_fp8,
)
from rtp_llm.models_py.utils.arch import is_sm12x
from rtp_llm.test.utils.numeric_util import calc_diff, per_block_cast_to_fp8


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm12x(),
    "CudaFp8VllmBlockwiseLinear requires sm_120 (consumer Blackwell)",
)
class CudaFp8VllmBlockwiseLinearNumericalTest(unittest.TestCase):

    test_shapes = [(256, 256), (384, 256), (256, 384)]
    # Boundary values around dispatch_blockwise_sm120 thresholds:
    #   M<=64 or M%4!=0 -> swap_ab
    #   64<M<=256 + M%4==0 -> pingpong
    #   M>256 + M%4==0 -> default
    test_batch_sizes = [1, 7, 31, 32, 64, 65, 128, 256, 257, 512]

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = "cuda"
        self.quant_config = init_quant_config("FP8_PER_BLOCK")

    def _make_weight(self, K: int, N: int):
        self.weight_bf16 = (
            torch.randn(N, K, dtype=torch.bfloat16, device=self.device) * 0.05
        )
        weight_fp8, weight_scales = per_block_cast_to_fp8(
            self.weight_bf16, use_ue8m0=False
        )
        scale_K = (K + 127) // 128
        scale_N = (N + 127) // 128
        self.weight_fp8 = weight_fp8.reshape(K, N)
        self.weight_scales = weight_scales.reshape(scale_K, scale_N)

    def _run(self, M: int, K: int, N: int, with_bias: bool):
        self._make_weight(K, N)
        bias = (
            torch.randn(N, dtype=torch.bfloat16, device=self.device) * 0.01
            if with_bias
            else None
        )
        linear = CudaFp8VllmBlockwiseLinear(
            weight=self.weight_fp8,
            weight_scales=self.weight_scales,
            bias=bias,
            quant_config=self.quant_config,
        )
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * 0.1
        out = linear(x)
        ref = x.float() @ self.weight_bf16.float().t()
        if bias is not None:
            ref = ref + bias.float()
        ref = ref.to(torch.bfloat16)
        diff = calc_diff(out, ref)
        self.assertLess(
            diff, 0.0011, f"M={M} K={K} N={N} with_bias={with_bias} diff={diff}"
        )
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_no_bias_all_dispatch_tiers(self):
        for K, N in self.test_shapes[:2]:
            for M in self.test_batch_sizes:
                with self.subTest(M=M, K=K, N=N):
                    self._run(M, K=K, N=N, with_bias=False)

    def test_with_bias_epilogue_fusion(self):
        for K, N in (self.test_shapes[0], self.test_shapes[2]):
            for M in [1, 33, 128, 257]:
                with self.subTest(M=M, K=K, N=N):
                    self._run(M, K=K, N=N, with_bias=True)

    def test_reject_fp16_input(self):
        K, N = self.test_shapes[0]
        self._make_weight(K, N)
        linear = CudaFp8VllmBlockwiseLinear(
            weight=self.weight_fp8,
            weight_scales=self.weight_scales,
            quant_config=self.quant_config,
        )
        input_fp16 = torch.randn(8, K, dtype=torch.float16, device=self.device)

        with self.assertRaisesRegex(
            ValueError, "Input tensor dtype must be bfloat16.*torch.float16"
        ):
            linear(input_fp16)

    def test_reject_unaligned_weight_shape(self):
        for K, N in [(320, 256), (256, 320)]:
            with self.subTest(K=K, N=N):
                self._make_weight(K, N)
                with self.assertRaisesRegex(
                    ValueError,
                    rf"K and N to be multiples of 128, got K={K} and N={N}",
                ):
                    CudaFp8VllmBlockwiseLinear(
                        weight=self.weight_fp8,
                        weight_scales=self.weight_scales,
                        quant_config=self.quant_config,
                    )


@unittest.skipIf(
    cutlass_scaled_mm_blockwise_sm120_fp8 is None,
    "SM120 FP8 blockwise op is only available on sm12x CUDA builds",
)
class CudaFp8VllmBlockwiseSM120BoundaryTest(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("SM120 FP8 blockwise tests require CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = "cuda"
        self.M = 8
        # Use at least two K scale groups so making A_sf contiguous actually
        # changes its MN-major stride and exercises the boundary check.
        self.K = 256
        self.N = 128

    def _make_op_inputs(self):
        input_tensor = torch.randn(
            self.M, self.K, dtype=torch.bfloat16, device=self.device
        ).contiguous()
        weight_bf16 = (
            torch.randn((self.N, self.K), dtype=torch.bfloat16, device=self.device)
            * 0.1
        ).contiguous()
        A, A_sf = sgl_per_token_group_quant_fp8(
            input_tensor,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )
        B, B_sf = per_block_cast_to_fp8(weight_bf16, use_ue8m0=False)
        D = torch.empty(self.M, self.N, dtype=torch.bfloat16, device=self.device)
        return D, A, B, A_sf, B_sf

    def test_rejects_wrong_input_scale_stride(self):
        D, A, B, A_sf, B_sf = self._make_op_inputs()
        bad_A_sf = A_sf.contiguous()
        with self.assertRaisesRegex(RuntimeError, "A_sf must use MN-major"):
            cutlass_scaled_mm_blockwise_sm120_fp8(D, A, B, bad_A_sf, B_sf)

    def test_rejects_cpu_bias_at_pybind_boundary(self):
        D, A, B, A_sf, B_sf = self._make_op_inputs()
        bias = torch.randn(self.N, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "bias must be a CUDA tensor"):
            cutlass_scaled_mm_blockwise_sm120_fp8(D, A, B, A_sf, B_sf, bias)

    def test_wrapper_moves_cpu_bias_to_output_device(self):
        weight = torch.randn(
            self.K, self.N, dtype=torch.float32, device=self.device
        ).to(torch.float8_e4m3fn)
        weight_scales = torch.rand(
            (self.K + 127) // 128,
            (self.N + 127) // 128,
            dtype=torch.float32,
            device=self.device,
        )
        bias = torch.randn(self.N, dtype=torch.bfloat16)
        linear = CudaFp8VllmBlockwiseLinear(weight, weight_scales, bias=bias)
        input_tensor = torch.randn(
            self.M, self.K, dtype=torch.bfloat16, device=self.device
        )
        output = linear(input_tensor)
        self.assertEqual(output.shape, (self.M, self.N))
        self.assertEqual(output.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
