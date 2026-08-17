"""sm12x-only numerical sanity tests for CudaFp8VllmBlockwiseLinear.

Quantizes a BF16 weight with per_block_cast_to_fp8 (block 128x128), runs
the kernel and compares against a fp32 reference matmul (+ optional bias/GELU).
Catches regressions in the three M-tier dispatch branches
(swap_ab / pingpong / default) and fused bias/GELU epilogues.
"""

import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
    _get_cutlass_scaled_mm_blockwise_sm120_fp8,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.test.sm120_test_utils import (
    make_blockwise_op_inputs,
)
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.test.utils.numeric_util import calc_diff, per_block_cast_to_fp8
from rtp_llm.utils.sm120_fp8_backend import SM120_FP8_BACKEND_ENV


class CudaFp8VllmBlockwiseLinearNumericalTest(unittest.TestCase):

    test_shapes = [(256, 256), (384, 256), (256, 384)]
    # Boundary values around dispatch_blockwise_sm120 thresholds:
    #   M<=64 or M%4!=0 -> swap_ab
    #   64<M<=256 + M%4==0 -> pingpong
    #   M>256 + M%4==0 -> default
    test_batch_sizes = [1, 7, 31, 32, 64, 65, 128, 256, 257, 512]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not is_sm120():
            raise unittest.SkipTest(
                "CudaFp8VllmBlockwiseLinear requires sm_120 (consumer Blackwell)"
            )

    def setUp(self):
        self.backend_env = mock.patch.dict(
            os.environ, {SM120_FP8_BACKEND_ENV: "cutlass"}
        )
        self.backend_env.start()
        self.addCleanup(self.backend_env.stop)
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

    def _run(self, M: int, K: int, N: int, with_bias: bool, use_gelu: bool = False):
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
        out = linear.forward_with_bias_gelu(x) if use_gelu else linear(x)
        ref = x.float() @ self.weight_bf16.float().t()
        if bias is not None:
            ref = ref + bias.float()
        if use_gelu:
            ref = F.gelu(ref, approximate="tanh")
        ref = ref.to(torch.bfloat16)
        diff = calc_diff(out, ref)
        self.assertLess(
            diff,
            0.0011,
            f"M={M} K={K} N={N} with_bias={with_bias} use_gelu={use_gelu} diff={diff}",
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

    def test_with_fused_bias(self):
        for K, N in (self.test_shapes[0], self.test_shapes[2]):
            for M in [1, 33, 128, 257, 512]:
                with self.subTest(M=M, K=K, N=N):
                    self._run(M, K=K, N=N, with_bias=True)

    def test_with_fused_bias_gelu(self):
        for M in [1, 65, 257]:
            with self.subTest(M=M):
                self._run(M, K=256, N=256, with_bias=True, use_gelu=True)

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

    def test_reject_noncontiguous_input(self):
        K, N = self.test_shapes[0]
        self._make_weight(K, N)
        linear = CudaFp8VllmBlockwiseLinear(
            weight=self.weight_fp8,
            weight_scales=self.weight_scales,
            quant_config=self.quant_config,
        )
        input_noncontiguous = torch.randn(
            K, 8, dtype=torch.bfloat16, device=self.device
        ).t()
        self.assertFalse(input_noncontiguous.is_contiguous())

        with self.assertRaisesRegex(ValueError, "input must be contiguous"):
            linear(input_noncontiguous)

    def test_empty_batch_returns_empty_output(self):
        K, N = self.test_shapes[0]
        self._make_weight(K, N)
        linear = CudaFp8VllmBlockwiseLinear(
            weight=self.weight_fp8,
            weight_scales=self.weight_scales,
            quant_config=self.quant_config,
        )

        output = linear(torch.empty(0, K, dtype=torch.bfloat16, device=self.device))

        self.assertEqual(output.shape, (0, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.device.type, "cuda")

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

    def test_reject_noncontiguous_weight_layout(self):
        K, N = self.test_shapes[0]
        self._make_weight(K, N)
        noncontiguous_weight = self.weight_fp8.reshape(N, K).t()
        self.assertFalse(noncontiguous_weight.is_contiguous())
        with self.assertRaisesRegex(ValueError, "weight must be contiguous"):
            CudaFp8VllmBlockwiseLinear(
                weight=noncontiguous_weight,
                weight_scales=self.weight_scales,
                quant_config=self.quant_config,
            )

    def test_reject_noncontiguous_weight_scale_layout(self):
        K, N = self.test_shapes[0]
        self._make_weight(K, N)
        noncontiguous_scales = self.weight_scales.t()
        self.assertFalse(noncontiguous_scales.is_contiguous())
        with self.assertRaisesRegex(ValueError, "weight scales must be contiguous"):
            CudaFp8VllmBlockwiseLinear(
                weight=self.weight_fp8,
                weight_scales=noncontiguous_scales,
                quant_config=self.quant_config,
            )

    def test_factory_selects_only_sm120_blockwise_backend(self):
        from rtp_llm.models_py.modules.factory.linear import LinearFactory

        K, N = self.test_shapes[0]
        self._make_weight(K, N)
        linear = LinearFactory.create_linear(
            weight=self.weight_fp8,
            bias=None,
            weight_scales=self.weight_scales,
            quant_config=self.quant_config,
        )
        self.assertIsInstance(linear, CudaFp8VllmBlockwiseLinear)

    def test_factory_non_square_weight_matches_reference(self):
        from rtp_llm.models_py.modules.factory.linear import LinearFactory

        M, K, N = 7, 384, 256
        self._make_weight(K, N)
        linear = LinearFactory.create_linear(
            weight=self.weight_fp8,
            bias=None,
            weight_scales=self.weight_scales,
            quant_config=self.quant_config,
        )
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * 0.1

        output = linear(x)
        reference = (x.float() @ self.weight_bf16.float().t()).to(torch.bfloat16)

        self.assertLess(calc_diff(output, reference), 0.0011)
        self.assertEqual(output.shape, (M, N))

    def test_factory_rejects_ue8m0_weight_scales(self):
        from rtp_llm.models_py.modules.factory.linear import LinearFactory

        K, N = self.test_shapes[0]
        self._make_weight(K, N)
        ue8m0_scales = torch.zeros(
            N, (K + 511) // 512, dtype=torch.int32, device=self.device
        )
        with self.assertRaisesRegex(ValueError, "requires float32 weight scales"):
            LinearFactory.create_linear(
                self.weight_fp8,
                None,
                ue8m0_scales,
                self.quant_config,
            )


class CudaFp8VllmBlockwiseSM120BoundaryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not is_sm120():
            raise unittest.SkipTest(
                "SM120 FP8 blockwise op requires an sm_120 CUDA device"
            )
        cls.gemm_op = _get_cutlass_scaled_mm_blockwise_sm120_fp8()
        if cls.gemm_op is None:
            raise RuntimeError("SM120 FP8 blockwise binding is missing")

    def setUp(self):
        self.backend_env = mock.patch.dict(
            os.environ, {SM120_FP8_BACKEND_ENV: "cutlass"}
        )
        self.backend_env.start()
        self.addCleanup(self.backend_env.stop)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = "cuda"
        self.M = 8
        # Use at least two K scale groups so making A_sf contiguous actually
        # changes its MN-major stride and exercises the boundary check.
        self.K = 256
        self.N = 128

    def _make_op_inputs(self):
        return make_blockwise_op_inputs(self.M, self.K, self.N, self.device)

    def test_rejects_wrong_input_scale_stride(self):
        D, A, B, A_sf, B_sf = self._make_op_inputs()
        bad_A_sf = A_sf.contiguous()
        with self.assertRaisesRegex(RuntimeError, "A_sf must use MN-major"):
            self.gemm_op(D, A, B, bad_A_sf, B_sf)

    def test_direct_binding_rejects_non_aligned_n(self):
        _, A, B, A_sf, B_sf = self._make_op_inputs()
        bad_n = 96
        B = B[:bad_n].contiguous()
        D = torch.empty(self.M, bad_n, dtype=torch.bfloat16, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "K and N must be multiples"):
            self.gemm_op(D, A, B, A_sf, B_sf)

    def test_wrapper_normalizes_cpu_bias_during_construction(self):
        weight_bf16 = torch.randn(
            self.N, self.K, dtype=torch.bfloat16, device=self.device
        )
        weight, weight_scales = per_block_cast_to_fp8(weight_bf16, use_ue8m0=False)
        weight = weight.reshape(self.K, self.N)
        weight_scales = weight_scales.reshape(
            (self.K + 127) // 128, (self.N + 127) // 128
        )
        bias = torch.randn(self.N, dtype=torch.bfloat16)
        linear = CudaFp8VllmBlockwiseLinear(weight, weight_scales, bias=bias)
        self.assertEqual(linear.bias.device.type, "cuda")
        input_tensor = torch.randn(
            self.M, self.K, dtype=torch.bfloat16, device=self.device
        )
        output = linear(input_tensor)
        cuda_bias_linear = CudaFp8VllmBlockwiseLinear(
            weight, weight_scales, bias=bias.to(self.device)
        )
        expected = cuda_bias_linear(input_tensor)
        self.assertEqual(output.shape, (self.M, self.N))
        self.assertEqual(output.device.type, "cuda")
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_direct_binding_empty_batch_is_noop(self):
        _, _, B, _, B_sf = make_blockwise_op_inputs(1, self.K, self.N)
        D = torch.empty(0, self.N, dtype=torch.bfloat16, device=self.device)
        A = torch.empty(0, self.K, dtype=torch.float8_e4m3fn, device=self.device)
        A_sf = torch.empty_strided(
            (0, self.K // 128),
            (1, 0),
            dtype=torch.float32,
            device=self.device,
        )

        self.gemm_op(D, A, B, A_sf, B_sf)
        # M == 0 returns before scale-layout validation and launches no kernel.
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
