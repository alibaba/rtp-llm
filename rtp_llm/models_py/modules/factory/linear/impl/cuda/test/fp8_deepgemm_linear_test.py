import logging
import os
import shutil
import unittest

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.test.utils.bench_util import bench
from rtp_llm.test.utils.numeric_util import calc_diff, per_block_cast_to_fp8


class CudaFp8DeepGEMMLinearTest(unittest.TestCase):

    def setUp(self):
        """Setup test environment"""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.skipTest("FP8 tests require CUDA")
        logging.getLogger(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear"
        ).setLevel(logging.WARNING)

        # Generate random weights and scales
        self.K = 12288
        self.N = 6144
        self.scale_K = (self.K + 127) // 128
        self.scale_N = (self.N + 127) // 128
        self.test_batch_sizes = [
            1,
            7,
            15,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            2049,
            2305,
            2561,
            2817,
            3073,
            3329,
            3585,
            3841,
            126543,
            126765,
            127899,
            127999,
        ]
        weight_fp32 = (
            torch.randn(
                self.K,
                self.N,
                dtype=torch.float32,
                device=self.device,
            )
            * 0.1
        )
        self.weight = weight_fp32.to(torch.float8_e4m3fn)
        self.weight_scales = (
            torch.rand(
                self.scale_K,
                self.scale_N,
                dtype=torch.float32,
                device=self.device,
            )
            * 0.01
            + 0.001
        )
        self.bias = torch.randn(self.N, dtype=torch.bfloat16, device=self.device)

    def _create_cuda_fp8_deepgemm_linear(
        self, with_bias: bool = False, with_bias_2d: bool = False
    ):
        """Helper method to create CudaFp8DeepGEMMLinear instance"""
        if with_bias_2d:
            bias = self.bias.unsqueeze(0)
        else:
            bias = self.bias
        return CudaFp8DeepGEMMLinear(
            weight=self.weight,
            weight_scales=self.weight_scales,
            input_scales=None,
            bias=bias if (with_bias or with_bias_2d) else None,
            config=None,
        )

    def test_dependency_availability(self):
        """Test dependency availability check - should fail if dependencies are missing"""
        # Test that we can at least import the module
        self.assertIsNotNone(CudaFp8DeepGEMMLinear)

    def test_can_handle_availability(self):
        """Test can handle availability"""
        # Test that can handle is not None
        self.assertIsNotNone(CudaFp8DeepGEMMLinear.can_handle)

    def test_module_creation(self):
        """Test CudaFp8DeepGEMMLinear module creation"""
        # Test with bias 2D
        cuda_fp8_deepgemm_linear_with_bias_2d = self._create_cuda_fp8_deepgemm_linear(
            with_bias_2d=True
        )
        self.assertEqual(cuda_fp8_deepgemm_linear_with_bias_2d.K, self.K)
        self.assertEqual(cuda_fp8_deepgemm_linear_with_bias_2d.N, self.N)
        self.assertIsNotNone(cuda_fp8_deepgemm_linear_with_bias_2d.weight)
        self.assertEqual(
            cuda_fp8_deepgemm_linear_with_bias_2d.weight.shape, (self.N, self.K)
        )
        self.assertIsNotNone(cuda_fp8_deepgemm_linear_with_bias_2d.weight_scales)
        self.assertEqual(
            cuda_fp8_deepgemm_linear_with_bias_2d.weight_scales.shape,
            (self.scale_N, self.scale_K),
        )
        self.assertIsNotNone(cuda_fp8_deepgemm_linear_with_bias_2d.bias)
        self.assertEqual(cuda_fp8_deepgemm_linear_with_bias_2d.bias.dim(), 2)
        # Test with bias 1D
        cuda_fp8_deepgemm_linear_with_bias_1d = self._create_cuda_fp8_deepgemm_linear(
            with_bias=True, with_bias_2d=False
        )
        self.assertEqual(cuda_fp8_deepgemm_linear_with_bias_1d.K, self.K)
        self.assertEqual(cuda_fp8_deepgemm_linear_with_bias_1d.N, self.N)
        self.assertIsNotNone(cuda_fp8_deepgemm_linear_with_bias_1d.weight)
        self.assertEqual(
            cuda_fp8_deepgemm_linear_with_bias_1d.weight.shape, (self.N, self.K)
        )
        self.assertIsNotNone(cuda_fp8_deepgemm_linear_with_bias_1d.weight_scales)
        self.assertEqual(
            cuda_fp8_deepgemm_linear_with_bias_1d.weight_scales.shape,
            (self.scale_N, self.scale_K),
        )
        self.assertIsNotNone(cuda_fp8_deepgemm_linear_with_bias_1d.bias)
        self.assertEqual(cuda_fp8_deepgemm_linear_with_bias_1d.bias.dim(), 1)
        # Test without bias
        cuda_fp8_deepgemm_linear_without_bias = self._create_cuda_fp8_deepgemm_linear(
            with_bias=False
        )
        self.assertEqual(cuda_fp8_deepgemm_linear_without_bias.K, self.K)
        self.assertEqual(cuda_fp8_deepgemm_linear_without_bias.N, self.N)
        self.assertIsNotNone(cuda_fp8_deepgemm_linear_without_bias.weight)
        self.assertEqual(
            cuda_fp8_deepgemm_linear_without_bias.weight.shape, (self.N, self.K)
        )
        self.assertIsNotNone(cuda_fp8_deepgemm_linear_without_bias.weight_scales)
        self.assertEqual(
            cuda_fp8_deepgemm_linear_without_bias.weight_scales.shape,
            (self.scale_N, self.scale_K),
        )
        self.assertIsNone(cuda_fp8_deepgemm_linear_without_bias.bias)

    def test_weight_validation(self):
        """Test weight validation"""
        # Validate weight dimension not equal to 2
        weight = torch.randn(
            (self.K, self.N, 1), dtype=torch.float32, device=self.device
        ).to(torch.float8_e4m3fn)
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(weight, self.weight_scales)
        self.assertIn(
            f"Weight and weight scale must be 2D tensors, but ", str(context.exception)
        )
        self.assertIn(
            f"got weight dim: {weight.dim()} and weight scale dim: {self.weight_scales.dim()}",
            str(context.exception),
        )
        weight = torch.randn((self.K), dtype=torch.float32, device=self.device).to(
            torch.float8_e4m3fn
        )
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(weight, self.weight_scales)
        self.assertIn(
            f"Weight and weight scale must be 2D tensors, but ", str(context.exception)
        )
        self.assertIn(
            f"got weight dim: {weight.dim()} and weight scale dim: {self.weight_scales.dim()}",
            str(context.exception),
        )
        # Validate weight shape not equal to (N, K)
        weight = torch.randn(
            (self.K, self.N + 1), dtype=torch.float32, device=self.device
        ).to(torch.float8_e4m3fn)
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(weight, self.weight_scales)
        self.assertIn(f"Weight scale dimension mismatch! ", str(context.exception))
        self.assertIn(
            f"N: {weight.shape[1]}, scale_N: {self.weight_scales.shape[1]}, K: {weight.shape[0]}, scale_K: {self.weight_scales.shape[0]}",
            str(context.exception),
        )
        weight = torch.randn(
            (self.K + 1, self.N), dtype=torch.float32, device=self.device
        ).to(torch.float8_e4m3fn)
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(weight, self.weight_scales)
        self.assertIn(f"Weight scale dimension mismatch! ", str(context.exception))
        self.assertIn(
            f"N: {weight.shape[1]}, scale_N: {self.weight_scales.shape[1]}, K: {weight.shape[0]}, scale_K: {self.weight_scales.shape[0]}",
            str(context.exception),
        )
        # Validate weight dtype not equal to float8_e4m3fn
        weight = torch.randn((self.K, self.N), dtype=torch.float32, device=self.device)
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(weight, self.weight_scales)
        self.assertIn(
            f"Weight dtype must be float8_e4m3fn, got ", str(context.exception)
        )
        self.assertIn(f"got {weight.dtype}", str(context.exception))

    def test_weight_scales_validation(self):
        """Test weight scale validation"""
        # Validate weight scale dimension not equal to 2
        weight_scales = torch.randn(
            (self.scale_K, self.scale_N, 1), dtype=torch.float32, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(self.weight, weight_scales)
        self.assertIn(
            f"Weight and weight scale must be 2D tensors, but ", str(context.exception)
        )
        self.assertIn(
            f"got weight dim: {self.weight.dim()} and weight scale dim: {weight_scales.dim()}",
            str(context.exception),
        )
        weight_scales = torch.randn(
            (self.scale_K), dtype=torch.float32, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(self.weight, weight_scales)
        self.assertIn(
            f"Weight and weight scale must be 2D tensors, but ", str(context.exception)
        )
        self.assertIn(
            f"got weight dim: {self.weight.dim()} and weight scale dim: {weight_scales.dim()}",
            str(context.exception),
        )
        # Validate weight scale shape not equal to (scale_N, scale_K)
        weight_scales = torch.randn(
            (self.scale_K, self.scale_N + 1), dtype=torch.float32, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(self.weight, weight_scales)
        self.assertIn(f"Weight scale dimension mismatch! ", str(context.exception))
        self.assertIn(
            f"N: {self.weight.shape[1]}, scale_N: {weight_scales.shape[1]}, K: {self.weight.shape[0]}, scale_K: {weight_scales.shape[0]}",
            str(context.exception),
        )
        weight_scales = torch.randn(
            (self.scale_K + 1, self.scale_N), dtype=torch.float32, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(self.weight, weight_scales)
        self.assertIn(f"Weight scale dimension mismatch! ", str(context.exception))
        self.assertIn(
            f"N: {self.weight.shape[1]}, scale_N: {weight_scales.shape[1]}, K: {self.weight.shape[0]}, scale_K: {weight_scales.shape[0]}",
            str(context.exception),
        )
        weight = torch.randn((7168, 2112), dtype=torch.float32, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scales = torch.randn((56, 17), dtype=torch.float32, device=self.device)
        cuda_fp8_deepgemm_linear = CudaFp8DeepGEMMLinear(weight, weight_scales)
        input_tensor = torch.randn(
            32,
            7168,
            dtype=torch.bfloat16,
            device=self.device,
        )
        output = cuda_fp8_deepgemm_linear(input_tensor)
        self.assertEqual(output.shape, (32, 2112))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.device.type, "cuda")
        # Validate weight scale dtype not equal to float32
        weight_scales = torch.randn(
            (self.scale_K, self.scale_N), dtype=torch.float16, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(self.weight, weight_scales)
        self.assertIn(f"Weight scale dtype must be float32, ", str(context.exception))
        self.assertIn(f"got {weight_scales.dtype}", str(context.exception))

    def test_bias_validation(self):
        """Test bias validation"""
        # Validate bias dimension not equal to 1 or 2
        bias = torch.randn((1, self.N, 1), dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(self.weight, self.weight_scales, bias=bias)
        self.assertIn(f"Bias dimension must be 1 or 2, ", str(context.exception))
        self.assertIn(f"got {bias.dim()}", str(context.exception))
        # Validate bias last dimension not equal to N
        bias = torch.randn((1, self.N + 1), dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(self.weight, self.weight_scales, bias=bias)
        self.assertIn(f"Bias last dimension must be {self.N}, ", str(context.exception))
        self.assertIn(f"got {bias.shape[-1]}", str(context.exception))
        # Validate the last dimension of bias not equal to N
        bias = torch.randn((self.N - 1,), dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(self.weight, self.weight_scales, bias=bias)
        self.assertIn(f"Bias last dimension must be {self.N}, ", str(context.exception))
        self.assertIn(f"got {bias.shape[-1]}", str(context.exception))
        # Validate first dimension of 2D bias not equal to 1
        bias = torch.randn((2, self.N), dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(self.weight, self.weight_scales, bias=bias)
        self.assertIn(f"Bias first dimension must be 1, ", str(context.exception))
        self.assertIn(f"got {bias.shape[0]}", str(context.exception))
        # Validate bias dtype not equal to float32
        bias = torch.randn((1, self.N), dtype=torch.float32, device=self.device)
        with self.assertRaises(ValueError) as context:
            CudaFp8DeepGEMMLinear(self.weight, self.weight_scales, bias=bias)
        self.assertIn(f"Bias dtype must be bfloat16, ", str(context.exception))
        self.assertIn("got torch.float32", str(context.exception))

    def test_input_validation(self):
        """Test input validation"""
        cuda_fp8_deepgemm_linear = self._create_cuda_fp8_deepgemm_linear(
            with_bias=False
        )
        # Test with 1D tensor (should raise ValueError)
        input_tensor = torch.randn(self.K, dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError) as context:
            cuda_fp8_deepgemm_linear(input_tensor)
        self.assertIn(f"Input tensor dimension must be 2", str(context.exception))
        self.assertIn(f"got {input_tensor.dim()}D tensor", str(context.exception))
        # Test with 3D tensor (should raise ValueError)
        input_tensor = torch.randn(
            32, self.K, 2, dtype=torch.bfloat16, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            cuda_fp8_deepgemm_linear(input_tensor)
        self.assertIn(f"Input tensor dimension must be 2", str(context.exception))
        self.assertIn(f"got {input_tensor.dim()}D tensor", str(context.exception))
        # Test with input tensor inner dimension not expected to be K (should raise ValueError)
        input_tensor = torch.randn(
            32, self.K + 1, dtype=torch.bfloat16, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            cuda_fp8_deepgemm_linear(input_tensor)
        self.assertIn(
            f"Input tensor inner dimension expected to be {self.K}",
            str(context.exception),
        )
        self.assertIn(f"got {input_tensor.shape[1]}", str(context.exception))
        # Test with bfloat16 input (should work)
        input_bf16 = torch.randn(32, self.K, dtype=torch.bfloat16, device=self.device)
        output = cuda_fp8_deepgemm_linear(input_bf16)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (32, self.N))
        # Test with float32 input (should raise ValueError)
        input_fp32 = torch.randn(32, self.K, dtype=torch.float32, device=self.device)
        with self.assertRaises(ValueError) as context:
            cuda_fp8_deepgemm_linear(input_fp32)
        self.assertIn(f"Input tensor dtype must be bfloat16", str(context.exception))
        self.assertIn("torch.float32", str(context.exception))
        # Test with float16 input (should raise ValueError)
        input_fp16 = torch.randn(32, self.K, dtype=torch.float16, device=self.device)
        with self.assertRaises(ValueError) as context:
            cuda_fp8_deepgemm_linear(input_fp16)
        self.assertIn(f"Input tensor dtype must be bfloat16", str(context.exception))
        self.assertIn("torch.float16", str(context.exception))
        # Test with int32 input (should raise ValueError)
        input_int32 = torch.randint(
            0, 10, (32, self.K), dtype=torch.int32, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            cuda_fp8_deepgemm_linear(input_int32)
        self.assertIn(f"Input tensor dtype must be bfloat16", str(context.exception))
        self.assertIn("torch.int32", str(context.exception))

    def test_output_dimension(self):
        """Test output dimension"""
        cuda_fp8_deepgemm_linear = self._create_cuda_fp8_deepgemm_linear(
            with_bias=False
        )
        for test_batch_size in self.test_batch_sizes:
            with self.subTest(batch_size=test_batch_size):
                input_tensor = torch.randn(
                    test_batch_size,
                    self.K,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                output = cuda_fp8_deepgemm_linear(input_tensor)
                expected_shape = (test_batch_size, self.N)
                self.assertEqual(output.shape, expected_shape)

    def test_output_dtype(self):
        """Test output dtype"""
        cuda_fp8_deepgemm_linear = self._create_cuda_fp8_deepgemm_linear(
            with_bias=False
        )
        for test_batch_size in self.test_batch_sizes:
            with self.subTest(batch_size=test_batch_size):
                input_tensor = torch.randn(
                    test_batch_size,
                    self.K,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                output = cuda_fp8_deepgemm_linear(input_tensor)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(output.device.type, "cuda")

    def test_output_correctness(self):
        """Test output correctness"""
        # Generate random weights and scale
        weight_bf16 = torch.randn(
            (self.N, self.K),
            dtype=torch.bfloat16,
            device=self.device,
        )
        weight_fp8, weight_scales = per_block_cast_to_fp8(weight_bf16, use_ue8m0=False)
        weight_fp8 = weight_fp8.reshape(self.K, self.N)
        weight_scales = weight_scales.reshape(self.scale_K, self.scale_N)
        # Initialize FP8 linear layer
        cuda_fp8_deepgemm_linear = CudaFp8DeepGEMMLinear(weight_fp8, weight_scales)
        # Test some batch sizes
        for test_batch_size in self.test_batch_sizes:
            with self.subTest(batch_size=test_batch_size):
                input_tensor = torch.randn(
                    test_batch_size,
                    self.K,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                output = cuda_fp8_deepgemm_linear(input_tensor)
                ref_output = (input_tensor.float() @ weight_bf16.float().t()).to(
                    torch.bfloat16
                )
                diff = calc_diff(output, ref_output)
                self.assertLess(diff, 0.001)
                self.assertEqual(output.shape, (test_batch_size, self.N))
                self.assertEqual(output.shape, ref_output.shape)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(output.dtype, ref_output.dtype)
                self.assertEqual(output.device.type, "cuda")
                self.assertEqual(output.device, ref_output.device)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_output_reproducibility(self):
        """Test output reproducibility"""
        cuda_fp8_deepgemm_linear = self._create_cuda_fp8_deepgemm_linear(
            with_bias=False
        )
        for test_batch_size in self.test_batch_sizes:
            with self.subTest(batch_size=test_batch_size):
                input_tensor = torch.randn(
                    test_batch_size,
                    self.K,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                output1 = cuda_fp8_deepgemm_linear(input_tensor)
                output2 = cuda_fp8_deepgemm_linear(input_tensor)
                self.assertEqual(output1.shape, output2.shape)
                self.assertEqual(output1.dtype, output2.dtype)
                self.assertEqual(output1.device, output2.device)
                diff = calc_diff(output1, output2)
                self.assertLess(diff, 1e-5)

    def test_bias_handling(self):
        """Test bias handling"""
        # Generate random weights and scale
        weight_bf16 = torch.randn(
            (self.N, self.K), dtype=torch.bfloat16, device=self.device
        )
        weight_fp8, weight_scales = per_block_cast_to_fp8(weight_bf16, use_ue8m0=False)
        weight_fp8 = weight_fp8.reshape(self.K, self.N)
        weight_scales = weight_scales.reshape(self.scale_K, self.scale_N)
        bias_bf16 = torch.randn((self.N,), dtype=torch.bfloat16, device=self.device)
        # Initialize FP8 linear layer
        cuda_fp8_deepgemm_linear = CudaFp8DeepGEMMLinear(
            weight_fp8, weight_scales, bias=bias_bf16
        )
        # Test some batch sizes
        for test_batch_size in self.test_batch_sizes:
            with self.subTest(batch_size=test_batch_size):
                input_tensor = torch.randn(
                    test_batch_size,
                    self.K,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                output = cuda_fp8_deepgemm_linear(input_tensor)
                ref_output = (
                    input_tensor.float() @ weight_bf16.float().t() + bias_bf16.float()
                ).to(torch.bfloat16)
                diff = calc_diff(output, ref_output)
                self.assertLess(diff, 0.001)
                self.assertEqual(output.shape, (test_batch_size, self.N))
                self.assertEqual(output.shape, ref_output.shape)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(output.dtype, ref_output.dtype)
                self.assertEqual(output.device.type, "cuda")
                self.assertEqual(output.device, ref_output.device)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    @unittest.skip("Skip profiling tests")
    def test_profile_cuda_fp8_deepgemm_linear(self):
        """Profile CUDA FP8 DeepGEMM linear"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise RuntimeError(
                "CUDA FP8 DeepGEMM linear tests require CUDA, but CUDA is not available"
            )
        # Generate random weights and scales
        K, N = 12288, 6144
        batch_sizes = (
            [1, 8, 16, 32, 48, 64, 80, 96, 112, 128]
            + [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960]
            + [2049, 2305, 2561, 2817, 3073, 3329, 3585, 3841]
        )
        weight_fp32 = (
            torch.randn(
                K,
                N,
                dtype=torch.float32,
                device=device,
            )
            * 0.1
        )
        weight = weight_fp32.to(torch.float8_e4m3fn)
        weight_scales = (
            torch.rand(
                (K + 127) // 128,
                (N + 127) // 128,
                dtype=torch.float32,
                device=device,
            )
            * 0.01
            + 0.001
        )
        # Create FP8 per block linear layer
        cuda_fp8_deepgemm_linear = CudaFp8DeepGEMMLinear(weight, weight_scales)
        # Create trace file directory if not exists
        if os.path.exists("./trace_files"):
            shutil.rmtree("./trace_files")
        os.makedirs("./trace_files", exist_ok=True)
        # Generate random input tensor
        for batch_size in batch_sizes:
            input_tensor = torch.randn(
                batch_size, K, dtype=torch.bfloat16, device=device
            )
            t_mean_new, t_min_new, t_max_new = bench(
                lambda: cuda_fp8_deepgemm_linear(input_tensor),
                num_warmups=20,
                num_tests=30,
                suppress_kineto_output=True,
                barrier_comm_profiling=False,
                trace_path=f"./trace_files/cuda_fp8_deepgemm_linear_{batch_size}.json",
            )
            print(f"Batch size {batch_size}: t_mean_new={t_mean_new:.6f}s")


if __name__ == "__main__":
    unittest.main()
