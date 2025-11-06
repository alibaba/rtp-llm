import logging
import os
import shutil
import unittest

import torch

from rtp_llm.test.utils.bench_util import bench
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import CudaFp8DeepGEMMLinear


class CudaFp8DeepGEMMLinearTest(unittest.TestCase):

    def setUp(self):
        """Setup test environment"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.skipTest("FP8 tests require CUDA")
        logging.getLogger(
            "rtp_llm.models_py.modules.cuda.linear.fp8_deepgemm_linear"
        ).setLevel(logging.WARNING)
        
        # Generate random weights and scales
        self.K = 12288
        self.N = 6144
        self.batch_sizes = [1, 32, 64, 128, 256]
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
        self.weight_scale = (
            torch.rand(
                (self.K + 127) // 128,
                (self.N + 127) // 128,
                dtype=torch.float32,
                device=self.device,
            )
            * 0.01
            + 0.001
        )
        self.bias = torch.randn(self.N, dtype=torch.bfloat16, device=self.device)

    def _create_fp8_linear(self, with_bias: bool = True, with_bias_2d: bool = False):
        """Helper method to create CudaFp8DeepGEMMLinear instance"""
        if with_bias_2d:
            bias = self.bias.unsqueeze(0)
        else:
            bias = self.bias
        return CudaFp8DeepGEMMLinear(
            weight=self.weight,
            weight_scale=self.weight_scale,
            bias=bias if with_bias else None,
        )

    def test_dependency_availability(self):
        """Test dependency availability check - should fail if dependencies are missing"""

        # Test that we can at least import the module
        self.assertIsNotNone(Fp8PerBlockLinear)

        # For unit tests, dependencies MUST be available - fail if not

    def test_module_creation(self):
        """Test CudaFp8DeepGEMMLinear module creation"""
        fp8_linear = self._create_fp8_linear(with_bias=True)
        self.assertEqual(fp8_linear.K, self.K)
        self.assertEqual(fp8_linear.N, self.N)
        self.assertIsNotNone(fp8_linear.weight)
        self.assertIsNotNone(fp8_linear.weight_scale)
        self.assertIsNotNone(fp8_linear.bias)

        fp8_linear_no_bias = self._create_fp8_linear(with_bias=False)
        self.assertEqual(fp8_linear_no_bias.K, self.K)
        self.assertEqual(fp8_linear_no_bias.N, self.N)
        self.assertIsNone(fp8_linear_no_bias.bias)

    def test_dependency_availability(self):
        """Test dependency availability check - should fail if dependencies are missing"""

        # Test that we can at least import the module
        self.assertIsNotNone(CudaFp8DeepGEMMLinear)

        # For unit tests, dependencies MUST be available - fail if not

    def test_input_dtype_validation(self):
        """Test input dtype validation - only bfloat16 is accepted"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=False)

        # Test with bfloat16 input (should work)
        input_bf16 = torch.randn(32, self.K, dtype=torch.bfloat16, device=self.device)
        output = fp8_linear(input_bf16)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (32, self.N))

        # Test with float32 input (should raise ValueError)
        input_fp32 = torch.randn(32, self.K, dtype=torch.float32, device=self.device)
        with self.assertRaises(ValueError) as context:
            fp8_linear(input_fp32)
        self.assertIn(
            "CudaFp8DeepGEMMLinear only accepts bfloat16 input", str(context.exception)
        )
        self.assertIn("torch.float32", str(context.exception))

        # Test with float16 input (should raise ValueError)
        input_fp16 = torch.randn(32, self.K, dtype=torch.float16, device=self.device)
        with self.assertRaises(ValueError) as context:
            fp8_linear(input_fp16)
        self.assertIn(
            "CudaFp8DeepGEMMLinear only accepts bfloat16 input", str(context.exception)
        )
        self.assertIn("torch.float16", str(context.exception))

        # Test with int32 input (should raise ValueError)
        input_int32 = torch.randint(
            0, 10, (32, self.K), dtype=torch.int32, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            fp8_linear(input_int32)
        self.assertIn(f"Input tensor dtype must be bfloat16", str(context.exception))
        self.assertIn("torch.int32", str(context.exception))

    def test_input_dimension_validation(self):
        """Test input dimension validation - only accept 2D tensor"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=False)

        # Test with 1D tensor (should raise ValueError)
        input_tensor = torch.randn(self.K, dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError) as context:
            fp8_linear(input_tensor)
        self.assertIn(f"Input tensor dimension must be 2", str(context.exception))
        self.assertIn(f"got {input_tensor.dim()}D tensor", str(context.exception))

        # Test with 3D tensor (should raise ValueError)
        input_tensor = torch.randn(
            32, self.K, 2, dtype=torch.bfloat16, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            fp8_linear(input_tensor)
        self.assertIn(f"Input tensor dimension must be 2", str(context.exception))
        self.assertIn(f"got {input_tensor.dim()}D tensor", str(context.exception))

        # Test with input tensor inner dimension not expected to be K (should raise ValueError)
        input_tensor = torch.randn(
            32, self.K + 1, dtype=torch.bfloat16, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            fp8_linear(input_tensor)
        self.assertIn(
            f"Input tensor inner dimension expected to be {self.K}",
            str(context.exception),
        )
        self.assertIn(f"got {input_tensor.shape[1]}", str(context.exception))

    def test_bias_validation(self):
        """Test bias validation"""
        # Dependencies must be available for unit tests

        bias = torch.randn((1, self.N, 1), dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError) as context:
            Fp8PerBlockLinear(self.weight, self.weight_scale, bias)
        self.assertIn(f"Bias dimension must be 1 or 2, ", str(context.exception))
        self.assertIn(f"got {bias.dim()}", str(context.exception))

        bias = torch.randn((1, self.N + 1), dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError) as context:
            Fp8PerBlockLinear(self.weight, self.weight_scale, bias)
        self.assertIn(f"Bias last dimension must be {self.N}, ", str(context.exception))
        self.assertIn(f"got {bias.shape[-1]}", str(context.exception))

        bias = torch.randn((self.N - 1,), dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError) as context:
            Fp8PerBlockLinear(self.weight, self.weight_scale, bias)
        self.assertIn(f"Bias last dimension must be {self.N}, ", str(context.exception))
        self.assertIn(f"got {bias.shape[-1]}", str(context.exception))

        bias = torch.randn((2, self.N), dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError) as context:
            Fp8PerBlockLinear(self.weight, self.weight_scale, bias)
        self.assertIn(f"Bias first dimension must be 1, ", str(context.exception))
        self.assertIn(f"got {bias.shape[0]}", str(context.exception))

        bias = torch.randn((1, self.N), dtype=torch.float32, device=self.device)
        with self.assertRaises(ValueError) as context:
            Fp8PerBlockLinear(self.weight, self.weight_scale, bias)
        self.assertIn(f"Bias dtype must be bfloat16, ", str(context.exception))
        self.assertIn("got torch.float32", str(context.exception))

    def test_forward_pass_with_dependencies(self):
        """Test forward pass when dependencies are available"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=True)

        for batch_size in self.batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.randn(
                    batch_size,
                    self.K,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                output = fp8_linear(input_tensor)

                expected_shape = (batch_size, self.N)
                self.assertEqual(output.shape, expected_shape)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(output.device.type, "cuda")
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_output_shape_validation(self):
        """Test output shape validation - only accept 2D tensor"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=False)

        # Test various batch sizes including edge cases
        test_batch_sizes = [1, 63, 64, 65, 127, 128, 129, 255, 256]

        for batch_size in test_batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.randn(
                    batch_size,
                    self.K,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                output = fp8_linear(input_tensor)
                self.assertEqual(output.shape, (batch_size, self.N))
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_bias_handling(self):
        """Test bias handling"""
        # Dependencies must be available for unit tests

        input_tensor = torch.randn(32, self.K, dtype=torch.bfloat16, device=self.device)

        fp8_linear_with_bias = self._create_fp8_linear(
            with_bias=True, with_bias_2d=False
        )
        output_with_bias = fp8_linear_with_bias(input_tensor)
        self.assertEqual(output_with_bias.shape, (32, self.N))

        fp8_linear_with_bias_2d = self._create_fp8_linear(
            with_bias=True, with_bias_2d=True
        )
        output_with_bias_2d = fp8_linear_with_bias_2d(input_tensor)
        self.assertEqual(output_with_bias_2d.shape, (32, self.N))

        fp8_linear_no_bias = self._create_fp8_linear(with_bias=False)
        output_no_bias = fp8_linear_no_bias(input_tensor)
        self.assertEqual(output_no_bias.shape, (32, self.N))

    def test_small_batch_sizes(self):
        """Test small batch size edge cases"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=True)

        input_1 = torch.randn(1, self.K, dtype=torch.bfloat16, device=self.device)
        output_1 = fp8_linear(input_1)
        self.assertEqual(output_1.shape, (1, self.N))

        input_8 = torch.randn(8, self.K, dtype=torch.bfloat16, device=self.device)
        output_8 = fp8_linear(input_8)
        self.assertEqual(output_8.shape, (8, self.N))

    def test_reproducibility(self):
        """Test result reproducibility"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=False)

        torch.manual_seed(42)
        input_tensor = torch.randn(64, self.K, dtype=torch.bfloat16, device=self.device)
        output1 = fp8_linear(input_tensor)
        output2 = fp8_linear(input_tensor)
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)

    def test_weight_reshaping(self):
        """Test weight reshaping logic"""
        fp8_linear = self._create_fp8_linear(with_bias=False)

        expected_weight_shape = (self.N, self.K)
        self.assertEqual(fp8_linear.weight.shape, expected_weight_shape)

        expected_scales_shape = (self.N // 128, self.K // 128)
        self.assertEqual(fp8_linear.weight_scale.shape, expected_scales_shape)

    def test_fp8_vs_quantized_bf16_accuracy(self):
        """Test accuracy comparison between FP8 linear and quant->dequant BF16 linear"""
        # Dependencies must be available for unit tests

        # Create FP8 linear layer
        fp8_linear = self._create_fp8_linear(with_bias=False)
        # Create equivalent BF16 linear layer with quantized weights
        bf16_linear = torch.nn.Linear(
            self.K,
            self.N,
            bias=False,
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Quantize BF16 weights to FP8 and then dequantize back to BF16
        with torch.no_grad():
            # Get the original FP8 weights
            weight_fp8 = self.weight.reshape(self.N, self.K)
            weight_scale = self.weight_scale.reshape(self.N // 128, self.K // 128)
            # Dequantize FP8 weights back to BF16
            weight_bf16_dequant = self._dequantize_fp8_to_bf16(weight_fp8, weight_scale)
            # Set the dequantized weights to BF16 linear layer
            bf16_linear.weight.data = weight_bf16_dequant.reshape(self.N, self.K)

        # Test with various batch sizes
        test_batch_sizes = [1, 16, 32, 64, 128]

        for batch_size in test_batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Generate test input
                torch.manual_seed(42)  # For reproducibility
                input_tensor = torch.randn(
                    batch_size,
                    self.K,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                # Forward pass through FP8 linear
                fp8_output = fp8_linear(input_tensor)
                # Forward pass through quantized BF16 linear
                bf16_output = bf16_linear(input_tensor)
                # Compare outputs
                # FP8 should be close to quantized BF16, but not identical due to quantization differences
                max_diff = torch.max(torch.abs(fp8_output - bf16_output)).item()
                mean_diff = torch.mean(torch.abs(fp8_output - bf16_output)).item()
                # Print comparison for debugging
                print(
                    f"Batch size {batch_size}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
                )
                # The difference should be reasonable (within quantization error)
                # FP8 quantization introduces some error, so we allow a reasonable tolerance
                # FP8 has limited precision, so we use more relaxed tolerances
                self.assertLess(max_diff, 0.15, f"Max difference too large: {max_diff}")
                self.assertLess(
                    mean_diff, 0.025, f"Mean difference too large: {mean_diff}"
                )
                # Both outputs should have the same shape and dtype
                self.assertEqual(fp8_output.shape, bf16_output.shape)
                self.assertEqual(fp8_output.dtype, bf16_output.dtype)
            print()

    def test_common_batch_sizes(self):
        """Test common batch sizes"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=False)
        bf16_linear = torch.nn.Linear(
            self.K,
            self.N,
            bias=False,
            dtype=torch.bfloat16,
            device=self.device,
        )
        weight_fp8 = self.weight.reshape(self.N, self.K)
        weight_scale = self.weight_scale.reshape(self.N // 128, self.K // 128)
        weight_bf16 = self._dequantize_fp8_to_bf16(weight_fp8, weight_scale)
        bf16_linear.weight.data = weight_bf16.reshape(self.N, self.K)

        # Quantize BF16 weights to FP8 and then dequantize back to BF16
        with torch.no_grad():
            # Get the original FP8 weights
            weight_fp8 = self.weight.reshape(self.N, self.K)
            weight_scale = self.weight_scale.reshape(self.N // 128, self.K // 128)
            # Dequantize FP8 weights back to BF16
            weight_bf16_dequant = self._dequantize_fp8_to_bf16(weight_fp8, weight_scale)
            # Set the dequantized weights to BF16 linear layer
            bf16_linear.weight.data = weight_bf16_dequant.reshape(self.N, self.K)

        for batch_size in range(1, 4097):
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.randn(
                    batch_size, self.K, dtype=torch.bfloat16, device=self.device
                )
                ref_output = bf16_linear(input_tensor)
                output = fp8_linear(input_tensor)
                self.assertEqual(output.shape, (batch_size, self.N))
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())
                max_diff = torch.max(torch.abs(output - ref_output)).item()
                mean_diff = torch.mean(torch.abs(output - ref_output)).item()
                self.assertLess(max_diff, 0.15, f"Max difference too large: {max_diff}")
                self.assertLess(
                    mean_diff, 0.025, f"Mean difference too large: {mean_diff}"
                )

    def _dequantize_fp8_to_bf16(
        self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        """Helper method to dequantize FP8 weights back to BF16"""
        scales_h = weight_scale.repeat_interleave(128, dim=0)[: self.N]
        scales_expanded = scales_h.repeat_interleave(128, dim=1)[:, : self.K]
        weight_bf16 = weight_fp8.float() * scales_expanded
        return weight_bf16.to(torch.bfloat16)

    @unittest.skip("Skip profiling tests")
    def test_profile_fp8_per_block_linear(self):
        """Profile FP8 per block linear"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise RuntimeError(
                "FP8 per block linear tests require CUDA, but CUDA is not available"
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
        weight_scale = (
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
        fp8_linear = Fp8PerBlockLinear(weight, weight_scale)
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
                lambda: fp8_linear(input_tensor),
                num_warmups=20,
                num_tests=30,
                suppress_kineto_output=True,
                barrier_comm_profiling=False,
                trace_path=f"./trace_files/fp8_per_block_linear_{batch_size}.json",
            )
            print(f"Batch size {batch_size}: t_mean_new={t_mean_new:.6f}s")


if __name__ == "__main__":
    unittest.main()
