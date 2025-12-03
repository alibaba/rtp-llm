import logging
import unittest

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)


class CudaFp8DeepGEMMLinearTest(unittest.TestCase):

    def setUp(self):
        """Setup test environment"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.skipTest("FP8 tests require CUDA")

        logging.getLogger(
            "rtp_llm.models_py.modules.cuda.linear.fp8_deepgemm_linear"
        ).setLevel(logging.WARNING)

        self.hidden_size = 512  # k
        self.output_size = 1024  # n
        self.batch_sizes = [1, 32, 64, 128, 256]

        weight_fp32 = (
            torch.randn(
                self.hidden_size,
                self.output_size,
                dtype=torch.float32,
                device=self.device,
            )
            * 0.1
        )
        self.weight = weight_fp32.to(torch.float8_e4m3fn)
        self.weight_scales = (
            torch.rand(
                (self.hidden_size + 127) // 128,
                (self.output_size + 127) // 128,
                dtype=torch.float32,
                device=self.device,
            )
            * 0.01
            + 0.001
        )

        self.bias = torch.randn(
            self.output_size, dtype=torch.bfloat16, device=self.device
        )

    def _create_fp8_linear(self, with_bias: bool = True):
        """Helper method to create CudaFp8DeepGEMMLinear instance"""
        return CudaFp8DeepGEMMLinear(
            weight=self.weight,
            weight_scales=self.weight_scales,
            bias=self.bias if with_bias else None,
            config=None,
        )

    def test_module_creation(self):
        """Test CudaFp8DeepGEMMLinear module creation"""
        fp8_linear = self._create_fp8_linear(with_bias=True)
        self.assertEqual(fp8_linear.hidden_size, self.hidden_size)
        self.assertEqual(fp8_linear.output_size, self.output_size)
        self.assertIsNotNone(fp8_linear.weight)
        self.assertIsNotNone(fp8_linear.weight_scales)
        self.assertIsNotNone(fp8_linear.bias)

        fp8_linear_no_bias = self._create_fp8_linear(with_bias=False)
        self.assertEqual(fp8_linear_no_bias.hidden_size, self.hidden_size)
        self.assertEqual(fp8_linear_no_bias.output_size, self.output_size)
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
        input_bf16 = torch.randn(
            32, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )
        output = fp8_linear(input_bf16)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (32, self.output_size))

        # Test with float32 input (should raise ValueError)
        input_fp32 = torch.randn(
            32, self.hidden_size, dtype=torch.float32, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            fp8_linear(input_fp32)
        self.assertIn(
            "CudaFp8DeepGEMMLinear only accepts bfloat16 input", str(context.exception)
        )
        self.assertIn("torch.float32", str(context.exception))

        # Test with float16 input (should raise ValueError)
        input_fp16 = torch.randn(
            32, self.hidden_size, dtype=torch.float16, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            fp8_linear(input_fp16)
        self.assertIn(
            "CudaFp8DeepGEMMLinear only accepts bfloat16 input", str(context.exception)
        )
        self.assertIn("torch.float16", str(context.exception))

        # Test with int32 input (should raise ValueError)
        input_int32 = torch.randint(
            0, 10, (32, self.hidden_size), dtype=torch.int32, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            fp8_linear(input_int32)
        self.assertIn(
            "CudaFp8DeepGEMMLinear only accepts bfloat16 input", str(context.exception)
        )
        self.assertIn("torch.int32", str(context.exception))

    def test_forward_pass_with_dependencies(self):
        """Test forward pass when dependencies are available"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=True)

        for batch_size in self.batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.randn(
                    batch_size,
                    self.hidden_size,
                    dtype=torch.bfloat16,
                    device=self.device,
                )

                output = fp8_linear(input_tensor)

                expected_shape = (batch_size, self.output_size)
                self.assertEqual(output.shape, expected_shape)

                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(output.device.type, "cuda")

                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_padding_logic(self):
        """Test padding logic for different input sizes"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=False)

        # Test various batch sizes including edge cases
        test_batch_sizes = [1, 7, 15, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256]

        for batch_size in test_batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.randn(
                    batch_size,
                    self.hidden_size,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                output = fp8_linear(input_tensor)
                self.assertEqual(output.shape, (batch_size, self.output_size))
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_bias_handling(self):
        """Test bias handling"""
        # Dependencies must be available for unit tests

        fp8_linear_with_bias = self._create_fp8_linear(with_bias=True)
        input_tensor = torch.randn(
            32, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )

        output_with_bias = fp8_linear_with_bias(input_tensor)
        self.assertEqual(output_with_bias.shape, (32, self.output_size))

        fp8_linear_no_bias = self._create_fp8_linear(with_bias=False)
        output_no_bias = fp8_linear_no_bias(input_tensor)
        self.assertEqual(output_no_bias.shape, (32, self.output_size))

    def test_small_batch_sizes(self):
        """Test small batch size edge cases"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=True)

        input_1 = torch.randn(
            1, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )
        output_1 = fp8_linear(input_1)
        self.assertEqual(output_1.shape, (1, self.output_size))

        input_8 = torch.randn(
            8, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )
        output_8 = fp8_linear(input_8)
        self.assertEqual(output_8.shape, (8, self.output_size))

    def test_reproducibility(self):
        """Test result reproducibility"""
        # Dependencies must be available for unit tests

        fp8_linear = self._create_fp8_linear(with_bias=False)

        torch.manual_seed(42)
        input_tensor = torch.randn(
            64, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )

        output1 = fp8_linear(input_tensor)
        output2 = fp8_linear(input_tensor)
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)

    def test_error_handling(self):
        """Test error handling"""
        fp8_linear = self._create_fp8_linear(with_bias=False)

        wrong_input = torch.randn(
            32, self.hidden_size + 1, dtype=torch.bfloat16, device=self.device
        )
        try:
            fp8_linear(wrong_input)
        except Exception:
            pass

    def test_weight_reshaping(self):
        """Test weight reshaping logic"""
        fp8_linear = self._create_fp8_linear(with_bias=False)

        expected_weight_shape = (self.output_size, self.hidden_size)
        self.assertEqual(fp8_linear.weight.shape, expected_weight_shape)

        expected_scales_shape = (self.output_size // 128, self.hidden_size // 128)
        self.assertEqual(fp8_linear.weight_scales.shape, expected_scales_shape)

    def test_fp8_vs_quantized_bf16_accuracy(self):
        """Test accuracy comparison between FP8 linear and quant->dequant BF16 linear"""
        # Dependencies must be available for unit tests

        # Create FP8 linear layer
        fp8_linear = self._create_fp8_linear(with_bias=False)

        # Create equivalent BF16 linear layer with quantized weights
        bf16_linear = torch.nn.Linear(
            self.hidden_size,
            self.output_size,
            bias=False,
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Quantize BF16 weights to FP8 and then dequantize back to BF16
        with torch.no_grad():
            # Get the original FP32 weights (before FP8 conversion)
            weight_fp32 = self.weight.float()

            # Quantize to FP8 and dequantize back to BF16
            weight_fp8 = weight_fp32.to(torch.float8_e4m3fn)
            weight_scales = self.weight_scales

            # Dequantize FP8 weights back to BF16
            weight_bf16_dequant = self._dequantize_fp8_to_bf16(
                weight_fp8, weight_scales
            )

            # Set the dequantized weights to BF16 linear layer
            bf16_linear.weight.data = weight_bf16_dequant.t()

        # Test with various batch sizes
        test_batch_sizes = [1, 16, 32, 64, 128]

        for batch_size in test_batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Generate test input
                torch.manual_seed(42)  # For reproducibility
                input_tensor = torch.randn(
                    batch_size,
                    self.hidden_size,
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

    def _dequantize_fp8_to_bf16(
        self, weight_fp8: torch.Tensor, weight_scales: torch.Tensor
    ) -> torch.Tensor:
        """Helper method to dequantize FP8 weights back to BF16"""
        # weight_fp8 shape: (hidden_size, output_size) = (512, 1024)
        # weight_scales shape: (hidden_size//128, output_size//128) = (4, 8)

        # Expand scales to match weight dimensions
        # First expand along hidden_size dimension (dim=0)
        scales_h = weight_scales.repeat_interleave(128, dim=0)[
            : self.hidden_size
        ]  # (512, 8)
        # Then expand along output_size dimension (dim=1)
        scales_expanded = scales_h.repeat_interleave(128, dim=1)[
            :, : self.output_size
        ]  # (512, 1024)

        # Dequantize: fp8_value * scale
        weight_bf16 = weight_fp8.float() * scales_expanded

        return weight_bf16.to(torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
