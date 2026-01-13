import logging
import os
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
    CudaFp4GEMMLinear,
    has_flashinfer_fp4,
)

from flashinfer import fp4_quantize
from rtp_llm.config.quant_config import init_quant_config


class CudaFp4GEMMLinearTest(unittest.TestCase):

    def setUp(self):
        """Setup test environment"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.skipTest("FP4 tests require CUDA")

        if not has_flashinfer_fp4():
            self.skipTest("flashinfer FP4 support is not available")

        logging.getLogger(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear"
        ).setLevel(logging.WARNING)

        self.hidden_size = 1024  # k
        self.output_size = 512  # n
        self.batch_sizes = [1, 32, 64, 128, 256] # m
        weight_fp16 = (
            torch.randn(
                self.output_size,
                self.hidden_size,
                dtype=torch.float16,
                device=self.device,
            )
            * 0.1
        )
        global_sf_weight = (448 * 6) / weight_fp16.float().abs().nan_to_num().max()
        self.weight_scale_2 = 1.0 / global_sf_weight.to(torch.float32)
        fp4_weight, weight_scale = fp4_quantize(weight_fp16,
                                                global_scale=global_sf_weight,
                                                is_sf_swizzled_layout=False)
        self.weight = fp4_weight
        self.weight_scales = weight_scale

        self.bias = torch.randn(
            self.output_size, dtype=torch.bfloat16, device=self.device
        )
        self.weight_fp16 = weight_fp16
        self.input_scale = self.weight_scale_2

    def _create_fp4_linear(self, with_bias: bool = True):
        """Helper method to create CudaFp4GEMMLinear instance"""
        return CudaFp4GEMMLinear(
            weight=self.weight,
            weight_scales=self.weight_scales,
            input_scales=self.input_scale,
            bias=self.bias if with_bias else None,
            quant_config=init_quant_config('modelopt_fp4'),
            weight_scale_2=self.weight_scale_2
        )

    def test_module_creation(self):
        """Test CudaFp4GEMMLinear module creation"""
        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "cutlass"
        fp4_linear = self._create_fp4_linear(with_bias=True)
        self.assertEqual(fp4_linear.hidden_size, self.hidden_size)
        self.assertEqual(fp4_linear.output_size, self.output_size)
        self.assertEqual(fp4_linear.backend, "cutlass")
        self.assertIsNotNone(fp4_linear.weight)
        self.assertIsNotNone(fp4_linear.weight_scales)
        self.assertIsNotNone(fp4_linear.weight_scale_2)
        self.assertIsNotNone(fp4_linear.input_scale)
        self.assertIsNotNone(fp4_linear.bias)

        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "trtllm"
        fp4_linear_no_bias = self._create_fp4_linear(with_bias=False)
        self.assertEqual(fp4_linear_no_bias.backend, "trtllm")
        self.assertEqual(fp4_linear_no_bias.hidden_size, self.hidden_size)
        self.assertEqual(fp4_linear_no_bias.output_size, self.output_size)
        self.assertIsNone(fp4_linear_no_bias.bias)

        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "sgl_cutlass"
        fp4_linear_sgl_cutlass = self._create_fp4_linear(with_bias=False)
        self.assertEqual(fp4_linear_sgl_cutlass.backend, "sgl_cutlass")
        self.assertEqual(fp4_linear_sgl_cutlass.hidden_size, self.hidden_size)
        self.assertEqual(fp4_linear_sgl_cutlass.output_size, self.output_size)
        self.assertIsNone(fp4_linear_sgl_cutlass.bias)

    def test_dependency_availability(self):
        """Test dependency availability check"""
        # Test that we can at least import the module
        self.assertIsNotNone(CudaFp4GEMMLinear)
        self.assertTrue(has_flashinfer_fp4())

    def test_input_dtype_validation(self):
        """Test input dtype validation - bfloat16 and float16 are accepted"""

        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "trtllm"
        fp4_linear = self._create_fp4_linear(with_bias=False)
        self.assertEqual(fp4_linear.backend, "trtllm")

        # Test with bfloat16 input (should work)
        input_bf16 = torch.randn(
            32, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )
        output = fp4_linear(input_bf16)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (32, self.output_size))

        # Test with float32 input (should raise ValueError)
        input_fp32 = torch.randn(
            32, self.hidden_size, dtype=torch.float32, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            fp4_linear(input_fp32)
        self.assertIn(
            "CudaFp4GEMMLinear accepts bfloat16 and float16 input", str(context.exception)
        )
        self.assertIn("torch.float32", str(context.exception))

        # Test with float16 input and backend:trtllm (should raise ValueError)
        input_fp16 = torch.randn(
            32, self.hidden_size, dtype=torch.float16, device=self.device
        )
        with self.assertRaises(ValueError) as context:
            fp4_linear(input_fp16)
        self.assertIn(
            "CudaFp4GEMMLinear with trtllm backend only supoorts bfloat16 input",
            str(context.exception)
        )
        self.assertIn("torch.float16", str(context.exception))

    def test_forward_pass_with_dependencies(self):
        """Test forward pass when dependencies are available"""
        fp4_linear = self._create_fp4_linear(with_bias=True)

        for batch_size in self.batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.randn(
                    batch_size,
                    self.hidden_size,
                    dtype=torch.bfloat16,
                    device=self.device,
                )

                output = fp4_linear(input_tensor)

                expected_shape = (batch_size, self.output_size)
                self.assertEqual(output.shape, expected_shape)

                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(output.device.type, "cuda")

                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_various_batch_sizes(self):
        """Test various batch sizes including edge cases"""
        fp4_linear = self._create_fp4_linear(with_bias=False)

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
                output = fp4_linear(input_tensor)
                self.assertEqual(output.shape, (batch_size, self.output_size))
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_bias_handling(self):
        """Test bias handling"""
        fp4_linear_with_bias = self._create_fp4_linear(
            with_bias=True
        )
        input_tensor = torch.randn(
            32, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )

        output_with_bias = fp4_linear_with_bias(input_tensor)
        self.assertEqual(output_with_bias.shape, (32, self.output_size))

        fp4_linear_no_bias = self._create_fp4_linear(
            with_bias=False
        )
        output_no_bias = fp4_linear_no_bias(input_tensor)
        self.assertEqual(output_no_bias.shape, (32, self.output_size))

        # Outputs should be different when bias is present
        self.assertFalse(torch.allclose(output_with_bias, output_no_bias))

    def test_small_batch_sizes(self):
        """Test small batch size edge cases"""
        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "cutlass"
        fp4_linear = self._create_fp4_linear(with_bias=True)

        input_1 = torch.randn(
            1, self.hidden_size, dtype=torch.float16, device=self.device
        )
        output_1 = fp4_linear(input_1)
        self.assertEqual(output_1.shape, (1, self.output_size))

        input_8 = torch.randn(
            8, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )
        output_8 = fp4_linear(input_8)
        self.assertEqual(output_8.shape, (8, self.output_size))

    def test_reproducibility(self):
        """Test result reproducibility"""
        # cutlass backend
        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "cutlass"
        fp4_linear = self._create_fp4_linear(with_bias=False)

        torch.manual_seed(42)
        input_tensor = torch.randn(
            64, self.hidden_size, dtype=torch.float16, device=self.device
        )

        output1 = fp4_linear(input_tensor)
        output2 = fp4_linear(input_tensor)
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)

        # trtllm backend
        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "trtllm"
        fp4_linear = self._create_fp4_linear(with_bias=True)

        torch.manual_seed(42)
        input_tensor = torch.randn(
            64, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )

        output1 = fp4_linear(input_tensor)
        output2 = fp4_linear(input_tensor)
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)

    def test_error_handling(self):
        """Test error handling"""
        fp4_linear = self._create_fp4_linear(with_bias=False)

        wrong_input = torch.randn(
            32, self.hidden_size + 1, dtype=torch.bfloat16, device=self.device
        )
        try:
            fp4_linear(wrong_input)
        except Exception:
            pass

    def test_weight_shape(self):
        """Test weight shape"""
        fp4_linear = self._create_fp4_linear(with_bias=False)

        # Weight should be stored in original shape [n, k]
        expected_weight_shape = (self.output_size, self.hidden_size // 2)
        self.assertEqual(fp4_linear.weight.shape, expected_weight_shape)
        
    def test_fp4_vs_bf16_accuracy(self):
        """Test accuracy comparison between FP4 linear and BF16 linear"""
        # Create FP4 linear layer, cutlass backend
        self._test_fp4_vs_bf16_accuracy_backend("cutlass")
        self._test_fp4_vs_bf16_accuracy_backend("trtllm")
        self._test_fp4_vs_bf16_accuracy_backend("sgl_cutlass")

    def _test_fp4_vs_bf16_accuracy_backend(self, backend):
        """Test accuracy comparison between FP4 linear and BF16 linear with different backends"""
        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = backend
        fp4_linear = self._create_fp4_linear(with_bias=False)

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

                fp4_linear.input_scale = 1.0 / (448.0 * 6.0 / input_tensor.float().abs().nan_to_num().max())
                fp4_linear.alpha = fp4_linear.input_scale * fp4_linear.weight_scale_2
                fp4_linear.input_scale_inv = 1.0 / fp4_linear.input_scale
                # Forward pass through FP4 linear
                fp4_output = fp4_linear(input_tensor)

                # Forward pass through BF16 linear
                bf16_output = (input_tensor.float() @ self.weight_fp16.float().t()).to(
                    torch.bfloat16
                )
                print(f"backend: {backend}, fp4_output: {fp4_output.float()}")
                print(f"backend: {backend}, bf16_output: {bf16_output.float()}")
                diff = calc_diff(fp4_output, bf16_output)
                self.assertLess(diff, 0.01)

                # Both outputs should have the same shape and dtype
                self.assertEqual(fp4_output.shape, bf16_output.shape)
                self.assertEqual(fp4_output.dtype, bf16_output.dtype)
                self.assertFalse(torch.isnan(fp4_output).any())
                self.assertFalse(torch.isinf(fp4_output).any())

    def test_fp4_vs_fp16_accuracy(self):
        """Test accuracy comparison between FP4 linear and BF16 linear"""
        self._test_fp4_vs_fp16_accuracy_backend("cutlass")
        self._test_fp4_vs_fp16_accuracy_backend("sgl_cutlass")

    def _test_fp4_vs_fp16_accuracy_backend(self, backend):
        """Test accuracy comparison between FP4 linear and BF16 linear with different backends"""
        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = backend
        fp4_linear = self._create_fp4_linear(with_bias=False)

        # Test with various batch sizes
        test_batch_sizes = [1, 16, 32, 64, 128]

        for batch_size in test_batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Generate test input
                torch.manual_seed(42)  # For reproducibility
                input_tensor = torch.randn(
                    batch_size,
                    self.hidden_size,
                    dtype=torch.float16,
                    device=self.device,
                )
                fp4_linear.input_scale = 1.0 / (448.0 * 6.0 / input_tensor.float().abs().nan_to_num().max())
                fp4_linear.alpha = fp4_linear.input_scale * fp4_linear.weight_scale_2
                fp4_linear.input_scale_inv = 1.0 / fp4_linear.input_scale
                # Forward pass through FP4 linear
                fp4_output = fp4_linear(input_tensor)

                # Forward pass through BF16 linear
                fp16_output = (input_tensor.float() @ self.weight_fp16.float().t()).to(
                    torch.float16
                )
                print(f"backend: {backend}, fp4_output: {fp4_output.float()}")
                print(f"backend: {backend}, fp16_output: {fp16_output.float()}")
                diff = calc_diff(fp4_output, fp16_output)
                self.assertLess(diff, 0.01)

                # Both outputs should have the same shape and dtype
                self.assertEqual(fp4_output.shape, fp16_output.shape)
                self.assertEqual(fp4_output.dtype, fp16_output.dtype)
                self.assertFalse(torch.isnan(fp4_output).any())
                self.assertFalse(torch.isinf(fp4_output).any())


if __name__ == "__main__":
    unittest.main()

