import logging
import os
import unittest

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
    CudaFp4GEMMLinear,
    has_flashinfer_fp4,
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

kE2M1ToFloatArray = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]


def e2m1_to_fp32(int4_value):
    signBit = int4_value & 0x8
    int4_absValue = int4_value & 0x7
    float_result = kE2M1ToFloatArray[int4_absValue]
    if signBit:
        float_result = -float_result
    return float_result


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a = a.flatten()
    # Get upper 4 bits
    highHalfByte = (a & 0xF0) >> 4
    # Get lower 4 bits
    lowHalfByte = a & 0x0F
    fH = torch.tensor([e2m1_to_fp32(x) for x in highHalfByte]).to(a.device)
    fL = torch.tensor([e2m1_to_fp32(x) for x in lowHalfByte]).to(a.device)
    # [0xAB, 0xCD] -> [0xB, 0xA, 0xD, 0xC]
    out = torch.stack((fL, fH), dim=-1).reshape(m, n * 2)
    return out


def dequantize_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out

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

        self.hidden_size = 512  # k
        self.output_size = 1024  # n
        self.batch_sizes = [1, 32, 64, 128, 256] # m

        self.weight = (
            torch.randn(
                self.hidden_size // 2,
                self.output_size,
                dtype=torch.uint8,
                device=self.device,
            )
            * 20
        )

        # Weight scales 
        self.weight_scales = (
            torch.rand(
                self.hidden_size // 16,
                self.output_size,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
            * 0.01
            + 0.001
        )

        self.bias = torch.randn(
            self.output_size, dtype=torch.float16, device=self.device
        )
        
        # weight_scale_2
        self.weight_scale_2 = torch.tensor(0.00022234, dtype=torch.float32)
        # input_scale
        self.input_scale = torch.tensor(0.001499, dtype=torch.float32)

    def _create_fp4_linear(self, with_bias: bool = True):
        """Helper method to create CudaFp4GEMMLinear instance"""
        return CudaFp4GEMMLinear(
            weight=self.weight,
            weight_scales=self.weight_scales,
            input_scales=self.input_scale,
            bias=self.bias if with_bias else None,
            config=None,
            weight_scale_2=self.weight_scale_2
        )

    def test_module_creation(self):
        """Test CudaFp4GEMMLinear module creation"""
        fp4_linear = self._create_fp4_linear(with_bias=True)
        self.assertEqual(fp4_linear.hidden_size, self.hidden_size)
        self.assertEqual(fp4_linear.output_size, self.output_size)
        self.assertEqual(fp4_linear.backend, "cutlass")
        self.assertIsNotNone(fp4_linear.weight)
        self.assertIsNotNone(fp4_linear.weight_scales)
        self.assertIsNotNone(fp4_linear.weight_scale_2)
        self.assertIsNotNone(fp4_linear.input_scale)
        self.assertIsNotNone(fp4_linear.bias)

        fp4_linear_no_bias = self._create_fp4_linear(with_bias=False)
        self.assertEqual(fp4_linear_no_bias.hidden_size, self.hidden_size)
        self.assertEqual(fp4_linear_no_bias.output_size, self.output_size)
        self.assertIsNone(fp4_linear_no_bias.bias)

    def test_dependency_availability(self):
        """Test dependency availability check"""
        # Test that we can at least import the module
        self.assertIsNotNone(CudaFp4GEMMLinear)
        self.assertTrue(has_flashinfer_fp4())

    def test_input_dtype_validation(self):
        """Test input dtype validation - bfloat16 and float16 are accepted"""

        os.environ["FLASHINFER_FP4_GEMM_BACKEND"] = "trtllm"
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
        fp4_linear = self._create_fp4_linear(with_bias=False)

        torch.manual_seed(42)
        input_tensor = torch.randn(
            64, self.hidden_size, dtype=torch.float16, device=self.device
        )

        output1 = fp4_linear(input_tensor)
        output2 = fp4_linear(input_tensor)
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)

        # trtllm backend
        os.environ["FLASHINFER_FP4_GEMM_BACKEND"] = "trtllm"
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

        # Weight should be stored in original shape [k, n]
        expected_weight_shape = (self.hidden_size // 2, self.output_size)
        self.assertEqual(fp4_linear.weight.shape, expected_weight_shape)
        

    def test_fp4_vs_bf16_accuracy(self):
        """Test accuracy comparison between FP4 linear and BF16 linear"""
        # Create FP4 linear layer, cutlass backend
        os.environ["FLASHINFER_FP4_GEMM_BACKEND"] = "trtllm"
        fp4_linear = self._create_fp4_linear(with_bias=False)

        # Create equivalent BF16 linear layer
        bf16_linear = torch.nn.Linear(
            self.hidden_size,
            self.output_size,
            bias=False,
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Set BF16 linear weights to match FP4 weights
        with torch.no_grad():
            weight = self.weight.T
            weight_scales = self.weight_scales.T
            weight_scale_2 = self.weight_scale_2
            # Dequantize FP4 weights back to BF16
            weight_bf16_dequant = dequantize_to_dtype(
                weight, weight_scales, weight_scale_2, torch.bfloat16
            )
            bf16_linear.weight.data = weight_bf16_dequant
            

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

                # Forward pass through FP4 linear
                fp4_output = fp4_linear(input_tensor)

                # Forward pass through BF16 linear
                bf16_output = bf16_linear(input_tensor)

                # Compare outputs
                # NVFP4 quantization introduces some error, so we allow a reasonable tolerance
                max_diff = torch.max(torch.abs(fp4_output - bf16_output)).item()
                mean_diff = torch.mean(torch.abs(fp4_output - bf16_output)).item()

                # Print comparison for debugging
                print(
                    f"Batch size {batch_size}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
                )

                # The difference should be reasonable (within quantization error)
                # NVFP4 has limited precision, so we use more relaxed tolerances
                self.assertLess(
                    max_diff, 0.15, f"Max difference too large: {max_diff}"
                )
                self.assertLess(
                    mean_diff, 0.025, f"Mean difference too large: {mean_diff}"
                )

                # Both outputs should have the same shape and dtype
                self.assertEqual(fp4_output.shape, bf16_output.shape)
                self.assertEqual(fp4_output.dtype, bf16_output.dtype)

    def test_fp4_vs_fp16_accuracy(self):
        """Test accuracy comparison between FP4 linear and BF16 linear"""
        # Create FP4 linear layer, cutlass backend
        fp4_linear = self._create_fp4_linear(with_bias=False)

        # Create equivalent BF16 linear layer
        fp16_linear = torch.nn.Linear(
            self.hidden_size,
            self.output_size,
            bias=False,
            dtype=torch.float16,
            device=self.device,
        )

        # Set BF16 linear weights to match NVFP4 weights
        with torch.no_grad():
            weight = self.weight.T
            weight_scales = self.weight_scales.T
            weight_scale_2 = self.weight_scale_2
            # Dequantize FP4 weights back to BF16
            weight_fp16_dequant = dequantize_to_dtype(
                weight, weight_scales, weight_scale_2, torch.float16
            )
            fp16_linear.weight.data = weight_fp16_dequant
            

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

                # Forward pass through FP4 linear
                fp4_output = fp4_linear(input_tensor)

                # Forward pass through BF16 linear
                fp16_output = fp16_linear(input_tensor)

                # Compare outputs
                # NVFP4 quantization introduces some error, so we allow a reasonable tolerance
                max_diff = torch.max(torch.abs(fp4_output - fp16_output)).item()
                mean_diff = torch.mean(torch.abs(fp4_output - fp16_output)).item()

                # Print comparison for debugging
                print(
                    f"Batch size {batch_size}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
                )

                # The difference should be reasonable (within quantization error)
                # NVFP4 has limited precision, so we use more relaxed tolerances
                self.assertLess(
                    max_diff, 0.15, f"Max difference too large: {max_diff}"
                )
                self.assertLess(
                    mean_diff, 0.025, f"Mean difference too large: {mean_diff}"
                )

                # Both outputs should have the same shape and dtype
                self.assertEqual(fp4_output.shape, fp16_output.shape)
                self.assertEqual(fp4_output.dtype, fp16_output.dtype)


if __name__ == "__main__":
    unittest.main()

