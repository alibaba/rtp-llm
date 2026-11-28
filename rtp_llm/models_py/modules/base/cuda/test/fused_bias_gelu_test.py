import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class FusedBiasGeluTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

    def test_matches_exact_gelu(self):
        torch.manual_seed(20260811)
        for dtype in (torch.float16, torch.bfloat16):
            for shape in ((1, 768), (17, 769), (93, 3072), (4096, 3072)):
                with self.subTest(dtype=dtype, shape=shape):
                    value = torch.randn(shape, device="cuda", dtype=dtype)
                    bias = torch.randn(shape[-1], device="cuda", dtype=dtype)
                    expected = F.gelu(value + bias)
                    actual = value.clone()
                    rtp_llm_ops.fused_bias_gelu(actual, bias)
                    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_accepts_row_bias(self):
        value = torch.randn((93, 3072), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn((1, 3072), device="cuda", dtype=torch.bfloat16)
        expected = F.gelu(value + bias)
        rtp_llm_ops.fused_bias_gelu(value, bias)
        torch.testing.assert_close(value, expected, rtol=2e-2, atol=2e-2)

    def test_empty_input(self):
        value = torch.empty((0, 769), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(769, device="cuda", dtype=torch.bfloat16)
        rtp_llm_ops.fused_bias_gelu(value, bias)
        self.assertEqual(value.shape, (0, 769))

    def test_misaligned_contiguous_tensors(self):
        for dtype in (torch.float16, torch.bfloat16):
            value_storage = torch.randn(3 * 768 + 1, device="cuda", dtype=dtype)
            bias_storage = torch.randn(768 + 1, device="cuda", dtype=dtype)
            value = value_storage[1:].reshape(3, 768)
            bias = bias_storage[1:]
            self.assertTrue(value.is_contiguous())
            self.assertTrue(bias.is_contiguous())
            expected = F.gelu(value.clone() + bias)
            rtp_llm_ops.fused_bias_gelu(value, bias)
            torch.testing.assert_close(value, expected, rtol=2e-2, atol=2e-2)

    def test_cuda_graph_capture(self):
        value = torch.randn((128, 3072), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(3072, device="cuda", dtype=torch.bfloat16)
        static = value.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rtp_llm_ops.fused_bias_gelu(static, bias)
        static.copy_(value)
        graph.replay()
        torch.testing.assert_close(static, F.gelu(value + bias), rtol=2e-2, atol=2e-2)

    def test_fused_quant_matches_separate_path(self):
        torch.manual_seed(20260812)
        value = torch.randn((17, 3072), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(3072, device="cuda", dtype=torch.bfloat16)
        activated = value.clone()
        rtp_llm_ops.fused_bias_gelu(activated, bias)
        expected_q, expected_s = sgl_per_token_group_quant_fp8(
            activated,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        actual_q = torch.empty_like(value, dtype=torch.float8_e4m3fn)
        actual_s = create_per_token_group_quant_fp8_output_scale(
            value.shape,
            value.device,
            group_size=128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        for shaped_bias in (bias, bias.unsqueeze(0)):
            with self.subTest(bias_shape=shaped_bias.shape):
                actual_q.view(torch.uint8).zero_()
                actual_s.zero_()
                rtp_llm_ops.fused_bias_gelu_quant_fp8(
                    value, shaped_bias, actual_q, actual_s
                )
                self.assertTrue(
                    torch.equal(
                        actual_q.view(torch.uint8), expected_q.view(torch.uint8)
                    )
                )
                self.assertTrue(torch.equal(actual_s, expected_s))

    def test_quantized_layernorm_preserves_rounding_across_warps(self):
        torch.manual_seed(113)
        for dtype in (torch.float16, torch.bfloat16):
            for width in (128, 384, 768, 1024):
                with self.subTest(dtype=dtype, width=width):
                    x, residual = torch.randn(2, 67, width, device="cuda", dtype=dtype)
                    bias, weight, beta = torch.randn(
                        3, width, device="cuda", dtype=dtype
                    )
                    reference, reference_residual = x.clone(), residual.clone()
                    rtp_llm_ops.fused_add_layernorm(
                        reference, reference_residual, bias, weight, beta, 1e-6
                    )
                    expected_q, expected_scales = sgl_per_token_group_quant_fp8(
                        reference,
                        128,
                        eps=1e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    quantized = torch.empty_like(expected_q)
                    scales = create_per_token_group_quant_fp8_output_scale(
                        x.shape, x.device, 128, True, True, True
                    )
                    rtp_llm_ops.fused_add_layernorm_quant_fp8(
                        x, residual, bias, weight, beta, quantized, scales, 1e-6
                    )
                    self.assertTrue(torch.equal(x, reference))
                    self.assertTrue(torch.equal(residual, reference_residual))
                    self.assertTrue(
                        torch.equal(
                            quantized.view(torch.uint8), expected_q.view(torch.uint8)
                        )
                    )
                    # Only valid groups occupy bytes of the final UE8M0 word.
                    for group in range(width // 128):
                        col, byte = divmod(group, 4)
                        shift = 8 * byte
                        self.assertTrue(
                            torch.equal(
                                (scales[:, col] >> shift) & 255,
                                (expected_scales[:, col] >> shift) & 255,
                            )
                        )

    def test_quantized_layernorm_accepts_vector_and_row_bias(self):
        bias = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        bias[::2] = -1
        weight = bias.new_tensor([1, 2, 4, 8]).repeat_interleave(128)
        beta = torch.zeros_like(bias)
        expected = (bias * weight).expand(17, -1).contiguous()
        expected_q, expected_s = sgl_per_token_group_quant_fp8(
            expected,
            128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        for shaped_bias in (bias, bias.unsqueeze(0)):
            with self.subTest(bias_shape=shaped_bias.shape):
                value = torch.zeros_like(expected)
                residual = torch.zeros_like(expected)
                quantized = torch.empty_like(expected_q)
                scales = create_per_token_group_quant_fp8_output_scale(
                    expected.shape, expected.device, 128, True, True, True
                )
                rtp_llm_ops.fused_add_layernorm_quant_fp8(
                    value, residual, shaped_bias, weight, beta, quantized, scales, 1e-6
                )
                self.assertTrue(torch.equal(value, expected))
                self.assertTrue(torch.equal(residual, bias.expand_as(residual)))
                self.assertTrue(
                    torch.equal(
                        quantized.view(torch.uint8), expected_q.view(torch.uint8)
                    )
                )
                self.assertTrue(torch.equal(scales, expected_s))


if __name__ == "__main__":
    unittest.main()
