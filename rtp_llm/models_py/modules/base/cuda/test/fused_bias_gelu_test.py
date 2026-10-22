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
        rtp_llm_ops.fused_bias_gelu_quant_fp8(value, bias, actual_q, actual_s)
        torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=0)
        torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)

    def test_fused_quant_float_scale_matches_separate_path(self):
        torch.manual_seed(20260831)
        value = torch.randn((17, 3072), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(3072, device="cuda", dtype=torch.bfloat16)
        activated = value.clone()
        rtp_llm_ops.fused_bias_gelu(activated, bias)
        expected_q, expected_s = sgl_per_token_group_quant_fp8(
            activated,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )
        actual_q = torch.empty_like(value, dtype=torch.float8_e4m3fn)
        actual_s = torch.empty_strided(
            expected_s.shape,
            expected_s.stride(),
            dtype=torch.float32,
            device=value.device,
        )
        rtp_llm_ops.fused_bias_gelu_quant_fp8(value, bias, actual_q, actual_s)
        torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=0)
        torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)

    def test_fused_quant_float_scale_can_skip_bias(self):
        torch.manual_seed(20260831)
        value = torch.randn((17, 3072), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(3072, device="cuda", dtype=torch.bfloat16)
        expected_q, expected_s = sgl_per_token_group_quant_fp8(
            F.gelu(value),
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )
        actual_q = torch.empty_like(value, dtype=torch.float8_e4m3fn)
        actual_s = torch.empty_strided(
            expected_s.shape,
            expected_s.stride(),
            dtype=torch.float32,
            device=value.device,
        )
        rtp_llm_ops.fused_bias_gelu_quant_fp8(value, bias, actual_q, actual_s, False)
        torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=0)
        torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)

    def test_fused_quant_float_scale_rejects_overlapping_layout(self):
        value = torch.randn((17, 256), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(value, dtype=torch.float8_e4m3fn)
        scales = torch.empty_strided(
            (17, 2), (1, 1), dtype=torch.float32, device=value.device
        )
        with self.assertRaisesRegex(RuntimeError, "stride"):
            rtp_llm_ops.fused_bias_gelu_quant_fp8(value, bias, output, scales, False)


if __name__ == "__main__":
    unittest.main()
