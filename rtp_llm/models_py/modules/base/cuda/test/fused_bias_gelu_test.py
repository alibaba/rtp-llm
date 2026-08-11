import unittest

import torch
import torch.nn.functional as F

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


if __name__ == "__main__":
    unittest.main()
