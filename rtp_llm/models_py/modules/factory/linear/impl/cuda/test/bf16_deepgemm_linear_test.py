"""Verify resumed projections and CUDA graph replay on SM100."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "requires SM100",
)
class Bf16DeepGemmLinearTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"RTP_BF16_LINEAR_BACKEND": "deepgemm"})
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(20260921)
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_resume_rows(self):
        for n, k in [(16, 4096), (128, 4096), (2048, 128), (4096, 2048), (6144, 4096)]:
            with self.subTest(n=n, k=k):
                weight = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
                layer = CudaF16Linear(weight)
                self.assertIsNotNone(layer._bf16_gemm)
                self.assertTrue(layer.weight.T.is_contiguous())
                x = torch.randn(3389, k, device="cuda", dtype=torch.bfloat16)
                full = layer(x)
                for m in [1, 13, 61, 129]:
                    suffix = layer(x[-m:])
                    self.assertTrue(torch.equal(full[-m:], suffix), (m, n, k))
                    reference = torch.nn.functional.linear(
                        x[-m:].float(), weight.T.float()
                    )
                    error = (suffix.float() - reference).norm() / reference.norm()
                    self.assertLess(error.item(), 0.005)
                self.assertEqual(tuple(layer(x[:0]).shape), (0, n))
                self.assertTrue(
                    torch.equal(layer(x[-13:].unsqueeze(0))[0], full[-13:])
                )

    def test_fallback(self):
        for dtype, n, k, use_bias in [
            (torch.float16, 16, 128, False),
            (torch.bfloat16, 17, 127, False),
            (torch.bfloat16, 16, 128, True),
        ]:
            with self.subTest(dtype=dtype, n=n, k=k, bias=use_bias):
                weight = torch.randn(k, n, device="cuda", dtype=dtype)
                bias = torch.randn(n, device="cuda", dtype=dtype) if use_bias else None
                layer = CudaF16Linear(weight, bias=bias)
                self.assertIsNone(layer._bf16_gemm)
                x = torch.randn(13, k, device="cuda", dtype=dtype)
                self.assertTrue(
                    torch.equal(layer(x), torch.nn.functional.linear(x, weight.T, bias))
                )

    def test_cuda_graph(self):
        layer = CudaF16Linear(
            torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16)
        )
        x = torch.randn(13, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            layer(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer(x)
        for _ in range(3):
            x.normal_()
            expected = layer(x)
            graph.replay()
            self.assertTrue(torch.equal(output, expected))


if __name__ == "__main__":
    unittest.main()
