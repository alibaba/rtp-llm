"""Keep HY4's final MTP residual and normalization on the same BF16 sum."""

import unittest

import torch

from rtp_llm.models_py.model_desc.generic_moe_mtp import _Hy4MtpFinalNorm


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHy4MtpFinalNorm(unittest.TestCase):
    def test_normalizes_returned_residual(self):
        for rows in (1, 4, 32, 128, 256, 1250, 4000, 8000, 12500):
            with self.subTest(rows=rows):
                generator = torch.Generator(device="cuda").manual_seed(931)
                hidden = torch.randn(
                    rows, 6144, device="cuda", generator=generator
                ).bfloat16()
                residual = (
                    torch.randn(rows, 6144, device="cuda", generator=generator)
                    * 0.003
                ).bfloat16()
                weight = torch.randn(
                    6144, device="cuda", generator=generator
                ).bfloat16()
                eps = 1e-6
                expected_residual = hidden + residual
                x = expected_residual.float()
                expected = (
                    x
                    * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
                    * weight.float()
                ).bfloat16()
                actual, actual_residual = _Hy4MtpFinalNorm(weight, eps)(
                    hidden, residual
                )
                self.assertIs(actual, hidden)
                self.assertIs(actual_residual, residual)
                torch.testing.assert_close(
                    actual_residual, expected_residual, rtol=0, atol=0
                )
                relative_l2 = (
                    (actual.float() - expected.float()).norm()
                    / expected.float().norm()
                ).item()
                self.assertLess(relative_l2, 1e-4)

    def test_cuda_graph_reuses_buffers(self):
        hidden = torch.randn(32, 6144, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(hidden)
        weight = torch.ones(6144, device="cuda", dtype=torch.bfloat16)
        norm = _Hy4MtpFinalNorm(weight, 1e-6)
        original_hidden, original_residual = hidden.clone(), residual.clone()
        for _ in range(3):
            norm(hidden, residual)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, summed = norm(hidden, residual)
        for scale in (1.0, 0.003):
            hidden.copy_(original_hidden * scale)
            residual.copy_(original_residual)
            expected = hidden + residual
            graph.replay()
            self.assertIs(output, hidden)
            self.assertIs(summed, residual)
            torch.testing.assert_close(summed, expected, rtol=0, atol=0)
            x = expected.float()
            ref = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            ref = ref.bfloat16()
            error = (output.float() - ref.float()).norm() / ref.float().norm()
            self.assertLess(error.item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
