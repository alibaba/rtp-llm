"""Byte parity for TP2/4/8 head groups and fixed-address graph refresh."""

import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.norm import Qwen4ExpFusedQKRMSNorm
from rtp_llm.models_py.modules.qwen4_exp.qsa.norm_triton import is_small_qk_supported


class GroupedQKNormTest(unittest.TestCase):
    def reference(self, x, q, k, heads, eps=1e-6):
        result = x.clone().reshape(x.shape[0], heads + 2, 256)
        for start, stop, gamma in ((0, heads, q), (heads, heads + 1, k)):
            value = result[:, start:stop].float()
            result[:, start:stop].copy_(
                (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps))
                * (1.0 + gamma.float())
            )
        return result.reshape_as(x)

    def test_real_head_groups_exact_bytes_and_v(self):
        torch.manual_seed(251026)
        for heads in (3, 6, 12):
            for dtype in (torch.bfloat16, torch.float32):
                q = torch.randn(256, device="cuda", dtype=dtype) * 0.1
                k = torch.randn_like(q) * 0.1
                module = Qwen4ExpFusedQKRMSNorm(q, k, heads, 1, 256)
                for rows in (1, 3, 4, 8, 16, 20, 24, 28, 31, 32):
                    with self.subTest(heads=heads, gamma=dtype, rows=rows):
                        x = torch.randn(
                            rows, (heads + 2) * 256, device="cuda", dtype=torch.bfloat16
                        )
                        self.assertTrue(is_small_qk_supported(x, q, k, heads, 1, 256))
                        expected = self.reference(x, q, k, heads)
                        actual = module(x.clone())
                        self.assertTrue(
                            torch.equal(
                                expected.view(torch.int16), actual.view(torch.int16)
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                actual[:, (heads + 1) * 256 :],
                                x[:, (heads + 1) * 256 :],
                            )
                        )

    def test_graph_refreshes_inputs_and_raw_gains(self):
        for heads in (6, 12):
            for rows in (1, 4, 16, 28):
                x = torch.randn(
                    rows, (heads + 2) * 256, device="cuda", dtype=torch.bfloat16
                )
                q = torch.full((256,), 0.003, device="cuda", dtype=torch.bfloat16)
                k = q.clone()
                module = Qwen4ExpFusedQKRMSNorm(q, k, heads, 1, 256)
                for _ in range(3):
                    module(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    module(x)
                for scale in (0.0, 0.001, 1.0, 100.0):
                    source = torch.randn_like(x) * scale
                    q.normal_(0, 0.1)
                    k.normal_(0, 0.1)
                    expected = self.reference(source, q, k, heads)
                    x.copy_(source)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(
                        torch.equal(x.view(torch.int16), expected.view(torch.int16))
                    )

    def test_unsupported_layout_keeps_fallback(self):
        for heads in (6, 12):
            q = torch.randn(256, device="cuda", dtype=torch.bfloat16)
            k = q.clone()
            module = Qwen4ExpFusedQKRMSNorm(q, k, heads, 1, 256)
            for rows in (33, 4093):
                x = torch.randn(
                    rows, (heads + 2) * 256, device="cuda", dtype=torch.bfloat16
                )
                self.assertFalse(is_small_qk_supported(x, q, k, heads, 1, 256))
                self.assertTrue(
                    torch.equal(module(x.clone()), self.reference(x, q, k, heads))
                )


if __name__ == "__main__":
    unittest.main()
