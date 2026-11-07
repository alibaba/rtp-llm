import unittest

import torch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    exact_head_rms_norm,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import apply_partial_rope
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_norm_rope import (
    try_indexer_norm_rope,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class IndexerNormRoPETest(unittest.TestCase):
    def tensors(self, batch, query, heads, dtype):
        qkv = torch.randn(
            batch, query, (heads + 1) * 128, device="cuda", dtype=torch.bfloat16
        )
        q = qkv[..., : heads * 128].view(batch, query, heads, 128)
        gamma = torch.randn(128, device="cuda", dtype=dtype) * 0.07
        angle = torch.randn(batch, query, 1, 32, device="cuda")
        cos = torch.cat((angle.cos(), angle.cos()), -1).to(torch.bfloat16)
        sin = torch.cat((angle.sin(), angle.sin()), -1).to(torch.bfloat16)
        return q, cos, sin, gamma

    def test_strided_raw_projection_and_fp32_raw_gamma(self):
        for batch, query, heads in (
            (1, 1, 4),
            (1, 4, 4),
            (4, 4, 4),
            (8, 4, 4),
            (1, 4096, 4),
            (1, 12288, 4),
        ):
            for dtype in (torch.bfloat16, torch.float32):
                tensors = self.tensors(batch, query, heads, dtype)
                q, cos, sin, gamma = tensors
                before = q.clone()
                expected = apply_partial_rope(
                    exact_head_rms_norm(q, gamma, 1e-6), cos, sin
                )
                actual = try_indexer_norm_rope(*tensors, 1e-6)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(q, before, rtol=0, atol=0)

    def test_graph_reads_live_raw_queries_gains_and_rotations(self):
        q, cos, sin, gamma = self.tensors(4, 4, 4, torch.float32)
        for _ in range(3):
            try_indexer_norm_rope(q, cos, sin, gamma, 1e-6)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_indexer_norm_rope(q, cos, sin, gamma, 1e-6)
        for seed in (7911, 7912):
            torch.manual_seed(seed)
            q.normal_()
            gamma.normal_(0, 0.07)
            cos.uniform_(-1, 1)
            sin.uniform_(-1, 1)
            expected = apply_partial_rope(exact_head_rms_norm(q, gamma, 1e-6), cos, sin)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
