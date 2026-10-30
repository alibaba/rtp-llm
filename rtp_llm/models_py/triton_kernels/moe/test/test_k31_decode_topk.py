import unittest

import torch

from rtp_llm.models_py.modules.base.cuda.select_topk import GroupTopK
from rtp_llm.models_py.modules.kimi_k3.router import KimiK3Router
from rtp_llm.models_py.triton_kernels.moe.k31_decode_topk import k31_decode_topk


class K31DecodeTopKTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9183)
        self.assertEqual(torch.cuda.get_device_capability(), (10, 3))

    def assert_routes_close(self, actual, expected):
        torch.testing.assert_close(actual[0], expected[0], atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)

    def test_score_boundary(self):
        bias = torch.zeros(896, device="cuda")
        near = torch.linspace(-1e-7, 1e-7, 896, device="cuda").repeat(16, 1)
        mixed = near.clone()
        mixed[:, ::4] = float("nan")
        mixed[:, 1::4] = float("inf")
        invalid = torch.full_like(near, float("nan"))
        partial = invalid.clone()
        partial[:, 100:108] = 0.0
        with_expert_zero = partial.clone()
        with_expert_zero[:, 0] = 0.0
        random = torch.randn((64, 896), device="cuda")
        cases = (near, mixed, invalid, partial, with_expert_zero, random[:16])
        for logits in cases:
            self.check_kernel(logits, bias, True, 1.0)

        for rows in (1, 8, 15, 16, 32, 64):
            for renormalize, scale in ((True, 2.5), (False, 2.5)):
                self.check_kernel(random[:rows], bias, renormalize, scale)

    def check_kernel(self, logits, bias, renormalize, scale):
        with self.subTest(rows=logits.shape[0], renormalize=renormalize, scale=scale):
            expected_weights = torch.empty((logits.shape[0], 16), device="cuda")
            expected_ids = torch.empty_like(expected_weights, dtype=torch.int32)
            actual_weights = torch.empty_like(expected_weights)
            actual_ids = torch.empty_like(expected_ids)
            GroupTopK().forward_fused_sigmoid(
                expected_weights, expected_ids, logits, bias, 16, renormalize, scale
            )
            k31_decode_topk(
                logits, bias, actual_weights, actual_ids, renormalize, scale
            )
            self.assert_routes_close(
                (actual_weights, actual_ids), (expected_weights, expected_ids)
            )

    def test_router_eager_and_graph(self):
        weight = torch.randn((7168, 896), device="cuda", dtype=torch.bfloat16) * 0.02
        bias = torch.randn(896, device="cuda") * 0.05
        kwargs = dict(top_k=16, groups=1, top_groups=1, renormalize=True, scale=1.0)
        baseline = KimiK3Router(weight, bias, **kwargs)
        optimized = KimiK3Router(weight, bias, **kwargs, enable_decode_topk=True)
        self.assertTrue(optimized._decode_topk)
        hidden = torch.randn((16, 7168), device="cuda", dtype=torch.bfloat16)
        self.assert_routes_close(optimized(hidden), baseline(hidden))
        self.assert_routes_close(optimized(hidden[:8]), baseline(hidden[:8]))
        for repeats in (4, 8):
            larger = hidden.repeat(repeats, 1)
            self.assert_routes_close(optimized(larger), baseline(larger))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = optimized(hidden)
        for _ in range(3):
            hidden.normal_()
            expected = baseline(hidden)
            graph.replay()
            self.assert_routes_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
