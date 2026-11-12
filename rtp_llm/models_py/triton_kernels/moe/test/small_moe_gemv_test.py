import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.small_moe_gemv import (
    is_supported,
    small_moe_gemv,
)
from rtp_llm.models_py.triton_kernels.moe.test.small_moe_fixture import (
    SmallMoeGemvTestFixture,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SmallMoeGemvTest(SmallMoeGemvTestFixture, unittest.TestCase):

    def test_eager_and_live_graph_routes(self):
        for m in (1, 2, 3, 4):
            p = self.payload(m)
            for _ in range(3):
                small_moe_gemv(
                    p.expert_x,
                    self.module.w1,
                    self.module.w2,
                    p.expert_topk_ids,
                    p.expert_topk_weights,
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = small_moe_gemv(
                    p.expert_x,
                    self.module.w1,
                    self.module.w2,
                    p.expert_topk_ids,
                    p.expert_topk_weights,
                )
            for spread, scale in ((512, 0.125), (1, 0.125), (3, 1.0), (512, 0.0)):
                with self.subTest(m=m, spread=spread, scale=scale):
                    p.expert_x.copy_(torch.randn_like(p.expert_x) * scale)
                    p.expert_topk_ids.copy_(
                        torch.randint(spread, (m, 10), device="cuda", dtype=torch.int32)
                    )
                    p.expert_topk_weights.copy_(
                        torch.rand_like(p.expert_topk_weights) * 0.1
                    )
                    expected = self.ref(p)
                    graph.replay()
                    torch.testing.assert_close(
                        actual, expected, atol=0.0078125, rtol=0.008
                    )

    def test_unsupported_falls_back_before_launch(self):
        p = self.payload(1)
        args = [
            p.expert_x,
            self.module.w1,
            self.module.w2,
            p.expert_topk_ids,
            p.expert_topk_weights,
        ]
        for index, tensor in (
            (0, p.expert_x.float()),
            (0, p.expert_x[:, ::2]),
            (3, p.expert_topk_ids.long()),
            (4, p.expert_topk_weights.half()),
        ):
            with self.subTest(index=index, dtype=tensor.dtype):
                values = list(args)
                values[index] = tensor
                self.assertFalse(is_supported(*values))
                self.assertIsNone(small_moe_gemv(*values))
        p = self.payload(5)
        self.assertIsNone(
            small_moe_gemv(
                p.expert_x,
                self.module.w1,
                self.module.w2,
                p.expert_topk_ids,
                p.expert_topk_weights,
            )
        )


if __name__ == "__main__":
    unittest.main()
