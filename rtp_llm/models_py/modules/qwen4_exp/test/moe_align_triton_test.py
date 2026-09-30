import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.moe_align_triton import moe_align_decode
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    moe_align_block_size_torch,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MoeAlignTritonTest(unittest.TestCase):
    def compare(self, ids, block, experts):
        actual = moe_align_decode(ids, block, experts)
        expected = moe_align_block_size_torch(ids, block, experts)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=0, rtol=0)

    def test_stable_layout_random_skewed_and_duplicate_routes(self):
        torch.manual_seed(42)
        for batch in (1, 3, 8, 17, 32):
            for block in (8, 16, 32, 64):
                for experts in (16, 512):
                    with self.subTest(batch=batch, block=block, experts=experts):
                        ids = torch.randint(
                            experts, (batch, 10), device="cuda", dtype=torch.int32
                        )
                        self.compare(ids, block, experts)
                        self.compare(torch.zeros_like(ids), block, experts)
                        self.compare(ids.remainder(3), block, experts)

    def test_cuda_graph_refreshes_routing_and_padding_in_both_directions(self):
        for batch in (1, 8, 32):
            ids = torch.randint(512, (batch, 10), device="cuda", dtype=torch.int32)
            for _ in range(3):
                moe_align_decode(ids, 16, 512)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = moe_align_decode(ids, 16, 512)
            for spread in (1, 512, 3, 512):
                ids.copy_(
                    torch.randint(spread, ids.shape, device="cuda", dtype=torch.int32)
                )
                graph.replay()
                expected = moe_align_block_size_torch(ids, 16, 512)
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, atol=0, rtol=0)

    def test_rejects_unbounded_or_non_power_of_two_geometry(self):
        for batch, experts in ((33, 512), (8, 511), (8, 1024)):
            ids = torch.zeros(batch, 10, device="cuda", dtype=torch.int32)
            with self.assertRaises(ValueError):
                moe_align_decode(ids, 16, experts)


if __name__ == "__main__":
    unittest.main()
