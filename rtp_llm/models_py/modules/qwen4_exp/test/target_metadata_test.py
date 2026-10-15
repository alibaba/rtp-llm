import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.target_metadata_triton import (
    try_target_metadata,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TargetMetadataTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 fast path")

    def reference(self, prefixes, q):
        visible = (
            prefixes[:, None]
            + torch.arange(q, dtype=torch.int32, device=prefixes.device)[None, :]
            + 1
        )
        query = (visible - 1).reshape(-1).to(torch.int64)
        return visible, visible // 4, query, query - query.remainder(4)

    def test_exact_geometry_and_integer_boundaries(self):
        for batch in (1, 3, 4, 5, 6, 7, 8, 17, 31):
            for q in (1, 2, 3, 4):
                prefixes = torch.randint(
                    -129, 32769, (batch,), device="cuda", dtype=torch.int32
                )
                prefixes[0] = torch.iinfo(torch.int32).max - 2
                actual = try_target_metadata(prefixes, q, 4)
                self.assertIsNotNone(actual)
                for a, b in zip(actual, self.reference(prefixes, q)):
                    self.assertTrue(torch.equal(a, b))

    def test_graph_refreshes_every_prefix_and_request_order(self):
        p = torch.tensor([127, 4093, 32749, 7, 1], device="cuda", dtype=torch.int32)
        for _ in range(3):
            try_target_metadata(p, 4, 4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_target_metadata(p, 4, 4)
        for step in range(4):
            p.copy_(p.flip(0) + step)
            graph.replay()
            for a, b in zip(actual, self.reference(p, 4)):
                self.assertTrue(torch.equal(a, b))

    def test_unsupported_layouts_return_none(self):
        p = torch.zeros(4, device="cuda", dtype=torch.int32)
        for value, q, ratio in (
            (p.cpu(), 4, 4),
            (p.long(), 4, 4),
            (p[::2], 4, 4),
            (p, 5, 4),
            (p, 4, 8),
            (p[:0], 4, 4),
            (torch.zeros(33, device="cuda", dtype=torch.int32), 4, 4),
        ):
            self.assertIsNone(try_target_metadata(value, q, ratio))


if __name__ == "__main__":
    unittest.main()
