"""Exercise real pooled-index expansion beyond 2**31 output elements."""

import unittest

import torch
from rtp_llm.models_py.modules.indexer_grouping import (
    fused_expand_indexer_groups_with_tail,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class LargeOffsetsTest(unittest.TestCase):
    def test_million_query_rows(self):
        rows = 1048576
        groups = torch.arange(512, dtype=torch.int32, device="cuda").expand(rows, -1)
        lengths = torch.full((rows,), 8193, dtype=torch.int32, device="cuda")
        output = fused_expand_indexer_groups_with_tail(groups, lengths, 4)
        torch.cuda.synchronize()
        self.assertEqual(output.shape, (rows, 2051))
        boundary = (2**31) // 2051
        selected = torch.tensor(
            [0, 1, boundary - 1, boundary, boundary + 1, rows - 1], device="cuda"
        )
        expected = torch.cat(
            (
                torch.arange(2048, dtype=torch.int32, device="cuda"),
                torch.tensor([8192, -1, -1], dtype=torch.int32, device="cuda"),
            )
        )
        torch.testing.assert_close(
            output[selected], expected.expand(selected.numel(), -1), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
