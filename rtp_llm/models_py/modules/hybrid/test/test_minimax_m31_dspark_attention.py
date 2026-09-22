import unittest

import torch

from rtp_llm.models_py.model_desc.minimax_m31_dspark import (
    _MiniMaxM31DSparkQueryContext,
)
from rtp_llm.models_py.modules.hybrid.msa_attention import (
    expand_dspark_visible_seq_lens,
)


class MiniMaxM31DSparkAttentionTest(unittest.TestCase):
    def test_query_context_supports_cuda_graph(self):
        context = _MiniMaxM31DSparkQueryContext()
        self.assertTrue(context.support_cuda_graph())
        self.assertIsNone(context.prepare_cuda_graph(None))

    def test_query_rows_see_request_local_block_tail(self):
        write_lengths = torch.tensor([11, 12, 13, 21, 22, 23], dtype=torch.int32)
        visible = expand_dspark_visible_seq_lens(write_lengths, 2, 6)
        torch.testing.assert_close(
            visible, torch.tensor([13, 13, 13, 23, 23, 23], dtype=torch.int32)
        )
        self.assertTrue(visible.is_contiguous())
        torch.testing.assert_close(
            write_lengths,
            torch.tensor([11, 12, 13, 21, 22, 23], dtype=torch.int32),
        )

    def test_rejects_incomplete_request_block(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            expand_dspark_visible_seq_lens(torch.arange(5), 2, 5)


if __name__ == "__main__":
    unittest.main()
