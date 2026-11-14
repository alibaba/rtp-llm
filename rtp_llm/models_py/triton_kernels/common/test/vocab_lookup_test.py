import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Fp8VocabEmbeddingTest(unittest.TestCase):
    def test_tp_reorder_and_hyper_repeat_match_live_graph(self):
        from rtp_llm.models_py.triton_kernels.common.fp8_vocab_lookup import (
            gather_hyper_rows,
        )

        for rows in (1, 4, 16, 4096):
            source = torch.randn(2, rows, 1280, device="cuda", dtype=torch.bfloat16)
            expected = (
                source.transpose(0, 1).contiguous().reshape(rows, 2560).repeat(1, 4)
            )
            actual = gather_hyper_rows(source, rows, 1280, 2, 4)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        source = torch.randn(2, 4, 1280, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            gather_hyper_rows(source, 4, 1280, 2, 4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gather_hyper_rows(source, 4, 1280, 2, 4)
        for gain in (0.01, 1.0, 3.0):
            source.normal_(0, gain)
            graph.replay()
            expected = source.transpose(0, 1).contiguous().reshape(4, 2560).repeat(1, 4)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
