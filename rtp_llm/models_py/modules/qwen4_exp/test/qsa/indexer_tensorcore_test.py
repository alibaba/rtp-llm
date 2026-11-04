import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score import (
    qsa_paged_indexer_score,
)


class IndexerTensorCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 is required")

    def test_weighted_relu_scores_and_topk_match_float64_reference(self):
        torch.manual_seed(1371)
        b, s, h, d, page, n = 2, 37, 4, 128, 32, 97
        q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16) * 0.3
        pool = torch.randn(9 * page, d, device="cuda", dtype=torch.bfloat16) * 0.3
        weights = torch.randn(b * s, h, device="cuda") * 0.1
        table = torch.tensor(
            [[5, 3, 7, 1], [4, 2, 8, 6]], device="cuda", dtype=torch.int32
        )
        lens = torch.randint(0, n + 1, (b, s), device="cuda", dtype=torch.int32)
        args = (q, weights, pool, table, lens)
        columns = torch.arange(n, device="cuda")
        keys = pool[(table[:, columns // page].long() * page + columns % page)].double()
        reference = torch.einsum("bshd,bnd->bshn", q.double(), keys).relu()
        reference = (reference * weights.view(b, s, h, 1).double()).sum(2)
        reference.masked_fill_(columns[None, None] >= lens[..., None], -float("inf"))
        with patch.dict(os.environ, RTP_QWEN4_PAGED_INDEXER_IMPL="mma"):
            output = qsa_paged_indexer_score(*args, block_size=page, max_ctx_len=n)
        torch.testing.assert_close(
            output.double(), reference.view(b * s, n), rtol=2e-5, atol=2e-6
        )
        with patch.dict(os.environ, RTP_QWEN4_PAGED_INDEXER_IMPL="vector"):
            legacy = qsa_paged_indexer_score(*args, block_size=page, max_ctx_len=n)
        torch.testing.assert_close(output, legacy, rtol=2e-5, atol=2e-6)
        # Only compare rows with a fully populated top-k boundary.
        live = lens.view(-1) >= 8
        torch.testing.assert_close(
            output[live].topk(8).indices.sort().values,
            legacy[live].topk(8).indices.sort().values,
            rtol=0,
            atol=0,
        )

    def test_graph_refreshes_pages_contexts_and_negative_weights(self):
        q = torch.ones(2, 17, 4, 128, device="cuda", dtype=torch.bfloat16)
        pool = torch.ones(4 * 32, 128, device="cuda", dtype=torch.bfloat16)
        pool[64:96] *= 2
        weights = torch.ones(34, 4, device="cuda")
        table = torch.ones(2, 1, device="cuda", dtype=torch.int32)
        lengths = torch.full((2, 17), 32, device="cuda", dtype=torch.int32)
        options = dict(block_size=32, max_ctx_len=37, validate_block_table=False)
        with patch.dict(os.environ, RTP_QWEN4_PAGED_INDEXER_IMPL="mma"):
            for _ in range(3):
                qsa_paged_indexer_score(q, weights, pool, table, lengths, **options)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = qsa_paged_indexer_score(
                    q, weights, pool, table, lengths, **options
                )
            weights[:, 0] = -0.5
            table[0, 0] = 2
            table[1, 0] = 99
            lengths.fill_(7)
            graph.replay()
        expected = torch.full_like(actual, -float("inf"))
        expected[:17, :7] = 128 * 2 * 2.5
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        graph.reset()


if __name__ == "__main__":
    unittest.main()
