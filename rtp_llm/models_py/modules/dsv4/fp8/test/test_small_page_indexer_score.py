"""Check 16-entry pages against an independent FP32 oracle on identical FP8 bytes."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import fp8_paged_indexer_score


class SmallPageIndexerScoreTest(unittest.TestCase):
    def test_paged_scores_and_graph_replay(self):
        torch.manual_seed(37)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for batch, next_n, tokens, heads in (
                (1, 1, 31, 32),
                (2, 3, 63, 1),
                (5, 7, 259, 17),
                (33, 4, 513, 63),
                (3, 4, 257, 16),
                (17, 4, 1024, 64),
                (65, 4, 1024, 64),
                (4, 1, 4097, 128),
                (3, 7, 1021, 128),
            ):
                with self.subTest(
                    batch=batch, next_n=next_n, tokens=tokens, heads=heads
                ):
                    self.check_shape(batch, next_n, tokens, heads)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    def check_shape(self, batch, next_n, tokens, heads):
        page, dim = 16, 128
        pages = (tokens + page - 1) // page
        q = (torch.randn(batch, next_n, heads, dim, device="cuda") * 0.2).to(
            torch.float8_e4m3fn
        )
        k = (torch.randn(batch, tokens, dim, device="cuda") * 0.3).to(
            torch.float8_e4m3fn
        )
        weights = torch.randn(batch * next_n, heads, device="cuda") * 0.05
        scales = torch.rand(batch, tokens, device="cuda") * 0.5 + 0.5
        lengths = torch.randint(
            1, tokens + 1, (batch, next_n), device="cuda", dtype=torch.int32
        )
        if batch * next_n > 1:
            lengths[0, 0] = 0
        table = (
            (torch.randperm(batch * pages, device="cuda") + 1)
            .reshape(batch, pages)
            .int()
        )
        cache = torch.zeros(
            (1 + batch * pages, page * 132), device="cuda", dtype=torch.uint8
        )
        for b in range(batch):
            kb = torch.zeros((pages * page, dim), device="cuda", dtype=torch.uint8)
            kb[:tokens] = k[b].view(torch.uint8)
            sb = torch.zeros(pages * page, device="cuda")
            sb[:tokens] = scales[b]
            cache[table[b].long(), : page * dim] = kb.reshape(pages, page * dim)
            cache[table[b].long(), page * dim :] = sb.view(torch.uint8).reshape(
                pages, page * 4
            )
        pool = cache.view(-1, 132)

        def score():
            return fp8_paged_indexer_score(
                q, weights, pool, table, lengths, page, tokens
            )

        expected = (
            torch.einsum("bnhd,btd->bnht", q.float(), k.float()).relu()
            * weights.reshape(batch, next_n, heads, 1)
        ).sum(2) * scales[:, None, :]
        valid = torch.arange(tokens, device="cuda")[None, :] < lengths.reshape(-1, 1)
        actual = score()
        torch.testing.assert_close(
            actual[valid],
            expected.reshape(batch * next_n, tokens)[valid],
            rtol=5e-4,
            atol=5e-4,
        )
        self.assertTrue(torch.isneginf(actual[~valid]).all().item())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = score()
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)
        lengths.zero_()
        graph.replay()
        self.assertTrue(torch.isneginf(captured).all().item())


if __name__ == "__main__":
    unittest.main()
