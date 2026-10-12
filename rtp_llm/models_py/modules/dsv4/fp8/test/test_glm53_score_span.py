"""Validate request-local output width against the old packed-K bound."""

import unittest

import torch
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import fp8_mqa_indexer_score


class ScoreSpanTest(unittest.TestCase):
    def test_local_span_preserves_valid_logits(self):
        self.assertTrue(torch.cuda.is_available(), "GPU test must not skip")
        torch.manual_seed(531021)
        for batch, query, keys in ((3, 17, 131), (39, 419, 2152)):
            rows, heads = batch * query, 32
            q = (torch.randn(rows, heads, 128, device="cuda") * 0.1).to(
                torch.float8_e4m3fn
            )
            k = (torch.randn(batch * keys, 128, device="cuda") * 0.1).to(
                torch.float8_e4m3fn
            )
            scale = torch.rand(batch * keys, device="cuda") * 0.1 + 0.01
            weights = torch.rand(rows, heads, device="cuda")
            req = torch.arange(batch, device="cuda").repeat_interleave(query)
            starts = (req * keys).int()
            lengths = (
                torch.arange(query, device="cuda").repeat(batch) % min(keys, 128)
                + keys
                - min(keys, 128)
                + 1
            )
            ends = starts + lengths.int()
            old = fp8_mqa_indexer_score(
                q, weights, k, scale, starts, ends, max_seqlen_k=batch * keys
            )
            new = fp8_mqa_indexer_score(
                q, weights, k, scale, starts, ends, max_seqlen_k=keys
            )
            columns = torch.arange(keys, device="cuda").unsqueeze(0)
            mask = columns < lengths.unsqueeze(1)
            torch.testing.assert_close(
                new[:, :keys][mask], old[:, :keys][mask], rtol=0, atol=0
            )
            print(
                "score widths:",
                batch,
                old.shape,
                new.shape,
                "valid logits bitwise identical",
            )


if __name__ == "__main__":
    unittest.main()
