"""Exact int64 hashes, EOS boundaries and fixed-address replay."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpNGramEmbedding
from rtp_llm.models_py.modules.qwen4_exp.ple.ple_hash_triton import try_hashed_ngram_ids
from rtp_llm.models_py.modules.qwen4_exp.test.ple_test import (
    _REAL_MULTIPLIERS,
    _REAL_OFFSETS,
    _REAL_VOCAB_SIZES,
    _ref_hashed_ids,
)

MODULE = "rtp_llm.models_py.modules.qwen4_exp.ple.ple_hash_triton"


class PLEHashTritonTest(unittest.TestCase):
    def module(self):
        obj = SimpleNamespace(
            ngram_size=3,
            heads_per_ngram=8,
            eos_token_id=248044,
            vocab_sizes=torch.tensor(_REAL_VOCAB_SIZES, device="cuda"),
            offsets=torch.tensor(_REAL_OFFSETS, device="cuda"),
            multipliers=torch.tensor(_REAL_MULTIPLIERS, device="cuda"),
        )
        obj._shift_right_ignore_eos = MethodType(
            Qwen4ExpNGramEmbedding._shift_right_ignore_eos, obj
        )
        return obj

    def call(self, obj, history, seq):
        return Qwen4ExpNGramEmbedding.hashed_ids(obj, history, seq)

    def reference(self, obj, history, seq):
        return _ref_hashed_ids(
            history.cpu(),
            seq,
            obj.vocab_sizes.cpu(),
            obj.offsets.cpu(),
            obj.multipliers.cpu(),
            3,
            obj.eos_token_id,
        ).cuda()

    def test_actual_metadata_prime_shapes_and_eos_resets(self):
        obj = self.module()
        torch.manual_seed(3016)
        for batch in (1, 3):
            for seq in (1, 3, 4, 8, 16, 31, 4093, 4096, 12288, 32768):
                for context in (0, 1, 2):
                    history = torch.randint(
                        0, 248320, (batch, seq + context), device="cuda"
                    )
                    if seq + context > 2:
                        history[:, 1::7] = obj.eos_token_id
                    actual = self.call(obj, history, seq)
                    self.assertTrue(
                        torch.equal(actual, self.reference(obj, history, seq)),
                        (batch, seq, context),
                    )

    def test_negative_overflow_and_noncontiguous_batch_stride(self):
        obj = self.module()
        history = torch.tensor(
            [-(2**63), -3, -1, 0, 248044, 7, 2**63 - 1, 248044, 248044], device="cuda"
        ).repeat(3, 1)
        self.assertTrue(
            torch.equal(
                self.call(obj, history, history.shape[1]),
                self.reference(obj, history, history.shape[1]),
            )
        )
        source = torch.randint(0, 248320, (6, 33), device="cuda")[::2]
        self.assertTrue(
            torch.equal(self.call(obj, source, 31), self.reference(obj, source, 31))
        )

    def test_graph_reads_new_tokens_and_metadata(self):
        obj = self.module()
        history = torch.randint(0, 248320, (3, 18), device="cuda")
        self.call(obj, history, 16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.call(obj, history, 16)
        for value in (0, 248044, 248319):
            history.fill_(value)
            obj.multipliers.add_(1)
            obj.offsets.add_(17)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(actual, self.reference(obj, history, 16)))

    def test_unsupported_keeps_original(self):
        obj = self.module()
        history = torch.arange(33, device="cuda").view(1, 33)
        with patch(MODULE + ".try_hashed_ngram_ids", return_value=None):
            expected = self.call(obj, history, 0)
        self.assertTrue(torch.equal(self.call(obj, history, 0), expected))
        cpu = history.cpu()
        self.assertIsNone(
            try_hashed_ngram_ids(
                cpu,
                31,
                obj.vocab_sizes,
                obj.offsets,
                obj.multipliers,
                ngram_size=3,
                heads_per_ngram=8,
                eos_token_id=248044,
            )
        )


if __name__ == "__main__":
    unittest.main()
