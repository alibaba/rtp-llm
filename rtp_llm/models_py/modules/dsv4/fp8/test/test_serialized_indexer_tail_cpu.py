"""Serialized indexer prefix/tail ordering contracts on CPU doubles.

The indexer tail runs inline in ``IndexerFP8.forward``; the attention
consumer chain (compressor → materialize → workspace attention) consumes the
returned tensor directly. These tests import ``IndexerFP8`` and
``AttentionFP8`` directly and exercise them with sentinel doubles so ordering
and error identity are checked without GPU kernels.
"""

from __future__ import annotations

import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import attention as attn_mod
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8


class SerializedIndexerTailTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.x = torch.zeros(2, 4)
        self.qr = torch.zeros(2, 3)
        self.qkv = types.SimpleNamespace(qr=self.qr)
        self.meta = object()
        self.workspace = object()
        self.common = types.SimpleNamespace(
            sp_int=0,
            workspace=self.workspace,
            csa_meta=types.SimpleNamespace(
                indexer_meta=self.meta,
                compressor_meta=object(),
                workspace_meta=object(),
            ),
        )
        self.topk = torch.tensor([[0, 2], [1, -1]], dtype=torch.int32)
        self.result = object()
        rng_patch = patch.object(
            attn_mod,
            "record_function_range",
            side_effect=lambda *_a, **_k: nullcontext(),
        )
        rng_patch.start()
        self.addCleanup(rng_patch.stop)

    def _indexer(self, state, prefix_exc=None, tail_exc=None):
        indexer = IndexerFP8.__new__(IndexerFP8)
        torch.nn.Module.__init__(indexer)
        events = self.events
        meta, x, qr, workspace = self.meta, self.x, self.qr, self.workspace

        def prefix(
            x_, qr_, meta_, *, workspace, cp_gather_stream=None, post_gather_stream=None
        ):
            assert x_ is x and qr_ is qr and meta_ is meta
            assert workspace is self.workspace
            assert cp_gather_stream is None and post_gather_stream is None
            events.append("prefix")
            if prefix_exc is not None:
                raise prefix_exc
            return state

        def tail(state_, meta_):
            assert state_ is state and meta_ is meta
            events.append("tail")
            if tail_exc is not None:
                raise tail_exc
            return self.topk

        indexer._prefill_score_prefix = prefix
        indexer._prefill_score_tail = tail
        return indexer

    def _attention(self, indexer):
        attn = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(attn)
        events = self.events
        common = self.common

        def compressor(x_, sp, *, meta, workspace):
            assert x_ is self.x and sp == 0
            assert meta is common.csa_meta.compressor_meta
            assert workspace is common.workspace
            events.append("compressor")

        def materialize(qkv_, common_):
            assert qkv_ is self.qkv and common_ is common
            events.append("materialize")
            return qkv_

        def consumer(qkv_, common_, workspace_meta, topk):
            assert workspace_meta is common.csa_meta.workspace_meta
            assert topk is self.topk
            assert torch.equal(topk, torch.tensor([[0, 2], [1, -1]], dtype=torch.int32))
            events.append("consumer")
            return self.result

        def warmup(qkv_, common_):
            raise AssertionError("warmup path must not run with bound workspace_meta")

        attn.indexer = indexer
        attn.compressor = compressor
        attn._materialize_prefill_q = materialize
        attn._attn_via_workspace = consumer
        attn._attn_fp8_swa_via_kv_full = warmup
        return attn

    def test_prefix_tail_consumer_order(self):
        state = object()
        indexer = self._indexer(state)
        attn = self._attention(indexer)
        out = AttentionFP8._forward_prefill_csa(attn, self.x, self.qkv, self.common)
        self.assertIs(out, self.result)
        self.assertEqual(
            self.events, ["prefix", "tail", "compressor", "materialize", "consumer"]
        )

    def test_prefix_failure_is_primary(self):
        sentinel = RuntimeError("prefix boom")
        indexer = self._indexer(object(), prefix_exc=sentinel)
        attn = self._attention(indexer)
        with self.assertRaises(RuntimeError) as ctx:
            AttentionFP8._forward_prefill_csa(attn, self.x, self.qkv, self.common)
        self.assertIs(ctx.exception, sentinel)
        self.assertEqual(self.events, ["prefix"])

    def test_tail_failure_is_primary(self):
        sentinel = RuntimeError("tail boom")
        indexer = self._indexer(object(), tail_exc=sentinel)
        attn = self._attention(indexer)
        with self.assertRaises(RuntimeError) as ctx:
            AttentionFP8._forward_prefill_csa(attn, self.x, self.qkv, self.common)
        self.assertIs(ctx.exception, sentinel)
        self.assertEqual(self.events, ["prefix", "tail"])

    def test_empty_prefix_returns_without_tail(self):
        empty = torch.empty((0, 2), dtype=torch.int32)
        indexer = self._indexer(empty)
        out = indexer.forward(self.x, self.qr, self.meta, workspace=self.workspace)
        self.assertIs(out, empty)
        self.assertEqual(self.events, ["prefix"])


if __name__ == "__main__":
    unittest.main()
