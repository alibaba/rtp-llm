"""Serialized indexer prefix/tail ordering contracts on CPU doubles.

The indexer tail runs inline in ``IndexerFP8.forward``; the attention
consumer chain (compressor → materialize → workspace attention) consumes the
returned tensor directly. These tests compile the real functions out of the
source files — no module import — and exercise them with sentinel doubles so
ordering and error identity are checked without GPU kernels.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types
import unittest
from contextlib import nullcontext
from runpy import run_path
from unittest.mock import patch

import torch

HERE = pathlib.Path(__file__).resolve()
FP8 = HERE.parents[1]
INDEXER_SRC = FP8 / "indexer.py"
ATTENTION_SRC = FP8 / "attention.py"
PACKAGE = "rtp_llm.models_py.modules.dsv4.fp8"


def _extract(src, class_name, method_names):
    """Parse ``src`` and compile ``class_name`` with only ``method_names``."""
    tree = ast.parse(src.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names
    ]
    assert {n.name for n in cls.body} == set(method_names), (
        class_name,
        {n.name for n in cls.body},
    )
    cls.bases = []
    cls.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    return module


def _indexer_cls():
    env = {"torch": torch}
    exec(
        compile(
            _extract(INDEXER_SRC, "IndexerFP8", {"forward"}), str(INDEXER_SRC), "exec"
        ),
        env,
    )
    cls = env["IndexerFP8"]
    # ``self.indexer(...)`` in _forward_prefill_csa resolves __call__ on the type.
    cls.__call__ = cls.forward
    return cls


def _attention_cls():
    env = {
        "torch": torch,
        "record_function_range": lambda *_a, **_k: nullcontext(),
    }
    exec(
        compile(
            _extract(
                ATTENTION_SRC,
                "AttentionFP8",
                {"_forward_prefill_csa", "_forward_prefill_compressed"},
            ),
            str(ATTENTION_SRC),
            "exec",
        ),
        env,
    )
    return env["AttentionFP8"]


def _indexer_packages(indexer_cls):
    """Fresh rtp_llm package chain whose fp8.indexer module exposes ``indexer_cls``."""
    indexer_mod = types.ModuleType(PACKAGE + ".indexer")
    indexer_mod.IndexerFP8 = indexer_cls
    mods = {}
    parts = PACKAGE.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        mod = types.ModuleType(name)
        mod.__path__ = []
        mods[name] = mod
        if i > 1:
            setattr(mods[".".join(parts[: i - 1])], parts[i - 1], mod)
    mods[PACKAGE].indexer = indexer_mod
    mods[PACKAGE + ".indexer"] = indexer_mod
    return mods


class SerializedIndexerTailTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.indexer_cls = _indexer_cls()
        self.attention_cls = _attention_cls()
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
        self._package_patch = patch.dict(
            sys.modules, _indexer_packages(self.indexer_cls)
        )
        self._package_patch.start()
        self.addCleanup(self._package_patch.stop)

    def _indexer(self, state, prefix_exc=None, tail_exc=None):
        indexer = self.indexer_cls.__new__(self.indexer_cls)
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
        attn = self.attention_cls.__new__(self.attention_cls)
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
        out = self.attention_cls._forward_prefill_csa(
            attn, self.x, self.qkv, self.common
        )
        self.assertIs(out, self.result)
        self.assertEqual(
            self.events, ["prefix", "tail", "compressor", "materialize", "consumer"]
        )

    def test_prefix_failure_is_primary(self):
        sentinel = RuntimeError("prefix boom")
        indexer = self._indexer(object(), prefix_exc=sentinel)
        attn = self._attention(indexer)
        with self.assertRaises(RuntimeError) as ctx:
            self.attention_cls._forward_prefill_csa(attn, self.x, self.qkv, self.common)
        self.assertIs(ctx.exception, sentinel)
        self.assertEqual(self.events, ["prefix"])

    def test_tail_failure_is_primary(self):
        sentinel = RuntimeError("tail boom")
        indexer = self._indexer(object(), tail_exc=sentinel)
        attn = self._attention(indexer)
        with self.assertRaises(RuntimeError) as ctx:
            self.attention_cls._forward_prefill_csa(attn, self.x, self.qkv, self.common)
        self.assertIs(ctx.exception, sentinel)
        self.assertEqual(self.events, ["prefix", "tail"])

    def test_empty_prefix_returns_without_tail(self):
        empty = torch.empty((0, 2), dtype=torch.int32)
        indexer = self._indexer(empty)
        out = indexer.forward(self.x, self.qr, self.meta, workspace=self.workspace)
        self.assertIs(out, empty)
        self.assertEqual(self.events, ["prefix"])


def setUpModule():
    run_path(str(FP8.parent / "test" / "cpu_test_utils.py"))[
        "isolate_cpu_test_module"
    ]()


if __name__ == "__main__":
    unittest.main()
