"""CPU tests for the D4 qualify-once cache module: flag parsing, hoist
gating, per-context cache semantics, identity sensitivity and the
frozen-context eager fallback. The attention-side wiring is byte-exactness
gated on GPU (parity sha); these tests cover the cache itself.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "_qualify_once_under_test", ROOT / "_v41_qualify_once.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qo = load_module()


class StubTensor:
    _next_ptr = 4096

    def __init__(self, shape=(1,), dtype="torch.int32", device="cuda:0"):
        self._ptr = StubTensor._next_ptr
        StubTensor._next_ptr += 64
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def data_ptr(self):
        return self._ptr


class FrozenCtx:
    __slots__ = ("cp_size",)

    def __init__(self):
        self.cp_size = 4


ON = {"DSV41_QUALIFY_PREFILL_ONCE": "1"}


class FlagTest(unittest.TestCase):
    def test_default_on(self):
        # R9 decision 1: qualify-once hoisting is the default (byte-exact).
        env = dict(os.environ)
        env.pop("DSV41_QUALIFY_PREFILL_ONCE", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(qo.qualify_prefill_once_enabled())

    def test_one_enables(self):
        with mock.patch.dict(os.environ, ON):
            self.assertTrue(qo.qualify_prefill_once_enabled())

    def test_escape_hatch_and_other_values_disable(self):
        for value in ("0", "true", "yes", ""):
            with mock.patch.dict(os.environ, {"DSV41_QUALIFY_PREFILL_ONCE": value}):
                self.assertFalse(qo.qualify_prefill_once_enabled())


class HoistGateTest(unittest.TestCase):
    def test_requires_flag_ctx_cp_on_and_prefill(self):
        ctx = SimpleNamespace()
        with mock.patch.dict(os.environ, ON):
            self.assertTrue(qo.hoist_active(ctx, True, False))
            self.assertFalse(qo.hoist_active(ctx, False, False))
            self.assertFalse(qo.hoist_active(ctx, True, True))
            self.assertFalse(qo.hoist_active(None, True, False))
        with mock.patch.dict(os.environ, {"DSV41_QUALIFY_PREFILL_ONCE": "0"}):
            self.assertFalse(qo.hoist_active(ctx, True, False))


class MetaCacheTest(unittest.TestCase):
    def setUp(self):
        self.ctx = SimpleNamespace()
        self.il = StubTensor(shape=(1,))
        self.pl = StubTensor(shape=(1,))
        self.key = qo.meta_key(self.il, self.pl, 32768)

    def test_miss_then_roundtrip(self):
        self.assertIsNone(qo.lookup_meta(self.ctx, self.key))
        lengths, ends = object(), object()
        qo.store_meta(self.ctx, self.key, lengths, ends)
        hit = qo.lookup_meta(self.ctx, self.key)
        self.assertIs(hit[0], lengths)
        self.assertIs(hit[1], ends)

    def test_identity_sensitive(self):
        qo.store_meta(self.ctx, self.key, object(), object())
        # different input_lengths buffer -> miss
        self.assertIsNone(
            qo.lookup_meta(
                self.ctx, qo.meta_key(StubTensor(shape=(1,)), self.pl, 32768)
            )
        )
        # different rope limit -> miss
        self.assertIsNone(
            qo.lookup_meta(self.ctx, qo.meta_key(self.il, self.pl, 65536))
        )
        # different shape -> miss
        self.assertIsNone(
            qo.lookup_meta(
                self.ctx, qo.meta_key(StubTensor(shape=(2,)), self.pl, 32768)
            )
        )

    def test_context_lifetime_isolation(self):
        qo.store_meta(self.ctx, self.key, object(), object())
        other = SimpleNamespace()
        self.assertIsNone(qo.lookup_meta(other, self.key))

    def test_frozen_context_falls_back_eager(self):
        frozen = FrozenCtx()
        self.assertIsNone(qo.lookup_meta(frozen, self.key))
        qo.store_meta(frozen, self.key, object(), object())  # no crash
        self.assertIsNone(qo.lookup_meta(frozen, self.key))
        fp = qo.tables_key("swa", StubTensor(), 99, object(), True)
        self.assertFalse(qo.tables_seen(frozen, fp))
        qo.tables_mark(frozen, fp)  # no crash
        self.assertFalse(qo.tables_seen(frozen, fp))


class TablesCacheTest(unittest.TestCase):
    def setUp(self):
        self.ctx = SimpleNamespace()
        self.table = StubTensor(shape=(1, 4))
        self.ends = object()

    def test_seen_after_mark(self):
        fp = qo.tables_key("swa", self.table, 128, self.ends, True)
        self.assertFalse(qo.tables_seen(self.ctx, fp))
        qo.tables_mark(self.ctx, fp)
        self.assertTrue(qo.tables_seen(self.ctx, fp))

    def test_fingerprint_sensitivity(self):
        base = qo.tables_key("swa", self.table, 128, self.ends, True)
        qo.tables_mark(self.ctx, base)
        variants = (
            qo.tables_key("indexer", self.table, 128, self.ends, True),
            qo.tables_key("swa", StubTensor(shape=(1, 4)), 128, self.ends, True),
            qo.tables_key("swa", self.table, 256, self.ends, True),
            qo.tables_key("swa", self.table, 128, object(), True),
            qo.tables_key("swa", self.table, 128, self.ends, False),
        )
        for fp in variants:
            self.assertFalse(qo.tables_seen(self.ctx, fp))

    def test_context_lifetime_isolation(self):
        fp = qo.tables_key("swa", self.table, 128, self.ends, True)
        qo.tables_mark(self.ctx, fp)
        self.assertFalse(qo.tables_seen(SimpleNamespace(), fp))


if __name__ == "__main__":
    unittest.main(verbosity=2)
