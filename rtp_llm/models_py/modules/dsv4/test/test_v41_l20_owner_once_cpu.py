"""CPU tests for the R9 L20 owner-once cache module: flag parsing,
fingerprint semantics, per-context cache, frozen-context eager fallback.
The require_owner wiring is byte-exactness gated on GPU (parity sha).
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
        "_l20_owner_once_under_test", ROOT / "_v41_l20_owner_once.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oo = load_module()


class StubTable:
    _next = 8192

    def __init__(self, shape=(2, 4)):
        self._ptr = StubTable._next
        StubTable._next += 256
        self.shape = shape

    def data_ptr(self):
        return self._ptr


class FrozenCtx:
    __slots__ = ("cp_size",)

    def __init__(self):
        self.cp_size = 4


def attention(kv=None, hca=None, idx=None):
    tables = {}
    if hca is not None:
        tables["hca_kv"] = hca
    if idx is not None:
        tables["indexer_kv"] = idx
    return SimpleNamespace(_block_tables_by_type=tables, _kv_cache=kv)


ON = {"DSV41_L20_OWNER_ONCE": "1"}


class FlagTest(unittest.TestCase):
    def test_default_off(self):
        env = dict(os.environ)
        env.pop("DSV41_L20_OWNER_ONCE", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(oo.l20_owner_once_enabled())

    def test_one_enables(self):
        with mock.patch.dict(os.environ, ON):
            self.assertTrue(oo.l20_owner_once_enabled())

    def test_other_values_disable(self):
        for value in ("0", "true", ""):
            with mock.patch.dict(os.environ, {"DSV41_L20_OWNER_ONCE": value}):
                self.assertFalse(oo.l20_owner_once_enabled())


class FingerprintTest(unittest.TestCase):
    def setUp(self):
        self.src = object()
        self.kv = object()
        self.hca = StubTable()
        self.idx = StubTable()
        self.att = attention(self.kv, self.hca, self.idx)

    def test_stable_for_same_objects(self):
        self.assertEqual(
            oo.owner_fingerprint(self.src, self.att),
            oo.owner_fingerprint(self.src, self.att),
        )

    def test_sensitive_to_source_kv_tables(self):
        base = oo.owner_fingerprint(self.src, self.att)
        self.assertNotEqual(base, oo.owner_fingerprint(object(), self.att))
        self.assertNotEqual(
            base,
            oo.owner_fingerprint(self.src, attention(object(), self.hca, self.idx)),
        )
        self.assertNotEqual(
            base,
            oo.owner_fingerprint(self.src, attention(self.kv, StubTable(), self.idx)),
        )
        self.assertNotEqual(
            base,
            oo.owner_fingerprint(
                self.src, attention(self.kv, self.hca, StubTable(shape=(2, 8)))
            ),
        )

    def test_none_without_fingerprintable_tables(self):
        self.assertIsNone(oo.owner_fingerprint(self.src, SimpleNamespace()))
        self.assertIsNone(
            oo.owner_fingerprint(self.src, attention(self.kv, object(), self.idx))
        )


class CacheTest(unittest.TestCase):
    def test_seen_after_mark(self):
        ctx = SimpleNamespace()
        fp = ("fp", 1)
        self.assertFalse(oo.owner_seen(ctx, fp))
        oo.owner_mark(ctx, fp)
        self.assertTrue(oo.owner_seen(ctx, fp))

    def test_context_lifetime_isolation(self):
        ctx = SimpleNamespace()
        oo.owner_mark(ctx, ("fp", 2))
        self.assertFalse(oo.owner_seen(SimpleNamespace(), ("fp", 2)))

    def test_frozen_context_falls_back_eager(self):
        frozen = FrozenCtx()
        self.assertFalse(oo.owner_seen(frozen, ("fp", 3)))
        oo.owner_mark(frozen, ("fp", 3))  # no crash
        self.assertFalse(oo.owner_seen(frozen, ("fp", 3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
