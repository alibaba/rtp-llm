"""CPU tests for the R19/D sync fixes: D1 worker-level expected-lengths
cache and D2 compact-CP transport metadata cache. Both are pure host-side
caching of read-only uploads; the tests pin the flag contracts (default
off = untouched legacy path), hit identity, key sensitivity, and the
bounded cache size. GPU/collective behavior is proven by the node parity
runs ([ELCACHE]/[TPMETA] announces + sha gate) and the sync-scan proof run.
"""

import importlib.util
import os
import pathlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


el = _load("_r19_el", "_v41_expected_lengths.py")


class ExpectedLengthsWorkerCacheTest(unittest.TestCase):
    def _cp(self, values=(1024,)):
        return SimpleNamespace(chunk_lengths_per_req=values)

    def _input_lengths(self, device=None):
        return torch.tensor([32768], dtype=torch.int32, device=device)

    def test_default_off_uses_legacy_path(self):
        clean = dict(os.environ)
        clean.pop("DSV41_EXPECTED_LENGTHS_WORKER_CACHE", None)
        clean.pop("DSV41_EXPECTED_LENGTHS_CACHE", None)
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertFalse(el.expected_lengths_worker_cache_enabled())
            a = el.expected_chunk_lengths(self._cp(), self._input_lengths())
            b = el.expected_chunk_lengths(self._cp(), self._input_lengths())
            self.assertEqual(a.tolist(), [1024])
            self.assertIsNot(a, b)  # fresh upload per call (legacy)

    def test_worker_cache_hit_identity_and_values(self):
        el._WORKER_CACHE.clear()
        with mock.patch.dict(os.environ, {"DSV41_EXPECTED_LENGTHS_WORKER_CACHE": "1"}):
            first = el.expected_chunk_lengths(self._cp(), self._input_lengths())
            second = el.expected_chunk_lengths(self._cp(), self._input_lengths())
            self.assertIs(first, second)  # same tensor object reused
            self.assertEqual(second.tolist(), [1024])
            # different values -> different entry
            other = el.expected_chunk_lengths(self._cp((768,)), self._input_lengths())
            self.assertIsNot(other, first)
            self.assertEqual(other.tolist(), [768])

    def test_worker_cache_bounded(self):
        el._WORKER_CACHE.clear()
        with mock.patch.dict(os.environ, {"DSV41_EXPECTED_LENGTHS_WORKER_CACHE": "1"}):
            for i in range(el._WORKER_CACHE_LIMIT + 10):
                el.expected_chunk_lengths(self._cp((i,)), self._input_lengths())
            self.assertLessEqual(len(el._WORKER_CACHE), el._WORKER_CACHE_LIMIT)

    def test_g2_context_cache_still_works_independently(self):
        with mock.patch.dict(os.environ, {"DSV41_EXPECTED_LENGTHS_CACHE": "1"}):
            cp = self._cp()
            a = el.expected_chunk_lengths(cp, self._input_lengths())
            b = el.expected_chunk_lengths(cp, self._input_lengths())
            self.assertIs(a, b)  # per-context cache hit


# ------------------------- D2 transport meta cache -------------------------
import sys
import types

# The transport module imports plan constants from its package; stub the
# relative import by loading the package path directly.
_pkg = types.ModuleType("_r19_tp_pkg")
_pkg.__path__ = [str(ROOT / "fp8")]
sys.modules["_r19_tp_pkg"] = _pkg
_spec = importlib.util.spec_from_file_location(
    "_r19_tp_pkg._v41_compact_cp_transport",
    ROOT / "fp8" / "_v41_compact_cp_transport.py",
)
tp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tp
_spec.loader.exec_module(tp)


def _plan(rows=1024, ratio=1, start=0, chunk_tokens=4096, tail_count=True, header=None):
    return SimpleNamespace(
        header=tuple(header if header is not None else range(16)),
        boundary_positions=tuple(range(rows)),
        rows=rows,
        ratio=ratio,
        start=start,
        chunk_tokens=chunk_tokens,
        tail_count=tail_count,
    )


class TransportMetaCacheTest(unittest.TestCase):
    def test_default_off_fresh_objects(self):
        clean = dict(os.environ)
        clean.pop("DSV41_PP_TRANSPORT_META_CACHE", None)
        with mock.patch.dict(os.environ, clean, clear=True):
            plans = (_plan(), _plan(ratio=2))
            a = tp.prepare_metadata_batch(plans, torch.device("cpu"))
            b = tp.prepare_metadata_batch(plans, torch.device("cpu"))
            self.assertIsNot(a, b)
            self.assertTrue(torch.equal(a[0].positions, b[0].positions))

    def test_cache_on_hit_identity_and_content(self):
        tp._META_CACHE.clear()
        with mock.patch.dict(os.environ, {"DSV41_PP_TRANSPORT_META_CACHE": "1"}):
            plans = (_plan(), _plan(ratio=2))
            a = tp.prepare_metadata_batch(plans, torch.device("cpu"))
            b = tp.prepare_metadata_batch(plans, torch.device("cpu"))
            self.assertIs(a, b)
            self.assertEqual(a[0].header.tolist(), list(range(16)))
            self.assertEqual(a[0].positions.tolist(), list(range(1024)))
            self.assertIsNone(a[0].tail_identity)
            self.assertEqual(a[1].tail_identity.tolist(), [[4094, 1], [4095, 1]])

    def test_key_sensitivity(self):
        tp._META_CACHE.clear()
        with mock.patch.dict(os.environ, {"DSV41_PP_TRANSPORT_META_CACHE": "1"}):
            # ratio=2 plans fold start/chunk_tokens into the tail identity
            # values, so different starts must produce different cache keys
            a = tp.prepare_metadata_batch(
                (_plan(ratio=2, start=0),), torch.device("cpu")
            )
            b = tp.prepare_metadata_batch(
                (_plan(ratio=2, start=4096),), torch.device("cpu")
            )
            self.assertIsNot(a, b)
            self.assertNotEqual(
                a[0].tail_identity.tolist(), b[0].tail_identity.tolist()
            )

    def test_validation_stacks_cached(self):
        tp._META_CACHE.clear()
        with mock.patch.dict(os.environ, {"DSV41_PP_TRANSPORT_META_CACHE": "1"}):
            groups = (tuple(_plan(ratio=2, start=s) for s in (0, 4096, 8192, 12288)),)
            a = tp.prepare_validation_stacks(groups, torch.device("cpu"))
            b = tp.prepare_validation_stacks(groups, torch.device("cpu"))
            self.assertIs(a, b)
            stack = a[0]
            self.assertEqual(tuple(stack.header.shape), (4, 16))
            self.assertEqual(tuple(stack.positions.shape), (4, 1024))
            self.assertEqual(stack.tail_state_exempt.tolist(), [True] * 4)

    def test_cache_bounded(self):
        tp._META_CACHE.clear()
        with mock.patch.dict(os.environ, {"DSV41_PP_TRANSPORT_META_CACHE": "1"}):
            for i in range(tp._META_CACHE_LIMIT + 10):
                tp.prepare_metadata_batch((_plan(start=i),), torch.device("cpu"))
            self.assertLessEqual(len(tp._META_CACHE), tp._META_CACHE_LIMIT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
