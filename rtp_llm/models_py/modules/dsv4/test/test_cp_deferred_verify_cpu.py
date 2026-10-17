"""CPU tests for deferred compact-CP geometry verification.

Check validation results, event ordering, invalid geometry and blocking fallback
using the production predicates with deterministic event doubles."""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from runpy import run_path
from unittest.mock import patch

import torch

HERE = pathlib.Path(__file__).resolve().parent
DSV4 = HERE.parent
FP8 = DSV4 / "fp8"
RUNTIME_PATH = FP8 / "_compact_cp_runtime.py"
RUNTIME_SRC = RUNTIME_PATH.read_text()
CP_PATH = DSV4 / "cp.py"
CP_SRC = CP_PATH.read_text()
PREFIX = "rtp_llm.models_py.modules.dsv4.fp8"


def _package(name, path=None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path] if path else []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(_package(parent), child, mod)
    return sys.modules[name]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(_package(parent), child, mod)
    return mod


def _zigzag_order():
    return torch.cat(
        [
            torch.cat(
                (
                    torch.arange(r * 512, (r + 1) * 512),
                    torch.arange((7 - r) * 512, (8 - r) * 512),
                )
            )
            for r in range(4)
        ]
    )


def _legacy_verdict(padding_mask, restore_indices):
    """The uncached blocking verdict expression, verbatim."""
    mask = padding_mask.detach().to(device="cpu").reshape(-1)
    restore = restore_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    order = _zigzag_order()
    return bool(
        mask.shape == (4096,)
        and torch.all(mask == 1)
        and restore.shape == (4096,)
        and torch.equal(restore, torch.argsort(order))
    )


def _cp_stub(**overrides):
    base = dict(
        cp_size=4,
        chunk_length=1024,
        padded_seq_len=4096,
        seq_len_full=4096,
        chunk_lengths_per_req=(1024,),
        kv_cache_sharded=False,
        prefix_length=0,
        compact_geometry_verified=False,
        _compact_geometry_pending=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _FakeEvent:
    def __init__(self, fail=False):
        self.syncs = 0
        self.fail = fail

    def synchronize(self):
        self.syncs += 1
        if self.fail:
            raise RuntimeError("event boom")


def _pending(mask, restore, *, fail_event=False):
    """A _PendingVerify instance with the launch machinery faked out."""
    pending = object.__new__(runtime._PendingVerify)
    pending._mask_src = mask
    pending._restore_src = restore
    pending._pinned_mask = mask.clone()
    pending._pinned_restore = restore.clone()
    pending._event = _FakeEvent(fail=fail_event)
    return pending


class VerifyContentParity(unittest.TestCase):
    def test_cached_zigzag_constant_is_byte_identical(self):
        order, expected = runtime._zigzag_constants()
        fresh_order = _zigzag_order()
        self.assertTrue(torch.equal(order, fresh_order))
        self.assertTrue(torch.equal(expected, torch.argsort(fresh_order)))
        # Cached: a second call returns the same objects.
        again = runtime._zigzag_constants()
        self.assertIs(again[0], order)
        self.assertIs(again[1], expected)

    def test_verdict_matrix_matches_legacy(self):
        valid_restore = torch.argsort(_zigzag_order())
        cases = {
            "valid": (torch.ones(4096, dtype=torch.int32), valid_restore),
            "bad_mask": (
                torch.cat(
                    [
                        torch.ones(4095, dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                    ]
                ),
                valid_restore,
            ),
            "bad_restore": (
                torch.ones(4096, dtype=torch.int32),
                torch.arange(4096, dtype=torch.long),  # identity, not zigzag
            ),
            "bad_shape": (
                torch.ones(2048, dtype=torch.int32),
                valid_restore,
            ),
            "bad_dtype_still_value_compared": (
                torch.ones(4096, dtype=torch.int64),
                valid_restore,
            ),
        }
        for name, (mask, restore) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(
                    runtime._verify_content(mask, restore),
                    _legacy_verdict(mask, restore),
                )
        # Sanity: the matrix actually contains both outcomes.
        verdicts = {runtime._verify_content(m, r) for m, r in cases.values()}
        self.assertEqual(verdicts, {True, False})


class PendingVerifyResolve(unittest.TestCase):
    def test_resolve_matches_legacy_verdict(self):
        valid_restore = torch.argsort(_zigzag_order())
        for mask, restore, want in (
            (torch.ones(4096, dtype=torch.int32), valid_restore, True),
            (torch.zeros(4096, dtype=torch.int32), valid_restore, False),
            (torch.ones(4096, dtype=torch.int32), torch.arange(4096), False),
        ):
            pending = _pending(mask, restore)
            with self.subTest(want=want):
                self.assertEqual(pending.resolve(), want)
                self.assertEqual(pending._event.syncs, 1)

    def test_resolve_reads_the_copied_bytes(self):
        # Mutant discipline: corrupt the PINNED copies; the verdict must flip
        # even though the sources are pristine — the resolve compares what the
        # device copied, not the source.
        valid_restore = torch.argsort(_zigzag_order())
        pending = _pending(torch.ones(4096, dtype=torch.int32), valid_restore)
        pending._pinned_restore[0] = pending._pinned_restore[1]
        self.assertFalse(pending.resolve())

    def test_resolve_exception_falls_back_to_legacy_blocking_read(self):
        valid_restore = torch.argsort(_zigzag_order())
        pending = _pending(
            torch.ones(4096, dtype=torch.int32), valid_restore, fail_event=True
        )
        # Corrupt the pinned copies: the fallback must ignore them and read
        # the (pristine) sources with the legacy expression.
        pending._pinned_mask.fill_(0)
        self.assertTrue(pending.resolve())


class ResolveEntryPoint(unittest.TestCase):
    def test_provisional_false_fails_safe(self):
        cp = _cp_stub()
        self.assertFalse(runtime.resolve_compact_geometry_verified(cp))

    def test_eager_true_is_returned_without_pending(self):
        cp = _cp_stub(compact_geometry_verified=True)
        self.assertTrue(runtime.resolve_compact_geometry_verified(cp))

    def test_pending_resolved_once_and_memoized(self):
        valid_restore = torch.argsort(_zigzag_order())
        cp = _cp_stub()
        cp._compact_geometry_pending = _pending(
            torch.ones(4096, dtype=torch.int32), valid_restore
        )
        self.assertTrue(runtime.resolve_compact_geometry_verified(cp))
        self.assertTrue(cp.compact_geometry_verified)
        self.assertIsNone(cp._compact_geometry_pending)
        # Second call: memoized, no re-resolve.
        self.assertTrue(runtime.resolve_compact_geometry_verified(cp))

    def test_pending_false_verdict_memoized(self):
        cp = _cp_stub()
        cp._compact_geometry_pending = _pending(
            torch.zeros(4096, dtype=torch.int32), torch.argsort(_zigzag_order())
        )
        self.assertFalse(runtime.resolve_compact_geometry_verified(cp))
        self.assertFalse(cp.compact_geometry_verified)
        self.assertIsNone(cp._compact_geometry_pending)
        # And stays on the safe path afterwards.
        self.assertFalse(runtime.resolve_compact_geometry_verified(cp))


class SelectGeometryResolvesFirst(unittest.TestCase):
    def test_select_geometry_forces_resolution(self):
        # CPU tensors can never take the compact path (fused must be CUDA),
        # so select_geometry returns None either way — but the pending verdict
        # MUST have been resolved and memoized on the context first.
        module = types.SimpleNamespace()
        meta = types.SimpleNamespace(positions=torch.zeros(4096, dtype=torch.long))
        fused = torch.zeros(4, 4)
        cp = _cp_stub()
        cp._compact_geometry_pending = _pending(
            torch.ones(4096, dtype=torch.int32), torch.argsort(_zigzag_order())
        )
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_COMPRESSOR": "1"}):
            out = runtime.select_geometry(module, cp, meta, fused)
        self.assertIsNone(out)  # CPU tensors: no compact path
        self.assertTrue(cp.compact_geometry_verified)  # but verdict resolved
        self.assertIsNone(cp._compact_geometry_pending)


class VerifiedGeometryHostPath(unittest.TestCase):
    """Host-resident sources keep the eager verdict (no pending)."""

    def _cp(self, **overrides):
        return _cp_stub(**overrides)

    def test_static_gates(self):
        mask = torch.ones(4096, dtype=torch.int32)
        restore = torch.argsort(_zigzag_order())
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_COMPRESSOR": "1"}):
            self.assertTrue(runtime.verified_geometry(self._cp(), mask, restore))
            self.assertFalse(
                runtime.verified_geometry(self._cp(cp_size=2), mask, restore)
            )
            self.assertFalse(
                runtime.verified_geometry(self._cp(prefix_length=4097), mask, restore)
            )
            self.assertFalse(
                runtime.verified_geometry(
                    self._cp(prefix_length=8 * 4096), mask, restore
                )
            )
            # Partial last chunk: not a full 4096-token chunk — static gate
            # declines before any content check (host-known, no readback).
            self.assertFalse(
                runtime.verified_geometry(self._cp(seq_len_full=3000), mask, restore)
            )
            self.assertFalse(
                runtime.verified_geometry(
                    self._cp(chunk_lengths_per_req=(512, 512)), mask, restore
                )
            )
        # Master flag off: never verified.
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_COMPRESSOR": "0"}):
            self.assertFalse(runtime.verified_geometry(self._cp(), mask, restore))

    def test_cpu_sources_take_eager_path_with_no_pending(self):
        mask = torch.ones(4096, dtype=torch.int32)
        restore = torch.argsort(_zigzag_order())
        cp = self._cp()
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_COMPRESSOR": "1"}):
            verdict = runtime.verified_geometry(cp, mask, restore)
        self.assertTrue(verdict)
        self.assertIsNone(cp._compact_geometry_pending)


class FlagAndSourceInvariants(unittest.TestCase):
    def test_deferred_flag_parse_fail_closed(self):
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_DEFERRED_VERIFY": "yes"}):
            with self.assertRaises(ValueError):
                runtime._deferred_verify_enabled()
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_DEFERRED_VERIFY": "0"}):
            self.assertFalse(runtime._deferred_verify_enabled())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_CP_COMPACT_DEFERRED_VERIFY", None)
            self.assertTrue(runtime._deferred_verify_enabled())  # default ON

    def test_plan_cache_flag_parse_fail_closed(self):
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_PLAN_CACHE": "yes"}):
            with self.assertRaises(ValueError):
                runtime._plan_cache_enabled()

    def test_deferred_branch_structure(self):
        tree = ast.parse(RUNTIME_SRC)
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "verified_geometry"
        )
        src = ast.get_source_segment(RUNTIME_SRC, fn)
        self.assertIn("_deferred_verify_enabled()", src)
        self.assertIn("padding_mask.is_cuda", src)
        self.assertIn("restore_indices.is_cuda", src)
        self.assertIn("is_current_stream_capturing()", src)
        self.assertIn("_PendingVerify(", src)
        pending_cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "_PendingVerify"
        )
        init_src = ast.get_source_segment(
            RUNTIME_SRC,
            next(
                n
                for n in pending_cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ),
        )
        self.assertIn("pin_memory=True", init_src)
        self.assertIn("non_blocking=True", init_src)
        self.assertIn("torch.cuda.Event()", init_src)
        resolve_src = ast.get_source_segment(
            RUNTIME_SRC,
            next(
                n
                for n in pending_cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "resolve"
            ),
        )
        self.assertIn("synchronize()", resolve_src)
        self.assertIn("_verify_content(", resolve_src)

    def test_select_geometry_consumes_the_resolver(self):
        tree = ast.parse(RUNTIME_SRC)
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "select_geometry"
        )
        src = ast.get_source_segment(RUNTIME_SRC, fn)
        self.assertIn("resolve_compact_geometry_verified(cp)", src)
        self.assertNotIn('getattr(cp, "compact_geometry_verified"', src)

    def test_cp_context_carries_the_pending_field(self):
        self.assertIn("_compact_geometry_pending", CP_SRC)
        self.assertIn("input_lengths_full_host", CP_SRC)


def setUpModule():
    global runtime
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()

    _package("rtp_llm")

    _package("rtp_llm.models_py")

    _package("rtp_llm.models_py.modules")

    _package("rtp_llm.models_py.modules.dsv4", str(DSV4))

    _package(PREFIX, str(FP8))

    _load(PREFIX + "._cp_packed_rows", FP8 / "_cp_packed_rows.py")

    runtime = _load(PREFIX + "._compact_cp_runtime", RUNTIME_PATH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
