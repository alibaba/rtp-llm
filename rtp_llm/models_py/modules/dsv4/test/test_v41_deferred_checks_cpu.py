"""CPU tests for the G3 deferred device-assert grouping.

Hermetic namespace-stub loading as in moe_cache_test_support.
"""

import contextlib
import importlib
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[5]
for suffix in [
    "",
    ".models_py",
    ".models_py.modules",
    ".models_py.modules.dsv4",
]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

deferred = importlib.import_module(
    "rtp_llm.models_py.modules.dsv4._v41_deferred_checks"
)


class DeferredChecksTest(unittest.TestCase):
    def setUp(self):
        deferred._pending.clear()
        deferred._registry_seen.clear()
        self.addCleanup(deferred._pending.clear)
        self.addCleanup(deferred._registry_seen.clear)

    def _capture_asserts(self):
        calls = []

        def fake_assert(cond, message=None):
            calls.append((cond, message))

        return calls, mock.patch.object(torch, "_assert_async", side_effect=fake_assert)

    def test_flag_default_off(self):
        env = dict(os.environ)
        env.pop("DSV41_GROUPED_ASSERTS", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(deferred.grouped_asserts_enabled())
        with mock.patch.dict(os.environ, {"DSV41_GROUPED_ASSERTS": "1"}):
            self.assertTrue(deferred.grouped_asserts_enabled())

    def test_flush_groups_into_single_assert_with_short_message(self):
        calls, patcher = self._capture_asserts()
        err = io.StringIO()
        with patcher, contextlib.redirect_stderr(err):
            deferred.add_check(torch.tensor(True), "a")
            deferred.add_check(torch.tensor(True), "b")
            deferred.add_check(torch.tensor(True), "c")
            self.assertEqual(deferred.pending_count(), 3)
            deferred.flush_checks("layer")
        self.assertEqual(len(calls), 1)
        cond, message = calls[0]
        self.assertTrue(bool(cond))
        self.assertIn("3 grouped", message)
        self.assertLessEqual(len(message), deferred._MAX_MESSAGE)
        registry = err.getvalue()
        for part in ("a", "b", "c"):
            self.assertIn(part, registry)
        self.assertEqual(deferred.pending_count(), 0)

    def test_registry_published_once_per_composition(self):
        calls, patcher = self._capture_asserts()
        err = io.StringIO()
        with patcher, contextlib.redirect_stderr(err):
            for _ in range(3):
                deferred.add_check(torch.tensor(True), "x")
                deferred.add_check(torch.tensor(True), "y")
                deferred.flush_checks("layer")
        self.assertEqual(len(calls), 3)
        self.assertEqual(err.getvalue().count("check registry"), 1)

    def test_flush_preserves_failure(self):
        calls, patcher = self._capture_asserts()
        err = io.StringIO()
        with patcher, contextlib.redirect_stderr(err):
            deferred.add_check(torch.tensor(True), "ok")
            deferred.add_check(torch.tensor(False), "bad")
            deferred.flush_checks("layer")
        self.assertEqual(len(calls), 1)
        self.assertFalse(bool(calls[0][0]))
        self.assertIn("bad", err.getvalue())

    def test_single_check_keeps_own_message(self):
        calls, patcher = self._capture_asserts()
        with patcher:
            deferred.add_check(torch.tensor(True), "solo")
            deferred.flush_checks("layer")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "solo")

    def test_long_single_message_is_truncated(self):
        calls, patcher = self._capture_asserts()
        with patcher:
            deferred.add_check(torch.tensor(True), "m" * 5000)
            deferred.flush_checks("layer")
        self.assertLessEqual(len(calls[0][1]), deferred._MAX_MESSAGE)

    def test_overflow_autoflush(self):
        calls, patcher = self._capture_asserts()
        err = io.StringIO()
        with patcher, contextlib.redirect_stderr(err):
            for i in range(deferred._MAX_PENDING):
                deferred.add_check(torch.tensor(True), f"m{i}")
            self.assertEqual(len(calls), 1)
            self.assertEqual(deferred.pending_count(), 0)

    def test_empty_flush_is_noop(self):
        calls, patcher = self._capture_asserts()
        with patcher:
            deferred.flush_checks("layer")
            deferred.flush_checks("stage")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
