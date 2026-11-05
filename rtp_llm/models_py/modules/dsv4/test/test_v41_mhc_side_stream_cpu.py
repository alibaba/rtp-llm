"""CPU tests for the R25/O1 mHC side-stream policy module.

Hermetic namespace-stub loading as in test_v41_deferred_checks_cpu. The
policy module needs torch only for the (untested here) stream factory;
flag semantics and the diagnostics accumulator are pure python.
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch  # noqa: F401  (policy module imports it)

ROOT = Path(__file__).resolve().parents[5]
for suffix in [
    "",
    ".models_py",
    ".models_py.modules",
    ".models_py.modules.dsv4",
    ".models_py.modules.dsv4.hc",
]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

side = importlib.import_module("rtp_llm.models_py.modules.dsv4.hc._v41_mhc_side_stream")


class SideStreamFlagTest(unittest.TestCase):
    def test_absent_is_on(self):
        # R27: adopted as default (escape = explicit "0").
        env = dict(os.environ)
        env.pop(side.MHC_SIDE_STREAM_FLAG, None)
        env.pop(side.MHC_SIDE_STATS_FLAG, None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(side.mhc_side_stream_enabled())
            self.assertFalse(side.mhc_side_stats_enabled())

    def test_value_based(self):
        for value, expected in (
            ("1", True),
            ("0", False),
            ("true", False),
            ("", False),
            ("2", False),
        ):
            with mock.patch.dict(
                os.environ, {side.MHC_SIDE_STREAM_FLAG: value}, clear=False
            ):
                self.assertEqual(side.mhc_side_stream_enabled(), expected, value)

    def test_stats_flag_independent(self):
        with mock.patch.dict(os.environ, {side.MHC_SIDE_STATS_FLAG: "1"}, clear=True):
            self.assertTrue(side.mhc_side_stats_enabled())
            # R27: the side stream itself is default-on; only the
            # diagnostics flag stays default-off.
            self.assertTrue(side.mhc_side_stream_enabled())


class StatsAccumulatorTest(unittest.TestCase):
    def setUp(self):
        self._saved = dict(side._STATS)

    def tearDown(self):
        side._STATS.clear()
        side._STATS.update(self._saved)

    def test_note_join_accumulates(self):
        side._STATS.update(joins=0, already_done=0, busy_ms=0.0, timed=0)
        side.mhc_side_note_join(True, 0.10)
        side.mhc_side_note_join(False, 0.30)
        self.assertEqual(side._STATS["joins"], 2)
        self.assertEqual(side._STATS["already_done"], 1)
        self.assertAlmostEqual(side._STATS["busy_ms"], 0.40)
        self.assertEqual(side._STATS["timed"], 2)

    def test_note_join_without_timing(self):
        side._STATS.update(joins=0, already_done=0, busy_ms=0.0, timed=0)
        side.mhc_side_note_join(True)
        self.assertEqual(side._STATS["joins"], 1)
        self.assertEqual(side._STATS["timed"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=0)
    print("OK")
