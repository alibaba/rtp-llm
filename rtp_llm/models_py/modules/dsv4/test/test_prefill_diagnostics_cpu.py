"""Execute diagnostic accounting and drain decisions with CPU event leaves.

These tests do not qualify CUDA event timing, model imports or GPU lifetimes.
"""

import ast
import contextlib
import importlib.util
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(relative, torch_leaf=None):
    spec = importlib.util.spec_from_file_location(
        "_diagnostic_under_test", ROOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.dict(sys.modules, {} if torch_leaf is None else {"torch": torch_leaf}),
    ):
        spec.loader.exec_module(module)
    return module


class Event:
    def __init__(self, value=0.0, ready=True, **unused):
        self.value, self.ready = value, ready
        self.waits = 0

    def query(self):
        return self.ready

    def synchronize(self):
        self.waits += 1
        self.ready = True

    def elapsed_time(self, other):
        return other.value - self.value

    def record(self):
        pass


def torch_leaf():
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(Event=Event, memory_stats=lambda: {}),
        distributed=types.SimpleNamespace(is_initialized=lambda: False),
    )


class DiagnosticContracts(unittest.TestCase):
    def test_cp_nonblocking_drain_and_forced_flush(self):
        mod = load("_cp_diagnostics.py")
        start, mid, end = Event(1), Event(3), Event(8, ready=False)
        with contextlib.redirect_stderr(io.StringIO()):
            mod._cp_gather_record("kind", start, mid, end, 64)
            mod._cp_gather_stats_drain()
            self.assertEqual(end.waits, 0)
            self.assertEqual(len(mod._cp_gather_stats["pending"]), 1)
            mod._cp_gather_stats_drain(force=True)
        self.assertEqual(end.waits, 1)
        self.assertEqual(mod._cp_gather_stats["launch_ms"], 2)
        self.assertEqual(mod._cp_gather_stats["restore_ms"], 5)
        self.assertEqual(mod._cp_gather_stats["total_ms"], 7)
        self.assertEqual(mod._cp_gather_stats["bytes"], 64)

    def test_cp_report_cadence_and_kind(self):
        mod = load("_cp_diagnostics.py")
        mod._CP_GATHER_STATS_EVERY = 2
        out = io.StringIO()
        with contextlib.redirect_stderr(out):
            mod._cp_gather_record("x", Event(1), None, Event(3), 32)
            self.assertNotIn("calls=", out.getvalue())
            mod._cp_gather_record("x", Event(4), None, Event(7), 64)
        self.assertIn("calls=2", out.getvalue())
        self.assertEqual(mod._cp_gather_stats["total_ms"], 5)
        self.assertEqual(mod._cp_gather_stats["restore_ms"], 0)
        self.assertEqual(mod._cp_gather_kind("foo.L17.bar"), "foo.L*.bar")

    def test_actual_sync_cp_caller_has_no_mid_event(self):
        mod = load("_cp_diagnostics.py")
        tree = ast.parse((ROOT / "cp.py").read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "cp_all_gather_full"
        )
        module = ast.parse("from __future__ import annotations\n")
        module.body.append(fn)
        ns = dict(
            torch=torch_leaf(),
            _CP_GATHER_STATS=True,
            _DEFAULT_CP_PROFILE_NAME="dsv4.cp.all_gather",
            _cp_gather_stats_drain=mod._cp_gather_stats_drain,
            _cp_gather_kind=mod._cp_gather_kind,
            _cp_gather_record=mod._cp_gather_record,
            _cp_gather_2d=lambda value, ctx: value,
            all_gather=lambda value, **kw: value,
            _cp_restore_gathered_full_2d=lambda value, ctx: value,
            record_function_range=lambda *a: contextlib.nullcontext(),
            Group=types.SimpleNamespace(TP="tp"),
        )
        exec(
            compile(ast.fix_missing_locations(module), str(ROOT / "cp.py"), "exec"), ns
        )
        tensor = types.SimpleNamespace(numel=lambda: 4, element_size=lambda: 2)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIs(ns[fn.name](tensor, types.SimpleNamespace(cp_size=4)), tensor)
        pending = mod._cp_gather_stats["pending"][0]
        self.assertIsNone(pending[2])
        self.assertEqual(pending[4], 32)

    def test_forward_disabled_and_phase_arithmetic(self):
        leaf = torch_leaf()
        leaf.cuda.memory_stats = lambda: (_ for _ in ()).throw(
            AssertionError("disabled read")
        )
        mod = load("prefill/_diagnostics.py", leaf)
        self.assertFalse(mod._FWD_STATS)
        self.assertFalse(mod._FWD_GPU)
        self.assertFalse(mod._FWD_PROFILE)
        self.assertIsNone(mod._fwd_stats_snap())
        out = io.StringIO()
        with contextlib.redirect_stderr(out):
            mod._fwd_stats_report_row(
                n_tokens=1024,
                cp_size=4,
                marks=dict(
                    entry=1.0,
                    cpctx=1.002,
                    pos=1.003,
                    embed=1.004,
                    meta=1.007,
                    loop=1.017,
                    tail=1.02,
                ),
                mem_before=(1, 2, 3),
                mem_after=(2, 2, 5),
            )
        self.assertIn("total=20.00", out.getvalue())
        self.assertIn("meta=3.00", out.getvalue())
        self.assertIn("loop=10.00", out.getvalue())
        self.assertIn("d_num_device_alloc=1", out.getvalue())
        self.assertIn("d_num_alloc_retries=2", out.getvalue())

    def test_forward_gpu_queue_does_not_wait_unless_forced(self):
        mod = load("prefill/_diagnostics.py", torch_leaf())
        start, end = Event(5), Event(12, ready=False)
        mod._FWD_GPU_PENDING.append((1024, 4, start, end))
        mod._fwd_gpu_drain()
        self.assertEqual(end.waits, 0)
        self.assertEqual(len(mod._FWD_GPU_PENDING), 1)
        with contextlib.redirect_stderr(io.StringIO()):
            mod._fwd_gpu_drain(force=True)
        self.assertEqual(end.waits, 1)
        self.assertEqual(mod._FWD_GPU_ROWS, [(1024, 4, 7.0)])


if __name__ == "__main__":
    unittest.main()
