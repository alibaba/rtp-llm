"""CPU tests for the eager (side-stream) authoritative count gather wiring.

Drives the REAL ``forward_ep_plan`` module and the REAL extracted
``NcclEpMxfp8Strategy`` count methods with CPU leaf doubles.  No CUDA, NCCL,
model weights or numerical qualification is implied; the CUDA branch of the
strategy hook is covered by source invariants and the later model leg.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from contextlib import nullcontext
from runpy import run_path
from unittest.mock import patch

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
WT = ROOT.parents[4]
PLAN_NAME = "rtp_llm.models_py.modules.dsv4.moe.forward_ep_plan"


def _package(name, path=None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path] if path else []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(_package(parent), child, mod)
    return sys.modules[name]


def _load_plan():
    if PLAN_NAME in sys.modules:
        return sys.modules[PLAN_NAME]
    dsv4 = ROOT
    moe = ROOT / "moe"
    _package("rtp_llm")
    _package("rtp_llm.models_py")
    _package("rtp_llm.models_py.modules")
    _package("rtp_llm.models_py.modules.dsv4", str(dsv4))
    _package("rtp_llm.models_py.modules.dsv4.moe", str(moe))
    spec = importlib.util.spec_from_file_location(PLAN_NAME, moe / "forward_ep_plan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PLAN_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


STRATEGY_PATH = ROOT / "moe/strategies/nccl_ep_mxfp8.py"
STRATEGY_SRC = STRATEGY_PATH.read_text()


def extract(path, names, env, cls=None):
    tree = ast.parse(path.read_text())
    body = (
        tree.body
        if cls is None
        else next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    )
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(
        names
    ), f"missing {set(names) - {n.name for n in nodes}} in {path}"
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    mod = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    exec(compile(mod, str(path), "exec"), env)
    return {n.name: env[n.name] for n in nodes}


# The extracted module-level flag parser needs its module constant; anchored by
# the SourceInvariants test below (asserts the real module defines this name).
_FLAG_NAME = "DSV4_MOE_EAGER_COUNT_GATHER"


class _BanHostSync:
    """Fail any ``Tensor.item`` / ``Tensor.cpu`` call inside the region."""

    def __enter__(self):
        def banned(*a, **k):
            raise AssertionError("host synchronizing call on the hot path")

        self._patchers = [
            patch.object(torch.Tensor, "item", banned),
            patch.object(torch.Tensor, "cpu", banned),
        ]
        for p in self._patchers:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self._patchers:
            p.stop()
        return False


class FakeStrategy:
    """Interface double: stage geometry + optional eager hooks + legacy gather."""

    name = "fork_nccl_mxfp8"
    requires_synchronized_chunk_schedule = True

    def __init__(self, ctx, rank, counts, events, *, with_hooks, lazy_via_tensor=False):
        self.cfg = types.SimpleNamespace(stage_context=ctx, ep_rank=rank, ep_size=4)
        self._counts = list(counts)
        self._events = events
        self._pending_counts = None
        self._chunk_extent_tensor = None
        self._count_gather = None
        self._count_tensor = None
        self._lazy_via_tensor = lazy_via_tensor
        self._ctx = ctx
        self._rank = rank
        if with_hooks:
            self.start_authoritative_counts_gather = self._start
            self.finish_authoritative_counts_gather = self._finish

    def _stage(self):
        return self._ctx.process_group, self._ctx.group_size, self._rank

    def _start(self, local_rows, device):
        self._events.append(("start", int(local_rows), str(device)))
        return {"counts": list(self._counts), "rows": int(local_rows)}

    def _finish(self, handle):
        self._events.append(("finish",))
        return list(handle["counts"])

    def _gather_counts(self, n_local, group, world, device):
        self._events.append(("lazy_gather", int(n_local)))
        if self._lazy_via_tensor:
            # Legacy device path: readback through the host-syncing API.
            return [int(v) for v in torch.tensor(self._counts).cpu().tolist()]
        return list(self._counts)

    def _exchange_counts(self, n_local, group, world, device):
        return self._gather_counts(n_local, group, world, device)


def make_ctx(rank=0):
    return types.SimpleNamespace(
        process_group=object(),
        world_ranks=(0, 1, 2, 3),
        group_rank=rank,
        group_size=4,
        pp_rank=0,
        generation=0,
    )


def make_model(strategies, width):
    layers = [
        types.SimpleNamespace(
            ffn=types.SimpleNamespace(
                _strategy=s, _is_decode_role=False, max_tokens_per_rank=width
            )
        )
        for s in strategies
    ]
    return types.SimpleNamespace(layers=layers, commit_only=False)


def drive_chunk_forward(model, strategies, local_rows, width, device):
    """One simulated chunk forward: per-layer extent + one subchunk count read."""
    cp_ctx = types.SimpleNamespace(cp_size=4, cp_rank=0, chunk_length=local_rows)
    observed = {}
    with plan.prefill_forward_scope(model, cp_ctx, local_rows, device):
        observed["extent0"] = strategies[0].synchronized_chunk_extent(
            local_rows, device
        )
        observed["extent1"] = strategies[1].synchronized_chunk_extent(
            local_rows, device
        )
        group, world, _ = strategies[0]._stage()
        with strategies[0].forward_subchunk_scope(0, width, local_rows):
            observed["sub0"] = tuple(
                strategies[0]._counts_for_forward(width, group, world, device)
            )
        with strategies[1].forward_subchunk_scope(width, width, local_rows):
            observed["sub1"] = tuple(
                strategies[1]._counts_for_forward(width, group, world, device)
            )
    return observed


class EagerCountGatherWiring(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict(os.environ, {"DSV4_MOE_FORWARD_COUNT_PLAN": "1"})
        self._env.start()

    def tearDown(self):
        self._env.stop()

    def test_eager_started_at_scope_entry_consumed_once_values_exact(self):
        events = []
        ctx = make_ctx()
        strategies = [
            FakeStrategy(ctx, 0, [8, 8, 8, 8], events, with_hooks=True),
            FakeStrategy(ctx, 0, [8, 8, 8, 8], events, with_hooks=True),
        ]
        model = make_model(strategies, width=2)
        with _BanHostSync():
            observed = drive_chunk_forward(model, strategies, 8, 2, torch.device("cpu"))
        self.assertEqual(observed["extent0"], 8)
        self.assertEqual(observed["extent1"], 8)
        self.assertEqual(observed["sub0"], (2, 2, 2, 2))
        self.assertEqual(observed["sub1"], (2, 2, 2, 2))
        # The gather launched exactly once, at scope entry, before any extent
        # consumption; the banked plan served every later call.
        self.assertEqual(events[:1], [("start", 8, "cpu")])
        self.assertEqual(events.count(("finish",)), 1)
        self.assertEqual([e for e in events if e[0] == "lazy_gather"], [])

    def test_eager_ragged_counts_feed_plan_math(self):
        events = []
        ctx = make_ctx()
        counts = [8, 6, 8, 8]  # rank-nonuniform but locally consistent
        strategies = [
            FakeStrategy(ctx, 0, counts, events, with_hooks=True),
            FakeStrategy(ctx, 0, counts, events, with_hooks=True),
        ]
        model = make_model(strategies, width=2)
        with _BanHostSync():
            observed = drive_chunk_forward(model, strategies, 8, 2, torch.device("cpu"))
        self.assertEqual(observed["extent0"], 8)  # max over ragged counts
        self.assertEqual(observed["sub0"], (2, 2, 2, 2))
        # subchunk at start=2: min(2, count-2) → ragged counts show through
        self.assertEqual(observed["sub1"], (2, 2, 2, 2))

    def test_lazy_path_without_hooks_still_syncs(self):
        """Control: the ban must bite the legacy gather (anti-vacuity)."""
        events = []
        ctx = make_ctx()
        strategies = [
            FakeStrategy(
                ctx, 0, [8, 8, 8, 8], events, with_hooks=False, lazy_via_tensor=True
            ),
            FakeStrategy(
                ctx, 0, [8, 8, 8, 8], events, with_hooks=False, lazy_via_tensor=True
            ),
        ]
        model = make_model(strategies, width=2)
        with self.assertRaises(AssertionError):
            with _BanHostSync():
                drive_chunk_forward(model, strategies, 8, 2, torch.device("cpu"))
        self.assertEqual(events[0][0], "lazy_gather")

    def test_local_count_mismatch_still_rejected_through_eager_path(self):
        events = []
        ctx = make_ctx()
        # counts[ep_rank]=6 disagrees with the forward's local_rows=8.
        strategies = [
            FakeStrategy(ctx, 0, [6, 8, 8, 8], events, with_hooks=True),
            FakeStrategy(ctx, 0, [6, 8, 8, 8], events, with_hooks=True),
        ]
        model = make_model(strategies, width=2)
        with self.assertRaisesRegex(RuntimeError, "disagrees with this forward"):
            drive_chunk_forward(model, strategies, 8, 2, torch.device("cpu"))
        self.assertEqual(events.count(("finish",)), 1)

    def test_no_eager_launch_when_scope_ineligible(self):
        events = []
        ctx = make_ctx()
        strategies = [
            FakeStrategy(ctx, 0, [8, 8, 8, 8], events, with_hooks=True),
            FakeStrategy(ctx, 0, [8, 8, 8, 8], events, with_hooks=True),
        ]
        model = make_model(strategies, width=2)
        cp_ctx = types.SimpleNamespace(cp_size=2, cp_rank=0, chunk_length=8)
        with plan.prefill_forward_scope(model, cp_ctx, 8, torch.device("cpu")):
            self.assertIsNone(plan.current_scope())
        self.assertEqual(events, [])


class RealHookGates(unittest.TestCase):
    """The real extracted start/finish methods: flag parse, CPU gate, finish math."""

    def test_flag_off_returns_none(self):
        st = types.SimpleNamespace(_eager_counts_state=None)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_MOE_EAGER_COUNT_GATHER", None)
            handle = strategy_methods["start_authoritative_counts_gather"](
                st, 8, torch.device("cpu")
            )
        self.assertIsNone(handle)

    def test_flag_invalid_value_raises(self):
        st = types.SimpleNamespace(_eager_counts_state=None)
        with patch.dict(os.environ, {"DSV4_MOE_EAGER_COUNT_GATHER": "yes"}):
            with self.assertRaises(ValueError):
                strategy_methods["start_authoritative_counts_gather"](
                    st, 8, torch.device("cpu")
                )

    def test_cpu_device_gate_does_not_touch_cuda(self):
        st = types.SimpleNamespace(_eager_counts_state=None)

        def no_cuda_stream(*a, **k):
            raise AssertionError("CUDA touched on the CPU gate path")

        with patch.dict(os.environ, {"DSV4_MOE_EAGER_COUNT_GATHER": "1"}):
            with patch.object(torch.cuda, "Stream", no_cuda_stream):
                handle = strategy_methods["start_authoritative_counts_gather"](
                    st, 8, torch.device("cpu")
                )
        self.assertIsNone(handle)

    def test_finish_reads_back_pinned_values(self):
        class _Event:
            def __init__(self):
                self.synced = 0

            def synchronize(self):
                self.synced += 1

        event = _Event()
        state = {"event": event, "pin": torch.arange(4, dtype=torch.int64) + 5}
        result = strategy_methods["finish_authoritative_counts_gather"](
            None, (state, 4)
        )
        self.assertEqual(result, [5, 6, 7, 8])
        self.assertEqual(event.synced, 1)

    def test_flag_off_scope_falls_back_to_lazy_gather(self):
        """Flag OFF + real start method → no eager handle → identical lazy path."""
        events = []
        ctx = make_ctx()
        strategies = [
            FakeStrategy(ctx, 0, [8, 8, 8, 8], events, with_hooks=False),
            FakeStrategy(ctx, 0, [8, 8, 8, 8], events, with_hooks=False),
        ]
        # Bind the REAL hook: with the flag off it must decline (return None).
        for s in strategies:
            s.start_authoritative_counts_gather = types.MethodType(
                strategy_methods["start_authoritative_counts_gather"], s
            )
            s._eager_counts_state = None
        model = make_model(strategies, width=2)
        with patch.dict(os.environ, {"DSV4_MOE_FORWARD_COUNT_PLAN": "1"}):
            os.environ.pop("DSV4_MOE_EAGER_COUNT_GATHER", None)
            observed = drive_chunk_forward(model, strategies, 8, 2, torch.device("cpu"))
        self.assertEqual(observed["extent0"], 8)
        self.assertEqual(observed["sub0"], (2, 2, 2, 2))
        # No eager handle was armed; the legacy gather served the establishment.
        self.assertEqual(
            [e for e in events if e[0] == "lazy_gather"], [("lazy_gather", 8)]
        )


class SourceInvariants(unittest.TestCase):
    """The CUDA branch of the real hook: checked structurally (GPU leg proves it)."""

    def _method_src(self, name):
        tree = ast.parse(STRATEGY_SRC)
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "NcclEpMxfp8Strategy"
        )
        node = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        return ast.get_source_segment(STRATEGY_SRC, node)

    def test_start_uses_side_stream_pinned_nonblocking_event(self):
        src = self._method_src("start_authoritative_counts_gather")
        self.assertIn("torch.cuda.Stream(", src)
        self.assertIn("pin_memory=True", src)
        self.assertIn("non_blocking=True", src)
        self.assertIn(".record(", src)
        self.assertIn("all_gather_into_tensor", src)
        # Dedicated buffers: never shares the synchronous path's tensors.
        self.assertNotIn("self._count_gather", src)
        self.assertNotIn("self._count_tensor", src)
        # The gather input stays a host-known scalar fill.
        self.assertIn('state["dev_in"].fill_(int(local_rows))', src)

    def test_scope_branches_prefer_scope_eager_gather(self):
        for name in ("synchronized_chunk_extent", "_counts_for_forward"):
            src = self._method_src(name)
            self.assertIn("scope.eager_gather", src, name)

    def test_finish_waits_event_then_reads_pinned(self):
        src = self._method_src("finish_authoritative_counts_gather")
        self.assertIn('state["event"].synchronize()', src)
        self.assertIn('state["pin"]', src)
        self.assertNotIn(".cuda()", src)

    def test_flag_name_constant_anchored(self):
        self.assertIn(
            '_EAGER_COUNT_GATHER_FLAG = "DSV4_MOE_EAGER_COUNT_GATHER"', STRATEGY_SRC
        )


class StartPosPositions(unittest.TestCase):
    """transformer.py's standalone forward builds positions without a D2H."""

    @staticmethod
    def _fn():
        path = ROOT / "transformer.py"
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_positions_from_start_pos"
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        mod = ast.fix_missing_locations(
            ast.Module(body=[future, node], type_ignores=[])
        )
        env = {"torch": torch}
        exec(compile(mod, str(path), "exec"), env)
        return env["_positions_from_start_pos"], path.read_text()

    def test_int_and_tensor_inputs_match_legacy_values(self):
        fn, _ = self._fn()
        legacy = 7 + torch.arange(5, dtype=torch.int64)
        self.assertTrue(torch.equal(fn(7, 5, torch.device("cpu")), legacy))
        with _BanHostSync():
            got = fn(torch.tensor([7], dtype=torch.int64), 5, torch.device("cpu"))
        self.assertTrue(torch.equal(got, legacy))
        self.assertEqual(got.dtype, torch.int64)

    def test_int32_tensor_promotes_like_legacy(self):
        fn, _ = self._fn()
        with _BanHostSync():
            got = fn(torch.tensor([7], dtype=torch.int32), 5, torch.device("cpu"))
        legacy = int(7) + torch.arange(5, dtype=torch.int64)
        self.assertTrue(torch.equal(got, legacy))
        self.assertEqual(got.dtype, torch.int64)

    def test_float_tensor_keeps_truncating_legacy_path(self):
        fn, _ = self._fn()
        got = fn(torch.tensor([7.9]), 3, torch.device("cpu"))
        legacy = int(7.9) + torch.arange(3, dtype=torch.int64)
        self.assertTrue(torch.equal(got, legacy))

    def test_forward_body_uses_helper(self):
        _, src = self._fn()
        self.assertIn("positions = _positions_from_start_pos(start_pos, S,", src)


def setUpModule():
    global _REAL_FLAG_FN, _strategy_env, plan, strategy_methods
    run_path(str(ROOT / "test" / "cpu_test_utils.py"))["isolate_cpu_test_module"]()

    plan = _load_plan()

    _REAL_FLAG_FN = extract(
        STRATEGY_PATH,
        {"_eager_count_gather_enabled"},
        {"os": os, "_EAGER_COUNT_GATHER_FLAG": _FLAG_NAME},
    )["_eager_count_gather_enabled"]

    _strategy_env = dict(
        torch=torch,
        os=os,
        Optional=None,  # annotations are strings via the future import
        current_scope=plan.current_scope,
        nullcontext=nullcontext,
        _count_fuse_enabled=lambda: True,
        _eager_count_gather_enabled=_REAL_FLAG_FN,
    )

    strategy_methods = extract(
        STRATEGY_PATH,
        {
            "synchronized_chunk_extent",
            "_counts_for_forward",
            "_take_pending_counts",
            "forward_subchunk_scope",
            "start_authoritative_counts_gather",
            "finish_authoritative_counts_gather",
        },
        _strategy_env,
        "NcclEpMxfp8Strategy",
    )

    FakeStrategy.synchronized_chunk_extent = strategy_methods[
        "synchronized_chunk_extent"
    ]

    FakeStrategy._counts_for_forward = strategy_methods["_counts_for_forward"]

    FakeStrategy._take_pending_counts = strategy_methods["_take_pending_counts"]

    FakeStrategy.forward_subchunk_scope = strategy_methods["forward_subchunk_scope"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
