"""CPU tests for cached compact-CP geometry plans.

Compare cached and uncached metadata, verify device-upload reuse and kill-switch
behavior, and reject incompatible rank or geometry reuse."""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from dataclasses import dataclass
from runpy import run_path
from unittest.mock import patch

import torch

HERE = pathlib.Path(__file__).resolve().parent
FP8 = HERE.parent
RUNTIME_PATH = FP8 / "_compact_cp_runtime.py"
RUNTIME_SRC = RUNTIME_PATH.read_text()
PACKED_PATH = FP8 / "_cp_packed_rows.py"
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


# Production geometries (measured model ABI): (state_entries, state_block,
# ratio, width) per compact role.
PRODUCTION_GEOMETRIES = {
    "csa_main": (8, 256, 4, 2048),
    "csa_indexer": (8, 256, 4, 512),
    "hca_main": (128, 256, 128, 2048),
}
STARTS = (0, 4096, 2 * 4096, 5 * 4096, 7 * 4096)


def _legacy_geometry_plan(g, device):
    """The uncached per-forward construction, verbatim, as the oracle."""
    tail_index_vals = []
    tail_dest_vals = []
    local_map = tuple(p for lo, hi in g.intervals() for p in range(lo, hi))
    lut = {p: i for i, p in enumerate(local_map)}
    tail_index_vals = [lut[p] for p in g.tails()]
    tail_dest_vals = [p for r in range(4) for p in g.tails(r)]
    boundary_vals = list(g.wire_boundaries())
    flat = torch.tensor(
        tail_index_vals + tail_dest_vals + boundary_vals,
        dtype=torch.long,
        device=device,
    )
    n_tail = len(tail_index_vals)
    n_dest = len(tail_dest_vals)
    tail_indices = flat[:n_tail]
    tail_dest = flat[n_tail : n_tail + n_dest]
    boundaries = flat[n_tail + n_dest :]
    local_boundaries = boundaries[
        g.rank * (1024 // g.ratio) : (g.rank + 1) * (1024 // g.ratio)
    ]
    covered_state = torch.zeros(4096, dtype=torch.bool, device=device)
    covered_state[tail_dest] = True
    return tail_indices, tail_dest, boundaries, local_boundaries, covered_state


@dataclass
class _Meta:
    """Dataclass stand-in for CompressorMeta (``replace`` requires one)."""

    positions: torch.Tensor
    b_idx: torch.Tensor
    token_to_req: torch.Tensor
    state_slots: torch.Tensor
    kv_slots: torch.Tensor
    seq_start_per_req: object = None
    cu_seq_per_req: object = None
    is_batched: bool = False


def _meta(kv_slots=None, start=0):
    kv_slots = torch.arange(4096, dtype=torch.int64) if kv_slots is None else kv_slots
    return _Meta(
        positions=torch.arange(4096, dtype=torch.long) + start,
        b_idx=torch.zeros(4096, dtype=torch.long),
        token_to_req=torch.zeros(4096, dtype=torch.long),
        state_slots=torch.full((4096,), -1, dtype=torch.long),
        kv_slots=kv_slots,
    )


def _legacy_chunk_plan(g, device, meta):
    """The uncached per-forward plan body (start-dependent parts), verbatim."""
    (
        tail_indices,
        tail_dest,
        boundaries,
        local_boundaries,
        covered_state,
    ) = _legacy_geometry_plan(g, device)
    masked = types.SimpleNamespace(
        positions=meta.positions.index_select(0, local_boundaries),
        b_idx=meta.b_idx.index_select(0, local_boundaries),
        state_slots=meta.state_slots.index_select(0, local_boundaries),
        kv_slots=meta.kv_slots.index_select(0, local_boundaries),
        token_to_req=meta.token_to_req.index_select(0, local_boundaries),
    )
    expected_positions = torch.arange(4096, device=device, dtype=torch.long) + g.start
    receiver = meta.kv_slots.index_select(0, boundaries)
    ordered_slots = receiver.sort().values
    boundary_kv_slots = masked.kv_slots
    expected_wire_ids = boundaries + g.start
    local_boundary_ids = local_boundaries + g.start
    checks = runtime._plan_metadata_checks(
        meta, expected_positions, covered_state, ordered_slots, receiver, 4096, g
    )
    for condition, message in checks:
        runtime._assert_device(condition, message)
    return (
        tail_indices,
        tail_dest,
        boundaries,
        local_boundaries,
        masked,
        receiver,
        boundary_kv_slots,
        expected_wire_ids,
        local_boundary_ids,
    )


def _assert_plan_equal(testcase, got, want):
    (
        tail_indices,
        tail_dest,
        boundaries,
        local_boundaries,
        masked,
        receiver,
        boundary_kv_slots,
        expected_wire_ids,
        local_boundary_ids,
    ) = got
    (
        w_tail_indices,
        w_tail_dest,
        w_boundaries,
        w_local_boundaries,
        w_masked,
        w_receiver,
        w_boundary_kv_slots,
        w_expected_wire_ids,
        w_local_boundary_ids,
    ) = want
    for name, a, b in (
        ("tail_indices", tail_indices, w_tail_indices),
        ("tail_dest", tail_dest, w_tail_dest),
        ("boundaries", boundaries, w_boundaries),
        ("local_boundaries", local_boundaries, w_local_boundaries),
        ("receiver", receiver, w_receiver),
        ("boundary_kv_slots", boundary_kv_slots, w_boundary_kv_slots),
        ("expected_wire_ids", expected_wire_ids, w_expected_wire_ids),
        ("local_boundary_ids", local_boundary_ids, w_local_boundary_ids),
    ):
        testcase.assertEqual(a.dtype, b.dtype, name)
        testcase.assertTrue(torch.equal(a, b), name)
    for name in ("positions", "b_idx", "state_slots", "kv_slots", "token_to_req"):
        testcase.assertTrue(
            torch.equal(getattr(masked, name), getattr(w_masked, name)),
            f"masked.{name}",
        )


class GeometryPlanCache(unittest.TestCase):
    def setUp(self):
        runtime._GEOMETRY_PLAN_CACHE.clear()

    def test_flag_parse_fail_closed(self):
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_PLAN_CACHE": "yes"}):
            with self.assertRaises(ValueError):
                runtime._plan_cache_enabled()
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_PLAN_CACHE": "0"}):
            self.assertFalse(runtime._plan_cache_enabled())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_CP_COMPACT_PLAN_CACHE", None)
            self.assertTrue(runtime._plan_cache_enabled())  # default ON

    def test_byte_identity_across_production_matrix(self):
        device = torch.device("cpu")
        for name, (se, sb, ratio, width) in PRODUCTION_GEOMETRIES.items():
            for rank in range(4):
                for start in STARTS:
                    g = runtime.Geometry(rank, start, se, sb, ratio, width)
                    meta = _meta(start=start)
                    got = runtime._build_chunk_plan(g, device, meta, 4096)
                    want = _legacy_chunk_plan(g, device, meta)
                    with self.subTest(geometry=name, rank=rank, start=start):
                        _assert_plan_equal(self, got, want)

    def test_byte_identity_with_alloc_like_kv_slots(self):
        # kv_slots are the allocation-dependent part of the plan; sweep a
        # shuffled block-table-like layout (still unique in-range slots).
        device = torch.device("cpu")
        rng = torch.Generator().manual_seed(1234)
        kv_slots = torch.randperm(4096, generator=rng).to(torch.int64)
        for name, (se, sb, ratio, width) in PRODUCTION_GEOMETRIES.items():
            g = runtime.Geometry(2, 3 * 4096, se, sb, ratio, width)
            meta = _meta(kv_slots=kv_slots, start=3 * 4096)
            with self.subTest(geometry=name):
                _assert_plan_equal(
                    self,
                    runtime._build_chunk_plan(g, device, meta, 4096),
                    _legacy_chunk_plan(g, device, meta),
                )

    def test_warm_cache_issues_no_tensor_upload(self):
        device = torch.device("cpu")
        g = runtime.Geometry(0, 0, 8, 256, 4, 2048)
        runtime._build_chunk_plan(g, device, _meta(), 4096)  # populate
        calls = []
        real_tensor = torch.tensor

        def counting(*args, **kwargs):
            calls.append(1)
            return real_tensor(*args, **kwargs)

        with patch.object(torch, "tensor", counting):
            runtime._build_chunk_plan(g, device, _meta(), 4096)
        self.assertEqual(len(calls), 0, "warm-cache plan must not upload")
        # Kill switch: per-call rebuild (the uncached behavior) uploads again.
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_PLAN_CACHE": "0"}):
            with patch.object(torch, "tensor", counting):
                runtime._build_chunk_plan(g, device, _meta(), 4096)
        self.assertEqual(len(calls), 1, "kill switch must rebuild per call")

    def test_cache_identity_and_kill_switch_freshness(self):
        device = torch.device("cpu")
        g = runtime.Geometry(1, 4096, 8, 256, 4, 2048)
        first = runtime._geometry_plan(g, device)
        second = runtime._geometry_plan(g, device)
        for a, b in zip(first, second):
            self.assertIs(a, b, "warm cache must return the same tensor objects")
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_PLAN_CACHE": "0"}):
            fresh = runtime._geometry_plan(g, device)
        for a, b in zip(first, fresh):
            self.assertIsNot(a, b, "kill switch must build fresh tensors")
            self.assertTrue(torch.equal(a, b))

    def test_cached_constants_are_not_mutated_by_consumers(self):
        device = torch.device("cpu")
        g = runtime.Geometry(0, 0, 8, 256, 4, 2048)
        runtime._geometry_plan(g, device)  # populate
        snapshot = [t.clone() for t in runtime._geometry_plan(g, device)]
        # Drive every consumer op of the cached constants (all read-only):
        meta = _meta()
        plan = runtime._build_chunk_plan(g, device, meta, 4096)
        (
            tail_indices,
            tail_dest,
            boundaries,
            local_boundaries,
            _masked,
            _receiver,
            _bkv,
            _ewire,
            _lbid,
        ) = plan
        local = torch.arange(4096, dtype=torch.float32)
        send = local.index_select(0, tail_indices)
        scratch = torch.zeros(4096, dtype=torch.float32)
        scratch.index_copy_(0, tail_dest, send.new_full((tail_dest.numel(),), 1.0))
        _ = boundaries + g.start
        _ = local_boundaries + g.start
        _ = meta.kv_slots.index_select(0, boundaries)
        _ = (meta.state_slots < 0) | runtime._geometry_plan(g, device)[4]
        for cached, before in zip(runtime._geometry_plan(g, device), snapshot):
            self.assertTrue(
                torch.equal(cached, before),
                "consumer mutated a cached plan constant",
            )

    def test_cache_key_separates_rank_and_geometry(self):
        device = torch.device("cpu")
        base = runtime._geometry_plan(runtime.Geometry(0, 0, 8, 256, 4, 2048), device)
        other_rank = runtime._geometry_plan(
            runtime.Geometry(1, 0, 8, 256, 4, 2048), device
        )
        other_geom = runtime._geometry_plan(
            runtime.Geometry(0, 0, 128, 256, 128, 2048), device
        )
        self.assertFalse(torch.equal(base[3], other_rank[3]))  # local_boundaries
        self.assertFalse(torch.equal(base[0], other_geom[0]))  # tail_indices
        self.assertNotEqual(base[2].numel(), other_geom[2].numel())  # boundaries

    def test_cross_rank_mixup_mutant_is_rejected_by_battery(self):
        # A covered_state that fails to cover a real tail row must trip the
        # missing-state-rows guard — this is what keeps a stale/poisoned cache
        # entry from silently scattering raw state to uncovered slots.
        g = runtime.Geometry(0, 0, 8, 256, 4, 2048)
        covered = runtime._geometry_plan(g, torch.device("cpu"))[4]
        meta = _meta()
        # One non-negative state slot inside a covered tail: passes only if
        # covered_state really marks the tail.
        meta.state_slots[g.tails()[0]] = 5
        receiver = meta.kv_slots.index_select(
            0, runtime._geometry_plan(g, torch.device("cpu"))[2]
        )
        ordered = receiver.sort().values
        checks = runtime._plan_metadata_checks(
            meta,
            torch.arange(4096, dtype=torch.long),
            covered,
            ordered,
            receiver,
            4096,
            g,
        )
        self.assertTrue(all(bool(c) for c, _ in checks))
        # Mutant: drop the tail scatter from covered_state → battery rejects.
        uncovered = torch.zeros(4096, dtype=torch.bool)
        checks_bad = runtime._plan_metadata_checks(
            meta,
            torch.arange(4096, dtype=torch.long),
            uncovered,
            ordered,
            receiver,
            4096,
            g,
        )
        failing = [m for c, m in checks_bad if not bool(c)]
        self.assertEqual(failing, ["compact CP missing state raw rows"])

    def test_poisoned_cache_changes_output(self):
        # Anti-vacuity for the cache being load-bearing: a poisoned entry MUST
        # change the derived plan (here: the wire-id order), i.e. the
        # byte-identity proofs above are not vacuously satisfied by accident.
        device = torch.device("cpu")
        g = runtime.Geometry(0, 0, 8, 256, 4, 2048)
        good = runtime._build_chunk_plan(g, device, _meta(), 4096)
        key = (g.rank, g.state_entries, g.state_block, g.ratio, g.width, str(device))
        poisoned = list(runtime._GEOMETRY_PLAN_CACHE[key])
        swapped = poisoned[2].clone()
        swapped[0], swapped[1] = swapped[1].item(), swapped[0].item()
        poisoned[2] = swapped
        runtime._GEOMETRY_PLAN_CACHE[key] = tuple(poisoned)
        bad = runtime._build_chunk_plan(g, device, _meta(), 4096)
        self.assertFalse(torch.equal(good[7], bad[7]))  # expected_wire_ids

    def test_call_sites_use_the_cache(self):
        tree = ast.parse(RUNTIME_SRC)
        init = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "CompactPending"
        ).body
        init_fn = next(
            n for n in init if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        src = ast.get_source_segment(RUNTIME_SRC, init_fn)
        self.assertIn("indices = _build_chunk_plan(g, device, meta, pool_cap)", src)
        self.assertNotIn("torch.tensor(", src)
        plan_src = ast.get_source_segment(
            RUNTIME_SRC,
            next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_build_chunk_plan"
            ),
        )
        self.assertNotIn("torch.tensor(", plan_src)
        self.assertIn("_geometry_plan(g, device)", plan_src)


def setUpModule():
    global packed, runtime
    run_path(str(HERE.parents[1] / "test" / "cpu_test_utils.py"))[
        "isolate_cpu_test_module"
    ]()

    _package("rtp_llm")

    _package("rtp_llm.models_py")

    _package("rtp_llm.models_py.modules")

    _package("rtp_llm.models_py.modules.dsv4")

    _package(PREFIX, str(FP8))

    packed = _load(PREFIX + "._cp_packed_rows", PACKED_PATH)

    runtime = _load(PREFIX + "._compact_cp_runtime", RUNTIME_PATH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
