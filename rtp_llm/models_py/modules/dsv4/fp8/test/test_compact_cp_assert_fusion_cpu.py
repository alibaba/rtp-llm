"""CPU tests for the compact-CP guard-preserving assert fusion.

Drives the REAL ``_compact_cp_runtime`` (loaded by file location with leaf
doubles): the fused production assert and the unfused per-check battery share
the same ``_plan_metadata_checks`` condition list, so both are driven with
identical violating fixtures here — the fusion cannot drift from the guards.

Also proves the production pack/scatter call sites already run with
``check=False`` (the ``_cp_packed_rows`` checked path keeps its single host
sync for non-plan callers) and that ``replicate`` issues exactly one async
device assert per layer with no host-synchronizing call.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types
import unittest
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


class _BanHostSync:
    def __enter__(self):
        def banned(*a, **k):
            raise AssertionError("host synchronizing call on the hot path")

        self._patchers = [
            patch.object(torch.Tensor, "item", banned),
            patch.object(torch.Tensor, "cpu", banned),
            patch.object(torch.Tensor, "tolist", banned),
        ]
        for p in self._patchers:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self._patchers:
            p.stop()
        return False


def _counting_asserts():
    """Wrap torch._assert_async: record messages, still execute the real op."""
    real = torch._assert_async
    calls = []

    def counting(condition, message):
        calls.append(message)
        return real(condition, message)

    return calls, patch.object(torch, "_assert_async", counting)


POOL_CAP = 4096


def _valid_meta():
    kv_slots = torch.arange(4096, dtype=torch.int64)
    return types.SimpleNamespace(
        positions=torch.arange(4096, dtype=torch.long),
        b_idx=torch.zeros(4096, dtype=torch.long),
        token_to_req=torch.zeros(4096, dtype=torch.long),
        state_slots=torch.full((4096,), -1, dtype=torch.long),
        kv_slots=kv_slots,
        seq_start_per_req=None,
        cu_seq_per_req=None,
        is_batched=False,
    )


def _battery(meta, pool_cap=POOL_CAP):
    expected_positions = torch.arange(4096, dtype=torch.long) + G.start
    covered_state = torch.zeros(4096, dtype=torch.bool)
    receiver = meta.kv_slots.index_select(0, BOUNDARIES)
    ordered_slots = receiver.sort().values
    return runtime._plan_metadata_checks(
        meta, expected_positions, covered_state, ordered_slots, receiver, pool_cap, G
    )


def _violating_fixtures():
    """One fixture per predicate; each flips exactly its own condition."""
    fixtures = []

    def add(name, mutate, message, batched=False):
        meta = _valid_meta()
        if batched:
            meta.is_batched = True
            meta.seq_start_per_req = torch.tensor([G.start], dtype=torch.long)
            meta.cu_seq_per_req = torch.tensor([0, 4096], dtype=torch.long)
        mutate(meta)
        fixtures.append((name, meta, message))

    def swap_positions(meta):
        meta.positions = meta.positions.clone()
        meta.positions[10], meta.positions[11] = (
            meta.positions[11].item(),
            meta.positions[10].item(),
        )

    add("positions", swap_positions, "compact CP nonsequential full metadata")
    add(
        "b_idx",
        lambda m: m.b_idx.__setitem__(3, 1),
        "compact CP multiple requests",
    )
    add(
        "token_to_req",
        lambda m: m.token_to_req.__setitem__(5, 2),
        "compact CP foreign token request mapping",
    )

    def bad_state(meta):
        meta.state_slots = meta.state_slots.clone()
        meta.state_slots[100] = 7  # nonnegative but not covered

    add("state_slots", bad_state, "compact CP missing state raw rows")

    def dup_receiver(meta):
        meta.kv_slots = meta.kv_slots.clone()
        meta.kv_slots[BOUNDARIES[6]] = meta.kv_slots[BOUNDARIES[5]]

    add("duplicate", dup_receiver, "compact CP duplicate destinations")

    def big_receiver(meta):
        meta.kv_slots = meta.kv_slots.clone()
        meta.kv_slots[BOUNDARIES[7]] = POOL_CAP  # == cap: out of range

    add("range_high", big_receiver, "compact CP invalid destination")

    def neg_receiver(meta):
        meta.kv_slots = meta.kv_slots.clone()
        meta.kv_slots[BOUNDARIES[7]] = -1

    add("range_neg", neg_receiver, "compact CP invalid destination")

    add(
        "prefix",
        lambda m: m.seq_start_per_req.fill_(G.start + 1),
        "compact CP foreign prefix",
        batched=True,
    )
    add(
        "window_values",
        lambda m: m.cu_seq_per_req.fill_(7),
        "compact CP bad raw windows",
        batched=True,
    )
    return fixtures


class FusedBatteryEquivalence(unittest.TestCase):
    def test_valid_meta_fused_passes_with_one_assert(self):
        checks = _battery(_valid_meta())
        self.assertEqual(len(checks), 6)  # 5 metadata + receiver range
        calls, counter = _counting_asserts()
        with counter:
            runtime._assert_fused(checks)
        self.assertEqual(len(calls), 1)
        for _, message in checks:
            self.assertIn(message, calls[0])

    def test_valid_batched_meta_battery_shape(self):
        meta = _valid_meta()
        meta.is_batched = True
        meta.seq_start_per_req = torch.tensor([G.start], dtype=torch.long)
        meta.cu_seq_per_req = torch.tensor([0, 4096], dtype=torch.long)
        checks = _battery(meta)
        self.assertEqual(len(checks), 8)
        calls, counter = _counting_asserts()
        with counter:
            runtime._assert_fused(checks)
        # The batched battery's joined message exceeds the 255-char CUDA
        # device-assert limit, so the assert splits into the fewest chunks that
        # fit — never a >250-char message, every predicate still asserted
        # verbatim, and the failing predicate's own message is what fires.
        joined = "; ".join(m for _, m in checks)
        if len(joined) <= 250:
            self.assertEqual(len(calls), 1)
        else:
            self.assertGreaterEqual(len(calls), 2)
            for c in calls:
                self.assertLessEqual(len(c), 250)
            for _, message in checks:
                self.assertTrue(any(message in c for c in calls), message)

    def test_every_violation_raises_fused_and_unfused_with_same_message(self):
        for name, meta, message in _violating_fixtures():
            checks = _battery(meta)
            failing = [m for c, m in checks if not bool(c)]
            self.assertEqual(
                failing, [message], f"{name}: expected exactly one failing predicate"
            )
            # Unfused battery: the first failing check raises with exactly its
            # own message (the pre-fusion behavior).
            # (driven from the same condition list — no drift possible)
            with self.assertRaises(RuntimeError) as ctx_unfused:
                for condition, msg in checks:
                    runtime._assert_device(condition, msg)
            self.assertEqual(str(ctx_unfused.exception), message)
            # Fused production assert: one raise, message content preserved.
            with self.assertRaises(RuntimeError) as ctx_fused:
                runtime._assert_fused(checks)
            self.assertIn(message, str(ctx_fused.exception))

    def test_fused_message_preserves_all_predicate_text(self):
        checks = _battery(_valid_meta())
        calls, counter = _counting_asserts()
        with counter:
            runtime._assert_fused(checks)
        self.assertEqual(len(calls), 1)
        for _, message in checks:
            self.assertIn(message, calls[0])

    def test_malformed_window_length_fails_closed(self):
        meta = _valid_meta()
        meta.is_batched = True
        meta.seq_start_per_req = torch.tensor([G.start], dtype=torch.long)
        meta.cu_seq_per_req = torch.tensor([0, 4096, 8192], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "compact CP bad raw windows"):
            _battery(meta)

    def test_missing_batched_windows_fail_closed(self):
        meta = _valid_meta()
        meta.is_batched = True
        with self.assertRaisesRegex(ValueError, "compact CP missing B1 raw windows"):
            _battery(meta)

    def test_no_pool_cap_skips_destination_check(self):
        checks = _battery(_valid_meta(), pool_cap=None)
        self.assertNotIn("compact CP invalid destination", [m for _, m in checks])
        self.assertEqual(len(checks), 5)

    def test_fused_reduces_assert_calls(self):
        checks = _battery(_valid_meta())
        calls, counter = _counting_asserts()
        with counter:
            runtime._assert_fused(checks)
        fused_calls = len(calls)
        calls.clear()
        with counter:
            for condition, msg in checks:
                runtime._assert_device(condition, msg)
        self.assertEqual(fused_calls, 1)
        self.assertEqual(len(calls), len(checks))


class PackedRowsCheckedPathSyncs(unittest.TestCase):
    """The _cp_packed_rows checked path keeps ONE host sync by contract; the
    production compact-CP caller passes check=False and must never hit it."""

    def _pool(self):
        layout = packed.MAIN_KV_LAYOUT  # data 576, scale 8
        data = torch.zeros(4, 8, layout.data_bytes, dtype=torch.uint8)
        scales = torch.zeros(4, 8, layout.scale_bytes, dtype=torch.uint8)
        return data, scales, layout

    def test_scatter_checked_path_syncs_and_unchecked_does_not(self):
        data, scales, layout = self._pool()
        data.view(-1)[:] = torch.arange(data.numel(), dtype=torch.int64).to(torch.uint8)
        slots = torch.tensor([0, 3, 5], dtype=torch.int64)
        packed_rows = packed.pack_rows(
            data,
            scales,
            slots,
            layout,
            num_blocks=4,
            entries_per_block=8,
            check=False,
        )
        with _BanHostSync():
            # Checked mode: the single .item() sync — the ban must catch it.
            with self.assertRaises(AssertionError):
                packed.scatter_packed_rows(
                    data,
                    scales,
                    packed_rows,
                    layout,
                    num_blocks=4,
                    entries_per_block=8,
                    check=True,
                )
            with self.assertRaises(AssertionError):
                packed.pack_rows(
                    data,
                    scales,
                    slots,
                    layout,
                    num_blocks=4,
                    entries_per_block=8,
                    check=True,
                )
            # Unchecked (production) mode: completes without any host sync.
            dst_data, dst_scales, _ = self._pool()
            packed.scatter_packed_rows(
                dst_data,
                dst_scales,
                packed_rows,
                layout,
                num_blocks=4,
                entries_per_block=8,
                check=False,
            )
        for slot in (0, 3, 5):
            self.assertTrue(
                torch.equal(
                    dst_data.view(-1, layout.data_bytes)[slot],
                    data.view(-1, layout.data_bytes)[slot],
                )
            )


def _make_handle(pool, kv_slots, start=0):
    """A staged CompactPending without __init__ (CUDA stream plumbing)."""
    g = runtime.Geometry(0, start, 4, 8, 4, 512)
    boundaries = torch.tensor(g.wire_boundaries(), dtype=torch.long)
    per = 1024 // g.ratio
    local_boundaries = boundaries[0:per]
    meta = _valid_meta()
    meta.kv_slots = kv_slots
    h = runtime.CompactPending.__new__(runtime.CompactPending)
    h.geometry = g
    h.meta = meta
    h.workspace = types.SimpleNamespace()
    h.local = torch.zeros(4, 4)
    h.group = object()
    h.role = "main"
    h.state = "staged"
    h.binding = None
    h.meta_identity = runtime._meta_identity(meta)
    h.owner = (id(h.workspace), h.local.device, id(meta), id(h.group))
    h.tail_indices = None
    h.tail_dest = None
    h.boundaries = boundaries
    h.local_boundaries = local_boundaries
    h.masked_meta = None
    h.receiver_slots = kv_slots.index_select(0, boundaries)
    h.boundary_kv_slots = kv_slots.index_select(0, local_boundaries)
    h.expected_wire_ids = boundaries + g.start
    h.local_boundary_ids = local_boundaries + g.start
    return h, meta


class ReplicateFusionAndCheckOff(unittest.TestCase):
    def _module(self, pool):
        return types.SimpleNamespace(_kv_pool_view=pool, head_dim=128)

    def test_replicate_one_assert_check_false_and_byte_exact(self):
        # Pool: 8 blocks x 512 entries x (128 data + 4 scale) — cap 4096.
        pool = torch.zeros(8, 512, 132, dtype=torch.uint8)
        kv_slots = torch.arange(4096, dtype=torch.int64)
        handle, meta = _make_handle(pool, kv_slots)
        module = self._module(pool)

        calls = []
        real_pack = runtime.pack_rows
        real_scatter = runtime.scatter_packed_rows

        def recording_pack(*a, **k):
            calls.append(("pack", k.get("check", "MISSING")))
            return real_pack(*a, **k)

        def recording_scatter(*a, **k):
            calls.append(("scatter", k.get("check", "MISSING")))
            return real_scatter(*a, **k)

        boundaries = handle.boundaries
        per = 1024 // handle.geometry.ratio

        def fake_all_gather(out, inp, group=None):
            n = inp.shape[0]
            for r in range(4):
                rows = inp.clone()
                ids = boundaries[r * n : (r + 1) * n] + handle.geometry.start
                rows[:, :8] = ids.contiguous().view(torch.uint8).reshape(n, 8)
                out[r * n : (r + 1) * n] = rows

        # Distinctive pool content so the scatter destinations are verifiable.
        data_view = pool.view(8, 512, 132)
        pattern = torch.arange(8 * 512 * 132, dtype=torch.int32).to(torch.uint8)
        data_view.copy_(pattern.reshape(8, 512, 132))
        before = pool.clone()

        assert_calls, counter = _counting_asserts()
        with (
            counter,
            _BanHostSync(),
            patch.object(runtime, "pack_rows", recording_pack),
            patch.object(runtime, "scatter_packed_rows", recording_scatter),
            patch.object(torch.distributed, "all_gather_into_tensor", fake_all_gather),
        ):
            handle.replicate(module)

        self.assertEqual(handle.state, "finished")
        self.assertEqual(
            calls, [("pack", False), ("scatter", False)]
        )  # both production sites unchecked
        self.assertEqual(len(assert_calls), 1)  # one fused wire-id assert
        self.assertEqual(assert_calls[0], "compact CP foreign logical rows")

        # Byte exactness: wire row i carried the pack bytes of local boundary
        # (i % per) and landed at pool slot receiver[i] == boundaries[i].
        layout = packed.MAIN_KV_LAYOUT
        db = 128
        slots = kv_slots.index_select(0, handle.local_boundaries)
        for i in (0, 1, 255, 256, 511, 1023):
            dst = int(handle.receiver_slots[i])
            src = int(slots[i % per])
            self.assertTrue(
                torch.equal(
                    pool[dst // 512, dst % 512, :db], before[src // 512, src % 512, :db]
                )
            )

    def test_replicate_rejects_foreign_wire_with_one_assert(self):
        pool = torch.zeros(8, 512, 132, dtype=torch.uint8)
        kv_slots = torch.arange(4096, dtype=torch.int64)
        handle, meta = _make_handle(pool, kv_slots)
        module = self._module(pool)

        def corrupt_all_gather(out, inp, group=None):
            out[:] = inp.repeat(4, 1)  # every rank's header = local ids: foreign
            out[0, :8] = 255  # definitely not the expected wire id

        with patch.object(
            torch.distributed, "all_gather_into_tensor", corrupt_all_gather
        ):
            with self.assertRaisesRegex(RuntimeError, "foreign logical rows"):
                handle.replicate(module)


class ReplicateSourceInvariants(unittest.TestCase):
    def _method_src(self, name, cls="CompactPending"):
        tree = ast.parse(RUNTIME_SRC)
        klass = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
        node = next(
            n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        return ast.get_source_segment(RUNTIME_SRC, node)

    def test_production_pack_scatter_sites_unchecked(self):
        src = self._method_src("replicate")
        self.assertEqual(src.count("check=False"), 2)
        # The per-layer receiver-range assert moved into the per-forward battery.
        self.assertNotIn("receiver >= 0", src)
        self.assertNotIn("self.boundaries + self.geometry.start", src)
        self.assertNotIn("index_select", src)

    def test_no_pageable_window_constant_left(self):
        self.assertNotIn("torch.tensor([0, 4096], device", RUNTIME_SRC)

    def test_init_cache_key_includes_pool_cap(self):
        src = self._method_src("__init__")
        self.assertIn("pool_cap", src)
        # the cache-miss body lives in ``_build_chunk_plan`` (called from
        # the miss branch); the fused battery still fires before any write.
        self.assertIn("_build_chunk_plan(g, device, meta, pool_cap)", src)
        tree = ast.parse(RUNTIME_SRC)
        plan_fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_build_chunk_plan"
        )
        plan_src = ast.get_source_segment(RUNTIME_SRC, plan_fn)
        self.assertIn("_assert_fused(", plan_src)
        self.assertNotIn("torch._assert_async(\n", plan_src)  # battery is fused

    def test_battery_lists_all_original_messages(self):
        src = RUNTIME_SRC
        for message in (
            "compact CP nonsequential full metadata",
            "compact CP multiple requests",
            "compact CP foreign token request mapping",
            "compact CP missing state raw rows",
            "compact CP duplicate destinations",
            "compact CP invalid destination",
            "compact CP foreign prefix",
            "compact CP bad raw windows",
            "compact CP foreign logical rows",
        ):
            self.assertIn(message, src)


def setUpModule():
    global BOUNDARIES, G, packed, runtime
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

    G = runtime.Geometry(0, 0, 4, 8, 4, 512)

    BOUNDARIES = torch.tensor(G.wire_boundaries(), dtype=torch.long)  # [1024]


if __name__ == "__main__":
    unittest.main(verbosity=2)
