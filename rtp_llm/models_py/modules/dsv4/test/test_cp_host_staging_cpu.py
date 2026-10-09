"""CPU tests for the pinned HtoD staging helper and the CP host-mirror scalars.

Loads the REAL ``cp.py`` via importlib with package stubs (no built rtp_llm
package needed).  CUDA is hidden: the staging CUDA branch is exercised through
the injectable ring (fake events/buffers stand in for pinned+event), while the
legacy path and the host-mirror derivations run for real on CPU tensors.
"""

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
CP_PATH = DSV4 / "cp.py"
CP_SRC = CP_PATH.read_text()
CP_NAME = "rtp_llm.models_py.modules.dsv4.cp"


def _package(name, path=None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path] if path else []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(_package(parent), child, mod)
    return sys.modules[name]


def _load_cp():
    if CP_NAME in sys.modules:
        return sys.modules[CP_NAME]
    _package("rtp_llm")
    _package("rtp_llm.models_py")
    _package("rtp_llm.models_py.modules")
    _package("rtp_llm.models_py.modules.dsv4", str(DSV4))
    dist = _package("rtp_llm.models_py.distributed")
    collective = types.ModuleType("rtp_llm.models_py.distributed.collective_torch")

    class _Group:
        TP = "TP"

    collective.Group = _Group
    collective.all_gather = lambda *a, **k: None
    collective._get_group = lambda g: g
    sys.modules[collective.__name__] = collective
    dist.collective_torch = collective

    spec = importlib.util.spec_from_file_location(CP_NAME, CP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[CP_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeEvent:
    def __init__(self):
        self.pending_query = False
        self.records = 0
        self.syncs = 0

    def record(self):
        self.records += 1
        self.pending_query = False

    def query(self):
        return self.pending_query

    def synchronize(self):
        self.syncs += 1
        self.pending_query = True


class _FakePinned:
    """Stand-in pinned buffer: records copies, produces CPU 'device' tensors."""

    def __init__(self, shape, dtype):
        self.buffer = torch.empty(shape, dtype=dtype)
        self.copies = 0

    def copy_(self, src):
        self.copies += 1
        self.buffer.copy_(src)
        return self.buffer

    def to(self, device=None, dtype=None, non_blocking=False):
        assert non_blocking, "staged transfer must be non_blocking"
        return self.buffer.to(device=device, dtype=dtype)


class PinnedRingLogic(unittest.TestCase):
    def test_ring_grows_rotates_and_fences(self):
        events = []

        def factory():
            ev = _FakeEvent()
            events.append(ev)
            return ev

        ring = cp._PinnedH2DRing(
            (3,),
            torch.int64,
            slots=3,
            event_factory=factory,
            buffer_factory=lambda: _FakePinned((3,), torch.int64),
        )

        e0 = ring.checkout()  # slot 0, never pending
        ring.commit(e0)
        e1 = ring.checkout()  # slot 0 pending+query False → grow slot 1
        ring.commit(e1)
        e2 = ring.checkout()  # grow slot 2
        ring.commit(e2)
        self.assertEqual(len(ring._entries), 3)
        self.assertIsNot(e0, e1)
        self.assertIsNot(e1, e2)

        # All three pending; query() False → grow impossible → sync oldest.
        e3 = ring.checkout()
        self.assertIs(e3, e0)
        self.assertEqual(events[0].syncs, 1)

        # After the fence, slot 0 is free; mark slot 1 completed via query.
        events[1].pending_query = True
        e4 = ring.checkout()
        self.assertIs(e4, e1)  # completed slot reused without a new alloc
        self.assertEqual(len(ring._entries), 3)

    def test_ring_rejects_zero_slots(self):
        with self.assertRaises(ValueError):
            cp._PinnedH2DRing((1,), torch.int64, slots=0, event_factory=_FakeEvent)


class StageHostToDeviceGates(unittest.TestCase):
    def test_flag_parse_fail_closed(self):
        with patch.dict(os.environ, {"DSV4_CP_PINNED_STAGE": "yes"}):
            with self.assertRaises(ValueError):
                cp._pinned_stage_enabled()
        with patch.dict(os.environ, {"DSV4_CP_PINNED_STAGE": "0"}):
            self.assertFalse(cp._pinned_stage_enabled())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_CP_PINNED_STAGE", None)
            self.assertTrue(cp._pinned_stage_enabled())  # default ON

    def test_cpu_device_uses_legacy_path(self):
        src = torch.arange(5, dtype=torch.int32)
        out = cp.stage_host_to_device(src, torch.device("cpu"), torch.int64)
        self.assertEqual(out.dtype, torch.int64)
        self.assertEqual(out.tolist(), [0, 1, 2, 3, 4])

    def test_flag_off_uses_legacy_path(self):
        src = torch.arange(5, dtype=torch.int32)
        with patch.dict(os.environ, {"DSV4_CP_PINNED_STAGE": "0"}):
            out = cp.stage_host_to_device(src, torch.device("cpu"), torch.int64)
        self.assertEqual(out.tolist(), [0, 1, 2, 3, 4])

    def test_cuda_branch_source_invariants(self):
        tree = ast.parse(CP_SRC)
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "stage_host_to_device"
        )
        src = ast.get_source_segment(CP_SRC, fn)
        self.assertIn("non_blocking=True", src)
        # No async read from caller-owned host memory: everything stages
        # through the never-freed ring buffers.
        self.assertNotIn("is_pinned()", src)
        self.assertIn("_PinnedH2DRing", src)
        self.assertIn("ring.checkout()", src)
        self.assertIn("ring.commit(entry)", src)


def _fake_cp_info(real_lengths, chunk_lengths, cp_size, *, full_mask=False):
    chunk_length = sum(chunk_lengths)
    padded = cp_size * chunk_length
    if full_mask:
        mask = torch.ones(padded, dtype=torch.int32)
    else:
        mask = torch.zeros(padded, dtype=torch.int32)
        mask[: int(sum(real_lengths))] = 1
    restore = torch.randperm(padded, dtype=torch.int64)
    return types.SimpleNamespace(
        prefill_qkv_padding_mask=mask,
        prefill_qkv_restore_indice=restore,
        prefill_actual_input_lengths_cpu=torch.tensor(real_lengths, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor(chunk_lengths, dtype=torch.int64),
    )


class HostMirrorDerivation(unittest.TestCase):
    """first_global_position must equal int(ctx.global_positions[0]) exactly."""

    def _check(self, cp_rank, prefix, real_lengths, chunk_lengths):
        cp_size = 4
        cp_info = _fake_cp_info(real_lengths, chunk_lengths, cp_size)
        device = torch.device("cpu")
        prefix_t = (
            torch.tensor([prefix] * len(real_lengths), dtype=torch.int64)
            if prefix
            else None
        )
        ctx = cp.build_cp_context_for_forward(
            cp_info,
            cp_size,
            cp_rank,
            sum(chunk_lengths),
            device,
            prefix_lengths=prefix_t,
        )
        self.assertIsNotNone(ctx.first_global_position)
        # The authoritative proof: the host mirror equals the device value.
        self.assertEqual(
            int(ctx.first_global_position), int(ctx.global_positions[0].item())
        )
        # And it matches the closed-form zigzag host formula.
        pair0 = chunk_lengths[0] // 2
        expect_local = min(cp_rank * pair0, max(int(real_lengths[0]) - 1, 0))
        self.assertEqual(int(ctx.first_global_position), prefix + expect_local)
        if prefix:
            self.assertEqual(
                tuple(ctx.prefix_lengths_full_host),
                tuple([prefix] * len(real_lengths)),
            )
        else:
            self.assertIsNone(ctx.prefix_lengths_full_host)
        return ctx

    def test_matrix_all_ranks_prefix_and_padding(self):
        for cp_rank in range(4):
            for prefix in (0, 100, 4096):
                self._check(cp_rank, prefix, [4096], [1024])  # full chunk
                self._check(cp_rank, prefix, [3000], [1024])  # padded chunk
                self._check(cp_rank, prefix, [97], [1024])  # clamp binds

    def test_two_request_chunks(self):
        for cp_rank in range(4):
            self._check(cp_rank, 0, [2000, 1500], [512, 512])
            self._check(cp_rank, 64, [2000, 1500], [512, 512])

    def test_first_position_for_meta_uses_host_value_without_item(self):
        cp_info = _fake_cp_info([4096], [1024], 4)
        ctx = cp.build_cp_context_for_forward(
            cp_info, 4, 2, 1024, torch.device("cpu"), prefix_lengths=None
        )
        with patch.object(
            torch.Tensor,
            "item",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("item banned")),
        ):
            got = cp.first_position_for_meta(ctx, ctx.global_positions)
        self.assertEqual(got, int(ctx.global_positions[0]))
        # Fallback: no CP context → legacy readback path still works.
        pos = torch.arange(10, 20, dtype=torch.int64)
        self.assertEqual(cp.first_position_for_meta(None, pos), 10)

    def test_first_position_none_when_no_rows(self):
        # Zero-length chunk: nothing derivable, callers must fall back.
        cp_info = _fake_cp_info([0], [0], 4, full_mask=True)
        cp_info.prefill_qkv_padding_mask = torch.zeros(0, dtype=torch.int32)
        cp_info.prefill_qkv_restore_indice = torch.zeros(0, dtype=torch.int64)
        cp_info.prefill_actual_input_lengths_cpu = None
        ctx = cp.build_cp_context(
            cp_info, 4, 0, 0, torch.device("cpu"), position_offset=0
        )
        self.assertIsNone(ctx.first_global_position)

    def test_first_position_empty_fallback_raises_descriptively(self):
        # Empty context with PP1 and fast generation: zero-token context
        # batch from the framework): the fallback must fail CLOSED with a
        # descriptive error, not an opaque IndexError on an empty tensor.
        cp_info = _fake_cp_info([0], [0], 4, full_mask=True)
        cp_info.prefill_qkv_padding_mask = torch.zeros(0, dtype=torch.int32)
        cp_info.prefill_qkv_restore_indice = torch.zeros(0, dtype=torch.int64)
        cp_info.prefill_actual_input_lengths_cpu = None
        ctx = cp.build_cp_context(
            cp_info, 4, 0, 0, torch.device("cpu"), position_offset=0
        )
        self.assertIsNone(ctx.first_global_position)
        empty = ctx.global_positions
        self.assertEqual(empty.numel(), 0)
        with self.assertRaises(RuntimeError) as ctx_mgr:
            cp.first_position_for_meta(ctx, empty)
        msg = str(ctx_mgr.exception)
        self.assertIn("empty positions", msg)
        self.assertIn("pp_size==1", msg)
        self.assertIn("enable_fast_gen", msg)
        # The pre-guard behavior was IndexError; prove the message differs so
        # the diagnosis is self-describing.
        self.assertNotIsInstance(ctx_mgr.exception, IndexError)

    def test_first_position_empty_no_cpctx_raises_descriptively(self):
        # No CP context at all (first is None) + empty positions: same guard.
        empty = torch.zeros(0, dtype=torch.int64)
        with self.assertRaises(RuntimeError) as ctx_mgr:
            cp.first_position_for_meta(None, empty)
        self.assertIn("empty positions", str(ctx_mgr.exception))

    def test_first_position_nonempty_fallback_byte_identical(self):
        # Non-empty fallback path must return exactly the legacy expression's
        # value for every shape/content (production PREFILL_CP parity).
        for data in ([0], [7], [10, 11, 12], [4095, 0, 3, 3, 3]):
            pos = torch.tensor(data, dtype=torch.int64)
            legacy = int(pos.reshape(-1)[0].item())
            self.assertEqual(cp.first_position_for_meta(None, pos), legacy)
        # 2-D input collapses like the legacy reshape(-1) did.
        pos2d = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)
        self.assertEqual(cp.first_position_for_meta(None, pos2d), 5)

    def test_first_position_mirror_wins_even_with_empty_positions(self):
        # Mirror precedence: when first_global_position is set the positions
        # tensor is never read (ban item()), even if it is (invalidly) empty.
        cp_info = _fake_cp_info([4096], [1024], 4)
        ctx = cp.build_cp_context_for_forward(
            cp_info, 4, 1, 1024, torch.device("cpu"), prefix_lengths=None
        )
        self.assertIsNotNone(ctx.first_global_position)
        empty = torch.zeros(0, dtype=torch.int64)
        with patch.object(
            torch.Tensor,
            "item",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("item banned")),
        ):
            self.assertEqual(
                cp.first_position_for_meta(ctx, empty),
                int(ctx.first_global_position),
            )

    def test_first_position_guard_source_pin(self):
        # Mutant pin: the guard text must exist in cp.py, positioned before the
        # legacy readback; deleting it must re-expose the IndexError behavior.
        src = CP_SRC
        fn = src.split("def first_position_for_meta", 1)[1]
        self.assertIn("flat.numel() == 0", fn)
        self.assertIn("raise RuntimeError", fn)
        self.assertLess(fn.index("flat.numel() == 0"), fn.index("flat[0].item()"))
        # Behavioral contrast: the legacy expression alone still IndexErrors.
        empty = torch.zeros(0, dtype=torch.int64)
        with self.assertRaises(IndexError):
            int(empty.reshape(-1)[0].item())


class VerifiedGeometryRouting(unittest.TestCase):
    def test_build_passes_host_sources_to_verified_geometry(self):
        tree = ast.parse(CP_SRC)
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "build_cp_context"
        )
        src = ast.get_source_segment(CP_SRC, fn)
        self.assertIn("padding_mask_host", src)
        self.assertIn("restore_indices_host", src)
        call = " ".join(src[src.index("verified_geometry(") :].split())
        self.assertIn("padding_mask_host if padding_mask_host is not None", call)
        self.assertIn("restore_indices_host if restore_indices_host is not None", call)

    def test_verified_geometry_runs_on_host_sources(self):
        # Real verified_geometry (loaded from the real module) must validate the
        # canonical 4096 geometry entirely from the CPU sources.
        fp8_runtime = "rtp_llm.models_py.modules.dsv4.fp8._compact_cp_runtime"
        _package("rtp_llm.models_py.modules.dsv4.fp8", str(DSV4 / "fp8"))
        spec = importlib.util.spec_from_file_location(
            fp8_runtime, DSV4 / "fp8/_compact_cp_runtime.py"
        )
        runtime = importlib.util.module_from_spec(spec)
        sys.modules[fp8_runtime] = runtime
        spec.loader.exec_module(runtime)

        cp_size, chunk_length = 4, 1024
        padded = cp_size * chunk_length
        order = torch.cat(
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
        cp_info = types.SimpleNamespace(
            prefill_qkv_padding_mask=torch.ones(padded, dtype=torch.int32),
            prefill_qkv_restore_indice=torch.argsort(order),
            prefill_actual_input_lengths_cpu=torch.tensor([4096], dtype=torch.int32),
            prefill_cp_chunk_lengths=torch.tensor([1024], dtype=torch.int64),
        )
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_COMPRESSOR": "1"}):
            ctx = cp.build_cp_context(
                cp_info, cp_size, 0, chunk_length, torch.device("cpu")
            )
        self.assertTrue(ctx.compact_geometry_verified)

    def test_verified_geometry_rejects_bad_mask_via_host_sources(self):
        cp_size, chunk_length = 4, 1024
        padded = cp_size * chunk_length
        order = torch.cat(
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
        mask = torch.ones(padded, dtype=torch.int32)
        mask[7] = 0  # invalid: a padded row inside the chunk
        cp_info = types.SimpleNamespace(
            prefill_qkv_padding_mask=mask,
            prefill_qkv_restore_indice=torch.argsort(order),
            prefill_actual_input_lengths_cpu=torch.tensor([4096], dtype=torch.int32),
            prefill_cp_chunk_lengths=torch.tensor([1024], dtype=torch.int64),
        )
        with patch.dict(os.environ, {"DSV4_CP_COMPACT_COMPRESSOR": "1"}):
            ctx = cp.build_cp_context(
                cp_info, cp_size, 0, chunk_length, torch.device("cpu")
            )
        self.assertFalse(ctx.compact_geometry_verified)


def setUpModule():
    global cp
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()

    cp = _load_cp()


if __name__ == "__main__":
    unittest.main(verbosity=2)
