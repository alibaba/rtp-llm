"""CPU byte-transport and caller-ordering tests with CUDA/collective leaf doubles.

These do not qualify numerical kernels, GPU binding or hardware stream lifetimes."""

import ast
import contextlib
import copy
import dataclasses
import gc
import hashlib
import importlib.util
import json
import os
import sys
import threading
import types
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import torch

FP8 = Path(__file__).resolve().parents[1]
DSV4 = FP8.parent
PREFIX = "rtp_llm.models_py.modules.dsv4.fp8"
METHODS = (
    "start_prefill",
    "wait_prefill_gather",
    "finish_prefill",
    "forward",
    "_cp_profile_name",
    "prepare_metadata",
    "_compute_state_slot_mapping",
    "_compute_kv_slot_mapping",
    "_launch",
)
SOURCE_PATHS = [
    FP8 / "_compact_cp_runtime.py",
    FP8 / "compressor.py",
    FP8 / "_cp_packed_rows.py",
    DSV4 / "cp.py",
    DSV4 / "prefill_workspace.py",
]
SOURCE_HASHES = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_PATHS
}


def package(name):
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
        if "." in name:
            par, child = name.rsplit(".", 1)
            setattr(package(par), child, m)
    return sys.modules[name]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    if "." in name:
        par, child = name.rsplit(".", 1)
        setattr(package(par), child, m)
    return m


package(PREFIX)
packed = load_file(PREFIX + "._cp_packed_rows", FP8 / "_cp_packed_rows.py")
runtime = load_file(PREFIX + "._compact_cp_runtime", FP8 / "_compact_cp_runtime.py")
workspace_module = load_file(
    "_actual_compact_prefill_workspace", DSV4 / "prefill_workspace.py"
)
collective = types.ModuleType("rtp_llm.models_py.distributed.collective_torch")
collective.Group = types.SimpleNamespace(TP="TP")
package("rtp_llm.models_py.distributed").collective_torch = collective
sys.modules[collective.__name__] = collective


def actual_callers():
    tree = ast.parse((FP8 / "compressor.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "CompressorFP8"
    )
    methods = [
        copy.deepcopy(n)
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in METHODS
    ]
    assert {n.name for n in methods} == set(METHODS)
    definitions = [
        copy.deepcopy(n)
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name in ("CompressorMeta", "_CompressorPending")
        or isinstance(n, ast.FunctionDef)
        and n.name in ("_build_prefill_positions", "_cp_sliced_state_read_needed")
    ]
    cp_tree = ast.parse((DSV4 / "cp.py").read_text())
    definitions += [
        copy.deepcopy(n)
        for n in cp_tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("_CP_ROLE_MAIN", "_CP_ROLE_INDEXER")
    ]
    definitions += [
        copy.deepcopy(n)
        for n in cp_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "cp_should_gather"
    ]
    definitions.append(
        ast.ClassDef(
            name="CompressorFP8", bases=[], keywords=[], body=methods, decorator_list=[]
        )
    )
    module_ast = ast.parse("from __future__ import annotations\n")
    module_ast.body += definitions
    ast.fix_missing_locations(module_ast)
    m = types.ModuleType("_actual_compact_compressor_callers")
    sys.modules[m.__name__] = m
    m.__dict__.update(
        torch=torch,
        dataclass=dataclasses.dataclass,
        Any=Any,
        Optional=Optional,
        record_function_range=lambda _: contextlib.nullcontext(),
    )
    exec(compile(module_ast, str(FP8 / "compressor.py"), "exec"), m.__dict__)
    return m


caller = actual_callers()
# Synthetic stress layouts are intentionally distinct from the production ABI.
LAYOUTS = {
    "indexer": (128, 4, 12, "indexer"),
    "csa": (512, 4, 12, "main"),
    "hca": (512, 128, 132, "main"),
}
# Production ABI: raw blocks contain 256 entries; generation state is separate.
MODELTEST_LAYOUTS = {
    "indexer": (128, 4, 8, "indexer"),
    "csa": (512, 4, 8, "main"),
    "hca": (512, 128, 128, "main"),
}
MODELTEST_KV = {"indexer": (64, 8448), "csa": (64, 38016), "hca": (2, 2304)}
MODELTEST_STATE_PAGE_BYTES = {"indexer": 16384, "csa": 65536, "hca": 524288}


def rows_for_rank(rank):
    return torch.cat(
        (
            torch.arange(rank * 512, (rank + 1) * 512),
            torch.arange((7 - rank) * 512, (8 - rank) * 512),
        )
    )


def cp_context(chunk=0, rank=0, **changes):
    order = torch.cat([rows_for_rank(r) for r in range(4)])
    x = dict(
        cp_size=4,
        cp_rank=rank,
        chunk_length=1024,
        padded_seq_len=4096,
        seq_len_full=4096,
        chunk_lengths_per_req=(1024,),
        kv_cache_sharded=False,
        prefix_length=chunk * 4096,
        compact_geometry_verified=True,
        relative_positions=rows_for_rank(rank),
        global_positions=rows_for_rank(rank) + chunk * 4096,
        local_is_real=torch.ones(1024, dtype=torch.bool),
        unpad_restore=torch.argsort(order),
    )
    x.update(changes)
    return types.SimpleNamespace(**x)


def make_pool(head, blocks=260, entries=32, offset=23, pad=19):
    width = 132 if head == 128 else 584
    stride = entries * width + pad
    raw = torch.full((offset + blocks * stride + 37,), 173, dtype=torch.uint8)
    pool = raw.as_strided((blocks, entries, width), (stride, width, 1), offset)
    return raw, pool


def make_module(kind="indexer", rank=0, *, modeltest=False):
    """Synthetic default retained; modeltest=True selects measured production ABI."""
    head, ratio, state_eb, role = (MODELTEST_LAYOUTS if modeltest else LAYOUTS)[kind]
    m = caller.CompressorFP8()
    m.head_dim, m.compress_ratio, m.overlap, m._cp_role = head, ratio, ratio == 4, role
    m._state_eb, m._state_tokens_per_block = state_eb, 256
    m._kv_tokens_per_block = 256 if modeltest else 128
    m._kv_eb = m._kv_tokens_per_block // ratio
    m._kv_cache_sharded = False
    m._profile_label = None
    m._wkv_wgate_fused = object()
    if modeltest:
        eb, stride = MODELTEST_KV[kind]
        m.raw_pool, m._kv_pool_view = make_pool(
            head, entries=eb, offset=0, pad=stride - eb * (132 if head == 128 else 584)
        )
    else:
        m.raw_pool, m._kv_pool_view = make_pool(head, entries=m._kv_eb)
    # Distinct receiver-local physical block tables, same logical sequence.
    m._kv_block_table = ((torch.arange(256) + 11 * rank) % 256 + 1)[None, :].int()
    # Current4096 rows plus one prior block: wraps across eight chunks but
    # does not overwrite the prior tail required by the first CSA boundary.
    m._state_block_table = torch.arange(1, 18)[None, :].int()
    m._state_pool_3d = torch.full(
        (18, state_eb, 2 * (1 + int(m.overlap)) * head), -777.0, dtype=torch.float32
    )
    m._cp_ctx = cp_context(rank=rank)
    m.launches = []
    return m


def metadata(m, chunk=0, batched=True):
    pos = torch.arange(chunk * 4096, (chunk + 1) * 4096, dtype=torch.long)
    idx = torch.zeros(4096, dtype=torch.long)
    # Actual original Python metadata builders, not reimplemented formulas.
    states = m._compute_state_slot_mapping(pos, idx, torch.tensor([(chunk + 1) * 4096]))
    kv = m._compute_kv_slot_mapping(pos, idx)
    return caller.CompressorMeta(
        pos,
        idx,
        states,
        kv,
        idx.int(),
        bool(chunk),
        batched,
        torch.tensor([chunk * 4096]),
        torch.tensor([0, 4096]),
    )


def workspace(width=2048):
    w = workspace_module.PrefillWorkspace(
        torch.device("cpu"),
        q_rows=1024,
        q_dim=8,
        reserve_cp=True,
        cp_rows=4096,
        main_w=width,
        idx_w=512,
        align_bytes=1,
    )
    w._union.fill_(205)
    return w


def full_projection(chunk, width):
    return torch.arange(chunk * 4096, (chunk + 1) * 4096, dtype=torch.float32)[
        :, None
    ] * 4 + (torch.arange(width, dtype=torch.float32)[None, :] % 4)


class Work:
    def __init__(self, world, state):
        self.world, self.state = world, state

    def wait(self):
        self.world.log("work_wait")
        if not self.state["done"].wait(8):
            raise RuntimeError("CPU collective playback did not receive all four ranks")
        if self.state.get("error"):
            raise self.state["error"]
        return True


class Stream:
    def __init__(self, world, name):
        self.world, self.name = world, name

    def wait_stream(self, stream):
        self.world.log("wait_stream", self.name, stream.name)

    def wait_event(self, event):
        self.world.log("wait_event", self.name, event.name)


class Event:
    def __init__(self, world):
        self.world, self.name = world, "tail_event"

    def record(self, stream):
        self.world.log("event_record", stream.name)


class World:
    """Threaded CPU tensor playback only; never torch.distributed initialization."""

    def __init__(self):
        self.local = threading.local()
        self.lock = threading.RLock()
        self.events, self.rounds, self.seq = [], {}, {}
        self.group = object()
        self.size = 4
        self.corrupt_wire = False
        self.fail_gather = False
        self.raw = None
        self.cuda = types.SimpleNamespace(
            current_stream=self.current_stream,
            stream=self.on_stream,
            Stream=lambda **kw: Stream(self, "cp"),
            Event=lambda: Event(self),
            is_current_stream_capturing=lambda: False,
        )

    def rank(self):
        return getattr(self.local, "rank", 0)

    @contextlib.contextmanager
    def at(self, rank):
        before = self.rank()
        self.local.rank = rank
        try:
            yield
        finally:
            self.local.rank = before

    def log(self, name, *data):
        with self.lock:
            self.events.append((self.rank(), name, *data))

    def current_stream(self, *args):
        return getattr(self.local, "stream", None) or Stream(self, "current")

    @contextlib.contextmanager
    def on_stream(self, s):
        prev = getattr(self.local, "stream", None)
        self.local.stream = s
        try:
            yield
        finally:
            self.local.stream = prev

    def all_gather(self, dest, src, group=None, async_op=False):
        self.log("gather", str(src.dtype), async_op, self.current_stream().name)
        if self.fail_gather:
            raise RuntimeError("injected collective failure")
        rank = self.rank()
        with self.lock:
            i = self.seq.get(rank, 0)
            self.seq[rank] = i + 1
            state = self.rounds.setdefault(
                (id(group), i), {"done": threading.Event(), "ranks": {}}
            )
            state["ranks"][rank] = (dest, src.clone())
            if len(state["ranks"]) == 4:
                try:
                    payload = torch.cat([state["ranks"][r][1] for r in range(4)])
                    if self.corrupt_wire and payload.dtype == torch.uint8:
                        payload[0, 0] ^= 1
                    for out, _ in state["ranks"].values():
                        out.copy_(payload)
                except Exception as exc:
                    state["error"] = exc
                state["done"].set()
        work = Work(self, state)
        if async_op:
            return work
        work.wait()
        return None

    def fallback(self, x, cp, **kwargs):
        self.log("fallback_start", kwargs.get("cp_role"))
        return types.SimpleNamespace(full=self.raw)

    def fallback_wait(self, h):
        self.log("fallback_wait")
        return h.full

    def parallel(self, fn):
        def run(rank):
            with self.at(rank), torch.inference_mode():
                return fn(rank)

        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(run, range(4)))

    @contextlib.contextmanager
    def leaves(self, cuda_metadata=True):
        with contextlib.ExitStack() as stack:
            for name, value in vars(self.cuda).items():
                stack.enter_context(patch.object(torch.cuda, name, value))
            stack.enter_context(
                patch.object(
                    torch.Tensor,
                    "record_stream",
                    lambda t, s: self.log(
                        "record_stream",
                        s.name,
                        t.data_ptr(),
                        t.untyped_storage().data_ptr(),
                    ),
                )
            )
            if cuda_metadata:
                stack.enter_context(
                    patch.object(torch.Tensor, "is_cuda", property(lambda _: True))
                )
            stack.enter_context(
                patch.object(
                    torch.distributed, "all_gather_into_tensor", self.all_gather
                )
            )
            stack.enter_context(
                patch.object(torch.distributed, "get_world_size", lambda g: self.size)
            )
            stack.enter_context(
                patch.object(torch.distributed, "get_rank", lambda g: self.rank())
            )
            stack.enter_context(
                patch.object(
                    collective, "_get_group", lambda _: self.group, create=True
                )
            )
            stack.enter_context(
                patch.dict(
                    caller.__dict__,
                    {
                        "_linear_bf16_bf16_fp32": lambda x, w: x,
                        "cp_all_gather_full_async": self.fallback,
                        "cp_wait_gather_full": self.fallback_wait,
                    },
                )
            )
            yield self


def install_writer(m, world, raw, chunk, expected_state):
    """CPU numerical leaf double; verify every required raw/history row exactly.
    Actual production caller supplies kv/score views + masked full metadata.
    """
    out_dim = raw.shape[1] // 2

    def writer(kv, score, meta, *, seq_start, boundary_meta=None):
        world.log("writer", m._cp_role)
        m.launches.append((meta, seq_start))
        assert kv.shape == score.shape == (4096, out_dim)
        assert seq_start is None if meta.is_batched else seq_start == chunk * 4096
        staged = torch.cat((kv, score), dim=1)
        valid_state = meta.state_slots >= 0
        torch.testing.assert_close(
            staged[valid_state], raw[valid_state], rtol=0, atol=0
        )
        slots = meta.state_slots[valid_state]
        m._state_pool_3d.view(-1, raw.shape[1])[slots] = staged[valid_state]
        # Reference state update uses full projection and original full metadata.
        original = metadata(m, chunk, meta.is_batched)
        all_state = original.state_slots >= 0
        expected_state.view(-1, raw.shape[1])[original.state_slots[all_state]] = raw[
            all_state
        ]
        assert meta.positions.shape == (4096,)
        writer_meta = meta if boundary_meta is None else boundary_meta
        assert writer_meta.positions.numel() == 1024 // m.compress_ratio
        own = writer_meta.positions - chunk * 4096
        assert torch.equal(writer_meta.cu_seq_per_req, torch.tensor([0, 4096]))
        assert torch.equal(
            writer_meta.token_to_req, torch.zeros_like(writer_meta.token_to_req)
        )
        for boundary in own.tolist():
            lo = boundary + 1 - (1 + int(m.overlap)) * m.compress_ratio
            if lo < 0 and chunk:
                positions = torch.arange(chunk * 4096 + lo, chunk * 4096)
                block_ids = m._state_block_table[
                    0, (positions // 256) % m._state_block_table.shape[1]
                ].long()
                historical = m._state_pool_3d[block_ids, positions % m._state_eb]
                wanted = (
                    positions[:, None].float() * 4
                    + torch.arange(raw.shape[1])[None, :] % 4
                )
                torch.testing.assert_close(historical, wanted, rtol=0, atol=0)
            torch.testing.assert_close(
                staged[max(lo, 0) : boundary + 1],
                raw[max(lo, 0) : boundary + 1],
                rtol=0,
                atol=0,
            )
        data, scales, _ = runtime.split_pool(m._kv_pool_view, m.head_dim)
        slots = writer_meta.kv_slots
        ids = writer_meta.positions
        data[slots // m._kv_eb, slots % m._kv_eb] = (
            (ids[:, None] + torch.arange(data.shape[2])[None, :]) % 251
        ).byte()
        scales[slots // m._kv_eb, slots % m._kv_eb] = (
            (ids[:, None] * 3 + torch.arange(scales.shape[2])[None, :]) % 251
        ).byte()

    m._launch = writer


class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        self.inference = torch.inference_mode()
        self.inference.__enter__()
        self.addCleanup(self.inference.__exit__, None, None, None)
        self.flag = patch.dict(os.environ, {runtime.FLAG: "1"})
        self.flag.start()
        self.addCleanup(self.flag.stop)

    def test_geometry_all_8_chunks_4_ranks_3_layouts_state_masks(self):
        total = 0
        for kind, (head, ratio, eb, role) in LAYOUTS.items():
            m = make_module(kind)
            width = 2 * (1 + int(m.overlap)) * head
            for chunk in range(8):
                meta = metadata(m, chunk)
                state_rows = set(torch.where(meta.state_slots >= 0)[0].tolist())
                expected_global = 192 if eb == 12 else 2112
                for rank in range(4):
                    g = runtime.Geometry(rank, chunk * 4096, eb, 256, ratio, width)
                    all_tails = set(p for r in range(4) for p in g.tails(r))
                    self.assertEqual(len(all_tails), expected_global)
                    self.assertEqual(state_rows, all_tails)
                    owned = {p for lo, hi in g.intervals() for p in range(lo, hi)}
                    staged = owned | all_tails
                    self.assertTrue(state_rows <= staged)
                    self.assertEqual(len(owned), 1024)
                    self.assertEqual(
                        set(g.wire_boundaries()), set(range(ratio - 1, 4096, ratio))
                    )
                    for p in owned:
                        if (p + 1) % ratio == 0:
                            self.assertTrue(
                                set(
                                    range(
                                        max(0, p + 1 - (1 + int(m.overlap)) * ratio),
                                        p + 1,
                                    )
                                )
                                <= staged
                            )
                    self.assertEqual(
                        g.bytes_per_rank()["raw_tail_receive"],
                        expected_global * width * 4,
                    )
                    self.assertEqual(
                        g.bytes_per_rank()["compressed_send"],
                        (1024 // ratio) * ((132 if head == 128 else 584) + 8),
                    )
                    total += 1
        self.assertEqual(total, 96)

    def test_verified_geometry_flag_and_malformed_maps(self):
        cp = cp_context()
        mask = torch.ones(4096, dtype=torch.int32)
        restore = cp.unpad_restore
        self.assertTrue(runtime.verified_geometry(cp, mask, restore))
        with patch.dict(os.environ, {runtime.FLAG: "0"}):
            self.assertFalse(runtime.verified_geometry(cp, mask, restore))
        for changes in (
            {"cp_size": 2},
            {"chunk_length": 1000},
            {"prefix_length": 1},
            {"prefix_length": 32768},
            {"kv_cache_sharded": True},
            {"chunk_lengths_per_req": (512, 512)},
            {"seq_len_full": 4000},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(
                    runtime.verified_geometry(cp_context(**changes), mask, restore)
                )
        for bad in (restore.flip(0), restore[:-1], torch.zeros_like(restore)):
            self.assertFalse(runtime.verified_geometry(cp, mask, bad))
        mask[8] = 0
        self.assertFalse(runtime.verified_geometry(cp, mask, restore))

    def test_select_geometry_off_cpu_capture_shapes_warmup(self):
        m = make_module()
        meta = metadata(m)
        x = full_projection(0, 512)[:1024].clone()
        world = World()
        self.assertIsNone(runtime.select_geometry(m, m._cp_ctx, meta, x))
        with world.leaves():
            self.assertIsNotNone(runtime.select_geometry(m, m._cp_ctx, meta, x))
            with patch.dict(os.environ, {runtime.FLAG: "0"}):
                self.assertIsNone(runtime.select_geometry(m, m._cp_ctx, meta, x))
            with patch.object(torch.cuda, "is_current_stream_capturing", lambda: True):
                self.assertIsNone(runtime.select_geometry(m, m._cp_ctx, meta, x))
            for bad in (x[:1000], x.double(), x[:, ::2], x.t()):
                self.assertIsNone(runtime.select_geometry(m, m._cp_ctx, meta, bad))
            self.assertIsNone(runtime.select_geometry(m, m._cp_ctx, None, x))
            m._kv_pool_view = None
            self.assertIsNone(runtime.select_geometry(m, m._cp_ctx, meta, x))
            self.assertIsNone(m.start_prefill(x, 0, meta=meta, workspace=workspace()))
            m.wait_prefill_gather(None)
            m.finish_prefill(None)
            self.assertIsNone(m.forward(x, 0, meta=meta, workspace=workspace()))
            self.assertFalse(world.events)

    def test_split_pool_offset_padding_disjoint_actual_pack_scatter(self):
        for head in (128, 512):
            backing, pool = make_pool(head, blocks=3, entries=4)
            data, scales, layout = runtime.split_pool(pool, head)
            self.assertEqual(
                data.untyped_storage().data_ptr(), backing.untyped_storage().data_ptr()
            )
            self.assertEqual(
                scales.storage_offset(), pool.storage_offset() + 4 * data.shape[2]
            )
            before = backing.clone()
            slots = torch.tensor([0, 5, 11])
            data[slots // 4, slots % 4] = 31
            scales[slots // 4, slots % 4] = 69
            payload = runtime.pack_rows(
                data,
                scales,
                slots,
                layout,
                num_blocks=3,
                entries_per_block=4,
                check=False,
            )
            other_raw, other = make_pool(head, blocks=3, entries=4, offset=37, pad=27)
            od, oscale, olayout = runtime.split_pool(other, head)
            runtime.scatter_packed_rows(
                od,
                oscale,
                payload,
                olayout,
                num_blocks=3,
                entries_per_block=4,
                check=False,
            )
            self.assertTrue(
                torch.equal(od[slots // 4, slots % 4], data[slots // 4, slots % 4])
            )
            self.assertTrue(
                torch.equal(
                    oscale[slots // 4, slots % 4], scales[slots // 4, slots % 4]
                )
            )
            changed = torch.zeros(backing.numel(), dtype=torch.bool)
            for s in slots.tolist():
                b, e = divmod(s, 4)
                for view in (data, scales):
                    start = (
                        view.storage_offset() + b * view.stride(0) + e * view.stride(1)
                    )
                    changed[start : start + view.shape[2]] = True
            self.assertTrue(torch.equal(backing[~changed], before[~changed]))
            for invalid in (
                pool.float(),
                pool[0],
                pool[:, :, ::2],
                torch.empty(3, 4, 1, dtype=torch.uint8),
            ):
                with self.subTest(head=head, shape=tuple(invalid.shape)):
                    with self.assertRaises((ValueError, RuntimeError)):
                        runtime.split_pool(invalid, head)

    def test_actual_caller_four_rank_8chunk_playback_all_layouts(self):
        # Retained synthetic12/132 state stress cases, not MODELTEST dimensions.
        self._playback_layouts(LAYOUTS)

    def _playback_layouts(self, layouts, *, modeltest=False):
        for kind, (head, ratio, eb, role) in layouts.items():
            mods = [make_module(kind, r, modeltest=modeltest) for r in range(4)]
            states = [m._state_pool_3d.clone() for m in mods]
            for chunk in range(8):
                with self.subTest(kind=kind, chunk=chunk):
                    width = 2 * (1 + int(mods[0].overlap)) * head
                    full = full_projection(chunk, width)
                    world = World()
                    world.raw = full
                    workspaces = [workspace(width) for _ in range(4)]
                    metas = [metadata(m, chunk, batched=True) for m in mods]
                    pending = []
                    for r, m in enumerate(mods):
                        m._cp_ctx = cp_context(chunk, r)
                        install_writer(m, world, full, chunk, states[r])
                    before = [m.raw_pool.clone() for m in mods]
                    with world.leaves():
                        for r, m in enumerate(mods):
                            with world.at(r):
                                local = full[rows_for_rank(r)].clone()
                                pending.append(
                                    m.start_prefill(
                                        local,
                                        chunk * 4096,
                                        meta=metas[r],
                                        workspace=workspaces[r],
                                        cp_gather_stream=Stream(world, "cp"),
                                    )
                                )
                        for r, m in enumerate(mods):
                            with world.at(r):
                                m.wait_prefill_gather(pending[r])
                                m.wait_prefill_gather(
                                    pending[r]
                                )  # idempotent stage, no writer
                            self.assertTrue(torch.equal(before[r], m.raw_pool))
                        world.parallel(lambda r: mods[r].finish_prefill(pending[r]))
                        for r in range(4):
                            with world.at(r):
                                with self.assertRaises(RuntimeError):
                                    mods[r].finish_prefill(pending[r])
                    for r, m in enumerate(mods):
                        torch.testing.assert_close(
                            m._state_pool_3d, states[r], rtol=0, atol=0
                        )
                        data, scales, _ = runtime.split_pool(m._kv_pool_view, head)
                        expected_backing = before[r].clone()
                        current_meta = metas[r]
                        at = torch.where(current_meta.kv_slots >= 0)[0]
                        slots, ids = (
                            current_meta.kv_slots[at],
                            current_meta.positions[at],
                        )
                        stride = m._kv_pool_view.stride(0)
                        base = (
                            m._kv_pool_view.storage_offset()
                            + (slots // m._kv_eb) * stride
                        )
                        db, sb = (128, 4) if head == 128 else (576, 8)
                        data_offsets = (
                            base[:, None]
                            + (slots % m._kv_eb)[:, None] * db
                            + torch.arange(db)[None, :]
                        )
                        scale_offsets = (
                            base[:, None]
                            + m._kv_eb * db
                            + (slots % m._kv_eb)[:, None] * sb
                            + torch.arange(sb)[None, :]
                        )
                        expected_backing[data_offsets] = (
                            (ids[:, None] + torch.arange(db)[None, :]) % 251
                        ).byte()
                        expected_backing[scale_offsets] = (
                            (ids[:, None] * 3 + torch.arange(sb)[None, :]) % 251
                        ).byte()
                        self.assertTrue(
                            torch.equal(m.raw_pool, expected_backing),
                            "full receiver backing/guards changed",
                        )
                        # Consumer reads all current and historical rows through receiver-local slots.
                        for c in range(chunk + 1):
                            mm = metadata(m, c)
                            b = torch.where(mm.kv_slots >= 0)[0]
                            slots, ids = mm.kv_slots[b], mm.positions[b]
                            self.assertTrue(
                                torch.equal(
                                    data[slots // m._kv_eb, slots % m._kv_eb],
                                    (
                                        (
                                            ids[:, None]
                                            + torch.arange(data.shape[2])[None, :]
                                        )
                                        % 251
                                    ).byte(),
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    scales[slots // m._kv_eb, slots % m._kv_eb],
                                    (
                                        (
                                            ids[:, None] * 3
                                            + torch.arange(scales.shape[2])[None, :]
                                        )
                                        % 251
                                    ).byte(),
                                )
                            )
                        p = pending[r].compact_handle
                        covered = torch.zeros(4096, dtype=torch.bool)
                        covered[p.tail_dest] = True
                        for lo, hi in p.geometry.intervals():
                            covered[lo:hi] = True
                        # Full scratch only restores owned intervals and globally published state tails.
                        torch.testing.assert_close(
                            p.scratch[covered], full[covered], rtol=0, atol=0
                        )
                        self.assertTrue(
                            torch.all(p.scratch[~covered].view(torch.uint8) == 205)
                        )
                        records = [ev for ev in world.events if ev[0] == r]
                        gather = [ev for ev in records if ev[1] == "gather"]
                        self.assertEqual(
                            [(e[2], e[3], e[4]) for e in gather],
                            [
                                ("torch.float32", True, "cp"),
                                ("torch.uint8", False, "current"),
                            ],
                        )
                    self.assertFalse(torch.equal(metas[0].kv_slots, metas[1].kv_slots))
                    wire = world.rounds[(id(world.group), 1)]
                    for rank, (_, payload) in wire["ranks"].items():
                        ids = payload[:, :8].contiguous().view(torch.int64).flatten()
                        self.assertTrue(
                            torch.equal(
                                ids,
                                pending[rank].compact_handle.local_boundaries
                                + chunk * 4096,
                            )
                        )

    def _check_modeltest_and_playback(self, kind):
        m = make_module(kind, modeltest=True)
        head, ratio, eb, role = MODELTEST_LAYOUTS[kind]
        width = 2 * (1 + int(m.overlap)) * head
        self.assertEqual(m._state_tokens_per_block, 256)
        self.assertEqual(m._kv_tokens_per_block, 256)
        self.assertEqual(
            m._state_pool_3d.stride(0) * m._state_pool_3d.element_size(),
            MODELTEST_STATE_PAGE_BYTES[kind],
        )
        kv_eb, stride = MODELTEST_KV[kind]
        self.assertEqual(
            tuple(m._kv_pool_view.shape[1:]), (kv_eb, 132 if head == 128 else 584)
        )
        self.assertEqual(m._kv_pool_view.stride(0), stride)
        self.assertEqual(m._kv_pool_view.storage_offset(), 0)
        for chunk in range(8):
            meta = metadata(m, chunk)
            for rank in range(4):
                g = runtime.Geometry(rank, chunk * 4096, eb, 256, ratio, width)
                tail = {p for r in range(4) for p in g.tails(r)}
                self.assertEqual(
                    tail, set(torch.where(meta.state_slots >= 0)[0].tolist())
                )
                self.assertEqual(len(tail), 128 if ratio == 4 else 2048)
                self.assertEqual(
                    g.bytes_per_rank()["raw_tail_receive"], len(tail) * width * 4
                )
        self._playback_layouts({kind: MODELTEST_LAYOUTS[kind]}, modeltest=True)

    def test_modeltest_indexer_eb8_kv64_stride8448_playback(self):
        self._check_modeltest_and_playback("indexer")

    def test_modeltest_csa_eb8_kv64_stride38016_playback(self):
        self._check_modeltest_and_playback("csa")

    def test_modeltest_hca_eb128_kv2_stride2304_playback(self):
        self._check_modeltest_and_playback("hca")

    def test_forward_uses_actual_compact_split_caller(self):
        world = World()
        full = full_projection(0, 512)
        world.raw = full
        mods = [make_module(rank=r) for r in range(4)]
        for m in mods:
            install_writer(m, world, full, 0, m._state_pool_3d.clone())
        ws = [workspace() for _ in range(4)]
        with world.leaves():
            results = world.parallel(
                lambda r: mods[r].forward(
                    full[rows_for_rank(r)], 0, meta=metadata(mods[r]), workspace=ws[r]
                )
            )
        self.assertEqual(results, [None] * 4)
        self.assertEqual([len(m.launches) for m in mods], [1] * 4)
        self.assertEqual(sum(e[1] == "gather" for e in world.events), 8)
        self.assertEqual(sum(e[1] == "fallback_start" for e in world.events), 0)

    def test_actual_forward_off_and_unsupported_use_old_gather(self):
        for flag, verified in [("0", True), ("1", False)]:
            world = World()
            world.raw = full_projection(0, 512)
            m = make_module()
            m._cp_ctx.compact_geometry_verified = verified
            captured = []
            m._launch = lambda kv, score, meta, **kw: captured.append((kv, score, meta))
            meta = metadata(m)
            with world.leaves(), patch.dict(os.environ, {runtime.FLAG: flag}):
                m.forward(
                    world.raw[rows_for_rank(0)], 0, meta=meta, workspace=workspace()
                )
            self.assertEqual(len(captured), 1)
            self.assertIs(captured[0][2], meta)
            self.assertTrue(torch.equal(torch.cat(captured[0][:2], 1), world.raw))
            self.assertEqual(sum(e[1] == "fallback_start" for e in world.events), 1)
            self.assertEqual(sum(e[1] == "gather" for e in world.events), 0)

    def test_bad_metadata_rejected_before_first_collective(self):
        changes = {
            "foreign_positions": lambda x: dataclasses.replace(
                x, positions=x.positions + 1
            ),
            "foreign_request": lambda x: dataclasses.replace(x, b_idx=x.b_idx + 1),
            "unstaged_state_write": lambda x: dataclasses.replace(
                x, state_slots=torch.zeros_like(x.state_slots)
            ),
            "duplicate_receiver": lambda x: dataclasses.replace(
                x, kv_slots=torch.zeros_like(x.kv_slots)
            ),
            "foreign_prefix": lambda x: dataclasses.replace(
                x, seq_start_per_req=torch.tensor([4096])
            ),
            "bad_raw_window": lambda x: dataclasses.replace(
                x, cu_seq_per_req=torch.tensor([0, 1024])
            ),
            "missing_raw_window": lambda x: dataclasses.replace(x, cu_seq_per_req=None),
        }
        for key, mutate in changes.items():
            with self.subTest(case=key):
                world = World()
                m = make_module()
                meta = mutate(metadata(m))
                before = m.raw_pool.clone()
                with world.leaves(), self.assertRaises((ValueError, RuntimeError)):
                    m.start_prefill(
                        full_projection(0, 512)[rows_for_rank(0)],
                        0,
                        meta=meta,
                        workspace=workspace(),
                    )
                self.assertFalse(any(e[1] == "gather" for e in world.events))
                self.assertTrue(torch.equal(before, m.raw_pool))

    def test_group_mismatch_rejected_before_gather(self):
        for size, actual_rank in [(8, 0), (4, 1)]:
            world = World()
            world.size = size
            m = make_module()
            with world.leaves(), world.at(actual_rank), self.assertRaises(RuntimeError):
                runtime.start(
                    m,
                    full_projection(0, 512)[:1024],
                    m._cp_ctx,
                    metadata(m),
                    workspace(),
                    "indexer",
                    None,
                )
            self.assertFalse(any(e[1] == "gather" for e in world.events))

    def test_rank_uniform_tail_then_two_role_payload_order(self):
        world = World()
        raw = {"indexer": full_projection(0, 512), "csa": full_projection(0, 2048)}
        mods = {k: [make_module(k, r) for r in range(4)] for k in raw}
        ws = [workspace() for _ in range(4)]
        handles = {k: [] for k in raw}
        for k in raw:
            for m in mods[k]:
                install_writer(m, world, raw[k], 0, m._state_pool_3d.clone())
        with world.leaves():
            for r in range(4):
                with world.at(r):
                    for k in ("indexer", "csa"):
                        m = mods[k][r]
                        handles[k].append(
                            m.start_prefill(
                                raw[k][rows_for_rank(r)],
                                0,
                                meta=metadata(m),
                                workspace=ws[r],
                                cp_gather_stream=Stream(world, "cp"),
                            )
                        )
            world.parallel(
                lambda r: mods["indexer"][r].finish_prefill(handles["indexer"][r])
            )
            for r in range(4):
                with world.at(r):
                    mods["csa"][r].wait_prefill_gather(handles["csa"][r])
            world.parallel(lambda r: mods["csa"][r].finish_prefill(handles["csa"][r]))
        for r in range(4):
            events = [x for x in world.events if x[0] == r and x[1] == "gather"]
            self.assertEqual(
                [x[2:4] for x in events],
                [("torch.float32", True)] * 2 + [("torch.uint8", False)] * 2,
            )
            self.assertNotEqual(
                handles["indexer"][r].compact_handle.scratch.data_ptr(),
                handles["csa"][r].compact_handle.scratch.data_ptr(),
            )

    def test_handle_keeps_input_meta_workspace_alive(self):
        world = World()
        m = make_module()
        meta = metadata(m)
        ws = workspace()
        raw = full_projection(0, 512)[:1024].clone()
        refs = [weakref.ref(x) for x in (ws, raw)]
        with world.leaves():
            handle = runtime.start(
                m, raw, m._cp_ctx, meta, ws, "indexer", Stream(world, "cp")
            )
            del raw, meta, ws
            gc.collect()
            self.assertTrue(all(r() is not None for r in refs))
            with self.assertRaises(RuntimeError):
                handle.owned_meta()
            with self.assertRaises(RuntimeError):
                handle.replicate(m)
        del handle
        gc.collect()
        self.assertTrue(all(r() is None for r in refs))

    def test_actual_launch_full_state_owned_boundary_and_full_raw_ABI(self):
        for kind in LAYOUTS:
            for batched in (True, False):
                with self.subTest(kind=kind, batched=batched):
                    world = World()
                    m = make_module(kind)
                    meta = metadata(m, 1, batched)
                    m._cp_ctx = cp_context(1, 2)
                    width = 2 * (1 + int(m.overlap)) * m.head_dim
                    raw = full_projection(1, width)
                    m._ensure_cos_sin_cache = lambda dev: torch.empty(1)
                    m.norm = types.SimpleNamespace(weight=torch.ones(m.head_dim))
                    m.norm_eps, m.rope_head_dim, m.ape = 1e-6, 64, torch.empty(1)
                    captures = []
                    with world.leaves(), world.at(2):
                        h = runtime.start(
                            m,
                            raw[rows_for_rank(2)],
                            m._cp_ctx,
                            meta,
                            workspace(),
                            "main",
                            Stream(world, "cp"),
                        )
                        # Only metadata selection is needed here; no pretend collective completion.
                        h.state = "staged"
                        owned = h.owned_meta()
                        with patch.dict(
                            caller.__dict__,
                            {
                                "run_save_partial_states": lambda *a, **kw: captures.append(
                                    ("state", a, kw)
                                ),
                                "run_fused_compress_kv_write": lambda *a, **kw: captures.append(
                                    ("fused", a, kw)
                                ),
                            },
                        ):
                            m._launch(
                                raw[:, : width // 2],
                                raw[:, width // 2 :],
                                h.meta,
                                seq_start=None if batched else 4096,
                                boundary_meta=owned,
                            )
                    self.assertEqual([c[0] for c in captures], ["state", "fused"])
                    state, fused = captures
                    self.assertEqual(state[1][3].numel(), 4096)
                    self.assertTrue(torch.equal(state[1][5], meta.state_slots))
                    self.assertEqual(fused[1][2].numel(), 1024 // m.compress_ratio)
                    self.assertEqual(fused[1][10].shape[0], 4096)
                    self.assertEqual(fused[1][11].shape[0], 4096)
                    self.assertTrue(
                        torch.equal(fused[1][2], meta.positions[h.local_boundaries])
                    )
                    self.assertTrue(
                        torch.equal(fused[1][9], meta.kv_slots[h.local_boundaries])
                    )
                    self.assertFalse(fused[2]["disable_raw_path"])
                    if batched:
                        self.assertTrue(
                            torch.equal(
                                fused[2]["cu_seq_per_req"], torch.tensor([0, 4096])
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                fused[2]["seq_start_per_req"], torch.tensor([4096])
                            )
                        )
                    else:
                        self.assertEqual(fused[1][13], 4096)
                        self.assertIsNone(fused[2]["cu_seq_per_req"])

    def test_inference_snapshot_and_private_mutation_guards(self):
        world = World()
        mods = [make_module(rank=r) for r in range(4)]
        metas = [metadata(m) for m in mods]
        handles = []
        self.assertTrue(metas[0].kv_slots.is_inference())
        with world.leaves():
            for r, m in enumerate(mods):
                with world.at(r):
                    handles.append(
                        runtime.start(
                            m,
                            full_projection(0, 512)[rows_for_rank(r)],
                            m._cp_ctx,
                            metas[r],
                            workspace(),
                            "indexer",
                            None,
                        )
                    )
            h = handles[0]
            saved = h.meta.kv_slots.clone()
            metas[0].kv_slots.add_(1)
            self.assertTrue(torch.equal(h.meta.kv_slots, saved))
            self.assertFalse(h.meta.kv_slots.is_inference())
            h.assert_binding(mods[0])
            with torch.inference_mode(False):
                h.meta.kv_slots.add_(1)
            before = h.scratch.clone()
            marker = len(world.events)
            with self.assertRaisesRegex(RuntimeError, "metadata"):
                h.stage()
            observed = [e[1] for e in world.events[marker:]]
            self.assertEqual(observed, ["wait_event", "work_wait"])
            self.assertTrue(torch.equal(h.scratch, before))

    def test_gathered_workspace_record_stream_and_staged_fast_rejection(self):
        world = World()
        mods = [make_module(rank=r) for r in range(4)]
        handles = []
        with world.leaves():
            for r, m in enumerate(mods):
                with world.at(r):
                    handles.append(
                        runtime.start(
                            m,
                            full_projection(0, 512)[rows_for_rank(r)],
                            m._cp_ctx,
                            metadata(m),
                            workspace(),
                            "indexer",
                            Stream(world, "cp"),
                        )
                    )
            h = handles[0]
            records = [e for e in world.events if e[0] == 0 and e[1] == "record_stream"]
            self.assertTrue(any(e[3] == h.send.data_ptr() for e in records))
            self.assertTrue(
                any(e[4] == h.gathered.untyped_storage().data_ptr() for e in records)
            )
            h.stage()
            with torch.inference_mode(False):
                h.meta.positions.add_(1)
            marker = len(world.events)
            with self.assertRaisesRegex(RuntimeError, "metadata"):
                h.stage()
            self.assertEqual(
                world.events[marker:], []
            )  # staged path rejects immediately

    def test_finish_caller_mutation_drains_before_reject(self):
        world = World()
        mods = [make_module(rank=r) for r in range(4)]
        pending = []
        with world.leaves():
            for r, m in enumerate(mods):
                with world.at(r):
                    pending.append(
                        m.start_prefill(
                            full_projection(0, 512)[rows_for_rank(r)],
                            0,
                            meta=metadata(m),
                            workspace=workspace(),
                        )
                    )
            with torch.inference_mode(False):
                pending[0].compact_handle.meta.positions.add_(1)
            marker = len(world.events)
            with self.assertRaisesRegex(RuntimeError, "metadata"):
                mods[0].finish_prefill(pending[0])
            self.assertEqual(
                [e[1] for e in world.events[marker:]], ["wait_event", "work_wait"]
            )

    def test_workspace_snapshot_and_indices_cached_across_serial_layers(self):
        world = World()
        m = make_module()
        meta = metadata(m)
        ws = workspace()
        local = full_projection(0, 512)[:1024]
        with world.leaves():
            h = runtime.start(m, local, m._cp_ctx, meta, ws, "indexer", None)
            # Completed previous layer is a precondition of same-role workspace reuse.
            h.state = "finished"
            h2 = runtime.start(m, local, m._cp_ctx, meta, ws, "indexer", None)
        self.assertIs(h.meta, h2.meta)
        for key in (
            "tail_indices",
            "tail_dest",
            "boundaries",
            "local_boundaries",
            "receiver_slots",
        ):
            self.assertIs(getattr(h, key), getattr(h2, key))
        self.assertEqual(len(ws._compact_cp_indices), 1)
        other = workspace()
        with world.leaves():
            h3 = runtime.start(m, local, m._cp_ctx, meta, other, "indexer", None)
        self.assertIsNot(h2.meta, h3.meta)
        self.assertIsNot(h2.tail_dest, h3.tail_dest)

    def test_pool_rebind_and_normal_table_version_reject_before_writer(self):
        for field in (
            "_kv_pool_view",
            "_state_pool_3d",
            "_kv_block_table",
            "_state_block_table",
        ):
            with self.subTest(field=field):
                world = World()
                mods = [make_module(rank=r) for r in range(4)]
                m = mods[0]
                with torch.inference_mode(False):
                    m._kv_block_table = m._kv_block_table.clone()
                    m._state_block_table = m._state_block_table.clone()
                called = []
                m._launch = lambda *a, **kw: called.append(True)
                with world.leaves():
                    pending = []
                    # The fixed caller really drains; provide every CPU peer,
                    # not a fake completed work handle or a single-rank gather.
                    for r, producer in enumerate(mods):
                        with world.at(r):
                            pending.append(
                                producer.start_prefill(
                                    full_projection(0, 512)[rows_for_rank(r)],
                                    0,
                                    meta=metadata(producer),
                                    workspace=workspace(),
                                )
                            )
                    p = pending[0]
                    if field.endswith("table"):
                        with torch.inference_mode(False):
                            getattr(m, field).add_(1)
                    else:
                        setattr(m, field, getattr(m, field).clone())
                    with self.assertRaisesRegex(RuntimeError, "pool/table"):
                        m.finish_prefill(p)
                self.assertEqual(called, [])

    def test_foreign_wire_rejected_without_remote_scatter(self):
        world = World()
        world.corrupt_wire = True
        mods = [make_module("hca", r) for r in range(4)]
        handles = []
        with world.leaves():
            for r, m in enumerate(mods):
                with world.at(r):
                    h = runtime.start(
                        m,
                        full_projection(0, 1024)[rows_for_rank(r)],
                        m._cp_ctx,
                        metadata(m),
                        workspace(),
                        "main",
                        None,
                    )
                    handles.append(h)
            for r, h in enumerate(handles):
                with world.at(r):
                    h.stage()
            before = [m.raw_pool.clone() for m in mods]

            def publish(r):
                try:
                    handles[r].replicate(mods[r])
                except RuntimeError as e:
                    return str(e)
                return "NO_REJECTION"

            errors = world.parallel(publish)
        self.assertTrue(all("foreign logical rows" in e for e in errors), errors)
        self.assertTrue(all(torch.equal(b, m.raw_pool) for b, m in zip(before, mods)))

    def test_gather_exception_propagates_without_writer_or_pool_write(self):
        world = World()
        world.fail_gather = True
        m = make_module()
        before = m.raw_pool.clone()
        calls = []
        m._launch = lambda *a, **kw: calls.append(True)
        with (
            world.leaves(),
            self.assertRaisesRegex(RuntimeError, "injected collective"),
        ):
            m.start_prefill(
                full_projection(0, 512)[:1024],
                0,
                meta=metadata(m),
                workspace=workspace(),
            )
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(before, m.raw_pool))

    def test_writer_exception_does_not_publish_or_allow_double_consume(self):
        world = World()
        mods = [make_module("hca", r) for r in range(4)]
        handles = []
        calls = []

        def broken(*a, **kw):
            calls.append("writer")
            raise RuntimeError("injected writer failure")

        mods[0]._launch = broken
        with world.leaves():
            for r, m in enumerate(mods):
                with world.at(r):
                    handles.append(
                        m.start_prefill(
                            full_projection(0, 1024)[rows_for_rank(r)],
                            0,
                            meta=metadata(m),
                            workspace=workspace(),
                        )
                    )
            with self.assertRaisesRegex(RuntimeError, "injected writer"):
                mods[0].finish_prefill(handles[0])
            self.assertEqual(
                sum(e[1] == "gather" and e[2] == "torch.uint8" for e in world.events), 0
            )
            with self.assertRaises(RuntimeError):
                mods[0].finish_prefill(handles[0])
        self.assertEqual(
            calls, ["writer"], "failed handle must not execute numerical writer twice"
        )

    def test_bad_token_request_mapping_rejected_before_first_collective(self):
        world = World()
        m = make_module()
        bad = dataclasses.replace(
            metadata(m), token_to_req=torch.ones(4096, dtype=torch.int32)
        )
        with world.leaves(), self.assertRaises((ValueError, RuntimeError)):
            m.start_prefill(
                full_projection(0, 512)[:1024], 0, meta=bad, workspace=workspace()
            )
        self.assertFalse(any(e[1] == "gather" for e in world.events))

    def test_no_cuda_initialization(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(torch.distributed.is_initialized())
        self.assertNotIn("rtp_llm.ops", sys.modules)
        self.assertNotIn("flashinfer", sys.modules)
        self.assertNotIn("deep_gemm", sys.modules)


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(
        json.dumps(
            {
                "torch_version": str(torch.__version__),
                "python_version": sys.version,
                "inference_mode_whole_calls": True,
                "package_imports": "stub namespaces/direct files; NOT native import binding",
                "source_hashes": SOURCE_HASHES,
                "actual_methods": list(METHODS),
                "scope": "CPU source/caller/byte protocol; NOT GPU or original compressor arithmetic proof",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    unittest.main(verbosity=2)
