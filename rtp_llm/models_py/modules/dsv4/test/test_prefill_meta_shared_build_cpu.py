"""CPU regression tests for shared prefill metadata construction.

Compare shared and independent builds across batch, prefix and ratio layouts.
The tests execute extracted production builders and CPU Triton metadata kernels;
they do not cover native imports or GPU execution."""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from runpy import run_path
from typing import Any, Dict, NamedTuple, Optional, Tuple
from unittest.mock import patch

import torch

HERE = pathlib.Path(__file__).resolve().parent
DSV4 = HERE.parent
ATTN_PATH = DSV4 / "fp8" / "attention.py"
ATTN_SRC = ATTN_PATH.read_text()
CP_PATH = DSV4 / "cp.py"
CP_SRC = CP_PATH.read_text()
PREFILL_META_PATH = DSV4 / "fp8" / "prefill_meta.py"
FLAG = "DSV4_FP8_PREFILL_META_SHARED_BUILD"

_PREFIX = "rtp_llm.models_py.modules.dsv4"


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


def _load_dsv4_tree():
    _package("rtp_llm")
    _package("rtp_llm.models_py")
    _package("rtp_llm.models_py.modules")
    _package(_PREFIX, str(DSV4))
    fp8 = _package(_PREFIX + ".fp8", str(DSV4 / "fp8"))
    dist = _package("rtp_llm.models_py.distributed")
    collective = types.ModuleType("rtp_llm.models_py.distributed.collective_torch")

    class _Group:
        TP = "TP"

    collective.Group = _Group
    collective.all_gather = lambda *a, **k: None
    collective._get_group = lambda g: g
    sys.modules[collective.__name__] = collective
    dist.collective_torch = collective
    # ``prefill_meta`` locally imports ``bind_attn_cache`` from the real
    # attention module (which needs native deps) — stub just that symbol.
    if _PREFIX + ".fp8.attention" not in sys.modules:
        import contextlib

        attn_stub = types.ModuleType(_PREFIX + ".fp8.attention")

        @contextlib.contextmanager
        def _bind_attn_cache(attn, kv_cache, block_tables_by_type):
            yield attn

        attn_stub.bind_attn_cache = _bind_attn_cache
        sys.modules[_PREFIX + ".fp8.attention"] = attn_stub
    # Real leaf modules (light deps only).
    _load(_PREFIX + "._profiler", DSV4 / "_profiler.py")
    cp = _load(_PREFIX + ".cp", DSV4 / "cp.py")
    _load(_PREFIX + ".kv_cache_utils", DSV4 / "kv_cache_utils.py")
    _load(_PREFIX + ".fp8._trap_utils", DSV4 / "fp8" / "_trap_utils.py")
    _load(_PREFIX + ".fp8._swa_cp_byte_sliced", DSV4 / "fp8" / "_swa_cp_byte_sliced.py")
    _load(_PREFIX + ".fp8._cp_attention_shard", DSV4 / "fp8" / "_cp_attention_shard.py")
    swa_ops = _load(
        _PREFIX + ".fp8._swa_ops_triton", DSV4 / "fp8" / "_swa_ops_triton.py"
    )
    fp8_kv_utils = _load(
        _PREFIX + ".fp8._kv_cache_utils", DSV4 / "fp8" / "_kv_cache_utils.py"
    )
    prefill_meta = _load(_PREFIX + ".fp8.prefill_meta", PREFILL_META_PATH)
    return cp, swa_ops, prefill_meta, fp8, fp8_kv_utils


# --- AST extraction of the real attention.py pieces -------------------------
def _extract_attn():
    tree = ast.parse(ATTN_SRC)
    want_funcs = {
        "_flat_1d",
        "_memo_identity",
        "_prefill_maxes_host",
        "_gather_len_max_host",
        "_suffix_gather_lens_max_host",
        "_any_prefix_continuation",
        "_first_prefix_int",
        "_build_suffix_pool_slot_mapping",
        "_sm120_paged_cache_prefill_enabled",
        "_use_cp_cache_hit_raw_q_merge",
        "_force_cp_cache_hit_raw_q_merge",
        "_force_all_cp_raw_q_merge",
    }
    want_classes = {
        "WorkspaceMeta",
        "SwaPrefillMeta",
        "PrefillMeta",
        "CsaPrefillMeta",
        "HcaPrefillMeta",
    }
    want_methods = {
        "_build_workspace_meta",
        "_build_swa_prefill_meta_varlen",
        "_build_shared_prefill_meta",
        "_build_csa_prefill_meta",
        "_build_hca_prefill_meta",
        "_build_compressor_meta",
        "_build_swa_cp_byte_compaction",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_funcs:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in want_classes:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "AttentionFP8":
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name in want_methods:
                    body.append(m)
    names = {n.name for n in body}
    want_all = want_funcs | want_classes | want_methods
    assert want_all == names, f"missing: {want_all - names}"
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    mod = ast.fix_missing_locations(ast.Module(body=[future] + body, type_ignores=[]))
    import logging
    import os as _os

    env = {
        "torch": torch,
        "Optional": Optional,
        "Tuple": Tuple,
        "Any": Any,
        "Dict": Dict,
        "NamedTuple": NamedTuple,
        "Union": __import__("typing").Union,
        "os": _os,
        "logging": logging,
        "record_function_range": _profiler.record_function_range,
        "_dsv4_pool_tokens_per_block": fp8_kv_utils.require_pool_tokens_per_block,
        "_dsv4_pool_owner_tokens_per_block": fp8_kv_utils.pool_physical_tokens_per_block,
        "make_compressed_k_pool_reader": _make_compressed_k_pool_reader_stub,
        "LocalPoolReader": _LocalPoolReaderStub,
        "is_sm120": lambda device=None: _IS_SM120[0],
        "is_sm12x": lambda: _IS_SM120[0],
        "prefer_raw_q_merge_attention_conservative": _cp_attention_shard.prefer_raw_q_merge_attention_conservative,
        "build_cp_full_prefill_positions": cp.build_cp_full_prefill_positions,
        "build_cp_full_prefill_positions_shared": cp.build_cp_full_prefill_positions_shared,
        "build_cp_byte_sliced_slot_compaction": _swa_cp_byte_sliced.build_cp_byte_sliced_slot_compaction,
        "_SWA_CP_RR_LOGGED_SITES": set(),
        "CPContext": object,
        "CPByteSlicedSlotCompaction": object,
        "PrefillWorkspace": object,
    }
    exec(compile(mod, str(ATTN_PATH), "exec"), env)
    return env


_IS_SM120 = [False]


class _LocalPoolReaderStub:
    """Quacks like ``LocalPoolReader`` for the ``isinstance`` gate.

    Stateless strategy object: equal by type for the meta comparators.
    """

    def __eq__(self, other):
        return type(other) is type(self)

    def __hash__(self):
        return hash(type(self))

    def fill(self, **kwargs):
        raise NotImplementedError

    def gather_packed(self, **kwargs):
        raise NotImplementedError


def _make_compressed_k_pool_reader_stub(
    cp_ctx=None,
    kv_cache_sharded=False,
    per_req_total_kv_lens=None,
    block_size=None,
    owner_block_size=None,
):
    # The non-sharded production path returns a plain local reader.
    assert not kv_cache_sharded, "sharded reader is out of scope for the CPU test"
    return _LocalPoolReaderStub()


def _recorded_call_counters():
    """Wrap torch.* factories used by the meta builders with counters."""
    counts = {}

    def wrap(name):
        real = getattr(torch, name)
        counts[name] = 0

        def wrapped(*a, **k):
            counts[name] += 1
            return real(*a, **k)

        return wrapped

    return counts, wrap


# --- Stub attention ---------------------------------------------------------
class _StubKvCache:
    seq_size_per_block = 256
    kernel_seq_size_per_block = 256

    def get_seq_size_per_block(self, tag):
        return self.seq_size_per_block

    def get_kernel_seq_size_per_block(self, tag):
        return self.kernel_seq_size_per_block


class _StubAttention:
    """Carries exactly what the extracted meta builders read off ``self``."""

    def __init__(
        self,
        *,
        compress_ratio: int,
        freqs_cis: torch.Tensor,
        cp_ctx,
        block_tables: dict,
        eb_by_type: dict,
        window_size: int = 64,
        kv_present: bool = True,
    ):
        self.compress_ratio = compress_ratio
        self.rope_head_dim = 64
        self.window_size = window_size
        self.freqs_cis = freqs_cis
        self._cp_ctx = cp_ctx
        self._kv_cache = _StubKvCache() if kv_present else None
        self._block_tables_by_type = block_tables
        self._eb_by_type = eb_by_type
        self.calls = {}

    # --- pool helpers (mirroring the real ones' pool-derived values) ---
    def _pool_entries_per_block(self, tag):
        return int(self._eb_by_type.get(tag, 0))

    def _swa_entries_per_block(self):
        return self._pool_entries_per_block(kv_cache_utils.SWA_KV)

    def _swa_cp_byte_sliced(self):
        return False

    def _pool_raw_u8(self, tag):
        return None

    def _ensure_freqs_cis_bound(self):
        return None

    # --- extracted real methods, bound ---

    # --- bucket builders: stubbed recorders by default (the shared-builder
    # tests), replaced with the real extracted methods in the CSA/HCA
    # threading tests. ---
    def _build_csa_prefill_meta(self, seqlen, sp_int, device, **kwargs):
        self.calls["csa"] = kwargs
        return ATT["CsaPrefillMeta"](
            indexer_meta=None, compressor_meta=None, workspace_meta=None
        )

    def _build_hca_prefill_meta(self, seqlen, sp_int, device, **kwargs):
        self.calls["hca"] = kwargs
        return ATT["HcaPrefillMeta"](compressor_meta=None, workspace_meta=None)


# --- Geometry fixtures ------------------------------------------------------
def _zigzag_restore(chunk_lengths, cp_size):
    total_chunk = sum(chunk_lengths)
    restore = torch.empty(cp_size * total_chunk, dtype=torch.int32)
    chunk_offset = 0
    seq_offset = 0
    for chunk in chunk_lengths:
        pair = chunk // 2
        padded = chunk * cp_size
        for rank in range(cp_size):
            dst_base = rank * total_chunk + chunk_offset
            even = torch.arange(
                seq_offset + rank * pair, seq_offset + rank * pair + pair
            )
            odd = torch.arange(
                seq_offset + padded - (rank + 1) * pair,
                seq_offset + padded - rank * pair,
            )
            restore[even.long()] = torch.arange(
                dst_base, dst_base + pair, dtype=torch.int32
            )
            restore[odd.long()] = torch.arange(
                dst_base + pair, dst_base + 2 * pair, dtype=torch.int32
            )
        chunk_offset += chunk
        seq_offset += padded
    return restore


def _make_cp_ctx(cp_rank, chunk_lengths, real_lengths, prefix):
    """A REAL CPContext from the real cp.py (flag off), on CPU.

    Production populates the host mirrors via ``build_cp_context_for_forward``'s
    pinned-prefix path; the direct builder leaves them None, so set them here
    to exactly the values the production path would carry.
    """
    parts = []
    for chunk, actual in zip(chunk_lengths, real_lengths):
        padded = chunk * 4
        part = torch.zeros(padded, dtype=torch.int32)
        part[:actual] = 1
        parts.append(part)
    info = types.SimpleNamespace(
        prefill_qkv_padding_mask=torch.cat(parts),
        prefill_qkv_restore_indice=_zigzag_restore(chunk_lengths, 4),
        prefill_actual_input_lengths_cpu=torch.tensor(real_lengths, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor(chunk_lengths, dtype=torch.int32),
    )
    with patch.dict(os.environ, {"DSV4_CP_CONTEXT_PLAN_CACHE": "0"}):
        ctx = cp.build_cp_context(
            info,
            4,
            cp_rank,
            sum(chunk_lengths),
            torch.device("cpu"),
            position_offset=int(prefix),
        )
    ctx.prefix_lengths_full_host = tuple([int(prefix)] * len(real_lengths))
    ctx.input_lengths_full_host = tuple(int(v) for v in real_lengths)
    return ctx


def _make_stubs(cp_ctx, ratios=(0, 4, 128), shared_freqs=True):
    """One stub attention per ratio bucket; CP-full pools bound."""
    n_reqs = len(cp_ctx.input_lengths_full_host)
    blocks = 6
    bt = torch.arange(1, n_reqs * blocks + 1, dtype=torch.int32).view(n_reqs, blocks)
    block_tables = {
        kv_cache_utils.SWA_KV: bt.clone(),
        kv_cache_utils.CSA_KV: bt.clone(),
        kv_cache_utils.HCA_KV: bt.clone(),
    }
    eb = {
        kv_cache_utils.SWA_KV: 32,
        kv_cache_utils.CSA_KV: 8,
        kv_cache_utils.HCA_KV: 8,
    }
    # Deterministic content (no RNG): separate ``_make_stubs`` calls must
    # produce byte-identical freqs so cross-fixture byte comparisons hold.
    freqs_csa_hca = (
        torch.arange(33000 * 8, dtype=torch.float32).reshape(33000, 8) / 1000.0
    )  # shared compress-rope object
    freqs_swa = freqs_csa_hca if shared_freqs else freqs_csa_hca.flip(0).clone()
    stubs = {}
    for r in ratios:
        stubs[r] = _StubAttention(
            compress_ratio=r,
            freqs_cis=freqs_swa if r == 0 else freqs_csa_hca,
            cp_ctx=cp_ctx,
            block_tables=block_tables,
            eb_by_type=eb,
        )
    return stubs


def _forward_args(cp_ctx, max_len):
    """The identical argument set every bucket build receives."""
    B = len(cp_ctx.input_lengths_full_host)
    chunk_length = cp_ctx.chunk_length
    positions = torch.arange(chunk_length, dtype=torch.long)  # rank-local stand-in
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32)
    il_local = torch.tensor(
        [chunk_length // B] * B, dtype=torch.int32
    )  # rank-local lengths
    cu_seqlens[1:] = torch.cumsum(il_local, 0)
    return dict(
        x=torch.zeros(chunk_length, 4),
        positions=cp_ctx.prefix_length,  # int start (production passes an int)
        sp_per_req=cp_ctx.prefix_lengths.to(torch.int64),
        cu_seqlens=cu_seqlens,
        batch_size=B,
        input_lengths=il_local,
        prefix_lengths=cp_ctx.prefix_lengths.to(torch.int32),
        position_ids=cp_ctx.global_positions.to(torch.long),
        req_id_per_token=cp_ctx.req_id_per_token.to(torch.int32),
        max_seqlen_q=max_len,
    )


def _meta_fields(meta):
    """Flatten a PrefillMeta (incl. nested) into {path: tensor-or-scalar}."""
    out = {}

    def walk(prefix, value):
        if prefix == "m.cp_ctx":
            # The CPContext object identity/content is out of scope here (the
            # (a) suite covers context byte-identity); both arms receive the
            # same-shaped context by construction.
            out[prefix] = "cp_ctx"
            return
        if isinstance(value, torch.Tensor):
            out[prefix] = value
        elif hasattr(value, "_fields"):  # NamedTuple
            for k, v in zip(value._fields, value):
                walk(f"{prefix}.{k}", v)
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                walk(f"{prefix}[{i}]", v)
        else:
            out[prefix] = value

    walk("m", meta)
    return out


def _assert_meta_equal(tc, got, want, msg=""):
    g, w = _meta_fields(got), _meta_fields(want)
    tc.assertEqual(set(g), set(w), msg)
    for k in g:
        gv, wv = g[k], w[k]
        if isinstance(gv, torch.Tensor):
            tc.assertEqual(gv.dtype, wv.dtype, f"{k} {msg}")
            tc.assertTrue(torch.equal(gv, wv), f"{k} {msg}")
        elif isinstance(gv, torch.device):
            tc.assertEqual(str(gv), str(wv), f"{k} {msg}")
        else:
            tc.assertEqual(gv, wv, f"{k} {msg}")


MATRIX = [
    # (cp_rank, chunk_lengths, real_lengths, prefixes-to-test)
    (0, (1024,), (4096,), (0, 4096)),
    (2, (1024,), (4096,), (4096, 28672)),
    (1, (1024,), (3000,), (0,)),  # partial tail
    (3, (768, 256), (3072, 1024), (0, 4096)),  # B=2
]


class SharedBuildEndToEnd(unittest.TestCase):
    """Drive the real ``_build_shared_prefill_meta`` across the 3 buckets
    sharing one memo; compare against legacy per-bucket builds field-by-field.
    """

    def _run(self, cp_rank, chunk_lengths, real_lengths, prefix, shared_flag):
        ctx = _make_cp_ctx(cp_rank, chunk_lengths, real_lengths, prefix)
        stubs = _make_stubs(ctx)
        args = _forward_args(ctx, max(real_lengths))
        shared = {} if shared_flag else None
        metas = {}
        for r in (0, 4, 128):
            metas[r] = stubs[r]._build_shared_prefill_meta(
                args["x"],
                args["positions"],
                sp_per_req=args["sp_per_req"],
                cu_seqlens=args["cu_seqlens"],
                batch_size=args["batch_size"],
                input_lengths=args["input_lengths"],
                prefix_lengths=args["prefix_lengths"],
                position_ids=args["position_ids"],
                req_id_per_token=args["req_id_per_token"],
                max_seqlen_q=args["max_seqlen_q"],
                shared=shared,
            )
        return metas, stubs

    def test_byte_identity_all_buckets_over_matrix(self):
        for cp_rank, chunk_lengths, real_lengths, prefixes in MATRIX:
            for prefix in prefixes:
                with self.subTest(rank=cp_rank, rl=real_lengths, sp=prefix):
                    legacy, _ = self._run(
                        cp_rank, chunk_lengths, real_lengths, prefix, False
                    )
                    shared, _ = self._run(
                        cp_rank, chunk_lengths, real_lengths, prefix, True
                    )
                    for r in (0, 4, 128):
                        _assert_meta_equal(self, shared[r], legacy[r], msg=f"ratio={r}")

    def test_shared_objects_across_buckets(self):
        metas, _ = self._run(0, (1024,), (4096,), 4096, True)
        swa, csa, hca = metas[0], metas[4], metas[128]
        # SWA Group-1 tensors shared by all three buckets.
        self.assertIs(csa.swa_meta.slot_mapping, swa.swa_meta.slot_mapping)
        self.assertIs(hca.swa_meta.slot_mapping, swa.swa_meta.slot_mapping)
        self.assertIs(csa.swa_meta.query_start_loc, swa.swa_meta.query_start_loc)
        self.assertIs(csa.swa_meta.combined_seq_lens, swa.swa_meta.combined_seq_lens)
        # topk + row_seqlens shared.
        self.assertIs(csa.topk_idxs, swa.topk_idxs)
        self.assertIs(hca.topk_idxs, swa.topk_idxs)
        self.assertIs(csa.row_seqlens_full, swa.row_seqlens_full)
        # freqs: CSA and HCA share the compress-rope object in this fixture.
        self.assertIs(hca.freqs_cis, csa.freqs_cis)

    def test_op_count_reduction(self):
        ctx = _make_cp_ctx(0, (1024,), (4096,), 4096)

        def counted_run(shared_flag):
            stubs = _make_stubs(ctx)
            args = _forward_args(ctx, 4096)
            counts = {"topk": 0, "slot_map": 0}
            real_topk = swa_ops.compute_window_topk_and_length_varlen
            real_slot = swa_ops.compute_swa_slot_mapping

            def c_topk(*a, **k):
                counts["topk"] += 1
                return real_topk(*a, **k)

            def c_slot(*a, **k):
                counts["slot_map"] += 1
                return real_slot(*a, **k)

            shared = {} if shared_flag else None
            with patch.object(swa_ops, "compute_window_topk_and_length_varlen", c_topk):
                with patch.object(swa_ops, "compute_swa_slot_mapping", c_slot):
                    for r in (0, 4, 128):
                        stubs[r]._build_shared_prefill_meta(
                            args["x"],
                            args["positions"],
                            sp_per_req=args["sp_per_req"],
                            cu_seqlens=args["cu_seqlens"],
                            batch_size=args["batch_size"],
                            input_lengths=args["input_lengths"],
                            prefix_lengths=args["prefix_lengths"],
                            position_ids=args["position_ids"],
                            req_id_per_token=args["req_id_per_token"],
                            max_seqlen_q=args["max_seqlen_q"],
                            shared=shared,
                        )
            return counts

        legacy = counted_run(False)
        shared = counted_run(True)
        self.assertEqual(legacy["topk"], 3)
        self.assertEqual(legacy["slot_map"], 3)
        self.assertEqual(shared["topk"], 1)
        self.assertEqual(shared["slot_map"], 1)

    def test_probe_mismatch_rebuilds_with_new_values(self):
        # A NEW forward (different prefix -> different cp_ctx) reusing the same
        # memo dict must NOT hit: the id(cp_ctx) probe component rejects the
        # stale entries and the rebuild produces the new forward's values.
        ctx_a = _make_cp_ctx(0, (1024,), (4096,), 0)
        ctx_b = _make_cp_ctx(0, (1024,), (4096,), 4096)
        shared: dict = {}
        stubs_a = _make_stubs(ctx_a)
        args_a = _forward_args(ctx_a, 4096)
        meta_a = stubs_a[4]._build_shared_prefill_meta(
            args_a["x"],
            args_a["positions"],
            sp_per_req=args_a["sp_per_req"],
            cu_seqlens=args_a["cu_seqlens"],
            batch_size=args_a["batch_size"],
            input_lengths=args_a["input_lengths"],
            prefix_lengths=args_a["prefix_lengths"],
            position_ids=args_a["position_ids"],
            req_id_per_token=args_a["req_id_per_token"],
            max_seqlen_q=args_a["max_seqlen_q"],
            shared=shared,
        )
        stubs_b = _make_stubs(ctx_b)
        args_b = _forward_args(ctx_b, 4096)
        meta_b = stubs_b[4]._build_shared_prefill_meta(
            args_b["x"],
            args_b["positions"],
            sp_per_req=args_b["sp_per_req"],
            cu_seqlens=args_b["cu_seqlens"],
            batch_size=args_b["batch_size"],
            input_lengths=args_b["input_lengths"],
            prefix_lengths=args_b["prefix_lengths"],
            position_ids=args_b["position_ids"],
            req_id_per_token=args_b["req_id_per_token"],
            max_seqlen_q=args_b["max_seqlen_q"],
            shared=shared,
        )
        # Rebuilt (not the stale objects), and values match a fresh legacy build.
        self.assertIsNot(meta_a.topk_idxs, meta_b.topk_idxs)
        legacy_b, _ = self._run(0, (1024,), (4096,), 4096, False)
        _assert_meta_equal(self, meta_b, legacy_b[4], msg="rebuilt-after-mismatch")

    def test_flag_off_distinct_objects_equal_values(self):
        metas, _ = self._run(2, (768, 256), (3072, 1024), 4096, False)
        self.assertIsNot(metas[0].topk_idxs, metas[4].topk_idxs)
        self.assertTrue(torch.equal(metas[0].topk_idxs, metas[4].topk_idxs))
        self.assertIsNot(metas[0].swa_meta.slot_mapping, metas[4].swa_meta.slot_mapping)
        self.assertTrue(
            torch.equal(metas[0].swa_meta.slot_mapping, metas[4].swa_meta.slot_mapping)
        )


class WorkspaceMetaSharing(unittest.TestCase):
    """Drive the REAL ``_build_workspace_meta`` twice (CSA then HCA ratio)
    over one shared memo."""

    def _build(self, stub, args, shared, dense):
        return stub._build_workspace_meta(
            args["x"].shape[0],
            int(args["positions"]),
            torch.device("cpu"),
            dense,
            use_varlen=True,
            batch_size=args["batch_size"],
            cu_seqlens=args["cu_seqlens"],
            input_lengths=args["input_lengths"],
            prefix_lengths=args["prefix_lengths"],
            sp_per_req=args["sp_per_req"],
            position_ids=args["position_ids"],
            req_id_per_token=args["req_id_per_token"],
            max_seqlen_q=args["max_seqlen_q"],
            swa_slot_mapping=None,
            shared=shared,
        )

    def test_default_kwarg_is_legacy(self):
        # The pre-existing bazel suites call these builders WITHOUT the new
        # ``shared`` kwarg — pin that the default is the legacy behavior.
        ctx = _make_cp_ctx(1, (1024,), (4096,), 8192)
        stubs = _make_stubs(ctx, ratios=(4,))
        args = _forward_args(ctx, 4096)
        # No ``shared`` kwarg at all (the legacy call shape).
        default_built = stubs[4]._build_workspace_meta(
            args["x"].shape[0],
            int(args["positions"]),
            torch.device("cpu"),
            False,
            use_varlen=True,
            batch_size=args["batch_size"],
            cu_seqlens=args["cu_seqlens"],
            input_lengths=args["input_lengths"],
            prefix_lengths=args["prefix_lengths"],
            sp_per_req=args["sp_per_req"],
            position_ids=args["position_ids"],
            req_id_per_token=args["req_id_per_token"],
            max_seqlen_q=args["max_seqlen_q"],
            swa_slot_mapping=None,
        )
        explicit_none = self._build(stubs[4], args, None, False)
        _assert_meta_equal(self, default_built, explicit_none, msg="default-kwarg")

    def test_ws_swa_shared_and_byte_identical(self):
        ctx = _make_cp_ctx(1, (1024,), (4096,), 8192)
        stubs = _make_stubs(ctx, ratios=(4, 128))
        args = _forward_args(ctx, 4096)
        shared: dict = {}
        csa = self._build(stubs[4], args, shared, False)
        hca = self._build(stubs[128], args, shared, True)
        self.assertIsNotNone(csa)
        self.assertIsNotNone(hca)
        # The SWA half is shared (same objects) across buckets...
        self.assertIs(hca.swa_cache_slot_mapping, csa.swa_cache_slot_mapping)
        self.assertIs(hca.swa_seq_lens, csa.swa_seq_lens)
        self.assertIs(hca.swa_bt_int32, csa.swa_bt_int32)
        self.assertIs(hca.qsl, csa.qsl)
        self.assertIs(hca.swa_cache_gather_lens, csa.swa_cache_gather_lens)
        # ...while the ratio-specific halves differ.
        self.assertIsNot(hca.cmp_seq_lens, csa.cmp_seq_lens)
        # Ratio sanity: N = (prefix + len) // ratio — CSA vs HCA.
        self.assertEqual(csa.N, (8192 + 4096) // 4)
        self.assertEqual(hca.N, (8192 + 4096) // 128)
        self.assertNotEqual(csa.N, hca.N)
        self.assertIsNotNone(hca.dense_cmp_topk)
        self.assertIsNone(csa.dense_cmp_topk)
        # Legacy comparison per ratio.
        legacy_csa = self._build(stubs[4], args, None, False)
        legacy_hca = self._build(stubs[128], args, None, True)
        _assert_meta_equal(self, csa, legacy_csa, msg="csa")
        _assert_meta_equal(self, hca, legacy_hca, msg="hca")

    def test_ws_suffix_readback_gone_under_ban(self):
        # The mirror-derived max_gather override + the memo keep the whole
        # second-bucket build free of .item()/.cpu()/.tolist() calls.
        ctx = _make_cp_ctx(1, (1024,), (4096,), 8192)
        stubs = _make_stubs(ctx, ratios=(4, 128))
        args = _forward_args(ctx, 4096)
        shared: dict = {}
        self._build(stubs[4], args, shared, False)

        def banned(*a, **k):
            raise AssertionError("host sync on hot path")

        with patch.object(torch.Tensor, "item", banned):
            with patch.object(torch.Tensor, "cpu", banned):
                with patch.object(torch.Tensor, "tolist", banned):
                    self._build(stubs[128], args, shared, True)

    def test_probe_mismatch_rebuilds(self):
        ctx_a = _make_cp_ctx(1, (1024,), (4096,), 8192)
        ctx_b = _make_cp_ctx(1, (1024,), (4096,), 12288)
        args_a = _forward_args(ctx_a, 4096)
        args_b = _forward_args(ctx_b, 4096)
        stubs_a = _make_stubs(ctx_a, ratios=(4,))
        stubs_b = _make_stubs(ctx_b, ratios=(4,))
        shared: dict = {}
        meta_a = self._build(stubs_a[4], args_a, shared, False)
        meta_b = self._build(stubs_b[4], args_b, shared, False)
        # Different prefix -> different values; the memo must have missed.
        self.assertIsNot(meta_a.swa_cache_slot_mapping, meta_b.swa_cache_slot_mapping)
        legacy_b = self._build(stubs_b[4], args_b, None, False)
        _assert_meta_equal(self, meta_b, legacy_b, msg="ws rebuild")

    def test_probe_stripped_mutant_stale_reuse_detected(self):
        # Mutant the ws_swa memo to skip the probe check: stale reuse across
        # two different forwards must produce observably wrong values — this
        # is why the probe exists.
        fn_src = ast.get_source_segment(
            ATTN_SRC,
            next(
                m
                for n in ast.parse(ATTN_SRC).body
                if isinstance(n, ast.ClassDef) and n.name == "AttentionFP8"
                for m in n.body
                if isinstance(m, ast.FunctionDef) and m.name == "_build_workspace_meta"
            ),
        )
        mutant_src = fn_src.replace(
            "if _ent is not None and _ent[0] == ws_probe:\n                    bundle = _ent[1]",
            "if _ent is not None:\n                    bundle = _ent[1]",
        )
        assert mutant_src != fn_src, "mutant edit did not apply"
        env = dict(ATT)  # reuse the extracted env (has all globals)
        exec(mutant_src, env)
        mutant = env["_build_workspace_meta"]

        ctx_a = _make_cp_ctx(1, (1024,), (4096,), 8192)
        ctx_b = _make_cp_ctx(1, (1024,), (4096,), 12288)
        stubs_a = _make_stubs(ctx_a, ratios=(4,))
        stubs_b = _make_stubs(ctx_b, ratios=(4,))
        # Bind the mutant on the B stub (instance attribute + explicit self).
        stubs_b[4]._build_workspace_meta = types.MethodType(mutant, stubs_b[4])
        args_a = _forward_args(ctx_a, 4096)
        args_b = _forward_args(ctx_b, 4096)
        shared: dict = {}
        self._build(stubs_a[4], args_a, shared, False)
        meta_b = self._build(stubs_b[4], args_b, shared, False)
        legacy_b = self._build(stubs_b[4], args_b, None, False)
        # The mutant stale-reused A's bundle: B's values are observably wrong.
        self.assertFalse(
            torch.equal(meta_b.swa_seq_lens, legacy_b.swa_seq_lens)
            and torch.equal(
                meta_b.swa_cache_slot_mapping, legacy_b.swa_cache_slot_mapping
            )
        )
        # And the REAL (probe-checked) build matches legacy exactly.
        fresh_shared: dict = {}
        stubs_b2 = _make_stubs(ctx_b, ratios=(4,))
        real_b = self._build(stubs_b2[4], args_b, fresh_shared, False)
        _assert_meta_equal(self, real_b, legacy_b, msg="real-after-mutant-check")


class CpFullPositionsMemo(unittest.TestCase):
    def test_shared_helper_memoizes_per_forward(self):
        ctx = _make_cp_ctx(2, (1024,), (4096,), 4096)
        shared: dict = {}
        calls = {"n": 0}
        real = cp.build_cp_full_prefill_positions

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        with patch.object(cp, "build_cp_full_prefill_positions", counting):
            a = cp.build_cp_full_prefill_positions_shared(
                ctx, torch.device("cpu"), shared
            )
            b = cp.build_cp_full_prefill_positions_shared(
                ctx, torch.device("cpu"), shared
            )
            c = cp.build_cp_full_prefill_positions_shared(
                ctx, torch.device("cpu"), shared
            )
        self.assertEqual(calls["n"], 1)
        self.assertIs(a, b)
        self.assertIs(b, c)
        direct = cp.build_cp_full_prefill_positions(ctx, torch.device("cpu"))
        for x, y in zip(a, direct):
            self.assertTrue(torch.equal(x, y))

    def test_none_shared_is_legacy(self):
        ctx = _make_cp_ctx(2, (1024,), (4096,), 4096)
        a = cp.build_cp_full_prefill_positions_shared(ctx, torch.device("cpu"), None)
        b = cp.build_cp_full_prefill_positions_shared(ctx, torch.device("cpu"), None)
        self.assertIsNot(a, b)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))

    def test_new_ctx_rebuilds(self):
        ctx_a = _make_cp_ctx(2, (1024,), (4096,), 4096)
        ctx_b = _make_cp_ctx(2, (1024,), (4096,), 8192)
        shared: dict = {}
        a = cp.build_cp_full_prefill_positions_shared(
            ctx_a, torch.device("cpu"), shared
        )
        b = cp.build_cp_full_prefill_positions_shared(
            ctx_b, torch.device("cpu"), shared
        )
        self.assertIsNot(a, b)
        self.assertFalse(torch.equal(a[0], b[0]))  # positions shift by prefix
        direct_b = cp.build_cp_full_prefill_positions(ctx_b, torch.device("cpu"))
        for x, y in zip(b, direct_b):
            self.assertTrue(torch.equal(x, y))


class PrefillMetaFlag(unittest.TestCase):
    def test_flag_parse_fail_closed(self):
        with patch.dict(os.environ, {FLAG: "yes"}):
            with self.assertRaises(ValueError):
                prefill_meta._meta_shared_build_enabled()
        with patch.dict(os.environ, {FLAG: "0"}):
            self.assertFalse(prefill_meta._meta_shared_build_enabled())
        with patch.dict(os.environ, {FLAG: "1"}):
            self.assertTrue(prefill_meta._meta_shared_build_enabled())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(FLAG, None)
            self.assertTrue(
                prefill_meta._meta_shared_build_enabled()
            )  # default ON (qualified)

    def test_propagate_passes_shared_only_when_on(self):
        seen = []

        class _Attn:
            compress_ratio = 4

            def _build_shared_prefill_meta(self, x, start_pos, **kwargs):
                seen.append(kwargs.get("shared", "ABSENT"))
                m = ATT["PrefillMeta"](
                    seqlen=1,
                    seqlen_full=1,
                    rd=64,
                    device=torch.device("cpu"),
                    cp_ctx=None,
                    cp_on=False,
                    freqs_cis=torch.zeros(1, 8),
                    topk_idxs=torch.zeros(1, 4, dtype=torch.int32),
                    sp_int=0,
                    any_cont=False,
                    row_seqlens_full=torch.ones(1, dtype=torch.long),
                    use_varlen=True,
                    sp_per_req=None,
                    cu_seqlens=None,
                    batch_size=1,
                    input_lengths=None,
                    prefix_lengths=None,
                    position_ids=None,
                    req_id_per_token=None,
                    max_seqlen_q=1,
                    swa_meta=None,
                    csa_meta=None,
                    hca_meta=None,
                )
                return m

            def _ensure_freqs_cis_bound(self):
                return None

            def _set_prefill_meta_shared(self, meta):
                self.meta = meta

        class _Layer:
            attn = _Attn()

        class _V4:
            layers = [_Layer()]

        for flag_value, want_shared in (("1", True), ("0", False)):
            seen.clear()
            with patch.dict(os.environ, {FLAG: flag_value}):
                prefill_meta.build_and_propagate_prefill_meta_fp8(
                    _V4(),
                    torch.zeros(4, 4),
                    0,
                    None,
                    None,
                    sp_per_req=None,
                    cu_seqlens=None,
                    batch_size=1,
                    input_lengths=None,
                    prefix_lengths=None,
                    position_ids=None,
                    req_id_per_token=None,
                    max_seqlen_q=1,
                    workspace=None,
                )
            self.assertEqual(len(seen), 1)
            if want_shared:
                self.assertIsInstance(seen[0], dict)
            else:
                self.assertIsNone(seen[0])


class WorkspaceMetaSm120Paged(unittest.TestCase):
    """The SM120 direct-paged block (``DSV4_SM120_PAGED_CACHE_PREFILL=1``)
    engages ``swa_pool_slot_mapping`` / ``cmp_pool_slot_mapping`` — the latter
    consumes the ``max_gather_host=N_max`` override.  Driven twice (CSA
    then HCA) over one shared memo; byte-identity vs legacy per ratio."""

    def setUp(self):
        self._old = _IS_SM120[0]
        _IS_SM120[0] = True
        self._env = patch.dict(os.environ, {"DSV4_SM120_PAGED_CACHE_PREFILL": "1"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        _IS_SM120[0] = self._old

    def test_paged_path_shared_and_byte_identical(self):
        ctx = _make_cp_ctx(1, (1024,), (4096,), 8192)
        stubs = _make_stubs(ctx, ratios=(4, 128))
        args = _forward_args(ctx, 4096)
        # swa_slot_mapping arg: what the callers pass from the SWA meta — a
        # per-token int64 write map (content inert for the suffix builders).
        swa_slot_mapping = torch.arange(ctx.seq_len_full, dtype=torch.long).contiguous()

        def build(stub, shared, dense):
            return stub._build_workspace_meta(
                args["x"].shape[0],
                int(args["positions"]),
                torch.device("cpu"),
                dense,
                use_varlen=True,
                batch_size=args["batch_size"],
                cu_seqlens=args["cu_seqlens"],
                input_lengths=args["input_lengths"],
                prefix_lengths=args["prefix_lengths"],
                sp_per_req=args["sp_per_req"],
                position_ids=args["position_ids"],
                req_id_per_token=args["req_id_per_token"],
                max_seqlen_q=args["max_seqlen_q"],
                swa_slot_mapping=swa_slot_mapping,
                shared=shared,
            )

        shared: dict = {}
        csa = build(stubs[4], shared, False)
        hca = build(stubs[128], shared, True)
        # Paged block engaged.
        self.assertIsNotNone(csa.cmp_pool_slot_mapping)
        self.assertIsNotNone(hca.cmp_pool_slot_mapping)
        self.assertIsNotNone(csa.swa_pool_slot_mapping)
        # Ratio-specific cmp pools differ.
        self.assertIsNot(csa.cmp_pool_slot_mapping, hca.cmp_pool_slot_mapping)
        # Shared SWA half (incl. the suffix read) stays shared.
        self.assertIs(hca.swa_cache_slot_mapping, csa.swa_cache_slot_mapping)
        # Byte-identity vs legacy per ratio.
        legacy_csa = build(stubs[4], None, False)
        legacy_hca = build(stubs[128], None, True)
        _assert_meta_equal(self, csa, legacy_csa, msg="csa-paged")
        _assert_meta_equal(self, hca, legacy_hca, msg="hca-paged")

    def test_paged_suffix_readbacks_gone_under_ban(self):
        # Under the mirror path the whole build (incl. the cmp_pool suffix
        # max via the N_max reuse) issues no .item()/.cpu()/.tolist().
        ctx = _make_cp_ctx(1, (1024,), (4096,), 8192)
        stubs = _make_stubs(ctx, ratios=(4, 128))
        args = _forward_args(ctx, 4096)
        swa_slot_mapping = torch.arange(ctx.seq_len_full, dtype=torch.long).contiguous()

        def banned(*a, **k):
            raise AssertionError("host sync on hot path")

        with patch.object(torch.Tensor, "item", banned):
            with patch.object(torch.Tensor, "cpu", banned):
                with patch.object(torch.Tensor, "tolist", banned):
                    for ratio, dense in ((4, False), (128, True)):
                        stubs[ratio]._build_workspace_meta(
                            args["x"].shape[0],
                            int(args["positions"]),
                            torch.device("cpu"),
                            dense,
                            use_varlen=True,
                            batch_size=args["batch_size"],
                            cu_seqlens=args["cu_seqlens"],
                            input_lengths=args["input_lengths"],
                            prefix_lengths=args["prefix_lengths"],
                            sp_per_req=args["sp_per_req"],
                            position_ids=args["position_ids"],
                            req_id_per_token=args["req_id_per_token"],
                            max_seqlen_q=args["max_seqlen_q"],
                            swa_slot_mapping=swa_slot_mapping,
                            shared={},
                        )


class SharedBuildFreqsDistinct(unittest.TestCase):
    """SWA bucket binds a different rope params object (base rope_theta) than
    the CSA/HCA buckets (shared compress-rope object): the freqs memo must
    miss for SWA<->CSA (different data_ptr) but hit CSA->HCA; values must stay
    byte-identical to legacy per-bucket builds either way."""

    def test_freqs_memo_selective_sharing(self):
        ctx = _make_cp_ctx(0, (1024,), (4096,), 4096)
        stubs = _make_stubs(ctx, shared_freqs=False)
        args = _forward_args(ctx, 4096)
        shared: dict = {}
        metas = {}
        for r in (0, 4, 128):
            metas[r] = stubs[r]._build_shared_prefill_meta(
                args["x"],
                args["positions"],
                sp_per_req=args["sp_per_req"],
                cu_seqlens=args["cu_seqlens"],
                batch_size=args["batch_size"],
                input_lengths=args["input_lengths"],
                prefix_lengths=args["prefix_lengths"],
                position_ids=args["position_ids"],
                req_id_per_token=args["req_id_per_token"],
                max_seqlen_q=args["max_seqlen_q"],
                shared=shared,
            )
        # CSA and HCA share the freqs gather object.
        self.assertIs(metas[4].freqs_cis, metas[128].freqs_cis)
        # SWA rebuilt its own (different rope params object).
        self.assertIsNot(metas[0].freqs_cis, metas[4].freqs_cis)
        # Byte-identity vs legacy.
        for r in (0, 4, 128):
            legacy = stubs[r]._build_shared_prefill_meta(
                args["x"],
                args["positions"],
                sp_per_req=args["sp_per_req"],
                cu_seqlens=args["cu_seqlens"],
                batch_size=args["batch_size"],
                input_lengths=args["input_lengths"],
                prefix_lengths=args["prefix_lengths"],
                position_ids=args["position_ids"],
                req_id_per_token=args["req_id_per_token"],
                max_seqlen_q=args["max_seqlen_q"],
                shared=None,
            )
            _assert_meta_equal(self, metas[r], legacy, msg=f"ratio={r}")


class CsaHcaBuilderThreading(unittest.TestCase):
    """Drive the REAL ``_build_csa_prefill_meta`` / ``_build_hca_prefill_meta``
    (extracted) with recording indexer/compressor stubs: proves the
    ``cp_full_positions`` memo is shared across the CSA compressor meta, the
    HCA compressor meta, and the nested indexer-compressor build; that the
    ``shared`` kwarg reaches ``indexer.prepare`` only when enabled; and that
    both bucket metas are byte-identical to the legacy (shared=None) builds.
    """

    def setUp(self):
        # Stub the heavy fp8 submodules the builders import locally.
        self._saved = {}
        indexer_stub = types.ModuleType(_PREFIX + ".fp8.indexer")
        compressor_stub = types.ModuleType(_PREFIX + ".fp8.compressor")

        class _IndexerBase:  # isinstance target
            pass

        indexer_stub.IndexerFP8 = _IndexerBase
        compressor_stub.build_prepare_metadata_args = lambda **k: dict(k)
        compressor_stub.CompressorFP8 = object
        compressor_stub.CompressorMeta = object
        for name, mod in (
            (_PREFIX + ".fp8.indexer", indexer_stub),
            (_PREFIX + ".fp8.compressor", compressor_stub),
        ):
            self._saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
        self._IndexerBase = _IndexerBase

    def tearDown(self):
        for name, saved in self._saved.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    class _RecordingIndexer:
        def __init__(self, base_cls):
            self.calls = []
            self._base = base_cls

        def prepare(self, *a, **k):
            self.calls.append(dict(kwargs=k))
            # Mimic the real IndexerFP8.prepare: the nested compressor meta
            # consumes the shared CP positions helper under CP.
            shared = k.get("shared")
            cp_ctx = getattr(self, "_cp_ctx", None)
            if cp_ctx is not None:
                cp.build_cp_full_prefill_positions_shared(
                    cp_ctx, torch.device("cpu"), shared
                )
            return ("indexer_meta",)

    def _make_csa_stub(self, ctx, ratio):
        outer = self

        class _Stub(_StubAttention):
            _build_csa_prefill_meta = ATT["_build_csa_prefill_meta"]
            _build_hca_prefill_meta = ATT["_build_hca_prefill_meta"]
            _build_compressor_meta = ATT["_build_compressor_meta"]

            def _set_compressor_pool_context(self):
                return None

            def _clear_compressor_pool_context(self):
                return None

        stub = _Stub(
            compress_ratio=ratio,
            freqs_cis=torch.zeros(33000, 8),
            cp_ctx=ctx,
            block_tables={
                kv_cache_utils.SWA_KV: torch.arange(1, 7, dtype=torch.int32).view(1, 6),
                kv_cache_utils.CSA_KV: torch.arange(1, 7, dtype=torch.int32).view(1, 6),
                kv_cache_utils.HCA_KV: torch.arange(1, 7, dtype=torch.int32).view(1, 6),
                kv_cache_utils.INDEXER_KV: torch.arange(1, 7, dtype=torch.int32).view(
                    1, 6
                ),
            },
            eb_by_type={
                kv_cache_utils.SWA_KV: 32,
                kv_cache_utils.CSA_KV: 8,
                kv_cache_utils.HCA_KV: 8,
                kv_cache_utils.INDEXER_KV: 8,
            },
        )

        class _Indexer(outer._IndexerBase, CsaHcaBuilderThreading._RecordingIndexer):
            def __init__(self):
                CsaHcaBuilderThreading._RecordingIndexer.__init__(
                    self, outer._IndexerBase
                )
                self._cp_ctx = ctx

        class _Compressor:
            def __init__(self):
                self.calls = []

            def prepare_metadata(self, positions, b_idx, **k):
                self.calls.append((positions, b_idx, k))
                return ("compressor_meta", positions, b_idx)

        stub.indexer = _Indexer()
        stub.compressor = _Compressor()
        return stub

    def _call(self, stub, args, shared):
        if int(stub.compress_ratio) == 4:
            return stub._build_csa_prefill_meta(
                args["x"].shape[0],
                int(args["positions"]),
                torch.device("cpu"),
                use_varlen=True,
                batch_size=args["batch_size"],
                cu_seqlens=args["cu_seqlens"],
                input_lengths=args["input_lengths"],
                prefix_lengths=args["prefix_lengths"],
                sp_per_req=args["sp_per_req"],
                position_ids=args["position_ids"],
                req_id_per_token=args["req_id_per_token"],
                max_seqlen_q=args["max_seqlen_q"],
                has_prefix=bool(int(args["positions"]) > 0),
                swa_slot_mapping=None,
                shared=shared,
            )
        return stub._build_hca_prefill_meta(
            args["x"].shape[0],
            int(args["positions"]),
            torch.device("cpu"),
            use_varlen=True,
            batch_size=args["batch_size"],
            cu_seqlens=args["cu_seqlens"],
            input_lengths=args["input_lengths"],
            prefix_lengths=args["prefix_lengths"],
            sp_per_req=args["sp_per_req"],
            position_ids=args["position_ids"],
            req_id_per_token=args["req_id_per_token"],
            max_seqlen_q=args["max_seqlen_q"],
            has_prefix=bool(int(args["positions"]) > 0),
            swa_slot_mapping=None,
            shared=shared,
        )

    def test_cp_positions_built_once_across_three_consumers(self):
        ctx = _make_cp_ctx(1, (1024,), (4096,), 4096)
        args = _forward_args(ctx, 4096)
        csa = self._make_csa_stub(ctx, 4)
        hca = self._make_csa_stub(ctx, 128)
        shared: dict = {}
        calls = {"n": 0}
        real = cp.build_cp_full_prefill_positions

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        with patch.object(cp, "build_cp_full_prefill_positions", counting):
            meta_csa = self._call(csa, args, shared)
            meta_hca = self._call(hca, args, shared)
        # 3 consumers (CSA compressor meta, CSA nested indexer meta, HCA
        # compressor meta), ONE real build.
        self.assertEqual(calls["n"], 1)
        # The indexer saw the shared kwarg.
        self.assertIs(csa.indexer.calls[0]["kwargs"].get("shared"), shared)
        # The shared positions object reached every consumer.
        pos = shared["cp_full_pos"][1][0]
        self.assertIs(csa.compressor.calls[0][0], pos)
        self.assertIs(hca.compressor.calls[0][0], pos)
        self.assertIsNotNone(meta_csa.workspace_meta)
        self.assertIsNotNone(meta_hca.workspace_meta)

    def test_legacy_builds_byte_identical(self):
        ctx = _make_cp_ctx(1, (1024,), (4096,), 4096)
        args = _forward_args(ctx, 4096)

        def run(shared):
            csa = self._make_csa_stub(ctx, 4)
            hca = self._make_csa_stub(ctx, 128)
            m_csa = self._call(csa, args, shared)
            m_hca = self._call(hca, args, shared)
            return m_csa, m_hca, csa, hca

        s_csa, s_hca, s_csa_stub, s_hca_stub = run({})
        l_csa, l_hca, l_csa_stub, l_hca_stub = run(None)
        # Whole-metas byte equal.
        _assert_meta_equal(self, s_csa, l_csa, msg="csa")
        _assert_meta_equal(self, s_hca, l_hca, msg="hca")
        # The compressor received identical positions by value (shared: same
        # object; legacy: rebuilt equal values).
        self.assertTrue(
            torch.equal(
                s_csa_stub.compressor.calls[0][0], l_csa_stub.compressor.calls[0][0]
            )
        )
        self.assertTrue(
            torch.equal(
                s_hca_stub.compressor.calls[0][0], l_hca_stub.compressor.calls[0][0]
            )
        )
        # Legacy path: the indexer got no shared kwarg.
        self.assertNotIn("shared", l_csa_stub.indexer.calls[0]["kwargs"])

    def test_probe_mismatch_rebuilds_positions(self):
        ctx_a = _make_cp_ctx(1, (1024,), (4096,), 4096)
        ctx_b = _make_cp_ctx(1, (1024,), (4096,), 8192)
        args_a = _forward_args(ctx_a, 4096)
        args_b = _forward_args(ctx_b, 4096)
        shared: dict = {}
        stub_a = self._make_csa_stub(ctx_a, 4)
        stub_b = self._make_csa_stub(ctx_b, 4)
        self._call(stub_a, args_a, shared)
        self._call(stub_b, args_b, shared)
        pa = stub_a.compressor.calls[0][0]
        pb = stub_b.compressor.calls[0][0]
        self.assertIsNot(pa, pb)
        self.assertFalse(torch.equal(pa, pb))  # prefix differs
        direct_b = cp.build_cp_full_prefill_positions(ctx_b, torch.device("cpu"))
        self.assertTrue(torch.equal(pb, direct_b[0]))


class SourceInvariants(unittest.TestCase):
    def test_flag_wiring(self):
        self.assertIn(
            '_META_SHARED_BUILD_FLAG = "DSV4_FP8_PREFILL_META_SHARED_BUILD"',
            PREFILL_META_PATH.read_text(),
        )
        self.assertIn(
            'os.environ.get(_META_SHARED_BUILD_FLAG, "1")',
            PREFILL_META_PATH.read_text(),
        )
        self.assertIn("shared=shared,", PREFILL_META_PATH.read_text())

    def test_memo_points_exist(self):
        for marker in (
            '"freqs_topk"',
            '"row_seqlens_full"',
            '"swa_g1"',
            '"ws_swa"',
            '"ws_swa_cache_map"',
            '"cp_full_pos"',
        ):
            self.assertIn(marker, ATTN_SRC + CP_SRC)
        # All builder signatures gained the optional shared kwarg.
        for sig in ("max_seqlen_q: int = 0,\n        shared: Optional[dict] = None,",):
            self.assertIn(sig, ATTN_SRC)
        self.assertIn(
            "shared: Optional[Any] = None,", (DSV4 / "fp8" / "indexer.py").read_text()
        )

    def test_legacy_call_sites_still_available(self):
        # The plain builder is still exported and the shared helper defaults
        # to the legacy build.
        self.assertIn("def build_cp_full_prefill_positions(", CP_SRC)
        self.assertIn("def build_cp_full_prefill_positions_shared(", CP_SRC)
        self.assertIn(
            "if shared is None:\n        return build_cp_full_prefill_positions(cp_ctx, device)",
            CP_SRC,
        )


def setUpModule():
    global ATT, _FP8_PKG, _cp_attention_shard, _profiler, _swa_cp_byte_sliced, cp
    global fp8_kv_utils, kv_cache_utils, prefill_meta, swa_ops
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()

    cp, swa_ops, prefill_meta, _FP8_PKG, fp8_kv_utils = _load_dsv4_tree()

    kv_cache_utils = sys.modules[_PREFIX + ".kv_cache_utils"]

    _profiler = sys.modules[_PREFIX + "._profiler"]

    _cp_attention_shard = sys.modules[_PREFIX + ".fp8._cp_attention_shard"]

    _swa_cp_byte_sliced = sys.modules[_PREFIX + ".fp8._swa_cp_byte_sliced"]

    ATT = _extract_attn()

    _StubKvCache.group_tags = [
        kv_cache_utils.SWA_KV,
        kv_cache_utils.CSA_KV,
        kv_cache_utils.HCA_KV,
    ]

    _StubAttention._build_workspace_meta = ATT["_build_workspace_meta"]

    _StubAttention._build_swa_prefill_meta_varlen = ATT[
        "_build_swa_prefill_meta_varlen"
    ]

    _StubAttention._build_shared_prefill_meta = ATT["_build_shared_prefill_meta"]

    _StubAttention._build_swa_cp_byte_compaction = ATT["_build_swa_cp_byte_compaction"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
