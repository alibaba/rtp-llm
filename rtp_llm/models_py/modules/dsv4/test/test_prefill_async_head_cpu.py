"""CPU regression tests for asynchronous prefill-head preparation.

Execute the metadata builders with CPU Triton kernels and compare consumed,
patched bundles with the eager path. Cover prediction rejection, stream fences,
caller inference-mode propagation and the default-on flag's kill switch.
These tests do not establish hardware stream ordering or model numerics."""

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
FP8 = DSV4 / "fp8"
ATTN_PATH = FP8 / "attention.py"
ATTN_SRC = ATTN_PATH.read_text()
CP_PATH = DSV4 / "cp.py"
INDEXER_PATH = FP8 / "indexer.py"
INDEXER_SRC = INDEXER_PATH.read_text()
COMPRESSOR_PATH = FP8 / "compressor.py"
COMPRESSOR_SRC = COMPRESSOR_PATH.read_text()
HEADPREBUILD_PATH = FP8 / "_head_prebuild.py"
PREFILL_META_PATH = FP8 / "prefill_meta.py"

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
    fp8 = _package(_PREFIX + ".fp8", str(FP8))
    dist = _package("rtp_llm.models_py.distributed")
    collective = types.ModuleType("rtp_llm.models_py.distributed.collective_torch")

    class _Group:
        TP = "TP"

    collective.Group = _Group
    collective.all_gather = lambda *a, **k: None
    collective._get_group = lambda g: g
    sys.modules[collective.__name__] = collective
    dist.collective_torch = collective
    # Real leaf modules (light deps only).
    _load(_PREFIX + "._profiler", DSV4 / "_profiler.py")
    cp = _load(_PREFIX + ".cp", CP_PATH)
    kvu = _load(_PREFIX + ".kv_cache_utils", DSV4 / "kv_cache_utils.py")
    _load(_PREFIX + ".fp8._trap_utils", FP8 / "_trap_utils.py")
    _load(_PREFIX + ".fp8._swa_cp_byte_sliced", FP8 / "_swa_cp_byte_sliced.py")
    _load(_PREFIX + ".fp8._cp_attention_shard", FP8 / "_cp_attention_shard.py")
    swa_ops = _load(_PREFIX + ".fp8._swa_ops_triton", FP8 / "_swa_ops_triton.py")
    fp8_kv_utils = _load(_PREFIX + ".fp8._kv_cache_utils", FP8 / "_kv_cache_utils.py")
    _load(_PREFIX + ".fp8._cp_packed_rows", FP8 / "_cp_packed_rows.py")
    compact_rt = _load(
        _PREFIX + ".fp8._compact_cp_runtime", FP8 / "_compact_cp_runtime.py"
    )
    fused_meta = _load(
        _PREFIX + ".fp8._fused_compressor_meta_triton",
        FP8 / "_fused_compressor_meta_triton.py",
    )
    return cp, kvu, swa_ops, fp8_kv_utils, compact_rt, fused_meta, fp8


# ---------------------------------------------------------------------------
# AST extraction: attention.py helpers + builders, indexer prepare, compressor meta
# ---------------------------------------------------------------------------
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
        "_build_suffix_cp_sliced_slot_mapping",
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
        "is_sm120": lambda device=None: False,
        "is_sm12x": lambda: False,
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


class _LocalPoolReaderStub:
    """Stateless strategy object: equal by type for the meta comparators."""

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
    assert not kv_cache_sharded, "sharded reader is out of scope for the CPU test"
    return _LocalPoolReaderStub()


def _extract_indexer():
    tree = ast.parse(INDEXER_SRC)
    want_classes = {"_IndexerFP8PrefillMeta"}
    want_methods = {"prepare"}
    want_funcs = {"_compressed_k_scalars_host", "_flat_1d"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in want_classes:
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in want_funcs:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "IndexerFP8":
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name in want_methods:
                    body.append(m)
    names = {n.name for n in body}
    assert (want_classes | want_methods | want_funcs) == names
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    mod = ast.fix_missing_locations(ast.Module(body=[future] + body, type_ignores=[]))
    import os as _os

    env = {
        "torch": torch,
        "os": _os,
        "Optional": Optional,
        "Any": Any,
        "Dict": Dict,
        "NamedTuple": NamedTuple,
        "record_function_range": _profiler.record_function_range,
        "build_cp_full_prefill_positions_shared": cp.build_cp_full_prefill_positions_shared,
        "CPContext": object,
        "CompressorMeta": object,
        "PrefillWorkspace": object,
    }
    exec(compile(mod, str(INDEXER_PATH), "exec"), env)
    return env


def _extract_compressor():
    tree = ast.parse(COMPRESSOR_SRC)
    want_classes = {"CompressorMeta"}
    want_methods = {"prepare_metadata"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in want_classes:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "CompressorFP8":
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name in want_methods:
                    body.append(m)
    names = {n.name for n in body}
    assert (
        want_classes | want_methods
    ) == names, f"missing: {(want_classes | want_methods) - names}"
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    mod = ast.fix_missing_locations(ast.Module(body=[future] + body, type_ignores=[]))
    import os as _os

    env = {
        "torch": torch,
        "os": _os,
        "Optional": Optional,
        "Any": Any,
        "Dict": Dict,
        "Tuple": Tuple,
        "dataclass": __import__("dataclasses").dataclass,
        "record_function_range": _profiler.record_function_range,
    }
    exec(compile(mod, str(COMPRESSOR_PATH), "exec"), env)
    return env


# The real prefill_meta + the module under test load AFTER the attention stub
# is installed (both import ``bind_attn_cache`` from it lazily).
import contextlib as _contextlib


@_contextlib.contextmanager
def _bind_attn_cache(attn, kv_cache, block_tables_by_type):
    prev_kv = getattr(attn, "_kv_cache", None)
    prev_bt = getattr(attn, "_block_tables_by_type", None)
    attn._kv_cache = kv_cache
    attn._block_tables_by_type = block_tables_by_type
    try:
        yield attn
    finally:
        attn._kv_cache = prev_kv
        attn._block_tables_by_type = prev_bt


# The patch path imports these helpers from the attention module.


# ---------------------------------------------------------------------------
# Stub pool plumbing
# ---------------------------------------------------------------------------


class _StubKvCache:
    seq_size_per_block = 256
    kernel_seq_size_per_block = 256

    def get_seq_size_per_block(self, tag):
        return self.seq_size_per_block

    def get_kernel_seq_size_per_block(self, tag):
        return self.kernel_seq_size_per_block


# entries-per-block per pool (fixture; both arms share it). Production
# geometry: 256 raw tokens per block-table row; the compressed pools hold
# 256/ratio entries per block (CSA r=4 → 64, HCA r=128 → 2, indexer r=4 → 64).


def _make_tables(n_blocks, variant=0):
    """One block table per pool tag with distinct, deterministic block ids.

    Block ids stay small (valid pool indices) but differ per ``variant`` so a
    stale-table reuse is visible in every slot field. Per tag the ids are
    rotated so a cross-tag mix-up would also show.
    """
    tables = {}
    for i, tag in enumerate(_KV_CACHE.group_tags):
        ids = torch.arange(1, n_blocks + 1, dtype=torch.int32)
        if variant:
            ids = ids.roll(3 + variant + i)
        tables[tag] = ids.view(1, n_blocks).contiguous()
    return tables


class _StubCompressor:
    """Carries the pool context + the REAL ``prepare_metadata``."""

    def __init__(self, compress_ratio, head_dim):
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = 1
        self._cp_ctx = None
        self._kv_cache_sharded = False
        self._kv_pool_view = None
        self._kv_block_table = None
        self._kv_eb = 0
        self._state_pool_3d = None
        self._state_block_table = None
        self._state_eb = 0
        self._state_tokens_per_block = 0
        self.freqs_cis = None

    def set_pool_context(
        self,
        kv_pool_view,
        kv_block_table,
        kv_eb,
        state_pool_view,
        state_block_table,
        state_eb,
        *,
        state_tokens_per_block,
        kv_tokens_per_block,
        kv_owner_tokens_per_block=0,
    ):
        self._kv_pool_view = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb
        self._state_pool_3d = state_pool_view
        self._state_block_table = state_block_table
        self._state_eb = state_eb
        self._state_tokens_per_block = state_tokens_per_block

    def clear_pool_context(self):
        self._kv_pool_view = None
        self._kv_block_table = None
        self._kv_eb = 0
        self._state_pool_3d = None
        self._state_block_table = None
        self._state_eb = 0

    def set_cp_ctx(self, ctx):
        self._cp_ctx = ctx


class _StubIndexer:
    """Carries the pool context + the REAL ``prepare``."""

    def __init__(self, compress_ratio, freqs_cis):
        self.compress_ratio = compress_ratio
        self.freqs_cis = freqs_cis
        self._cp_ctx = None
        self._kv_pool_view = None
        self._kv_block_table = None
        self._kv_eb = 0
        self._state_block_table = None
        self._state_eb = 0
        self._kv_owner_tokens_per_block = 0
        self.compressor = _StubCompressor(compress_ratio, 128)

    def set_cp_ctx(self, ctx):
        self._cp_ctx = ctx

    def set_pool_context(
        self,
        kv_pool_view,
        kv_block_table,
        kv_eb,
        state_pool_view,
        state_block_table,
        state_eb,
        *,
        state_tokens_per_block,
        kv_tokens_per_block,
        kv_owner_tokens_per_block=0,
    ):
        self._kv_pool_view = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb
        self._state_block_table = state_block_table
        self._state_eb = state_eb

    def _propagate_pool_to_nested(self):
        self.compressor.set_pool_context(
            self._kv_pool_view,
            self._kv_block_table,
            self._kv_eb,
            None,
            self._state_block_table,
            self._state_eb,
            state_tokens_per_block=256,
            kv_tokens_per_block=256,
        )

    def _clear_nested_pool(self):
        self.compressor.clear_pool_context()


class _StubAttention:
    """Carries what the extracted builders + patch path read off ``self``."""

    def __init__(self, *, compress_ratio, freqs_cis, cp_ctx, window_size=64):
        self.compress_ratio = compress_ratio
        self.rope_head_dim = 64
        self.window_size = window_size
        self.freqs_cis = freqs_cis
        self._cp_ctx = cp_ctx
        self._kv_cache = _KV_CACHE
        self._block_tables_by_type = None
        self._prefill_meta_shared = None
        self.compressor = (
            _StubCompressor(compress_ratio, 512) if compress_ratio in (4, 128) else None
        )
        self.indexer = (
            _StubIndexer(compress_ratio, freqs_cis) if compress_ratio == 4 else None
        )

    # --- pool helpers ---
    def _pool_entries_per_block(self, tag):
        return int(_EB.get(tag, 0))

    def _swa_entries_per_block(self):
        return int(_EB[kv_cache_utils.SWA_KV])

    def _swa_cp_byte_sliced(self):
        return False

    def _pool_raw_u8(self, tag):
        return None

    def _ensure_freqs_cis_bound(self):
        return None

    def _set_prefill_meta_shared(self, meta):
        self._prefill_meta_shared = meta

    def set_cp_ctx(self, ctx):
        # Mirror ``V4Transformer._propagate_cp_ctx``'s traversal.
        self._cp_ctx = ctx
        if self.compressor is not None:
            self.compressor.set_cp_ctx(ctx)
        if self.indexer is not None:
            self.indexer.set_cp_ctx(ctx)
            self.indexer.compressor.set_cp_ctx(ctx)

    def _set_compressor_pool_context(self):
        # Mirror of the production bind: compressor gets its ratio's pools,
        # the indexer gets the INDEXER pools — from the CURRENT
        # ``_block_tables_by_type``. The fake pool view is sized so the
        # slot values land inside the ``pool_rows`` guard.
        tables = self._block_tables_by_type

        def _pool_view(tag):
            n_blocks = int(tables[tag].shape[1]) if tables.get(tag) is not None else 1
            eb = self._pool_entries_per_block(tag)
            return torch.zeros(n_blocks + 64, max(eb, 1), 8)

        if self.compressor is not None:
            kv_at, state_at = {
                4: (kv_cache_utils.CSA_KV, kv_cache_utils.CSA_STATE),
                128: (kv_cache_utils.HCA_KV, kv_cache_utils.HCA_STATE),
            }[self.compress_ratio]
            self.compressor.set_pool_context(
                _pool_view(kv_at),
                tables.get(kv_at),
                self._pool_entries_per_block(kv_at),
                None,
                tables.get(state_at),
                self._pool_entries_per_block(state_at),
                state_tokens_per_block=256,
                kv_tokens_per_block=256,
            )
        if self.indexer is not None:
            self.indexer.set_pool_context(
                _pool_view(kv_cache_utils.INDEXER_KV),
                tables.get(kv_cache_utils.INDEXER_KV),
                self._pool_entries_per_block(kv_cache_utils.INDEXER_KV),
                None,
                tables.get(kv_cache_utils.INDEXER_STATE),
                self._pool_entries_per_block(kv_cache_utils.INDEXER_STATE),
                state_tokens_per_block=256,
                kv_tokens_per_block=256,
            )

    def _clear_compressor_pool_context(self):
        if self.compressor is not None:
            self.compressor.clear_pool_context()
        if self.indexer is not None:
            self.indexer._kv_pool_view = None
            self.indexer._kv_block_table = None
            self.indexer._kv_eb = 0
            self.indexer._state_block_table = None
            self.indexer._state_eb = 0

    # --- extracted real methods, bound ---


# ``_build_csa_prefill_meta`` does function-level imports of the compressor and
# indexer modules (which need native deps); install stub modules exposing the
# referenced symbols with the fixture classes bound.


def _build_prepare_metadata_args_unsupported(*a, **k):
    raise AssertionError(
        "non-CP compressor meta path is out of scope for async-head tests"
    )


class _StubLayer:
    def __init__(self, attn):
        self.attn = attn


class _StubV4:
    def __init__(self, attns):
        self.layers = [_StubLayer(a) for a in attns]
        self.fp8_kv_cache = True


# ---------------------------------------------------------------------------
# Geometry fixtures
# ---------------------------------------------------------------------------


def _recipe_cp_info():
    """The framework's per-chunk inputs for a full 4096-token CP4 chunk."""
    _, restore = compact_rt._zigzag_constants()
    info = types.SimpleNamespace(
        prefill_qkv_padding_mask=torch.ones(4096, dtype=torch.int32),
        prefill_qkv_restore_indice=restore.to(torch.int32).contiguous(),
        prefill_actual_input_lengths_cpu=torch.tensor([4096], dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor([1024], dtype=torch.int32),
    )
    return info


def _build_ctx(cp_rank, prefix):
    info = _recipe_cp_info()
    with patch.dict(os.environ, {"DSV4_CP_CONTEXT_PLAN_CACHE": "0"}):
        ctx = cp.build_cp_context_for_forward(
            info,
            4,
            cp_rank,
            1024,
            torch.device("cpu"),
            prefix_lengths=torch.tensor([prefix], dtype=torch.int32),
            kv_cache_sharded=False,
        )
    return ctx


def _meta_kwargs(ctx):
    """The meta-build arg set, mirroring prefill/forward.py's prep block."""
    return dict(
        sp_per_req=ctx.prefix_lengths.to(torch.int64).contiguous(),
        cu_seqlens=torch.tensor([0, 1024], dtype=torch.int32),
        batch_size=1,
        input_lengths=torch.tensor([1024], dtype=torch.int32),
        prefix_lengths=ctx.prefix_lengths.to(torch.int32).contiguous(),
        position_ids=ctx.global_positions.to(torch.long),
        req_id_per_token=ctx.req_id_per_token.to(torch.int32).contiguous(),
        max_seqlen_q=1024,
    )


def _make_attns(ctx, shared_freqs=None):
    freqs = (
        shared_freqs
        if shared_freqs is not None
        else (torch.arange(33000 * 8, dtype=torch.float32).reshape(33000, 8) / 1000.0)
    )
    attns = {
        0: _StubAttention(compress_ratio=0, freqs_cis=freqs, cp_ctx=None),
        4: _StubAttention(compress_ratio=4, freqs_cis=freqs, cp_ctx=None),
        128: _StubAttention(compress_ratio=128, freqs_cis=freqs, cp_ctx=None),
    }
    for a in attns.values():
        a.set_cp_ctx(ctx)
    return attns


def _legacy_build(cp_rank, prefix, tables):
    """The legacy eager path: context + 3-bucket metas for chunk (rank, prefix)."""
    ctx = _build_ctx(cp_rank, prefix)
    attns = _make_attns(ctx)
    args = _meta_kwargs(ctx)
    sp_int = int(ctx.global_positions[0].item())
    metas = {}
    for r, attn in attns.items():
        attn._block_tables_by_type = tables
        with _bind_attn_cache(attn, _KV_CACHE, tables):
            metas[r] = attn._build_shared_prefill_meta(
                torch.zeros(1024, 4),
                sp_int,
                **args,
                shared=None,
            )._replace(workspace=None)
        attn._block_tables_by_type = None
    return ctx, metas, attns


def _prebuilt_build(cp_rank, prefix, prev_tables):
    """The builder body (synchronous drive): previous chunk's table bound."""
    ctx_prev = _build_ctx(cp_rank, prefix - 4096) if prefix > 0 else None
    key = head_prebuild.predicted_continuation_key(ctx_prev, torch.device("cpu"))
    # Use a fresh attn set for the builder (mirrors production binding).
    ctx_for_builders = _build_ctx(cp_rank, prefix)
    attns = _make_attns(ctx_for_builders)
    v4 = _StubV4([attns[0], attns[4], attns[128]])
    bundle = head_prebuild._build_bundle_inner(
        key=key,
        v4=v4,
        kv_cache=_KV_CACHE,
        block_tables_by_type=prev_tables,
        device=torch.device("cpu"),
        cp_size=4,
        cp_rank=cp_rank,
        predicted_prefix=prefix,
        shared=None,
    )
    return bundle, attns, key


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------

_SKIP_FIELDS = ("m.cp_ctx", "m.workspace")


def _meta_fields(meta):
    import dataclasses as _dc

    out = {}

    def walk(prefix, value):
        if prefix in _SKIP_FIELDS:
            return
        if isinstance(value, torch.Tensor):
            out[prefix] = value
        elif hasattr(value, "_fields"):  # NamedTuple
            for k, v in zip(value._fields, value):
                walk(f"{prefix}.{k}", v)
        elif _dc.is_dataclass(value) and not isinstance(value, type):
            for f in _dc.fields(value):
                walk(f"{prefix}.{f.name}", getattr(value, f.name))
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
            tc.assertEqual(tuple(gv.shape), tuple(wv.shape), f"{k} shape {msg}")
            tc.assertTrue(torch.equal(gv, wv), f"{k} {msg}")
        elif isinstance(gv, torch.device):
            tc.assertEqual(str(gv), str(wv), f"{k} {msg}")
        elif isinstance(gv, _LocalPoolReaderStub):
            tc.assertIsInstance(wv, _LocalPoolReaderStub, f"{k} {msg}")
        else:
            tc.assertEqual(gv, wv, f"{k} {msg}")


_CTX_TENSORS = (
    "relative_positions",
    "global_positions",
    "local_is_real",
    "unpad_restore",
    "req_id_per_token",
    "prefix_lengths",
    "input_lengths_global",
    "cu_seqlens_global",
)
_CTX_SCALARS = (
    "cp_size",
    "cp_rank",
    "chunk_length",
    "padded_seq_len",
    "seq_len_full",
    "prefix_length",
    "seq_len_total",
    "unpad_restore_is_prefix",
    "kv_cache_sharded",
    "first_global_position",
    "prefix_lengths_full_host",
    "input_lengths_full_host",
)


def _assert_ctx_equal(tc, got, want, msg=""):
    for name in _CTX_SCALARS:
        tc.assertEqual(getattr(got, name), getattr(want, name), f"ctx.{name} {msg}")
    for name in _CTX_TENSORS:
        gv, wv = getattr(got, name), getattr(want, name)
        tc.assertEqual(gv is None, wv is None, f"ctx.{name} None-ness {msg}")
        if gv is not None:
            tc.assertEqual(gv.dtype, wv.dtype, f"ctx.{name} dtype {msg}")
            tc.assertTrue(torch.equal(gv, wv), f"ctx.{name} {msg}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
RECIPE_RANKS = (0, 1, 2, 3)
# Chunk starts for an 8-chunk 32K request (prefix = chunk_idx * 4096); 0 is the
# request's first chunk (never a prediction target — nothing precedes it — but
# it exercises the prefix=0 edge of the builders).
RECIPE_PREFIXES = (0, 4096, 2 * 4096, 5 * 4096, 7 * 4096)


class PrebuiltByteIdentity(unittest.TestCase):
    """The prebuilt+patched meta/context equals the legacy eager build."""

    def test_byte_identity_matrix(self):
        for rank in RECIPE_RANKS:
            for prefix in RECIPE_PREFIXES:
                with self.subTest(rank=rank, prefix=prefix):
                    # Previous chunk's table covers the previous range; the
                    # real chunk's table extends it with NEW distinct ids.
                    prev_blocks = max(1, prefix // 256)
                    new_blocks = (prefix + 4096) // 256
                    prev_tables = _make_tables(prev_blocks, variant=0)
                    new_tables = _make_tables(new_blocks, variant=1)

                    ctx_legacy, metas_legacy, _ = _legacy_build(
                        rank, prefix, new_tables
                    )

                    if prefix == 0:
                        continue  # no prediction target precedes chunk 0

                    bundle, attns_cand, key = _prebuilt_build(rank, prefix, prev_tables)
                    # Key must equal the real forward's key.
                    info = _recipe_cp_info()
                    real_key = head_prebuild.key_from_forward_inputs(
                        info,
                        torch.tensor([prefix], dtype=torch.int32),
                        torch.device("cpu"),
                        4,
                        rank,
                        1024,
                    )
                    self.assertEqual(key, real_key)

                    # Anti-vacuity: the stale (pre-patch) slot fields must
                    # differ from legacy somewhere (the previous table is
                    # narrower → clamped/masked), proving the patch is needed.
                    stale_csa = bundle.meta_by_ratio[4].csa_meta
                    self.assertFalse(
                        torch.equal(
                            stale_csa.compressor_meta.kv_slots,
                            metas_legacy[4].csa_meta.compressor_meta.kv_slots,
                        ),
                        "stale slots must differ pre-patch (anti-vacuity)",
                    )

                    # Patch + propagate.
                    head_prebuild.patch_prebuilt_metas(bundle, _KV_CACHE, new_tables)
                    for r in (0, 4, 128):
                        bundle.meta_by_ratio[r] = bundle.meta_by_ratio[r]._replace(
                            workspace=None
                        )
                        _assert_ctx_equal(
                            self,
                            bundle.cp_ctx,
                            ctx_legacy,
                            msg=f"rank={rank} prefix={prefix}",
                        )
                        _assert_meta_equal(
                            self,
                            bundle.meta_by_ratio[r],
                            metas_legacy[r],
                            msg=f"ratio={r} rank={rank} prefix={prefix}",
                        )

    def test_patch_is_load_bearing_on_every_slot_field(self):
        """Every patched field must change vs the stale pre-patch value."""
        rank, prefix = 1, 3 * 4096
        prev_tables = _make_tables(max(1, prefix // 256), variant=0)
        new_tables = _make_tables((prefix + 4096) // 256, variant=1)
        _, metas_legacy, _ = _legacy_build(rank, prefix, new_tables)
        bundle, _, _ = _prebuilt_build(rank, prefix, prev_tables)

        def slots_differ(stale, fresh):
            if stale is None and fresh is None:
                return False
            if (stale is None) != (fresh is None):
                return True
            if stale.shape != fresh.shape:
                return True
            return not torch.equal(stale, fresh)

        # Capture the stale (pre-patch) slot fields, then patch.
        stale = {}
        for r in (0, 4, 128):
            meta = bundle.meta_by_ratio[r]
            entry = {"swa_slot_mapping": meta.swa_meta.slot_mapping}
            if r == 4:
                entry["csa_kv_slots"] = meta.csa_meta.compressor_meta.kv_slots
                entry["idx_block_table"] = meta.csa_meta.indexer_meta.block_table_i32
                entry["ws_swa_cache"] = (
                    meta.csa_meta.workspace_meta.swa_cache_slot_mapping
                )
            if r == 128:
                entry["hca_kv_slots"] = meta.hca_meta.compressor_meta.kv_slots
                entry["ws_swa_cache"] = (
                    meta.hca_meta.workspace_meta.swa_cache_slot_mapping
                )
            stale[r] = entry
        head_prebuild.patch_prebuilt_metas(bundle, _KV_CACHE, new_tables)
        for r in (0, 4, 128):
            meta = bundle.meta_by_ratio[r]
            legacy = metas_legacy[r]
            _assert_meta_equal(self, meta, legacy, msg=f"ratio={r}")
            # The stale SWA slot mapping must have differed from legacy (the
            # previous chunk's table is narrower + differently arranged), and
            # the patched field must now equal legacy.
            self.assertTrue(
                slots_differ(
                    stale[r]["swa_slot_mapping"], legacy.swa_meta.slot_mapping
                ),
                f"ratio={r}: stale SWA slot mapping must differ pre-patch",
            )
            self.assertTrue(
                torch.equal(meta.swa_meta.slot_mapping, legacy.swa_meta.slot_mapping),
                f"ratio={r}: patched SWA slot mapping must equal legacy",
            )
        # CSA-specific stale fields also differed pre-patch.
        self.assertTrue(
            slots_differ(
                stale[4]["csa_kv_slots"],
                metas_legacy[4].csa_meta.compressor_meta.kv_slots,
            )
        )
        self.assertTrue(
            slots_differ(
                stale[4]["idx_block_table"],
                metas_legacy[4].csa_meta.indexer_meta.block_table_i32,
            )
        )
        self.assertTrue(
            slots_differ(
                stale[4]["ws_swa_cache"],
                metas_legacy[4].csa_meta.workspace_meta.swa_cache_slot_mapping,
            )
        )


class PrebuiltValidation(unittest.TestCase):
    """Fail-closed validation at consume: key + content gates."""

    def _seed_singleton(self, bundle, key):
        """Point the module's builder singleton at a pre-seeded local builder."""
        builder = head_prebuild.AsyncHeadBuilder()
        builder._result = (key, bundle)
        saved = head_prebuild._BUILDER
        head_prebuild._BUILDER = builder
        self.addCleanup(lambda: setattr(head_prebuild, "_BUILDER", saved))
        return builder

    def test_key_hit_and_content_pass(self):
        rank, prefix = 2, 4096
        prev_tables = _make_tables(32)
        bundle, _, key = _prebuilt_build(rank, prefix, prev_tables)
        self._seed_singleton(bundle, key)
        with patch.dict(os.environ, {"DSV4_PREFILL_ASYNC_HEAD": "1"}):
            got = head_prebuild.consume_async_head(
                _recipe_cp_info(),
                torch.tensor([prefix], dtype=torch.int32),
                torch.device("cpu"),
                4,
                rank,
                1024,
            )
        self.assertIs(got, bundle)

    def test_key_mismatch_declines(self):
        rank, prefix = 2, 4096
        prev_tables = _make_tables(32)
        bundle, _, key = _prebuilt_build(rank, prefix, prev_tables)
        with patch.dict(os.environ, {"DSV4_PREFILL_ASYNC_HEAD": "1"}):
            # Wrong prefix (request boundary / cache hit): the key differs.
            self._seed_singleton(bundle, key)
            got = head_prebuild.consume_async_head(
                _recipe_cp_info(),
                torch.tensor([prefix + 4096], dtype=torch.int32),
                torch.device("cpu"),
                4,
                rank,
                1024,
            )
            self.assertIsNone(got)
            # Wrong rank.
            self._seed_singleton(bundle, key)
            got2 = head_prebuild.consume_async_head(
                _recipe_cp_info(),
                torch.tensor([prefix], dtype=torch.int32),
                torch.device("cpu"),
                4,
                (rank + 1) % 4,
                1024,
            )
            self.assertIsNone(got2)

    def test_content_mismatch_declines(self):
        rank, prefix = 2, 4096
        prev_tables = _make_tables(32)
        bundle, _, key = _prebuilt_build(rank, prefix, prev_tables)
        self._seed_singleton(bundle, key)
        bad_info = _recipe_cp_info()
        bad_info.prefill_qkv_padding_mask = bad_info.prefill_qkv_padding_mask.clone()
        bad_info.prefill_qkv_padding_mask[100] = 0  # padding in a "full" chunk
        with patch.dict(os.environ, {"DSV4_PREFILL_ASYNC_HEAD": "1"}):
            got = head_prebuild.consume_async_head(
                bad_info,
                torch.tensor([prefix], dtype=torch.int32),
                torch.device("cpu"),
                4,
                rank,
                1024,
            )
        self.assertIsNone(got)

    def test_partial_tail_chunk_declines(self):
        """A partial last chunk (real 3000 < padded 4096) never consumes a
        full-chunk prediction: the key's real-lengths field differs."""
        rank, prefix = 2, 4096
        prev_tables = _make_tables(32)
        bundle, _, key = _prebuilt_build(rank, prefix, prev_tables)
        self._seed_singleton(bundle, key)
        partial_info = _recipe_cp_info()
        partial_info.prefill_actual_input_lengths_cpu = torch.tensor(
            [3000], dtype=torch.int32
        )
        with patch.dict(os.environ, {"DSV4_PREFILL_ASYNC_HEAD": "1"}):
            got = head_prebuild.consume_async_head(
                partial_info,
                torch.tensor([prefix], dtype=torch.int32),
                torch.device("cpu"),
                4,
                rank,
                752,  # rank-local partial chunk length (3000 padded zigzag)
            )
        self.assertIsNone(got)

    def test_builder_device_work_uses_only_its_own_stream(self):
        """Source invariant: every CUDA device op in the builder wrapper runs
        under the dedicated builder stream (never the main/current stream)."""
        src = HEADPREBUILD_PATH.read_text()
        fn = ast.parse(src)
        wrapper = None
        for node in fn.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_build_with_stream_fence"
            ):
                wrapper = node
                break
        self.assertIsNotNone(wrapper)
        # The CUDA branch must wrap the build in ``with torch.cuda.stream(<the
        # builder stream>)`` and record the completion event on it; the build
        # only runs outside a stream context on non-CUDA. ``ast.dump``
        # decomposes attribute chains, so assert the structural pieces.
        with_items = [n for n in ast.walk(wrapper) if isinstance(n, ast.With)]
        self.assertTrue(with_items, "the CUDA branch must run under a stream context")
        ctx_expr = ast.dump(with_items[0].items[0].context_expr)
        self.assertIn("attr='stream'", ctx_expr)
        self.assertIn("attr='cuda'", ctx_expr)
        self.assertIn("id='_builder_stream'", ast.dump(wrapper))
        self.assertIn("cuda", ast.dump(wrapper))

    def test_builder_exception_produces_none(self):
        builder = head_prebuild.AsyncHeadBuilder()

        def bad():
            raise RuntimeError("boom")

        builder.kick((1,), bad)
        self.assertIsNone(builder.consume((1,)))
        self.assertEqual(builder._failed, 1)

    def test_no_prediction_outstanding_returns_none(self):
        builder = head_prebuild.AsyncHeadBuilder()
        self.assertIsNone(builder.consume((1, 2)))

    def test_recipe_domain_gate(self):
        ctx = _build_ctx(0, 4096)
        self.assertTrue(head_prebuild._recipe_domain_ok(ctx))
        # Non-CP / wrong geometry decline.
        self.assertFalse(head_prebuild._recipe_domain_ok(None))
        ctx_sharded = _build_ctx(0, 4096)
        ctx_sharded.kv_cache_sharded = True
        self.assertFalse(head_prebuild._recipe_domain_ok(ctx_sharded))
        # A partial-tail chunk (real_lengths != 4096) declines.
        info = _recipe_cp_info()
        info.prefill_actual_input_lengths_cpu = torch.tensor([3000], dtype=torch.int32)
        ctx_partial = cp.build_cp_context_for_forward(
            info,
            4,
            0,
            1024,
            torch.device("cpu"),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            kv_cache_sharded=False,
        )
        self.assertFalse(head_prebuild._recipe_domain_ok(ctx_partial))


class PrebuiltThreadDiscipline(unittest.TestCase):
    """The real builder thread: runs off-thread, delivers, fails closed."""

    def test_builder_runs_off_thread_with_same_values(self):
        builder = head_prebuild.AsyncHeadBuilder()
        main_tid = __import__("threading").get_ident()
        seen = {}

        def build():
            seen["tid"] = __import__("threading").get_ident()
            return ("bundle-marker", torch.arange(4))

        builder.kick(("k",), build)
        got = builder.consume(("k",))
        self.assertIsNotNone(got)
        self.assertNotEqual(seen["tid"], main_tid)
        self.assertEqual(got[0], "bundle-marker")

    def test_kick_after_result_replaces_and_no_leak_of_stale(self):
        builder = head_prebuild.AsyncHeadBuilder()
        builder.kick(("a",), lambda: "A")
        self.assertEqual(builder.consume(("a",)), "A")
        builder.kick(("b",), lambda: "B")
        self.assertEqual(builder.consume(("b",)), "B")
        # A kick whose key never matches is dropped by consume.
        builder.kick(("c",), lambda: "C")
        self.assertIsNone(builder.consume(("not-c",)))

    def test_builder_mirrors_main_thread_inference_mode(self):
        """The worker must mirror both caller states, not force inference mode."""
        seen = []
        real = head_prebuild._build_under_inference_mode

        def spy(kwargs, inference_mode):
            seen.append(bool(inference_mode))
            return real(kwargs, inference_mode)

        def drive(rank, prefix):
            prev_tables = _make_tables(max(1, prefix // 256), variant=0)
            ctx_prev = _build_ctx(rank, prefix)
            attns = _make_attns(ctx_prev)
            v4 = _StubV4([attns[0], attns[4], attns[128]])
            head_prebuild.maybe_kick_async_head(
                v4, _KV_CACHE, prev_tables, ctx_prev, torch.device("cpu")
            )
            # Join the builder so the spy has run before we assert.
            head_prebuild.consume_async_head(
                _recipe_cp_info(),
                torch.tensor([prefix + 4096], dtype=torch.int32),
                torch.device("cpu"),
                4,
                rank,
                1024,
            )

        saved = head_prebuild._BUILDER
        self.addCleanup(lambda: setattr(head_prebuild, "_BUILDER", saved))
        with patch.dict(
            os.environ,
            {"DSV4_PREFILL_ASYNC_HEAD": "1", "DSV4_FP8_PREFILL_META_SHARED_BUILD": "0"},
        ):
            with patch.object(head_prebuild, "_build_under_inference_mode", spy):
                # Serving contract: main thread inside inference_mode.
                head_prebuild._BUILDER = None
                with torch.inference_mode():
                    drive(3, 2 * 4096)
                self.assertEqual(seen, [True])
                # Not in inference mode: the builder must NOT force it on.
                seen.clear()
                head_prebuild._BUILDER = None
                drive(3, 2 * 4096)
                self.assertEqual(seen, [False])

    def test_end_to_end_kick_consume_patch_matches_legacy(self):
        """Full flow: kick on the builder thread → consume → patch → byte-equal.

        Drives ``maybe_kick_async_head`` (which posts to the real builder
        thread) for the chunk at ``prefix`` and consumes it for the chunk at
        ``prefix + 4096``, then patches against the new chunk's table and
        compares every ratio bucket's meta against the legacy eager build.
        """
        rank, prefix = 3, 2 * 4096
        prev_tables = _make_tables(max(1, prefix // 256), variant=0)
        new_tables = _make_tables((prefix + 2 * 4096) // 256, variant=1)

        # The "current" chunk's context (what forward N−1 produced).
        ctx_prev = _build_ctx(rank, prefix)
        attns = _make_attns(ctx_prev)
        v4 = _StubV4([attns[0], attns[4], attns[128]])

        saved = head_prebuild._BUILDER
        head_prebuild._BUILDER = None
        self.addCleanup(lambda: setattr(head_prebuild, "_BUILDER", saved))
        with patch.dict(
            os.environ,
            {
                "DSV4_PREFILL_ASYNC_HEAD": "1",
                "DSV4_FP8_PREFILL_META_SHARED_BUILD": "0",
            },
        ):
            head_prebuild.maybe_kick_async_head(
                v4, _KV_CACHE, prev_tables, ctx_prev, torch.device("cpu")
            )
            # The next chunk's real forward consumes the bundle.
            bundle = head_prebuild.consume_async_head(
                _recipe_cp_info(),
                torch.tensor([prefix + 4096], dtype=torch.int32),
                torch.device("cpu"),
                4,
                rank,
                1024,
            )
            self.assertIsNotNone(bundle, "the prediction must hit for the continuation")
            ok = prefill_meta.propagate_prebuilt_prefill_meta_fp8(
                v4, bundle, _KV_CACHE, new_tables, None
            )
            self.assertTrue(ok)

        # Legacy eager reference for the consumed chunk.
        _, metas_legacy, _ = _legacy_build(rank, prefix + 4096, new_tables)
        for r in (0, 4, 128):
            _assert_meta_equal(
                self,
                bundle.meta_by_ratio[r],
                metas_legacy[r],
                msg=f"ratio={r} rank={rank}",
            )
        ctx_legacy = _build_ctx(rank, prefix + 4096)
        _assert_ctx_equal(self, bundle.cp_ctx, ctx_legacy)


class PatchOutOfDomain(unittest.TestCase):
    """A bundle carrying fields the patch does not reproduce must fail closed."""

    def test_workspace_sm120_paged_maps_decline(self):
        rank, prefix = 0, 4096
        prev_tables = _make_tables(max(1, prefix // 256), variant=0)
        new_tables = _make_tables((prefix + 4096) // 256, variant=1)
        bundle, _, _ = _prebuilt_build(rank, prefix, prev_tables)
        # Mutate the prebuilt CSA workspace meta into the non-recipe shape.
        csa = bundle.meta_by_ratio[4].csa_meta
        wm = csa.workspace_meta
        bad_wm = wm._replace(swa_pool_slot_mapping=torch.zeros(1, 1, dtype=torch.long))
        bundle.meta_by_ratio[4] = bundle.meta_by_ratio[4]._replace(
            csa_meta=csa._replace(workspace_meta=bad_wm)
        )
        with self.assertRaises(head_prebuild._PatchOutOfDomain):
            head_prebuild.patch_prebuilt_metas(bundle, _KV_CACHE, new_tables)

    def test_propagate_fails_closed_on_patch_error(self):
        rank, prefix = 0, 4096
        prev_tables = _make_tables(max(1, prefix // 256), variant=0)
        new_tables = _make_tables((prefix + 4096) // 256, variant=1)
        bundle, attns, _ = _prebuilt_build(rank, prefix, prev_tables)
        csa = bundle.meta_by_ratio[4].csa_meta
        bad_wm = csa.workspace_meta._replace(use_cp_raw_q_merge=True)
        bundle.meta_by_ratio[4] = bundle.meta_by_ratio[4]._replace(
            csa_meta=csa._replace(workspace_meta=bad_wm)
        )
        v4 = _StubV4([attns[0], attns[4], attns[128]])
        ok = prefill_meta.propagate_prebuilt_prefill_meta_fp8(
            v4, bundle, _KV_CACHE, new_tables, None
        )
        self.assertFalse(ok)
        # The fallback contract: nothing was propagated.
        for a in attns.values():
            self.assertIsNone(a._prefill_meta_shared)


class AsyncHeadFlag(unittest.TestCase):
    def test_flag_fail_closed_parse(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_PREFILL_ASYNC_HEAD", None)
            self.assertTrue(head_prebuild.async_head_enabled())  # default ON
        with patch.dict(os.environ, {"DSV4_PREFILL_ASYNC_HEAD": "0"}):
            self.assertFalse(head_prebuild.async_head_enabled())
        with patch.dict(os.environ, {"DSV4_PREFILL_ASYNC_HEAD": "1"}):
            self.assertTrue(head_prebuild.async_head_enabled())
        with patch.dict(os.environ, {"DSV4_PREFILL_ASYNC_HEAD": "yes"}):
            with self.assertRaises(ValueError):
                head_prebuild.async_head_enabled()

    def test_flag_off_consume_returns_none_and_kick_noop(self):
        saved = head_prebuild._BUILDER
        head_prebuild._BUILDER = None
        self.addCleanup(lambda: setattr(head_prebuild, "_BUILDER", saved))
        with patch.dict(os.environ, {"DSV4_PREFILL_ASYNC_HEAD": "0"}):
            got = head_prebuild.consume_async_head(
                _recipe_cp_info(),
                torch.tensor([0], dtype=torch.int32),
                torch.device("cpu"),
                4,
                0,
                1024,
            )
            self.assertIsNone(got)
            # kick: no-op (no thread, no singleton creation).
            head_prebuild.maybe_kick_async_head(
                None, None, None, None, torch.device("cpu")
            )
            self.assertIsNone(head_prebuild._BUILDER)


def setUpModule():
    global ATT, COMP, IDX, _EB, _FP8_PKG, _KV_CACHE, _attn_stub_mod
    global _compressor_stub_mod, _cp_attention_shard, _indexer_stub_mod, _profiler
    global _swa_cp_byte_sliced, compact_rt, cp, fp8_kv_utils, fused_meta
    global head_prebuild, kv_cache_utils, prefill_meta, swa_ops
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()

    cp, kv_cache_utils, swa_ops, fp8_kv_utils, compact_rt, fused_meta, _FP8_PKG = (
        _load_dsv4_tree()
    )

    _profiler = sys.modules[_PREFIX + "._profiler"]

    _cp_attention_shard = sys.modules[_PREFIX + ".fp8._cp_attention_shard"]

    _swa_cp_byte_sliced = sys.modules[_PREFIX + ".fp8._swa_cp_byte_sliced"]

    ATT = _extract_attn()

    IDX = _extract_indexer()

    COMP = _extract_compressor()

    _attn_stub_mod = types.ModuleType(_PREFIX + ".fp8.attention")

    _attn_stub_mod.bind_attn_cache = _bind_attn_cache

    for _name in (
        "_build_suffix_pool_slot_mapping",
        "_suffix_gather_lens_max_host",
        "_flat_1d",
    ):
        setattr(_attn_stub_mod, _name, ATT[_name])

    _attn_stub_mod._build_swa_cp_byte_compaction = ATT["_build_swa_cp_byte_compaction"]

    sys.modules[_PREFIX + ".fp8.attention"] = _attn_stub_mod

    setattr(_FP8_PKG, "attention", _attn_stub_mod)

    prefill_meta = _load(_PREFIX + ".fp8.prefill_meta", PREFILL_META_PATH)

    head_prebuild = _load(_PREFIX + ".fp8._head_prebuild", HEADPREBUILD_PATH)

    _StubKvCache.group_tags = [
        kv_cache_utils.SWA_KV,
        kv_cache_utils.CSA_KV,
        kv_cache_utils.HCA_KV,
        kv_cache_utils.INDEXER_KV,
        kv_cache_utils.INDEXER_STATE,
        kv_cache_utils.CSA_STATE,
        kv_cache_utils.HCA_STATE,
    ]

    _KV_CACHE = _StubKvCache()

    _EB = {
        kv_cache_utils.SWA_KV: 32,
        kv_cache_utils.CSA_KV: 64,
        kv_cache_utils.HCA_KV: 2,
        kv_cache_utils.INDEXER_KV: 64,
        kv_cache_utils.INDEXER_STATE: 8,
        kv_cache_utils.CSA_STATE: 8,
        kv_cache_utils.HCA_STATE: 8,
    }

    _StubCompressor.prepare_metadata = COMP["prepare_metadata"]

    _StubIndexer.prepare = IDX["prepare"]

    _StubAttention._build_workspace_meta = ATT["_build_workspace_meta"]

    _StubAttention._build_swa_prefill_meta_varlen = ATT[
        "_build_swa_prefill_meta_varlen"
    ]

    _StubAttention._build_shared_prefill_meta = ATT["_build_shared_prefill_meta"]

    _StubAttention._build_csa_prefill_meta = ATT["_build_csa_prefill_meta"]

    _StubAttention._build_hca_prefill_meta = ATT["_build_hca_prefill_meta"]

    _StubAttention._build_compressor_meta = ATT["_build_compressor_meta"]

    _StubAttention._build_swa_cp_byte_compaction = ATT["_build_swa_cp_byte_compaction"]

    _compressor_stub_mod = types.ModuleType(_PREFIX + ".fp8.compressor")

    _compressor_stub_mod.build_prepare_metadata_args = (
        _build_prepare_metadata_args_unsupported
    )

    _compressor_stub_mod.CompressorFP8 = _StubCompressor

    _compressor_stub_mod.CompressorMeta = COMP["CompressorMeta"]

    sys.modules[_PREFIX + ".fp8.compressor"] = _compressor_stub_mod

    setattr(_FP8_PKG, "compressor", _compressor_stub_mod)

    _indexer_stub_mod = types.ModuleType(_PREFIX + ".fp8.indexer")

    _indexer_stub_mod.IndexerFP8 = _StubIndexer

    sys.modules[_PREFIX + ".fp8.indexer"] = _indexer_stub_mod

    setattr(_FP8_PKG, "indexer", _indexer_stub_mod)


if __name__ == "__main__":
    unittest.main()
