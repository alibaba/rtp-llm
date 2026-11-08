"""CPU regression tests for synchronous FP8 prefill metadata.

Execute the actual metadata builders with CPU Triton kernels and compare
their outputs against independent Python coordinate oracles — no
fallback-vs-fallback comparisons: every expected coordinate below is
derived in plain Python from the CP4 zigzag recipe.

Covered:

* ``AttentionFP8`` shared-meta builders (SWA / CSA / HCA buckets) under a
  CP4 context: global positions, indexer ks/ke/T, workspace N/M and the
  fresh-K flat slot stream.
* Eager ``build_and_propagate_prefill_meta_fp8`` sharing: same-ratio meta
  identity, Group-1 SWA write-field aliasing and clear behavior.
* ``CompressorFP8.forward``'s ordinary full CP gather: exactly one
  ``cp_all_gather_full_async`` → ``cp_wait_gather_full`` → ``_launch``
  sequence, for regular (4096) and irregular (40) geometry.

These tests do not establish hardware stream ordering or model numerics.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from runpy import run_path
from typing import Any, Dict, NamedTuple, Optional, Tuple
from unittest.mock import Mock, patch

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


_GENERIC_PROFILER = "rtp_llm.models_py.utils.profiler"


def _load_generic_profiler():
    """Load main's torch-only generic profiler that ``dsv4._profiler`` wraps."""
    if _GENERIC_PROFILER in sys.modules:
        return
    _package("rtp_llm.models_py.utils")
    path = DSV4.parents[1] / "utils" / "profiler.py"
    spec = importlib.util.spec_from_file_location(_GENERIC_PROFILER, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_GENERIC_PROFILER] = mod
    spec.loader.exec_module(mod)


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
    _package("rtp_llm.device")
    _load("rtp_llm.device.device_type", DSV4.parents[2] / "device" / "device_type.py")
    _package("rtp_llm.models_py")
    _package("rtp_llm.models_py.modules")
    _load_generic_profiler()
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
    fused_meta = _load(
        _PREFIX + ".fp8._fused_compressor_meta_triton",
        FP8 / "_fused_compressor_meta_triton.py",
    )
    return cp, kvu, swa_ops, fp8_kv_utils, fused_meta, fp8


# ---------------------------------------------------------------------------
# AST extraction: attention.py helpers + builders, indexer prepare, compressor
# meta + forward. The extracted symbols run REAL metadata code — only their
# module-level globals are provided by the fixture environment.
# ---------------------------------------------------------------------------
def _extract_attn():
    tree = ast.parse(ATTN_SRC)
    want_funcs = {
        "_flat_1d",
        "_build_suffix_pool_slot_mapping",
        "_build_suffix_cp_sliced_slot_mapping",
        "_swa_paged_slots_fully_covered",
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
        # Keep the actual cross-ratio reuse validator so the eager
        # propagation path exercises the real storage/view checks.
        "_validate_reusable_prefill_common",
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
        "build_cp_byte_sliced_slot_compaction": _swa_cp_byte_sliced.build_cp_byte_sliced_slot_compaction,
        "kv_tag_for_compress_ratio": kvu.kv_tag_for_compress_ratio,
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


def _make_compressed_k_pool_reader_stub(
    cp_ctx=None,
    kv_cache_sharded=False,
    per_req_total_kv_lens=None,
    block_size=None,
    owner_block_size=None,
    total_kv_len=None,
):
    assert not kv_cache_sharded, "sharded reader is out of scope for the CPU test"
    return _LocalPoolReaderStub()


def _extract_indexer():
    tree = ast.parse(INDEXER_SRC)
    want_classes = {"_IndexerFP8PrefillMeta"}
    want_methods = {"prepare"}
    want_funcs = {"_flat_1d"}
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
        "build_cp_full_prefill_positions": cp.build_cp_full_prefill_positions,
        "CPContext": object,
        "CompressorMeta": object,
        "PrefillWorkspace": object,
    }
    exec(compile(mod, str(INDEXER_PATH), "exec"), env)
    return env


def _extract_compressor():
    tree = ast.parse(COMPRESSOR_SRC)
    want_classes = {"CompressorMeta", "_CPStateReadSelection"}
    want_methods = {"prepare_metadata", "forward"}
    want_funcs = {
        "_cache_cp_state_read_selection",
        "_cp_sliced_state_read_needed",
        "_select_cp_state_read_blocks",
        "_select_cp_state_read_tail_blocks",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in want_classes:
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in want_funcs:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "CompressorFP8":
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name in want_methods:
                    body.append(m)
    names = {n.name for n in body}
    assert (
        want_classes | want_methods | want_funcs
    ) == names, f"missing: {(want_classes | want_methods | want_funcs) - names}"
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
        "_CP_STATE_READ_SELECTION_CACHE": __import__("weakref").WeakKeyDictionary(),
        "record_function_range": _profiler.record_function_range,
        "cp_should_gather": cp.cp_should_gather,
        "CPContext": object,
        # Scoped recorders — the full-gather test installs its own
        # assertions through ``_gather_spy`` per call.
        "cp_all_gather_full_async": _GATHER_SPY.start,
        "cp_wait_gather_full": _GATHER_SPY.wait,
        # ``forward`` only reaches this helper on the meta=None fallback;
        # the fixture always supplies meta, so a call here is a bug.
        "_build_prefill_positions": _build_prefill_positions_unexpected,
    }
    exec(compile(mod, str(COMPRESSOR_PATH), "exec"), env)
    return env


def _build_prefill_positions_unexpected(*args, **kwargs):
    raise AssertionError(
        "compressor.forward must use the caller-supplied CompressorMeta in "
        "this fixture; the meta=None fallback is out of scope"
    )


class _GatherSpy:
    """Scoped ``cp_all_gather_full_async`` / ``cp_wait_gather_full`` recorders.

    ``start`` mirrors the production signature: ``(local_2d, cp_ctx,
    stream=None, *, profile_name=None, workspace, cp_role)``. ``wait``
    takes the produced handle. The test configures ``handle`` / ``full``
    and inspects ``start_calls`` / ``wait_calls`` afterwards.
    """

    def __init__(self):
        self.handle = object()
        self.full: Optional[torch.Tensor] = None
        self.start_calls = []
        self.wait_calls = []

    def reset(self):
        self.handle = object()
        self.full = None
        self.start_calls = []
        self.wait_calls = []

    def start(self, local_2d, cp_ctx, stream=None, **kwargs):
        self.start_calls.append(((local_2d, cp_ctx, stream), kwargs))
        return self.handle

    def wait(self, handle):
        self.wait_calls.append(handle)
        return self.full


_GATHER_SPY = _GatherSpy()


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


# entries-per-block per pool (fixture). Production geometry: 256 raw tokens
# per block-table row; the compressed pools hold 256/ratio entries per block
# (CSA r=4 → 64, HCA r=128 → 2, indexer r=4 → 64).


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
        self._kv_tokens_per_block = 0
        self._kv_owner_tokens_per_block = 0
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
        self._kv_tokens_per_block = kv_tokens_per_block
        self._kv_owner_tokens_per_block = (
            kv_owner_tokens_per_block or kv_tokens_per_block
        )

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
        self._kv_owner_tokens_per_block = 0

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
    """Carries what the extracted builders + propagation path read off ``self``."""

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

    # --- extracted real methods, bound in ``setUpModule`` ---


# ``_build_csa_prefill_meta`` does function-level imports of the compressor and
# indexer modules (which need native deps); install stub modules exposing the
# referenced symbols with the fixture classes bound.


def _build_prepare_metadata_args_unsupported(*a, **k):
    raise AssertionError(
        "non-CP compressor meta path is out of scope for synchronous-meta tests"
    )


class _StubLayer:
    def __init__(self, attn):
        self.attn = attn


class _StubV4:
    def __init__(self, attns):
        self.layers = [_StubLayer(a) for a in attns]
        self.fp8_kv_cache = True
        self.max_seq_len = 32832


# ---------------------------------------------------------------------------
# Geometry fixtures
# ---------------------------------------------------------------------------


def _cp_zigzag_slices(seq_len, cp_size, cp_rank):
    """Global positions of rank ``cp_rank``'s local tokens for one request.

    The CP zigzag splits the padded request into ``2 * cp_size`` contiguous
    slices; rank ``r`` owns slice ``r`` and slice ``2 * cp_size - 1 - r``.
    """
    half = seq_len // (2 * cp_size)
    front = list(range(cp_rank * half, (cp_rank + 1) * half))
    back = list(range(seq_len - (cp_rank + 1) * half, seq_len - cp_rank * half))
    return front + back


def _recipe_cp_info():
    """The framework's per-chunk inputs for a full 4096-token CP4 chunk."""
    gather_order = []
    for r in range(4):
        gather_order += _cp_zigzag_slices(4096, 4, r)
    assert sorted(gather_order) == list(range(4096))
    restore = torch.empty(4096, dtype=torch.int32)
    for gather_index, global_pos in enumerate(gather_order):
        restore[global_pos] = gather_index
    info = types.SimpleNamespace(
        prefill_qkv_padding_mask=torch.ones(4096, dtype=torch.int32),
        prefill_qkv_restore_indice=restore.contiguous(),
        prefill_actual_input_lengths_cpu=torch.tensor([4096], dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor([1024], dtype=torch.int32),
    )
    return info


def _build_ctx(cp_rank, prefix):
    info = _recipe_cp_info()
    return cp.build_cp_context_for_forward(
        info,
        4,
        cp_rank,
        1024,
        torch.device("cpu"),
        prefix_lengths=torch.tensor([prefix], dtype=torch.int32),
        kv_cache_sharded=False,
    )


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


def _legacy_build(cp_rank, prefix, tables, shared_freqs=None):
    """The eager path: context + 3-bucket metas for chunk (rank, prefix)."""
    ctx = _build_ctx(cp_rank, prefix)
    attns = _make_attns(ctx, shared_freqs)
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
            )._replace(workspace=None)
        attn._block_tables_by_type = None
    return ctx, metas, attns


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class SynchronousMetadataTest(unittest.TestCase):
    """Actual builders vs independent Python coordinate oracles."""

    def test_cp4_builders_match_coordinate_oracle(self):
        for rank in (0, 1, 2, 3):
            for prefix in (0, 4096, 28672):
                with self.subTest(rank=rank, prefix=prefix):
                    ctx, metas, attns = _legacy_build(rank, prefix, _make_tables(128))
                    # Global positions: prefix + this rank's zigzag slices.
                    gp = torch.tensor(
                        [prefix + i for i in _cp_zigzag_slices(4096, 4, rank)],
                        dtype=torch.long,
                    )
                    self.assertTrue(
                        torch.equal(ctx.global_positions, gp),
                        "ctx.global_positions",
                    )

                    # Indexer meta (ratio-4 bucket): T=(p+4096)//4 compressed
                    # tokens, end_pos is the absolute sequence end, ks is
                    # all-zero (B==1 request starts at flat-K offset 0), ke is
                    # the per-token causal clamp to T.
                    idx = metas[4].csa_meta.indexer_meta
                    T = (prefix + 4096) // 4
                    self.assertEqual(idx.T, T)
                    self.assertEqual(idx.end_pos, prefix + 4096)
                    self.assertTrue(
                        torch.equal(idx.ks, torch.zeros(1024, dtype=torch.int32)),
                        "indexer ks",
                    )
                    want_ke = torch.clamp((gp.to(torch.int64) + 1) // 4, max=T).to(
                        torch.int32
                    )
                    self.assertTrue(torch.equal(idx.ke, want_ke), "indexer ke")
                    self.assertEqual(
                        idx.cu_kv_seqlens.tolist(), [0, T], "indexer cu_kv_seqlens"
                    )

                    # Workspace meta for the compressed buckets: N compressed
                    # rows, then min(prefix, win-1) prefix-tail rows, then the
                    # 4096 fresh rows — so M = N + 4096 + min(p, 63).
                    for ratio in (4, 128):
                        meta = metas[ratio]
                        wm = (
                            meta.csa_meta.workspace_meta
                            if ratio == 4
                            else meta.hca_meta.workspace_meta
                        )
                        N = (prefix + 4096) // ratio
                        want_M = N + 4096 + min(prefix, 63)
                        self.assertEqual(wm.N, N, f"ratio={ratio} N")
                        self.assertEqual(wm.M, want_M, f"ratio={ratio} M")
                        self.assertEqual(
                            wm.cmp_seq_lens.tolist(), [N], f"ratio={ratio} cmp_seq_lens"
                        )
                        self.assertEqual(
                            wm.swa_gather_lens.tolist(),
                            [4096 + min(prefix, 63)],
                            f"ratio={ratio} swa_gather_lens",
                        )
                        self.assertEqual(
                            wm.new_k_slot_in_flat.tolist(),
                            [N + min(prefix, 63) + i for i in range(4096)],
                            f"ratio={ratio} new_k_slot_in_flat",
                        )
                    for ratio in (0, 4, 128):
                        self.assertEqual(
                            metas[ratio].any_cont,
                            prefix > 0,
                            f"ratio={ratio} any_cont",
                        )
                        self.assertEqual(
                            metas[ratio].row_seqlens_full.tolist(),
                            [4096],
                            f"ratio={ratio} row_seqlens_full",
                        )

    def test_eager_metadata_propagates_and_clears_by_ratio(self):
        """Actual ``build_and_propagate_prefill_meta_fp8`` + clear."""
        for prefix in (0, 4096):
            with self.subTest(prefix=prefix):
                ctx = _build_ctx(2, prefix)
                shared_freqs = (
                    torch.arange(33000 * 8, dtype=torch.float32).reshape(33000, 8)
                    / 1000.0
                )
                ratios = (0, 4, 128, 4, 128)
                attns = [
                    _StubAttention(
                        compress_ratio=r, freqs_cis=shared_freqs, cp_ctx=None
                    )
                    for r in ratios
                ]
                for a in attns:
                    a.set_cp_ctx(ctx)
                v4 = _StubV4(attns)
                tables = _make_tables(128)
                marker = object()

                prefill_meta.build_and_propagate_prefill_meta_fp8(
                    v4,
                    torch.zeros(1024, 4),
                    int(ctx.global_positions[0].item()),
                    _KV_CACHE,
                    tables,
                    **_meta_kwargs(ctx),
                    workspace=marker,
                )

                slot_for = [a._prefill_meta_shared for a in attns]
                # Same-ratio layers share one meta object.
                self.assertIs(slot_for[1], slot_for[3], "ratio4 identity")
                self.assertIs(slot_for[2], slot_for[4], "ratio128 identity")
                self.assertIsNot(slot_for[0], slot_for[1])
                # The forward's workspace object is threaded onto every meta.
                for i, meta in enumerate(slot_for):
                    self.assertIs(meta.workspace, marker, f"layer {i} workspace")

                m0, m4, m128 = slot_for[0], slot_for[1], slot_for[2]
                # Ratio-specific geometry stays independent.
                self.assertEqual(m4.csa_meta.workspace_meta.N, (prefix + 4096) // 4)
                self.assertEqual(m128.hca_meta.workspace_meta.N, (prefix + 4096) // 128)
                self.assertEqual(m4.csa_meta.indexer_meta.T, (prefix + 4096) // 4)
                self.assertEqual(m4.csa_meta.indexer_meta.end_pos, prefix + 4096)
                # Reused common fields alias the ratio-0 source tensors.
                for m in (m4, m128):
                    self.assertIs(m.topk_idxs, m0.topk_idxs, "topk_idxs shared")
                    self.assertIs(
                        m.row_seqlens_full,
                        m0.row_seqlens_full,
                        "row_seqlens_full shared",
                    )
                    self.assertIsNot(m.swa_meta, m0.swa_meta)
                    for field in (
                        "slot_mapping",
                        "query_start_loc",
                        "combined_seq_lens",
                        "topk_length_kv_full",
                        "slot_compaction",
                    ):
                        self.assertIs(
                            getattr(m.swa_meta, field),
                            getattr(m0.swa_meta, field),
                            f"swa_meta.{field} shared",
                        )
                # Compressed-RoPE buckets reuse one gathered freqs_cis.
                self.assertIs(m4.freqs_cis, m128.freqs_cis, "freqs_cis shared")
                for m in (m0, m4, m128):
                    self.assertEqual(
                        m.freqs_cis_source_id, id(shared_freqs), "freqs source id"
                    )

                # The real clear walks every layer and empties the slot.
                prefill_meta.clear_prefill_meta_shared_fp8(v4)
                for a in attns:
                    self.assertIsNone(a._prefill_meta_shared, "cleared slot")

    def test_full_cp_gather_is_used_for_ordinary_and_irregular_geometry(self):
        """``forward`` runs the ordinary full gather — no compact transport."""
        for full_len in (4096, 40):
            for rank in (0, 1, 2, 3):
                for start_pos in (0, 4096):
                    with self.subTest(
                        full_len=full_len, rank=rank, start_pos=start_pos
                    ):
                        self._drive_forward(full_len, rank, start_pos)

    def _drive_forward(self, full_len, rank, start_pos):
        full = torch.arange(full_len * 8, dtype=torch.float32).reshape(full_len, 8)
        local = full[
            torch.tensor(_cp_zigzag_slices(full_len, 4, rank), dtype=torch.long)
        ].contiguous()
        self.assertEqual(tuple(local.shape), (full_len // 4, 8))

        harness = types.SimpleNamespace(
            overlap=1,
            head_dim=2,
            _state_pool_3d=torch.zeros(1, 1, 1),
            _kv_pool_view=torch.zeros(1, 1, 1),
            _state_block_table=None,
            _kv_block_table=None,
            _kv_eb=1,
            _cp_ctx=types.SimpleNamespace(cp_size=4, seq_len_full=full_len),
            _project_prefill=lambda x: x,
            _cp_profile_name=lambda _label: "test",
            _cp_role="main",
        )
        meta = types.SimpleNamespace(
            positions=torch.arange(full_len, dtype=torch.long),
            is_batched=True,
        )
        marker = object()

        launched = []

        def _launch_spy(kv_flat, score_flat, meta_arg, seq_start=None, **kwargs):
            launched.append((kv_flat, score_flat, meta_arg, seq_start, kwargs))

        harness._launch = _launch_spy

        _GATHER_SPY.reset()
        _GATHER_SPY.full = full
        result = COMP["forward"](harness, local, start_pos, meta=meta, workspace=marker)
        self.assertIsNone(result)

        # Exactly one ordinary full-gather start + wait + launch.
        self.assertEqual(len(_GATHER_SPY.start_calls), 1, "gather starts")
        self.assertEqual(len(_GATHER_SPY.wait_calls), 1, "gather waits")
        self.assertEqual(len(launched), 1, "launches")

        (args, kwargs) = _GATHER_SPY.start_calls[0]
        self.assertTrue(torch.equal(args[0], local), "gather payload")
        self.assertIs(args[1], harness._cp_ctx, "gather ctx")
        self.assertIsNone(args[2], "gather stream on CPU")
        self.assertIs(kwargs["workspace"], marker, "gather workspace")
        self.assertEqual(kwargs["cp_role"], "main", "gather role")
        self.assertIs(_GATHER_SPY.wait_calls[0], _GATHER_SPY.handle, "wait handle")

        kv_flat, score_flat, meta_arg, seq_start, _ = launched[0]
        self.assertTrue(
            torch.equal(torch.cat([kv_flat, score_flat], dim=1), full),
            "cat(kv,score) == gathered full",
        )
        self.assertIs(meta_arg, meta, "launch meta")
        self.assertIsNone(seq_start, "batched meta disables scalar seq_start")


def setUpModule():
    global ATT, COMP, IDX, _EB, _FP8_PKG, _KV_CACHE, _attn_stub_mod
    global _compressor_stub_mod, _cp_attention_shard, _indexer_stub_mod, _profiler
    global _swa_cp_byte_sliced, cp, fp8_kv_utils, fused_meta
    global kv_cache_utils, kvu, prefill_meta, swa_ops
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()

    cp, kv_cache_utils, swa_ops, fp8_kv_utils, fused_meta, _FP8_PKG = _load_dsv4_tree()
    kvu = kv_cache_utils

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
        "_flat_1d",
    ):
        setattr(_attn_stub_mod, _name, ATT[_name])

    _attn_stub_mod._build_swa_cp_byte_compaction = ATT["_build_swa_cp_byte_compaction"]

    sys.modules[_PREFIX + ".fp8.attention"] = _attn_stub_mod

    setattr(_FP8_PKG, "attention", _attn_stub_mod)

    arch_stub = _package("rtp_llm.models_py.utils.arch")
    arch_stub.is_sm120 = lambda device=None: False
    prefill_meta = _load(_PREFIX + ".fp8.prefill_meta", PREFILL_META_PATH)

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

    _StubAttention._validate_reusable_prefill_common = ATT[
        "_validate_reusable_prefill_common"
    ]

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
