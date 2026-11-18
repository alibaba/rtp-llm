"""BF16 Indexer scoring over a forward-owned CP8 shared-prefix page bank.

This adapter changes K addressing, not Q quantization, head reduction,
candidate publication or token selection. Native page256 MXFP4 contains the
64-byte payload plane followed by the packed four-byte SF plane. Dense and
sparse calls use the installed weighted BF16 DeepGEMM contract unchanged.
Unsupported admission is prelaunch; execution failures never retry serially.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import torch

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.fp8._v41_sparse_prefill_indexer import (
    SparsePrefillPlan,
)
from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    SharedNativeCache,
    _source_tensor_key,
)

PAGE, HEADS, PACKED_DIM = 256, 32, 64


def _raw(value):
    return (
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and not value.is_neg()
        and not value.is_conj()
        and not value.requires_grad
    )


def _device_supported(tensor):
    return (
        tensor.is_cuda
        and torch.version.hip is None
        and _quant.is_ppu()
        and torch.cuda.get_device_capability(tensor.device) == (8, 9)
        and not torch.cuda.is_current_stream_capturing()
    )


@lru_cache(maxsize=1)
def _deep_gemm():
    try:
        import deep_gemm
    except ImportError:
        return None
    names = (
        "get_paged_mqa_logits_metadata",
        "fp8_fp4_paged_mqa_logits",
        "get_paged_sparse_mqa_logits_metadata",
        "fp8_fp4_paged_sparse_mqa_logits",
        "get_num_sms",
    )
    return (
        deep_gemm
        if all(callable(getattr(deep_gemm, name, None)) for name in names)
        else None
    )


def _bank_supported(bank, state, ratio):
    from rtp_llm.models_py.modules.dsv41.ppu.cp_prefill_shared_indexer import (
        SharedIndexerBank,
    )

    if (
        type(bank) is not SharedIndexerBank
        or bank.cache is not state.cache
        or not bank.valid()
        or type(ratio) is not int
        or ratio not in (1, 2)
        or state.cache.geometry.ratio(bank.source) != ratio
        or len(bank.request_counts) != 8
        or bank.request_counts != tuple(count // ratio for count in state.cache.lengths)
        or type(bank.prefix_rows) is not int
        or bank.prefix_rows <= 0
        or bank.prefix_rows % PAGE
        or any(
            type(count) is not int or not bank.prefix_rows <= count <= 131072
            for count in bank.request_counts
        )
    ):
        return False
    pages, table, host = bank.pages, bank.request_table, bank.request_table_host
    width = max((count + PAGE - 1) // PAGE for count in bank.request_counts)
    return (
        _raw(pages)
        and pages.dtype == torch.uint8
        and pages.ndim == 4
        and pages.shape[0] > 0
        and pages.shape[1:] == (PAGE, 1, 68)
        and pages.stride(1) == pages.stride(2) == 68
        and pages.stride(3) == 1
        and pages.stride(0) >= PAGE * 68
        and pages.stride(0) % 512 == 0
        and pages.data_ptr() % 16 == 0
        and _raw(table)
        and table.device == pages.device
        and table.dtype == torch.int32
        and table.shape == (8, width)
        and table.is_contiguous()
        and isinstance(host, tuple)
        and len(host) == 8
        and all(
            isinstance(row, tuple)
            and len(row) == width
            and all(type(page) is int and 0 <= page < pages.shape[0] for page in row)
            for row in host
        )
    )


def _scope_key(state, request_slices, positions, request_ids):
    """Authenticate host ragged row geometry and the current CP tensor owners."""
    cp, cache = getattr(state, "context", None), getattr(state, "cache", None)
    runtime = getattr(state, "runtime", None)
    model, policy = getattr(runtime, "model", None), getattr(runtime, "policy", None)
    if (
        not getattr(state, "prefill", False)
        or getattr(state, "idle", True)
        or torch.is_grad_enabled()
        or any(
            getattr(state, flag, False)
            for flag in ("is_graph", "is_graph_commit", "verify", "proposal")
        )
        or getattr(getattr(state, "attn", None), "is_cuda_graph", False)
        or type(cache) is not SharedNativeCache
        or not cache._is_prefill
        or not cache.sharded
        or not isinstance(cp, CPContext)
        or cp.cp_size != 8
        or not cp.kv_cache_sharded
        or type(cp.cp_rank) is not int
        or not 0 <= cp.cp_rank < 8
        or model is None
        or getattr(model, "_is_speculative", False)
        or getattr(model, "_is_decode_role", False)
        or getattr(model, "_prefill_cp_size", None) != 8
        or getattr(getattr(model, "parallelism_config", None), "tp_size", None) != 8
        or getattr(policy, "mode", None) != "compact"
        or getattr(policy, "keep_tokens", None) != 140
        or getattr(policy, "window", None) != 128
        or getattr(policy, "max_rollback_tokens", 0) != 0
        or getattr(policy, "encoder_replay", False)
    ):
        return None
    chunks, lengths, prefixes = (
        cp.chunk_lengths_per_req,
        state.lengths,
        state.query_prefixes,
    )
    if (
        not all(
            isinstance(values, tuple) and len(values) == 8
            for values in (chunks, lengths, prefixes)
        )
        or any(
            type(value) is not int or value < 0
            for value in (*chunks, *lengths, *prefixes)
        )
        or cache.lengths != tuple(p + n for p, n in zip(prefixes, lengths))
        or cp.input_lengths_global_host != lengths
        or cp.prefix_lengths_host != prefixes
        or sum(chunks) <= 0
        or cp.chunk_length != sum(chunks)
        or cp.padded_seq_len != sum(chunks) * 8
    ):
        return None
    expected, start = [], 0
    for chunk in chunks:
        expected.append(slice(start, start + chunk))
        start += chunk
    if (
        request_slices != tuple(expected)
        or not all(_raw(value) for value in (positions, request_ids, cp.local_is_real))
        or positions is not cp.global_positions
        or request_ids is not cp.req_id_per_token
        or positions.shape != (start,)
        or positions.dtype != torch.int64
        or not positions.is_contiguous()
        or request_ids.shape != (start,)
        or request_ids.dtype not in (torch.int32, torch.int64)
        or not request_ids.is_contiguous()
        or request_ids.device != positions.device
        or cp.local_is_real.shape != (start,)
        or cp.local_is_real.dtype != torch.bool
        or not cp.local_is_real.is_contiguous()
        or cp.local_is_real.device != positions.device
    ):
        return None
    from rtp_llm.models_py.modules.dsv41.ppu.layers import cp8_yoco_operator_eligible

    if not cp8_yoco_operator_eligible(state):
        return None
    return (
        id(cp),
        bool(state.compacted),
        chunks,
        lengths,
        prefixes,
        id(cache),
        tuple(
            _source_tensor_key(value)
            for value in (positions, request_ids, cp.local_is_real)
        ),
    )


def _row_keys(rows):
    return tuple(
        _source_tensor_key(value)
        for value in (rows.request_ids, rows.tables, rows.visible)
    )


@dataclass(frozen=True)
class CPIndexerRows:
    state: object
    bank: object
    positions: torch.Tensor
    source_request_ids: torch.Tensor
    request_slices: tuple[slice, ...]
    ratio: int
    scope_key: tuple
    request_ids_host: tuple[int, ...]
    request_ids: torch.Tensor
    tables: torch.Tensor
    visible: torch.Tensor
    keys: tuple
    _dense: dict = field(default_factory=dict, repr=False, compare=False)
    _sparse: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def count(self):
        return len(self.request_ids_host)

    @property
    def width(self):
        return max(
            self.bank.request_counts[request] for request in self.request_ids_host
        )

    def valid(self):
        return (
            self.bank.valid()
            and self.scope_key
            == _scope_key(
                self.state,
                self.request_slices,
                self.positions,
                self.source_request_ids,
            )
            and self.keys == _row_keys(self)
        )

    def slice(self, chunk):
        """A nonempty contiguous row tile, preserving true request IDs and mapping."""
        if not self.valid():
            raise RuntimeError("stale CP paged Indexer row publication")
        if not isinstance(chunk, slice) or chunk.step not in (None, 1):
            raise ValueError("CP paged Indexer rows require a contiguous slice")
        start, stop, _ = chunk.indices(self.count)
        if not 0 <= start < stop <= self.count:
            raise ValueError("CP paged Indexer row slice is empty")
        ids, tables, visible = (
            self.request_ids[start:stop],
            self.tables[start:stop],
            self.visible[start:stop],
        )
        provisional = CPIndexerRows(
            self.state,
            self.bank,
            self.positions,
            self.source_request_ids,
            self.request_slices,
            self.ratio,
            self.scope_key,
            self.request_ids_host[start:stop],
            ids,
            tables,
            visible,
            (),
        )
        return CPIndexerRows(
            self.state,
            self.bank,
            self.positions,
            self.source_request_ids,
            self.request_slices,
            self.ratio,
            self.scope_key,
            provisional.request_ids_host,
            ids,
            tables,
            visible,
            _row_keys(provisional),
        )


def can_prepare_rows(state, *, request_slices, positions, request_ids):
    """Allocation-free capability handshake before cache.prepare_source()."""
    return (
        _scope_key(state, request_slices, positions, request_ids) is not None
        and _device_supported(positions)
        and _deep_gemm() is not None
    )


def prepare_rows(state, bank, *, request_slices, positions, request_ids, ratio):
    """Prelaunch eager CP8/YOCO140 admission; no collectives or device downloads.

    CPContext's trusted host chunks describe arbitrary ragged request rows.
    IDs are constructed from that same host geometry, so sparse paired rows
    can never borrow another request's page table. A bank with an unaligned
    shared prefix is rejected before metadata allocation or provider launch.
    """
    scope = _scope_key(state, request_slices, positions, request_ids)
    if (
        scope is None
        or not _bank_supported(bank, state, ratio)
        or positions.device != bank.pages.device
        or not _device_supported(positions)
        or _deep_gemm() is None
    ):
        return None
    host_ids = tuple(
        request
        for request, chunk in enumerate(state.context.chunk_lengths_per_req)
        for _ in range(chunk)
    )
    from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

    # The existing pinned allocator records each asynchronous copy before
    # releasing its host storage. Keep uploads on the current scoring stream;
    # pageable torch.tensor(..., device=...) would drain that stream twice.
    ids = _upload(host_ids, positions.device, torch.int32)
    tables = bank.request_table.index_select(0, ids).contiguous()
    counts = _upload(bank.request_counts, positions.device, torch.int64).index_select(
        0, ids
    )
    # CP padding positions are already clamped inside their own request. Keep
    # the same score window as the existing candidate/TopK consumer, then let
    # the unchanged downstream valid-row mask discard the padding outputs.
    visible = ((positions + 1) // ratio).clamp_min(0)
    visible = torch.minimum(visible, counts).to(torch.int32).contiguous()
    provisional = CPIndexerRows(
        state,
        bank,
        positions,
        request_ids,
        request_slices,
        ratio,
        scope,
        host_ids,
        ids,
        tables,
        visible,
        (),
    )
    return CPIndexerRows(
        state,
        bank,
        positions,
        request_ids,
        request_slices,
        ratio,
        scope,
        host_ids,
        ids,
        tables,
        visible,
        _row_keys(provisional),
    )


def _query_supported(q_payload, q_sf, weights, rows):
    if not isinstance(rows, CPIndexerRows) or not rows.valid():
        return False
    values = q_payload, q_sf, weights
    return (
        all(_raw(value) and value.device == rows.bank.pages.device for value in values)
        and q_payload.dtype == torch.int8
        and q_payload.shape == (rows.count, HEADS, PACKED_DIM)
        and q_payload.is_contiguous()
        and q_payload.data_ptr() % 16 == 0
        and q_sf.dtype == torch.int32
        and q_sf.shape == (rows.count, HEADS)
        and q_sf.is_contiguous()
        and q_sf.data_ptr() % 16 == 0
        and weights.dtype in (torch.bfloat16, torch.float32)
        and weights.shape == (rows.count, HEADS)
        and weights.stride(1) == 1
        and weights.stride(0) >= HEADS
        and weights.data_ptr() % 16 == 0
        and all(
            value.untyped_storage().data_ptr()
            != rows.bank.pages.untyped_storage().data_ptr()
            for value in values
        )
        and _device_supported(q_payload)
    )


def _check_logits(logits, rows, width):
    if (
        not isinstance(logits, torch.Tensor)
        or logits.device != rows.bank.pages.device
        or logits.dtype != torch.bfloat16
        or logits.shape != (rows.count, width)
        or logits.stride(1) != 1
        or logits.stride(0) < width
    ):
        raise RuntimeError("unexpected CP paged Indexer BF16 logits ABI")
    return logits


def score_dense(q_payload, q_sf, weights, rows):
    """Raw BF16 request-local columns; callers mask/pool before disposable TopK."""
    if not _query_supported(q_payload, q_sf, weights, rows):
        return None
    dg = _deep_gemm()
    if dg is None:
        return None
    from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_indexer_layout

    flat = cp_prefill_indexer_layout.score(
        q_payload, q_sf, weights, rows, getattr(dg, "fp8_fp4_mqa_logits", None),
    )
    if flat is not None:
        return _check_logits(flat, rows, rows.width)
    if not rows._dense:
        lengths = rows.visible[:, None].contiguous()
        schedule = dg.get_paged_mqa_logits_metadata(
            lengths,
            PAGE,
            dg.get_num_sms(),
            indices=rows.request_ids,
            metadata_extra=(1, HEADS, PACKED_DIM * 2, torch.int8),
        )
        rows._dense.update(
            lengths=lengths,
            schedule=schedule,
            keys=tuple(_source_tensor_key(value) for value in (lengths, schedule)),
        )
    lengths, schedule = rows._dense["lengths"], rows._dense["schedule"]
    if rows._dense["keys"] != tuple(
        _source_tensor_key(value) for value in (lengths, schedule)
    ):
        raise RuntimeError("stale CP paged dense metadata")
    logits = dg.fp8_fp4_paged_mqa_logits(
        (
            q_payload.view(rows.count, 1, HEADS, PACKED_DIM),
            q_sf.view(rows.count, 1, HEADS),
        ),
        rows.bank.pages,
        weights.to(torch.bfloat16).contiguous(),
        lengths,
        rows.tables,
        schedule,
        rows.width,
        indices=rows.request_ids,
    )
    return _check_logits(logits, rows, rows.width)


def _sparse_keys(plan):
    return tuple(
        _source_tensor_key(value)
        for value in (plan.row_ks, plan.row_ke, plan.sparse_indices, plan.end)
    )


def score_sparse(q_payload, q_sf, weights, rows, plan):
    """Use RTP's existing local candidate compiler and unchanged remap plan.

    `plan.end` is a compact slot count for native metadata. `plan.row_ke`
    remains logical causal visibility for the existing selection/remap path.
    Padded duplicate IDs beyond that compact prefix must never be scheduled.
    """
    if (
        not _query_supported(q_payload, q_sf, weights, rows)
        or not isinstance(plan, SparsePrefillPlan)
        or rows.ratio != 1
        or rows.bank.source != 20
        or plan.rows != rows.count
        or plan.block_size != 8
        or plan.sparse_indices.shape[1] != 2048
        or plan.row_key_offsets is not None
        or plan.key_count < rows.width
        or not all(
            _raw(value)
            and value.device == rows.bank.pages.device
            and value.is_contiguous()
            and value.dtype == torch.int32
            for value in (plan.row_ks, plan.row_ke, plan.sparse_indices, plan.end)
        )
        or plan.row_ke.shape != plan.end.shape
        or plan.row_ks.shape != plan.end.shape
        or plan.end.shape != (rows.count,)
    ):
        return None
    dg = _deep_gemm()
    if dg is None:
        return None
    key, current = id(plan), _sparse_keys(plan)
    if key not in rows._sparse:
        metadata = dg.get_paged_sparse_mqa_logits_metadata(
            plan.end,
            rows.tables,
            rows.request_ids,
            PAGE,
            plan.sparse_indices,
            torch.int8,
            plan.block_size,
        )
        rows._sparse[key] = plan, current, metadata, _source_tensor_key(metadata)
    owner, stamp, metadata, metadata_stamp = rows._sparse[key]
    if (
        owner is not plan
        or current != stamp
        or metadata_stamp != _source_tensor_key(metadata)
    ):
        raise RuntimeError("stale CP paged sparse metadata")
    logits = dg.fp8_fp4_paged_sparse_mqa_logits(
        (
            q_payload.view(rows.count, 1, HEADS, PACKED_DIM),
            q_sf.view(rows.count, 1, HEADS),
        ),
        rows.bank.pages,
        weights.to(torch.bfloat16).contiguous(),
        metadata,
        2048,
        plan.block_size,
    )
    return _check_logits(logits, rows, plan.sparse_columns)
