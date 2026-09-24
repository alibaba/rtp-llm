# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-only FP4 prefill scoring with request-local sparse columns.

Adapted from vLLM's model_executor/kernels/attention/dsa/sparse_mqa_logits.py.
RTP has already restored request-local logical K order in PrefillIndexerKeys,
so every row starts at zero; prefix tokens are included in ``visible``.
Candidate sorting additionally removes duplicates to satisfy the installed
DeepGEMM metadata contract. Valid slots are an ascending prefix, followed by
the last valid block, as in vLLM. The metadata builder receives the compact
``end``, so only the unique prefix participates in its merge-path schedule.
The independent logical ``row_ke`` retains causal bounds for remapping.

The sparse scorer performs BF16 intermediate head reduction, unlike the old
FP32 dense scorer. Its result must be validated against that BF16 contract;
casting dense FP32 logits at the end is not an exact reference.

Plans are immutable snapshots of one request/chunk's candidates and bounds.
The caller owns their forward-local, byte-bounded cache and must invalidate
it whenever candidates, request/prefix bounds, or logical K order change.
No automatic cache keyed only by shape or device pointers is used here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_indexer import PrefillIndexerKeys

MAX_CHUNK_ROWS = 4096
MAX_BATCH_REQUESTS = 128
_WARMED_STREAMS: set[tuple[int, int]] = set()


@dataclass(frozen=True)
class SparsePrefillPlan:
    row_ks: torch.Tensor  # [M] int32 zeros, request-local packed K starts
    row_ke: torch.Tensor  # [M] int32 logical visible lengths, clamped to N
    sparse_indices: torch.Tensor  # [M, S] int32 sorted block IDs, last-ID padded
    end: torch.Tensor  # [M] int32 valid sparse-token columns, not logical length
    metadata: torch.Tensor  # DeepGEMM uint8 schedule; retains no input pointers
    key_count: int
    block_size: int
    row_key_offsets: torch.Tensor | None = None  # [M] token offsets into shared K slab

    @property
    def rows(self) -> int:
        return self.sparse_indices.shape[0]

    @property
    def sparse_columns(self) -> int:
        return self.sparse_indices.shape[1] * self.block_size

    @property
    def nbytes(self) -> int:
        """Owned plan buffers, including the relatively large DG schedule."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.row_ks,
                self.row_ke,
                self.sparse_indices,
                self.end,
                self.metadata,
            )
        ) + (
            0
            if self.row_key_offsets is None
            else self.row_key_offsets.numel() * self.row_key_offsets.element_size()
        )


@lru_cache(maxsize=1)
def _get_deep_gemm():
    try:
        import deep_gemm
    except ImportError:
        return None
    if not all(
        callable(getattr(deep_gemm, name, None))
        for name in ("get_sparse_mqa_logits_metadata", "fp8_fp4_sparse_mqa_logits")
    ):
        return None
    return deep_gemm


def _plan_supported(candidates, visible, key_count, block_size):
    if not (
        torch.version.hip is None
        and candidates.is_cuda
        and candidates.ndim == 2
        and candidates.dtype == torch.int32
        and 0 < candidates.shape[0] <= MAX_CHUNK_ROWS
        and 4 <= candidates.shape[1] <= 4096
        and candidates.shape[1] & (candidates.shape[1] - 1) == 0
        and candidates.stride(1) == 1
        and candidates.stride(0) >= candidates.shape[1]
        and visible.device == candidates.device
        and visible.ndim == 1
        and visible.numel() == candidates.shape[0]
        and visible.dtype in (torch.int32, torch.int64)
        and visible.stride(0) > 0
        and 0 < key_count <= 1 << 20
        and block_size == 8
    ):
        return False
    return (
        torch.cuda.get_device_capability(candidates.device)[0] == 10
        and _get_deep_gemm() is not None
    )


def _score_supported(q_payload, q_sf, keys, weights, rows, key_count, device):
    return (
        isinstance(keys, PrefillIndexerKeys)
        and q_payload.device == device
        and q_payload.dtype == torch.int8
        and q_payload.shape == (rows, 32, 64)
        and q_payload.is_contiguous()
        and q_sf.device == device
        and q_sf.dtype == torch.int32
        and q_sf.shape == (rows, 32)
        and q_sf.is_contiguous()
        and keys.quant.device == device
        and keys.quant.dtype == torch.int8
        and keys.quant.shape == (key_count, 64)
        and keys.quant.is_contiguous()
        and keys.scale.device == device
        and keys.scale.dtype == torch.int32
        and keys.scale.shape == (key_count,)
        and keys.scale.is_contiguous()
        and weights.device == device
        and weights.dtype in (torch.bfloat16, torch.float32)
        and weights.shape == (rows, 32)
        and weights.stride(1) == 1
        and weights.stride(0) >= 32
        and not (torch.is_grad_enabled() and weights.requires_grad)
    )


def _row_keys_supported(candidates, row_key_offsets, row_key_counts):
    if row_key_offsets is None and row_key_counts is None:
        return True
    return all(
        isinstance(value, torch.Tensor)
        and value.device == candidates.device
        and value.dtype == torch.int32
        and value.shape == (candidates.shape[0],)
        and value.is_contiguous()
        for value in (row_key_offsets, row_key_counts)
    )


def is_supported(
    q_payload: torch.Tensor,
    q_sf: torch.Tensor,
    keys: PrefillIndexerKeys,
    weights: torch.Tensor,
    candidates: torch.Tensor,
    visible: torch.Tensor,
    block_size: int = 8,
    topk: int = 512,
    *,
    row_key_offsets: torch.Tensor | None = None,
    row_key_counts: torch.Tensor | None = None,
) -> bool:
    """Whole scoring boundary gate; the caller separately gates DeepSelect.

    The production DeepSelect input stride needs 1024-byte alignment and its
    int32 output row needs 32-byte alignment. Plan-only tests may use smaller
    K through prepare_plan, but this gate rejects those selection layouts.
    """
    if (
        not isinstance(keys, PrefillIndexerKeys)
        or not _plan_supported(candidates, visible, len(keys), block_size)
        or not _row_keys_supported(candidates, row_key_offsets, row_key_counts)
    ):
        return False
    width = candidates.shape[1] * block_size
    return (
        0 < topk <= min(4096, width)
        and topk % 8 == 0
        and width * 2 % 1024 == 0
        and _score_supported(
            q_payload,
            q_sf,
            keys,
            weights,
            candidates.shape[0],
            len(keys),
            candidates.device,
        )
    )


# Prefix reuse changes the logical key count for every request. It is only a
# clamp bound; specializing it recompiles the expensive K2048 sorting network.
@triton.jit(
    do_not_specialize=[
        "KEY_COUNT",
        "request_start",
        "request_stop",
        "REQUESTS",
    ]
)
def _prepare_sparse_prefill_plan_kernel(
    candidates,
    visible,
    row_ks,
    row_ke,
    sparse_indices,
    sparse_end,
    CANDIDATE_STRIDE: tl.constexpr,
    VISIBLE_STRIDE: tl.constexpr,
    K: tl.constexpr,
    KEY_COUNT,
    BLOCK: tl.constexpr,
    key_offsets=None,
    key_counts=None,
    row_offsets=None,
    BATCHED: tl.constexpr = False,
    request_ids=None,
    request_start=0,
    request_stop=0,
    LOOKUP: tl.constexpr = False,
    REQUESTS=0,
    POSITIONS_RATIO: tl.constexpr = 0,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, K)
    candidate = tl.load(candidates + row * CANDIDATE_STRIDE + column)
    length = tl.load(visible + row * VISIBLE_STRIDE)
    if POSITIONS_RATIO:
        # Preserve the eager integer add's wrap before signed floor division.
        length = (length + 1).to(length.dtype)
        if POSITIONS_RATIO == 2:
            length = length >> 1
    key_offset = 0
    key_count = KEY_COUNT
    if BATCHED:
        if LOOKUP:
            request = tl.load(request_ids + row).to(tl.int64)
            in_group = (request >= request_start) & (request < request_stop)
            ids = tl.arange(0, 128)  # MAX_BATCH_REQUESTS; independent of batch size.
            counts = tl.load(key_counts + ids, ids < REQUESTS, other=0).to(tl.int64)
            padded = tl.cdiv(tl.maximum(counts, 0), 256) * 256
            key_offset = tl.sum(
                tl.where((ids >= request_start) & (ids < request), padded, 0)
            )
            key_count = tl.load(key_counts + request, in_group, other=-1).to(tl.int64)
        else:
            key_offset = tl.load(key_offsets + row).to(tl.int64)
            key_count = tl.load(key_counts + row).to(tl.int64)
        valid_request = (
            (key_offset >= 0)
            & (key_offset % BLOCK == 0)
            & (key_count >= 0)
            & (key_offset + key_count <= KEY_COUNT)
        )
        key_count = tl.where(valid_request, key_count, 0)
        key_offset = tl.where(valid_request, key_offset, 0)
        tl.store(row_offsets + row, key_offset)
    length = tl.minimum(tl.maximum(length, 0), key_count).to(tl.int32)
    tl.store(row_ks + row, 0)
    tl.store(row_ke + row, length)
    max_block = tl.cdiv(length, BLOCK)
    valid = (candidate >= 0) & (candidate < max_block)
    sentinel: tl.constexpr = 2147483647
    sorted_blocks = tl.sort(tl.where(valid, candidate, sentinel), descending=False)
    previous = tl.gather(sorted_blocks, tl.maximum(column - 1, 0), axis=0)
    unique = (sorted_blocks != sentinel) & ((column == 0) | (sorted_blocks != previous))
    compact_column = tl.cumsum(unique.to(tl.int32)) - 1
    valid_blocks = tl.sum(unique.to(tl.int32))
    last_block = tl.max(tl.where(unique, sorted_blocks, -1))
    tl.store(
        sparse_indices + row * K + compact_column,
        sorted_blocks + key_offset // BLOCK,
        unique,
    )
    # Prefix and padding stores target disjoint addresses; no second kernel.
    tl.store(
        sparse_indices + row * K + column,
        tl.where(valid_blocks > 0, last_block + key_offset // BLOCK, 0),
        column >= valid_blocks,
    )
    tail = tl.minimum(tl.maximum(length - last_block * BLOCK, 0), BLOCK)
    count = tl.where(valid_blocks > 0, (valid_blocks - 1) * BLOCK + tail, 0)
    tl.store(sparse_end + row, count)


def prepare_plan(
    candidates: torch.Tensor,
    visible: torch.Tensor,
    key_count: int,
    block_size: int = 8,
    *,
    row_key_offsets: torch.Tensor | None = None,
    row_key_counts: torch.Tensor | None = None,
    request_ids: torch.Tensor | None = None,
    request_key_counts: torch.Tensor | None = None,
    request_start: int = 0,
    request_stop: int | None = None,
    positions_ratio: int | None = None,
) -> SparsePrefillPlan | None:
    """Expand/sort/deduplicate one request-local chunk and build its DG schedule.

    Negative and out-of-causal-range candidate IDs are discarded. Repeated
    IDs represent one block, matching candidate-mask set membership. This
    operation stays on the device, including visible's int64-to-int32 clamp.
    A stream must run eager preparation once before CUDA graph capture,
    because the DG schedule builder initializes a per-stream workspace.

    For a shared K slab, supply both contiguous int32 [M] row_key_offsets
    (token offsets, aligned to block_size) and row_key_counts (request-local
    lengths). Candidate IDs and visible remain request-local. Invalid spans
    are masked empty on-device. The caller must supply the true request spans;
    the helper cannot infer their ownership from the concatenated slab.
    No dense [M, sum(request lengths)] scores or K copies are materialized.
    Ragged groups can instead look up GPU per-request tables via request_ids.
    K starts are derived from 256-token-padded counts from request_start.
    Only IDs in [request_start, request_stop) read their counts. Counts exclude
    alignment padding. The fixed request tile supports at most 128 requests.
    With positions_ratio=1 or 2, visible is a positions vector; compute
    (positions + 1) // ratio on-device in its original integer dtype.
    """
    if positions_ratio is not None and (
        type(positions_ratio) is not int or positions_ratio not in (1, 2)
    ):
        return None
    if not _plan_supported(
        candidates, visible, key_count, block_size
    ) or not _row_keys_supported(candidates, row_key_offsets, row_key_counts):
        return None
    lookup = request_ids is not None or request_key_counts is not None
    if lookup:
        if (
            row_key_offsets is not None
            or row_key_counts is not None
            or not isinstance(request_ids, torch.Tensor)
            or request_ids.device != candidates.device
            or request_ids.dtype not in (torch.int32, torch.int64)
            or request_ids.shape != (candidates.shape[0],)
            or not request_ids.is_contiguous()
            or not all(
                isinstance(value, torch.Tensor)
                and value.device == candidates.device
                and value.dtype == torch.int32
                and value.ndim == 1
                and value.is_contiguous()
                for value in (request_key_counts,)
            )
            or type(request_start) is not int
        ):
            return None
        request_stop = (
            request_key_counts.numel() if request_stop is None else request_stop
        )
        if (
            type(request_stop) is not int
            or not 0 <= request_start < request_stop <= request_key_counts.numel()
            or request_key_counts.numel() > MAX_BATCH_REQUESTS
        ):
            return None
    stream = torch.cuda.current_stream(candidates.device)
    stream_key = (stream.device.index, stream.cuda_stream)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and stream_key not in _WARMED_STREAMS:
        return None
    rows, count = candidates.shape
    batched = lookup or row_key_offsets is not None
    bounds = torch.empty(
        (4 if batched else 3, rows), dtype=torch.int32, device=candidates.device
    )
    row_ks, row_ke, sparse_end = bounds[:3].unbind(0)
    offsets = bounds[3] if batched else None
    indices = torch.empty((rows, count), dtype=torch.int32, device=candidates.device)
    _prepare_sparse_prefill_plan_kernel[(rows,)](
        candidates,
        visible,
        row_ks,
        row_ke,
        indices,
        sparse_end,
        candidates.stride(0),
        visible.stride(0),
        count,
        key_count,
        block_size,
        POSITIONS_RATIO=positions_ratio or 0,
        **(
            {
                "row_offsets": offsets,
                "BATCHED": True,
                "LOOKUP": True,
                "request_ids": request_ids,
                "key_counts": request_key_counts,
                "request_start": request_start,
                "request_stop": request_stop,
                "REQUESTS": request_key_counts.numel(),
            }
            if lookup
            else (
                {
                    "key_offsets": row_key_offsets,
                    "key_counts": row_key_counts,
                    "row_offsets": offsets,
                    "BATCHED": True,
                }
                if batched
                else {}
            )
        ),
        num_warps=16 if count >= 1024 else 4,
    )
    # The pinned DeepGEMM non-paged metadata ABI uses (ke - ks) ONLY to
    # derive ceil_div(compact_length, BLOCK). Actual K addresses come from
    # sparse_indices, and the scoring kernel is bounded by key_count.
    # Give it compact length, not logical visibility: repeated padding
    # violates its unique-prefix merge-path contract and can assert.
    # See scheduler/sm100_sparse_mqa_logits_metadata.cuh:get_num_kv_blocks.
    # row_ke remains the logical causal boundary for selection/remapping.
    metadata = _get_deep_gemm().get_sparse_mqa_logits_metadata(
        row_ks,
        sparse_end,
        key_count,
        indices,
        torch.int8,
        block_size,
        use_unaligned_ks=True,
    )
    if not capturing:
        _WARMED_STREAMS.add(stream_key)
    return SparsePrefillPlan(
        row_ks, row_ke, indices, sparse_end, metadata, key_count, block_size, offsets
    )


def score(
    q_payload: torch.Tensor,
    q_sf: torch.Tensor,
    keys: PrefillIndexerKeys,
    weights: torch.Tensor,
    plan: SparsePrefillPlan,
) -> torch.Tensor | None:
    """Compute BF16 sparse logits; padding outside plan.end is not a valid score.

    FP4 payload and packed group-32 UE8M0 scales are consumed unchanged.
    Head weights are converted to BF16 (not multiplied by Q scales). This
    precision change follows vLLM's sparse contract and is intentional.
    """
    if not (
        isinstance(plan, SparsePrefillPlan)
        and q_payload.is_cuda
        and _score_supported(
            q_payload,
            q_sf,
            keys,
            weights,
            plan.rows,
            plan.key_count,
            plan.sparse_indices.device,
        )
    ):
        return None
    return _get_deep_gemm().fp8_fp4_sparse_mqa_logits(
        (q_payload, q_sf),
        (keys.quant, keys.scale),
        weights.to(torch.bfloat16),
        plan.metadata,
        plan.sparse_indices.shape[1],
        plan.block_size,
        use_unaligned_ks=True,
    )


@triton.jit
def _remap_sparse_prefill_topk_kernel(
    columns,
    indices,
    sparse_end,
    visible,
    logits,
    output,
    COLUMN_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    LOGITS_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    SPARSE_COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_LOGITS: tl.constexpr,
    TILE: tl.constexpr,
    key_offsets=None,
    BATCHED: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, TILE)
    column = tl.load(columns + row * COLUMN_STRIDE + slot, slot < TOPK, other=-1)
    count = tl.load(sparse_end + row)
    length = tl.load(visible + row)
    valid = (slot < TOPK) & (column >= 0) & (column < count) & (column < SPARSE_COLUMNS)
    block = tl.load(indices + row * INDEX_STRIDE + column // BLOCK, valid, other=0)
    position = block * BLOCK + column % BLOCK
    if BATCHED:
        position -= tl.load(key_offsets + row)
    valid = valid & (position >= 0) & (position < length)
    if HAS_LOGITS:
        value = tl.load(
            logits + row * LOGITS_STRIDE + column, valid, other=float("nan")
        ).to(tl.float32)
        valid = valid & (tl.abs(value) < float("inf"))
    tl.store(
        output + row * OUTPUT_STRIDE + slot, tl.where(valid, position, -1), slot < TOPK
    )


def remap(
    columns: torch.Tensor,
    plan: SparsePrefillPlan,
    *,
    logits: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Map selected sparse columns back to request-local compressed positions.

    -1, out-of-prefix, and causal-tail positions remain -1. Passing logits
    also preserves RTP's filtering of selected NaN and +/-inf scores. The
    caller can supply a chunk slice of its int32 final output buffer.
    """
    if not (
        isinstance(plan, SparsePrefillPlan)
        and columns.is_cuda
        and columns.device == plan.sparse_indices.device
        and columns.dtype == torch.int32
        and columns.ndim == 2
        and columns.shape[0] == plan.rows
        and 0 < columns.shape[1] <= 4096
        and columns.stride(1) == 1
        and columns.stride(0) >= columns.shape[1]
    ):
        return None
    if logits is not None and not (
        logits.device == columns.device
        and logits.dtype == torch.bfloat16
        and logits.shape == (plan.rows, plan.sparse_columns)
        and logits.stride(1) == 1
        and logits.stride(0) >= plan.sparse_columns
    ):
        return None
    if out is not None and not (
        out.device == columns.device
        and out.dtype == torch.int32
        and out.shape == columns.shape
        and out.stride(1) == 1
        and out.stride(0) >= out.shape[1]
    ):
        return None
    if out is None:
        out = torch.empty(columns.shape, dtype=torch.int32, device=columns.device)
    _remap_sparse_prefill_topk_kernel[(plan.rows,)](
        columns,
        plan.sparse_indices,
        plan.end,
        plan.row_ke,
        logits,
        out,
        columns.stride(0),
        plan.sparse_indices.stride(0),
        logits.stride(0) if logits is not None else 0,
        out.stride(0),
        columns.shape[1],
        plan.sparse_columns,
        plan.block_size,
        logits is not None,
        triton.next_power_of_2(columns.shape[1]),
        **(
            {"key_offsets": plan.row_key_offsets, "BATCHED": True}
            if plan.row_key_offsets is not None
            else {}
        ),
    )
    return out


def _joined_keys(keys):
    """Metadata-only alias of one bounded pool slab, never concatenate K."""
    if not keys or any(not isinstance(key, PrefillIndexerKeys) for key in keys):
        return None
    first = keys[0]
    if any(
        key.quant.dtype != torch.int8
        or key.scale.dtype != torch.int32
        or key.quant.shape != (len(key), 64)
        or key.scale.shape != (len(key),)
        or not key.quant.is_contiguous()
        or not key.scale.is_contiguous()
        or key.quant.device != first.quant.device
        or key.scale.device != first.quant.device
        for key in keys
    ):
        return None
    from ._v41_grouped_prefill_score import _slab_view

    quant = _slab_view([key.quant for key in keys], padded=True)
    scale = _slab_view([key.scale for key in keys], padded=True)
    if (
        quant is None
        or scale is None
        or not 0 < scale.numel() <= 1 << 20
        or first.scale.data_ptr() % 16
    ):
        return None
    offsets = tuple(
        key.scale.storage_offset() - first.scale.storage_offset() for key in keys
    )
    return (
        PrefillIndexerKeys(quant, scale),
        offsets,
    )


def _batch_groups(keys, slices, rows):
    """Bound query rows and K slabs; ragged groups use GPU request tables."""
    if len(keys) != len(slices) or not slices:
        return None
    cursor, groups, pieces = 0, [], []
    for request, span in enumerate(slices):
        if (
            not isinstance(span, slice)
            or span.step not in (None, 1)
            or span.start != cursor
            or not isinstance(span.stop, int)
            or not cursor <= span.stop <= rows
        ):
            return None
        cursor = span.stop
        start = span.start
        while start < span.stop:
            if pieces and (
                _joined_keys([keys[p[0]] for p in pieces] + [keys[request]]) is None
            ):
                groups.append(tuple(pieces))
                pieces = []
            capacity = MAX_CHUNK_ROWS - (pieces[-1][2] - pieces[0][1] if pieces else 0)
            stop = min(span.stop, start + capacity)
            pieces.append((request, start, stop))
            start = stop
            if stop - pieces[0][1] == MAX_CHUNK_ROWS:
                groups.append(tuple(pieces))
                pieces = []
    if pieces:
        groups.append(tuple(pieces))
    if cursor != rows:
        return None
    result = []
    for pieces in groups:
        joined = _joined_keys([keys[p[0]] for p in pieces])
        if joined is None:
            return None
        slab, offsets = joined
        result.append((pieces, slab, offsets))
    return result


def try_batched_sparse(
    q,
    sf,
    weights,
    globals_by_req,
    slices,
    positions,
    ratio,
    candidates,
    block_size,
    topk,
    shared,
    out,
    *,
    req_ids=None,
    key_counts=None,
) -> bool:
    """One sparse plan/scorer/selector/remap per bounded packed query group.

    False is returned only during metadata preflight, before device work or
    output writes. Request order, K order and positions must remain immutable
    for the caller's forward-local shared dictionary, as for single plans.
    """
    if (
        not isinstance(q, torch.Tensor)
        or not q.is_cuda
        or q.ndim != 3
        or not globals_by_req
        or not 2 <= len(globals_by_req) <= MAX_BATCH_REQUESTS
        or slices is None
        or ratio not in (1, 2)
        or topk != 512
        or block_size != 8
        or not isinstance(candidates, torch.Tensor)
        or candidates.ndim != 2
        or candidates.shape[0] != q.shape[0]
        or positions.device != q.device
        or positions.shape != (q.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.stride(0) <= 0
        or out.device != q.device
        or out.dtype != torch.int32
        or out.shape != (q.shape[0], topk)
        or out.stride(1) != 1
        or out.stride(0) < topk
    ):
        return False
    keys = [entry[1] for entry in globals_by_req]
    if any(not isinstance(key, PrefillIndexerKeys) for key in keys):
        return False
    if (
        not isinstance(req_ids, torch.Tensor)
        or req_ids.device != q.device
        or req_ids.dtype not in (torch.int32, torch.int64)
        or req_ids.shape != (q.shape[0],)
        or not req_ids.is_contiguous()
        or not all(
            isinstance(value, torch.Tensor)
            and value.device == q.device
            and value.dtype == torch.int32
            and value.shape == (len(keys),)
            and value.is_contiguous()
            for value in (key_counts,)
        )
    ):
        return False
    groups = _batch_groups(keys, slices, q.shape[0])
    if not groups:
        return False
    from . import _v41_deepselect as deepselect

    if not deepselect.is_available(q.device):
        return False
    stream = torch.cuda.current_stream(q.device)
    if (
        torch.cuda.is_current_stream_capturing()
        and (stream.device.index, stream.cuda_stream) not in _WARMED_STREAMS
    ):
        return False
    for pieces, slab, offsets in groups:
        span = slice(pieces[0][1], pieces[-1][2])
        first = pieces[0][0]
        expected = tuple(
            sum((len(keys[b]) + 255) // 256 * 256 for b in range(first, p[0]))
            for p in pieces
        )
        if offsets != expected:
            return False
        if not is_supported(
            q[span],
            sf[span],
            slab,
            weights[span],
            candidates[span],
            positions[span],
            block_size,
            topk,
        ):
            return False

    source = shared.get("candidates", candidates)
    if source is not candidates:
        return False
    cache = shared.get("prefill_sparse_plans")
    if cache is None or cache[0] is not source:
        cache = [source, {}, 0]
        shared["prefill_sparse_plans"] = cache
    limit = max(
        0, int(os.environ.get("DSV41_SPARSE_PREFILL_PLAN_MAX_BYTES", 256 * 1024**2))
    )
    for pieces, slab, offsets in groups:
        start, stop = pieces[0][1], pieces[-1][2]
        span = slice(start, stop)
        counts = tuple(len(keys[p[0]]) for p in pieces)
        key = (
            "batched",
            pieces,
            offsets,
            counts,
            ratio,
            block_size,
            # Tensor keys use object identity and retain these small immutable
            # descriptors, preventing allocator address reuse from hitting a
            # stale plan. Actual K bytes/addresses do not affect the schedule.
            positions,
            positions.stride(0),
            req_ids,
            key_counts,
        )
        plan = cache[1].get(key)
        if plan is None:
            plan = prepare_plan(
                candidates[span],
                positions[span],
                len(slab),
                block_size,
                request_ids=req_ids[span],
                request_key_counts=key_counts,
                request_start=pieces[0][0],
                request_stop=pieces[-1][0] + 1,
                positions_ratio=ratio,
            )
            if plan is None:
                raise RuntimeError("batched sparse plan rejected after preflight")
            if cache[2] + plan.nbytes <= limit:
                cache[1][key] = plan
                cache[2] += plan.nbytes
        logits = score(q[span], sf[span], slab, weights[span], plan)
        if logits is None:
            raise RuntimeError("batched sparse scorer rejected after preflight")
        columns = deepselect.try_select_sparse_tokens(logits, plan.end)
        if (
            columns is None
            or remap(columns, plan, logits=logits, out=out[span]) is None
        ):
            raise RuntimeError(
                "batched sparse selection/remap rejected after preflight"
            )
        del logits, columns, plan
    return True
