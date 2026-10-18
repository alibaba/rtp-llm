# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 decode index scoring directly from the source layer's packed FP4 cache.

The cache has grouped storage within each block: all 64-byte FP4 payload
rows, then all packed-UE8M0 int32 scales. Its nominal [blocks, entries, 68]
view is not a per-row payload/scale interleave. No full-capacity
dequantization or [T,H,K] score is made; DeepGEMM's MX-mode paged kernel
reads the fused bytes directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._indexer_score import PagedMQAMetadata


@dataclass(frozen=True)
class DecodeIndexerKeys:
    """The owning layer's physical cache plus its logical compressed coverage."""

    pool: torch.Tensor
    block_table: torch.Tensor
    capacity: int
    logical_entries_per_block: int


@dataclass(frozen=True)
class DecodeIndexerMetadata:
    lengths: torch.Tensor
    safe_table: torch.Tensor
    paged: PagedMQAMetadata


# Adapted from vLLM's v1/attention/ops/metadata.py (Apache-2.0): fixed-span
# request mapping, with RTP's sentinel pages and logical/physical page geometry.
@triton.jit
def _fused_decode_metadata_kernel(
    positions,
    table,
    lengths,
    physical_lengths,
    safe_table,
    expanded_table,
    requests,
    QUERY_LEN: tl.constexpr,
    RATIO: tl.constexpr,
    LOGICAL: tl.constexpr,
    PHYSICAL: tl.constexpr,
    COLS: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    request = token // QUERY_LEN
    length = ((tl.load(positions + token).to(tl.int64) + 1) // RATIO).to(tl.int32)
    last = tl.maximum(length - 1, 0)
    extent = (last // LOGICAL) * PHYSICAL + last % LOGICAL + 1
    tl.store(lengths + token, length)
    tl.store(physical_lengths + token, tl.where(length > 0, extent, 0))
    tl.store(requests + token, request)
    columns = tl.arange(0, BLOCK)
    blocks = tl.load(
        table + request * STRIDE0 + columns * STRIDE1, columns < COLS, other=0
    )
    blocks = tl.maximum(blocks, 0)
    tl.store(expanded_table + token * COLS + columns, blocks, columns < COLS)
    if token % QUERY_LEN == 0:
        tl.store(safe_table + request * COLS + columns, blocks, columns < COLS)


def get_decode_metadata(
    shared: dict,
    keys: DecodeIndexerKeys,
    positions: torch.Tensor,
    *,
    batch_size: int,
    query_len: int,
    compress_ratio: int,
) -> DecodeIndexerMetadata | None:
    """Share within one forward; captured producers rerun on every graph replay.

    The first model layer clears ``decode_indexer_metadata`` before capture/eager
    forwards. Graph replay reruns the captured producers on live device inputs.
    Input references prevent allocator reuse from aliasing a cached key. Pools
    are deliberately not cached: different owners may share the same page table.
    Without RoPE position reuse, retain the uncached fallback instead of pinning
    duplicate metadata for each source layer's fresh position conversion.
    """
    table, pool = keys.block_table, keys.pool
    rows = batch_size * query_len
    if (
        os.environ.get("DSV41_FUSE_INDEXER_METADATA", "1") != "1"
        or os.environ.get("DSV41_REUSE_DECODE_ROPE", "1") != "1"
        or batch_size <= 0
        or query_len <= 0
        or compress_ratio not in (1, 2)
        or not positions.is_cuda
        or positions.shape != (rows,)
        or positions.dtype not in (torch.int32, torch.int64)
        or not positions.is_contiguous()
        or table.device != positions.device
        or table.ndim != 2
        or table.shape[0] != batch_size
        or table.shape[1] <= 0
        or table.dtype not in (torch.int32, torch.int64)
        or pool.device != positions.device
        or pool.dtype != torch.uint8
        or pool.ndim != 3
        or pool.shape[2] != 68
        or not pool.is_contiguous()
        or keys.logical_entries_per_block not in (64, 128)
        or keys.logical_entries_per_block > pool.shape[1]
        or not 0 < keys.capacity <= table.shape[1] * keys.logical_entries_per_block
        or not is_supported(positions.device, pool.shape[1])
    ):
        return None
    from ._indexer_score import PagedMQAMetadata, _deep_gemm, _get_num_sms

    num_sms = _get_num_sms(positions.device)
    key = (
        positions.data_ptr(),
        positions.shape,
        positions.stride(),
        positions.dtype,
        positions.device,
        table.data_ptr(),
        table.shape,
        table.stride(),
        table.dtype,
        query_len,
        compress_ratio,
        keys.logical_entries_per_block,
        pool.shape[1],
        keys.capacity,
        num_sms,
    )
    cache = shared.setdefault("decode_indexer_metadata", {})
    cached = cache.get(key)
    if cached is not None:
        return cached[2]
    options = dict(device=positions.device, dtype=torch.int32)
    lengths = torch.empty(rows, **options)
    physical_lengths = torch.empty((rows, 1), **options)
    safe_table = torch.empty(table.shape, **options)
    expanded_table = torch.empty((rows, table.shape[1]), **options)
    requests = torch.empty(rows, **options)
    _fused_decode_metadata_kernel[(rows,)](
        positions,
        table,
        lengths,
        physical_lengths,
        safe_table,
        expanded_table,
        requests,
        query_len,
        compress_ratio,
        keys.logical_entries_per_block,
        pool.shape[1],
        table.shape[1],
        table.stride(0),
        table.stride(1),
        triton.next_power_of_2(table.shape[1]),
        num_warps=4,
    )
    schedule = _deep_gemm.get_paged_mqa_logits_metadata(
        physical_lengths, pool.shape[1], num_sms, indices=requests
    )
    result = DecodeIndexerMetadata(
        lengths,
        safe_table,
        PagedMQAMetadata(physical_lengths, expanded_table, requests, schedule),
    )
    cache[key] = (positions, table, result)
    return result


@triton.jit
def _prepare_paged_metadata_kernel(
    logical_ptr,
    physical_ptr,
    table_ptr,
    safe_table_ptr,
    ROWS: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE0: tl.constexpr,
    TABLE_STRIDE1: tl.constexpr,
    LOGICAL: tl.constexpr,
    PHYSICAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(logical_ptr + row, row < ROWS, other=0)
    last = tl.maximum(length - 1, 0)
    extent = (last // LOGICAL) * PHYSICAL + last % LOGICAL + 1
    tl.store(physical_ptr + row, tl.where(length > 0, extent, 0), row < ROWS)
    block = tl.load(
        table_ptr
        + (row // TABLE_COLS) * TABLE_STRIDE0
        + (row % TABLE_COLS) * TABLE_STRIDE1,
        row < TABLE_ROWS * TABLE_COLS,
        other=0,
    )
    # RTP's block<=0 sentinel must never reach DG as a negative address.
    # Block zero is safe to read; the output kernel restores its zero-K score
    # without relying on the contents of its unallocated cache storage.
    tl.store(safe_table_ptr + row, tl.maximum(block, 0), row < TABLE_ROWS * TABLE_COLS)


@triton.jit
def _compact_logits_kernel(
    physical_ptr,
    logical_ptr,
    lengths_ptr,
    safe_table_ptr,
    CAPACITY: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    PHYSICAL_STRIDE: tl.constexpr,
    LOGICAL: tl.constexpr,
    PHYSICAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(lengths_ptr + row)
    physical_col = (col // LOGICAL) * PHYSICAL + col % LOGICAL
    block = tl.load(
        safe_table_ptr + (row // QUERY_LEN) * TABLE_COLS + col // LOGICAL,
        col < CAPACITY,
        other=0,
    )
    visible = (col < CAPACITY) & (col < length)
    value = tl.load(
        physical_ptr + row * PHYSICAL_STRIDE + physical_col,
        visible & (block > 0),
        other=0.0,
    )
    tl.store(
        logical_ptr + row * CAPACITY + col,
        tl.where(visible, value, -float("inf")),
        col < CAPACITY,
    )


@triton.jit
def _compact_logits_bounded_kernel(
    physical_ptr,
    logical_ptr,
    lengths_ptr,
    safe_table_ptr,
    CAPACITY: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    PHYSICAL_STRIDE: tl.constexpr,
    LOGICAL: tl.constexpr,
    PHYSICAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Length-bounded compaction: one CTA per row, dynamic loop over visible
    columns, no stores beyond context_lens.

    Columns >= length keep the allocator contents; every bounded decode
    consumer (candidate pool, topk_v3, DeepSelect, finite epilogue) clamps its
    reads by the same per-row lengths. A caller falling back to an unbounded
    reader must restore the -inf padding first (see score_decode_indexer).

    Versus the padded kernel this removes both the ~94%-of-capacity dead
    stores and the capacity-wide grid of mostly-empty CTAs (the static-grid
    form is dispatch/wave-latency bound, not store-bandwidth bound). The
    full-tile gather is unpredicated: physical_col stays inside the scored
    physical row for every col < CAPACITY, block-zero reads are safe by the
    sentinel design, and the block<=0 zero restoration happens in registers.
    """
    row = tl.program_id(0)
    length = tl.minimum(tl.load(lengths_ptr + row), CAPACITY)
    table_base = safe_table_ptr + (row // QUERY_LEN) * TABLE_COLS
    store_base = logical_ptr + row * CAPACITY
    load_base = physical_ptr + row * PHYSICAL_STRIDE
    n_full = (length // BLOCK) * BLOCK
    for start in tl.range(0, n_full, BLOCK):
        col = start + tl.arange(0, BLOCK)
        physical_col = (col // LOGICAL) * PHYSICAL + col % LOGICAL
        block = tl.load(table_base + col // LOGICAL, col < CAPACITY, other=0)
        value = tl.load(load_base + physical_col)
        tl.store(store_base + col, tl.where(block > 0, value, 0.0))
    if n_full < length:
        col = n_full + tl.arange(0, BLOCK)
        physical_col = (col // LOGICAL) * PHYSICAL + col % LOGICAL
        block = tl.load(table_base + col // LOGICAL, col < CAPACITY, other=0)
        # Tail columns can pass CAPACITY; keep every access in bounds.
        value = tl.load(load_base + physical_col, col < CAPACITY, other=0.0)
        visible = col < length
        tl.store(
            store_base + col,
            tl.where(visible & (block > 0), value, 0.0),
            visible,
        )


def is_supported(
    device: torch.device,
    block_size: int,
    num_heads: int = 32,
    head_dim: int = 128,
) -> bool:
    """Gate known unsupported paths; execution failures must still propagate."""
    if (
        torch.device(device).type != "cuda"
        or num_heads != 32
        or head_dim != 128
        or block_size not in (64, 128)
    ):
        return False
    from ._indexer_score import has_fp8_fp4_paged_mqa_logits

    if not has_fp8_fp4_paged_mqa_logits():
        return False
    major = torch.cuda.get_device_capability(device)[0]
    return major == 10


def prepare_indexer_q(
    q: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_head_dim: int = 64,
):
    """RoPE the queries, then quantize to the group-32 UE8M0 FP4 MX form.

    ``q`` is pre-RoPE BF16 [B,S,32,128], ``weights`` is FP32 [B,S,32] (used
    raw by DeepGEMM's MX mode — the query scales live in the packed SF), and
    ``frequencies`` are complex64 [B*S, rope_head_dim/2]. Returns
    ``(payload [B,S,32,64] int8, sf [B,S,32] int32)``.
    """
    from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import quantize_rows_fp4
    from rtp_llm.models_py.modules.dsv41.rope import apply_rotary_emb

    b, s, h, d = q.shape
    flat = q.reshape(b * s, h, d).clone()
    apply_rotary_emb(
        flat[..., -rope_head_dim:].unsqueeze(0), freqs_cis.reshape(b * s, -1)
    )
    payload, sf = quantize_rows_fp4(flat)
    return payload.view(b, s, h, d // 2), sf.view(b, s, h)


def prepare_indexer_q_and_weights(
    q: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int = 64,
    *,
    weights_out_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q from raw head weights and a complete frequency lookup table.

    Unsupported query layouts retain the gather/cast/scale/clone/RoPE/quant
    chain; CUDA/JIT errors in the supported fused path still propagate.
    Select the consumer's weight dtype, with the same post-scaling rounding
    in the fallback. Current DeepGEMM FP4 scoring consumes BF16 weights.
    """
    from ._v41_indexer_q_triton import try_fused_indexer_q

    if weights_out_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("indexer weights must use torch.float32 or torch.bfloat16")
    result = try_fused_indexer_q(
        q,
        weights,
        freqs_cis,
        positions,
        rope_head_dim,
        weights_out_dtype=weights_out_dtype,
    )
    if result is not None:
        return result
    scaled_weights = weights.float() * (q.shape[-1] * q.shape[-2]) ** -0.5
    payload, sf = prepare_indexer_q(
        q, scaled_weights, freqs_cis[positions], rope_head_dim
    )
    if weights_out_dtype != torch.float32:
        scaled_weights = scaled_weights.to(weights_out_dtype)
    return payload, sf, scaled_weights


def score_decode_indexer(
    q: torch.Tensor | None,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    pool: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    max_ctx_len: int,
    rope_head_dim: int = 64,
    logical_entries_per_block: int | None = None,
    positions: torch.Tensor | None = None,
    preserve_score_dtype: bool = False,
    pad_invisible: bool = True,
    prepared_query: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    metadata: DecodeIndexerMetadata | None = None,
) -> torch.Tensor | None:
    """Return [B*S,max_ctx_len], or None for an unsupported fast path.

    Default output is FP32. ``preserve_score_dtype`` retains DeepGEMM's native
    BF16/FP32 scores for decode selectors, with no additional rounding.

    ``prepared_query`` supplies payload, packed scales and scaled head weights
    from the caller's query branch. It permits q=None and avoids a BF16 query
    projection allocation when a native GEMM epilogue produced the payload.

    ``pad_invisible=False`` skips the -inf materialization of columns >=
    context_lens, leaving them at the allocator contents (stable across CUDA
    graph replays of one buffer). Only callers whose readers all clamp by the
    same per-row lengths may use it; before handing the result to any
    unbounded reader (e.g. a full-row torch.topk fallback) the caller must
    restore the padding, as attention_v41 does with a masked_fill_.

    Pool and block table must come from the same KV source layer. Pool is the
    contiguous, unpadded uint8 [blocks,typed_entries_per_block,68] view over
    the per-block planar FP4 layout (payload plane then packed-UE8M0 scale
    plane); entries are physical cache rows, not raw allocator blocks. With
    uniform V4.1 cache layout, ratio-2 occupies only the first half of every
    physical block; pass that half as ``logical_entries_per_block``. Its scale
    region still follows ALL physical payload rows, so reinterpreting it as a
    half-sized cache block is incorrect. Score the physical extent and
    compact logits. ``context_lens[b,s]`` is the device-side causal visible
    key count for that individual speculative token, in [0,max_ctx_len].
    Static max_ctx_len must fit the table coverage. No device scalar is read
    during capture/replay.

    DeepGEMM leaves columns >= context_lens undefined. With the default
    pad_invisible=True the output kernel sets them to -inf and restores score
    zero for visible unallocated table entries (block<=0), matching the old
    zero-K dequantization. Candidate selection, short-context ordering and
    invalid-index semantics stay in the caller. Returned logits are already
    causally masked for every reader that clamps by context_lens; no repeated
    caller mask is needed. The compaction kernel never reads DeepGEMM's
    unwritten columns.

    With positions, weights are the unscaled projection and freqs_cis is the
    full frequency table, enabling fused Q preparation. Without positions,
    preserve the original scaled-FP32-weights/gathered-frequencies contract.

    ``metadata`` must be prepared from the same live positions and block table
    for this forward. It owns both logical lengths for selection/compaction and
    physical lengths for the unchanged DG scheduler/scorer.
    """
    if prepared_query is None:
        b, s, h, d = q.shape
        device = q.device
    else:
        payload = prepared_query[0]
        b, s, h, packed_d = payload.shape
        d, device = packed_d * 2, payload.device
    block_size = pool.shape[1]
    logical_entries = (
        block_size if logical_entries_per_block is None else logical_entries_per_block
    )
    if not is_supported(device, block_size, h, d):
        return None
    from ._indexer_score import fp8_fp4_paged_indexer_score

    if prepared_query is not None:
        q_payload, q_sf, weights = prepared_query
    elif positions is None:
        q_payload, q_sf = prepare_indexer_q(q, weights, freqs_cis, rope_head_dim)
    else:
        q_payload, q_sf, weights = prepare_indexer_q_and_weights(
            q,
            weights,
            freqs_cis,
            positions,
            rope_head_dim,
            weights_out_dtype=torch.bfloat16,
        )
    if metadata is None:
        lengths = context_lens.to(torch.int32).contiguous()
        physical_lengths = torch.empty_like(lengths)
        safe_table = torch.empty(block_table.shape, dtype=torch.int32, device=device)
        _prepare_paged_metadata_kernel[
            (triton.cdiv(max(b * s, block_table.numel()), 128),)
        ](
            lengths,
            physical_lengths,
            block_table,
            safe_table,
            b * s,
            b,
            block_table.shape[1],
            block_table.stride(0),
            block_table.stride(1),
            logical_entries,
            block_size,
            128,
        )
    else:
        lengths = metadata.lengths
        physical_lengths = metadata.paged.lengths.view(b, s)
        safe_table = metadata.safe_table
    physical_capacity = (
        max_ctx_len
        if logical_entries == block_size
        else triton.cdiv(max_ctx_len, logical_entries) * block_size
    )
    logits = fp8_fp4_paged_indexer_score(
        q_payload,
        q_sf,
        pool,
        weights.reshape(b * s, h),
        safe_table,
        physical_lengths,
        block_size,
        physical_capacity,
        return_bf16=True,
        metadata=None if metadata is None else metadata.paged,
    )
    compact = torch.empty(
        b * s,
        max_ctx_len,
        dtype=logits.dtype if preserve_score_dtype else torch.float32,
        device=device,
    )
    if pad_invisible:
        _compact_logits_kernel[(b * s, triton.cdiv(max_ctx_len, 1024))](
            logits,
            compact,
            lengths,
            safe_table,
            max_ctx_len,
            s,
            block_table.shape[1],
            logits.stride(0),
            logical_entries,
            block_size,
            1024,
        )
    else:
        # 8192-column tiles benchmarked fastest at production decode shapes
        # (T=192, 256k/512k capacity, 32k contexts): 3.5/5.3 us per call vs
        # 30.8/59.7 us for the padded static grid (graph-replay GPU timing).
        _compact_logits_bounded_kernel[(b * s,)](
            logits,
            compact,
            lengths,
            safe_table,
            max_ctx_len,
            s,
            block_table.shape[1],
            logits.stride(0),
            logical_entries,
            block_size,
            8192,
            num_warps=8,
        )
    return compact
