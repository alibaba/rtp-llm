# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched V4.1 decode selection and fused candidate metadata.

Candidate pooling/flags follow vLLM's attention/dsa/candidate_blocks.py.
The L20 source can publish DeepInfer-style complete, sorted visible prefixes;
other callers retain the historical score-filtered candidate contract.
Supported candidate-block selections reuse RTP's exact radix-select kernel;
other candidate counts retain torch.topk. Candidate scores remain FP32.
Output order and cutoff-tie choices are unspecified; the newest block is pinned.
Token K512 uses native DeepSelect when supported, with radix-select fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_deepselect as deepselect
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_candidate_topk_v3 as candidate_topk
from rtp_llm.models_py.modules.dsv41.fp8.indexer import _TOPK_V3_OK, _get_topk_workspace
from rtp_llm.ops.compute_ops import rtp_llm_ops


@dataclass
class DecodeSlotPlan:
    """Forward-local slot output; shared TopK always remains in local coordinates."""

    request_ids: torch.Tensor
    block_table: torch.Tensor
    entries: int
    logical_entries: int
    slots: torch.Tensor | None = None


def _slot_fusion_enabled() -> bool:
    """Default-on slot publication; ``DSV41_FUSED_DECODE_TOPK=0`` keeps the
    separate per-layer translator launch for reference checks."""
    return os.environ.get("DSV41_FUSED_DECODE_TOPK", "1") == "1"


def make_slot_plan(request_ids, block_table, *, entries, logical_entries):
    if not _slot_fusion_enabled():
        return None
    if not (
        request_ids.is_cuda
        and request_ids.ndim == 1
        and request_ids.is_contiguous()
        and request_ids.dtype == torch.int32
        and block_table.device == request_ids.device
        and block_table.dtype == torch.int32
        and block_table.ndim == 2
        and block_table.shape[0] > 0
        and block_table.shape[1] > 0
        and block_table.stride(1) == 1
        and block_table.stride(0) >= block_table.shape[1]
        and entries == logical_entries
        and entries > 0
    ):
        return None
    return DecodeSlotPlan(request_ids, block_table, entries, logical_entries)


def is_supported(logits: torch.Tensor, lengths: torch.Tensor, topk: int) -> bool:
    return (
        _TOPK_V3_OK
        and logits.is_cuda
        and logits.dtype in (torch.float32, torch.bfloat16)
        and logits.ndim == 2
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and logits.shape[0] > 0
        and topk in (512, 1024, 2048)
        and logits.shape[1] >= topk
        and lengths.device == logits.device
        and lengths.dtype == torch.int32
        and lengths.is_contiguous()
        and lengths.numel() == logits.shape[0]
    )


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _candidate_pool_kernel(
    logits,
    lengths,
    scores,
    block_lengths,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NBLOCKS: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    columns = blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    visible = tl.load(lengths + row)
    values = tl.load(
        logits + row * STRIDE + columns,
        (blocks[:, None] < NBLOCKS)
        & (offsets[None, :] < BLOCK_SIZE)
        & (columns < WIDTH)
        & (columns < visible),
        other=-float("inf"),
    ).to(tl.float32)
    pooled = tl.reduce(values, 1, _max_with_nan)
    pooled = tl.where(
        (visible > 0) & (blocks == (visible - 1) // BLOCK_SIZE),
        float("inf"),
        pooled,
    )
    tl.store(scores + row * SCORE_STRIDE + blocks, pooled, blocks < NBLOCKS)
    if tl.program_id(1) == 0:
        tl.store(block_lengths + row, tl.cdiv(tl.minimum(visible, WIDTH), BLOCK_SIZE))


@triton.jit
def _candidate_pool_bounded_kernel(
    logits,
    lengths,
    scores,
    block_lengths,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    """Length-bounded candidate pooling: one CTA per row, dynamic loop over the
    visible blocks only, no scores written beyond ``block_lengths[row]``.

    The static ``_candidate_pool_kernel`` launches a capacity-wide grid
    ``[rows, cdiv(nblocks, TILE)]``; at production decode shapes (width 524288,
    block_size 8 => nblocks 65536, visible ~32.8k => ~4096 visible blocks) 93.75%
    of the axis-1 programs load fully-masked ``-inf`` and store padding. Like the
    round-14 compaction kernel, that grid is dispatch/wave-latency bound, not
    store-bandwidth bound: measured 52.9 us static vs 6.6 us bounded (graph-replay
    GPU timing, TILE=256/num_warps=8), an 8.1x removal of the only repo-side
    indexer kernel >20 us left after round 14.

    Blocks ``>= block_lengths[row]`` keep the allocator contents. This is safe
    only for the topk_v3 selection path, whose readers all clamp by the same
    per-row ``block_lengths``: topk_v3 scans ``seq_len = block_lengths[row]``
    (topk_v3.cu:77, vector loads never straddle seq_len) and the store-flags
    epilogue reads scores only at indices topk_v3 selected (< block_lengths).
    The unbounded ``scores.topk`` fallback and the ``nblocks <= count`` dense
    path read the full score row, so ``select_candidates`` retains the padded
    static kernel for those. Visible-block scores and block_lengths are bitwise
    identical to the static kernel (see test_v41_decode_topk).
    """
    row = tl.program_id(0).to(tl.int64)
    visible = tl.minimum(tl.load(lengths + row), WIDTH)
    nblocks_vis = tl.cdiv(visible, BLOCK_SIZE)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    load_base = logits + row * STRIDE
    store_base = scores + row * SCORE_STRIDE
    last_block = (visible - 1) // BLOCK_SIZE
    for start in tl.range(0, nblocks_vis, TILE):
        blocks = start + tl.arange(0, TILE)
        columns = blocks[:, None] * BLOCK_SIZE + offsets[None, :]
        # Full tiles gather unpredicated within the visible span; a single
        # masked condition keeps the partial last block and the tile tail in
        # bounds (columns < visible <= WIDTH, blocks < nblocks_vis).
        values = tl.load(
            load_base + columns,
            (blocks[:, None] < nblocks_vis)
            & (offsets[None, :] < BLOCK_SIZE)
            & (columns < visible),
            other=-float("inf"),
        ).to(tl.float32)
        pooled = tl.reduce(values, 1, _max_with_nan)
        pooled = tl.where((visible > 0) & (blocks == last_block), float("inf"), pooled)
        tl.store(store_base + blocks, pooled, blocks < nblocks_vis)
    tl.store(block_lengths + row, nblocks_vis)


@triton.jit
def _candidate_store_flags_kernel(
    scores,
    candidates,
    flags,
    K: tl.constexpr,
    NBLOCKS: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(NBLOCKS, 1024)):
        blocks = tile * 1024 + offsets
        tl.store(flags + row * NBLOCKS + blocks, 0, blocks < NBLOCKS)
    columns = tl.arange(0, triton.next_power_of_2(K))
    if NBLOCKS <= K:
        # Every block fits: no selection or initialized index buffer is needed.
        index = columns
    else:
        index = tl.load(candidates + row * K + columns, columns < K, other=-1)
    domain = (columns < K) & (index >= 0) & (index < NBLOCKS)
    score = tl.load(scores + row * SCORE_STRIDE + index, domain, other=-float("inf"))
    valid = domain & (score > -float("inf"))
    # +inf pins the newest block; NaN and -inf remain invalid, as in torch.
    tl.store(candidates + row * K + columns, tl.where(valid, index, -1), columns < K)
    tl.debug_barrier()
    tl.store(flags + row * NBLOCKS + index, 1, valid)


@triton.jit
def _candidate_store_prefix_kernel(
    candidates, flags, block_lengths,
    K: tl.constexpr, NBLOCKS: tl.constexpr, SEARCH_STEPS: tl.constexpr,
):
    """Publish sorted visible IDs, not score-filtered holes (decode only)."""
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(NBLOCKS, 1024)):
        blocks = tile * 1024 + offsets
        tl.store(flags + row * NBLOCKS + blocks, 0, blocks < NBLOCKS)
    lanes = tl.arange(0, K)
    visible = tl.maximum(tl.minimum(tl.load(block_lengths + row), NBLOCKS), 0)
    expected = tl.minimum(visible, K)
    if NBLOCKS <= K:
        ids = lanes
    else:
        ids = tl.load(candidates + row * K + lanes)
    values = tl.sort(tl.where((ids >= 0) & (ids < visible), ids, 2147483647))
    count = tl.sum((values != 2147483647).to(tl.int32), 0)
    if count < expected:
        # A full-width torch.topk fallback can choose padding on all--inf ties.
        # Complete it with distinct visible IDs, rather than inventing inactive
        # metadata fillers. Finite, length-bounded TopK skips this branch.
        lo = tl.full((K,), 0, tl.int32)
        hi = tl.full((K,), K, tl.int32)
        for _ in range(SEARCH_STEPS):
            mid = (lo + hi) // 2
            value = tl.gather(values, tl.minimum(mid, K - 1), 0)
            right = (mid < K) & (value < lanes)
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(right, hi, mid)
        present = tl.gather(values, tl.minimum(lo, K - 1), 0) == lanes
        fillers = tl.sort(tl.where((lanes < expected) & ~present, lanes, 2147483647))
        filler = tl.gather(fillers, tl.maximum(lanes - count, 0), 0)
        values = tl.sort(tl.where(lanes < expected, tl.where(lanes < count, values, filler), 2147483647))
    # Preserve the partial context tail even if NaN scores displaced its +inf
    # pin. It is the largest logical ID, so replacing the last slot stays sorted.
    has_last = tl.sum(((lanes < expected) & (values == visible - 1)).to(tl.int32), 0) != 0
    values = tl.where((expected > 0) & ~has_last & (lanes == expected - 1), visible - 1, values)
    valid = lanes < expected
    tl.store(candidates + row * K + lanes, tl.where(valid, values, -1))
    tl.debug_barrier()
    tl.store(flags + row * NBLOCKS + values, 1, valid)


@triton.jit
def _candidate_mask_kernel(
    logits,
    flags,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * TILE + tl.arange(0, TILE)
    keep = tl.load(
        flags + row * NBLOCKS + columns // BLOCK_SIZE,
        columns < WIDTH,
        other=0,
    )
    tl.store(
        logits + row * STRIDE + columns, -float("inf"), (columns < WIDTH) & (keep == 0)
    )


@triton.jit
def _finite_topk_kernel(
    logits,
    lengths,
    output,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * TILE + tl.arange(0, TILE)
    length = tl.load(lengths + row)
    index = tl.load(output + row * K + columns, columns < K, other=-1)
    value = tl.load(
        logits + row * STRIDE + index,
        (columns < K) & (index >= 0) & (index < WIDTH),
        other=-float("inf"),
    )
    # topk_v3 already emits the required ascending short-row result. Long
    # rows can contain candidate-mask holes; lengths alone do not filter them.
    keep = (length <= K) | (tl.abs(value) < float("inf"))
    tl.store(output + row * K + columns, tl.where(keep, index, -1), columns < K)


@triton.jit
def _finite_topk_slots_kernel(
    logits,
    lengths,
    output,
    requests,
    table,
    slots,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    ENTRIES: tl.constexpr,
    LOGICAL: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * TILE + tl.arange(0, TILE)
    length = tl.load(lengths + row)
    index = tl.load(output + row * K + columns, columns < K, other=-1)
    value = tl.load(
        logits + row * STRIDE + index,
        (columns < K) & (index >= 0) & (index < WIDTH),
        other=-float("inf"),
    )
    keep = (length <= K) | (tl.abs(value) < float("inf"))
    local = tl.where(keep, index, -1)
    tl.store(output + row * K + columns, local, columns < K)
    request = tl.load(requests + row).to(tl.int64)
    block = local // LOGICAL
    valid = (
        (columns < K)
        & (local >= 0)
        & (block >= 0)
        & (block < TABLE_COLS)
        & (request >= 0)
        & (request < TABLE_ROWS)
    )
    base = tl.load(table + request * TABLE_STRIDE + block, valid, other=0)
    physical = base * ENTRIES + local % ENTRIES
    tl.store(
        slots + row * K + columns,
        tl.where(valid & (base > 0), physical, -1),
        columns < K,
    )


def select_candidates(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    block_size: int,
    topk_blocks: int,
    *, complete: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``complete`` publishes sorted valid-prefix IDs for the decode source.

    It keeps selected NaN/-inf-score blocks (unlike legacy publication), pins
    the last visible block and pads only after min(visible_blocks, K). Callers
    must publish the matching prefix contract together with these tensors.
    """
    rows, width = logits.shape
    nblocks = triton.cdiv(width, block_size)
    count = min(topk_blocks, nblocks)
    if complete and count not in (512, 1024, 2048):
        raise ValueError("Complete decode candidates require K512/K1024/K2048")
    # Keep score rows aligned for topk_v3's vectorized loads, without widening
    # its logical selection domain or reading uninitialized padding.
    score_stride = triton.cdiv(nblocks, 256) * 256
    scores = torch.empty(
        (rows, score_stride), dtype=torch.float32, device=logits.device
    )[:, :nblocks]
    block_lengths = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    # The topk_v3 selection path clamps every score read by the per-row
    # block_lengths, so pooling can skip the capacity-wide padding and the
    # mostly-empty static grid (see _candidate_pool_bounded_kernel). The
    # scores.topk fallback and the nblocks<=count dense epilogue read the full
    # score row and keep the padded static kernel.
    use_candidate_v3 = topk_blocks == 2048 and candidate_topk.is_available(
        "decode", logits.device
    )
    bounded_pool = (
        nblocks > count
        and count in (512, 1024, 2048)
        and (topk_blocks != 2048 or use_candidate_v3)
    )
    if bounded_pool:
        _candidate_pool_bounded_kernel[(rows,)](
            logits,
            lengths,
            scores,
            block_lengths,
            width,
            logits.stride(0),
            block_size,
            score_stride,
            256,
            num_warps=8,
        )
    else:
        _candidate_pool_kernel[(rows, triton.cdiv(nblocks, 128))](
            logits,
            lengths,
            scores,
            block_lengths,
            width,
            logits.stride(0),
            block_size,
            nblocks,
            score_stride,
            128,
        )
    if use_candidate_v3:
        # Full K2048 output already has the packed layout used by publication.
        candidates = candidate_topk.select(scores, block_lengths)
        if count != 2048:
            candidates = candidates[:, :count].contiguous()
    else:
        candidates = torch.empty((rows, count), dtype=torch.int32, device=logits.device)
    if topk_blocks == 2048 and not use_candidate_v3:
        candidates.copy_(scores.topk(count, dim=-1, sorted=False).indices)
    elif topk_blocks != 2048 and nblocks > count:
        if count in (512, 1024, 2048):
            rtp_llm_ops.topk_v3(
                scores,
                block_lengths,
                candidates,
                _get_topk_workspace(logits.device),
                count,
                nblocks,
            )
        else:
            candidates.copy_(scores.topk(count, dim=-1, sorted=False).indices)
    flags = torch.empty((rows, nblocks), dtype=torch.uint8, device=logits.device)
    if complete:
        _candidate_store_prefix_kernel[(rows,)](
            candidates, flags, block_lengths, count, nblocks, count.bit_length(), num_warps=8,
        )
    else:
        _candidate_store_flags_kernel[(rows,)](
            scores, candidates, flags, count, nblocks, score_stride
        )
    return candidates, flags


def mask_candidates(logits: torch.Tensor, flags: torch.Tensor, block_size: int) -> None:
    rows, width = logits.shape
    _candidate_mask_kernel[(rows, triton.cdiv(width, 1024))](
        logits, flags, width, logits.stride(0), flags.shape[1], block_size, 1024
    )


def select_tokens(
    logits: torch.Tensor, lengths: torch.Tensor, topk: int, *, slot_plan=None
) -> torch.Tensor:
    rows, width = logits.shape
    if slot_plan is not None:
        if (
            slot_plan.request_ids.numel() != rows
            or slot_plan.request_ids.device != logits.device
        ):
            raise ValueError("Slot mapping does not match selected query rows")
        slot_plan.slots = None
    output = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    selected = None
    if topk == 512:
        selected = deepselect.try_select_tokens(
            logits,
            lengths,
            out=output,
            filter_finite=False,
            enabled=os.environ.get("DSV41_DECODE_DEEPSELECT", "1") == "1",
        )
    if selected is None:
        rtp_llm_ops.topk_v3(
            logits, lengths, output, _get_topk_workspace(logits.device), topk, width
        )
    # Decode retains all visible indices on short rows, even candidate-mask holes.
    # Native selection must not filter them before this shared publication step.
    if slot_plan is None:
        _finite_topk_kernel[(rows, triton.cdiv(topk, 256))](
            logits, lengths, output, width, logits.stride(0), topk, 256
        )
    else:
        slots = torch.empty_like(output)
        table = slot_plan.block_table
        _finite_topk_slots_kernel[(rows, triton.cdiv(topk, 256))](
            logits,
            lengths,
            output,
            slot_plan.request_ids,
            table,
            slots,
            width,
            logits.stride(0),
            topk,
            table.shape[0],
            table.shape[1],
            table.stride(0),
            slot_plan.entries,
            slot_plan.logical_entries,
            256,
        )
        slot_plan.slots = slots
    return output
