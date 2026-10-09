"""Build DSpARK query write slots and noncausal attention metadata together."""

from __future__ import annotations

import os
from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl


class DSparkMetadata(NamedTuple):
    query_slots: torch.Tensor  # int64 [B * gamma]
    global_indices: torch.Tensor  # int32 [B * gamma, ceil128(window + gamma)]
    topk_length: torch.Tensor  # int32 [B], shared by every query of a request


@triton.jit
def _pool_slot(
    block_table,
    request,
    position,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    ENTRIES: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    safe_position = tl.maximum(position, 0)
    column = safe_position // TOKENS_PER_BLOCK
    valid = (position >= 0) & (column < TABLE_COLS)
    block = tl.load(block_table + request * TABLE_STRIDE + column, valid, other=0).to(
        tl.int64
    )
    return tl.where(valid & (block > 0), block * ENTRIES + safe_position % ENTRIES, -1)


@triton.jit
def _dspark_metadata_kernel(
    query_positions,
    prefix_lengths,
    active_requests,
    block_table,
    query_slots,
    global_indices,
    topk_length,
    GAMMA: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    ENTRIES: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    request = row // GAMMA
    prefix = tl.load(prefix_lengths + request).to(tl.int64)
    active = tl.load(active_requests + request) != 0
    committed = tl.minimum(prefix, WINDOW)
    columns = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    position = tl.where(
        columns < committed,
        prefix - committed + columns,
        tl.where(columns < committed + GAMMA, prefix + columns - committed, -1),
    )
    position = tl.where(active & (columns < TOPK), position, -1)
    slots = _pool_slot(
        block_table,
        request,
        position,
        TABLE_COLS,
        TABLE_STRIDE,
        ENTRIES,
        TOKENS_PER_BLOCK,
    )
    tl.store(global_indices + row * TOPK + columns, slots, columns < TOPK)

    # Only the first tile owns the scalar outputs; no CTA writes another row.
    if tile == 0:
        query_position = tl.load(query_positions + row).to(tl.int64)
        query_slot = _pool_slot(
            block_table,
            request,
            query_position,
            TABLE_COLS,
            TABLE_STRIDE,
            ENTRIES,
            TOKENS_PER_BLOCK,
        )
        tl.store(query_slots + row, query_slot)
        if row % GAMMA == 0:
            tl.store(topk_length + request, tl.where(active, committed + GAMMA, 0))


def is_supported(
    query_positions: torch.Tensor,
    prefix_lengths: torch.Tensor,
    active_requests: torch.Tensor,
    block_table: torch.Tensor,
    *,
    gamma: int,
    window_size: int,
    entries_per_block: int,
    tokens_per_block: int,
) -> bool:
    """Use shape/device metadata only; unsupported inputs keep the eager path."""
    if os.environ.get("DSV4_FUSED_DSPARK_METADATA", "1") == "0":
        return False
    if (
        gamma <= 0
        or window_size < 0
        or entries_per_block <= 0
        or tokens_per_block <= 0
        or prefix_lengths.ndim != 1
        or active_requests.shape != prefix_lengths.shape
        or query_positions.shape
        not in (
            (prefix_lengths.numel() * gamma,),
            (prefix_lengths.numel(), gamma),
        )
        or block_table.ndim != 2
        or block_table.shape[0] < prefix_lengths.numel()
        or block_table.shape[1] == 0
    ):
        return False
    tensors = (query_positions, prefix_lengths, active_requests, block_table)
    integers = (torch.int32, torch.int64)
    return (
        query_positions.device.type == "cuda"
        and all(
            t.device == query_positions.device and t.is_contiguous() for t in tensors
        )
        and all(
            t.dtype in integers for t in (query_positions, prefix_lengths, block_table)
        )
        and active_requests.dtype in (torch.bool, *integers)
    )


def try_build_dspark_metadata(
    query_positions: torch.Tensor,
    prefix_lengths: torch.Tensor,
    active_requests: torch.Tensor,
    block_table: torch.Tensor,
    *,
    gamma: int,
    window_size: int,
    entries_per_block: int,
    tokens_per_block: int,
) -> Optional[DSparkMetadata]:
    """Return exact eager metadata in one launch, or None for unsupported input.

    Query writes follow query_positions even for inactive requests, as in the
    old writer path. Only attention indices/lengths use active_requests. Invalid
    positions, out-of-range block columns and unallocated blocks map to -1.
    The caller owns the forward-local outputs; no metadata is cached here.
    """
    if not is_supported(
        query_positions,
        prefix_lengths,
        active_requests,
        block_table,
        gamma=gamma,
        window_size=window_size,
        entries_per_block=entries_per_block,
        tokens_per_block=tokens_per_block,
    ):
        return None
    batch = prefix_lengths.numel()
    rows = batch * gamma
    topk = triton.cdiv(window_size + gamma, 128) * 128
    device = query_positions.device
    result = DSparkMetadata(
        torch.empty(rows, dtype=torch.int64, device=device),
        torch.empty((rows, topk), dtype=torch.int32, device=device),
        torch.empty(batch, dtype=torch.int32, device=device),
    )
    if rows:
        _dspark_metadata_kernel[(rows, triton.cdiv(topk, 256))](
            query_positions,
            prefix_lengths,
            active_requests,
            block_table,
            *result,
            GAMMA=gamma,
            WINDOW=window_size,
            TOPK=topk,
            TABLE_COLS=block_table.shape[1],
            TABLE_STRIDE=block_table.stride(0),
            ENTRIES=entries_per_block,
            TOKENS_PER_BLOCK=tokens_per_block,
            BLOCK=256,
            num_warps=4,
        )
    return result
