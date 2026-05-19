"""Fused metadata builders for the DSV4 vLLM-style compressor."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _map_compressor_metadata_kernel(
    positions_ptr,
    b_idx_ptr,
    state_bt_ptr,
    state_bt_stride,
    kv_bt_ptr,
    kv_bt_stride,
    state_slots_ptr,
    kv_slots_ptr,
    token_to_req_ptr,
    n_elements,
    STATE_EB: tl.constexpr,
    STATE_MAX_BLOCKS: tl.constexpr,
    KV_EB: tl.constexpr,
    KV_MAX_BLOCKS: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    pos = tl.load(positions_ptr + offs, mask=mask, other=0).to(tl.int64)
    req = tl.load(b_idx_ptr + offs, mask=mask, other=0).to(tl.int64)
    tl.store(token_to_req_ptr + offs, req.to(tl.int32), mask=mask)

    state_block_in_seq_raw = pos // STATE_EB
    state_in_capacity = state_block_in_seq_raw < STATE_MAX_BLOCKS
    state_block_in_seq = tl.maximum(tl.minimum(state_block_in_seq_raw, STATE_MAX_BLOCKS - 1), 0)
    state_in_block = pos % STATE_EB
    state_block_id = tl.load(
        state_bt_ptr + req * state_bt_stride + state_block_in_seq,
        mask=mask,
        other=0,
    ).to(tl.int64)
    state_slot = state_block_id * STATE_EB + state_in_block
    state_slot = tl.where(state_in_capacity & (state_block_id > 0), state_slot, -1)
    tl.store(state_slots_ptr + offs, state_slot, mask=mask)

    tokens_per_block: tl.constexpr = KV_EB * RATIO
    boundary = ((pos + 1) % RATIO) == 0
    kv_block_in_seq = pos // tokens_per_block
    kv_in_block = (pos % tokens_per_block) // RATIO
    kv_in_capacity = kv_block_in_seq < KV_MAX_BLOCKS
    kv_block_safe = tl.minimum(kv_block_in_seq, KV_MAX_BLOCKS - 1)
    kv_block_id = tl.load(
        kv_bt_ptr + req * kv_bt_stride + kv_block_safe,
        mask=mask,
        other=0,
    ).to(tl.int64)
    kv_valid = boundary & kv_in_capacity & (kv_block_id > 0)
    kv_slot = kv_block_id * KV_EB + kv_in_block
    kv_slot = tl.where(kv_valid, kv_slot, -1)
    tl.store(kv_slots_ptr + offs, kv_slot, mask=mask)


@triton.jit
def _prefill_compressor_metadata_kernel(
    state_bt_ptr,
    state_bt_stride,
    kv_bt_ptr,
    kv_bt_stride,
    positions_ptr,
    b_idx_ptr,
    state_slots_ptr,
    kv_slots_ptr,
    token_to_req_ptr,
    n_elements,
    seq_len,
    start_pos,
    STATE_EB: tl.constexpr,
    STATE_MAX_BLOCKS: tl.constexpr,
    KV_EB: tl.constexpr,
    KV_MAX_BLOCKS: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    req = (offs // seq_len).to(tl.int64)
    local_pos = (offs % seq_len).to(tl.int64)
    pos = start_pos + local_pos
    tl.store(positions_ptr + offs, pos, mask=mask)
    tl.store(b_idx_ptr + offs, req, mask=mask)
    tl.store(token_to_req_ptr + offs, req.to(tl.int32), mask=mask)

    state_block_in_seq_raw = pos // STATE_EB
    state_in_capacity = state_block_in_seq_raw < STATE_MAX_BLOCKS
    state_block_in_seq = tl.maximum(tl.minimum(state_block_in_seq_raw, STATE_MAX_BLOCKS - 1), 0)
    state_in_block = pos % STATE_EB
    state_block_id = tl.load(
        state_bt_ptr + req * state_bt_stride + state_block_in_seq,
        mask=mask,
        other=0,
    ).to(tl.int64)
    state_slot = state_block_id * STATE_EB + state_in_block
    state_slot = tl.where(state_in_capacity & (state_block_id > 0), state_slot, -1)
    tl.store(state_slots_ptr + offs, state_slot, mask=mask)

    tokens_per_block: tl.constexpr = KV_EB * RATIO
    boundary = ((pos + 1) % RATIO) == 0
    kv_block_in_seq = pos // tokens_per_block
    kv_in_block = (pos % tokens_per_block) // RATIO
    kv_in_capacity = kv_block_in_seq < KV_MAX_BLOCKS
    kv_block_safe = tl.minimum(kv_block_in_seq, KV_MAX_BLOCKS - 1)
    kv_block_id = tl.load(
        kv_bt_ptr + req * kv_bt_stride + kv_block_safe,
        mask=mask,
        other=0,
    ).to(tl.int64)
    kv_valid = boundary & kv_in_capacity & (kv_block_id > 0)
    kv_slot = kv_block_id * KV_EB + kv_in_block
    kv_slot = tl.where(kv_valid, kv_slot, -1)
    tl.store(kv_slots_ptr + offs, kv_slot, mask=mask)


def _can_fuse(
    state_block_table: Optional[torch.Tensor],
    state_eb: int,
    kv_block_table: Optional[torch.Tensor],
    kv_eb: int,
) -> bool:
    return (
        state_block_table is not None
        and kv_block_table is not None
        and state_block_table.is_cuda
        and kv_block_table.is_cuda
        and state_eb > 0
        and kv_eb > 0
        and int(state_block_table.shape[1]) > 0
        and int(kv_block_table.shape[1]) > 0
    )


def map_compressor_metadata(
    positions: torch.Tensor,
    b_idx: torch.Tensor,
    state_block_table: Optional[torch.Tensor],
    state_eb: int,
    kv_block_table: Optional[torch.Tensor],
    kv_eb: int,
    compress_ratio: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return fused ``(state_slots, kv_slots, token_to_req)`` or ``None``."""
    if (
        not positions.is_cuda
        or not b_idx.is_cuda
        or positions.dtype != torch.long
        or b_idx.dtype != torch.long
        or not _can_fuse(state_block_table, state_eb, kv_block_table, kv_eb)
    ):
        return None

    assert state_block_table is not None
    assert kv_block_table is not None
    n_elements = int(positions.numel())
    state_slots = torch.empty_like(positions)
    kv_slots = torch.empty_like(positions)
    token_to_req = torch.empty((n_elements,), device=positions.device, dtype=torch.int32)
    if n_elements == 0:
        return state_slots, kv_slots, token_to_req

    block = 256
    grid = (triton.cdiv(n_elements, block),)
    _map_compressor_metadata_kernel[grid](
        positions,
        b_idx,
        state_block_table,
        state_block_table.stride(0),
        kv_block_table,
        kv_block_table.stride(0),
        state_slots,
        kv_slots,
        token_to_req,
        n_elements,
        STATE_EB=int(state_eb),
        STATE_MAX_BLOCKS=int(state_block_table.shape[1]),
        KV_EB=int(kv_eb),
        KV_MAX_BLOCKS=int(kv_block_table.shape[1]),
        RATIO=int(compress_ratio),
        BLOCK=block,
    )
    return state_slots, kv_slots, token_to_req


def build_prefill_compressor_metadata(
    start_pos: int,
    bsz: int,
    seqlen: int,
    device: torch.device,
    state_block_table: Optional[torch.Tensor],
    state_eb: int,
    kv_block_table: Optional[torch.Tensor],
    kv_eb: int,
    compress_ratio: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return fused ``(positions, b_idx, state_slots, kv_slots, token_to_req)``.

    The prefill hot path normally has ``bsz == 1``; this helper still supports
    larger batches by generating the same flattened token-major layout as
    ``_build_prefill_positions``.
    """
    if not _can_fuse(state_block_table, state_eb, kv_block_table, kv_eb):
        return None

    assert state_block_table is not None
    assert kv_block_table is not None
    n_elements = int(bsz) * int(seqlen)
    positions = torch.empty((n_elements,), device=device, dtype=torch.long)
    b_idx = torch.empty((n_elements,), device=device, dtype=torch.long)
    state_slots = torch.empty_like(positions)
    kv_slots = torch.empty_like(positions)
    token_to_req = torch.empty((n_elements,), device=device, dtype=torch.int32)
    if n_elements == 0:
        return positions, b_idx, state_slots, kv_slots, token_to_req

    block = 256
    grid = (triton.cdiv(n_elements, block),)
    _prefill_compressor_metadata_kernel[grid](
        state_block_table,
        state_block_table.stride(0),
        kv_block_table,
        kv_block_table.stride(0),
        positions,
        b_idx,
        state_slots,
        kv_slots,
        token_to_req,
        n_elements,
        int(seqlen),
        int(start_pos),
        STATE_EB=int(state_eb),
        STATE_MAX_BLOCKS=int(state_block_table.shape[1]),
        KV_EB=int(kv_eb),
        KV_MAX_BLOCKS=int(kv_block_table.shape[1]),
        RATIO=int(compress_ratio),
        BLOCK=block,
    )
    return positions, b_idx, state_slots, kv_slots, token_to_req
