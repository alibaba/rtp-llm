"""CP-aware slot mapping builder for DSV4 prefill writer (Stage 5b-2).

Under ``prefill_cp_config.kv_cache_sharded=true`` the DSV4 paged KV pools
are RR-sharded across CP ranks: logical block ``g_blk`` is owned by rank
``g_blk % cp_size``. Each rank's per-request ``block_table`` only contains
the 1/cp_size physical blocks it owns, in compact form (local block ``l`` ↔
global logical block ``cp_rank + l*cp_size``).

The fixed/SWA pools (INDEXER_STATE / CSA_STATE / HCA_STATE / SWA_KV) are
different: each logical block exists on every rank, but its entries are
split into contiguous intra-block slices. A rank writes only offsets
``[cp_rank * local_eb, (cp_rank + 1) * local_eb)`` within the full logical
block; decode later loads every peer slice into the full block.

The writer (compressor / SWA / indexer write paths) computes a slot
mapping ``[N]`` int64 telling each token where to land in the local
pool. Under sharding:

  * Tokens whose logical block is OWNED by this rank → real slot id
    ``local_block_id * eb + in_block_offset`` (with the existing boundary
    / valid checks layered on top).
  * Tokens whose logical block is NON-OWNED → ``-1`` sentinel; the
    writer kernel skips ``-1`` slots so non-owned data simply isn't
    written.

This is the central primitive consumed by Stage 5b's writer-side
modifications. Pure tensor logic; no NCCL involvement (the gather happens
on the reader side).

"""

from __future__ import annotations

from typing import Tuple

import torch


def cp_global_block_to_local(
    positions: torch.Tensor,
    block_table_local: torch.Tensor,
    b_idx: torch.Tensor,
    tokens_per_block: int,
    cp_size: int,
    cp_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve per-token global logical block → local physical block id.

    Args:
      positions: ``[N]`` int64 — global token position within the request
                 (NOT the in-batch flat offset; per-request 0..T_r-1).
      block_table_local: ``[B, local_max_blocks]`` int64 — this rank's
                 per-request block table. Local block ``l`` at request
                 ``b`` maps to global logical block
                 ``cp_rank + l * cp_size``.
      b_idx: ``[N]`` int64 — per-token request id.
      tokens_per_block: pool's ``seq_size_per_block`` (== virtual block's
                 token granularity for this group).
      cp_size, cp_rank: CP geometry.

    Returns:
      (block_id, owned_mask) where:
        * ``block_id`` ``[N]`` int64 — physical block id from
          ``block_table_local``; for non-owned tokens the value is the
          rank's local block 0 (a safe placeholder; ``owned_mask`` says
          to ignore it).
        * ``owned_mask`` ``[N]`` bool — True iff this token's logical
          block is owned by ``cp_rank`` and present in ``block_table_local``.
    """
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if b_idx.dtype != torch.int64:
        b_idx = b_idx.to(torch.int64)
    bt_long = block_table_local.to(torch.int64)
    max_local_blocks = int(bt_long.shape[1])
    if max_local_blocks <= 0:
        return torch.zeros_like(positions), torch.zeros_like(
            positions, dtype=torch.bool
        )

    g_blk = positions // tokens_per_block
    owner = g_blk % cp_size
    owned_by_rank = owner == cp_rank
    local_blk = g_blk // cp_size
    in_capacity = local_blk < max_local_blocks
    owned_mask = owned_by_rank & in_capacity
    # For non-owned or out-of-capacity tokens we'd index past the local
    # block_table extent; clamp to 0 to keep indexing safe. The value is
    # then masked away by ``owned_mask``.
    local_blk_safe = torch.where(owned_mask, local_blk, torch.zeros_like(local_blk))
    block_id = bt_long[b_idx, local_blk_safe]
    return block_id, owned_mask


def cp_kv_slot_mapping(
    positions: torch.Tensor,
    block_table_local: torch.Tensor,
    b_idx: torch.Tensor,
    tokens_per_block: int,
    kv_eb: int,
    ratio: int,
    cp_size: int,
    cp_rank: int,
    owner_tokens_per_block: int | None = None,
) -> torch.Tensor:
    """CP-aware KV-pool slot mapping for compressor write path.

    Mirrors the non-CP ``_compute_kv_slot_mapping`` semantics:
      * Only boundary tokens (``(pos+1) % ratio == 0``) get a real slot.
      * ``block_id == 0`` is treated as "unallocated" → slot = -1.
      * Slots overflowing the pool's row count are masked elsewhere
        (caller responsibility).

    With CP sharding additionally:
      * Non-owned logical blocks → slot = -1.

    ``tokens_per_block`` is the KV pool's kernel row size.  For DSV4 FP8
    FULL pools the framework block table is expanded to kernel-block ids, but
    cache keys / cache-store FULL blocks are physical blocks.  In that case
    ``owner_tokens_per_block`` is the physical block size: ownership is decided
    by the physical block, while the final slot still indexes the kernel block
    row from ``block_table_local``.
    """
    if ratio <= 0 or kv_eb <= 0:
        return torch.full_like(positions, -1, dtype=torch.int64)
    owner_tpb = int(owner_tokens_per_block or tokens_per_block)
    if owner_tpb <= 0 or owner_tpb % int(tokens_per_block) != 0:
        return torch.full_like(positions, -1, dtype=torch.int64)

    if owner_tpb == int(tokens_per_block):
        block_id, owned_mask = cp_global_block_to_local(
            positions, block_table_local, b_idx, tokens_per_block, cp_size, cp_rank
        )
        in_block_compressed = (positions % tokens_per_block) // ratio
    else:
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)
        if b_idx.dtype != torch.int64:
            b_idx = b_idx.to(torch.int64)
        bt_long = block_table_local.to(torch.int64)
        max_local_kernel_blocks = int(bt_long.shape[1])
        if max_local_kernel_blocks <= 0:
            return torch.full_like(positions, -1, dtype=torch.int64)

        kernel_blocks_per_owner_block = owner_tpb // int(tokens_per_block)
        owner_block = positions // owner_tpb
        owner = owner_block % int(cp_size)
        owned_by_rank = owner == int(cp_rank)
        local_owner_block = owner_block // int(cp_size)
        kernel_in_owner_block = (positions % owner_tpb) // int(tokens_per_block)
        local_kernel_block = (
            local_owner_block * kernel_blocks_per_owner_block + kernel_in_owner_block
        )
        in_capacity = local_kernel_block < max_local_kernel_blocks
        owned_mask = owned_by_rank & in_capacity
        local_kernel_block_safe = torch.where(
            owned_mask, local_kernel_block, torch.zeros_like(local_kernel_block)
        )
        block_id = bt_long[b_idx, local_kernel_block_safe]
        in_block_compressed = (positions % int(tokens_per_block)) // ratio
    slot = block_id * kv_eb + in_block_compressed
    boundary = ((positions + 1) % ratio) == 0
    # CP path uses ``block_id > 0`` (stricter than the non-CP path which uses
    # ``>= 0``). This is intentional: under page-RR the rank-local
    # block_table is padded for short requests, and the CP slot mapper
    # conservatively treats block_id == 0 as the unallocated sentinel to
    # avoid writing through padding rows. Pinned by
    # ``test_unallocated_block_zero_yields_minus_one``.
    valid = owned_mask & boundary & (block_id > 0)
    return torch.where(valid, slot, torch.full_like(slot, -1))


def cp_state_slot_mapping(
    positions: torch.Tensor,
    block_table_local: torch.Tensor,
    b_idx: torch.Tensor,
    state_eb: int,
    tokens_per_block: int,
    cp_size: int,
    cp_rank: int,
    seq_end_per_req: torch.Tensor | None = None,
) -> torch.Tensor:
    """CP-aware fixed STATE slot mapping for intra-block slices.

    ``state_eb`` is this rank's local entries per physical block. The full
    logical ring has ``state_eb * cp_size`` entries. Only positions whose
    full-ring offset belongs to ``cp_rank`` get a real local slot.
    """
    if state_eb <= 0 or tokens_per_block <= 0 or cp_size <= 0:
        return torch.full_like(positions, -1, dtype=torch.int64)
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if b_idx.dtype != torch.int64:
        b_idx = b_idx.to(torch.int64)
    bt_long = block_table_local.to(torch.int64)
    max_blocks = int(bt_long.shape[1])
    if max_blocks <= 0:
        return torch.full_like(positions, -1, dtype=torch.int64)

    full_eb = int(state_eb) * int(cp_size)
    block_in_seq_raw = positions // int(tokens_per_block)
    block_in_seq = block_in_seq_raw % max_blocks
    logical_offset = positions % full_eb
    owner_rank = logical_offset // int(state_eb)
    local_offset = logical_offset - owner_rank * int(state_eb)
    block_id = bt_long[b_idx, block_in_seq]

    valid = (owner_rank == int(cp_rank)) & (block_id > 0)
    if seq_end_per_req is not None:
        seq_end = seq_end_per_req.to(device=positions.device, dtype=torch.int64)[b_idx]
        block_end = (block_in_seq_raw + 1) * int(tokens_per_block)
        effective_end = torch.minimum(block_end, seq_end)
        valid = valid & ((positions + full_eb) >= effective_end)

    slot = block_id * int(state_eb) + local_offset
    return torch.where(valid, slot, torch.full_like(slot, -1))
