"""CP-aware slot mapping builder for DSV4 prefill writer (Stage 5b-2).

Under ``prefill_cp_config.kv_cache_sharded=true`` the pool is RR-sharded
across CP ranks: logical block ``g_blk`` is owned by rank ``g_blk %
cp_size``. Each rank's per-request ``block_table`` only contains the
1/cp_size physical blocks it owns, in compact form (local block ``l`` ↔
global logical block ``cp_rank + l*cp_size``).

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

Only paged KV pools use this ownership mapping. STATE pools are intentionally
excluded: every CP rank keeps the full STATE block table.
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
) -> torch.Tensor:
    """CP-aware KV-pool slot mapping for compressor write path.

    Mirrors the non-CP ``_compute_kv_slot_mapping`` semantics:
      * Only boundary tokens (``(pos+1) % ratio == 0``) get a real slot.
      * ``block_id == 0`` is treated as "unallocated" → slot = -1.
      * Slots overflowing the pool's row count are masked elsewhere
        (caller responsibility).

    With CP sharding additionally:
      * Non-owned logical blocks → slot = -1.
    """
    if ratio <= 0 or kv_eb <= 0:
        return torch.full_like(positions, -1, dtype=torch.int64)
    block_id, owned_mask = cp_global_block_to_local(
        positions, block_table_local, b_idx, tokens_per_block, cp_size, cp_rank
    )
    in_block_compressed = (positions % tokens_per_block) // ratio
    slot = block_id * kv_eb + in_block_compressed
    boundary = ((positions + 1) % ratio) == 0
    valid = owned_mask & boundary & (block_id > 0)
    return torch.where(valid, slot, torch.full_like(slot, -1))


def cp_state_slot_mapping(
    positions: torch.Tensor,
    block_table_local: torch.Tensor,
    b_idx: torch.Tensor,
    state_eb: int,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Deprecated guardrail.

    STATE pools are not CP-sharded in the DSV4 FP8 path: every rank keeps the
    full STATE block table. Applying CP ownership masking here would make the
    writer skip non-owned positions and leave decode-visible garbage. Keep this
    symbol only to fail fast if old scaffolding tries to use it.
    """
    del positions, block_table_local, b_idx, state_eb, cp_size, cp_rank
    raise RuntimeError(
        "cp_state_slot_mapping is invalid: DSV4 STATE pools are not CP-sharded"
    )
