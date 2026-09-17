"""Compute global pool-slot indices from per-request block tables.

Bridges ``DSv4DecodeAttnMetadataFP8``'s per-request positions (start_pos,
compressed indices) and the framework BlockPool's flat slot space.

For a multi-entry KV pool with ``pool_entries_per_block = E``::

    block_id   = block_table[req, abs_pos // pool_tokens_per_block]   # int32
    in_block   = abs_pos % ring_entries
    global_slot = block_id * E + in_block

For state pools with ``E = 1`` slot per page::

    global_slot = block_table[req, slot_in_compressor]   # block_id IS the slot

Negative ``abs_pos`` (sentinel "no write this step") propagates as -1.

Everything stays on device — no D2H, no Python loops, suitable for
CUDA-graph capture.
"""

from __future__ import annotations

import torch


_SLOT_INDEX_DTYPES = (torch.int32, torch.int64)


def compute_kv_pool_slot_mapping(
    block_table: torch.Tensor,
    abs_pos: torch.Tensor,
    pool_entries_per_block: int,
    pool_tokens_per_block: int,
    ring_entries: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token global pool slot for a multi-entry KV pool.

    Args:
        block_table: ``[B, max_blocks_per_req]`` int32 device tensor.
            ``block_table[r, k]`` is the physical block id holding the
            k-th block worth of pool entries for request r.
        abs_pos: ``[B * q_len_per_req]`` int32 device tensor of absolute
            entry indices within each request's pool stream.
              * SWA: token absolute position (pos = start_pos + s).
              * CSA-K / HCA-K / INDEXER-K: compressed-K index
                ``= (start_pos + s + 1) // ratio - 1``; sentinel -1 for
                non-boundary tokens.
            Shape order is request-major: token i belongs to request
            ``i // q_len_per_req``.
        pool_entries_per_block: pool's flat slot multiplier / tensor second
            dimension ``E``.
        pool_tokens_per_block: raw-token coverage of one block-table row.
        ring_entries: in-block modulo domain. Pass ``pool_entries_per_block``
            for non-ring paged pools.
        valid_mask: optional ``[B * q_len_per_req]`` bool. When provided,
            slots whose mask entry is False are forced to -1. If None,
            ``abs_pos < 0`` alone is used as the skip signal.

    Returns:
        ``[B * q_len_per_req]`` int64 of global flat slots. ``-1`` marks
        skip (caller passes ``mask_negative=True`` to the write op).
    """
    pool_entries_per_block = int(pool_entries_per_block)
    pool_tokens_per_block = int(pool_tokens_per_block)
    ring_entries = int(ring_entries)
    if pool_entries_per_block <= 0:
        raise ValueError(
            "pool_entries_per_block must be positive, "
            f"got {pool_entries_per_block}"
        )
    if pool_tokens_per_block <= 0:
        raise ValueError(
            "pool_tokens_per_block must be positive, "
            f"got {pool_tokens_per_block}"
        )
    if ring_entries <= 0:
        raise ValueError(f"ring_entries must be positive, got {ring_entries}")
    if (
        block_table.dim() != 2
        or block_table.shape[0] <= 0
        or block_table.shape[1] <= 0
    ):
        raise ValueError(
            "block_table must have non-empty shape [B, max_blocks_per_req], "
            f"got {tuple(block_table.shape)}"
        )
    if abs_pos.dim() != 1:
        raise ValueError(f"abs_pos must be 1D, got shape={tuple(abs_pos.shape)}")
    if block_table.dtype not in _SLOT_INDEX_DTYPES:
        raise TypeError(
            "block_table must use int32 or int64 indices, "
            f"got dtype={block_table.dtype}"
        )
    if abs_pos.dtype not in _SLOT_INDEX_DTYPES:
        raise TypeError(
            f"abs_pos must use int32 or int64 indices, got dtype={abs_pos.dtype}"
        )
    if block_table.device != abs_pos.device:
        raise ValueError(
            "block_table and abs_pos must be on the same device, "
            f"got block_table={block_table.device}, abs_pos={abs_pos.device}"
        )
    if valid_mask is not None and valid_mask.shape != abs_pos.shape:
        raise ValueError(
            "valid_mask must match abs_pos shape, "
            f"got valid_mask={tuple(valid_mask.shape)}, abs_pos={tuple(abs_pos.shape)}"
        )
    if valid_mask is not None and valid_mask.dtype != torch.bool:
        raise TypeError(f"valid_mask must be bool, got dtype={valid_mask.dtype}")
    if valid_mask is not None and valid_mask.device != abs_pos.device:
        raise ValueError(
            "valid_mask and abs_pos must be on the same device, "
            f"got valid_mask={valid_mask.device}, abs_pos={abs_pos.device}"
        )
    if ring_entries > pool_entries_per_block:
        raise ValueError(
            "ring_entries cannot exceed pool_entries_per_block, "
            f"got ring_entries={ring_entries}, entries={pool_entries_per_block}"
        )
    if abs_pos.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=abs_pos.device)

    B = block_table.shape[0]
    T_total = abs_pos.shape[0]
    if T_total % B != 0:
        raise ValueError(f"abs_pos ({T_total}) must be divisible by batch ({B})")
    q_len = T_total // B

    # Validate the index dtype before promoting to int64 so the arithmetic does
    # not overflow int32 for large global positions.
    abs_pos_i64 = abs_pos.to(torch.long)
    # Per-token request index (request-major flat layout).
    req_idx = (
        torch.arange(B, device=abs_pos.device, dtype=torch.long)
        .view(B, 1)
        .expand(B, q_len)
        .reshape(-1)
    )

    # Negative abs_pos → skip; clamp before the gather so the index is in
    # range, then mask the output back to -1.
    skip = abs_pos_i64 < 0
    if valid_mask is not None:
        skip = skip | (~valid_mask)
    safe_pos = torch.where(skip, torch.zeros_like(abs_pos_i64), abs_pos_i64)

    block_in_seq = safe_pos // pool_tokens_per_block
    in_block = safe_pos % ring_entries

    # OOB rows are invalid, not aliases for the final block-table column. Point
    # the defensive gather at row 0, then include row validity in the output
    # mask so the result is always -1.
    max_blocks = block_table.shape[1]
    valid_block_row = block_in_seq < max_blocks
    safe_block_in_seq = torch.where(
        valid_block_row, block_in_seq, torch.zeros_like(block_in_seq)
    )

    # Gather block_id per token.
    # block_table's integer/device contract was checked above; promote only to
    # make the returned flat slots unconditionally int64.
    flat_bt = block_table.to(torch.long)
    block_id = flat_bt[req_idx, safe_block_in_seq]
    global_slot = block_id * pool_entries_per_block + in_block

    # C++ BlockPool reserves block 0 and uses -1 for NULL_BLOCK_IDX.
    # Normalize every unallocated block-table entry to the single writer/
    # attention sentinel, -1. Leaving values like ``-256 + in_block`` in
    # slot tensors is unsafe because downstream kernels only understand -1.
    skip = skip | (~valid_block_row) | (block_id <= 0)
    return torch.where(skip, torch.full_like(global_slot, -1), global_slot)
