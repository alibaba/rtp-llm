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
    if abs_pos.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=abs_pos.device)
    pool_entries_per_block = int(pool_entries_per_block)
    pool_tokens_per_block = int(pool_tokens_per_block)
    ring_entries = int(ring_entries)
    assert pool_entries_per_block > 0
    assert pool_tokens_per_block > 0
    assert ring_entries > 0

    B = block_table.shape[0]
    T_total = abs_pos.shape[0]
    assert T_total % B == 0, f"abs_pos ({T_total}) must be divisible by batch ({B})"
    q_len = T_total // B

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
        skip = skip | (~valid_mask.to(torch.bool))
    safe_pos = torch.where(skip, torch.zeros_like(abs_pos_i64), abs_pos_i64)

    block_in_seq = safe_pos // pool_tokens_per_block
    in_block = safe_pos % ring_entries

    # Clamp block_in_seq to [0, max_blocks-1] to keep gather in range when
    # abs_pos overflows the request's allocation (shouldn't happen for valid
    # tokens; defensive against sentinel rows beyond [:bsz]).
    max_blocks = block_table.shape[1]
    block_in_seq = block_in_seq.clamp_(0, max_blocks - 1)

    # Gather block_id per token.
    flat_bt = block_table.to(torch.long)
    block_id = flat_bt[req_idx, block_in_seq]
    global_slot = block_id * pool_entries_per_block + in_block

    # C++ BlockPool reserves block 0 and uses -1 for NULL_BLOCK_IDX.
    # Normalize every unallocated block-table entry to the single writer/
    # attention sentinel, -1. Leaving values like ``-256 + in_block`` in
    # slot tensors is unsafe because downstream kernels only understand -1.
    skip = skip | (block_id <= 0)
    return torch.where(skip, torch.full_like(global_slot, -1), global_slot)
