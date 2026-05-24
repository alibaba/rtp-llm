"""Compute global pool-slot indices from per-request block tables.

Bridges ``DSv4DecodeAttnMetadataFP8``'s per-request positions (start_pos,
compressed indices) and the framework BlockPool's flat slot space.

For a multi-entry KV pool with ``entries_per_block = E``::

    block_id   = block_table[req, abs_pos // E]   # int32
    in_block   = abs_pos %  E
    global_slot = block_id * E + in_block

For state pools with ``E = 1`` slot per page::

    global_slot = block_table[req, slot_in_compressor]   # block_id IS the slot

Negative ``abs_pos`` (sentinel "no write this step") propagates as -1.

Everything stays on device — no D2H, no Python loops, suitable for
CUDA-graph capture.
"""

from __future__ import annotations

from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8._pool_handle import (
    PoolHandle,
    _adhoc_kv_handle,
    _adhoc_state_handle,
)
from rtp_llm.models_py.modules.dsv4.fp8._slot_resolver import (
    SentinelStrict,
    resolve_pool_slot,
)


def compute_kv_pool_slot_mapping(
    block_table: torch.Tensor,
    abs_pos: torch.Tensor,
    entries_per_block: int,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    handle: Optional[PoolHandle] = None,
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
        entries_per_block: pool's entries-per-block (e.g. SWA 256, CSA 64).
        valid_mask: optional ``[B * q_len_per_req]`` bool. When provided,
            slots whose mask entry is False are forced to -1. If None,
            ``abs_pos < 0`` alone is used as the skip signal.

    Returns:
        ``[B * q_len_per_req]`` int64 of global flat slots. ``-1`` marks
        skip (caller passes ``mask_negative=True`` to the write op).

    When ``handle`` is None a degenerate KV handle (ratio=1) is
    synthesised; the resolver math is bit-equal to the historical inline
    body (GT_ZERO + ``eff_pos_already_resolved=True``).
    """
    if abs_pos.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=abs_pos.device)
    if handle is None:
        handle = _adhoc_kv_handle(eb=int(entries_per_block), ratio=1, is_state=False)
    # ``abs_pos < 0`` is the historical skip sentinel; fold it into the
    # caller-supplied ``valid_mask`` so :func:`resolve_pool_slot`'s
    # ``GT_ZERO`` rule + valid_mask reproduce the same drop set.
    abs_pos_i64 = abs_pos.to(torch.long)
    pos_ok = abs_pos_i64 >= 0
    combined_mask = (
        pos_ok if valid_mask is None else (pos_ok & valid_mask.to(torch.bool))
    )
    slot, _ = resolve_pool_slot(
        handle,
        abs_pos_i64,
        block_table,
        sentinel_strict=SentinelStrict.GT_ZERO,
        eff_pos_already_resolved=True,
        valid_mask=combined_mask,
    )
    return slot


def compute_state_pool_slot_mapping(
    block_table: torch.Tensor,
    positions: torch.Tensor,
    req_idx: torch.Tensor,
    entries_per_block: int,
    *,
    cyclic: bool = False,
    handle: Optional[PoolHandle] = None,
) -> torch.Tensor:
    """State-pool addressing (linear canonical; cyclic per-call-site flag).

    Replaces ``decode_attn_metadata._compute_state_pool_slot_mapping``
    (M06 §3.3). The default (``cyclic=False``) matches
    ``_save_partial_states_kernel`` semantics — linear
    ``block_id * state_eb + in_block`` with capacity clamp.

    ``cyclic=True`` preserves the decode-meta rolling-buffer addressing.
    Panel C CD-4 enforcement: when ``cyclic=True`` the caller MUST hand
    in a narrowed ``block_table`` view whose width matches
    ``handle.max_state_blocks``. The shim asserts at the call site
    boundary BEFORE forwarding to :func:`resolve_pool_slot`, surfacing
    contract violations at the consumer rather than silently using the
    wrong modulus deep inside the resolver.
    """
    if positions.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=positions.device)
    if handle is None:
        # On Path B (no descriptor) `cyclic=True` is illegal — we have no
        # canonical modulus. The resolver assert below will fire.
        handle = _adhoc_state_handle(eb=int(entries_per_block))
    if cyclic:
        assert handle.is_state, (
            f"compute_state_pool_slot_mapping: cyclic=True requires a "
            f"STATE-pool handle (got is_state={handle.is_state})"
        )
        assert handle.max_state_blocks is not None, (
            "compute_state_pool_slot_mapping: cyclic=True requires "
            "handle.max_state_blocks (None ⇒ legacy Path-B handle; "
            "caller must take Path A with region_descs available, see "
            "_pool_handle.make_pool_handle)"
        )
        assert block_table.shape[1] == handle.max_state_blocks, (
            f"compute_state_pool_slot_mapping: STATE-pool cyclic-modulus "
            f"contract violation — block_table.shape[1]="
            f"{block_table.shape[1]} != handle.max_state_blocks="
            f"{handle.max_state_blocks}. M09 R3 owns the per-pool "
            f"narrowing of unified_block_table[:, :max_state_blocks[at]] "
            f"before invocation (Panel C CD-4)."
        )
    slot, _ = resolve_pool_slot(
        handle,
        positions,
        block_table,
        req_idx=req_idx,
        sentinel_strict=SentinelStrict.GT_ZERO,
        cyclic=cyclic,
        eff_pos_already_resolved=True,
    )
    return slot
