"""DeepSeek-V4 decode-path KV-write ops (BF16, per-request batched).

Two pure-torch ops scatter freshly-computed K rows into the per-layer
``register_buffer`` cache via the slot mappings produced by
:mod:`decode_attn_metadata`. Each request can carry its own
``start_pos`` — the slot mapping has the per-request stride baked in.

Layout (mirrors the prefill code's ``self.kv_cache``):
  * ``swa_buffer``         : ``[B_max, window_size, head_dim]``
  * ``compressed_buffer``  : ``[B_max, max_seq_len // ratio, head_dim]``

Both buffers are flattened to ``[B_max * T_dim, head_dim]`` at write
time and indexed by the (already flat-stride) slots the metadata
builder hands us. Negative slots are skipped (used by the compressed
op for non-boundary tokens).

Implementation choice: pure ``torch.index_copy_`` — no triton kernel
needed. At V4 limits, a forward step writes at most ``B * q_len`` rows
per layer (~hundreds for max-batch decode, vs. tens of millions for
attention itself), so launch overhead, not bandwidth, dominates and
``index_copy_`` is the right tool.

Both ops run on CPU and CUDA (kept device-agnostic for unit testing).
"""

from __future__ import annotations

import torch
from torch import Tensor


def write_swa_k_decode(
    k_state: Tensor,
    slot_mapping: Tensor,
    swa_buffer: Tensor,
) -> None:
    """In-place SWA K write.

    For each token ``i``::

        swa_buffer[b_idx[i], t_idx[i]] = k_state[i]

    The SWA slot mapping never emits -1 (every token lands on a valid ring
    slot), so this op is unconditional — no mask, no D2H sync, no boolean
    indexing. All ops are CUDA-graph-capture safe.

    Args:
        k_state: ``[T_total, head_dim]`` bf16 — flat over batch.
        slot_mapping: ``[T_total]`` int32 — flat slot index in the
            ``[B_max * window_size]`` ring space. Per-request offset baked
            in by the metadata builder; never -1.
        swa_buffer: ``[B_max, window_size, head_dim]`` bf16 — write
            target, modified in place.
    """
    if k_state.numel() == 0 or slot_mapping.numel() == 0:
        return

    win = swa_buffer.shape[1]
    slot_mapping_long = (
        slot_mapping.long() if slot_mapping.dtype != torch.long else slot_mapping
    )
    if k_state.dtype != swa_buffer.dtype:
        k_state = k_state.to(swa_buffer.dtype)
    # 2D indexing handles non-contiguous slice (e.g. kv_cache[:, :win, :]).
    b_idx = slot_mapping_long // win
    t_idx = slot_mapping_long % win
    swa_buffer[b_idx, t_idx] = k_state


def write_compressed_k_decode(
    k_state: Tensor,
    slot_mapping: Tensor,
    compressed_buffer: Tensor,
) -> None:
    """In-place compressed-K write (skips non-boundary tokens).

    For each ``i`` with ``slot_mapping[i] >= 0``::

        compressed_buffer[b_idx[i], t_idx[i]] = k_state[i]

    Tokens with ``slot_mapping[i] == -1`` are no-ops (non-boundary positions).

    CUDA-graph safe implementation: no boolean indexing, no D2H sync, no
    data-dependent output shapes. Uses safe-redirect + delta-encode +
    ``index_put_(accumulate=True)``:
      * ``-1`` slots are redirected to slot 0 (any valid slot).
      * The delta for invalid slots is zero → accumulate has no effect.
      * For valid slots, delta = ``k_state - existing`` → after atomic add
        the slot holds ``existing + (k_state - existing) = k_state``.
      * If two tokens (both -1 redirected) hit slot 0, the deltas are all
        zero → no corruption. If two valid tokens share a slot (impossible
        by construction), the result would be wrong but this never happens.

    Args:
        k_state: ``[T_total, head_dim]`` bf16.
        slot_mapping: ``[T_total]`` int32 — slot in the
            ``[B_max * (max_seq_len // ratio)]`` flat space; ``-1`` = skip.
        compressed_buffer: ``[B_max, max_seq_len // ratio, head_dim]``
            bf16 — write target.
    """
    if k_state.numel() == 0 or slot_mapping.numel() == 0:
        return

    Tc = compressed_buffer.shape[1]
    slot_mapping_long = (
        slot_mapping.long() if slot_mapping.dtype != torch.long else slot_mapping
    )
    valid_mask = slot_mapping_long >= 0  # [T_total] bool — stays on device

    # Redirect -1 to slot 0; invalid slots will contribute a zero delta.
    safe_slot = torch.where(
        valid_mask, slot_mapping_long, torch.zeros_like(slot_mapping_long)
    )
    b_idx = safe_slot // Tc
    t_idx = safe_slot % Tc

    if k_state.dtype != compressed_buffer.dtype:
        k_state = k_state.to(compressed_buffer.dtype)

    # delta-encode: valid slots → k_state - existing; invalid → 0.
    existing = compressed_buffer[b_idx, t_idx]
    delta = torch.where(
        valid_mask.unsqueeze(-1),
        k_state - existing,
        torch.zeros_like(k_state),
    )
    # atomic add — well-defined for repeated indices (slot-0 collisions add 0).
    compressed_buffer.index_put_((b_idx, t_idx), delta, accumulate=True)
