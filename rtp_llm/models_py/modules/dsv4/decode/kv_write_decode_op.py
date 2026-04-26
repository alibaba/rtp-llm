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

        swa_buffer.view(-1, head_dim)[slot_mapping[i]] = k_state[i]

    Args:
        k_state: ``[T_total, head_dim]`` bf16 — flat over batch
            (``T_total = B * q_len``).
        slot_mapping: ``[T_total]`` int32 — slot in flattened SWA buffer
            ``[B_max * window_size, head_dim]``. Per-request stride is
            already baked in by the metadata builder. Negative entries
            are defensively skipped (no-op).
        swa_buffer: ``[B_max, window_size, head_dim]`` bf16 — write
            target, modified in place.
    """
    if k_state.numel() == 0 or slot_mapping.numel() == 0:
        return

    win = swa_buffer.shape[1]

    # Defensive: drop slot==-1 (currently the SWA mapping never emits -1,
    # but keeping the mask makes this op robust if a caller stitches the
    # SWA + compressed mappings together).
    if slot_mapping.dtype != torch.long:
        slot_mapping_long = slot_mapping.long()
    else:
        slot_mapping_long = slot_mapping
    valid_mask = slot_mapping_long >= 0
    if not bool(valid_mask.all()):
        slot_mapping_long = slot_mapping_long[valid_mask]
        k_state = k_state[valid_mask]
        if slot_mapping_long.numel() == 0:
            return

    # The SWA buffer is typically a non-contiguous slice
    # (``self.kv_cache[:, :win]`` of a larger ``[:, win + Tc]`` cache),
    # so the flat ``view(-1, head_dim)`` approach is unsafe — use 2D
    # ``(b, t)`` indexing on the original tensor instead.  This fans out
    # to a single launched ``__setitem__`` which is also in-place.
    if k_state.dtype != swa_buffer.dtype:
        k_state = k_state.to(swa_buffer.dtype)
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

        compressed_buffer.view(-1, head_dim)[slot_mapping[i]] = k_state[i]

    Tokens whose ``slot_mapping`` entry is ``-1`` are no-ops — these are
    the non-compression-boundary positions emitted by
    :func:`_build_compressed_slot_mapping`.

    Args:
        k_state: ``[T_total, head_dim]`` bf16 — every "alive" token's
            compressed K. The per-token mask is carried via
            ``slot_mapping``, not via slicing the input, so the caller
            doesn't have to re-pack ``k_state``.
        slot_mapping: ``[T_total]`` int32 — slot in flattened compressed
            buffer ``[B_max * (max_seq_len // ratio), head_dim]``;
            ``-1`` means "skip".
        compressed_buffer: ``[B_max, max_seq_len // ratio, head_dim]``
            bf16 — write target, modified in place.
    """
    if k_state.numel() == 0 or slot_mapping.numel() == 0:
        return

    Tc = compressed_buffer.shape[1]

    if slot_mapping.dtype != torch.long:
        slot_mapping_long = slot_mapping.long()
    else:
        slot_mapping_long = slot_mapping
    valid_mask = slot_mapping_long >= 0
    if not bool(valid_mask.any()):
        return
    slot_mapping_long = slot_mapping_long[valid_mask]
    k_state = k_state[valid_mask]

    # Same non-contiguous-slice concern as ``write_swa_k_decode`` —
    # ``compressed_buffer`` is ``self.kv_cache[:, win:]`` and therefore
    # non-contiguous.  Use 2D ``(b, t)`` indexing.
    if k_state.dtype != compressed_buffer.dtype:
        k_state = k_state.to(compressed_buffer.dtype)
    b_idx = slot_mapping_long // Tc
    t_idx = slot_mapping_long % Tc
    compressed_buffer[b_idx, t_idx] = k_state
