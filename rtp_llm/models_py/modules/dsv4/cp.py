"""Context-Parallel helpers for DeepSeek-V4.

Mirrors the SGLang PR #23600 NSA-CP pattern: the framework zigzag-splits
prefill tokens across the CP group (``tp_size``-wide in RTP-LLM's
convention) and populates ``attention_inputs.context_parallel_info``
with ``prefill_qkv_padding_mask`` + ``prefill_qkv_restore_indice``.

V4 needs the full sequence visible at two specific sites: ``Compressor``
(S-dim pooling) and ``Indexer`` (full-context scoring).  Everything
else — attention Q projection, per-layer ``kv_cache`` write, FFN / MoE,
LM head — runs rank-local so MoE's EP dispatch naturally sees only the
1/cp_size token slice that actually belongs to this rank.
"""

from typing import Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather


def cp_all_gather_to_full(
    local: torch.Tensor,
    cp_info,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """All-gather a rank-local ``[1, S_local, *]`` tensor across the CP
    group and restore the logical sequence order via
    ``cp_info.prefill_qkv_restore_indice`` / ``prefill_qkv_padding_mask``.

    Returns the full sequence tensor ``[1, S_total, *]`` with the real
    tokens in their original order (padding rows stripped by the
    padding-mask index).
    """
    assert local.dim() >= 2, f"expected [1, S, ...] got shape {tuple(local.shape)}"
    B = local.size(0)
    assert B == 1, "V4 is B=1 only"

    padding_mask = cp_info.prefill_qkv_padding_mask
    restore_indices = cp_info.prefill_qkv_restore_indice
    dev = local.device
    if padding_mask.device != dev:
        padding_mask = padding_mask.to(dev)
        restore_indices = restore_indices.to(dev)

    total_ag = int(padding_mask.shape[0])
    local_chunk_total = total_ag // cp_size

    # Pad/truncate the rank-local slice up to the padded chunk size.
    S_rank = local.size(1)
    trailing_shape = local.shape[2:]
    if S_rank < local_chunk_total:
        pad = torch.zeros(
            (1, local_chunk_total - S_rank) + trailing_shape,
            dtype=local.dtype, device=dev,
        )
        local_padded = torch.cat([local, pad], dim=1)
    elif S_rank > local_chunk_total:
        local_padded = local[:, :local_chunk_total]
    else:
        local_padded = local

    # Flatten all leading dims except the last feature dim for the
    # all_gather helper (which concatenates along dim 0 for 1-D).
    # local_padded: [1, local_chunk_total, *F] -> [local_chunk_total, prod(F)]
    local_chunk = local_padded.reshape(local_chunk_total, -1)
    gathered = all_gather(local_chunk, group=Group.TP)  # [cp_size * local_chunk_total, prod(F)]
    # Strip zigzag padding and restore logical order.
    unpad_restore = restore_indices[padding_mask == 1].to(torch.long)
    full = gathered[unpad_restore]                      # [S_total_real, prod(F)]
    return full.view((1, full.shape[0]) + trailing_shape)


def cp_compute_global_positions(
    cp_info, cp_size: int, cp_rank: int, S_rank: int, device: torch.device,
) -> torch.Tensor:
    """Return a ``[S_rank]`` int64 tensor mapping each rank-local real
    token (index 0..S_rank-1) to its GLOBAL position in the full
    prefill sequence.

    Used by Attention / Indexer to index ``freqs_cis`` correctly under
    CP: rank-local tokens are scattered zigzag through the full seq,
    so the straightforward ``freqs_cis[:S_rank]`` slice picks wrong
    rotation angles.
    """
    padding_mask = cp_info.prefill_qkv_padding_mask.to(device)
    restore_indices = cp_info.prefill_qkv_restore_indice.to(device)
    total_ag = int(padding_mask.shape[0])
    local_chunk_total = total_ag // cp_size
    unpad_restore = restore_indices[padding_mask == 1].to(torch.long)

    inv_restore = torch.full((total_ag,), -1, dtype=torch.long, device=device)
    inv_restore[unpad_restore] = torch.arange(unpad_restore.shape[0], device=device)
    local_inv = inv_restore[cp_rank * local_chunk_total:(cp_rank + 1) * local_chunk_total]
    # local_inv[i] = global-logical-position of this rank's token at
    # local index i, or -1 if padding.  We keep only real tokens.
    valid = local_inv >= 0
    global_positions = local_inv[valid]                # [S_rank_real]
    if global_positions.numel() >= S_rank:
        return global_positions[:S_rank]
    pad = torch.zeros(S_rank - global_positions.numel(), dtype=torch.long, device=device)
    return torch.cat([global_positions, pad], dim=0)


def cp_should_gather(cp_info, cp_size: int, start_pos: int) -> bool:
    """Prefill-only CP: gather kicks in exactly when CP metadata is
    bound, ``cp_size > 1``, and we are running the prefill pass
    (``start_pos == 0``).  In decode each rank already has the single
    new token it needs (the gathered kv_cache from prefill is still
    valid), so gather is a no-op there."""
    return cp_info is not None and cp_size > 1 and start_pos == 0
