"""Native FlashMLA dual-pool decode for architecture-selected V4.1 caches."""

from __future__ import annotations

import torch


def fp4_dual_decode_attention(
    *,
    q: torch.Tensor,
    swa_pool_3d: torch.Tensor,
    global_pool_3d: torch.Tensor,
    attn_sink: torch.Tensor,
    swa_topk_3d: torch.Tensor,
    global_topk_3d: torch.Tensor,
    swa_block_table: torch.Tensor,
    sched_meta,
    fp8_op,
) -> torch.Tensor:
    """Return BF16 ``[B, S, H, 512]`` attention with one joint softmax.

    Blackwell uses 528B SWA and 288B FP4 GLOBAL token-interleaved records.
    Hopper uses 584B V4 page-planar records for both pools. In either format
    physical page strides may include padding. Sparse indices address
    ``page * entries_per_page + offset`` and use -1 for masked entries.
    """
    return fp8_op.forward(
        q,
        swa_pool_3d,
        attn_sink,
        swa_topk_3d,
        sched_meta,
        block_table=swa_block_table,
        topk_length=None,
        extra_k_cache=global_pool_3d,
        extra_topk_idxs=global_topk_3d,
        extra_topk_length=None,
    )
