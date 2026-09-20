"""Native FlashMLA dual-pool decode for V4.1 SWA FP8 and GLOBAL FP4."""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import FP4_GLOBAL_ENTRY_BYTES
from rtp_llm.models_py.modules.dsv4.fp8._v41_swa_triton import (
    ENTRY_BYTES as SWA_ENTRY_BYTES,
)


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

    FlashMLA 49e2000 accepts a V4.1 528B SWA pool and a 288B FP4 extra
    pool. Both nominal 3D views represent page-planar payload/scale storage;
    their physical page strides may include padding. Sparse indices address
    ``page * entries_per_page + offset`` and use -1 for masked entries.
    """
    if swa_pool_3d.shape[-1] != SWA_ENTRY_BYTES:
        raise ValueError("V4.1 FlashMLA requires a 528B SWA cache")
    if global_pool_3d.shape[-1] != FP4_GLOBAL_ENTRY_BYTES:
        raise ValueError("V4.1 FlashMLA requires a 288B FP4 GLOBAL cache")
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
