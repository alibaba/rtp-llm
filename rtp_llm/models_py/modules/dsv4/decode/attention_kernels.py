"""DSv4 decode FlashMLA attention kernel dispatch — extracted from
``Attention._forward_decode_body``.

Mirrors the prefill kernel-family triad (``_attn_via_workspace`` /
``_attn_fp8_swa_via_kv_full`` / ``_attn_fp8_swa_via_concat``) for the
decode path. FP8-only; BF16 fallback removed (see :meth:`Attention.forward_decode`).

Two variants:
    * :func:`attn_fp8_swa_paged` — SWA-only layer (``compress_ratio == 0``).
      One FlashMLA call over the FP8 SWA pool using per-request indices.
    * :func:`attn_fp8_dual_paged` — CSA / HCA layer. One FlashMLA call
      reading both the SWA pool and the compressed pool via
      ``extra_k_cache`` + ``extra_indices_in_kvcache``; softmax is
      merged in-kernel. Matches vLLM ``deepseek_v4_attention.py:849-865``.
"""

from __future__ import annotations

from typing import Any

import torch

from rtp_llm.models_py.modules.dsv4.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


def attn_fp8_swa_paged(
    *,
    q: torch.Tensor,  # [B, 1, H, D] bf16
    swa_pool_3d: torch.Tensor,  # [num_blocks, entries_per_block, 584] uint8
    attn_sink: torch.Tensor,  # [H] bf16 / fp32
    swa_abs_idx: torch.Tensor,  # [B, win] int32, -1 padding per FlashMLA contract
    cache_seqlens: torch.Tensor,  # [B] int32
    swa_block_table: torch.Tensor,  # [B, max_blocks_per_req] int32
    sched_meta: Any,  # FlashMLA sched_meta from DSv4DecodeAttnMetadata
    fp8_op: SparseAttnV4DecodeFp8Op,
    dbg_tag: str = "",
) -> torch.Tensor:
    """SWA-only FP8 FlashMLA — returns ``[B, 1, H, D_v]`` bf16 output.

    Contract notes:
      * ``swa_abs_idx`` already holds per-request abs positions with ``-1``
        padding, which is exactly what FlashMLA's ``indices`` expects —
        no re-cast or re-contiguous needed (iter3.2 optimization).
      * ``cache_seqlens`` is precomputed once per step into
        ``attn_metadata.cache_seqlens_i32`` and sliced ``[:bsz]`` by the
        caller.
      * ``sched_meta`` is owned by :class:`DSv4DecodeAttnMetadata` (see
        :func:`~decode_attn_metadata.get_or_build_sched_meta`); lifetime
        is tied to the decode step (eager) or the capture impl (CUDA graph).
    """
    with torch.profiler.record_function(
        f"{dbg_tag}/fmla_fp8_swa" if dbg_tag else "fmla_fp8_swa"
    ):
        return fp8_op.forward(
            q,
            swa_pool_3d,
            attn_sink,
            swa_abs_idx,
            sched_meta,
            cache_seqlens=cache_seqlens,
            block_table=swa_block_table,
        )


def attn_fp8_dual_paged(
    *,
    q: torch.Tensor,  # [B, 1, H, D] bf16
    swa_pool_3d: torch.Tensor,  # [num_blocks, eb_swa, 584] uint8
    cmp_pool_3d: torch.Tensor,  # [num_blocks, eb_cmp, 584] uint8
    attn_sink: torch.Tensor,
    swa_topk_3d: torch.Tensor,  # [B, 1, win] int32 global slots into SWA pool
    cmp_topk_3d: torch.Tensor,  # [B, 1, K_cmp] int32 global slots into cmp pool
    cache_seqlens: torch.Tensor,  # [B] int32
    swa_block_table: torch.Tensor,  # [B, max_blocks_per_req] int32
    sched_meta: Any,  # FlashMLA sched_meta from DSv4DecodeAttnMetadata
    fp8_op: SparseAttnV4DecodeFp8Op,
    dbg_tag: str = "",
) -> torch.Tensor:
    """Dual-pool (SWA + compressed) FP8 FlashMLA — returns ``[B, 1, H, D_v]``.

    The FlashMLA kernel merges the two pools' attention in-kernel via
    ``extra_k_cache`` / ``extra_indices_in_kvcache`` / merged-softmax.
    Eliminates the legacy "dequant both pools → BF16 cat → TileLang
    sparse_attn" round-trip.

    Callers must feed indices ``[B, 1, K]``-contiguous (iter3.2 — the
    translator already emits int32 so no dtype cast is needed). CSA and
    HCA layers share the same ``sched_meta`` entry (the planner doesn't
    differentiate extra_topk), see
    :func:`~decode_attn_metadata.get_or_build_sched_meta`.
    """
    with torch.profiler.record_function(
        f"{dbg_tag}/fmla_fp8_dual" if dbg_tag else "fmla_fp8_dual"
    ):
        return fp8_op.forward(
            q,
            swa_pool_3d,
            attn_sink,
            swa_topk_3d,
            sched_meta,
            cache_seqlens=cache_seqlens,
            block_table=swa_block_table,
            extra_k_cache=cmp_pool_3d,
            extra_topk_idxs=cmp_topk_3d,
        )
