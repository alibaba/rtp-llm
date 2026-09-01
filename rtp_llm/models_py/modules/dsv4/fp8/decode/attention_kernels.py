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

import os
from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)

# opt_flash_mla diagnostic: when ``DSV4_TOPK_DEBUG=1`` assert each effective
# length fits the indices width on the eager path. Off by default — the check
# forces a device->host sync per call, so it must not run in production. Lengths
# are clamped to the indices width by construction (build/_update_topk_lengths),
# so this is purely a guard for future changes. See opt_flash_mla.
_TOPK_DEBUG = os.environ.get("DSV4_TOPK_DEBUG", "0") not in ("0", "", "false", "False")


def _debug_check_topk_length(tag: str, length, indices) -> None:
    """Assert effective ``topk_length`` values fit the indices last-dim width.

    Eager-only (skipped while a CUDA graph stream is capturing, where the
    device->host sync would be illegal). A value exceeding the indices width
    would make FlashMLA scan past the buffer (IMA surfacing later as a sticky
    CUBLAS error). Gated by ``DSV4_TOPK_DEBUG`` so it is zero-cost by default.
    """
    if not _TOPK_DEBUG or length is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        pass
    width = int(indices.shape[-1])
    mx = int(length.max())
    mn = int(length.min())
    if mx > width or mn < 0:
        raise AssertionError(
            f"[opt_flash_mla][{tag}] topk_length out of range: "
            f"min={mn} max={mx} indices_width={width} "
            f"length_shape={tuple(length.shape)} indices_shape={tuple(indices.shape)}"
        )


def attn_fp8_swa_paged(
    *,
    q: torch.Tensor,  # [B, 1, H, D] bf16
    swa_pool_3d: torch.Tensor,  # [num_blocks, entries_per_block, 584] uint8
    attn_sink: torch.Tensor,  # [H] bf16 / fp32
    swa_topk_3d: torch.Tensor,  # [B, 1, win] int32 global slots into SWA pool
    swa_block_table: torch.Tensor,  # [B, max_blocks_per_req] int32
    sched_meta: Any,  # FlashMLA sched_meta from DSv4DecodeAttnMetadataFP8
    fp8_op: SparseAttnV4DecodeFp8Op,
    topk_length: Optional[torch.Tensor] = None,  # [B] int32 SWA effective length
) -> torch.Tensor:
    """SWA-only FP8 FlashMLA — returns ``[B, 1, H, D_v]`` bf16 output.

    Contract notes:
      * ``swa_topk_3d`` holds per-request **global pool slot ids** (``-1``
        padding for entries before sequence start). FlashMLA's sparse FP8
        kernel reads ``pool[indices[i]]`` with no block-table indirection,
        so the caller MUST translate abs positions via
        :func:`~paged_topk_translator.translate_local_to_global_slots`
        before invoking this op.
      * ``sched_meta`` is owned by :class:`DSv4DecodeAttnMetadataFP8` (see
        :func:`~decode_attn_metadata.get_or_build_sched_meta`); lifetime
        is tied to the decode step (eager) or the capture impl (CUDA graph).
      * ``topk_length`` (opt_flash_mla) is the per-request SWA effective length
        ``[B] int32``; FlashMLA only scans ``swa_topk_3d[:, :, :topk_length[b]]``.
        ``None`` keeps the legacy full-capture-width scan.
    """
    _debug_check_topk_length("swa", topk_length, swa_topk_3d)
    return fp8_op.forward(
        q,
        swa_pool_3d,
        attn_sink,
        swa_topk_3d,
        sched_meta,
        block_table=swa_block_table,
        topk_length=topk_length,
    )


def attn_fp8_dual_paged(
    *,
    q: torch.Tensor,  # [B, 1, H, D] bf16
    swa_pool_3d: torch.Tensor,  # [num_blocks, eb_swa, 584] uint8
    cmp_pool_3d: torch.Tensor,  # [num_blocks, eb_cmp, 584] uint8
    attn_sink: torch.Tensor,
    swa_topk_3d: torch.Tensor,  # [B, 1, win] int32 global slots into SWA pool
    cmp_topk_3d: torch.Tensor,  # [B, 1, K_cmp] int32 global slots into cmp pool
    swa_block_table: torch.Tensor,  # [B, max_blocks_per_req] int32
    sched_meta: Any,  # FlashMLA sched_meta from DSv4DecodeAttnMetadataFP8
    fp8_op: SparseAttnV4DecodeFp8Op,
    topk_length: Optional[torch.Tensor] = None,  # [B] int32 SWA effective length
    extra_topk_length: Optional[torch.Tensor] = None,  # [B] int32 cmp effective length
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

    ``topk_length`` / ``extra_topk_length`` (opt_flash_mla) are per-request
    ``[B] int32`` effective lengths on ``swa_topk_3d`` / ``cmp_topk_3d``;
    FlashMLA only scans the leftmost ``length[b]`` indices of each (``-1``
    entries inside that bound are skipped). This shrinks the HCA scan from the
    CUDA-graph capture width (e.g. 8192) to the true ``(seq_len)//ratio``.
    ``None`` keeps the legacy full-capture-width scan.
    """
    _debug_check_topk_length("dual.swa", topk_length, swa_topk_3d)
    _debug_check_topk_length("dual.extra", extra_topk_length, cmp_topk_3d)
    return fp8_op.forward(
        q,
        swa_pool_3d,
        attn_sink,
        swa_topk_3d,
        sched_meta,
        block_table=swa_block_table,
        topk_length=topk_length,
        extra_k_cache=cmp_pool_3d,
        extra_topk_idxs=cmp_topk_3d,
        extra_topk_length=extra_topk_length,
    )
