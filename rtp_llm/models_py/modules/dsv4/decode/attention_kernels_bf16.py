"""DSv4 decode BF16 sparse attention dispatch — extracted from
``AttentionVLLM._forward_decode_body``.

Mirrors the source project's :mod:`rtp_llm.models_py.modules.dsv4.decode.attention_kernels`
(FP8 FlashMLA path) for the BF16 KV-cache. Two variants:

    * :func:`attn_bf16_swa_paged` — SWA-only layer (``compress_ratio == 0``).
      Zero-copy: pool view fed directly to the TileLang sparse_attn kernel
      via per-request global slot indices.
    * :func:`attn_bf16_dual_paged` — CSA / HCA layer. The TileLang kernel
      can't take two KV tensors, so we gather both pools into a packed
      scratch (``gather_dual_pool_kv_packed``) and call sparse_attn with
      identity ``[swa_topk | cmp_topk]`` indices. Memory cost noted in
      :func:`paged_topk_translator.gather_dual_pool_kv_packed`.

Topk index layout: SWA half occupies ``[0, win)``, compressed half occupies
``[win, win+K_cmp)`` in the packed kv. ``-1`` entries in the source global-slot
arrays propagate as ``-1`` (masked) into the packed topk.
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.decode.paged_topk_translator import (
    gather_dual_pool_kv_packed,
)
from rtp_llm.models_py.modules.dsv4.decode.sparse_attn_decode_op import (
    SparseAttnV4DecodeOp,
)


def attn_bf16_swa_paged(
    *,
    q: torch.Tensor,  # [B, q_len, H, D] bf16
    swa_pool_view: torch.Tensor,  # [num_global_slots, D] bf16
    attn_sink: torch.Tensor,
    swa_global_topk: torch.Tensor,  # [B, win] int32 global slots into SWA pool
    bsz: int,
    q_len: int,
    win: int,
    sparse_op: SparseAttnV4DecodeOp,
) -> torch.Tensor:
    """SWA-only BF16 TileLang sparse attn — returns ``[B, q_len, H, D_v]``.

    Zero-copy: ``swa_pool_view.unsqueeze(0)`` feeds the kernel directly with
    indirect reads via the global slot indices; no gather buffer.
    """
    T = bsz * q_len
    q_packed = q.transpose(0, 1).contiguous() if q_len > 1 else q.transpose(0, 1)
    with record_function_range("dsv4.attn.sparse_attn"):
        o_packed = sparse_op.forward(
            q_packed,
            swa_pool_view.unsqueeze(0),
            attn_sink,
            swa_global_topk.view(1, T, win).contiguous(),
        )
    return o_packed.transpose(0, 1)


def attn_bf16_dual_paged(
    *,
    q: torch.Tensor,  # [B, 1, H, D] bf16  (q_len == 1 only)
    swa_pool_view: torch.Tensor,  # [num_global_slots, D] bf16
    cmp_pool_view: torch.Tensor,  # [num_global_slots, D] bf16
    swa_global_topk: torch.Tensor,  # [B, win] int32 global slots
    cmp_global_topk: torch.Tensor,  # [B, K_cmp] int32 global slots
    attn_sink: torch.Tensor,
    head_dim: int,
    bsz: int,
    q_len: int,
    win: int,
    sparse_op: SparseAttnV4DecodeOp,
) -> torch.Tensor:
    """Dual-pool BF16 sparse attn — returns ``[B, 1, H, D_v]``.

    Pipeline:
      1. ``gather_dual_pool_kv_packed`` packs SWA + compressed pool slots into
         ``kv_packed [B, win+K_cmp, D]``.
      2. Build identity packed topk: ``[arange(win) | arange(K_cmp)+win]``,
         masking -1 entries from the source global indices.
      3. One ``sparse_op.forward`` call.
    """
    assert q_len == 1, f"attn_bf16_dual_paged supports q_len=1 only (got {q_len})"
    K_cmp = cmp_global_topk.shape[-1]

    with record_function_range("dsv4.attn.kv_gather_dense_or_paged"):
        kv_packed_4d = gather_dual_pool_kv_packed(
            swa_pool_view,
            cmp_pool_view,
            swa_global_topk,
            cmp_global_topk,
            head_dim,
            bsz,
            q_len,
        )
    kv_packed = kv_packed_4d.view(bsz, win + K_cmp, head_dim)

    swa_valid = (swa_global_topk >= 0).view(bsz, q_len, win)
    cmp_valid = (cmp_global_topk >= 0).view(bsz, q_len, K_cmp)
    swa_topk = (
        torch.arange(win, device=kv_packed.device, dtype=torch.int32)
        .view(1, 1, win)
        .expand(bsz, q_len, win)
    )
    cmp_topk = (
        torch.arange(K_cmp, device=kv_packed.device, dtype=torch.int32)
        .add_(win)
        .view(1, 1, K_cmp)
        .expand(bsz, q_len, K_cmp)
    )
    packed_topk = torch.cat(
        [
            torch.where(swa_valid, swa_topk, torch.full_like(swa_topk, -1)),
            torch.where(cmp_valid, cmp_topk, torch.full_like(cmp_topk, -1)),
        ],
        dim=-1,
    )
    with record_function_range("dsv4.attn.sparse_attn"):
        return sparse_op.forward(q, kv_packed, attn_sink, packed_topk.contiguous())
