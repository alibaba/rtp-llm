"""Whole-LSA-attention-in-one-FlashMLA-call path (optional, SM90+ only).

Instead of running the dense window (``swa_dense_fwd``) plus the dynamic sparse
set (``sparse_mla_triton``) and merging the two partial softmaxes, this entry
builds the *union* slot list per query row -- sink tokens + sliding window +
dynamic sparse slots -- and evaluates all of it with one
``flash_mla_sparse_fwd`` call, which needs no merge at all.

Measured on B300 for the prefill shape (t=2048, ctx128k, 2048 slots/row):
0.7508 ms / MFU 15.6% against 1.1049 ms / 10.6% for the composed SWA+sparse+merge
path (baseline-log #5.138), with numerically equivalent results.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_flashmla import (
    flashmla_available,
    sparse_flashmla_fwd,
)

__all__ = ["lsa_attention_flashmla", "lsa_flashmla_available"]


def lsa_flashmla_available() -> bool:
    return flashmla_available()


def _build_union_slots(
    positions: torch.Tensor,
    dyn_indices: torch.Tensor,
    window: int,
    init_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Sink prefix + sliding window + dynamic slots, one row per query token."""
    t = positions.shape[0]
    dyn = dyn_indices
    if dyn.dim() == 3:
        dyn = dyn.squeeze(1)
    dyn = dyn.to(torch.int64)
    win_len = min(window, int(positions.max().item()) + 1 if positions.numel() else window)

    win = positions[:, None] - (window - 1) + torch.arange(window, device=device)[None, :]
    win = torch.where(win < 0, torch.full_like(win, -1), win)

    sink = torch.arange(init_tokens, device=device, dtype=torch.int64).expand(t, -1)
    parts = [sink, win, dyn]
    union = torch.cat(parts, dim=1)
    # FlashMLA wants the slot-list length to be a multiple of 128; -1 marks pads.
    s_len = union.shape[1]
    if s_len % 128:
        pad = torch.full((t, (s_len + 127) // 128 * 128 - s_len), -1,
                         dtype=union.dtype, device=device)
        union = torch.cat([union, pad], dim=1)
    return union


def lsa_attention_flashmla(
    q_abs: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    dyn_indices: torch.Tensor,
    *,
    window: int = 1024,
    init_tokens: int = 16,
    kv_lora_rank: int = 512,
    scale: Optional[float] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full LSA attention in a single FlashMLA sparse call (no merge).

    Same contract as ``lsa_forward.lsa_attention``: returns ``(out [T, H, DV], lse
    [T, H])`` with lse in natural log. Falls back to the composed path when
    FlashMLA is not runnable on this device (compute capability < 9).
    """
    if not flashmla_available():
        from rtp_llm.models_py.triton_kernels.lsa_attention.lsa_forward import (
            lsa_attention,
        )

        return lsa_attention(
            q_abs, kv, positions, dyn_indices, window=window, init_tokens=init_tokens,
            kv_lora_rank=kv_lora_rank, scale=scale, out_dtype=out_dtype,
        )

    union = _build_union_slots(positions, dyn_indices, window, init_tokens, q_abs.device)
    return sparse_flashmla_fwd(
        q_abs, kv, union, kv_lora_rank=kv_lora_rank, out_dtype=out_dtype, scale=scale,
    )
