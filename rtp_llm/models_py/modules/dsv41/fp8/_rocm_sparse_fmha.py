"""Reference-math sparse MLA forward for ROCm (v0, correctness first).

Replaces ``flash_mla.flash_mla_sparse_fwd`` on gfx942 with the joint-softmax
semantics of the model's reference sparse attention: window keys and
compressed global keys arrive in ONE absolute index list and share ONE
softmax, one scale and one per-head additive sink logit:

    logits[r,h,k] = q[r,h,:] . kv[idx[r,k],:] * sm_scale
    p[r,h,k]      = exp(logits - rowmax)
    denom[r,h]    = sum_k p + exp(sink[h] - rowmax)
    out[r,h,:]    = sum_k p[r,h,k] * kv[idx[r,k],:] / denom

Invalid slots (index < 0, or slot >= topk_length when given) contribute to
neither numerator nor denominator; rows whose keys are all invalid decay to
the sink exactly as the reference ``_sparse_attn`` does (same clamp_min
(-1e30) row-max semantics).

Input/output contract mirrors the native wrapper's call site:
``q`` [R, H, D] BF16 (already RoPE'd), ``kv`` [S, 1, D] or [S, D] BF16,
``indices`` [R, K] int (compact valid prefix, -1 padding), ``topk_length``
[R] optional, ``attn_sink`` [H] optional fp32/bf16; returns [R, H, D] BF16.

This gather + fp32-einsum path materializes [chunk, K, D] intermediates and
is the correctness bring-up path; a fused Triton FMHA (and the graph-safe
decode op) is the follow-up performance mechanism.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

_DEFAULT_ROW_CHUNK = 1024


def _row_chunk() -> int:
    try:
        return max(1, int(os.environ.get("DSV41_ROCM_SPARSE_Q_CHUNK", "1024")))
    except ValueError:
        return _DEFAULT_ROW_CHUNK


def rocm_sparse_mla_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    row_chunk: Optional[int] = None,
) -> torch.Tensor:
    if q.dim() != 3:
        raise ValueError(f"expected q [R, H, D], got {tuple(q.shape)}")
    kv_flat = kv.reshape(kv.shape[0], -1) if kv.dim() == 3 else kv
    rows, heads, dim = q.shape
    out = torch.empty_like(q)
    if rows == 0:
        return out
    if kv_flat.shape[-1] != dim:
        raise ValueError(
            f"kv head dim {kv_flat.shape[-1]} != q head dim {dim}"
        )
    # Callers pass indices as [R, K] or with an inner q_len axis ([R, 1, K]);
    # normalize to one key axis per query row.
    if indices.dim() != 2:
        indices = indices.reshape(rows, -1)
    if indices.dtype != torch.int64:
        indices = indices.to(torch.int64)
    topk = indices.shape[1]
    k_slots = torch.arange(topk, device=q.device)
    if topk_length is not None:
        topk_length = topk_length.reshape(rows)
    sink = None
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).reshape(1, heads, 1)
    chunk = row_chunk if row_chunk is not None else _row_chunk()
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        idx = indices[start:stop]
        valid = idx >= 0
        if topk_length is not None:
            valid = valid & (
                k_slots[None, :] < topk_length[start : stop, None].to(torch.int64)
            )
        sel = kv_flat.index_select(0, idx.clamp_min(0).reshape(-1)).view(
            stop - start, topk, dim
        )
        qc = q[start:stop].to(torch.float32)
        sc = sel.to(torch.float32)
        logits = torch.einsum("rhd,rkd->rhk", qc, sc) * sm_scale
        logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
        row_max = logits.amax(dim=-1, keepdim=True).clamp_min(-1e30)
        probs = torch.exp(logits - row_max)
        denom = probs.sum(dim=-1, keepdim=True)
        if sink is not None:
            denom = denom + torch.exp(sink - row_max)
        acc = torch.einsum("rhk,rkd->rhd", probs, sc)
        out[start:stop] = (acc / denom).to(q.dtype)
    return out
