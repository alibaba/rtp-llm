"""GQA block-sparse attention baseline (pure torch, prefill shape).

The indexer produces ``selected_indices [B, S, K]`` (-1 padded); this module
gathers K and V entries at those indices and computes softmax attention using
the standard GQA head mapping (H_q // H_kv query heads per KV head).

Contract::

    q [B, H_q, S, D],  k [B, H_kv, T, D],  v [B, H_kv, T, D]
    selected_indices [B, S, K] int32
    -> o [B, H_q, S, D]

For each (b, h_q, s):
    h_kv    = h_q // (H_q // H_kv)
    gather  k_gathered, v_gathered from (b, h_kv, sel[t], :)
    scores  = q[b, h_q, s] @ k_gathered^T / sqrt(D)
    attn    = softmax(scores, masked -inf at -1 positions)
    o       = attn @ v_gathered
"""

from typing import Optional

import torch
import torch.nn.functional as F


def gather_and_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    K = selected_indices.shape[-1]
    dev = q.device
    if scale is None:
        scale = 1.0 / float(D) ** 0.5

    q_per_kv = H_q // H_kv
    out = torch.zeros(B, H_q, S, D, device=dev, dtype=q.dtype)
    for start in range(0, S, 32):
        end = min(start + 32, S)
        clamped = selected_indices[:, start:end].clamp(min=0).long()
        mask = selected_indices[:, start:end] >= 0
        for h_q in range(H_q):
            h_kv = h_q // q_per_kv
            ks, vs = k[:, h_kv], v[:, h_kv]
            indices = clamped.unsqueeze(-1).expand(-1, -1, -1, D)
            k_sel = ks.unsqueeze(1).expand(-1, end - start, -1, -1).gather(2, indices)
            v_sel = vs.unsqueeze(1).expand(-1, end - start, -1, -1).gather(2, indices)
            qh = q[:, h_q, start:end]
            scores = torch.matmul(qh.unsqueeze(-2), k_sel.transpose(-2, -1))
            scores = scores.squeeze(-2) * scale
            scores = torch.where(mask, scores, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = torch.where(mask.any(dim=-1, keepdim=True), attn, 0.0)
            out[:, h_q, start:end] = torch.matmul(attn.unsqueeze(-2), v_sel).squeeze(-2)
            del k_sel, v_sel
    return out
