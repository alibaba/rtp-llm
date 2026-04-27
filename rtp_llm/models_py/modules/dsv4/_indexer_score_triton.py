"""V4 Indexer score Triton kernel — fused einsum + ReLU + weighted-sum.

Replaces the chunked materialization in ``indexer.py`` (final_plan.md P0):

    index_score = einsum("bshd,btd->bsht", q_f, kv_f)        # [B,S,H,T]
    index_score = (index_score.relu_() * w_f.unsqueeze(-1)).sum(dim=2)  # [B,S,T]

with a single Triton kernel that streams the head dimension and never
materializes ``[B,S,H,T]``.  Output ``[B,S,T] fp32``, identical math.

Memory cost drops from ``S*H*T*4`` bytes (the ``relu`` intermediate) to
zero — ``index_score`` itself is the only live tensor.

Optional causal mask for prefill (``apply_causal_mask=True``):
    out[b, s, t] += -inf  if  t >= (q_pos[s] + 1) // compress_ratio

Folds the mask into the kernel so we don't allocate a separate ``[B,S,T]``
mask tensor for the long-context case.

Caller passes:
  q       [B, S, H, D] bf16
  kv      [B, T, D]    bf16
  weights [B, S, H]    bf16/fp32 (cast inside kernel)
  q_pos   [B, S] int32 — global Q position for each row, OR None when not
           applying causal mask (decode / start_pos > 0)
  compress_ratio: int — divisor for the causal kv-column threshold

Returns:
  out     [B, S, T] fp32

Shapes constraints (V4-Flash defaults):
  H = 64, D = 128 — both fit in registers per program; we tile S × T.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_indexer_score_fwd(
    q_ptr,           # [B, S, H, D] bf16
    kv_ptr,          # [B, T, D]    bf16
    w_ptr,           # [B, S, H]    fp32 (caller casts)
    q_pos_ptr,       # [B, S] int32, or 0 if APPLY_MASK==False
    out_ptr,         # [B, S, T] fp32

    # strides
    q_b: tl.constexpr, q_s: tl.constexpr, q_h: tl.constexpr,   # q strides; d-stride==1 assumed (contiguous)
    kv_b: tl.constexpr, kv_t: tl.constexpr,                     # kv strides; d-stride==1 assumed
    w_b: tl.constexpr, w_s: tl.constexpr,                       # w strides; h-stride==1 assumed
    qpos_b: tl.constexpr,                                       # q_pos stride for B; s-stride==1
    out_b: tl.constexpr, out_s: tl.constexpr,                   # out strides; t-stride==1 assumed

    # geometry
    S: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,

    # tile sizes
    BLOCK_S: tl.constexpr,
    BLOCK_T: tl.constexpr,

    # toggles
    APPLY_MASK: tl.constexpr,
):
    """One program covers one (b, s_block, t_block) tile.

    Inner loop streams over H heads, computing
      score[s, t] = sum_h relu(sum_d q[s,h,d] * k[t,d]) * w[s,h]
    using fp32 accumulators.  D and H are kept as compile-time constants
    so each per-head dot is unrolled into a 1-shot tile load + reduce.
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_t = tl.program_id(2)

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    s_mask = s_off < S
    t_mask = t_off < T

    # K tile [BLOCK_T, D] — shared across all S rows in this program.
    d_idx = tl.arange(0, D)
    k_ptrs = (
        kv_ptr
        + pid_b * kv_b
        + t_off[:, None] * kv_t
        + d_idx[None, :]
    )
    k_tile = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_T, D]

    # Causal mask (apply once per program; per-tile).  q_pos[s] is the global
    # position of S row s; the row may attend to compressed-KV columns
    # [0, (q_pos+1) // ratio).
    if APPLY_MASK:
        qp_ptrs = q_pos_ptr + pid_b * qpos_b + s_off
        q_pos = tl.load(qp_ptrs, mask=s_mask, other=0).to(tl.int32)  # [BLOCK_S]
        # threshold[s] = (q_pos + 1) // ratio
        thr = (q_pos + 1) // COMPRESS_RATIO  # [BLOCK_S]
        # mask[s, t] = (t < thr[s])
        # broadcast t_off [BLOCK_T] vs thr [BLOCK_S]
        causal = t_off[None, :] < thr[:, None]   # [BLOCK_S, BLOCK_T]
    else:
        causal = tl.full((BLOCK_S, BLOCK_T), True, dtype=tl.int1)

    acc = tl.zeros((BLOCK_S, BLOCK_T), dtype=tl.float32)

    for h in tl.static_range(H):
        # q[s, h, d] tile [BLOCK_S, D]
        q_ptrs = (
            q_ptr
            + pid_b * q_b
            + s_off[:, None] * q_s
            + h * q_h
            + d_idx[None, :]
        )
        q_tile = tl.load(q_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_S, D]

        # dot: [BLOCK_S, D] x [BLOCK_T, D]^T = [BLOCK_S, BLOCK_T]
        score = tl.dot(q_tile, tl.trans(k_tile))

        # ReLU (V4 indexer math), then per-head weighted accumulate
        score = tl.where(score > 0.0, score, 0.0)
        w_ptrs = w_ptr + pid_b * w_b + s_off * w_s + h
        w = tl.load(w_ptrs, mask=s_mask, other=0.0).to(tl.float32)  # [BLOCK_S]
        acc += score * w[:, None]

    # Write masked output: invalid s/t rows stay as 0 (won't be read by
    # topk).  Causal-masked entries get -inf so the downstream topk
    # naturally drops them.
    final = tl.where(causal, acc, float("-inf"))
    out_ptrs = (
        out_ptr
        + pid_b * out_b
        + s_off[:, None] * out_s
        + t_off[None, :]
    )
    write_mask = s_mask[:, None] & t_mask[None, :]
    tl.store(out_ptrs, final, mask=write_mask)


def v4_indexer_score(
    q: torch.Tensor,                       # [B, S, H, D] bf16, contiguous in last dim
    kv: torch.Tensor,                      # [B, T, D]    bf16
    weights: torch.Tensor,                 # [B, S, H]    bf16 or fp32
    q_pos: Optional[torch.Tensor] = None,  # [B, S] int32 — global Q position; None for no mask
    compress_ratio: int = 1,
) -> torch.Tensor:
    """V4 indexer fused score.  See module docstring for the kernel math.

    When ``q_pos`` is None (decode / continuation), the causal mask is
    skipped — the caller is responsible for any masking afterwards.
    """
    assert q.dtype == torch.bfloat16, f"q dtype={q.dtype}"
    assert kv.dtype == torch.bfloat16, f"kv dtype={kv.dtype}"
    assert q.is_contiguous(), "q must be contiguous"
    assert kv.is_contiguous(), "kv must be contiguous"
    assert q.dim() == 4 and kv.dim() == 3
    B, S, H, D = q.shape
    Bk, T, Dk = kv.shape
    assert B == Bk and D == Dk, f"q/kv batch/D mismatch: q={q.shape} kv={kv.shape}"
    assert weights.shape == (B, S, H), f"weights shape={weights.shape} expected {(B,S,H)}"

    weights = weights.contiguous()
    if weights.dtype != torch.float32:
        weights = weights.float()

    apply_mask = q_pos is not None
    if apply_mask:
        q_pos = q_pos.contiguous()
        assert q_pos.shape == (B, S), f"q_pos shape={q_pos.shape} expected {(B,S)}"
        assert q_pos.dtype in (torch.int32, torch.int64), f"q_pos dtype={q_pos.dtype}"
        if q_pos.dtype != torch.int32:
            q_pos = q_pos.to(torch.int32)
    else:
        # passing a tiny dummy avoids a Triton constexpr branch on a real ptr
        q_pos = torch.empty(1, dtype=torch.int32, device=q.device)

    out = torch.empty((B, S, T), dtype=torch.float32, device=q.device)
    if S == 0 or T == 0:
        return out

    BLOCK_S = 32 if S >= 32 else triton.next_power_of_2(S)
    BLOCK_T = 64 if T >= 64 else triton.next_power_of_2(T)

    grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(T, BLOCK_T))

    _v4_indexer_score_fwd[grid](
        q, kv, weights, q_pos, out,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(1),
        weights.stride(0), weights.stride(1),
        q_pos.stride(0) if apply_mask else 0,
        out.stride(0), out.stride(1),
        S=S, T=T, H=H, D=D,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_S=BLOCK_S, BLOCK_T=BLOCK_T,
        APPLY_MASK=apply_mask,
        num_warps=4,
        num_stages=2,
    )
    return out
