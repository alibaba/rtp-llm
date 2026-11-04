"""Triton Lightning-indexer scoring for LongCat LSA (baseline version).

Computes the DSA-style indexer scores used by LSA slot selection::

    I[t, s] = sum_j w[t, j] * ReLU(q_idx[t, j] . k_idx[s]) * scale

This baseline kernel materializes the full ``[T, N]`` score matrix; the fused
score+top-k path (and the hierarchical two-stage selection) is part of run plan
P2. Slot selection itself lives in :mod:`.reference` (``select_slots``) and is
reused unchanged so that kernel and reference share the same SI semantics.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _indexer_scores_kernel(
    QIDX,
    KIDX,
    WEIGHTS,
    OUT,
    T,
    N,
    scale,
    HI: tl.constexpr,
    DI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Score tile ``[BLOCK_M, BLOCK_N]`` over the indexer heads."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, DI)
    m_mask = offs_m < T
    n_mask = offs_n < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for j in tl.static_range(HI):
        q = tl.load(
            QIDX + offs_m[:, None] * (HI * DI) + j * DI + offs_d[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        k = tl.load(
            KIDX + offs_n[:, None] * DI + offs_d[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        dots = tl.dot(q, tl.trans(k)) * scale
        relu = tl.maximum(dots, 0.0)
        w = tl.load(WEIGHTS + offs_m * HI + j, mask=m_mask, other=0.0)
        acc += relu * w.to(tl.float32)[:, None]
    tl.store(
        OUT + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def indexer_scores_triton(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
    *,
    scale: Optional[float] = None,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Materialize ``[T, N]`` indexer scores with a Triton MQA kernel.

    Args:
        q_idx: ``[T, HI, DI]`` indexer queries.
        k_idx: ``[N, DI]`` shared indexer keys.
        weights: ``[T, HI]`` per-head weights.
        scale: dot scale; defaults to ``1 / sqrt(DI)`` (applied before ReLU;
            uniform positive scaling does not change the top-k ordering).
    Returns:
        ``[T, N]`` float32 scores.
    """
    if not q_idx.is_cuda:
        raise ValueError("indexer_scores_triton requires CUDA tensors")
    if q_idx.dim() != 3 or k_idx.dim() != 2 or weights.dim() != 2:
        raise ValueError("expect q_idx [T, HI, DI], k_idx [N, DI], weights [T, HI]")
    t_count, heads, head_dim = q_idx.shape
    n_count, k_dim = k_idx.shape
    if k_dim != head_dim or weights.shape != (t_count, heads):
        raise ValueError("shape mismatch between q_idx / k_idx / weights")

    q_idx = q_idx.contiguous()
    k_idx = k_idx.contiguous()
    weights = weights.contiguous()
    out = torch.empty((t_count, n_count), device=q_idx.device, dtype=torch.float32)
    grid = (triton.cdiv(t_count, block_m), triton.cdiv(n_count, block_n))
    _indexer_scores_kernel[grid](
        q_idx,
        k_idx,
        weights,
        out,
        t_count,
        n_count,
        scale if scale is not None else head_dim**-0.5,
        HI=heads,
        DI=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out