"""Triton sparse MLA attention for GPUs without a FlashMLA kernel.

The installed FlashMLA wheel only carries ``sm_100a``/``sm_103a`` cubins.
L20 (SM89) therefore needs a real sparse-attention implementation for both
prefill and decode rather than a dense fallback (a 1M-token dense attention
would be quadratic).

For one query row all query heads share the same top-k slot list.  This module
exploits that: one program owns one query row and a small head tile, gathers
the selected KV rows once, and runs the QK/PV tensor-core dot products for all
heads in the tile.  It supports the BF16 dense prefill KV layout and the
dequantized decode layout (the caller dequantizes packed FP8/FP4 pools first).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_TILE = 16
BLOCK_N = 16


@triton.jit
def _sparse_mla_kernel(
    q_ptr,  # [NQ, H, D] contiguous
    kv_ptr,  # [T, D] contiguous
    idx_ptr,  # [NQ, K] int32, -1 = masked
    len_ptr,  # [NQ] int32 or 0
    sink_ptr,  # [H] float32 or 0
    out_ptr,  # [NQ, H, D] contiguous
    lse_ptr,  # [NQ, H] float32
    max_ptr,  # [NQ, H] float32
    NQ,
    H,
    K,
    sm_scale,
    D: tl.constexpr = 512,
    BH: tl.constexpr = 16,
    BN: tl.constexpr = 16,
    HAS_LEN: tl.constexpr = False,
    HAS_SINK: tl.constexpr = False,
):
    row = tl.program_id(0)
    head_base = tl.program_id(1) * BH
    if row >= NQ:
        return

    heads = head_base + tl.arange(0, BH)
    h_mask = heads < H
    d = tl.arange(0, D)

    q = tl.load(
        q_ptr + (row * H + heads[:, None]) * D + d[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    if HAS_LEN:
        length = tl.load(len_ptr + row).to(tl.int32)
    else:
        length = K

    if HAS_SINK:
        sink = tl.load(sink_ptr + heads, mask=h_mask, other=-float("inf")).to(
            tl.float32
        )
        m_i = sink
        l_i = tl.where(h_mask, 1.0, 0.0)
    else:
        m_i = tl.full((BH,), -float("inf"), tl.float32)
        l_i = tl.zeros((BH,), tl.float32)

    acc = tl.zeros((BH, D), tl.float32)
    seen = tl.zeros((BH,), tl.int1)

    for start_n in range(0, K, BN):
        cols = start_n + tl.arange(0, BN)
        n_mask = cols < K
        idx = tl.load(
            idx_ptr + row * K + cols,
            mask=n_mask,
            other=-1,
        ).to(tl.int64)
        valid = (idx >= 0) & n_mask & (cols < length)
        seen = seen | (tl.max(valid.to(tl.int32), axis=0) > 0)
        kv = tl.load(
            kv_ptr + idx[:, None] * D + d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        qk = tl.dot(q, tl.trans(kv)) * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        block_max = tl.max(qk, axis=1)
        new_m = tl.maximum(m_i, block_max)
        alpha = tl.where(m_i == -float("inf"), 1.0, tl.exp(m_i - new_m))
        p = tl.where(valid[None, :], tl.exp(qk - new_m[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
        m_i = new_m

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    out = acc * inv_l[:, None]
    tl.store(
        out_ptr + (row * H + heads[:, None]) * D + d[None, :],
        out.to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    lse = tl.where(
        (l_i > 0) & seen,
        m_i + tl.log(tl.maximum(l_i, 1e-30)),
        float("inf"),
    )
    tl.store(lse_ptr + row * H + heads, lse, mask=h_mask)
    tl.store(max_ptr + row * H + heads, m_i, mask=h_mask)


def _launch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    sm_scale: float,
):
    nq, h, d = q.shape
    k = indices.shape[1]
    out = torch.empty((nq, h, d), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((nq, h), dtype=torch.float32, device=q.device)
    maximum = torch.empty((nq, h), dtype=torch.float32, device=q.device)
    if nq == 0 or k == 0:
        if k == 0:
            out.zero_()
            lse.fill_(float("inf"))
            maximum.fill_(float("-inf"))
        return out, maximum, lse

    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.contiguous()
    if topk_length is not None:
        topk_length = topk_length.contiguous()
    if attn_sink is not None:
        attn_sink = attn_sink.to(device=q.device, dtype=torch.float32).contiguous()

    grid = (nq, triton.cdiv(h, HEAD_TILE))
    _sparse_mla_kernel[grid](
        q,
        kv,
        indices,
        topk_length if topk_length is not None else q,
        attn_sink if attn_sink is not None else q,
        out,
        lse,
        maximum,
        nq,
        h,
        k,
        float(sm_scale),
        D=d,
        BH=HEAD_TILE,
        BN=BLOCK_N,
        HAS_LEN=topk_length is not None,
        HAS_SINK=attn_sink is not None,
        num_warps=4,
        num_stages=2,
    )
    return out, maximum, lse


def sparse_mla_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
):
    """BF16 sparse MLA.

    Args:
      q: ``[NQ, H, D]`` (prefill) or ``[B, S, H, D]`` (decode).
      kv: ``[T, D]`` or ``[T, 1, D]`` BF16.
      indices: ``[NQ, K]``, ``[NQ, 1, K]`` or ``[B, S, 1, K]`` int32.
      topk_length: optional ``[NQ]`` int32 left-aligned valid lengths.
    """
    if q.dim() == 4:
        b, s, h, d = q.shape
        q = q.reshape(b * s, h, d)
        if indices.dim() == 4:
            indices = indices.reshape(b * s, indices.shape[-1])
        elif indices.dim() == 3:
            indices = indices.reshape(b * s, indices.shape[-1])
        if topk_length is not None:
            topk_length = topk_length.reshape(b * s)
    else:
        if indices.dim() == 3:
            indices = indices.reshape(indices.shape[0], indices.shape[-1])
    if kv.dim() == 3:
        kv = kv[:, 0, :]
    return _launch(q, kv, indices, topk_length, attn_sink, sm_scale)