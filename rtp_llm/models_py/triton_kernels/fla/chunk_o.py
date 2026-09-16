# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
from rtp_llm.models_py.triton_kernels.fla.op import exp, exp2, safe_exp
from rtp_llm.models_py.triton_kernels.fla.utils import (
    check_shared_mem,
    is_amd,
    is_amd_cdna3,
    is_nvidia_hopper,
)

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
# @triton.autotune(
#     configs=[
#         triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
#         for BK in BKV_LIST
#         for BV in BKV_LIST
#         for num_warps in NUM_WARPS
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "V", "BT"],
# )
@triton.jit(do_not_specialize=["T", "T_FLAT"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    T_FLAT,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_LOG2: tl.constexpr,
    V_HEAD_MAJOR: tl.constexpr,
    G_HEAD_MAJOR: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_b64, i_h64 = i_b.to(tl.int64), i_h.to(tl.int64)

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b64 * NT + i_t
        bos, eos = i_b64 * T, i_b64 * T + T

    # Promote indices before any stride multiplication. Long packed sequences
    # and head-major tensors can otherwise overflow int32 while computing the
    # element offset, even if the completed expression is cast to int64 later.
    i_tg64 = i_tg.to(tl.int64)
    bos64 = bos.to(tl.int64)

    # offset calculation
    q += (bos64 * Hg + i_h64 // (H // Hg)) * K
    k += (bos64 * Hg + i_h64 // (H // Hg)) * K
    if V_HEAD_MAJOR:
        if IS_VARLEN:
            v += (i_h64 * T_FLAT + bos64) * V
        else:
            v += ((i_b64 * H + i_h64) * T_FLAT) * V
    else:
        v += (bos64 * H + i_h64) * V
    o += (bos64 * H + i_h64) * V
    h += (i_tg64 * H + i_h64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        # V-first h view (matches chunk_delta_h.py / SGL main): (V, K) + (K, 1).
        p_h = tl.make_block_ptr(
            h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV] — transpose b_h at dot time
        b_o += tl.dot(b_q, tl.trans(b_h.to(b_q.dtype)))
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        if G_HEAD_MAJOR:
            if IS_VARLEN:
                g += i_h64 * T_FLAT + bos64
            else:
                g += (i_b64 * H + i_h64) * T_FLAT
            g_stride = 1
        else:
            g += bos64 * H + i_h64
            g_stride = H
        p_g = tl.make_block_ptr(g, (T,), (g_stride,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        if IS_LOG2:
            # AMD path: g is in log2 domain (RCP_LN2-scaled cumsum upstream).
            # Within each chunk b_g[i] - b_g[j] ≤ 0 for i ≥ j (cumsum of
            # logsigmoid ≤ 0), so exp2 ∈ (0, 1]. The m_A mask below zeros
            # out i < j entries.
            b_o = b_o * exp2(b_g)[:, None]
            b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
        else:
            # NVIDIA path: g is in natural-log domain, keep exp/safe_exp to
            # preserve bit-level semantics with the original implementation.
            b_o = b_o * exp(b_g)[:, None]
            b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    if IS_LOG2:
        # AMD path: also mask out OOB rows/cols at the last chunk so b_A
        # accumulator never multiplies garbage data when T % BT != 0.
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    else:
        # NVIDIA path: keep the original lower-triangular mask only, to be
        # bit-level identical to the pre-optimization implementation.
        o_i = tl.arange(0, BT)
        m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    v_stride_t = V if V_HEAD_MAJOR else H * V
    p_v = tl.make_block_ptr(
        v, (T, V), (v_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.zeros_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        T_FLAT=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=64 if is_amd else 128,
        BV=128 if is_amd else 64,
        IS_LOG2=is_amd,
        V_HEAD_MAJOR=False,
        G_HEAD_MAJOR=False,
        num_warps=(4 if h.dtype == torch.float32 else 1) if is_amd else 4,
        num_stages=1 if is_amd_cdna3 else 2,
    )
    return o


def chunk_fwd_o_head_major_vk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Run RTP K6 directly on AITER's head-major K5 outputs.

    AITER returns ``v`` as ``[B, H, T, V]`` and cumulative gates as
    ``[B, H, T]``. Compile-time layout flags avoid transpose/contiguous copies;
    the returned output keeps RTP's token-major ``[B, T, H, V]`` contract.
    """
    if v.ndim != 4 or g.ndim != 3:
        raise ValueError("head-major RTP K6 expects rank-4 v and rank-3 g")
    B, T, Hg, K = q.shape
    v_batch, H, T_flat, V = v.shape
    if v_batch != B or g.shape != (B, H, T_flat):
        raise ValueError("head-major RTP K6 received incompatible v/g shapes")
    if T_flat != T:
        raise ValueError(f"expected v token extent {T}, got {T_flat}")

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = K**-0.5
    output = torch.empty(B, T, H, V, dtype=v.dtype, device=v.device)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        output,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        T_FLAT=T_flat,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=64 if is_amd else 128,
        BV=128 if is_amd else 64,
        IS_LOG2=is_amd,
        V_HEAD_MAJOR=True,
        G_HEAD_MAJOR=True,
        num_warps=(4 if h.dtype == torch.float32 else 1) if is_amd else 4,
        num_stages=1 if is_amd_cdna3 else 2,
    )
    return output
