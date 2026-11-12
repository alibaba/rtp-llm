# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Adapted for rtp-llm: forward-only, supports USE_EXP2 for log2-space gates.
# Copied from fla/ops/gla/chunk.py: chunk_gla_fwd_kernel_o + chunk_gla_fwd_o_gk

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.autotune_cache import (
    autotune_cache_kwargs,
    cuda_cached_autotune,
)
from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
from rtp_llm.models_py.triton_kernels.fla.op import exp, exp2


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@cuda_cached_autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "TRANSPOSE_STATE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
        p_q_m0 = (p_q_i0 >= 0) & (p_q_i0 < (T))
        p_q_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
        p_q_m1 = (p_q_i1 >= 0) & (p_q_i1 < (K))
        p_q = (
            (q + (bos * H + i_h) * K)
            + p_q_i0[:, None] * (H * K)
            + p_q_i1[None, :] * (1)
        )
        p_g_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
        p_g_m0 = (p_g_i0 >= 0) & (p_g_i0 < (T))
        p_g_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
        p_g_m1 = (p_g_i1 >= 0) & (p_g_i1 < (K))
        p_g = (
            (g + (bos * H + i_h) * K)
            + p_g_i0[:, None] * (H * K)
            + p_g_i1[None, :] * (1)
        )
        if TRANSPOSE_STATE:
            p_h_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_h_m0 = (p_h_i0 >= 0) & (p_h_i0 < (V))
            p_h_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
            p_h_m1 = (p_h_i1 >= 0) & (p_h_i1 < (K))
            p_h = (
                (h + (i_tg * H + i_h) * K * V)
                + p_h_i0[:, None] * (K)
                + p_h_i1[None, :] * (1)
            )
        else:
            p_h_i0 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
            p_h_m0 = (p_h_i0 >= 0) & (p_h_i0 < (K))
            p_h_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_h_m1 = (p_h_i1 >= 0) & (p_h_i1 < (V))
            p_h = (
                (h + (i_tg * H + i_h) * K * V)
                + p_h_i0[:, None] * (V)
                + p_h_i1[None, :] * (1)
            )

        # [BT, BK]
        b_q = tl.load(p_q, mask=p_q_m0[:, None] & p_q_m1[None, :], other=0)
        # [BT, BK]
        b_g = tl.load(p_g, mask=p_g_m0[:, None] & p_g_m1[None, :], other=0).to(
            tl.float32
        )
        # [BT, BK]
        if USE_EXP2:
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        else:
            b_qg = (b_q * exp(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, mask=p_h_m0[:, None] & p_h_m1[None, :], other=0)
        if i_k >= 0:
            if TRANSPOSE_STATE:
                b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    b_o *= scale
    p_v_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
    p_v_m0 = (p_v_i0 >= 0) & (p_v_i0 < (T))
    p_v_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
    p_v_m1 = (p_v_i1 >= 0) & (p_v_i1 < (V))
    p_v = (v + (bos * H + i_h) * V) + p_v_i0[:, None] * (H * V) + p_v_i1[None, :] * (1)
    p_o_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
    p_o_m0 = (p_o_i0 >= 0) & (p_o_i0 < (T))
    p_o_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
    p_o_m1 = (p_o_i1 >= 0) & (p_o_i1 < (V))
    p_o = (o + (bos * H + i_h) * V) + p_o_i0[:, None] * (H * V) + p_o_i1[None, :] * (1)
    p_A_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
    p_A_m0 = (p_A_i0 >= 0) & (p_A_i0 < (T))
    p_A_i1 = tl.arange(0, BT).to(tl.int64) + (0)
    p_A_m1 = (p_A_i1 >= 0) & (p_A_i1 < (BT))
    p_A = (
        (A + (bos * H + i_h) * BT) + p_A_i0[:, None] * (H * BT) + p_A_i1[None, :] * (1)
    )
    # [BT, BV]
    b_v = tl.load(p_v, mask=p_v_m0[:, None] & p_v_m1[None, :], other=0)
    # [BT, BT]
    b_A = tl.load(p_A, mask=p_A_m0[:, None] & p_A_m1[None, :], other=0)
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=p_o_m0[:, None] & p_o_m1[None, :])


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Please ensure zeros, since vllm will use padding v
    o = torch.zeros_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return o
