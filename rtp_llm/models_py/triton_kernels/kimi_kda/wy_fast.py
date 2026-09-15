# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Adapted for rtp-llm: forward-only, no backward.

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.autotune_cache import (
    autotune_cache_kwargs,
    cuda_cached_autotune,
)
from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
from rtp_llm.models_py.triton_kernels.fla.op import exp2


@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@cuda_cached_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kda_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
    p_b_m0 = (p_b_i0 >= 0) & (p_b_i0 < (T))
    p_b = (beta + bos * H + i_h) + p_b_i0 * (H)
    b_b = tl.load(p_b, mask=p_b_m0, other=0)

    p_A_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
    p_A_m0 = (p_A_i0 >= 0) & (p_A_i0 < (T))
    p_A_i1 = tl.arange(0, BT).to(tl.int64) + (0)
    p_A_m1 = (p_A_i1 >= 0) & (p_A_i1 < (BT))
    p_A = (
        (A + (bos * H + i_h) * BT) + p_A_i0[:, None] * (H * BT) + p_A_i1[None, :] * (1)
    )
    b_A = tl.load(p_A, mask=p_A_m0[:, None] & p_A_m1[None, :], other=0)

    for i_v in range(tl.cdiv(V, BV)):
        p_v_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
        p_v_m0 = (p_v_i0 >= 0) & (p_v_i0 < (T))
        p_v_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
        p_v_m1 = (p_v_i1 >= 0) & (p_v_i1 < (V))
        p_v = (
            (v + (bos * H + i_h) * V)
            + p_v_i0[:, None] * (H * V)
            + p_v_i1[None, :] * (1)
        )
        p_u_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
        p_u_m0 = (p_u_i0 >= 0) & (p_u_i0 < (T))
        p_u_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
        p_u_m1 = (p_u_i1 >= 0) & (p_u_i1 < (V))
        p_u = (
            (u + (bos * H + i_h) * V)
            + p_u_i0[:, None] * (H * V)
            + p_u_i1[None, :] * (1)
        )
        b_v = tl.load(p_v, mask=p_v_m0[:, None] & p_v_m1[None, :], other=0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(
            p_u, b_u.to(p_u.dtype.element_ty), mask=p_u_m0[:, None] & p_u_m1[None, :]
        )

    for i_k in range(tl.cdiv(K, BK)):
        p_w_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
        p_w_m0 = (p_w_i0 >= 0) & (p_w_i0 < (T))
        p_w_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
        p_w_m1 = (p_w_i1 >= 0) & (p_w_i1 < (K))
        p_w = (
            (w + (bos * H + i_h) * K)
            + p_w_i0[:, None] * (H * K)
            + p_w_i1[None, :] * (1)
        )
        p_k_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
        p_k_m0 = (p_k_i0 >= 0) & (p_k_i0 < (T))
        p_k_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
        p_k_m1 = (p_k_i1 >= 0) & (p_k_i1 < (K))
        p_k = (
            (k + (bos * H + i_h) * K)
            + p_k_i0[:, None] * (H * K)
            + p_k_i1[None, :] * (1)
        )
        b_k = tl.load(p_k, mask=p_k_m0[:, None] & p_k_m1[None, :], other=0)
        b_kb = b_k * b_b[:, None]

        p_gk_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
        p_gk_m0 = (p_gk_i0 >= 0) & (p_gk_i0 < (T))
        p_gk_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
        p_gk_m1 = (p_gk_i1 >= 0) & (p_gk_i1 < (K))
        p_gk = (
            (gk + (bos * H + i_h) * K)
            + p_gk_i0[:, None] * (H * K)
            + p_gk_i1[None, :] * (1)
        )
        b_gk = tl.load(p_gk, mask=p_gk_m0[:, None] & p_gk_m1[None, :], other=0).to(
            tl.float32
        )
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_q_m0 = (p_q_i0 >= 0) & (p_q_i0 < (T))
            p_q_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
            p_q_m1 = (p_q_i1 >= 0) & (p_q_i1 < (K))
            p_q = (
                (q + (bos * H + i_h) * K)
                + p_q_i0[:, None] * (H * K)
                + p_q_i1[None, :] * (1)
            )
            p_qg_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_qg_m0 = (p_qg_i0 >= 0) & (p_qg_i0 < (T))
            p_qg_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
            p_qg_m1 = (p_qg_i1 >= 0) & (p_qg_i1 < (K))
            p_qg = (
                (qg + (bos * H + i_h) * K)
                + p_qg_i0[:, None] * (H * K)
                + p_qg_i1[None, :] * (1)
            )
            b_q = tl.load(p_q, mask=p_q_m0[:, None] & p_q_m1[None, :], other=0)
            b_qg = b_q * exp2(b_gk)
            tl.store(
                p_qg,
                b_qg.to(p_qg.dtype.element_ty),
                mask=p_qg_m0[:, None] & p_qg_m1[None, :],
            )
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0
            ).to(tl.float32)
            b_kg = b_k * tl.where(
                (i_t * BT + tl.arange(0, BT) < T)[:, None],
                exp2(b_gn[None, :] - b_gk),
                0,
            )
            p_kg_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_kg_m0 = (p_kg_i0 >= 0) & (p_kg_i0 < (T))
            p_kg_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
            p_kg_m1 = (p_kg_i1 >= 0) & (p_kg_i1 < (K))
            p_kg = (
                (kg + (bos * H + i_h) * K)
                + p_kg_i0[:, None] * (H * K)
                + p_kg_i1[None, :] * (1)
            )
            tl.store(
                p_kg,
                b_kg.to(p_kg.dtype.element_ty),
                mask=p_kg_m0[:, None] & p_kg_m1[None, :],
            )

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(
            p_w, b_w.to(p_w.dtype.element_ty), mask=p_w_m0[:, None] & p_w_m1[None, :]
        )


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    qg = torch.empty_like(q) if q is not None else None
    kg = torch.empty_like(k) if gk is not None else None
    recompute_w_u_fwd_kda_kernel[(NT, B * H)](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u, qg, kg
