# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Token-parallel implementation of KDA intra chunk kernel
# Adapted for rtp-llm: forward-only.

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.autotune_cache import (
    autotune_cache_kwargs,
    cuda_cached_autotune,
)
from rtp_llm.models_py.triton_kernels.fla.op import exp2


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@cuda_cached_autotune(
    configs=[
        triton.Config({"BH": BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "N"])
def chunk_kda_fwd_kernel_intra_token_parallel(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N

        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    q += bos * H * K
    k += bos * H * K
    g += bos * H * K
    Aqk += bos * H * BT
    Akk += bos * H * BC
    beta += bos * H

    BK: tl.constexpr = triton.next_power_of_2(K)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = (i_hg * BH + o_h) < H
    m_k = o_k < K

    p_q_i0 = tl.arange(0, BH).to(tl.int64) + (i_hg * BH)
    p_q_m0 = (p_q_i0 >= 0) & (p_q_i0 < (H))
    p_q_i1 = tl.arange(0, BK).to(tl.int64) + (0)
    p_q_m1 = (p_q_i1 >= 0) & (p_q_i1 < (K))
    p_q = (q + i_t * H * K) + p_q_i0[:, None] * (K) + p_q_i1[None, :] * (1)
    p_k_i0 = tl.arange(0, BH).to(tl.int64) + (i_hg * BH)
    p_k_m0 = (p_k_i0 >= 0) & (p_k_i0 < (H))
    p_k_i1 = tl.arange(0, BK).to(tl.int64) + (0)
    p_k_m1 = (p_k_i1 >= 0) & (p_k_i1 < (K))
    p_k = (k + i_t * H * K) + p_k_i0[:, None] * (K) + p_k_i1[None, :] * (1)
    p_g_i0 = tl.arange(0, BH).to(tl.int64) + (i_hg * BH)
    p_g_m0 = (p_g_i0 >= 0) & (p_g_i0 < (H))
    p_g_i1 = tl.arange(0, BK).to(tl.int64) + (0)
    p_g_m1 = (p_g_i1 >= 0) & (p_g_i1 < (K))
    p_g = (g + i_t * H * K) + p_g_i0[:, None] * (K) + p_g_i1[None, :] * (1)
    p_beta_i0 = tl.arange(0, BH).to(tl.int64) + (i_hg * BH)
    p_beta_m0 = (p_beta_i0 >= 0) & (p_beta_i0 < (H))
    p_beta = (beta + i_t * H) + p_beta_i0 * (1)
    # [BH, BK]
    b_q = tl.load(p_q, mask=p_q_m0[:, None] & p_q_m1[None, :], other=0).to(tl.float32)
    b_k = tl.load(p_k, mask=p_k_m0[:, None] & p_k_m1[None, :], other=0).to(tl.float32)
    b_g = tl.load(p_g, mask=p_g_m0[:, None] & p_g_m1[None, :], other=0).to(tl.float32)
    b_k = b_k * tl.load(p_beta, mask=p_beta_m0, other=0).to(tl.float32)[:, None]

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        p_kj_i0 = tl.arange(0, BH).to(tl.int64) + (i_hg * BH)
        p_kj_m0 = (p_kj_i0 >= 0) & (p_kj_i0 < (H))
        p_kj_i1 = tl.arange(0, BK).to(tl.int64) + (0)
        p_kj_m1 = (p_kj_i1 >= 0) & (p_kj_i1 < (K))
        p_kj = (k + j * H * K) + p_kj_i0[:, None] * (K) + p_kj_i1[None, :] * (1)
        p_gj_i0 = tl.arange(0, BH).to(tl.int64) + (i_hg * BH)
        p_gj_m0 = (p_gj_i0 >= 0) & (p_gj_i0 < (H))
        p_gj_i1 = tl.arange(0, BK).to(tl.int64) + (0)
        p_gj_m1 = (p_gj_i1 >= 0) & (p_gj_i1 < (K))
        p_gj = (g + j * H * K) + p_gj_i0[:, None] * (K) + p_gj_i1[None, :] * (1)
        # [BH, BK]
        b_kj = tl.load(p_kj, mask=p_kj_m0[:, None] & p_kj_m1[None, :], other=0).to(
            tl.float32
        )
        b_gj = tl.load(p_gj, mask=p_gj_m0[:, None] & p_gj_m1[None, :], other=0).to(
            tl.float32
        )

        b_kgj = b_kj * exp2(b_g - b_gj)

        b_kgj = tl.where(m_k[None, :], b_kgj, 0.0)
        # [BH]
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

        tl.store(
            Aqk + i_t * H * BT + (i_hg * BH + o_h) * BT + j % BT,
            b_Aqk.to(Aqk.dtype.element_ty),
            mask=m_h,
        )
        tl.store(
            Akk + i_t * H * BC + (i_hg * BH + o_h) * BC + j - i_ts,
            b_Akk.to(Akk.dtype.element_ty),
            mask=m_h,
        )


def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> None:
    B, T, H, K = q.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    def grid(meta):
        return (B * T, triton.cdiv(H, meta["BH"]))

    chunk_kda_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )
    return Aqk, Akk
