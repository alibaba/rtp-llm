# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Adapted for rtp-llm: forward-only, supports USE_EXP2 for log2-space gates.

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.autotune_cache import (
    autotune_cache_kwargs,
    cuda_cached_autotune,
)
from rtp_llm.models_py.triton_kernels.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from rtp_llm.models_py.triton_kernels.fla.op import exp, exp2


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@cuda_cached_autotune(
    configs=[
        # Keep the K3 state update numerically aligned with the original
        # FLA 0.5.1 implementation used by the sealed Dummy golden.  On
        # SM103, autotuning may otherwise select the 3-stage variant.  Its
        # different tensor-core scheduling changes the FP32 recurrent state
        # by a few ulps and can eventually flip near-tied MoE routes.
        triton.Config({"BV": 32}, num_warps=4, num_stages=2)
    ],
    key=[
        "H",
        "HV",
        "K",
        "V",
        "BT",
        "USE_EXP2",
        "TRANSPOSE_STATE",
    ],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K * V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
        else:
            p_h0_1 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_h0_2 = tl.make_block_ptr(
                    h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_h0_2 = tl.make_block_ptr(
                    h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_3 = tl.make_block_ptr(
                    h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_h0_3 = tl.make_block_ptr(
                    h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_h0_4 = tl.make_block_ptr(
                    h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_h0_4 = tl.make_block_ptr(
                    h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_h1 = tl.make_block_ptr(
                h + i_t_int64 * HV * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, 0),
                (BV, 64),
                (1, 0),
            )
        else:
            p_h1 = tl.make_block_ptr(
                h + i_t_int64 * HV * K * V,
                (K, V),
                (V, 1),
                (0, i_v * BV),
                (64, BV),
                (1, 0),
            )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_h2 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 64),
                    (BV, 64),
                    (1, 0),
                )
            else:
                p_h2 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (64, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_h3 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 128),
                    (BV, 64),
                    (1, 0),
                )
            else:
                p_h3 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (128, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_h4 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 192),
                    (BV, 64),
                    (1, 0),
                )
            else:
                p_h4 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (192, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(
                tl.float32
            )
            p_g = tl.make_block_ptr(
                g + (bos * HV + i_h).to(tl.int64), (T,), (HV,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * HV * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            ).to(tl.float32)
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_h1 *= exp(b_gk_last1)[None, :]
            else:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                ).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_h2 *= exp(b_gk_last2)[None, :]
                else:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[:, None]
                    else:
                        b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                ).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_h3 *= exp(b_gk_last3)[None, :]
                else:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[:, None]
                    else:
                        b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                ).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_h4 *= exp(b_gk_last4)[None, :]
                else:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[:, None]
                    else:
                        b_h4 *= exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
        else:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(
                    ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(
                    ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(
                    ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if transpose_state_layout:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = (
            k.new_zeros(N, HV, V, K, dtype=torch.float32)
            if output_final_state
            else None
        )
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = (
            k.new_zeros(N, HV, K, V, dtype=torch.float32)
            if output_final_state
            else None
        )

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * HV)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return h, v_new, final_state


def chunk_gated_delta_rule_fwd_h_cublas(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Deterministic K3 chunk-state recurrence using CUDA BLAS GEMMs.

    The original FLA Triton kernel accumulates a BF16 tensor-core product into
    a non-zero FP32 recurrent state.  Triton 3.6 and 3.7 lower that operation
    with different reduction groupings, which changes a few BF16 KDA outputs
    and can cascade into different MoE expert IDs.  CUDA BLAS' full-K
    ``baddbmm(..., out_dtype=float32)`` reproduces the sealed K3 Dummy state
    and output bit-for-bit on the supported CUDA13 stack.

    This is the correctness backend.  The Triton implementation remains
    available for profiling and can replace it once a compiler-version-stable
    kernel has component parity.  The Python chunk loop is deliberately
    simple; performance optimization is out of scope for the accuracy phase.
    """

    del chunk_indices
    if g is not None or gk is None:
        raise ValueError(
            "the K3 CUBLAS state backend requires gk and does not support g"
        )
    if not use_exp2:
        raise ValueError("the K3 CUBLAS state backend requires exp2 gates")
    if not k.is_cuda or not w.is_cuda or not u.is_cuda or not gk.is_cuda:
        raise ValueError("the K3 CUBLAS state backend requires CUDA tensors")
    if k.dtype != w.dtype or k.dtype != u.dtype:
        raise ValueError("k, w and u must have the same dtype")

    batch, token_count, key_heads, key_dim = k.shape
    value_heads, value_dim = u.shape[2], u.shape[3]
    if value_heads % key_heads != 0:
        raise ValueError(
            f"value heads {value_heads} must be divisible by key heads {key_heads}"
        )
    if w.shape != (batch, token_count, value_heads, key_dim):
        raise ValueError(
            f"unexpected w shape {tuple(w.shape)} for k/u shapes "
            f"{tuple(k.shape)}/{tuple(u.shape)}"
        )
    if gk.shape != (batch, token_count, value_heads, key_dim):
        raise ValueError(
            f"unexpected gk shape {tuple(gk.shape)} for value heads {value_heads}"
        )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be FP32")

    if cu_seqlens is None:
        sequence_ranges = [
            (batch_index, 0, token_count) for batch_index in range(batch)
        ]
        chunks_per_sequence = (token_count + chunk_size - 1) // chunk_size
        total_chunks = chunks_per_sequence
        h = k.new_empty(
            batch,
            total_chunks,
            value_heads,
            value_dim if transpose_state_layout else key_dim,
            key_dim if transpose_state_layout else value_dim,
        )
    else:
        if batch != 1:
            raise ValueError("cu_seqlens requires flattened inputs with batch size 1")
        boundaries = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != token_count:
            raise ValueError(
                f"invalid cu_seqlens boundaries {boundaries} for T={token_count}"
            )
        sequence_ranges = [
            (0, boundaries[index], boundaries[index + 1])
            for index in range(len(boundaries) - 1)
        ]
        total_chunks = sum(
            (stop - start + chunk_size - 1) // chunk_size
            for _, start, stop in sequence_ranges
        )
        h = k.new_empty(
            1,
            total_chunks,
            value_heads,
            value_dim if transpose_state_layout else key_dim,
            key_dim if transpose_state_layout else value_dim,
        )

    sequence_count = len(sequence_ranges)
    final_state = (
        torch.empty(
            sequence_count,
            value_heads,
            value_dim if transpose_state_layout else key_dim,
            key_dim if transpose_state_layout else value_dim,
            dtype=torch.float32,
            device=k.device,
        )
        if output_final_state
        else None
    )
    v_new = torch.empty_like(u) if save_new_value else None
    key_head_repeat = value_heads // key_heads
    global_chunk_index = 0

    for sequence_index, (batch_index, seq_start, seq_stop) in enumerate(
        sequence_ranges
    ):
        if initial_state is None:
            state = torch.zeros(
                value_heads,
                key_dim,
                value_dim,
                dtype=torch.float32,
                device=k.device,
            )
        else:
            state = initial_state[sequence_index]
            if transpose_state_layout:
                state = state.transpose(-1, -2)
            state = state.contiguous()

        local_chunk_index = 0
        for start in range(seq_start, seq_stop, chunk_size):
            stop = min(start + chunk_size, seq_stop)
            length = stop - start
            stored_state = state.to(k.dtype)
            target_chunk = (
                global_chunk_index if cu_seqlens is not None else local_chunk_index
            )
            if transpose_state_layout:
                h[batch_index, target_chunk] = stored_state.transpose(-1, -2)
            else:
                h[batch_index, target_chunk] = stored_state

            w_chunk = w[batch_index, start:stop].permute(1, 0, 2).contiguous()
            u_chunk = u[batch_index, start:stop].permute(1, 0, 2).contiguous()
            new_value = torch.baddbmm(
                u_chunk.float(),
                w_chunk,
                stored_state,
                beta=1,
                alpha=-1,
                out_dtype=torch.float32,
            )
            new_value_bf16 = new_value.to(k.dtype)
            if v_new is not None:
                v_new[batch_index, start:stop] = new_value_bf16.permute(1, 0, 2)

            decay = torch.exp2(gk[batch_index, stop - 1].float()).unsqueeze(-1)
            decayed_state = state * decay
            k_chunk = k[batch_index, start:stop]
            if key_head_repeat != 1:
                k_chunk = k_chunk.repeat_interleave(key_head_repeat, dim=1)
            k_chunk = k_chunk.permute(1, 2, 0).contiguous()
            state = decayed_state + torch.bmm(
                k_chunk,
                new_value_bf16,
                out_dtype=torch.float32,
            )

            local_chunk_index += 1
            if cu_seqlens is not None:
                global_chunk_index += 1

        if final_state is not None:
            if transpose_state_layout:
                final_state[sequence_index] = state.transpose(-1, -2)
            else:
                final_state[sequence_index] = state

    return h, v_new, final_state
