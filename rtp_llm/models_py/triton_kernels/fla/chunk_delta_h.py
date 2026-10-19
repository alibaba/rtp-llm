# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.offset import linear_offset_64
from rtp_llm.models_py.triton_kernels.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from rtp_llm.models_py.triton_kernels.fla.op import exp, exp2, safe_exp
from rtp_llm.models_py.triton_kernels.fla.utils import (
    is_amd,
    is_amd_cdna3,
    is_amd_cdna4,
    is_nvidia_hopper,
)

logger = logging.getLogger(__name__)

# TODO(V-first migration): Gluon CDNA4 chunk-h path is disabled until the mfma
# layout is ported from K-first to V-first cache layout to match the Triton
# fallback + chunk_o + fused_recurrent paths. No CDNA4 hardware available for
# validation at the moment. Track progress alongside V-first unification work.
_GLUON_PATH_VFIRST_DONE = False
_gluon_fallback_warned = False

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]


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
# @triton.autotune(
#     configs=[
#         triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4]
#         for num_stages in [2, 3, 4]
#         for BV in [32, 64]
#     ],
#     key=["H", "K", "V", "BT", "USE_G"],
#     use_cuda_graph=use_cuda_graph,
# )
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
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_LOG2: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
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

    # [BV, BK] — V-first layout matches SGL main / FLA state_v_first=True so
    # the on-disk SSM cache (already declared as (..., V, K)) and the chunk
    # pipeline agree on the same byte formula `v*K + k`.
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += linear_offset_64(boh, H * K * V) + linear_offset_64(i_h, K * V)
    v += linear_offset_64(bos, H * V) + linear_offset_64(i_h, V)
    k += linear_offset_64(bos, Hg * K) + linear_offset_64(i_h // (H // Hg), K)
    w += linear_offset_64(bos, H * K) + linear_offset_64(i_h, K)
    if SAVE_NEW_VALUE:
        v_new += linear_offset_64(bos, H * V) + linear_offset_64(i_h, V)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 += linear_offset_64(i_nh, K * V)
    if STORE_FINAL_STATE:
        ht += linear_offset_64(i_nh, K * V)

    # load initial state — V-first view: (V, K) + strides (K, 1)
    if USE_INITIAL_STATE:
        p_h0_1_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
        p_h0_1_m0 = (p_h0_1_i0 >= 0) & (p_h0_1_i0 < (V))
        p_h0_1_i1 = tl.arange(0, 64).to(tl.int64) + (0)
        p_h0_1_m1 = (p_h0_1_i1 >= 0) & (p_h0_1_i1 < (K))
        p_h0_1 = (h0) + p_h0_1_i0[:, None] * (K) + p_h0_1_i1[None, :] * (1)
        b_h1 += tl.load(
            p_h0_1, mask=p_h0_1_m0[:, None] & p_h0_1_m1[None, :], other=0
        ).to(tl.float32)
        if K > 64:
            p_h0_2_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_h0_2_m0 = (p_h0_2_i0 >= 0) & (p_h0_2_i0 < (V))
            p_h0_2_i1 = tl.arange(0, 64).to(tl.int64) + (64)
            p_h0_2_m1 = (p_h0_2_i1 >= 0) & (p_h0_2_i1 < (K))
            p_h0_2 = (h0) + p_h0_2_i0[:, None] * (K) + p_h0_2_i1[None, :] * (1)
            b_h2 += tl.load(
                p_h0_2, mask=p_h0_2_m0[:, None] & p_h0_2_m1[None, :], other=0
            ).to(tl.float32)
        if K > 128:
            p_h0_3_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_h0_3_m0 = (p_h0_3_i0 >= 0) & (p_h0_3_i0 < (V))
            p_h0_3_i1 = tl.arange(0, 64).to(tl.int64) + (128)
            p_h0_3_m1 = (p_h0_3_i1 >= 0) & (p_h0_3_i1 < (K))
            p_h0_3 = (h0) + p_h0_3_i0[:, None] * (K) + p_h0_3_i1[None, :] * (1)
            b_h3 += tl.load(
                p_h0_3, mask=p_h0_3_m0[:, None] & p_h0_3_m1[None, :], other=0
            ).to(tl.float32)
        if K > 192:
            p_h0_4_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_h0_4_m0 = (p_h0_4_i0 >= 0) & (p_h0_4_i0 < (V))
            p_h0_4_i1 = tl.arange(0, 64).to(tl.int64) + (192)
            p_h0_4_m1 = (p_h0_4_i1 >= 0) & (p_h0_4_i1 < (K))
            p_h0_4 = (h0) + p_h0_4_i0[:, None] * (K) + p_h0_4_i1[None, :] * (1)
            b_h4 += tl.load(
                p_h0_4, mask=p_h0_4_m0[:, None] & p_h0_4_m1[None, :], other=0
            ).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        h_chunk = h + linear_offset_64(i_t, stride_h)
        p_h1_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
        p_h1_m0 = (p_h1_i0 >= 0) & (p_h1_i0 < (V))
        p_h1_i1 = tl.arange(0, 64).to(tl.int64) + (0)
        p_h1_m1 = (p_h1_i1 >= 0) & (p_h1_i1 < (K))
        p_h1 = (h_chunk) + p_h1_i0[:, None] * (K) + p_h1_i1[None, :] * (1)
        tl.store(
            p_h1,
            b_h1.to(p_h1.dtype.element_ty),
            mask=p_h1_m0[:, None] & p_h1_m1[None, :],
        )
        if K > 64:
            p_h2_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_h2_m0 = (p_h2_i0 >= 0) & (p_h2_i0 < (V))
            p_h2_i1 = tl.arange(0, 64).to(tl.int64) + (64)
            p_h2_m1 = (p_h2_i1 >= 0) & (p_h2_i1 < (K))
            p_h2 = (h_chunk) + p_h2_i0[:, None] * (K) + p_h2_i1[None, :] * (1)
            tl.store(
                p_h2,
                b_h2.to(p_h2.dtype.element_ty),
                mask=p_h2_m0[:, None] & p_h2_m1[None, :],
            )
        if K > 128:
            p_h3_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_h3_m0 = (p_h3_i0 >= 0) & (p_h3_i0 < (V))
            p_h3_i1 = tl.arange(0, 64).to(tl.int64) + (128)
            p_h3_m1 = (p_h3_i1 >= 0) & (p_h3_i1 < (K))
            p_h3 = (h_chunk) + p_h3_i0[:, None] * (K) + p_h3_i1[None, :] * (1)
            tl.store(
                p_h3,
                b_h3.to(p_h3.dtype.element_ty),
                mask=p_h3_m0[:, None] & p_h3_m1[None, :],
            )
        if K > 192:
            p_h4_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_h4_m0 = (p_h4_i0 >= 0) & (p_h4_i0 < (V))
            p_h4_i1 = tl.arange(0, 64).to(tl.int64) + (192)
            p_h4_m1 = (p_h4_i1 >= 0) & (p_h4_i1 < (K))
            p_h4 = (h_chunk) + p_h4_i0[:, None] * (K) + p_h4_i1[None, :] * (1)
            tl.store(
                p_h4,
                b_h4.to(p_h4.dtype.element_ty),
                mask=p_h4_m0[:, None] & p_h4_m1[None, :],
            )

        # b_v = b_w @ b_h.T  — b_h is now (BV, 64); transpose at dot time.
        p_w_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
        p_w_m0 = (p_w_i0 >= 0) & (p_w_i0 < (T))
        p_w_i1 = tl.arange(0, 64).to(tl.int64) + (0)
        p_w_m1 = (p_w_i1 >= 0) & (p_w_i1 < (K))
        p_w = (w) + p_w_i0[:, None] * (stride_w) + p_w_i1[None, :] * (1)
        b_w = tl.load(p_w, mask=p_w_m0[:, None] & p_w_m1[None, :], other=0)
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_w_m0 = (p_w_i0 >= 0) & (p_w_i0 < (T))
            p_w_i1 = tl.arange(0, 64).to(tl.int64) + (64)
            p_w_m1 = (p_w_i1 >= 0) & (p_w_i1 < (K))
            p_w = (w) + p_w_i0[:, None] * (stride_w) + p_w_i1[None, :] * (1)
            b_w = tl.load(p_w, mask=p_w_m0[:, None] & p_w_m1[None, :], other=0)
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_w_m0 = (p_w_i0 >= 0) & (p_w_i0 < (T))
            p_w_i1 = tl.arange(0, 64).to(tl.int64) + (128)
            p_w_m1 = (p_w_i1 >= 0) & (p_w_i1 < (K))
            p_w = (w) + p_w_i0[:, None] * (stride_w) + p_w_i1[None, :] * (1)
            b_w = tl.load(p_w, mask=p_w_m0[:, None] & p_w_m1[None, :], other=0)
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_w_m0 = (p_w_i0 >= 0) & (p_w_i0 < (T))
            p_w_i1 = tl.arange(0, 64).to(tl.int64) + (192)
            p_w_m1 = (p_w_i1 >= 0) & (p_w_i1 < (K))
            p_w = (w) + p_w_i0[:, None] * (stride_w) + p_w_i1[None, :] * (1)
            b_w = tl.load(p_w, mask=p_w_m0[:, None] & p_w_m1[None, :], other=0)
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
        p_v_m0 = (p_v_i0 >= 0) & (p_v_i0 < (T))
        p_v_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
        p_v_m1 = (p_v_i1 >= 0) & (p_v_i1 < (V))
        p_v = (v) + p_v_i0[:, None] * (stride_v) + p_v_i1[None, :] * (1)
        b_v = tl.load(p_v, mask=p_v_m0[:, None] & p_v_m1[None, :], other=0) - b_v

        if SAVE_NEW_VALUE:
            p_v_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_v_m0 = (p_v_i0 >= 0) & (p_v_i0 < (T))
            p_v_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_v_m1 = (p_v_i1 >= 0) & (p_v_i1 < (V))
            p_v = (v_new) + p_v_i0[:, None] * (stride_v) + p_v_i1[None, :] * (1)
            tl.store(
                p_v,
                b_v.to(p_v.dtype.element_ty),
                mask=p_v_m0[:, None] & p_v_m1[None, :],
            )

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            g_base = g + linear_offset_64(bos, H) + i_h
            g_last = g_base + linear_offset_64(last_idx, H)
            if IS_LOG2:
                # AMD path: g is in log2 domain (RCP_LN2-scaled cumsum upstream).
                # The fp32 promotions, int64 index and m_t mask are robustness
                # tweaks added together with the log2 rewrite for MI355X/MI308X.
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g_last).to(tl.float32)
                p_g_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
                p_g_m0 = (p_g_i0 >= 0) & (p_g_i0 < (T))
                p_g = (g_base) + p_g_i0 * (H)
                b_g = tl.load(p_g, mask=p_g_m0, other=0).to(tl.float32)
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                # NVIDIA path: g is in natural-log domain. Keep the original
                # exp/safe_exp formulation and integer indexing for bit-level
                # parity with the pre-optimization implementation.
                b_g_last = tl.load(g_last)
                p_g_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
                p_g_m0 = (p_g_i0 >= 0) & (p_g_i0 < (T))
                p_g = (g_base) + p_g_i0 * (H)
                b_g = tl.load(p_g, mask=p_g_m0, other=0)
                b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        if USE_GK:
            gk_base = gk + linear_offset_64(bos, H * K) + linear_offset_64(i_h, K)
            gk_last = gk_base + linear_offset_64(last_idx, H * K)
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk_last + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            # b_h is (BV, 64) — broadcast K-dim weight on the trailing axis.
            if IS_LOG2:
                b_h1 *= exp2(b_gk_last1.to(tl.float32))[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk_last + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                if IS_LOG2:
                    b_h2 *= exp2(b_gk_last2.to(tl.float32))[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk_last + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                if IS_LOG2:
                    b_h3 *= exp2(b_gk_last3.to(tl.float32))[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk_last + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                if IS_LOG2:
                    b_h4 *= exp2(b_gk_last4.to(tl.float32))[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        # b_h += (b_k @ b_v).T  — outer-product is (64, BV); transpose to (BV, 64)
        # to match the V-first b_h layout.
        p_k_i0 = tl.arange(0, 64).to(tl.int64) + (0)
        p_k_m0 = (p_k_i0 >= 0) & (p_k_i0 < (K))
        p_k_i1 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
        p_k_m1 = (p_k_i1 >= 0) & (p_k_i1 < (T))
        p_k = (k) + p_k_i0[:, None] * (1) + p_k_i1[None, :] * (stride_k)
        b_k = tl.load(p_k, mask=p_k_m0[:, None] & p_k_m1[None, :], other=0)
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k_i0 = tl.arange(0, 64).to(tl.int64) + (64)
            p_k_m0 = (p_k_i0 >= 0) & (p_k_i0 < (K))
            p_k_i1 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_k_m1 = (p_k_i1 >= 0) & (p_k_i1 < (T))
            p_k = (k) + p_k_i0[:, None] * (1) + p_k_i1[None, :] * (stride_k)
            b_k = tl.load(p_k, mask=p_k_m0[:, None] & p_k_m1[None, :], other=0)
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k_i0 = tl.arange(0, 64).to(tl.int64) + (128)
            p_k_m0 = (p_k_i0 >= 0) & (p_k_i0 < (K))
            p_k_i1 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_k_m1 = (p_k_i1 >= 0) & (p_k_i1 < (T))
            p_k = (k) + p_k_i0[:, None] * (1) + p_k_i1[None, :] * (stride_k)
            b_k = tl.load(p_k, mask=p_k_m0[:, None] & p_k_m1[None, :], other=0)
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k_i0 = tl.arange(0, 64).to(tl.int64) + (192)
            p_k_m0 = (p_k_i0 >= 0) & (p_k_i0 < (K))
            p_k_i1 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_k_m1 = (p_k_i1 >= 0) & (p_k_i1 < (T))
            p_k = (k) + p_k_i0[:, None] * (1) + p_k_i1[None, :] * (stride_k)
            b_k = tl.load(p_k, mask=p_k_m0[:, None] & p_k_m1[None, :], other=0)
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    # epilogue — V-first store
    if STORE_FINAL_STATE:
        p_ht_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
        p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (V))
        p_ht_i1 = tl.arange(0, 64).to(tl.int64) + (0)
        p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (K))
        p_ht = (ht) + p_ht_i0[:, None] * (K) + p_ht_i1[None, :] * (1)
        tl.store(
            p_ht,
            b_h1.to(p_ht.dtype.element_ty),
            mask=p_ht_m0[:, None] & p_ht_m1[None, :],
        )
        if K > 64:
            p_ht_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (V))
            p_ht_i1 = tl.arange(0, 64).to(tl.int64) + (64)
            p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (K))
            p_ht = (ht) + p_ht_i0[:, None] * (K) + p_ht_i1[None, :] * (1)
            tl.store(
                p_ht,
                b_h2.to(p_ht.dtype.element_ty),
                mask=p_ht_m0[:, None] & p_ht_m1[None, :],
            )
        if K > 128:
            p_ht_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (V))
            p_ht_i1 = tl.arange(0, 64).to(tl.int64) + (128)
            p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (K))
            p_ht = (ht) + p_ht_i0[:, None] * (K) + p_ht_i1[None, :] * (1)
            tl.store(
                p_ht,
                b_h3.to(p_ht.dtype.element_ty),
                mask=p_ht_m0[:, None] & p_ht_m1[None, :],
            )
        if K > 192:
            p_ht_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (V))
            p_ht_i1 = tl.arange(0, 64).to(tl.int64) + (192)
            p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (K))
            p_ht = (ht) + p_ht_i0[:, None] * (K) + p_ht_i1[None, :] * (1)
            tl.store(
                p_ht,
                b_h4.to(p_ht.dtype.element_ty),
                mask=p_ht_m0[:, None] & p_ht_m1[None, :],
            )


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
# @triton.autotune(
#     configs=[
#         triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4]
#         for num_stages in [2, 3, 4]
#         for BV in [32, 64]
#     ],
#     key=["H", "K", "V", "BT", "USE_G"],
#     use_cuda_graph=use_cuda_graph,
# )
@triton.jit(do_not_specialize=["T"])
def _chunk_gated_delta_rule_selected_kernel(
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
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_LOG2: tl.constexpr,
    state_positions=None,
    selected_states=None,
    NUM_STATE_POSITIONS: tl.constexpr = 0,
    SELECT_STATES: tl.constexpr = False,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
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

    if SELECT_STATES:
        i_s = tl.program_id(2)
        # Validate in the original integer width before narrowing: large int64
        # positions must not wrap into a valid prefix or an out-of-bounds load.
        prefix = tl.load(state_positions + i_n * NUM_STATE_POSITIONS + i_s)
        valid_prefix = (prefix >= 0) & (prefix <= T)
        prefix = tl.where(valid_prefix, prefix, 0).to(tl.int32)
        # At chunk boundaries replay the preceding complete chunk. This also
        # covers the sequence end when no following chunk-start state exists.
        chunk_start = (tl.maximum(prefix, 1) - 1) // BT * BT
        boh += chunk_start // BT
        bos += chunk_start
        T = prefix - chunk_start
        NT = tl.cdiv(T, BT)

    # [BV, BK] — V-first layout matches SGL main / FLA state_v_first=True so
    # the on-disk SSM cache (already declared as (..., V, K)) and the chunk
    # pipeline agree on the same byte formula `v*K + k`.
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += linear_offset_64(boh, H * K * V) + linear_offset_64(i_h, K * V)
    v += linear_offset_64(bos, H * V) + linear_offset_64(i_h, V)
    k += linear_offset_64(bos, Hg * K) + linear_offset_64(i_h // (H // Hg), K)
    w += linear_offset_64(bos, H * K) + linear_offset_64(i_h, K)
    if SAVE_NEW_VALUE:
        v_new += linear_offset_64(bos, H * V) + linear_offset_64(i_h, V)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 += linear_offset_64(i_nh, K * V)
    if STORE_FINAL_STATE:
        if SELECT_STATES:
            ht = (
                selected_states
                + linear_offset_64(i_n * NUM_STATE_POSITIONS + i_s, H * K * V)
                + linear_offset_64(i_h, K * V)
            )
        else:
            ht += linear_offset_64(i_nh, K * V)

    # load initial state — V-first view: (V, K) + strides (K, 1)
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    if SELECT_STATES:
        # h is FP32, so restoring a chunk start preserves the accumulator used
        # by the original recurrence. Do not reconstruct from rounded v_new.
        if prefix > 0:
            p_h1 = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            b_h1 = tl.load(p_h1, boundary_check=(0, 1)).to(tl.float32)
            if K > 64:
                p_h2 = tl.make_block_ptr(
                    h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
                b_h2 = tl.load(p_h2, boundary_check=(0, 1)).to(tl.float32)
            if K > 128:
                p_h3 = tl.make_block_ptr(
                    h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
                b_h3 = tl.load(p_h3, boundary_check=(0, 1)).to(tl.float32)
            if K > 192:
                p_h4 = tl.make_block_ptr(
                    h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
                b_h4 = tl.load(p_h4, boundary_check=(0, 1)).to(tl.float32)
        elif not valid_prefix:
            b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
            if K > 64:
                b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
            if K > 128:
                b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
            if K > 192:
                b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # main recurrence
    for i_t in range(NT):
        if not SELECT_STATES:
            h_chunk = h + linear_offset_64(i_t, stride_h)
            p_h1 = tl.make_block_ptr(
                h_chunk, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(
                    h_chunk, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(
                    h_chunk, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(
                    h_chunk, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        # b_v = b_w @ b_h.T  — b_h is now (BV, 64); transpose at dot time.
        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            g_base = g + linear_offset_64(bos, H) + i_h
            g_last = g_base + linear_offset_64(last_idx, H)
            if IS_LOG2:
                # AMD path: g is in log2 domain (RCP_LN2-scaled cumsum upstream).
                # The fp32 promotions, int64 index and m_t mask are robustness
                # tweaks added together with the log2 rewrite for MI355X/MI308X.
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g_last).to(tl.float32)
                p_g = tl.make_block_ptr(g_base, (T,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                # NVIDIA path: g is in natural-log domain. Keep the original
                # exp/safe_exp formulation and integer indexing for bit-level
                # parity with the pre-optimization implementation.
                b_g_last = tl.load(g_last)
                p_g = tl.make_block_ptr(g_base, (T,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))
                b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        if USE_GK:
            gk_base = gk + linear_offset_64(bos, H * K) + linear_offset_64(i_h, K)
            gk_last = gk_base + linear_offset_64(last_idx, H * K)
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk_last + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            # b_h is (BV, 64) — broadcast K-dim weight on the trailing axis.
            if IS_LOG2:
                b_h1 *= exp2(b_gk_last1.to(tl.float32))[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk_last + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                if IS_LOG2:
                    b_h2 *= exp2(b_gk_last2.to(tl.float32))[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk_last + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                if IS_LOG2:
                    b_h3 *= exp2(b_gk_last3.to(tl.float32))[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk_last + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                if IS_LOG2:
                    b_h4 *= exp2(b_gk_last4.to(tl.float32))[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        # b_h += (b_k @ b_v).T  — outer-product is (64, BV); transpose to (BV, 64)
        # to match the V-first b_h layout.
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    # epilogue — V-first store
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def _validate_selected_state_output_alias(
    selected_states: Optional[torch.Tensor],
    **named_tensors: Optional[torch.Tensor],
) -> None:
    """Reject shared output/input storage using host metadata only.

    Check storage ranges conservatively so views and noncontiguous public API
    inputs are covered before any input-contiguity copies are made.
    """
    if selected_states is None or selected_states.numel() == 0:
        return
    output_storage = selected_states.untyped_storage()
    output_start = output_storage.data_ptr()
    output_end = output_start + output_storage.nbytes()
    for name, tensor in named_tensors.items():
        if (
            tensor is None
            or tensor.numel() == 0
            or tensor.device != selected_states.device
        ):
            continue
        input_storage = tensor.untyped_storage()
        input_start = input_storage.data_ptr()
        input_end = input_start + input_storage.nbytes()
        if output_start < input_end and input_start < output_end:
            raise ValueError(f"selected_states must not share storage with {name}")


def _validate_selected_states(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor],
    gk: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    state_dtype: Optional[torch.dtype],
    chunk_size: int,
    state_positions: torch.Tensor,
    selected_states: torch.Tensor,
) -> None:
    """Check snapshot support using tensor metadata only (no device sync)."""
    if k.device.type != "cuda" or is_amd:
        raise NotImplementedError("selected GDN states currently require NVIDIA CUDA")
    if gk is not None:
        raise NotImplementedError("selected GDN states do not support gk")
    if state_dtype not in (None, torch.float32):
        raise ValueError("selected GDN states require FP32 chunk states")
    if chunk_size != 64:
        raise ValueError("selected GDN states require chunk_size=64")
    if k.ndim != 4 or w.ndim != 4 or u.ndim != 4:
        raise ValueError("k, w and u must have rank 4")
    B, T, Hg, K = k.shape
    H, V = u.shape[-2:]
    if min(B, Hg, H, K, V) <= 0 or K > 256 or H % Hg:
        raise ValueError("invalid GDN batch or head dimensions")
    if u.shape[:2] != (B, T) or w.shape != (B, T, H, K):
        raise ValueError("k, w and u have incompatible shapes")
    if k.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("k must have floating-point dtype supported by GDN")
    if w.dtype != k.dtype or u.dtype != k.dtype:
        raise ValueError("k, w and u must have the same dtype")
    if cu_seqlens is None:
        N = B
    else:
        if (
            B != 1
            or cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("cu_seqlens must be an integer vector for a packed batch")
        N = cu_seqlens.numel() - 1
    if state_positions.ndim != 2 or state_positions.shape[0] != N:
        raise ValueError("state_positions must have shape [N, S]")
    if state_positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("state_positions must have int32 or int64 dtype")
    S = state_positions.shape[1]
    if S > 65535:
        raise ValueError("at most 65535 state positions per sequence are supported")
    if selected_states.shape != (N, S, H, V, K):
        raise ValueError("selected_states must have shape [N, S, H, V, K]")
    if selected_states.dtype != torch.float32:
        raise ValueError("selected_states must have FP32 dtype")
    if g is not None and (g.shape != (B, T, H) or not g.is_floating_point()):
        raise ValueError("g must be a floating-point scalar gate of shape [B, T, H]")
    if initial_state is not None and (
        initial_state.shape != (N, H, V, K)
        or initial_state.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError("initial_state must have floating-point shape [N, H, V, K]")
    for name, tensor in (
        ("k", k),
        ("w", w),
        ("u", u),
        ("g", g),
        ("initial_state", initial_state),
        ("cu_seqlens", cu_seqlens),
        ("state_positions", state_positions),
        ("selected_states", selected_states),
    ):
        if tensor is not None:
            if tensor.device != k.device:
                raise ValueError(f"{name} must be on the same device as k")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
    _validate_selected_state_output_alias(
        selected_states,
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        state_positions=state_positions,
    )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    state_dtype: Optional[torch.dtype] = None,
    state_positions: Optional[torch.Tensor] = None,
    selected_states: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Args:
        state_dtype: dtype of the per-chunk hidden state buffer ``h``.
            If explicitly provided by the caller, it is honored on both the
            generic Triton fallback path and the Gluon/AMD CDNA4 path.
            If ``None``, defaults to fp32 on all backends to preserve
            precision (the kernel accumulates in fp32 internally).
        state_positions: Optional contiguous int32/int64 CUDA tensor [N, S]
            of sequence-local processed prefix lengths. Position 0 selects the
            initial state; position p selects the state after tokens [0, p).
            Unsorted positions and duplicates are supported. Negative positions
            (including -1 padding) and positions beyond the sequence length
            produce zero states without reading outside the input.
        selected_states: Caller-owned contiguous FP32 CUDA tensor [N, S, H, V, K].
            Must be supplied together with state_positions. The existing return
            tuple is unchanged. Snapshot extraction requires FP32 chunk states,
            NVIDIA CUDA, chunk_size=64 and scalar g (gk is not supported).
            Tensor metadata is checked without copying position values to CPU.
            The input/output buffers must not alias.

    With no requested positions, the original kernel is the only launch. With
    positions, a second specialization replays at most one chunk per position
    using the original recurrence and its exact FP32 chunk-start accumulator.
    """
    if (state_positions is None) != (selected_states is None):
        raise ValueError(
            "state_positions and selected_states must be provided together"
        )
    if state_positions is not None:
        _validate_selected_states(
            k,
            w,
            u,
            g,
            gk,
            initial_state,
            cu_seqlens,
            state_dtype,
            chunk_size,
            state_positions,
            selected_states,
        )
    # Gluon dispatch for AMD CDNA4 (MI355X): ~10-18% faster for TP2 H≤32 T≤64K
    try:
        from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h_gluon import (
            _is_gluon_beneficial,
            chunk_gated_delta_rule_fwd_h_gluon,
        )

        _gluon_available = True
    except ImportError:
        _gluon_available = False

    if _gluon_available:
        H = u.shape[-2]
        T = k.shape[1]
        K = k.shape[-1]
        V = u.shape[-1]
        global _gluon_fallback_warned
        if not _GLUON_PATH_VFIRST_DONE and not _gluon_fallback_warned:
            logger.warning(
                "Gluon CDNA4 chunk-h disabled pending V-first layout port; "
                "MI355X falling back to Triton path."
            )
            _gluon_fallback_warned = True
        if (
            _GLUON_PATH_VFIRST_DONE
            and state_positions is None
            and gk is None
            and _is_gluon_beneficial(
                H, T, K, k_dtype=k.dtype, v_dim=V, cu_seqlens=cu_seqlens
            )
        ):
            return chunk_gated_delta_rule_fwd_h_gluon(
                k=k,
                w=w,
                u=u,
                g=g,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
                save_new_value=save_new_value,
                cu_seqlens=cu_seqlens,
                state_dtype=state_dtype,
            )
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
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

    # Generic Triton fallback: the kernel accumulates ``b_h*`` in fp32 and
    # only casts on store. Defaulting to fp32 here preserves precision and
    # matches upstream fla. Callers that knowingly want a narrower buffer
    # (e.g. to save memory) must opt in by passing ``state_dtype`` explicitly,
    # rather than having it silently inferred from ``k.dtype``.
    if state_dtype is None:
        h_dtype = torch.float32
    else:
        h_dtype = state_dtype
    # V-first cache layout, matches the on-disk ssm_states declaration
    # (B, blocks, H, V, K) and SGL main / FLA state_v_first=True.
    h = k.new_empty(B, NT, H, V, K, dtype=h_dtype)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

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
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=64 if is_amd_cdna3 else (16 if is_amd_cdna4 else 32),
        IS_LOG2=is_amd,
        num_warps=4,
        num_stages=2,
    )
    if state_positions is not None and state_positions.shape[1] > 0:
        _chunk_gated_delta_rule_selected_kernel[
            (triton.cdiv(V, 32), N * H, state_positions.shape[1])
        ](
            k=k,
            v=u,
            w=w,
            v_new=None,
            g=g,
            gk=None,
            h=h,
            h0=initial_state,
            ht=selected_states,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BV=32,
            IS_LOG2=False,
            state_positions=state_positions,
            selected_states=selected_states,
            NUM_STATE_POSITIONS=state_positions.shape[1],
            SELECT_STATES=True,
            num_warps=4,
            num_stages=2,
        )
    return h, v_new, final_state
