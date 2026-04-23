#!/usr/bin/env python3
"""Standalone chunk-GDN benchmark: old 3-kernel vs new fused+exp2 pipeline.

No rtp-llm dependency — only requires torch + triton.

Production shapes (Qwen3.5-397B TP2, 64K prefill):
  B=1, T=65536, Hg=8 (k-heads/TP), H=32 (v-heads/TP), DK=DV=128, BT=64

Usage:
  python bench_standalone.py              # default 64K
  python bench_standalone.py --T 4096     # shorter sequence
  python bench_standalone.py --stages     # per-stage + fused config sweep
"""

import argparse
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════
# Section 1: Triton utility ops
# ═══════════════════════════════════════════════════════════════════

exp = tl.exp
exp2 = tl.math.exp2


@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float("-inf")))


# ═══════════════════════════════════════════════════════════════════
# Section 2: Shared kernels (cumsum, recompute_w_u, fwd_h, fwd_o)
# ═══════════════════════════════════════════════════════════════════


# ── cumsum ──────────────────────────────────────────────────────
@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
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
    p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum(g, chunk_size, scale=None, cu_seqlens=None):
    B, T, H = g.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    o = torch.empty_like(g, dtype=torch.float)
    chunk_local_cumsum_scalar_kernel[(NT, B * H)](
        s=g,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=False,
        REVERSE=False,
        num_warps=8,
        num_stages=3,
    )
    return o


# ── recompute_w_u ──────────────────────────────────────────────
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_EXP2: tl.constexpr,
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
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    if USE_EXP2:
        b_g = exp2(tl.load(p_g, boundary_check=(0,)))
    else:
        b_g = exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(k, v, beta, A, g_cumsum, use_exp2=False):
    B, T, Hg, K = k.shape
    V = v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]
    NT = triton.cdiv(T, BT)
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=64,
        USE_EXP2=use_exp2,
        num_warps=4,
        num_stages=3,
    )
    return w, u


# ── fwd_h (recurrence) ────────────────────────────────────────
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
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h(
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
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
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

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v2 = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v2, b_v.to(p_v2.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(
                tl.float32
            )
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def fwd_h(k, w, u, g, use_exp2=False):
    B, T, Hg, K = k.shape
    V = u.shape[-1]
    H = u.shape[-2]
    BT = 64
    NT = triton.cdiv(T, BT)
    h = k.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), B * H)

    chunk_gated_delta_rule_fwd_kernel_h[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=None,
        h=h,
        h0=None,
        ht=None,
        cu_seqlens=None,
        chunk_offsets=None,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=16,
        USE_EXP2=use_exp2,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new


# ── fwd_o (output) ─────────────────────────────────────────────
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
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
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
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
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        if USE_EXP2:
            b_o = b_o * exp2(b_g)[:, None]
            b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
        else:
            b_o = b_o * exp(b_g)[:, None]
            b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def fwd_o(q, k, v_new, h, g, scale, use_exp2=False):
    B, T, Hg, K = q.shape
    V = v_new.shape[-1]
    H = v_new.shape[-2]
    BT = min(64, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)
    o = torch.zeros_like(v_new)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v_new,
        h,
        g,
        o,
        None,
        None,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=128,
        USE_EXP2=use_exp2,
        num_warps=1,
        num_stages=2,
    )
    return o


# ═══════════════════════════════════════════════════════════════════
# Section 3: OLD-only kernels (kkt, solve_tril, merge)
# ═══════════════════════════════════════════════════════════════════


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g_cumsum,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
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
    o_t = tl.arange(0, BT)
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))
    if USE_G:
        p_g = tl.make_block_ptr(
            g_cumsum + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_g = tl.load(p_g, boundary_check=(0,))
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])
    b_A *= b_beta[:, None]
    b_A = tl.where(o_t[:, None] > o_t[None, :], b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
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
    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16
    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)
    )
    p_Ai = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = -tl.where(tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :], b_A, 0)
    o_i = tl.arange(0, 16)
    for i in range(1, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        mask = o_i == i
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += o_i[:, None] == o_i[None, :]
    tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
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
    A += (bos * H + i_h) * 64
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 64

    A_21 = tl.load(
        tl.make_block_ptr(
            A, (T, 64), (H * 64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    A_32 = tl.load(
        tl.make_block_ptr(
            A, (T, 64), (H * 64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    A_31 = tl.load(
        tl.make_block_ptr(
            A, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    A_43 = tl.load(
        tl.make_block_ptr(
            A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    A_42 = tl.load(
        tl.make_block_ptr(
            A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    A_41 = tl.load(
        tl.make_block_ptr(
            A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    Ai_11 = tl.load(
        tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 64, 0), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    Ai_22 = tl.load(
        tl.make_block_ptr(
            Ad, (T, 16), (H * 16, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    Ai_33 = tl.load(
        tl.make_block_ptr(
            Ad, (T, 16), (H * 16, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    Ai_44 = tl.load(
        tl.make_block_ptr(
            Ad, (T, 16), (H * 16, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)

    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    Ai_32 = -tl.dot(
        tl.dot(Ai_33, A_32, input_precision="ieee"), Ai_22, input_precision="ieee"
    )
    Ai_43 = -tl.dot(
        tl.dot(Ai_44, A_43, input_precision="ieee"), Ai_33, input_precision="ieee"
    )
    Ai_31 = -tl.dot(
        Ai_33,
        tl.dot(A_31, Ai_11, input_precision="ieee")
        + tl.dot(A_32, Ai_21, input_precision="ieee"),
        input_precision="ieee",
    )
    Ai_42 = -tl.dot(
        Ai_44,
        tl.dot(A_42, Ai_22, input_precision="ieee")
        + tl.dot(A_43, Ai_32, input_precision="ieee"),
        input_precision="ieee",
    )
    Ai_41 = -tl.dot(
        Ai_44,
        tl.dot(A_41, Ai_11, input_precision="ieee")
        + tl.dot(A_42, Ai_21, input_precision="ieee")
        + tl.dot(A_43, Ai_31, input_precision="ieee"),
        input_precision="ieee",
    )

    fill = tl.zeros((16, 16), dtype=tl.bfloat16)
    tl.store(
        tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64, 0), (16, 16), (1, 0)),
        Ai_11.to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 16), (16, 16), (1, 0)
        ),
        Ai_22.to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 32), (16, 16), (1, 0)
        ),
        Ai_33.to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 48), (16, 16), (1, 0)
        ),
        Ai_44.to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
        ),
        Ai_21.to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
        ),
        Ai_31.to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0)
        ),
        Ai_32.to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
        ),
        Ai_41.to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0)
        ),
        Ai_42.to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0)
        ),
        Ai_43.to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 0, 16), (16, 16), (1, 0)
        ),
        fill,
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 0, 32), (16, 16), (1, 0)
        ),
        fill,
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 0, 48), (16, 16), (1, 0)
        ),
        fill,
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 32), (16, 16), (1, 0)
        ),
        fill,
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 48), (16, 16), (1, 0)
        ),
        fill,
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 48), (16, 16), (1, 0)
        ),
        fill,
        boundary_check=(0, 1),
    )


def solve_tril(A, output_dtype):
    B, T, H, BT = A.shape
    Ad = torch.empty(B, T, H, 16, device=A.device, dtype=torch.float32)
    NT16 = triton.cdiv(T, 16)
    solve_tril_16x16_kernel[NT16, B * H](
        A=A,
        Ad=Ad,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        BT=BT,
        num_warps=1,
        num_stages=4,
    )
    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=output_dtype)
    NT = triton.cdiv(T, BT)
    merge_16x16_to_64x64_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        BT=BT,
        num_warps=4,
        num_stages=3,
    )
    return Ai


def kkt_fwd(k, beta, g_cumsum):
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = 64
    NT = triton.cdiv(T, BT)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=torch.float32)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
        A=A,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BK=64,
        num_warps=8,
        num_stages=3,
    )
    return A


# ═══════════════════════════════════════════════════════════════════
# Section 4: NEW-only kernel (fused kkt+solve)
# ═══════════════════════════════════════════════════════════════════


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
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
    if i_t * BT >= T:
        return
    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC
    k += (bos * Hg + i_h // (H // Hg)) * K
    A += (bos * H + i_h) * BT
    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T
    p_b0 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
    p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
    if USE_G:
        p_g0 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
        p_g1 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
        p_g2 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
        p_g3 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
        b_g0 = tl.load(p_g0, boundary_check=(0,)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0,)).to(tl.float32)
        b_g2 = tl.load(p_g2, boundary_check=(0,)).to(tl.float32)
        b_g3 = tl.load(p_g3, boundary_check=(0,)).to(tl.float32)
    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(
            k, (T, K), (Hg * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))
        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(
                k, (T, K), (Hg * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))
            if i_tc2 < T:
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (Hg * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))
                if i_tc3 < T:
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (Hg * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1))
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    if USE_G:
        if USE_EXP2:
            b_A00 *= tl.where(
                m_d & m_tc0[:, None] & m_tc0[None, :],
                exp2(b_g0[:, None] - b_g0[None, :]),
                0.0,
            )
            b_A11 *= tl.where(
                m_d & m_tc1[:, None] & m_tc1[None, :],
                exp2(b_g1[:, None] - b_g1[None, :]),
                0.0,
            )
            b_A22 *= tl.where(
                m_d & m_tc2[:, None] & m_tc2[None, :],
                exp2(b_g2[:, None] - b_g2[None, :]),
                0.0,
            )
            b_A33 *= tl.where(
                m_d & m_tc3[:, None] & m_tc3[None, :],
                exp2(b_g3[:, None] - b_g3[None, :]),
                0.0,
            )
            b_A10 *= tl.where(
                m_tc1[:, None] & m_tc0[None, :],
                exp2(b_g1[:, None] - b_g0[None, :]),
                0.0,
            )
            b_A20 *= tl.where(
                m_tc2[:, None] & m_tc0[None, :],
                exp2(b_g2[:, None] - b_g0[None, :]),
                0.0,
            )
            b_A21 *= tl.where(
                m_tc2[:, None] & m_tc1[None, :],
                exp2(b_g2[:, None] - b_g1[None, :]),
                0.0,
            )
            b_A30 *= tl.where(
                m_tc3[:, None] & m_tc0[None, :],
                exp2(b_g3[:, None] - b_g0[None, :]),
                0.0,
            )
            b_A31 *= tl.where(
                m_tc3[:, None] & m_tc1[None, :],
                exp2(b_g3[:, None] - b_g1[None, :]),
                0.0,
            )
            b_A32 *= tl.where(
                m_tc3[:, None] & m_tc2[None, :],
                exp2(b_g3[:, None] - b_g2[None, :]),
                0.0,
            )
        else:
            b_A00 *= tl.where(
                m_d & m_tc0[:, None] & m_tc0[None, :],
                exp(b_g0[:, None] - b_g0[None, :]),
                0.0,
            )
            b_A11 *= tl.where(
                m_d & m_tc1[:, None] & m_tc1[None, :],
                exp(b_g1[:, None] - b_g1[None, :]),
                0.0,
            )
            b_A22 *= tl.where(
                m_d & m_tc2[:, None] & m_tc2[None, :],
                exp(b_g2[:, None] - b_g2[None, :]),
                0.0,
            )
            b_A33 *= tl.where(
                m_d & m_tc3[:, None] & m_tc3[None, :],
                exp(b_g3[:, None] - b_g3[None, :]),
                0.0,
            )
            b_A10 *= tl.where(
                m_tc1[:, None] & m_tc0[None, :], exp(b_g1[:, None] - b_g0[None, :]), 0.0
            )
            b_A20 *= tl.where(
                m_tc2[:, None] & m_tc0[None, :], exp(b_g2[:, None] - b_g0[None, :]), 0.0
            )
            b_A21 *= tl.where(
                m_tc2[:, None] & m_tc1[None, :], exp(b_g2[:, None] - b_g1[None, :]), 0.0
            )
            b_A30 *= tl.where(
                m_tc3[:, None] & m_tc0[None, :], exp(b_g3[:, None] - b_g0[None, :]), 0.0
            )
            b_A31 *= tl.where(
                m_tc3[:, None] & m_tc1[None, :], exp(b_g3[:, None] - b_g1[None, :]), 0.0
            )
            b_A32 *= tl.where(
                m_tc3[:, None] & m_tc2[None, :], exp(b_g3[:, None] - b_g2[None, :]), 0.0
            )
    else:
        b_A00 = tl.where(m_d, b_A00, 0.0)
        b_A11 = tl.where(m_d, b_A11, 0.0)
        b_A22 = tl.where(m_d, b_A22, 0.0)
        b_A33 = tl.where(m_d, b_A33, 0.0)
    b_A00 *= b_b0[:, None]
    b_A11 *= b_b1[:, None]
    b_A22 *= b_b2[:, None]
    b_A33 *= b_b3[:, None]
    b_A10 *= b_b1[:, None]
    b_A20 *= b_b2[:, None]
    b_A21 *= b_b2[:, None]
    b_A30 *= b_b3[:, None]
    b_A31 *= b_b3[:, None]
    b_A32 *= b_b3[:, None]
    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33
    for i in range(2, min(BC, T - i_tc0)):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.0), 0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.0), 0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.0), 0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.0), 0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a, b_Ai33)
    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision="ieee"), b_Ai00, input_precision="ieee"
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision="ieee"), b_Ai11, input_precision="ieee"
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision="ieee"), b_Ai22, input_precision="ieee"
    )
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision="ieee")
        + tl.dot(b_A21, b_Ai10, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision="ieee")
        + tl.dot(b_A32, b_Ai21, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision="ieee")
        + tl.dot(b_A31, b_Ai10, input_precision="ieee")
        + tl.dot(b_A32, b_Ai20, input_precision="ieee"),
        input_precision="ieee",
    )
    tl.store(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0)),
        b_Ai00.to(A.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)),
        b_Ai10.to(A.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0)),
        b_Ai11.to(A.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)),
        b_Ai20.to(A.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)),
        b_Ai21.to(A.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)),
        b_Ai22.to(A.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)),
        b_Ai30.to(A.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)),
        b_Ai31.to(A.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)),
        b_Ai32.to(A.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)),
        b_Ai33.to(A.dtype.element_ty),
        boundary_check=(0, 1),
    )


def fused_kkt_solve(k, g, beta, use_exp2=True):
    B, T, Hg, K = k.shape
    H = beta.shape[2]
    BT = 64
    NT = triton.cdiv(T, BT)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BC=16,
        BK=64,
        USE_EXP2=use_exp2,
        num_warps=1,
        num_stages=1,
    )
    return A


# ═══════════════════════════════════════════════════════════════════
# Section 5: Pipeline wrappers
# ═══════════════════════════════════════════════════════════════════

RCP_LN2 = 1.0 / 0.6931471805599453


def fwd_h_orig(k, w, u, g, use_exp2=False):
    """Original RTP config: BV=32, warps=4."""
    B, T, Hg, K = k.shape
    V = u.shape[-1]
    H = u.shape[-2]
    BT = 64
    NT = triton.cdiv(T, BT)
    h = k.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), B * H)

    chunk_gated_delta_rule_fwd_kernel_h[grid](
        k=k, v=u, w=w, v_new=v_new, g=g, gk=None,
        h=h, h0=None, ht=None,
        cu_seqlens=None, chunk_offsets=None,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT,
        BV=32,
        USE_EXP2=use_exp2,
        num_warps=4, num_stages=2,
    )
    return h, v_new


def fwd_o_orig(q, k, v_new, h, g, scale, use_exp2=False):
    """Original RTP config: BK=128, BV=64, warps=4."""
    B, T, Hg, K = q.shape
    V = v_new.shape[-1]
    H = v_new.shape[-2]
    BT = min(64, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)
    o = torch.zeros_like(v_new)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q, k, v_new, h, g, o, None, None, scale,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT,
        BK=128, BV=64,
        USE_EXP2=use_exp2,
        num_warps=4, num_stages=2,
    )
    return o


def old_pipeline(q, k, v, g, beta, scale):
    """RTP original: 3-kernel path + original tile configs."""
    g_cum = chunk_local_cumsum(g, chunk_size=64)
    A = kkt_fwd(k, beta, g_cum)
    A = solve_tril(A, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, A, g_cum, use_exp2=False)
    h, v_new = fwd_h_orig(k, w, u, g_cum, use_exp2=False)
    o = fwd_o_orig(q, k, v_new, h, g_cum, scale, use_exp2=False)
    return o


def new_pipeline(q, k, v, g, beta, scale):
    """Optimized: fused kkt+solve + exp2 + MI355X tile tuning."""
    g_cum = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
    A = fused_kkt_solve(k, g_cum, beta, use_exp2=True)
    w, u = recompute_w_u_fwd(k, v, beta, A, g_cum, use_exp2=True)
    h, v_new = fwd_h(k, w, u, g_cum, use_exp2=True)
    o = fwd_o(q, k, v_new, h, g_cum, scale, use_exp2=True)
    return o


# ═══════════════════════════════════════════════════════════════════
# Section 6: Benchmark
# ═══════════════════════════════════════════════════════════════════


def bench_fn(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return sum(times) / len(times), min(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=65536, help="sequence length")
    parser.add_argument("--Hg", type=int, default=8, help="k/q heads per TP")
    parser.add_argument("--H", type=int, default=32, help="v heads per TP")
    parser.add_argument("--DK", type=int, default=128)
    parser.add_argument("--DV", type=int, default=128)
    parser.add_argument(
        "--stages", action="store_true", help="run per-stage + config sweep"
    )
    args = parser.parse_args()

    B, T, Hg, H, DK, DV = 1, args.T, args.Hg, args.H, args.DK, args.DV
    SCALE = DK**-0.5

    print(f"Shape: B={B} T={T} Hg={Hg} H={H} DK={DK} DV={DV}  chunks={T//64}")
    print()

    torch.manual_seed(42)
    q = torch.randn(B, T, Hg, DK, device="cuda", dtype=torch.bfloat16)
    k = F.normalize(
        torch.randn(B, T, Hg, DK, device="cuda", dtype=torch.bfloat16), p=2, dim=-1
    )
    v = torch.randn(B, T, H, DV, device="cuda", dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16))
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16).sigmoid()

    # ── precision ───────────────────────────────────────────────
    print("=== Precision ===")
    torch.cuda.synchronize()
    o_old = old_pipeline(q, k, v, g, beta, SCALE)
    torch.cuda.synchronize()
    o_new = new_pipeline(q, k, v, g, beta, SCALE)
    torch.cuda.synchronize()

    diff = (o_old.float() - o_new.float()).abs()
    rel = diff / (o_old.float().abs() + 1e-8)
    print(f"  max_abs_diff:  {diff.max().item():.6e}")
    print(f"  mean_abs_diff: {diff.mean().item():.6e}")
    print(f"  max_rel_err:   {rel.max().item():.6e}")
    print(f"  mean_rel_err:  {rel.mean().item():.6e}")
    print(f"  → {'PASS' if rel.mean().item() < 1e-2 else 'FAIL'}")
    print()

    # ── end-to-end ──────────────────────────────────────────────
    print("=== End-to-End ===")
    avg_old, min_old = bench_fn(lambda: old_pipeline(q, k, v, g, beta, SCALE))
    avg_new, min_new = bench_fn(lambda: new_pipeline(q, k, v, g, beta, SCALE))
    print(f"  old pipeline:  avg={avg_old:8.0f} us  min={min_old:8.0f} us")
    print(f"  new pipeline:  avg={avg_new:8.0f} us  min={min_new:8.0f} us")
    print(f"  speedup: {avg_old/avg_new:.3f}x  savings: {avg_old-avg_new:.0f} us")
    print()

    # ── per-stage ───────────────────────────────────────────────
    if args.stages:
        print("=== Per-Stage Breakdown ===")
        g_old = chunk_local_cumsum(g, chunk_size=64)
        g_new = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)

        def _b(fn, name):
            avg, _ = bench_fn(fn)
            print(f"  {name:40s} {avg:8.0f} us")
            return avg

        _b(lambda: chunk_local_cumsum(g, chunk_size=64), "OLD cumsum")
        _b(lambda: kkt_fwd(k, beta, g_old), "OLD kkt")
        A_raw = kkt_fwd(k, beta, g_old)
        _b(lambda: solve_tril(A_raw, k.dtype), "OLD solve_tril")
        A_sol = solve_tril(A_raw, k.dtype)
        _b(lambda: recompute_w_u_fwd(k, v, beta, A_sol, g_old), "OLD recompute_w_u")

        print()
        _b(
            lambda: chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2),
            "NEW cumsum (scaled)",
        )
        _b(lambda: fused_kkt_solve(k, g_new, beta), "NEW fused_kkt_solve")
        A_fused = fused_kkt_solve(k, g_new, beta)
        _b(
            lambda: recompute_w_u_fwd(k, v, beta, A_fused, g_new, use_exp2=True),
            "NEW recompute_w_u (exp2)",
        )

        print()
        print("=== Fused Kernel Config Sweep ===")
        NT = triton.cdiv(T, 64)
        A_tmp = torch.zeros(B, T, H, 64, device="cuda", dtype=k.dtype)
        for bk in [32, 64]:
            for nw in [1, 2, 4, 8]:

                def _run(bk=bk, nw=nw):
                    A_tmp.zero_()
                    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
                        k=k,
                        g=g_new,
                        beta=beta,
                        A=A_tmp,
                        cu_seqlens=None,
                        chunk_indices=None,
                        T=T,
                        H=H,
                        Hg=Hg,
                        K=DK,
                        BT=64,
                        BC=16,
                        BK=bk,
                        USE_EXP2=True,
                        num_warps=nw,
                        num_stages=1,
                    )

                _b(_run, f"fused BK={bk} warps={nw}")

    print("\nDone.")


if __name__ == "__main__":
    main()
