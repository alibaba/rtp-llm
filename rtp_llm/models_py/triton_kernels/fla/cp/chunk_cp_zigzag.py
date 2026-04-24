# -*- coding: utf-8 -*-
# Context Parallelism — Zigzag variant
#
# Each rank computes on its own zigzag tokens directly, no QKV all-gather needed.
# Each rank has 2 segments (front half + back half of zigzag).
# Total 2*cp_size segments form a causal chain.
#
# Phase 0: Conv1d with P2P exchange of tail tokens (kernel_size-1 tokens)
# Phase 1: Each rank computes (b, M) for its 2 segments locally
# Phase 2: All-gather (b, M) from all ranks, reorder to causal order
# Phase 3: cp_merge to compute h0_true for each segment
# Phase 4: Rerun Step5+Step6 with correct h0
#
# Pure-Python helpers (exchange_conv_context / prepend_conv_context /
# strip_conv_context / zigzag_causal_order / causal_positions /
# build_segment_cu_seqlens) live in `cp.utils`.

from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_o import chunk_fwd_o
from rtp_llm.models_py.triton_kernels.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.cp.utils import (
    build_segment_cu_seqlens,
    causal_positions,
    zigzag_causal_order,
)
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd
from rtp_llm.models_py.triton_kernels.fla.op import exp
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd

# ---------------------------------------------------------------------------
# cp_merge_kernel: walk the causal chain of (M, b) affines applied to h0.
# Reads from `ag_hm` indirectly via `causal_order_ptr` so we never have to
# materialise a reordered copy of the gathered buffer.
# ---------------------------------------------------------------------------


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_cp_compute_br_kernel(
    k,
    v,
    w,
    g,
    ht,
    cu_seqlens,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_n * T
    NT = tl.cdiv(T, BT)

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    for i_t in range(NT):
        # b_v = v[t] - w[t] @ h
        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_v = tl.dot(tl.load(p_w, boundary_check=(0, 1)), b_h1.to(w.dtype.element_ty))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_v += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_h2.to(w.dtype.element_ty)
            )
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_v += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_h3.to(w.dtype.element_ty)
            )
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_v += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_h4.to(w.dtype.element_ty)
            )

        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            # Mask positions >= T inside this chunk: boundary-checked g loads
            # return 0, but multiplying b_v by exp(b_g_last - 0) instead of 0
            # leaks fp32 garbage into the b_h accumulator (fails on short
            # sequences where T < BT).
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            # Use plain `exp` (NOT safe_exp): zigzag-CP's `chunk_local_cumsum`
            # path can produce ULP-level differences vs the legacy single-card
            # cumsum. safe_exp's `<= 0` branch amplifies that ULP noise into
            # full magnitude divergence at chunk boundaries; plain exp keeps the
            # decay continuous and stays within bf16 ULP of legacy.
            # m_t mask is currently a no-op (sub-seq lens are chunk-multiples),
            # kept for future cases where chunks may have intra-chunk padding.
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
        b_h1 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_h2 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_h3 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_h4 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v)

    # store only final state
    ht_base = ht + i_nh * K * V
    tl.store(
        tl.make_block_ptr(ht_base, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)),
        b_h1.to(ht.dtype.element_ty),
        boundary_check=(0, 1),
    )
    if K > 64:
        tl.store(
            tl.make_block_ptr(
                ht_base, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            ),
            b_h2.to(ht.dtype.element_ty),
            boundary_check=(0, 1),
        )
    if K > 128:
        tl.store(
            tl.make_block_ptr(
                ht_base, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            ),
            b_h3.to(ht.dtype.element_ty),
            boundary_check=(0, 1),
        )
    if K > 192:
        tl.store(
            tl.make_block_ptr(
                ht_base, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            ),
            b_h4.to(ht.dtype.element_ty),
            boundary_check=(0, 1),
        )


def compute_br(k, w, u, g, cu_seqlens=None):
    """Run recurrence with h0=0, return only h_final. No h[t] or v_new stored."""
    B, T, Hg, K = k.shape
    H = w.shape[2]
    V = u.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    ht = torch.empty(N, H, K, V, dtype=torch.float32, device=k.device)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_cp_compute_br_kernel[grid](
        k=k,
        v=u,
        w=w,
        g=g,
        ht=ht,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=64,
        BV=32,
        num_warps=4,
        num_stages=2,
    )
    return ht


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_cp_compute_M_total_kernel(
    k,
    w,
    g,
    M_total,
    cu_seqlens,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_col, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_n * T
    NT = tl.cdiv(T, BT)

    # δh = I[:, i_col*BK : (i_col+1)*BK]
    o_row = tl.arange(0, 64)
    o_col = tl.arange(0, BK) + i_col * BK
    b_dh1 = (o_row[:, None] == o_col[None, :]).to(tl.float32)
    if K > 64:
        b_dh2 = ((o_row + 64)[:, None] == o_col[None, :]).to(tl.float32)
    if K > 128:
        b_dh3 = ((o_row + 128)[:, None] == o_col[None, :]).to(tl.float32)
    if K > 192:
        b_dh4 = ((o_row + 192)[:, None] == o_col[None, :]).to(tl.float32)

    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_k = Hg * K
    stride_w = H * K

    for i_t in range(NT):
        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_wdh = tl.dot(
            tl.load(p_w, boundary_check=(0, 1)), b_dh1.to(w.dtype.element_ty)
        )
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_wdh += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_dh2.to(w.dtype.element_ty)
            )
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_wdh += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_dh3.to(w.dtype.element_ty)
            )
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_wdh += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_dh4.to(w.dtype.element_ty)
            )

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            # Same mask reasoning as compute_br_kernel.
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            # See compute_br_kernel comment: plain `exp` (no safe_exp) keeps
            # decay continuous so ULP noise doesn't get amplified by `<= 0`
            # branch.
            b_wdh = b_wdh * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last_exp = exp(b_g_last)
            b_dh1 = b_dh1 * b_g_last_exp
            if K > 64:
                b_dh2 = b_dh2 * b_g_last_exp
            if K > 128:
                b_dh3 = b_dh3 * b_g_last_exp
            if K > 192:
                b_dh4 = b_dh4 * b_g_last_exp

        b_wdh = b_wdh.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_dh1 -= tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_wdh)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_dh2 -= tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_wdh)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_dh3 -= tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_wdh)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_dh4 -= tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_wdh)

    M_base = M_total + i_nh * K * K
    tl.store(
        tl.make_block_ptr(M_base, (K, K), (K, 1), (0, i_col * BK), (64, BK), (1, 0)),
        b_dh1.to(M_total.dtype.element_ty),
        boundary_check=(0, 1),
    )
    if K > 64:
        tl.store(
            tl.make_block_ptr(
                M_base, (K, K), (K, 1), (64, i_col * BK), (64, BK), (1, 0)
            ),
            b_dh2.to(M_total.dtype.element_ty),
            boundary_check=(0, 1),
        )
    if K > 128:
        tl.store(
            tl.make_block_ptr(
                M_base, (K, K), (K, 1), (128, i_col * BK), (64, BK), (1, 0)
            ),
            b_dh3.to(M_total.dtype.element_ty),
            boundary_check=(0, 1),
        )
    if K > 192:
        tl.store(
            tl.make_block_ptr(
                M_base, (K, K), (K, 1), (192, i_col * BK), (64, BK), (1, 0)
            ),
            b_dh4.to(M_total.dtype.element_ty),
            boundary_check=(0, 1),
        )


def compute_M_total(k, w, g, cu_seqlens=None):
    B, T, Hg, K = k.shape
    H = w.shape[2]
    BK = min(64, K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    M_total = torch.empty(N, H, K, K, dtype=torch.float32, device=k.device)

    def grid(meta):
        return (triton.cdiv(K, meta["BK"]), N * H)

    chunk_cp_compute_M_total_kernel[grid](
        k=k,
        w=w,
        g=g,
        M_total=M_total,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=64,
        BK=BK,
        num_warps=4,
        num_stages=2,
    )
    return M_total


@triton.jit(do_not_specialize=["num_ranks"])
def cp_merge_kernel(
    h_out,
    ag_hm,  # all-gather raw layout (NCCL order, NOT reordered)
    h0,
    causal_order_ptr,  # [2*cp_size] int tensor: causal_pos -> ag_hm row
    num_ranks,  # number of affines to apply (causal positions 0..num_ranks-1)
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    HAS_H0: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H

    stride_rank = N * H * K * (V + K)
    ag_base = (i_n * H + i_h) * K * (V + K)

    if HAS_H0:
        p_h0 = tl.make_block_ptr(
            h0 + (i_n * H + i_h) * K * V,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)

    for r in range(num_ranks):
        # Look up actual ag_hm row for causal position r — avoids materializing
        # a reordered copy of the gathered buffer (which is huge for large N).
        r_actual = tl.load(causal_order_ptr + r).to(tl.int64)
        base = r_actual * stride_rank + ag_base
        p_b = tl.make_block_ptr(
            ag_hm + base, (K, V), (V + K, 1), (0, i_v * BV), (BK, BV), (1, 0)
        )
        p_m = tl.make_block_ptr(
            ag_hm + base + V, (K, K), (V + K, 1), (0, 0), (BK, BK), (1, 0)
        )
        b_b = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)
        b_m = tl.load(p_m, boundary_check=(0, 1)).to(tl.float32)
        b_h = tl.dot(b_m, b_h) + b_b

    p_out = tl.make_block_ptr(
        h_out + (i_n * H + i_h) * K * V, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
    )
    tl.store(p_out, b_h.to(p_out.dtype.element_ty), boundary_check=(0, 1))


def cp_merge(ag_hm, h0, num_ranks, N, H, K, V, causal_order):
    """Triton kernel merge: iterate `num_ranks` affine transforms on h0,
    reading affines from `ag_hm` in causal order via `causal_order` lookup.

    Args:
        ag_hm: all-gather raw layout, shape [2*cp_size, N, H, K, V+K].
        causal_order: [2*cp_size] int tensor; position r in causal chain
            corresponds to row `causal_order[r]` of `ag_hm`.
    """
    BK = triton.next_power_of_2(K)
    BV = 32
    h_out = torch.empty(N, H, K, V, dtype=torch.float32, device=ag_hm.device)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    cp_merge_kernel[grid](
        h_out=h_out,
        ag_hm=ag_hm,
        h0=h0,
        causal_order_ptr=causal_order,
        num_ranks=num_ranks,
        N=N,
        H=H,
        K=K,
        V=V,
        BV=BV,
        BK=BK,
        HAS_H0=h0 is not None,
        num_warps=4,
        num_stages=2,
    )
    return h_out


def chunk_gated_delta_rule_fwd_cp_zigzag(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cp_group: dist.ProcessGroup,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    seg_cu: Optional[torch.LongTensor] = None,
    causal_order: Optional[torch.Tensor] = None,
):
    """
    CP-parallel gated delta rule forward — zigzag variant.

    Each rank computes on its own zigzag tokens. No QKV all-gather needed.
    Communication is limited to SSM state affine pairs (b, M).

    Input tokens are laid out as [seg0, seg1] per sequence, where seg0 is the
    front half and seg1 is the back half of this rank's zigzag assignment.
    """
    rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)
    num_segs = 2 * cp_size

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # ---- Phase 1: compute (b, M) for both segments in one pass ----
    if seg_cu is None:
        seg_cu = build_segment_cu_seqlens(cu_seqlens)

    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=seg_cu)

    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=seg_cu, output_dtype=torch.float32
    )

    A = solve_tril(A=A, cu_seqlens=seg_cu, output_dtype=k.dtype)

    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=seg_cu)

    b = compute_br(k=k, w=w, u=u, g=g, cu_seqlens=seg_cu)  # [2*N, H, K, V]
    M = compute_M_total(k=k, w=w, g=g, cu_seqlens=seg_cu)  # [2*N, H, K, K]

    b0 = b[0::2].contiguous()  # [N, H, K, V]
    b1 = b[1::2].contiguous()
    M0 = M[0::2].contiguous()  # [N, H, K, K]
    M1 = M[1::2].contiguous()

    N = cu_seqlens.shape[0] - 1
    H = w.shape[2]
    K = k.shape[3]
    V = v.shape[-1]

    # ---- Phase 2: all-gather affine pairs ----
    packed = torch.stack(
        [
            torch.cat([b0, M0], dim=-1),
            torch.cat([b1, M1], dim=-1),
        ],
        dim=0,
    )  # [2, N, H, K, V+K]

    gathered = torch.empty(
        num_segs, *packed.shape[1:], device=packed.device, dtype=packed.dtype
    )
    dist.all_gather_into_tensor(
        gathered.view(num_segs, -1),
        packed.view(2, -1),
        group=cp_group,
    )

    # No physical reorder of `gathered` — cp_merge_kernel reads ag_hm rows
    # via `causal_order` lookup. Saves a 537MB+ transient buffer per layer
    # at high N (and avoids the PyTorch advanced-indexing grid-limit bug).
    if causal_order is None:
        causal_order = torch.tensor(
            zigzag_causal_order(cp_size), dtype=torch.long, device=packed.device
        )

    # ---- Phase 3: cp_merge to get h0 for each segment ----
    h0_global = initial_state.float() if initial_state is not None else None
    seg0_pos, seg1_pos = causal_positions(rank, cp_size)

    h0_seg0 = cp_merge(gathered, h0_global, seg0_pos, N, H, K, V, causal_order)
    h0_seg1 = cp_merge(gathered, h0_global, seg1_pos, N, H, K, V, causal_order)

    # ---- Phase 4: rerun Step5 + Step6 with correct h0 (single pass) ----
    # Interleave h0_seg0 and h0_seg1 to match seg_cu layout:
    # seg_cu sequences are [seg0_seq0, seg1_seq0, seg0_seq1, seg1_seq1, ...]
    h0_combined = torch.empty(2 * N, H, K, V, dtype=torch.float32, device=k.device)
    h0_combined[0::2] = h0_seg0
    h0_combined[1::2] = h0_seg1

    h_all, v_new_all, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0_combined,
        output_final_state=False,
        cu_seqlens=seg_cu,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new_all,
        h=h_all,
        g=g,
        scale=scale,
        cu_seqlens=seg_cu,
    )

    # final_state: result of applying every rank's affine in causal order to h0
    final_state = (
        cp_merge(gathered, h0_global, num_segs, N, H, K, V, causal_order)
        if output_final_state
        else None
    )

    return o, h_all, final_state
