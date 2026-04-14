# -*- coding: utf-8 -*-
# Context Parallelism — Prefix Scan variant
#
# Phase 1: All ranks run Step1-4 + Step5(h0=0) + compute M_total (parallel)
# Phase 2: Inclusive prefix scan of (M, b) in log2(P) rounds
# Phase 3: Broadcast h0_global, each rank computes h0_true locally
# Phase 4: All ranks rerun Step5 with correct h0 + Step6 (parallel)
#
# Affine pair: f(x) = M @ x + b
# Combine: (M2, b2) ∘ (M1, b1) = (M2 @ M1, M2 @ b1 + b2)

import math
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
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd
from rtp_llm.models_py.triton_kernels.fla.op import exp, safe_exp
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd

# ---------------------------------------------------------------------------
# Lightweight kernel: only compute h_final (= b_r), no h[t] or v_new stored.
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
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
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


# ---------------------------------------------------------------------------
# Kernel: compute M_total [N, H, K, K]
# Propagate identity columns through all chunks to get transfer matrix.
# ---------------------------------------------------------------------------


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
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_wdh = b_wdh * safe_exp(b_g_last - b_g)[:, None]
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


# ---------------------------------------------------------------------------
# Inclusive prefix scan of affine pairs using Hillis-Steele algorithm.
# All communication is parallel within each round.
# After scan, rank r holds: inclusive[r] = f_r ∘ f_{r-1} ∘ ... ∘ f_0
# ---------------------------------------------------------------------------


def _bmm(A, B):
    """Batched matmul for [N, H, K, K] tensors, reshaped to [N*H, K, K]."""
    N, H, K, _ = A.shape
    return torch.bmm(A.view(-1, K, K), B.view(-1, K, K)).view(N, H, K, K)


def _bmv(A, b):
    """Batched mat-vec for [N,H,K,K] @ [N,H,K,V] → [N,H,K,V]."""
    N, H, K, _ = A.shape
    V = b.shape[-1]
    return torch.bmm(A.view(-1, K, K), b.view(-1, K, V)).view(N, H, K, V)


def inclusive_prefix_scan_affine(M_local, b_local, cp_group):
    """
    Hillis-Steele inclusive prefix scan.
    After scan, rank r holds (M_inc, b_inc) = f_r ∘ ... ∘ f_0.
    """
    rank = dist.get_rank(cp_group)
    world_size = dist.get_world_size(cp_group)
    num_rounds = int(math.ceil(math.log2(world_size)))

    M_cur = M_local.clone()
    b_cur = b_local.clone()

    for d in range(num_rounds):
        stride = 1 << d
        src = rank - stride
        dst = rank + stride

        # Allocate recv buffers
        M_recv = torch.empty_like(M_cur)
        b_recv = torch.empty_like(b_cur)

        # All sends and recvs in parallel using isend/irecv
        ops = []
        if dst < world_size:
            ops.append(dist.isend(M_cur, dst=dst, group=cp_group))
            ops.append(dist.isend(b_cur, dst=dst, group=cp_group))
        if src >= 0:
            ops.append(dist.irecv(M_recv, src=src, group=cp_group))
            ops.append(dist.irecv(b_recv, src=src, group=cp_group))

        for op in ops:
            op.wait()

        if src >= 0:
            # combine: (M_cur, b_cur) ∘ (M_recv, b_recv)
            b_cur = _bmv(M_cur, b_recv) + b_cur
            M_cur = _bmm(M_cur, M_recv)

    return M_cur, b_cur


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule_fwd_cp_scan(
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
):
    """
    CP-parallel gated delta rule forward — prefix scan variant.

    Phase 1: All ranks Step1-4 + Step5(h0=0) + M_total  (parallel)
    Phase 2: Inclusive prefix scan in log2(P) rounds
    Phase 3: Broadcast h0_global + local h0_true computation
    Phase 4: All ranks Step5(h0_true) + Step6  (parallel)
    """
    rank = dist.get_rank(cp_group)
    world_size = dist.get_world_size(cp_group)

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # ---- Step 1-4 (all parallel) ----
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )

    B = k.shape[0]
    H = w.shape[2]
    K = k.shape[3]
    V = u.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    # ---- Phase 1: compute b_r and M_r (all parallel) ----
    h_partial_final = compute_br(k=k, w=w, u=u, g=g, cu_seqlens=cu_seqlens)
    M_total = compute_M_total(k=k, w=w, g=g, cu_seqlens=cu_seqlens)

    # ---- Phase 2: inclusive prefix scan (log2(P) rounds) ----
    M_inc, b_inc = inclusive_prefix_scan_affine(M_total, h_partial_final, cp_group)
    # Now rank r has: inclusive[r] = f_r ∘ ... ∘ f_0
    # meaning: true_final[r] = M_inc @ h0_global + b_inc

    # ---- Phase 3: compute h0_true locally + shift ----
    h0_global = (
        initial_state.float()
        if initial_state is not None
        else torch.zeros(N, H, K, V, dtype=torch.float32, device=k.device)
    )

    # All ranks compute true_final[rank] = M_inc @ h0_global + b_inc
    true_final_local = _bmv(M_inc, h0_global) + b_inc

    # Shift: rank r-1 sends true_final to rank r (all parallel)
    h0_true = torch.empty(N, H, K, V, dtype=torch.float32, device=k.device)
    ops = []
    if rank > 0:
        ops.append(dist.irecv(h0_true, src=rank - 1, group=cp_group))
    if rank < world_size - 1:
        ops.append(dist.isend(true_final_local, dst=rank + 1, group=cp_group))
    for op in ops:
        op.wait()
    if rank == 0:
        h0_true = h0_global

    # ---- Phase 4: rerun Step5 with correct h0 + Step6 (all parallel) ----
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0_true,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    final_state = true_final_local if output_final_state else None

    return o, h, final_state
