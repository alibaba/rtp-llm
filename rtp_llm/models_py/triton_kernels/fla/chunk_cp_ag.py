# -*- coding: utf-8 -*-
# Context Parallelism — All-gather variant (FLA-style)
#
# Same math as chunk_cp_scan.py, but uses all-gather + sequential merge
# instead of Hillis-Steele prefix scan.
#
# Phase 1: All ranks compute (b_r, M_r) locally (parallel)
# Phase 2: All-gather (b_r, M_r) from all ranks
# Phase 3: Sequential merge to compute h0_true
# Phase 4: All ranks rerun Step5 with correct h0 + Step6 (parallel)

from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

# Reuse compute_br and compute_M_total from scan variant
from rtp_llm.models_py.triton_kernels.fla.chunk_cp_scan import (
    compute_br,
    compute_M_total,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_o import chunk_fwd_o
from rtp_llm.models_py.triton_kernels.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd


@triton.jit(do_not_specialize=["num_ranks"])
def cp_merge_kernel(
    h_out,
    ag_hm,
    h0,
    num_ranks,
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
        base = r * stride_rank + ag_base
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


def cp_merge(ag_hm, h0, num_ranks, N, H, K, V):
    """Triton kernel merge: iterate num_ranks affine transforms on h0."""
    BK = triton.next_power_of_2(K)
    BV = 32
    h_out = torch.empty(N, H, K, V, dtype=torch.float32, device=ag_hm.device)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    cp_merge_kernel[grid](
        h_out=h_out,
        ag_hm=ag_hm,
        h0=h0,
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


def chunk_gated_delta_rule_fwd_cp_allgather(
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
    CP-parallel gated delta rule forward — all-gather variant (FLA-style).

    Same phases as scan variant, but Phase 2 uses all-gather + sequential merge.
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
    b_r = compute_br(k=k, w=w, u=u, g=g, cu_seqlens=cu_seqlens)  # [N, H, K, V]
    M_r = compute_M_total(k=k, w=w, g=g, cu_seqlens=cu_seqlens)  # [N, H, K, K]

    # ---- Phase 2: all-gather (b_r, M_r) from all ranks ----
    packed = torch.cat([b_r, M_r], dim=-1)  # [N, H, K, V+K]

    gathered = torch.empty(
        world_size, *packed.shape, device=packed.device, dtype=packed.dtype
    )
    dist.all_gather_into_tensor(
        gathered.view(world_size, -1),
        packed.view(-1),
        group=cp_group,
    )

    # ---- Phase 3: Triton merge to compute h0_true and true_final_local ----
    h0_global = initial_state.float() if initial_state is not None else None

    h0_true = cp_merge(gathered, h0_global, rank, N, H, K, V)
    true_final_local = cp_merge(gathered, h0_global, rank + 1, N, H, K, V)

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
