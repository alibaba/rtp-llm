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

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

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


def exchange_conv_context(
    local_qkv: torch.Tensor,
    cu_seqlens_cpu: list,
    half_lengths_cpu: list,
    rank: int,
    cp_size: int,
    cp_group: dist.ProcessGroup,
    ctx_len: int = 3,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Exchange conv1d context tokens between ranks via all_gather.

    Args:
        local_qkv: [dim, local_total] channel-first
        cu_seqlens_cpu: pre-computed local_cu.tolist()
        half_lengths_cpu: pre-computed (local_lengths // 2).tolist()
        rank, cp_size: rank info
        cp_group: process group
        ctx_len: number of context tokens (kernel_size - 1)

    Returns:
        (seg0_ctx, seg1_ctx): each is [dim, ctx_len * batch] or None
    """
    batch_size = len(half_lengths_cpu)
    dim = local_qkv.shape[0]
    device = local_qkv.device

    seg0_tail = torch.zeros(
        dim, ctx_len * batch_size, device=device, dtype=local_qkv.dtype
    )
    seg1_tail = torch.zeros(
        dim, ctx_len * batch_size, device=device, dtype=local_qkv.dtype
    )
    for b in range(batch_size):
        s = cu_seqlens_cpu[b]
        h = half_lengths_cpu[b]
        n = min(h, ctx_len)
        if n > 0:
            seg0_tail[:, b * ctx_len + ctx_len - n : (b + 1) * ctx_len] = local_qkv[
                :, s + h - n : s + h
            ]
            seg1_tail[:, b * ctx_len + ctx_len - n : (b + 1) * ctx_len] = local_qkv[
                :, s + 2 * h - n : s + 2 * h
            ]

    assert cp_size == 2, "Only cp_size=2 supported for now"

    send_buf = seg0_tail if rank == 0 else seg1_tail
    all_tails = [torch.empty_like(send_buf) for _ in range(cp_size)]
    dist.all_gather(all_tails, send_buf, group=cp_group)

    if rank == 0:
        return None, all_tails[1]
    else:
        return all_tails[0], seg0_tail


def prepend_conv_context(
    local_qkv: torch.Tensor,
    cu_seqlens_cpu: list,
    half_lengths_cpu: list,
    seg0_ctx: Optional[torch.Tensor],
    seg1_ctx: torch.Tensor,
    ctx_len: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepend conv context tokens before each segment for correct conv1d.

    Args:
        local_qkv: [dim, local_total] channel-first
        cu_seqlens_cpu: pre-computed local_cu.tolist()
        half_lengths_cpu: pre-computed (local_lengths // 2).tolist()
        seg0_ctx: [dim, ctx_len * batch] or None
        seg1_ctx: [dim, ctx_len * batch]
        ctx_len: number of context tokens

    Returns:
        (padded_qkv, padded_seg_cu): padded input and new seg cu_seqlens
    """
    batch_size = len(half_lengths_cpu)
    dim = local_qkv.shape[0]
    device = local_qkv.device

    has_seg0_ctx = seg0_ctx is not None
    extra_per_seq = ctx_len * (1 + int(has_seg0_ctx))
    new_total = local_qkv.shape[1] + extra_per_seq * batch_size

    padded = torch.zeros(new_total, dim, device=device, dtype=local_qkv.dtype).T
    padded_seg_cu_cpu = [0]

    offset_src = 0
    offset_dst = 0
    for b in range(batch_size):
        half = half_lengths_cpu[b]

        if has_seg0_ctx:
            padded[:, offset_dst : offset_dst + ctx_len] = seg0_ctx[
                :, b * ctx_len : (b + 1) * ctx_len
            ]
            offset_dst += ctx_len
        padded[:, offset_dst : offset_dst + half] = local_qkv[
            :, offset_src : offset_src + half
        ]
        seg0_len = half + (ctx_len if has_seg0_ctx else 0)
        padded_seg_cu_cpu.append(padded_seg_cu_cpu[-1] + seg0_len)
        offset_dst += half
        offset_src += half

        padded[:, offset_dst : offset_dst + ctx_len] = seg1_ctx[
            :, b * ctx_len : (b + 1) * ctx_len
        ]
        offset_dst += ctx_len
        padded[:, offset_dst : offset_dst + half] = local_qkv[
            :, offset_src : offset_src + half
        ]
        seg1_len = half + ctx_len
        padded_seg_cu_cpu.append(padded_seg_cu_cpu[-1] + seg1_len)
        offset_dst += half
        offset_src += half

    padded_seg_cu = torch.tensor(padded_seg_cu_cpu, dtype=torch.long, device=device)
    return padded, padded_seg_cu, padded_seg_cu_cpu


def strip_conv_context(
    conv_output: torch.Tensor,
    cu_seqlens_cpu: list,
    half_lengths_cpu: list,
    padded_seg_cu_cpu: list,
    seg0_has_ctx: bool,
    local_total: int,
    ctx_len: int = 3,
) -> torch.Tensor:
    """Strip prepended context tokens from conv1d output.

    Returns: [dim, local_total] with context tokens removed.
    """
    batch_size = len(half_lengths_cpu)
    dim = conv_output.shape[0]
    device = conv_output.device

    stripped = torch.empty(local_total, dim, device=device, dtype=conv_output.dtype).T
    offset_dst = 0
    for b in range(batch_size):
        half = half_lengths_cpu[b]

        seg0_start = padded_seg_cu_cpu[2 * b] + (ctx_len if seg0_has_ctx else 0)
        stripped[:, offset_dst : offset_dst + half] = conv_output[
            :, seg0_start : seg0_start + half
        ]
        offset_dst += half

        seg1_start = padded_seg_cu_cpu[2 * b + 1] + ctx_len
        stripped[:, offset_dst : offset_dst + half] = conv_output[
            :, seg1_start : seg1_start + half
        ]
        offset_dst += half

    return stripped


def zigzag_causal_order(cp_size: int) -> list:
    """Map all-gather layout to causal order.

    All-gather layout: [rank0_seg0, rank0_seg1, rank1_seg0, rank1_seg1, ...]
    Causal order (zigzag): rank0_seg0, rank1_seg0, ..., rankN_seg0,
                           rankN_seg1, ..., rank1_seg1, rank0_seg1

    Returns indices into the all-gather layout that produce causal order.
    """
    num_segs = 2 * cp_size
    order = []
    for pos in range(num_segs):
        if pos < cp_size:
            rank = pos
            seg = 0
        else:
            rank = num_segs - 1 - pos
            seg = 1
        order.append(rank * 2 + seg)
    return order


def causal_positions(rank: int, cp_size: int) -> Tuple[int, int]:
    """Return the causal chain positions of this rank's seg0 and seg1."""
    seg0_pos = rank
    seg1_pos = 2 * cp_size - 1 - rank
    return seg0_pos, seg1_pos


def build_segment_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Build cu_seqlens that treats each sequence's two halves as separate sequences.

    Input cu_seqlens: [0, L0, L0+L1, ...]  (batch+1 entries)
    Output: [0, L0/2, L0, L0+L1/2, L0+L1, ...]  (2*batch+1 entries)

    This lets all kernels process seg0 and seg1 in a single pass.
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    half_lengths = lengths // 2
    batch_size = lengths.shape[0]
    seg_cu = torch.zeros(
        2 * batch_size + 1, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    for b in range(batch_size):
        seg_cu[2 * b + 1] = seg_cu[2 * b] + half_lengths[b]
        seg_cu[2 * b + 2] = seg_cu[2 * b + 1] + half_lengths[b]
    return seg_cu


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


def _compute_both_segments(k, v, beta, g, cu_seqlens, seg_cu=None):
    """Run Step1-4 on both segments in one pass.

    Constructs seg_cu_seqlens to treat seg0/seg1 as 2*batch sequences.
    Returns (w, u, g_cumsum, seg_cu, b0, M0, b1, M1) where b0/M0 are for seg0, b1/M1 for seg1.
    """
    if seg_cu is None:
        seg_cu = build_segment_cu_seqlens(cu_seqlens)
    batch_size = cu_seqlens.shape[0] - 1

    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=seg_cu)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=seg_cu, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=seg_cu, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=seg_cu)
    b = compute_br(k=k, w=w, u=u, g=g, cu_seqlens=seg_cu)  # [2*batch, H, K, V]
    M = compute_M_total(k=k, w=w, g=g, cu_seqlens=seg_cu)  # [2*batch, H, K, K]

    b0 = b[0::2].contiguous()  # [batch, H, K, V]
    b1 = b[1::2].contiguous()
    M0 = M[0::2].contiguous()  # [batch, H, K, K]
    M1 = M[1::2].contiguous()

    return w, u, g, seg_cu, b0, M0, b1, M1


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
    w, u, g, seg_cu, b0, M0, b1, M1 = _compute_both_segments(
        k, v, beta, g, cu_seqlens, seg_cu
    )

    batch_size = cu_seqlens.shape[0] - 1
    N = batch_size
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

    # Reorder to causal order
    if causal_order is None:
        causal_order = torch.tensor(
            zigzag_causal_order(cp_size), dtype=torch.long, device=packed.device
        )
    gathered_causal = gathered[causal_order].contiguous()

    # ---- Phase 3: cp_merge to get h0 for each segment ----
    h0_global = initial_state.float() if initial_state is not None else None
    seg0_pos, seg1_pos = causal_positions(rank, cp_size)

    h0_seg0 = cp_merge(gathered_causal, h0_global, seg0_pos, N, H, K, V)
    h0_seg1 = cp_merge(gathered_causal, h0_global, seg1_pos, N, H, K, V)

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

    # Split h into seg0 and seg1 chunks for output
    # h_all has NT_total chunks; split by segment
    h = h_all

    # final_state: pass through all segments
    final_state = None
    if output_final_state:
        final_state = cp_merge(gathered_causal, h0_global, num_segs, N, H, K, V)

    return o, h, final_state
