# Adapted from SGLang `gdn_fused_proj.py:_scatter_fused_proj_kernel`.
# Splits a packed [Q|K|V] tensor into three contiguous buffers shaped as
# (1, M, n_heads, head_dim), avoiding the .view() -> .contiguous() copy that
# torch.split + view triggers when the slice stride doesn't match the target
# contig stride.
#
# Also provides the fused split+contiguous+l2norm variant (scatter_qkv_l2norm):
# one pass over the packed tensor that l2-normalizes q/k per (token, head) in
# fp32 while scattering, replacing scatter_qkv/torch.split + the separate
# fused_l2norm_qk call chain.

from typing import Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.offset import linear_offset_64


@triton.jit
def _scatter_qkv_kernel(
    src_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    stride_src_row,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLK_QK: tl.constexpr,
    BLK_V: tl.constexpr,
):
    row = tl.program_id(0)
    src = linear_offset_64(row, stride_src_row)

    offs_qk = tl.arange(0, BLK_QK)
    mask_qk = offs_qk < K_DIM
    q_dst = linear_offset_64(row, K_DIM) + offs_qk
    k_dst = linear_offset_64(row, K_DIM) + offs_qk
    tl.store(
        q_ptr + q_dst, tl.load(src_ptr + src + offs_qk, mask=mask_qk), mask=mask_qk
    )
    tl.store(
        k_ptr + k_dst,
        tl.load(src_ptr + src + K_DIM + offs_qk, mask=mask_qk),
        mask=mask_qk,
    )

    offs_v = tl.arange(0, BLK_V)
    mask_v = offs_v < V_DIM
    v_dst = linear_offset_64(row, V_DIM) + offs_v
    tl.store(
        v_ptr + v_dst,
        tl.load(src_ptr + src + 2 * K_DIM + offs_v, mask=mask_v),
        mask=mask_v,
    )


def scatter_qkv(
    mixed_qkv: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a packed (M, k_dim*2 + v_dim) tensor into (1, M, n_heads, head_dim) buffers.

    Replaces:
        q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim=-1)
        q = q.view(1, M, num_k_heads, head_k_dim)       # triggers .contiguous()
        k = k.view(1, M, num_k_heads, head_k_dim)       # triggers .contiguous()
        v = v.view(1, M, num_v_heads, head_v_dim)       # triggers .contiguous()
    """
    assert mixed_qkv.dim() == 2 and mixed_qkv.is_contiguous(), (
        f"mixed_qkv must be 2D contiguous, got shape={tuple(mixed_qkv.shape)} "
        f"stride={mixed_qkv.stride()} contig={mixed_qkv.is_contiguous()}"
    )
    M = mixed_qkv.shape[0]
    k_dim = num_k_heads * head_k_dim
    v_dim = num_v_heads * head_v_dim
    assert (
        mixed_qkv.shape[1] == 2 * k_dim + v_dim
    ), f"expected last dim 2*{k_dim} + {v_dim} = {2*k_dim + v_dim}, got {mixed_qkv.shape[1]}"

    dtype, device = mixed_qkv.dtype, mixed_qkv.device
    q = torch.empty((1, M, num_k_heads, head_k_dim), dtype=dtype, device=device)
    k = torch.empty((1, M, num_k_heads, head_k_dim), dtype=dtype, device=device)
    v = torch.empty((1, M, num_v_heads, head_v_dim), dtype=dtype, device=device)

    BLK_QK = triton.next_power_of_2(k_dim)
    BLK_V = triton.next_power_of_2(v_dim)
    _scatter_qkv_kernel[(M,)](
        mixed_qkv,
        q,
        k,
        v,
        mixed_qkv.stride(0),
        K_DIM=k_dim,
        V_DIM=v_dim,
        BLK_QK=BLK_QK,
        BLK_V=BLK_V,
        num_warps=4,
    )
    return q, k, v


@triton.jit
def _scatter_qkv_l2norm_kernel(
    src_ptr,  # [M, 2*k_dim + v_dim] packed [Q|K|V], contiguous rows
    q_ptr,  # [M, H, K] contiguous output (normalized)
    k_ptr,  # [M, H, K] contiguous output (normalized)
    v_ptr,  # [M, H, V] contiguous output (pass-through)
    M,
    stride_src_row,
    eps,
    H: tl.constexpr,  # head count (requires num_k_heads == num_v_heads)
    K: tl.constexpr,  # head_k_dim
    V: tl.constexpr,  # head_v_dim
    BT: tl.constexpr,  # rows per program
    BK: tl.constexpr,  # padded head_k_dim
    BV: tl.constexpr,  # padded head_v_dim
):
    # Fused split + contiguous + l2norm: each program handles BT rows of one
    # head across q/k/v. The q/k normalization replicates the fp32 formula of
    # fla/l2norm.py fused_l2norm_qk_kernel_small (load -> fp32, sum(x*x),
    # rsqrt(var + eps), scale, cast back) so outputs are bit-comparable with
    # scatter_qkv + fused_l2norm_qk.
    i_blk = tl.program_id(0)
    i_h = tl.program_id(1)
    rows = i_blk * BT + tl.arange(0, BT)
    row_mask = rows < M

    cols_k = tl.arange(0, BK)
    mask_qk = row_mask[:, None] & (cols_k < K)[None, :]
    src_base = src_ptr + rows[:, None] * stride_src_row + i_h * K

    # q: normalize per (row, head), store contiguous
    b_q = tl.load(src_base + cols_k[None, :], mask=mask_qk, other=0.0).to(tl.float32)
    b_q_rstd = tl.math.rsqrt(tl.sum(b_q * b_q, axis=1) + eps)
    b_q = b_q * b_q_rstd[:, None]
    dst_q = q_ptr + (rows[:, None] * H + i_h) * K
    tl.store(dst_q + cols_k[None, :], b_q.to(dst_q.dtype.element_ty), mask=mask_qk)

    # k: normalize per (row, head), store contiguous
    b_k = tl.load(
        src_base + H * K + cols_k[None, :], mask=mask_qk, other=0.0
    ).to(tl.float32)
    b_k_rstd = tl.math.rsqrt(tl.sum(b_k * b_k, axis=1) + eps)
    b_k = b_k * b_k_rstd[:, None]
    dst_k = k_ptr + (rows[:, None] * H + i_h) * K
    tl.store(dst_k + cols_k[None, :], b_k.to(dst_k.dtype.element_ty), mask=mask_qk)

    # v: pass-through copy, store contiguous
    cols_v = tl.arange(0, BV)
    mask_v = row_mask[:, None] & (cols_v < V)[None, :]
    b_v = tl.load(
        src_ptr + rows[:, None] * stride_src_row + 2 * H * K + i_h * V + cols_v[None, :],
        mask=mask_v,
        other=0.0,
    )
    dst_v = v_ptr + (rows[:, None] * H + i_h) * V
    tl.store(dst_v + cols_v[None, :], b_v, mask=mask_v)


def scatter_qkv_l2norm(
    mixed_qkv: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a packed (M, k_dim*2 + v_dim) tensor and l2-normalize q/k in one pass.

    Replaces the chain
        scatter_qkv(mixed_qkv, ...) + fused_l2norm_qk(q, k)
    (or torch.split + .contiguous() + l2norm) with a single BT-tiled kernel
    that reads mixed_qkv once: q/k are normalized per (token, head) in fp32
    with the same formula as fla/l2norm.py fused_l2norm_qk, v is copied
    through bit-exactly.

    Returns q/k/v as (1, M, num_heads, head_dim) contiguous buffers with q/k
    pre-normalized, so downstream FLA paths run with their in-kernel l2norm
    disabled.

    Requires num_k_heads == num_v_heads (the per-head tiling maps one head
    across q/k/v); callers with GVA layouts must use scatter_qkv instead.
    """
    assert num_k_heads == num_v_heads, (
        f"scatter_qkv_l2norm requires matching head counts, got "
        f"{num_k_heads} vs {num_v_heads}"
    )
    assert mixed_qkv.dim() == 2, (
        f"mixed_qkv must be 2D, got shape={tuple(mixed_qkv.shape)}"
    )
    if not mixed_qkv.is_contiguous():
        mixed_qkv = mixed_qkv.contiguous()
    M = mixed_qkv.shape[0]
    H = num_k_heads
    K = head_k_dim
    V = head_v_dim
    assert mixed_qkv.shape[1] == 2 * H * K + H * V, (
        f"expected last dim 2*{H * K} + {H * V}, got {mixed_qkv.shape[1]}"
    )

    dtype, device = mixed_qkv.dtype, mixed_qkv.device
    q = torch.empty((1, M, H, K), dtype=dtype, device=device)
    k = torch.empty((1, M, H, K), dtype=dtype, device=device)
    v = torch.empty((1, M, H, V), dtype=dtype, device=device)

    BT = 8
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    _scatter_qkv_l2norm_kernel[(triton.cdiv(M, BT), H)](
        mixed_qkv,
        q,
        k,
        v,
        M,
        mixed_qkv.stride(0),
        eps,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=2,
        num_stages=1,
    )
    return q, k, v
