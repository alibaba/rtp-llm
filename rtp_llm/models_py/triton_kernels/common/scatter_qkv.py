# Adapted from SGLang `gdn_fused_proj.py:_scatter_fused_proj_kernel`.
# Splits a packed [Q|K|V] tensor into three contiguous buffers shaped as
# (1, M, n_heads, head_dim), avoiding the .view() -> .contiguous() copy that
# torch.split + view triggers when the slice stride doesn't match the target
# contig stride.

from typing import Tuple

import torch
import triton
import triton.language as tl


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
    src = row * stride_src_row

    offs_qk = tl.arange(0, BLK_QK)
    mask_qk = offs_qk < K_DIM
    q_dst = row * K_DIM + offs_qk
    k_dst = row * K_DIM + offs_qk
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
    v_dst = row * V_DIM + offs_v
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
    if mixed_qkv.dim() != 2 or not mixed_qkv.is_contiguous():
        raise ValueError(
            f"mixed_qkv must be 2D contiguous, got shape={tuple(mixed_qkv.shape)} "
            f"stride={mixed_qkv.stride()} contig={mixed_qkv.is_contiguous()}"
        )
    M = mixed_qkv.shape[0]
    k_dim = num_k_heads * head_k_dim
    v_dim = num_v_heads * head_v_dim
    expected_last_dim = 2 * k_dim + v_dim
    if mixed_qkv.shape[1] != expected_last_dim:
        raise ValueError(
            f"expected last dim 2*{k_dim} + {v_dim} = {expected_last_dim}, "
            f"got {mixed_qkv.shape[1]}"
        )

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
