# Adapted from SGLang `gdn_fused_proj.py:_scatter_fused_proj_kernel`.
# Splits a packed [Q|K|V] tensor into three contiguous buffers shaped as
# (1, M, n_heads, head_dim), avoiding the .view() -> .contiguous() copy that
# torch.split + view triggers when the slice stride doesn't match the target
# contig stride.

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.offset import linear_offset_64

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError:
    gluon = None


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


if gluon is not None:

    @gluon.jit(do_not_specialize=["m"])
    def _scatter_qkv_l2norm_sm90_multi_head(
        packed, q_out, k_out, v_out, m, H: gl.constexpr
    ):
        # Four warps own the 128 columns of each head, preserving the 1D
        # reference reduction tree. An eight-warp CTA tiles four Q/K pairs
        # (not necessarily a whole token); each pair also copies three V heads.
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [2, 4], [1, 0])
        heads = gl.program_id(0) * 4 + gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
        col = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
        token = (heads // H).to(gl.int64)
        head = heads % H
        row_width: gl.constexpr = 5 * H * 128
        src = token[:, None] * row_width + head[:, None] * 128 + col[None, :]
        dst = heads[:, None].to(gl.int64) * 128 + col[None, :]
        valid = heads[:, None] < m * H
        q = gl.load(packed + src, valid, other=0).to(gl.float32)
        k = gl.load(packed + src + H * 128, valid, other=0).to(gl.float32)
        qi = 1.0 / gl.sqrt(gl.sum(q * q, axis=1) + 1e-6)
        ki = 1.0 / gl.sqrt(gl.sum(k * k, axis=1) + 1e-6)
        gl.store(q_out + dst, q * qi[:, None], valid)
        gl.store(k_out + dst, k * ki[:, None], valid)
        for part in gl.static_range(3):
            v_col = (head * 3 + part)[:, None] * 128 + col[None, :]
            v = gl.load(packed + token[:, None] * row_width + 2 * H * 128 + v_col, valid, other=0)
            gl.store(v_out + token[:, None] * (3 * H * 128) + v_col, v, valid)


def try_scatter_qkv_l2norm_sm90(
    packed: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return normalized Q/K and raw V for qualified Qwen4 prefill geometry."""
    if not (
        os.environ.get("QWEN4_GDN_SM90_FUSED_SCATTER_NORM") == "1"
        and gluon is not None
        and (num_k_heads, num_v_heads) in ((2, 6), (8, 24))
        and head_k_dim == head_v_dim == 128
        and packed.ndim == 2
        and 2048 <= packed.shape[0] <= 32768
        and packed.shape[1] == (2 * num_k_heads + num_v_heads) * 128
        and packed.is_cuda
        and packed.is_contiguous()
        and packed.dtype == torch.bfloat16
        and packed.data_ptr() % 16 == 0
        and torch.version.hip is None
        and torch.cuda.get_device_capability(packed.device) == (9, 0)
    ):
        return None
    m = packed.shape[0]
    q = packed.new_empty((1, m, num_k_heads, 128))
    k = torch.empty_like(q)
    v = packed.new_empty((1, m, num_v_heads, 128))
    _scatter_qkv_l2norm_sm90_multi_head[(triton.cdiv(m * num_k_heads, 4),)](
        packed, q, k, v, m, H=num_k_heads, num_warps=8, num_stages=1
    )
    return q, k, v
