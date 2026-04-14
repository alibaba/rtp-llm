# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.utils import input_guard

BT_LIST = [8, 16, 32, 64, 128]


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
#     ],
#     key=["D"],
# )
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.jit
def l2norm_fwd_strided_kernel(
    x,
    y,
    stride_x_row: tl.int64,
    D,
    BD: tl.constexpr,
    HEADS_PER_TOKEN: tl.constexpr,
    eps,
):
    """L2-normalize rows from a non-contiguous input into contiguous output.

    Input layout:  each token has HEADS_PER_TOKEN * D contiguous elements,
                   but tokens are separated by stride_x_row (> HEADS_PER_TOKEN * D).
    Output layout: fully contiguous (T * HEADS_PER_TOKEN, D).

    program_id(0) iterates over T * HEADS_PER_TOKEN rows.
    """
    i_t = tl.program_id(0)
    # Map flat row index to (token, head_within_token)
    token = i_t // HEADS_PER_TOKEN
    head = i_t % HEADS_PER_TOKEN
    # Input: strided by token, contiguous within token
    x += token * stride_x_row + head * D
    # Output: always contiguous
    y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


# @triton.autotune(
#     configs=[
#         triton.Config({"BT": BT}, num_warps=num_warps)
#         for num_warps in [1, 2, 4, 8, 16]
#         for BT in BT_LIST
#     ],
#     key=["D", "NB"],
# )
@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    D = x.shape[-1]

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    # Always use contiguous path — the strided kernel has poor memory access patterns
    # that cause 10-30x slowdown on large token counts (e.g. 2M rows from 64K prefill).
    # The implicit copy from .reshape() is much cheaper than strided reads.
    x = x.reshape(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T = x.shape[0]

    if D <= 512:
        # Batched kernel: process BT rows per block to reduce launch count.
        # For D=128 with 2M rows, this reduces blocks from 2M to ~32K.
        BT = max(16, min(64, 8192 // BD))
        num_warps = min(max(BT * BD // 256, 1), 8)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        NB = triton.cdiv(T, 2048)
        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            NB=NB,
            T=T,
            D=D,
            BD=BD,
            BT=BT,
            num_warps=num_warps,
            num_stages=3,
        )
    else:
        num_warps = min(max(BD // 256, 1), 8)
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=num_warps,
            num_stages=3,
        )

    return y.view(x_shape_og)


@triton.jit
def l2norm_fwd_qk_kernel(
    x,
    y_q,
    y_k,
    stride_x_row: tl.int64,
    D,
    BD: tl.constexpr,
    H_K: tl.constexpr,
    eps,
):
    """L2-normalize q and k heads from a packed non-contiguous input into
    two separate contiguous outputs in a single kernel launch.

    Input: packed [token0: q_h0..q_h{H_K-1}, k_h0..k_h{H_K-1}, ...][token1: ...]
           with tokens separated by stride_x_row.
    Output: y_q is contiguous [T*H_K, D], y_k is contiguous [T*H_K, D].

    program_id(0) iterates over T * H_K * 2 rows total.
    """
    i_t = tl.program_id(0)
    total_heads = H_K * 2
    token = i_t // total_heads
    head_in_token = i_t % total_heads
    is_k = head_in_token >= H_K
    head_local = tl.where(is_k, head_in_token - H_K, head_in_token)

    # Input: strided by token, q+k heads adjacent within token
    x += token * stride_x_row + head_in_token * D

    # Output: separate contiguous buffers
    out_row = token * H_K + head_local
    y = tl.where(is_k, y_k, y_q)
    y += out_row * D

    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


def l2norm_fwd_qk(
    q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalize q and k, returning contiguous outputs.

    Uses the standard contiguous l2norm path which has much better memory
    access patterns than the strided kernel (10-30x faster for large prefills).
    """
    return l2norm_fwd(q, eps), l2norm_fwd(k, eps)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, eps=1e-6, output_dtype=None):
        return l2norm_fwd(x, eps, output_dtype)


def l2norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):

    def __init__(self, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)
