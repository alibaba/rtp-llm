# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source was licensed under the MIT license.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch
import triton
import triton.language as tl
@triton.heuristics(
    {
        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
    }
)
@triton.jit
def layer_norm_gated_fwd_kernel(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    T,  # number of rows in x
    H: tl.constexpr,  # number of heads
    g_stride_n,
    D: tl.constexpr,  # number of columns in x
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(
            residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(
            residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (
        (b_x - b_mean[:, None]) * b_rstd[:, None]
        if not IS_RMS_NORM
        else b_x * b_rstd[:, None]
    )
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    # swish/sigmoid output gate
    o_t = i_t * BT + tl.arange(0, BT)
    o_g = (o_t // H) * g_stride_n + (o_t % H) * D
    b_g = tl.load(
        g + o_g[:, None] + o_d[None, :],
        mask=(o_t[:, None] < T) & m_d[None, :],
        other=0.0,
    ).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


class KimiK3GatedNorm(torch.nn.Module):
    """Native K3 sigmoid RMSNorm with vLLM's per-head reduction layout.

    Kernel source: vLLM c3b48446349569512749db7f6e2164aa8a33437d,
    third_party/flash_linear_attention/ops/fused_norm_gate.py. Keep the
    16-row, eight-warp layout: changing it changes BF16 rounding.
    """

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        if weight.ndim != 1 or weight.numel() != 128:
            raise ValueError("Kimi K3 gated norm requires a 128-element weight")
        self.weight = weight
        self.eps = eps
        self.launch_pdl = weight.is_cuda and torch.cuda.get_device_capability(weight.device)[0] >= 9

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 128 or gate.shape != x.shape:
            raise ValueError("Kimi K3 gated norm expects matching [rows, 128] inputs")
        if not x.is_cuda or x.device != gate.device or x.device != self.weight.device:
            raise ValueError("Kimi K3 gated norm inputs and weight must share a CUDA device")
        # The native kernel requires packed x, while gate supports row strides.
        x = x.contiguous()
        if gate.stride(-1) != 1:
            gate = gate.contiguous()
        y = torch.empty_like(x)
        rows = x.shape[0]
        if rows == 0:
            return y
        rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
        layer_norm_gated_fwd_kernel[(triton.cdiv(rows, 16),)](
            x, gate, y, self.weight, None, None, None, None, rstd,
            self.eps, rows, H=1, g_stride_n=gate.stride(0), D=128,
            BT=16, BD=128, ACTIVATION="sigmoid", IS_RMS_NORM=True,
            num_warps=8, launch_pdl=self.launch_pdl,
        )
        return y
