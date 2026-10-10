# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    SHARED_WEIGHT: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    if not SHARED_WEIGHT:
        W += group * N
        if HAS_BIAS:
            B += group * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        if ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)
        else:
            x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        if ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)
        else:
            y *= z * tl.sigmoid(z)
    # Write output
    tl.store(Y + cols, y, mask=mask)


def layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation="silu",
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    shared_weight = weight.shape == (group_size,)
    assert shared_weight or weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == weight.shape
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            SHARED_WEIGHT=shared_weight,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            ACTIVATION=activation,
            num_warps=num_warps,
        )
    return out, mean, rstd


class RmsNormGated(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        group_size: Optional[int] = None,
        eps: float = 1e-6,
        activation: str = "silu",
    ):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.activation = activation
        if group_size is None:
            self.group_size = self.weight.shape[-1]
        else:
            self.group_size = group_size
        if self.bias is not None:
            assert (
                self.bias.shape[-1] == self.weight.shape[-1]
            ), "Bias dimension must be equal to weight dimension, weight_shape: {}, bias_shape: {}".format(
                self.weight.shape, self.bias.shape
            )
        assert (
            self.weight.shape[-1] % self.group_size == 0
        ), "Weight dimension must be divisible by group size, weight_shape: {}, group_size: {}".format(
            self.weight.shape, self.group_size
        )

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        assert (
            self.weight.shape[-1] == self.group_size
            or x.shape[-1] == self.weight.shape[-1]
        ), "Weight must match input or group size, input_shape: {}, weight_shape: {}".format(
            x.shape, self.weight.shape
        )

        return layer_norm_fwd(
            x,
            self.weight,
            self.bias,
            self.eps,
            z=gate,
            out=None,
            group_size=self.group_size,
            norm_before_gate=True,
            is_rms_norm=True,
            activation=self.activation,
        )[0]


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source was licensed under the MIT license.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

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


class FusedRMSNormGated(torch.nn.Module):
    """Fused sigmoid RMSNorm with vLLM's 128-wide per-head reduction layout.

    Kernel source: vLLM c3b48446349569512749db7f6e2164aa8a33437d,
    third_party/flash_linear_attention/ops/fused_norm_gate.py. Keep the
    16-row, eight-warp layout: changing it changes BF16 rounding.
    """

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        if weight.ndim != 1 or weight.numel() != 128:
            raise ValueError("FusedRMSNormGated requires a 128-element weight")
        self.weight = weight
        self.eps = eps
        self.launch_pdl = weight.is_cuda and torch.cuda.get_device_capability(weight.device)[0] >= 9

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 128 or gate.shape != x.shape:
            raise ValueError("FusedRMSNormGated expects matching [rows, 128] inputs")
        if not x.is_cuda or x.device != gate.device or x.device != self.weight.device:
            raise ValueError("FusedRMSNormGated inputs and weight must share a CUDA device")
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


# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Q/KV RMSNorm adapted from vLLM c3b484463; Apache-2.0 licensed.
@triton.jit
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
    # CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
    # the launch at max-num-batched-tokens >= 65536 with "invalid argument".
    # int64: q_in_stride can be ~24K (128 heads × 192) and overflows int32
    # past num_tokens ~87K under large chunked prefill.
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    # RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
    # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
    # keep x, rrms, and w all in fp32 and perform a single cast at store.
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    # The weight load does not depend on the producer's output, so issue it
    # before the PDL wait: gamma streams in while the producer finishes,
    # leaving a single dependent global round trip (x) after the wait.
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


@lru_cache(None)
def _supports_pdl(device_index):
    return torch.cuda.get_device_capability(device_index)[0] >= 9


def fused_q_kv_rmsnorm(q, kv, q_weight, kv_weight, eps):
    if q.ndim != 2 or kv.ndim != 2 or q.shape[0] != kv.shape[0]:
        raise ValueError("MLA Q/KV norms require rank-2 Q/KV with equal token counts")
    for x, weight in ((q, q_weight), (kv, kv_weight)):
        if not x.is_cuda or x.device != q.device or weight.device != q.device:
            raise ValueError("MLA Q/KV norms require CUDA tensors on the same device")
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError("MLA Q/KV norms require BF16 tensors")
        if x.stride(-1) != 1 or not weight.is_contiguous():
            raise ValueError("MLA Q/KV norms require contiguous features and weights")
        if weight.ndim != 1 or weight.numel() != x.shape[1] or x.shape[1] <= 0:
            raise ValueError("K3 MLA norm weight shape does not match input")
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    kv_out = torch.empty(kv.shape, device=kv.device, dtype=kv.dtype)
    if q.shape[0]:
        block = triton.next_power_of_2(max(q.shape[1], kv.shape[1]))
        _fused_q_kv_rmsnorm_kernel[(q.shape[0], 2)](
            q, q_out, q_weight, q.stride(0), q_out.stride(0),
            kv, kv_out, kv_weight, kv.stride(0), kv_out.stride(0), eps,
            Q_SIZE=q.shape[1], KV_SIZE=kv.shape[1], BLOCK_SIZE=block,
            launch_pdl=_supports_pdl(q.device.index),
            num_warps=8 if block >= 2048 else 4,
        )
    return q_out, kv_out
