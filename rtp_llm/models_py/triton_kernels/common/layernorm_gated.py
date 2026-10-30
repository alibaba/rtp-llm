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
        "HAS_ROW_INDICES": lambda args: args["row_indices"] is not None,
        "STORE_RSTD": lambda args: args["rstd"] is not None,
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
    T,  # number of output head rows
    row_indices,  # optional output-token to input-token map; -1 pads with zero
    input_tokens,
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
    HAS_ROW_INDICES: tl.constexpr,
    STORE_RSTD: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    o_t = i_t * BT + tl.arange(0, BT)
    if HAS_ROW_INDICES:
        token = tl.load(row_indices + o_t // H, mask=o_t < T, other=-1)
        valid = (o_t < T) & (token >= 0) & (token < input_tokens)
        source_rows = token * H + o_t % H
        b_x = tl.load(
            x + source_rows[:, None] * D + o_d[None, :],
            mask=valid[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
    else:
        source_rows = o_t
        p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        if HAS_ROW_INDICES:
            b_x += tl.load(
                residual + source_rows[:, None] * D + o_d[None, :],
                mask=valid[:, None] & m_d[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
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

    if STORE_RSTD:
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
    o_g = (source_rows // H) * g_stride_n + (source_rows % H) * D
    b_g = tl.load(
        g + o_g[:, None] + o_d[None, :],
        mask=(valid[:, None] if HAS_ROW_INDICES else o_t[:, None] < T) & m_d[None, :],
        other=0.0,
    ).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)
    if HAS_ROW_INDICES:
        b_y = tl.where(valid[:, None], b_y, 0.0)

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
        self.launch_pdl = (
            weight.is_cuda and torch.cuda.get_device_capability(weight.device)[0] >= 9
        )

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        *,
        row_indices: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize heads, optionally gathering input tokens into padded output rows."""
        if x.ndim not in (2, 3) or x.shape[-1] != 128 or gate.shape != x.shape:
            raise ValueError("FusedRMSNormGated expects matching [..., 128] inputs")
        if not x.is_cuda or x.device != gate.device or x.device != self.weight.device:
            raise ValueError(
                "FusedRMSNormGated inputs and weight must share a CUDA device"
            )
        # The native kernel requires packed x, while gate supports row strides.
        shape = x.shape
        heads = x.shape[1] if x.ndim == 3 else 1
        input_tokens = x.shape[0]
        if row_indices is not None:
            if (
                row_indices.ndim != 1
                or row_indices.dtype != torch.int64
                or not row_indices.is_contiguous()
                or row_indices.device != x.device
                or x.dtype != torch.bfloat16
                or gate.dtype != x.dtype
                or self.weight.dtype != x.dtype
            ):
                raise ValueError(
                    "Mapped gated RMSNorm requires BF16 inputs and a CUDA int64 token map"
                )
            shape = (row_indices.numel(), *shape[1:])
        if out is not None and (
            out.shape != shape
            or out.dtype != x.dtype
            or out.device != x.device
            or not out.is_contiguous()
            or out.numel() > 0
            and out.untyped_storage().data_ptr()
            in (
                x.untyped_storage().data_ptr(),
                gate.untyped_storage().data_ptr(),
                self.weight.untyped_storage().data_ptr(),
            )
        ):
            raise ValueError(
                "Gated RMSNorm output must be a separate contiguous tensor with the output shape"
            )
        x = x.contiguous().view(-1, 128)
        if gate.stride(-1) != 1 or (gate.ndim == 3 and gate.stride(1) != 128):
            gate = gate.contiguous()
        y = torch.empty(shape, dtype=x.dtype, device=x.device) if out is None else out
        rows = y.numel() // 128
        if rows == 0:
            return y.reshape(shape)
        rstd = (
            None
            if row_indices is not None
            else torch.empty((rows,), dtype=torch.float32, device=x.device)
        )
        layer_norm_gated_fwd_kernel[(triton.cdiv(rows, 16),)](
            x,
            gate,
            y,
            self.weight,
            None,
            None,
            None,
            None,
            rstd,
            self.eps,
            rows,
            row_indices=row_indices,
            input_tokens=input_tokens,
            H=heads,
            g_stride_n=gate.stride(0),
            D=128,
            BT=16,
            BD=128,
            ACTIVATION="sigmoid",
            IS_RMS_NORM=True,
            num_warps=8,
            launch_pdl=self.launch_pdl,
        )
        return y.reshape(shape)


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
    q_fp8_ptr,
    q_scales_ptr,
    q_amax_ptr,
    scale_rows,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
    QUANTIZE_Q: tl.constexpr,
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
    y = (x * rrms * w).to(row_out.dtype.element_ty)
    tl.store(row_out + block, y, mask=mask)
    if QUANTIZE_Q and pid_task == 0:
        rounded = tl.where(mask, y.to(tl.float32), 0.0)
        tl.store(q_amax_ptr + token_idx, tl.max(tl.abs(rounded), axis=0))
        groups = tl.reshape(rounded, (BLOCK_SIZE // 128, 128))
        amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 1e-4)
        bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        quantized = tl.reshape(
            tl.clamp(groups / scale[:, None], -448.0, 448.0), (BLOCK_SIZE,)
        )
        tl.store(q_fp8_ptr + token_idx * Q_SIZE + block, quantized, mask=mask)
        packed = tl.sum(
            tl.reshape(exponent, (BLOCK_SIZE // 512, 4))
            << (tl.arange(0, 4)[None, :] * 8),
            axis=1,
        )
        packs = tl.arange(0, BLOCK_SIZE // 512)
        tl.store(
            q_scales_ptr + packs * scale_rows + token_idx, packed, packs < Q_SIZE // 512
        )


@lru_cache(None)
def _supports_pdl(device_index):
    return torch.cuda.get_device_capability(device_index)[0] >= 9


def fused_q_kv_rmsnorm(
    q, kv, q_weight, kv_weight, eps, *, packed_output=False, quantize_q=False
):
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
            raise ValueError("MLA norm weight shape does not match input")
    if packed_output:
        packed = q.new_empty((q.shape[0], q.shape[1] + kv.shape[1]))
        q_out, kv_out = packed.split((q.shape[1], kv.shape[1]), dim=-1)
    else:
        q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        kv_out = torch.empty(kv.shape, device=kv.device, dtype=kv.dtype)
    scale_rows = triton.cdiv(q.shape[0], 4) * 4
    if quantize_q:
        if q.shape[1] % 512 or packed_output:
            raise ValueError(
                "Q encoding requires 512-aligned Q and separate Q/KV outputs"
            )
        q_fp8 = q.new_empty(q.shape, dtype=torch.float8_e4m3fn)
        q_scales = q.new_empty((q.shape[1] // 512, scale_rows), dtype=torch.int32)
        q_amax = q.new_empty(q.shape[0], dtype=torch.float32)
    else:
        q_fp8 = q_scales = q_amax = q_out
    if q.shape[0]:
        block = triton.next_power_of_2(max(q.shape[1], kv.shape[1]))
        _fused_q_kv_rmsnorm_kernel[(q.shape[0], 2)](
            q,
            q_out,
            q_weight,
            q.stride(0),
            q_out.stride(0),
            kv,
            kv_out,
            kv_weight,
            kv.stride(0),
            kv_out.stride(0),
            eps,
            q_fp8,
            q_scales,
            q_amax,
            scale_rows,
            Q_SIZE=q.shape[1],
            KV_SIZE=kv.shape[1],
            BLOCK_SIZE=block,
            launch_pdl=_supports_pdl(q.device.index),
            QUANTIZE_Q=quantize_q,
            num_warps=8 if block >= 2048 else 4,
        )
    if quantize_q:
        return q_out, kv_out, (q_fp8, q_scales.T[: q.shape[0]]), q_amax
    return packed if packed_output else (q_out, kv_out)


@triton.jit
def _rounded_add_rmsnorm_kernel(
    Hidden,
    Delta,
    Weight,
    Residual,
    Output,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    offsets = row * WIDTH + cols
    active = cols < WIDTH
    hidden = tl.load(Hidden + offsets, active, other=0).to(tl.float32)
    delta = tl.load(Delta + offsets, active, other=0).to(tl.float32)
    residual = (hidden + delta).to(Residual.dtype.element_ty)
    tl.store(Residual + offsets, residual, active)
    x = residual.to(tl.float32)
    weight = tl.load(Weight + cols, active, other=0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x) / WIDTH + EPS)
    tl.store(Output + offsets, x * rrms * weight, active)


def rounded_add_rmsnorm(hidden, delta, weight, eps):
    """Fuse BF16 residual addition and RMSNorm of the rounded sum."""
    residual, output = torch.empty_like(hidden), torch.empty_like(hidden)
    if hidden.shape[0]:
        width = hidden.shape[1]
        _rounded_add_rmsnorm_kernel[(hidden.shape[0],)](
            hidden,
            delta,
            weight,
            residual,
            output,
            width,
            eps,
            triton.next_power_of_2(width),
            num_warps=8,
        )
    return residual, output


def _validate_q_kv_norm(q, kv, q_weight, kv_weight):
    if q.ndim != 2 or kv.ndim != 2 or q.shape[0] != kv.shape[0]:
        raise ValueError("K3 MLA norms require rank-2 Q/KV with equal token counts")
    for x, weight in ((q, q_weight), (kv, kv_weight)):
        if not x.is_cuda or x.device != q.device or weight.device != q.device:
            raise ValueError("K3 MLA norms require CUDA tensors on the same device")
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError("K3 MLA norms require BF16 tensors")
        if x.stride(-1) != 1 or not weight.is_contiguous():
            raise ValueError("K3 MLA norms require contiguous features and weights")
        if weight.ndim != 1 or weight.numel() != x.shape[1] or x.shape[1] <= 0:
            raise ValueError("K3 MLA norm weight shape does not match input")


@triton.jit(do_not_specialize=["N"])
def _grouped_mla_norm_kernel(
    Q,
    KV,
    QW,
    KVW,
    QO,
    KVO,
    SUFFIX,
    Q_BIAS,
    N,
    QS: tl.constexpr,
    KVS: tl.constexpr,
    KVOS: tl.constexpr,
    SS: tl.constexpr,
    SUFFIX_DIM: tl.constexpr,
    QSIZE: tl.constexpr,
    KVSIZE: tl.constexpr,
    Q_EPS: tl.constexpr,
    KV_EPS: tl.constexpr,
    ROWS: tl.constexpr,
    QB: tl.constexpr,
    KVB: tl.constexpr,
    PDL: tl.constexpr,
    Q_LAYER_NORM: tl.constexpr,
    Q_HAS_BIAS: tl.constexpr,
):
    # Each branch keeps its own reduction width and output layout.
    tile = tl.program_id(0)
    if Q_LAYER_NORM:
        # Eight Indexer rows use one warp per row, preserving the original
        # LayerNorm reduction layout (four adjacent features per lane).
        indexer_tiles = tl.cdiv(N, 8)
        first_branch = tile < indexer_tiles
    else:
        first_branch = tl.program_id(1) == 0
    if first_branch:
        if Q_LAYER_NORM:
            q_rows = (tile * 8 + tl.arange(0, 8)).to(tl.int64)
        else:
            q_rows = (tile * ROWS + tl.arange(0, ROWS)).to(tl.int64)
        qc = tl.arange(0, QB)
        qw = tl.load(QW + qc, qc < QSIZE, other=0).to(tl.float32)
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        qx = tl.load(
            Q + q_rows[:, None] * QS + qc[None, :],
            (q_rows[:, None] < N) & (qc[None, :] < QSIZE),
            other=0,
        ).to(tl.float32)
        if Q_LAYER_NORM:
            centered = qx - (tl.sum(qx, 1) / QSIZE)[:, None]
            centered = tl.where(qc[None, :] < QSIZE, centered, 0.0)
            variance = tl.sum(centered * centered, 1) / QSIZE
            rstd = 1 / tl.sqrt(variance + Q_EPS)
            qy = (centered * rstd[:, None]) * qw[None, :]
            if Q_HAS_BIAS:
                bias = tl.load(Q_BIAS + qc, qc < QSIZE, other=0).to(tl.float32)
                qy = qy + bias[None, :]
        else:
            qy = (
                qx * tl.rsqrt(tl.sum(qx * qx, 1) / QSIZE + Q_EPS)[:, None] * qw[None, :]
            )
        tl.store(
            QO + q_rows[:, None] * QSIZE + qc[None, :],
            qy,
            (q_rows[:, None] < N) & (qc[None, :] < QSIZE),
        )
    else:
        if Q_LAYER_NORM:
            tile -= indexer_tiles
        rows = (tile * ROWS + tl.arange(0, ROWS)).to(tl.int64)
        kc = tl.arange(0, KVB)
        kw = tl.load(KVW + kc, kc < KVSIZE, other=0).to(tl.float32)
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        kx = tl.load(
            KV + rows[:, None] * KVS + kc[None, :],
            (rows[:, None] < N) & (kc[None, :] < KVSIZE),
            other=0,
        ).to(tl.float32)
        ky = kx * tl.rsqrt(tl.sum(kx * kx, 1) / KVSIZE + KV_EPS)[:, None] * kw[None, :]
        tl.store(
            KVO + rows[:, None] * KVOS + kc[None, :],
            ky,
            (rows[:, None] < N) & (kc[None, :] < KVSIZE),
        )
        if SUFFIX_DIM:
            sc = tl.arange(0, KVB)
            sv = tl.load(
                SUFFIX + rows[:, None] * SS + sc[None, :],
                (rows[:, None] < N) & (sc[None, :] < SUFFIX_DIM),
                other=0,
            )
            tl.store(
                KVO + rows[:, None] * KVOS + KVSIZE + sc[None, :],
                sv,
                (rows[:, None] < N) & (sc[None, :] < SUFFIX_DIM),
            )


def grouped_q_kv_rmsnorm(q, kv, q_weight, kv_weight, eps, *, kv_suffix=None):
    """Prefill BF16 norm: four rows per CTA, one launch, FP32 then one cast."""
    _validate_q_kv_norm(q, kv, q_weight, kv_weight)
    if kv_suffix is not None and (
        kv_suffix.ndim != 2
        or kv_suffix.shape[0] != q.shape[0]
        or kv_suffix.device != kv.device
        or kv_suffix.dtype != kv.dtype
        or kv_suffix.stride(-1) != 1
    ):
        raise ValueError(
            "KV suffix must have matching BF16 rows and contiguous features"
        )
    if q.shape[0] < 128 or (q.shape[1], kv.shape[1]) != (1536, 512):
        # Keep the established short-input reduction and rounding. Packing is
        # fused only into the grouped producer that already owns long Prefill.
        q_out, kv_out = fused_q_kv_rmsnorm(q, kv, q_weight, kv_weight, eps)
        if kv_suffix is not None:
            kv_out = torch.cat((kv_out, kv_suffix), dim=-1)[:, : kv.shape[1]]
        return q_out, kv_out
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    suffix_dim = 0 if kv_suffix is None else kv_suffix.shape[1]
    storage = torch.empty(
        (kv.shape[0], kv.shape[1] + suffix_dim), device=kv.device, dtype=kv.dtype
    )
    kv_out = storage[:, : kv.shape[1]]
    if not q.shape[0]:
        return q_out, kv_out
    _grouped_mla_norm_kernel[(triton.cdiv(q.shape[0], 4), 2)](
        q,
        kv,
        q_weight,
        kv_weight,
        q_out,
        kv_out,
        kv if kv_suffix is None else kv_suffix,
        None,
        q.shape[0],
        q.stride(0),
        kv.stride(0),
        storage.stride(0),
        0 if kv_suffix is None else kv_suffix.stride(0),
        suffix_dim,
        q.shape[1],
        kv.shape[1],
        eps,
        eps,
        4,
        triton.next_power_of_2(q.shape[1]),
        triton.next_power_of_2(kv.shape[1]),
        _supports_pdl(q.device.index),
        False,
        False,
        num_warps=8,
    )
    return q_out, kv_out


def grouped_indexer_k_sfa_kv_norm(
    key,
    latent,
    suffix,
    key_weight,
    key_bias,
    kv_weight,
    *,
    key_eps,
    kv_eps,
):
    """Normalize natural-order K3.1 projection slices with one launch."""
    if (
        key.ndim != 2
        or latent.ndim != 2
        or (key.shape[1], latent.shape[1]) != (128, 512)
    ):
        raise ValueError(
            "grouped DSA norm requires 128-column Indexer K and 512-column SFA latent"
        )
    rows = key.shape[0]
    if (
        latent.shape[0] != rows
        or suffix.ndim != 2
        or suffix.shape[0] != rows
        or suffix.shape[1] > latent.shape[1]
    ):
        raise ValueError(
            "grouped DSA norm requires matching token rows and a packed KV suffix"
        )
    if not key.is_cuda or any(
        x.device != key.device or x.dtype != torch.bfloat16 or x.stride(-1) != 1
        for x in (key, latent, suffix)
    ):
        raise ValueError(
            "grouped DSA norm requires BF16 CUDA slices with contiguous features"
        )
    affine = [(key_weight, 128), (kv_weight, 512)]
    if key_bias is not None:
        affine.append((key_bias, 128))
    if any(
        x.device != key.device or x.shape != (width,) or not x.is_contiguous()
        for x, width in affine
    ):
        raise ValueError(
            "grouped DSA norm requires matching contiguous affine parameters"
        )

    key_out = torch.empty(key.shape, device=key.device, dtype=key.dtype)
    suffix_width = suffix.shape[1]
    kv_out = torch.empty(
        (rows, 512 + suffix_width), device=key.device, dtype=latent.dtype
    )
    if rows:
        _grouped_mla_norm_kernel[(triton.cdiv(rows, 8) + triton.cdiv(rows, 4),)](
            key,
            latent,
            key_weight,
            kv_weight,
            key_out,
            kv_out,
            suffix,
            key_bias,
            rows,
            key.stride(0),
            latent.stride(0),
            kv_out.stride(0),
            suffix.stride(0),
            suffix_width,
            128,
            512,
            key_eps,
            kv_eps,
            4,
            128,
            512,
            _supports_pdl(key.device.index),
            True,
            key_bias is not None,
            num_warps=8,
        )
    latent_out, suffix_out = kv_out.split((512, suffix_width), dim=-1)
    return key_out, latent_out, suffix_out


@triton.jit(do_not_specialize=["N"])
def _independent_q_kv_rmsnorm_kernel(
    X,
    W,
    OUT,
    SUFFIX,
    N,
    WIDTH: tl.constexpr,
    XS: tl.constexpr,
    OS: tl.constexpr,
    SS: tl.constexpr,
    SUFFIX_DIM: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    rows = (tl.program_id(0) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    weight = tl.load(W + cols, cols < WIDTH, other=0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(
        X + rows[:, None] * XS + cols[None, :],
        (rows[:, None] < N) & (cols[None, :] < WIDTH),
        other=0,
    ).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 1) / WIDTH + EPS)[:, None] * weight[None, :]
    tl.store(
        OUT + rows[:, None] * OS + cols[None, :],
        y,
        (rows[:, None] < N) & (cols[None, :] < WIDTH),
    )
    if SUFFIX_DIM:
        suffix = tl.load(
            SUFFIX + rows[:, None] * SS + cols[None, :],
            (rows[:, None] < N) & (cols[None, :] < SUFFIX_DIM),
            other=0,
        )
        tl.store(
            OUT + rows[:, None] * OS + WIDTH + cols[None, :],
            suffix,
            (rows[:, None] < N) & (cols[None, :] < SUFFIX_DIM),
        )


def independent_q_kv_rmsnorm(
    x, weight, eps, *, q_width, kv_width, full_rows, suffix=None
):
    """One Q or KV branch using the original full-extent reduction policy."""
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.device != x.device
        or weight.shape != (x.shape[1],)
        or not weight.is_contiguous()
        or x.stride(-1) != 1
        or x.shape[1] not in (q_width, kv_width)
        or not 0 <= x.shape[0] <= full_rows
    ):
        raise ValueError(
            "independent MLA norm requires BF16 Q/KV and original full row extent"
        )
    if suffix is not None and (
        suffix.ndim != 2
        or suffix.shape[0] != x.shape[0]
        or suffix.dtype != x.dtype
        or suffix.device != x.device
        or suffix.stride(-1) != 1
        or x.shape[1] != kv_width
        or suffix.shape[1] > kv_width
    ):
        raise ValueError("packed independent KV norm suffix mismatch")
    grouped = full_rows >= 128 and (q_width, kv_width) == (1536, 512)
    rows = 4 if grouped else 1
    block = triton.next_power_of_2(x.shape[1] if grouped else max(q_width, kv_width))
    suffix_width = 0 if suffix is None else suffix.shape[1]
    storage = torch.empty(
        (x.shape[0], x.shape[1] + suffix_width), dtype=x.dtype, device=x.device
    )
    if x.shape[0]:
        _independent_q_kv_rmsnorm_kernel[(triton.cdiv(x.shape[0], rows),)](
            x,
            weight,
            storage,
            x if suffix is None else suffix,
            x.shape[0],
            x.shape[1],
            x.stride(0),
            storage.stride(0),
            0 if suffix is None else suffix.stride(0),
            suffix_width,
            eps,
            rows,
            block,
            _supports_pdl(x.device.index),
            num_warps=8 if grouped or block >= 2048 else 4,
        )
    return storage[:, : x.shape[1]]
