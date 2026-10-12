# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Adapted from rtp_llm.models_py.triton_kernels.fla.fused_recurrent, whose
# original flash-linear-attention implementation is Copyright (c) 2023-2025,
# Songlin Yang, Yu Zhang, under the MIT license.
"""Experimental paged GDN gating/recurrence fusion with optional output norm."""

import math
import os

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import layer_norm_fwd
from rtp_llm.models_py.triton_kernels.fla.op import exp


@triton.jit
def _post_conv_kernel(
    Q,
    K,
    V,
    A,
    B,
    A_LOG,
    DT,
    Z,
    WEIGHT,
    STATE,
    BLOCK_MAP,
    LENGTHS,
    OUT,
    H: tl.constexpr,
    HV: tl.constexpr,
    DIM: tl.constexpr,
    STEPS: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    ZS: tl.constexpr,
    SS: tl.constexpr,
    MS: tl.constexpr,
    PAGE: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    SHARED_WEIGHT: tl.constexpr,
    SIGMOID_GATE: tl.constexpr,
    FUSED_NORM: tl.constexpr,
    VALUE_TILE: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    key_head = head // (HV // H)
    length = tl.load(LENGTHS + request).to(tl.int64)
    if length <= 0:
        return
    read_slot = (length - 2) // PAGE
    read_block = tl.load(BLOCK_MAP + request * MS + read_slot).to(tl.int64)
    if read_block <= 0:
        return
    k_index = tl.arange(0, DIM)
    v_index = tl.program_id(2) * VALUE_TILE + tl.arange(0, VALUE_TILE)
    state_offset = head * DIM * DIM + v_index[:, None] * DIM + k_index[None, :]
    state = tl.load(STATE + read_block * SS + state_offset).to(tl.float32)
    a_log = tl.load(A_LOG + head).to(tl.float32)
    dt = tl.load(DT + head).to(tl.float32)
    if FUSED_NORM:
        weight_offset = v_index if SHARED_WEIGHT else head * DIM + v_index
        weight = tl.load(WEIGHT + weight_offset).to(tl.float32)
    for step in range(STEPS):
        token = request * STEPS + step
        q = tl.load(Q + token * QS + key_head * QH + k_index).to(tl.float32)
        k = tl.load(K + token * KS + key_head * KH + k_index).to(tl.float32)
        value = tl.load(V + token * VS + head * VH + v_index).to(tl.float32)
        a = tl.load(A + token * AS + head).to(tl.float32) + dt
        b = tl.load(B + token * BS + head).to(tl.float32)
        softplus = tl.where(a <= 20.0, tl.log(1.0 + tl.exp(a)), a)
        g = -tl.exp(a_log) * softplus
        # CUDA's existing gating producer stores beta in BF16 before recurrence.
        beta = tl.sigmoid(b).to(tl.bfloat16).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        q *= SCALE
        state *= exp(g)
        value -= tl.sum(state * k[None, :], 1)
        value *= beta
        state += value[:, None] * k[None, :]
        core = tl.sum(state * q[None, :], 1).to(tl.bfloat16).to(tl.float32)
        if FUSED_NORM:
            inv_rms = 1.0 / tl.sqrt(tl.sum(core * core) / DIM + EPS)
            normalized = core * inv_rms * weight
            z = tl.load(Z + token * ZS + head * DIM + v_index).to(tl.float32)
            gate = tl.sigmoid(z) if SIGMOID_GATE else z * tl.sigmoid(z)
            result = normalized * gate
        else:
            result = core
        tl.store(OUT + token * HV * DIM + head * DIM + v_index, result)
        write_slot = (length - 1) // PAGE + step
        write_block = tl.load(BLOCK_MAP + request * MS + write_slot).to(tl.int64)
        tl.store(STATE + write_block * SS + state_offset, state)


def is_supported(
    q, k, v, a, b, a_log, dt_bias, z, weight, state, block_map, lengths, activation
):
    if not (
        q.is_cuda and torch.version.hip is None and q.ndim == k.ndim == v.ndim == 4
    ):
        return False
    batch, steps, heads, dim = q.shape
    # Keep one-token draft steps on the established path: fusion costs more
    # graph replay time at this size, even when the summed kernel time is lower.
    value_heads = v.shape[2]
    tensors = (q, k, v, a, b, a_log, dt_bias, z, weight, state, block_map, lengths)
    return (
        all(t.device == q.device for t in tensors)
        and q.dtype
        == k.dtype
        == v.dtype
        == a.dtype
        == b.dtype
        == z.dtype
        == torch.bfloat16
        and dim == 128
        and k.shape == q.shape
        and v.shape == (batch, steps, value_heads, dim)
        and 2 <= steps <= 4
        and heads > 0
        and value_heads % heads == 0
        and all(t.stride(0) == steps * t.stride(1) for t in (q, k, v))
        and all(t.stride(-1) == 1 for t in (q, k, v, a, b, z, weight))
        and a.shape == b.shape == (batch * steps, value_heads)
        and z.shape == (batch * steps, value_heads * dim)
        and a_log.shape == dt_bias.shape == (value_heads,)
        and a_log.is_contiguous()
        and dt_bias.is_contiguous()
        and a_log.dtype in (torch.float32, torch.bfloat16)
        and dt_bias.dtype in (torch.float32, torch.bfloat16)
        and weight.numel() in (dim, value_heads * dim)
        and weight.ndim == 1
        and weight.dtype == torch.bfloat16
        and state.ndim == 4
        and state.shape[1:] == (value_heads, dim, dim)
        and state.dtype in (torch.bfloat16, torch.float32)
        and state.stride(1) == dim * dim
        and state.stride(2) == dim
        and state.stride(3) == 1
        and block_map.ndim == 2
        and block_map.shape[0] == batch
        and block_map.stride(1) == 1
        and block_map.dtype == torch.int32
        and lengths.shape == (batch,)
        and lengths.dtype == torch.int32
        and lengths.is_contiguous()
        and activation in ("silu", "sigmoid")
    )


def try_gdn_post_conv(
    q,
    k,
    v,
    a,
    b,
    a_log,
    dt_bias,
    z,
    weight,
    state,
    block_map,
    lengths,
    *,
    page_size,
    eps,
    activation
):
    if page_size <= 0 or not is_supported(
        q, k, v, a, b, a_log, dt_bias, z, weight, state, block_map, lengths, activation
    ):
        return None
    batch, steps, heads, dim = q.shape
    # Keep low-traffic singleton requests on the established model path.
    # Explicitly set one to reproduce the unrestricted experimental variant.
    min_batch = int(os.environ.get("RTP_LLM_QWEN4_GDN_POST_CONV_MIN_BATCH", "2"))
    if min_batch < 1:
        raise ValueError("Qwen4 GDN post-conv minimum batch must be positive")
    if batch < min_batch:
        return None
    value_heads = v.shape[2]
    layout = os.environ.get("RTP_LLM_QWEN4_GDN_POST_CONV_LAYOUT", "split")
    if layout not in ("fused", "split"):
        raise ValueError("Qwen4 GDN post-conv layout must be 'fused' or 'split'")
    fused_norm = layout == "fused"
    value_tile = dim if fused_norm else 8
    output = torch.empty(
        (batch * steps, value_heads * dim), device=q.device, dtype=q.dtype
    )
    if batch:
        _post_conv_kernel[(batch, value_heads, dim // value_tile)](
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            z,
            weight,
            state,
            block_map,
            lengths,
            output,
            H=heads,
            HV=value_heads,
            DIM=dim,
            STEPS=steps,
            QS=q.stride(1),
            QH=q.stride(2),
            KS=k.stride(1),
            KH=k.stride(2),
            VS=v.stride(1),
            VH=v.stride(2),
            AS=a.stride(0),
            BS=b.stride(0),
            ZS=z.stride(0),
            SS=state.stride(0),
            MS=block_map.stride(0),
            PAGE=page_size,
            EPS=eps,
            SCALE=1.0 / math.sqrt(dim),
            SHARED_WEIGHT=weight.numel() == dim,
            SIGMOID_GATE=activation == "sigmoid",
            FUSED_NORM=fused_norm,
            VALUE_TILE=value_tile,
            num_warps=4 if fused_norm else 1,
            num_stages=1,
        )
    if not fused_norm:
        return layer_norm_fwd(
            output,
            weight,
            None,
            eps,
            z=z,
            group_size=dim,
            norm_before_gate=True,
            is_rms_norm=True,
            activation=activation,
        )[0]
    return output
