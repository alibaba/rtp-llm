"""Prefill gated RMSNorm + group FP8 quant, preserving the BF16 boundary."""

import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)


@gluon.jit
def _gated_rmsnorm_fp8_rows(
    X,
    Z,
    W,
    B,
    Q,
    S,
    M,
    SX: gl.constexpr,
    SZ: gl.constexpr,
    GROUPS: gl.constexpr,
    EPS: gl.constexpr,
    SS: gl.constexpr,
    SHARED: gl.constexpr,
    BIAS: gl.constexpr,
    SILU: gl.constexpr,
    UE8M0: gl.constexpr,
    V2: gl.constexpr,
    BT: gl.constexpr,
):
    # Four adjacent channels per lane, one head per warp. Explicit layout
    # preserves the reference reduction and prevents inter-warp conversions.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    r = gl.program_id(0) * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
    c = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    token = r // GROUPS
    group = r % GROUPS
    mask = r[:, None] < M * GROUPS
    off = group[:, None] * 128 + c[None, :]
    x = gl.load(X + token[:, None].to(gl.int64) * SX + off, mask, 0).to(gl.float32)
    z = gl.load(Z + token[:, None].to(gl.int64) * SZ + off, mask, 0).to(gl.float32)
    wi = c[None, :] if SHARED else off
    w = gl.load(W + wi).to(gl.float32)
    var = gl.sum(x * x, 1) / 128
    y = (x * (1.0 / gl.sqrt(var + EPS))[:, None]) * w
    if BIAS:
        y += gl.load(B + wi).to(gl.float32)
    gate = 1.0 / (1.0 + gl.exp(-z))
    if SILU:
        gate = z * gate
    y = (y * gate).to(gl.bfloat16).to(gl.float32)
    amax = gl.maximum(gl.max(gl.abs(y), 1), 1e-4)
    if V2:
        scale = amax * (1.0 / 448.0)
        if UE8M0:
            bits = scale.to(gl.int32, bitcast=True)
            exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(gl.int32)
            scale = (exponent << 23).to(gl.float32, bitcast=True)
            multiplier = ((254 - exponent) << 23).to(gl.float32, bitcast=True)
        else:
            multiplier = gl.div_rn(448.0, amax)
        quant = y * multiplier[:, None]
    else:
        scale = gl.div_rn(amax, 448.0)
        if UE8M0:
            scale = gl.exp2(gl.ceil(gl.log2(gl.maximum(scale, 1e-10))))
        quant = gl.div_rn(y, scale[:, None])
    quant = gl.minimum(gl.maximum(quant, -448.0), 448.0)
    gl.store(Q + r[:, None].to(gl.int64) * 128 + c[None, :], quant, mask)
    if UE8M0:
        exponent = (scale.to(gl.int32, bitcast=True) >> 23) & 255
        byte_offset = (token + (group // 4) * SS) * 4 + group % 4
        gl.store(
            S.to(gl.pointer_type(gl.uint8)) + byte_offset,
            exponent.to(gl.uint8),
            token < M,
        )
    else:
        gl.store(S + token + group * SS, scale, token < M)


def gated_rmsnorm_fp8(
    x,
    gate,
    weight,
    bias=None,
    eps=1e-6,
    activation="silu",
    *,
    scale_ue8m0=False,
    tile_rows=16
):
    if (
        x.ndim != 2
        or gate.shape != x.shape
        or x.dtype != torch.bfloat16
        or gate.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.shape[1] % 512
        or weight.shape not in ((128,), (x.shape[1],))
        or x.stride(1) != 1
        or gate.stride(1) != 1
        or weight.stride(0) != 1
        or activation not in ("silu", "sigmoid")
        or tile_rows not in (4, 8, 16, 32)
    ):
        raise ValueError(
            "Requires BF16 128-wide groups, a multiple of four groups and contiguous feature axes"
        )
    if not x.is_cuda or not all(t.device == x.device for t in (gate, weight)):
        raise ValueError("Expected tensors on the same CUDA device")
    if bias is not None and (
        bias.shape != weight.shape
        or bias.dtype != weight.dtype
        or bias.device != x.device
        or bias.stride(0) != 1
    ):
        raise ValueError("Bias must match weight")
    quant_mode = os.getenv("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
    if quant_mode not in ("auto", "legacy", "v2"):
        raise ValueError("DSV4_FP8_QUANT_KERNEL must be auto, legacy, or v2")
    use_v2 = quant_mode == "v2" or (
        quant_mode == "auto" and x.numel() >= 4 * 1024 * 1024
    )
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    s = create_per_token_group_quant_fp8_output_scale(
        x.shape, x.device, 128, True, True, scale_ue8m0
    )
    m, n = x.shape
    if m:
        _gated_rmsnorm_fp8_rows[(triton.cdiv(m * (n // 128), tile_rows),)](
            x,
            gate,
            weight,
            bias,
            q,
            s,
            m,
            x.stride(0),
            gate.stride(0),
            n // 128,
            eps,
            s.stride(1),
            weight.numel() == 128,
            bias is not None,
            activation == "silu",
            scale_ue8m0,
            use_v2,
            tile_rows,
            num_warps=4,
        )
    return q, s
