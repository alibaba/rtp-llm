"""Prefill gated RMSNorm + group FP8 quant, preserving the BF16 boundary."""

import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from rtp_llm.models_py.utils.prefill_input_log import trace_triton


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
    AMAX_FLOOR: gl.constexpr,
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
    amax = gl.maximum(gl.max(gl.abs(y), 1), AMAX_FLOOR)
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


# Norm arithmetic and row scheduling follow vLLM's Apache-2.0 FLA
# layernorm_guard.py (Copyright contributors to vLLM and Tri Dao). Keep the
# original RTP FP8 arithmetic below the register-only BF16 rounding boundary.
@gluon.jit
def _gated_rmsnorm_fp8_rows_vllm(
    X,
    Z,
    W,
    B,
    Q,
    S,
    Y,
    M,
    SX: gl.constexpr,
    SZ: gl.constexpr,
    SY: gl.constexpr,
    GROUPS: gl.constexpr,
    EPS,
    SS: gl.constexpr,
    SHARED: gl.constexpr,
    BIAS: gl.constexpr,
    SILU: gl.constexpr,
    UE8M0: gl.constexpr,
    V2: gl.constexpr,
    BT: gl.constexpr,
    AMAX_FLOOR: gl.constexpr,
    WRITE_Y: gl.constexpr,
):
    # Match the actual vLLM 128-channel reductions. With two or four rows,
    # each half warp reduces eight adjacent channels per lane. A single row
    # instead uses a full warp and four channels per lane.
    if BT == 1:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, 1], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [1, 1], [1, 0])
    rows = gl.program_id(0) * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
    c = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    if SHARED:
        token = rows // GROUPS
        group = rows % GROUPS
    else:
        token = rows
        group = gl.full((BT,), gl.program_id(1), gl.int32, gl.SliceLayout(1, layout))
    mask = token[:, None] < M
    off = group[:, None] * 128 + c[None, :]
    x = gl.load(X + token[:, None].to(gl.int64) * SX + off, mask, 0).to(gl.float32)
    z = gl.load(Z + token[:, None].to(gl.int64) * SZ + off, mask, 0).to(gl.float32)
    wi = c[None, :] if SHARED else off
    w = gl.load(W + wi).to(gl.float32)
    var = gl.sum(x * x, 1) / 128
    y = (x * gl.rsqrt(var + EPS)[:, None]) * w
    if BIAS:
        y += gl.load(B + wi).to(gl.float32)
    gate = 1.0 / (1.0 + gl.exp(-z))
    if SILU:
        gate = z * gate
    y_bf16 = (y * gate).to(gl.bfloat16)
    if WRITE_Y:
        # Optional caller-owned observation buffer; production does not allocate it.
        gl.store(Y + token[:, None].to(gl.int64) * SY + off, y_bf16, mask)
    y = y_bf16.to(gl.float32)
    amax = gl.maximum(gl.max(gl.abs(y), 1), AMAX_FLOOR)
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
    gl.store(Q + token[:, None].to(gl.int64) * (GROUPS * 128) + off, quant, mask)
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
    tile_rows=16,
    vllm_math=None,
    amax_floor=1e-10,
    out_y=None,
):
    """Fuse gated RMSNorm and FP8 quantization without a BF16 intermediate.

    vllm_math selects vLLM's 128-channel FP32 reduction/rsqrt schedule. When
    omitted, RTP_LENGTH_ALIGN_GATED_NORM controls it (default on). FP8 floor,
    encoding and legacy/v2 policy are independent. tile_rows tunes only the
    original path. amax_floor defaults to 1e-10 for GDN out_proj. out_y
    optionally observes BF16 norm in a caller-owned buffer.
    """
    if vllm_math is None:
        vllm_math = os.getenv("RTP_LENGTH_ALIGN_GATED_NORM", "1") == "1"
    if not isinstance(vllm_math, bool):
        raise TypeError("vllm_math must be bool or None")
    if amax_floor not in (1e-4, 1e-10):
        raise ValueError("amax_floor must be 1e-4 or 1e-10")
    if not vllm_math and out_y is not None:
        raise ValueError("out_y requires vllm_math=True")
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
    if out_y is not None and (
        out_y.shape != x.shape
        or out_y.dtype != x.dtype
        or out_y.device != x.device
        or out_y.stride(1) != 1
    ):
        raise ValueError(
            "out_y must match input shape, dtype and device with contiguous channels"
        )
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
    if m and vllm_math:
        shared = weight.numel() == 128
        rows = m * (n // 128) if shared else m
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        rows_per_block = min(triton.next_power_of_2(triton.cdiv(rows, 2 * sm_count)), 4)
        trace_triton(
            "gated_rmsnorm_fp8:_gated_rmsnorm_fp8_rows_vllm[grid]",
            _gated_rmsnorm_fp8_rows_vllm,
            (triton.cdiv(rows, rows_per_block), 1 if shared else n // 128),
            x,
            gate,
            weight,
            bias,
            q,
            s,
            out_y,
            m,
            x.stride(0),
            gate.stride(0),
            out_y.stride(0) if out_y is not None else 0,
            n // 128,
            eps,
            s.stride(1),
            shared,
            bias is not None,
            activation == "silu",
            scale_ue8m0,
            use_v2,
            rows_per_block,
            amax_floor,
            out_y is not None,
            num_warps=1,
        )
    elif m:
        trace_triton(
            "gated_rmsnorm_fp8:_gated_rmsnorm_fp8_rows[triton.cdiv(m * (n // 128), tile_rows),]",
            _gated_rmsnorm_fp8_rows,
            (triton.cdiv(m * (n // 128), tile_rows),),
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
            amax_floor,
            num_warps=4,
        )
    return q, s
