"""Fuse GDN gated norm with the existing FP32 or UE8M0 scale boundary.

The BF16 norm rounding is kept before amax and FP8 conversion. Head traversal
follows the vLLM norm, while the reduction layout and scale encoding retain
RTP's current contract.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)


@gluon.jit(
    do_not_specialize=["M", "SS"],
    do_not_specialize_on_alignment=["M", "SS"],
)
def _gdn_norm_quant(
    X,
    Z,
    W,
    Q,
    S,
    M,
    N: gl.constexpr,
    ZX: gl.constexpr,
    EPS: gl.constexpr,
    GROUPS: gl.constexpr,
    SS,
    GH: gl.constexpr,
    WARPS: gl.constexpr,
    UE8M0: gl.constexpr,
    LEGACY: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [WARPS, 1], [1, 0])
    flat = gl.program_id(0) * GH + gl.arange(0, GH, layout=gl.SliceLayout(1, layout))
    row = flat // GROUPS
    head = flat % GROUPS
    col = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    offset = head[:, None] * 128 + col[None, :]
    valid = flat[:, None] < M * GROUPS
    value = gl.load(X + row[:, None] * N + offset, valid, other=0.0).to(gl.float32)
    variance = gl.sum(value * value, 1) / 128.0
    inverse = 1.0 / gl.sqrt(variance + EPS)
    weight = gl.load(W + col).to(gl.float32)
    gate = gl.load(Z + row[:, None] * ZX + offset, valid, other=0.0).to(gl.float32)
    normalized = ((value * inverse[:, None]) * weight[None, :]) * (
        1.0 / (1.0 + gl.exp(-gate))
    )
    rounded = normalized.to(gl.bfloat16).to(gl.float32)
    absolute = gl.abs(rounded)
    amax = gl.maximum(gl.max(gl.where(absolute == absolute, absolute, 0.0), 1), 1.0e-4)
    scale_float = amax * (1.0 / 448.0)
    if UE8M0:
        bits = scale_float.to(gl.uint32, bitcast=True)
        exponent = (
            ((bits >> 23) & 255).to(gl.int32)
            - 127
            + ((bits & 0x7FFFFF) != 0).to(gl.int32)
        )
        reciprocal = ((127 - exponent).to(gl.uint32) << 23).to(gl.float32, bitcast=True)
    else:
        # Match the CUDA v2 quantizer division before FP8 rounding.
        reciprocal = gl.div_rn(448.0, amax)
    if LEGACY:
        # The legacy CUDA path rounds both divisions independently.
        scale_float = gl.div_rn(amax, 448.0)
        scaled = gl.div_rn(rounded, scale_float[:, None])
    else:
        scaled = rounded * reciprocal[:, None]
    scaled = gl.where(scaled == scaled, scaled, -448.0)
    quant = gl.minimum(gl.maximum(scaled, -448.0), 448.0)
    gl.store(Q + row[:, None] * N + offset, quant, valid)
    if UE8M0:
        scale_offset = ((head // 4) * SS + row) * 4 + head % 4
        gl.store(
            S.to(gl.pointer_type(gl.uint8)) + scale_offset,
            exponent + 127,
            flat < M * GROUPS,
        )
    else:
        gl.store(S + row + head * SS, scale_float, flat < M * GROUPS)


def norm_quant(x, z, weight, eps, groups=8, warps=1, *, scale_ue8m0=True):
    rows, width = x.shape
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and width in (1536, 3072)
        and 2048 <= rows <= (8192 if scale_ue8m0 else 32768)
        and z.device == x.device
        and z.shape == x.shape
        and z.dtype == x.dtype
        and z.stride(1) == 1
        and z.stride(0) > 0
        and weight.device == x.device
        and weight.shape == (128,)
        and weight.dtype in (torch.bfloat16, torch.float32)
        and weight.is_contiguous()
        and groups == 8
        and warps == (1 if scale_ue8m0 else 2)
    ):
        raise ValueError("unsupported GDN norm/FP8 input layout")
    return _launch_norm_quant(x, z, weight, eps, groups, warps, scale_ue8m0=scale_ue8m0)


def _launch_norm_quant(
    x, z, weight, eps, groups=8, warps=1, *, scale_ue8m0=True, legacy=False
):
    """Launch after the public helper or model dispatch validates the layout."""
    rows, width = x.shape
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = create_per_token_group_quant_fp8_output_scale(
        x.shape, x.device, 128, True, True, scale_ue8m0
    )
    _gdn_norm_quant[(triton.cdiv(rows * (width // 128), groups),)](
        x,
        z,
        weight,
        q,
        s,
        M=rows,
        N=width,
        ZX=z.stride(0),
        EPS=eps,
        GROUPS=width // 128,
        SS=s.stride(1),
        GH=groups,
        WARPS=warps,
        UE8M0=scale_ue8m0,
        LEGACY=legacy,
        num_warps=warps,
    )
    return q, s
