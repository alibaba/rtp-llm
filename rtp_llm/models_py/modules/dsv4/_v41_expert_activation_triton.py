# SPDX-License-Identifier: MIT
# Copyright (c) 2023 DeepSeek
"""SM120 exact expert preparation/postquant around native FP32 SiLU.

Group-32 quantization follows _v41_mxfp8_triton; see fp8/_v41_cache_triton.LICENSE.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .v41_expert import _check_limit, _check_rows


@triton.jit
def _mul_rn(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prepare_gate(
    GATE,
    ACTIVE,
    OUT,
    GROUPS: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    M: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    CLAMP: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    group = (tl.program_id(0) * BLOCK_G + tl.arange(0, BLOCK_G)).to(tl.int64)
    row = group // GROUPS_PER_ROW
    column = (group % GROUPS_PER_ROW * 32)[:, None] + tl.arange(0, 32)[None, :]
    live = group < GROUPS
    active = tl.load(ACTIVE + row, live, 0)
    offsets = (row // M * GS0 + row % M * GS1)[:, None] + column
    gate = tl.load(GATE + offsets, live[:, None] & active[:, None], 0).to(tl.float32)
    if CLAMP > 0:
        gate = tl.where(gate != gate, gate, tl.minimum(gate, CLAMP))
    output = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(OUT + output, gate, live[:, None])


@triton.jit
def _post_quantize(
    ACTIVATED,
    UP,
    ACTIVE,
    WEIGHTS,
    Q,
    S,
    GROUPS: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    M: tl.constexpr,
    US0: tl.constexpr,
    US1: tl.constexpr,
    CLAMP: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    group = (tl.program_id(0) * BLOCK_G + tl.arange(0, BLOCK_G)).to(tl.int64)
    row = group // GROUPS_PER_ROW
    column = (group % GROUPS_PER_ROW * 32)[:, None] + tl.arange(0, 32)[None, :]
    live = group < GROUPS
    active = tl.load(ACTIVE + row, live, 0)
    output = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    offsets = (row // M * US0 + row % M * US1)[:, None] + column
    up = tl.load(UP + offsets, live[:, None] & active[:, None], 0).to(tl.float32)
    activated = tl.load(ACTIVATED + output, live[:, None], 0)
    weights = tl.load(WEIGHTS + row, live & active, 0)
    if CLAMP > 0:
        up = tl.where(up != up, up, tl.clamp(up, -CLAMP, CLAMP))
    values = _mul_rn(weights[:, None], _mul_rn(activated, up))
    values = values.to(tl.bfloat16).to(tl.float32)
    tl.device_assert(
        tl.sum((tl.abs(values) < float("inf")).to(tl.int32), 1) == 32,
        "V4.1 activation quantization requires finite inputs",
    )
    maximum = tl.maximum(tl.max(tl.abs(values), 1), 0.0001)
    scaled = maximum * (1.0 / 448.0)
    bits = scaled.to(tl.uint32, bitcast=True)
    encoded = (bits >> 23) + (bits & 8388607 != 0).to(tl.uint32)
    scale = (encoded << 23).to(tl.float32, bitcast=True)
    normalized = tl.clamp(tl.div_rn(values, scale[:, None]), -448.0, 448.0)
    payload = normalized.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(
        tl.uint8, bitcast=True
    )
    tl.store(Q + output, payload, live[:, None])
    tl.store(S + group, tl.where(active, encoded, 0).to(tl.uint8), live)


def can_quantize_swiglu(
    gate, up, valid_rows, route_weights, clamp_limit, backend, *, allow_contiguous=False
):
    if not all(
        isinstance(t, torch.Tensor) for t in (gate, up, valid_rows, route_weights)
    ):
        return False
    shape_supported = (
        gate.ndim == 3 and 0 < gate.shape[0] <= 96 and 0 < gate.shape[1] <= 256
    ) or (allow_contiguous and gate.ndim == 2 and gate.shape[0] > 0)
    return (
        backend == "triton"
        and gate.device.type == "cuda"
        and torch.cuda.get_device_capability(gate.device) == (12, 0)
        and shape_supported
        and 0 < gate.shape[-1] <= 4096
        and gate.shape[-1] % 32 == 0
        and up.shape == gate.shape
        and gate.dtype == up.dtype == torch.bfloat16
        and up.device == gate.device
        and gate.stride(-1) == up.stride(-1) == 1
        and valid_rows.shape == route_weights.shape == gate.shape[:-1]
        and valid_rows.dtype == torch.bool
        and route_weights.dtype == torch.float32
        and valid_rows.device == route_weights.device == gate.device
        and valid_rows.is_contiguous()
        and route_weights.is_contiguous()
        and not (
            torch.is_grad_enabled()
            and (gate.requires_grad or up.requires_grad or route_weights.requires_grad)
        )
        and type(clamp_limit) in (int, float)
        and 0 <= clamp_limit <= 1000000.0
    )


def quantize_swiglu_rows(
    gate, up, valid_rows, *, route_weights=None, clamp_limit=10.0, backend="triton"
):
    if gate.ndim == 2:
        values, scales = quantize_swiglu_rows(
            gate.unsqueeze(0),
            up.unsqueeze(0),
            valid_rows.unsqueeze(0),
            route_weights=route_weights.unsqueeze(0),
            clamp_limit=clamp_limit,
            backend=backend,
        )
        return values.squeeze(0), scales
    _check_limit(clamp_limit)
    active = _check_rows(gate.shape[:-1], gate.device, valid_rows, route_weights)
    prepared = torch.empty(gate.shape, dtype=torch.float32, device=gate.device)
    values = torch.empty(gate.shape, dtype=torch.float8_e4m3fn, device=gate.device)
    scales = torch.empty(
        (gate.numel() // gate.shape[-1], gate.shape[-1] // 32),
        dtype=torch.uint8,
        device=gate.device,
    )
    groups = gate.numel() // 32
    grid = (triton.cdiv(groups, 16),)
    _prepare_gate[grid](
        gate,
        active,
        prepared,
        groups,
        gate.shape[-1] // 32,
        gate.shape[1],
        gate.stride(0),
        gate.stride(1),
        CLAMP=float(clamp_limit),
        BLOCK_G=16,
        num_warps=4,
        enable_fp_fusion=False,
    )
    activated = F.silu(prepared)
    _post_quantize[grid](
        activated,
        up,
        active,
        route_weights,
        values.view(torch.uint8),
        scales,
        groups,
        gate.shape[-1] // 32,
        gate.shape[1],
        up.stride(0),
        up.stride(1),
        CLAMP=float(clamp_limit),
        BLOCK_G=16,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (values, scales)
