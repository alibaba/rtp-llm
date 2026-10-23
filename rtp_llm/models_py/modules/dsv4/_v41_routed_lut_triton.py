# SPDX-License-Identifier: MIT
# Copyright (c) 2023 DeepSeek
"""SM120 routed activation with an owner-retained native FP32 SiLU table.

The table is initialized with the same CUDA FP32 SiLU as the maintained path.
Group-32 arithmetic follows _v41_expert_activation_triton, including mul.rn.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4._v41_expert_activation_triton import (
    _mul_rn,
    can_quantize_swiglu,
)
from rtp_llm.models_py.modules.dsv4.v41_expert import _check_limit


@dataclass(frozen=True)
class SiluTable:
    """Retain the table and event for every CUDA graph that references them."""

    values: torch.Tensor
    ready: torch.cuda.Event
    clamp_limit: float


def prepare_silu_table(device, clamp_limit=10.0):
    """Create caller-owned state outside capture/timing; never cache globally."""
    _check_limit(clamp_limit)
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("The routed SiLU table requires SM120")
    if clamp_limit > 1000000.0:
        raise ValueError("Clamp exceeds the maintained activation contract")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Initialize the SiLU table before CUDA graph capture")
        bits = torch.arange(65536, dtype=torch.int32, device=device).to(torch.uint16)
        gate = bits.view(torch.bfloat16).float()
        if clamp_limit:
            gate = gate.clamp(max=float(clamp_limit))
        values = F.silu(gate)
        # Graph capture must emit a wait node for initialization outside capture.
        ready = torch.cuda.Event(external=True)
        ready.record(torch.cuda.current_stream(device))
    return SiluTable(values, ready, float(clamp_limit))


@triton.jit
def _routed_lut_quantize(
    GATE,
    UP,
    VALID,
    WEIGHTS,
    TABLE,
    Q,
    S,
    GROUPS: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    M: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    US0: tl.constexpr,
    US1: tl.constexpr,
    CLAMP: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    group = (tl.program_id(0) * BLOCK_G + tl.arange(0, BLOCK_G)).to(tl.int64)
    row = group // GROUPS_PER_ROW
    live = group < GROUPS
    valid = tl.load(VALID + row, live, 0)
    weight = tl.load(WEIGHTS + row, live & valid, 0)
    active = valid & (weight != 0)
    weight = tl.where(active, weight, 0.0)
    column = (group % GROUPS_PER_ROW * 32)[:, None] + tl.arange(0, 32)[None, :]
    gate_offset = (row // M * GS0 + row % M * GS1)[:, None] + column
    up_offset = (row // M * US0 + row % M * US1)[:, None] + column
    gate = tl.load(GATE + gate_offset, live[:, None] & active[:, None], 0)
    activated = tl.load(TABLE + gate.to(tl.uint16, bitcast=True).to(tl.int32))
    up = tl.load(UP + up_offset, live[:, None] & active[:, None], 0).to(tl.float32)
    if CLAMP > 0:
        up = tl.where(up != up, up, tl.clamp(up, -CLAMP, CLAMP))
    values = _mul_rn(weight[:, None], _mul_rn(activated, up))
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
    output = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(Q + output, payload, live[:, None])
    tl.store(S + group, tl.where(active, encoded, 0).to(tl.uint8), live)


def is_supported(gate, up, valid, weights, table, clamp_limit=10.0):
    return (
        isinstance(table, SiluTable)
        and can_quantize_swiglu(
            gate, up, valid, weights, clamp_limit, "triton", allow_contiguous=True
        )
        and table.clamp_limit == float(clamp_limit)
        and table.values.device == gate.device
        and table.values.dtype == torch.float32
        and table.values.shape == (65536,)
        and table.values.is_contiguous()
    )


def quantize_swiglu_lut(gate, up, valid, weights, table, *, clamp_limit=10.0):
    """Return None for unsupported inputs so callers can use the maintained path."""
    if not is_supported(gate, up, valid, weights, table, clamp_limit):
        return None
    values = torch.empty(gate.shape, dtype=torch.float8_e4m3fn, device=gate.device)
    scales = torch.empty(
        (gate.numel() // gate.shape[-1], gate.shape[-1] // 32),
        dtype=torch.uint8,
        device=gate.device,
    )
    # The owner may drop the table after enqueue, including on another stream.
    stream = torch.cuda.current_stream(gate.device)
    stream.wait_event(table.ready)
    table.values.record_stream(stream)
    if gate.ndim == 2:
        gate, up = gate.unsqueeze(0), up.unsqueeze(0)
    groups = gate.numel() // 32
    _routed_lut_quantize[(triton.cdiv(groups, 16),)](
        gate,
        up,
        valid,
        weights,
        table.values,
        values.view(torch.uint8),
        scales,
        groups,
        gate.shape[-1] // 32,
        gate.shape[1],
        gate.stride(0),
        gate.stride(1),
        up.stride(0),
        up.stride(1),
        CLAMP=float(clamp_limit),
        BLOCK_G=16,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return values, scales
