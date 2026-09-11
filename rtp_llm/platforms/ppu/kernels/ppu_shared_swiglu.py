"""Merged BF16 PPU shared-expert SwiGLU without FP32 materialization.

SG's merged gate/up contract: clamp and SiLU/multiply in FP32, round once to
BF16 before the existing block-FP8 activation quantizer. This is deliberately
separate from SG's routed MXFP4 kernel, which has a different rounding boundary.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_swiglu(
    gateup,
    output,
    SIZE: tl.constexpr,
    H: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pos = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = pos // H, pos % H
    gate = tl.load(gateup + row * (2 * H) + col, pos < SIZE, 0).to(tl.float32)
    up = tl.load(gateup + row * (2 * H) + H + col, pos < SIZE, 0).to(tl.float32)
    if LIMIT > 0:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    value = (gate * tl.sigmoid(gate)) * up
    tl.store(output + pos, value, pos < SIZE)


def silu_mul_merged_bf16(gate_up, limit=0.0):
    if (
        not gate_up.is_cuda
        or gate_up.dtype != torch.bfloat16
        or not gate_up.is_contiguous()
        or gate_up.ndim < 2
        or gate_up.shape[-1] % 2
    ):
        raise ValueError("PPU shared SwiGLU requires contiguous BF16 [..., 2H]")
    if torch.cuda.get_device_name(gate_up.device) != "ZW-M890P":
        raise RuntimeError("PPU shared SwiGLU requires ZW-M890P")
    h = gate_up.shape[-1] // 2
    out = torch.empty(
        (*gate_up.shape[:-1], h), dtype=gate_up.dtype, device=gate_up.device
    )
    if out.numel():
        _shared_swiglu[(triton.cdiv(out.numel(), 1024),)](
            gate_up,
            out,
            out.numel(),
            h,
            float(limit),
            1024,
            num_warps=4,
        )
    return out
