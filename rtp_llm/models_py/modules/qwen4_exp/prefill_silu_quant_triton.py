"""Preserve the shared FFN's BF16 activation and dynamic per-tensor FP8 scale."""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_partial(
    GATE, Y, PARTIAL, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr
):
    program = tl.program_id(0)
    offsets = program * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // D, offsets % D
    gate = tl.load(GATE + row * (2 * D) + col, offsets < N, other=0).to(tl.float32)
    up = tl.load(GATE + row * (2 * D) + D + col, offsets < N, other=0).to(tl.float32)
    value = ((gate * tl.sigmoid(gate)) * up).to(tl.bfloat16)
    tl.store(Y + offsets, value, offsets < N)
    absolute = tl.abs(value.to(tl.float32))
    absolute = tl.where((offsets < N) & (absolute == absolute), absolute, 0.0)
    tl.store(PARTIAL + program, tl.max(absolute, 0))


@triton.jit
def _partial_quant(
    Y,
    PARTIAL,
    Q,
    SCALE,
    N: tl.constexpr,
    OUT: tl.constexpr,
    PARTS: tl.constexpr,
    PTILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    part = tl.arange(0, PTILE)
    maximum = tl.max(tl.load(PARTIAL + part, part < PARTS, other=0.0), 0)
    scale = maximum * (1.0 / 448.0)
    # Match the native quantizer's subnormal scale flush and NaN clipping.
    scale = tl.where(scale < 1.1754943508222875e-38, 0.0, scale)
    if tl.program_id(0) == 0:
        tl.store(SCALE, scale)
    inverse = tl.div_rn(1.0, tl.maximum(scale, 1.0e-9))
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Y + offsets, offsets < N, other=0.0).to(tl.float32) * inverse
    value = tl.where(value == value, value, 448.0)
    value = tl.maximum(-448.0, tl.minimum(448.0, value))
    tl.store(
        Q + offsets, tl.where(offsets < N, value, 0.0).to(tl.float8e4nv), offsets < OUT
    )


def is_supported(gate_up, output=None):
    if not (
        isinstance(gate_up, torch.Tensor)
        and gate_up.is_cuda
        and torch.version.hip is None
        and gate_up.dtype == torch.bfloat16
        and gate_up.ndim == 2
        and gate_up.is_contiguous()
        and not gate_up.requires_grad
        and 4093 <= gate_up.shape[0] <= 16384
        and gate_up.shape[1] in (640, 1280)
        and torch.cuda.get_device_capability(gate_up.device) == (12, 0)
    ):
        return False
    rows, width = gate_up.shape[0], gate_up.shape[1] // 2
    return output is None or (
        isinstance(output, torch.Tensor)
        and output.device == gate_up.device
        and output.dtype == torch.float8_e4m3fn
        and output.is_contiguous()
        and output.untyped_storage().data_ptr() != gate_up.untyped_storage().data_ptr()
        and tuple(output.shape) in ((rows, width), (triton.cdiv(rows, 16) * 16, width))
    )


def try_silu_mul_quant(gate_up, *, output=None):
    if not is_supported(gate_up, output):
        return None
    rows, width = gate_up.shape[0], gate_up.shape[1] // 2
    size = rows * width
    producer_block = 2048 if size <= 4 * 1024 * 1024 else 4096
    parts = triton.cdiv(size, producer_block)
    activated = torch.empty((rows, width), device=gate_up.device, dtype=torch.bfloat16)
    partial = torch.empty(parts, device=gate_up.device, dtype=torch.float32)
    scale = torch.empty(1, device=gate_up.device, dtype=torch.float32)
    if output is None:
        output = torch.empty(
            (triton.cdiv(rows, 16) * 16, width),
            device=gate_up.device,
            dtype=torch.float8_e4m3fn,
        )
    _silu_partial[(parts,)](
        gate_up,
        activated,
        partial,
        N=size,
        D=width,
        BLOCK=producer_block,
        num_warps=4 if producer_block == 2048 else 8,
        enable_fp_fusion=False,
    )
    _partial_quant[(triton.cdiv(output.numel(), 4096),)](
        activated,
        partial,
        output,
        scale,
        N=size,
        OUT=output.numel(),
        PARTS=parts,
        PTILE=triton.next_power_of_2(parts),
        BLOCK=4096,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, scale
