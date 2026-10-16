"""Preserve the stable TP sum's FP32 rank order in one CUDA kernel."""

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.triton_kernels.common.host_math import ceil_div
from triton.language.extra.cuda import libdevice

_VECTOR_REDUCE = os.environ.get("GLM53_TP_REDUCE_VECTOR", "0") == "1"


@triton.jit(do_not_specialize=["SIZE"])
def _sum_rank_order(
    parts,
    output,
    SIZE: tl.int64,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    ALIGN: tl.constexpr = 1,
):
    # ALIGN is 8 only after checking SIZE % 8 on the host. Encoding the
    # divisibility explicitly also survives identity-int64 argument lowering.
    size = (SIZE.to(tl.int64) // ALIGN) * ALIGN
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    value = tl.full((BLOCK,), 0.0, tl.float32)
    for rank in tl.static_range(WORLD):
        # CUDA's native BF16 conversions can flush subnormal values. Torch's
        # BF16 -> float cast preserves their bit pattern.
        raw = tl.load(parts + rank * size + i, i < size, 0)
        bits = raw.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
        part = bits.to(tl.float32, bitcast=True)
        # Explicit rounded addition prevents reassociation of cancellation.
        value = libdevice.add_rn(value, part)
    # Round finite floats to BF16 with ties to even, including subnormals.
    bits = value.to(tl.uint32, bitcast=True)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).to(tl.uint16)
    rounded = tl.where(
        value != value,
        value.to(tl.bfloat16).to(tl.uint16, bitcast=True),
        rounded,
    )
    tl.store(output + i, rounded.to(tl.bfloat16, bitcast=True), i < size)


@lru_cache(maxsize=128)
def _rank_sum_runner(device, size, world, alignment=1):
    # Keep compiled code only; the public runner reads the current CUDA stream.
    grid = (ceil_div(size, 1024), 1, 1)
    kernel = _sum_rank_order.warmup(
        torch.bfloat16,
        torch.bfloat16,
        size,
        world,
        1024,
        alignment,
        grid=grid,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return kernel[grid]


def sum_bf16_rank_order(parts: torch.Tensor, output: torch.Tensor):
    size, world = output.numel(), parts.shape[0]
    if not size:
        return
    alignment = 8 if _VECTOR_REDUCE and size % 8 == 0 else 1
    if (
        parts.dtype == output.dtype == torch.bfloat16
        and parts.data_ptr() % 16 == output.data_ptr() % 16 == 0
        and parts.device.index == torch.cuda.current_device()
        and not torch.cuda.is_current_stream_capturing()
    ):
        _rank_sum_runner(parts.device.index, size, world, alignment)(
            parts, output, size, world, 1024, alignment
        )
        return
    # Dtype-only warmup assumes aligned pointers. Preserve JIT specialization
    # for unaligned views, other current devices, and graph capture.
    _sum_rank_order[(ceil_div(output.numel(), 1024),)](
        parts,
        output,
        output.numel(),
        parts.shape[0],
        1024,
        alignment,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
