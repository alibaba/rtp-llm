"""One launch for the shared MLP's clamped SiLU and product."""

import math
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_COMPILED_DISPATCH = (
    os.environ.get("GLM53_CLAMPED_SWIGLU_COMPILED_DISPATCH", "0") == "1"
)


@triton.jit(do_not_specialize=["SIZE"])
def _clamped_swiglu(
    x, output, SIZE, WIDTH: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    source = (index // WIDTH) * (2 * WIDTH) + index % WIDTH
    gate = tl.load(x + source, index < SIZE, 0).to(tl.float32)
    up = tl.load(x + source + WIDTH, index < SIZE, 0).to(tl.float32)
    gate = tl.where(gate == gate, tl.minimum(gate, LIMIT), gate)
    up = tl.where(up == up, tl.minimum(tl.maximum(up, -LIMIT), LIMIT), up)
    # Eager clamp and SiLU each materialize BF16 before the next operation.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    activated = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate))
    activated = activated.to(tl.bfloat16).to(tl.float32)
    tl.store(output + index, activated * up, index < SIZE)


@lru_cache(maxsize=128)
def _runner(device, width, limit, blocks):
    # Cache code only. An odd runtime SIZE avoids divisibility specialization;
    # the wrapper bounds its int32 ABI and retains the original unaligned path.
    grid = (blocks, 1, 1)
    kernel = _clamped_swiglu.warmup(
        torch.bfloat16,
        torch.bfloat16,
        13,
        width,
        limit,
        256,
        grid=grid,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return kernel[grid]


def is_supported(x: torch.Tensor, limit: float) -> bool:
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[1] > 0
        and x.shape[1] % 2 == 0
        and not x.requires_grad
        and math.isfinite(limit)
        and limit > 0
    )


def clamped_swiglu(x: torch.Tensor, limit: float):
    """Return None for unsupported inputs so the caller retains its old path."""
    if not is_supported(x, limit):
        return None
    width = x.shape[1] // 2
    output = x.new_empty((x.shape[0], width))
    count = output.numel()
    if count:
        blocks = (count + 255) // 256
        if (
            _COMPILED_DISPATCH
            and count < 2**31
            and x.data_ptr() % 16 == 0
            and x.device.index == torch.cuda.current_device()
            and not torch.cuda.is_current_stream_capturing()
        ):
            _runner(x.device.index, width, limit, blocks)(
                x, output, count, width, limit, 256
            )
            return output
        _clamped_swiglu[(blocks,)](
            x,
            output,
            count,
            width,
            limit,
            256,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return output
