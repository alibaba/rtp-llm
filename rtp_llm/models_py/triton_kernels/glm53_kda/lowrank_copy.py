"""Copy a pair of strided BF16 low-rank projections with one CUDA launch."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS"])
def _copy_pair(
    forget,
    gate,
    out_forget,
    out_gate,
    ROWS: tl.int64,
    FORGET_STRIDE: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024).to(tl.int64)
    row, column = index // 128, index % 128
    valid = row < ROWS
    tl.store(
        out_forget + index,
        tl.load(forget + row * FORGET_STRIDE + column, valid, 0),
        valid,
    )
    tl.store(
        out_gate + index,
        tl.load(gate + row * GATE_STRIDE + column, valid, 0),
        valid,
    )


@lru_cache(maxsize=128)
def _launcher(device, rows, forget_stride, gate_stride):
    # Device is part of the key because compiled CUDA handles belong to a device.
    # warmup(dtype) specializes aligned pointers without retaining input tensors.
    # ROWS is runtime int64: changing batch size reuses the same compiled kernel.
    grid = ((rows + 7) // 8, 1, 1)
    kernel = _copy_pair.warmup(
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        rows,
        forget_stride,
        gate_stride,
        grid=grid,
        num_warps=4,
    )
    # The public runner resolves the current CUDA stream on each invocation.
    return kernel[grid]


def is_supported(forget, gate):
    return (
        forget.is_cuda
        and torch.version.hip is None
        and forget.device == gate.device
        and forget.dtype == gate.dtype == torch.bfloat16
        and forget.ndim == gate.ndim == 2
        and forget.shape == gate.shape
        and forget.shape[1] == 128
        and forget.shape[0] > 1
        and forget.stride(1) == gate.stride(1) == 1
        and not forget.requires_grad
        and not gate.requires_grad
        and forget.data_ptr() % 16 == gate.data_ptr() % 16 == 0
        and forget.device.index == torch.cuda.current_device()
        and not torch.cuda.is_current_stream_capturing()
    )


def copy_lowrank_pair(forget, gate):
    """Return contiguous copies, or None so the caller keeps its original path."""
    if not is_supported(forget, gate):
        return None
    rows = forget.shape[0]
    forget_stride, gate_stride = forget.stride(0), gate.stride(0)
    run = _launcher(forget.device.index, rows, forget_stride, gate_stride)
    out_forget = torch.empty_like(forget, memory_format=torch.contiguous_format)
    out_gate = torch.empty_like(gate, memory_format=torch.contiguous_format)
    run(forget, gate, out_forget, out_gate, rows, forget_stride, gate_stride)
    return out_forget, out_gate
