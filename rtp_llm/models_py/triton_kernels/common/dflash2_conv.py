# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 dynamic grouped depthwise convolution on CUDA and ROCm.

Rows are request-major fixed-width query blocks (anchor plus mask tokens).
Tap zero reads the current row; every other tap is causal and is zero at the
request boundary. Delta has one coefficient per group while the learned base
has one coefficient per channel. Both are accumulated in FP32.

Mathematical contract follows the Apache-2.0 DFlash2 implementation in
vllm/model_executor/models/qwen3_dflash2.py (vLLM PR #52816).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_conv_kernel(
    Hidden,
    Delta,
    Base,
    Out,
    H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TAPS: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    hidden_row_stride: tl.constexpr,
    delta_row_stride: tl.constexpr,
    delta_tap_stride: tl.constexpr,
    delta_group_stride: tl.constexpr,
    base_tap_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = channel < H
    group = channel // GROUP_SIZE
    value = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(TAPS):
        coefficient = tl.load(Base + tap * base_tap_stride + channel, live, other=0).to(
            tl.float32
        )
        coefficient += tl.load(
            Delta
            + row * delta_row_stride
            + tap * delta_tap_stride
            + group * delta_group_stride,
            live,
            other=0,
        ).to(tl.float32)
        source = tl.load(
            Hidden + (row - tap) * hidden_row_stride + channel,
            live & (row % QUERY_WIDTH >= tap),
            other=0,
        ).to(tl.float32)
        value += coefficient * source
    tl.store(Out + row * H + channel, value, live)


def grouped_conv(
    hidden: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    query_width: int,
    group_size: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one convolution side; all tensors must be on the same GPU.

    ``delta`` may be a strided view of [rows, 2, taps, groups], which lets
    prepare/finish reuse the two halves of a single coefficient projection.
    No tensor data is read on the host, including during graph capture.
    """
    if hidden.ndim != 2 or delta.ndim != 3 or base.ndim != 2:
        raise ValueError(
            "DFlash2 convolution expects hidden[N,H], delta[N,T,G], base[T,H]"
        )
    rows, hidden_size = hidden.shape
    taps = base.shape[0]
    if (
        query_width <= 0
        or rows % query_width
        or group_size <= 0
        or hidden_size <= 0
        or hidden_size % group_size
        or taps <= 0
        or base.shape[1] != hidden_size
        or delta.shape != (rows, taps, hidden_size // group_size)
    ):
        raise ValueError("invalid DFlash2 convolution block/group/weight shape")
    if hidden.device.type != "cuda" or any(
        tensor.device != hidden.device for tensor in (delta, base)
    ):
        raise ValueError(
            "DFlash2 convolution requires CUDA or HIP tensors on one device"
        )
    if hidden.dtype not in (torch.bfloat16, torch.float16, torch.float32) or any(
        tensor.dtype != hidden.dtype for tensor in (delta, base)
    ):
        raise ValueError("DFlash2 convolution requires matching floating point dtypes")
    if hidden.stride(-1) != 1 or base.stride(-1) != 1:
        raise ValueError("DFlash2 convolution requires contiguous hidden channels")
    if out is None:
        out = torch.empty((rows, hidden_size), device=hidden.device, dtype=hidden.dtype)
    elif (
        out.shape != hidden.shape
        or out.device != hidden.device
        or out.dtype != hidden.dtype
        or not out.is_contiguous()
    ):
        raise ValueError(
            "DFlash2 convolution output must be a matching contiguous tensor"
        )
    if rows and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (hidden, delta, base)
    ):
        raise ValueError("DFlash2 convolution output must not alias an input")
    if rows:
        _grouped_conv_kernel[(rows, triton.cdiv(hidden_size, 256))](
            hidden,
            delta,
            base,
            out,
            hidden_size,
            group_size,
            taps,
            query_width,
            hidden.stride(0),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            base.stride(0),
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["grouped_conv"]
