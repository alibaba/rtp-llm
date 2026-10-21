# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Alibaba Group Holding Limited
"""SM120 FP32 indexer weighting, head reduction and causal masking."""

import torch
import triton
import triton.language as tl


@triton.jit
def _multiply_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _postprocess(
    Dots,
    Weights,
    Visible,
    Output,
    K: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    keys = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    # Match the CUDA sum over 32 strided heads with four accumulators.
    # Scalar RN operations preserve separate multiplication/addition rounding.
    a0 = tl.full((BLOCK,), 0, tl.float32)
    a1 = tl.full((BLOCK,), 0, tl.float32)
    a2 = tl.full((BLOCK,), 0, tl.float32)
    a3 = tl.full((BLOCK,), 0, tl.float32)
    for base in tl.static_range(0, 32, 4):
        d0 = tl.load(Dots + row * DS0 + base * DS1 + keys * DS2, keys < K, 0)
        d1 = tl.load(Dots + row * DS0 + (base + 1) * DS1 + keys * DS2, keys < K, 0)
        d2 = tl.load(Dots + row * DS0 + (base + 2) * DS1 + keys * DS2, keys < K, 0)
        d3 = tl.load(Dots + row * DS0 + (base + 3) * DS1 + keys * DS2, keys < K, 0)
        w0 = tl.load(Weights + row * WS0 + base * WS1)
        w1 = tl.load(Weights + row * WS0 + (base + 1) * WS1)
        w2 = tl.load(Weights + row * WS0 + (base + 2) * WS1)
        w3 = tl.load(Weights + row * WS0 + (base + 3) * WS1)
        a0 = _add_rn(a0, _multiply_rn(tl.where(d0 < 0, 0.0, d0), w0))
        a1 = _add_rn(a1, _multiply_rn(tl.where(d1 < 0, 0.0, d1), w1))
        a2 = _add_rn(a2, _multiply_rn(tl.where(d2 < 0, 0.0, d2), w2))
        a3 = _add_rn(a3, _multiply_rn(tl.where(d3 < 0, 0.0, d3), w3))
    visible = tl.load(Visible + row)
    score = tl.where(
        keys < visible, _add_rn(_add_rn(_add_rn(a0, a1), a2), a3), float("-inf")
    )
    tl.store(Output + row * K + keys, score, keys < K)


def try_postprocess(dots, weights, visible):
    """Return fused scores for qualified layouts, or None for the eager path."""
    if (
        dots.device.type != "cuda"
        or dots.ndim != 3
        or dots.shape[1] != 32
        or not 1 < dots.shape[2] < 2**31
        or dots.dtype != torch.float32
        or not dots.is_contiguous()
        or weights.shape != dots.shape[:2]
        or weights.dtype != torch.float32
        or weights.device != dots.device
        or visible.shape != (dots.shape[0],)
        or visible.dtype not in (torch.int32, torch.int64)
        or visible.device != dots.device
        or not visible.is_contiguous()
        or (torch.is_grad_enabled() and (dots.requires_grad or weights.requires_grad))
    ):
        return None
    if torch.cuda.get_device_capability(dots.device) != (12, 0):
        return None
    out = torch.empty(
        (dots.shape[0], dots.shape[2]), dtype=torch.float32, device=dots.device
    )
    if dots.shape[0]:
        _postprocess[(dots.shape[0], triton.cdiv(dots.shape[2], 256))](
            dots,
            weights,
            visible,
            out,
            dots.shape[2],
            *dots.stride(),
            *weights.stride(),
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
