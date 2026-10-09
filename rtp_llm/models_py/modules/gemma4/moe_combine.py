"""Experimental MoE combine with BF16 weighted stores and original slot order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _contribution(HIDDEN, SCALES, INVERSE, token, col, slot: tl.constexpr,
                  ROWS, WIDTH: tl.constexpr, HS: tl.constexpr, TOP_K: tl.constexpr):
    original = token * TOP_K + slot
    sorted_row = tl.load(INVERSE + original, mask=token < ROWS, other=0)
    h = tl.load(HIDDEN + sorted_row * HS + col, mask=token < ROWS, other=0).to(tl.float32)
    scale = tl.load(SCALES + original, mask=token < ROWS, other=0).to(tl.float32)
    return (h * scale).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _combine_kernel(HIDDEN, SCALES, INVERSE, OUT, ROWS,
                    WIDTH: tl.constexpr, HS: tl.constexpr,
                    TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token, col = index // WIDTH, index % WIDTH
    acc0 = tl.full((BLOCK,), 0, tl.float32)
    acc1 = tl.full((BLOCK,), 0, tl.float32)
    acc2 = tl.full((BLOCK,), 0, tl.float32)
    acc3 = tl.full((BLOCK,), 0, tl.float32)
    for group in tl.static_range((TOP_K + 3) // 4):
        if group * 4 < TOP_K:
            acc0 = acc0 + _contribution(HIDDEN, SCALES, INVERSE, token, col, group * 4, ROWS, WIDTH, HS, TOP_K)
        if group * 4 + 1 < TOP_K:
            acc1 = acc1 + _contribution(HIDDEN, SCALES, INVERSE, token, col, group * 4 + 1, ROWS, WIDTH, HS, TOP_K)
        if group * 4 + 2 < TOP_K:
            acc2 = acc2 + _contribution(HIDDEN, SCALES, INVERSE, token, col, group * 4 + 2, ROWS, WIDTH, HS, TOP_K)
        if group * 4 + 3 < TOP_K:
            acc3 = acc3 + _contribution(HIDDEN, SCALES, INVERSE, token, col, group * 4 + 3, ROWS, WIDTH, HS, TOP_K)
    total = ((acc0 + acc1) + acc2) + acc3
    tl.store(OUT + index, total, mask=token < ROWS)


def combine(hidden, inverse, scales):
    if (
        not hidden.is_cuda or hidden.dtype != torch.bfloat16 or hidden.dim() != 2
        or hidden.numel() == 0 or hidden.stride(1) != 1 or hidden.stride(0) <= 0
        or hidden.shape[1] < 128 or hidden.shape[1] % 4
        or scales.device != hidden.device or scales.dtype != torch.bfloat16
        or scales.dim() != 2 or not scales.is_contiguous() or scales.shape[0] <= 0
        or not 1 <= scales.shape[1] <= 32
        or hidden.shape[0] != scales.numel()
        or inverse.device != hidden.device or inverse.dtype != torch.int64
        or inverse.dim() != 1 or not inverse.is_contiguous()
        or inverse.numel() != scales.numel()
    ):
        return None
    rows, width, top_k = scales.shape[0], hidden.shape[1], scales.shape[1]
    output = torch.empty((rows, width), device=hidden.device, dtype=hidden.dtype)
    _combine_kernel[(triton.cdiv(rows * width, 1024),)](
        hidden, scales, inverse, output, rows, width, hidden.stride(0), top_k, 1024,
        num_warps=4, enable_fp_fusion=False,
    )
    return output
