"""Preserve the reference HC head's ordered FP32 products and BF16 store."""

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_head_sum(X, P, Y, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    result = tl.load(X + row * 4 * D + col, col < D, 0).to(tl.float32) * (
        tl.load(P + row * 4) + EPS
    )
    for lane in tl.static_range(1, 4):
        value = tl.load(X + row * 4 * D + lane * D + col, col < D, 0).to(tl.float32)
        weight = tl.load(P + row * 4 + lane) + EPS
        result = result + value * weight
    tl.store(Y + row * D + col, result, col < D)


def is_supported(x, pre):
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim in (3, 4)
        and x.shape[-2:] == (4, 4096)
        and x.is_contiguous()
        and pre.shape == x.shape[:-1]
        and pre.dtype == torch.float32
        and pre.device == x.device
        and pre.is_contiguous()
    )


def hc_head_sum(x, pre, eps):
    if not is_supported(x, pre):
        raise ValueError("PPU HC head sum requires BF16 Flash inputs and FP32 gates")
    rows = x.numel() // (4 * 4096)
    output = torch.empty((*x.shape[:-2], 4096), device=x.device, dtype=x.dtype)
    if rows:
        # PyTorch materializes each product and adds lanes 0, 1, 2, 3 in order.
        # Neither a tree reduction nor FMA preserves its BF16 result bits.
        _hc_head_sum[(rows, 16)](
            x,
            pre,
            output,
            4096,
            float(eps),
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
