"""SM120 text indexer RoPE with the existing BF16 rounding boundaries."""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _text_rope(
    POS,
    INV_FREQ,
    COS,
    SIN,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    frequency = tl.arange(0, 32)
    position = tl.load(POS + row * STRIDE, row < ROWS, other=0).to(tl.float32)
    inverse = tl.load(INV_FREQ + frequency)
    angle = (position[:, None] * INV_SCALE) * inverse[None, :]
    cosine = libdevice.cos(angle)
    sine = libdevice.sin(angle)
    offset = row[:, None] * 64 + frequency[None, :]
    tl.store(COS + offset, cosine, row[:, None] < ROWS)
    tl.store(COS + offset + 32, cosine, row[:, None] < ROWS)
    tl.store(SIN + offset, sine, row[:, None] < ROWS)
    tl.store(SIN + offset + 32, sine, row[:, None] < ROWS)


def try_build_text_rope(positions, config, style, *, dtype, device, inv_freq=None):
    """Return None outside the validated canonical text geometry.

    The caller retains the full validation/error path for unsupported configs.
    Canonical text has equal T/H/W axes, so its MRoPE frequencies equal Base
    RoPE without materializing the repeated position matrix or axis indices.
    """
    if not (
        positions.is_cuda
        and positions.device == device
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device) == (12, 0)
        and positions.dtype in (torch.int32, torch.int64)
        and positions.ndim == 1
        and positions.numel() > 0
        and positions.stride(0) > 0
        and dtype == torch.bfloat16
        and int(config.dim) == 64
        and bool(getattr(config, "indexer_is_neox_style", True))
        and isinstance(inv_freq, torch.Tensor)
        and inv_freq.device == device
        and inv_freq.dtype == torch.float32
        and inv_freq.shape == (32,)
        and inv_freq.is_contiguous()
    ):
        return None
    factor = int(config.index_factor)
    if style == "Base":
        if factor not in (1, 3):
            return None
    elif style == "Mrope":
        sections = (
            int(config.mrope_dim1),
            int(config.mrope_dim2),
            int(config.mrope_dim3),
        )
        if not (
            factor == 3
            and bool(config.mrope_interleaved)
            and min(sections) >= 0
            and sum(sections) == 32
            and sections[1] <= len(range(1, 32, 3))
            and sections[2] <= len(range(2, 32, 3))
        ):
            return None
    else:
        return None
    base, scale = float(config.base), float(getattr(config, "scale", 1.0))
    if not (math.isfinite(base) and base > 0 and math.isfinite(scale) and scale > 0):
        return None
    cos = torch.empty((positions.numel(), 64), dtype=dtype, device=device)
    sin = torch.empty_like(cos)
    _text_rope[(triton.cdiv(positions.numel(), 4),)](
        positions,
        inv_freq,
        cos,
        sin,
        ROWS=positions.numel(),
        STRIDE=positions.stride(0),
        INV_SCALE=1.0 / scale,
        BLOCK=4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return cos, sin


@triton.jit
def _partial_rope(X, COS, SIN, OUT, HEADS: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, 128)
    paired = tl.where(column < 32, column + 32, column - 32)
    value = tl.load(X + row * 128 + column)
    other = tl.load(X + row * 128 + paired, column < 64, other=0)
    cosine = tl.load(COS + row // HEADS * 64 + column, column < 64, other=0)
    sine = tl.load(SIN + row // HEADS * 64 + column, column < 64, other=0)
    rotated = tl.where(column < 32, -other.to(tl.float32), other.to(tl.float32))
    # Eager BF16 materializes each product before adding them. Preserve both
    # products and the final cast, including when the fused kernel uses FP32.
    first = (value.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
    second = (rotated * sine.to(tl.float32)).to(tl.bfloat16)
    output = (first.to(tl.float32) + second.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * 128 + column, tl.where(column < 64, output, value))


def try_apply_partial_rope(x, cos, sin):
    if not (
        x.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(x.device) == (12, 0)
        and x.ndim in (3, 4)
        and x.shape[-1] == 128
        and x.numel() > 0
        and cos.shape == (*x.shape[:-2], 1, 64)
        and sin.shape == cos.shape
        and x.dtype == cos.dtype == sin.dtype == torch.bfloat16
        and x.device == cos.device == sin.device
        and x.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
        and not any(t.requires_grad for t in (x, cos, sin))
    ):
        return None
    out = torch.empty_like(x)
    _partial_rope[(x.numel() // 128,)](
        x, cos, sin, out, HEADS=x.shape[-2], num_warps=4, enable_fp_fusion=False
    )
    return out
