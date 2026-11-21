"""Fuse DeepSeek V4's BF16 Q normalization and partial rotary embedding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _q_norm_rope(
    X, F, Y, D: tl.constexpr, RD: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    off = tl.arange(0, D)
    x = tl.load(X + row * D + off).to(tl.float32)
    # Official Q normalization is BF16 torch arithmetic, unlike weighted
    # RMSNorm. Preserve each materialized rounding boundary inside one launch.
    square = (x * x).to(tl.bfloat16).to(tl.float32)
    mean = (tl.sum(square, 0) / D).to(tl.bfloat16).to(tl.float32)
    var = (mean + EPS).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(var).to(tl.bfloat16).to(tl.float32)
    y = (x * inv).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + off, y, off < D - RD)

    pair = tl.arange(0, RD // 2)
    re = D - RD + 2 * pair
    im = re + 1
    real = (
        (tl.load(X + row * D + re).to(tl.float32) * inv).to(tl.bfloat16).to(tl.float32)
    )
    imag = (
        (tl.load(X + row * D + im).to(tl.float32) * inv).to(tl.bfloat16).to(tl.float32)
    )
    cos = tl.load(F + (row // H) * RD + 2 * pair)
    sin = tl.load(F + (row // H) * RD + 2 * pair + 1)
    # Match the complex multiply used by the official PPU torch reference.
    tl.store(Y + row * D + re, tl.fma(real, cos, -imag * sin))
    tl.store(Y + row * D + im, tl.fma(imag, cos, real * sin))


def q_norm_rope(x, freqs_cis, rope_head_dim=64, *, eps=1e-6, out=None):
    """Normalize BF16 ``[..., heads, 512]`` Q with one frequency per token."""
    if x.dtype != torch.bfloat16 or x.ndim < 3 or x.shape[-1] != 512:
        raise ValueError("DSV4 Q must be BF16 [..., heads, 512]")
    if rope_head_dim != 64 or x.shape[-2] <= 0:
        raise ValueError("DSV4 Q requires positive heads and 64 rotary dimensions")
    if not x.is_contiguous():
        raise ValueError("DSV4 Q must be contiguous")
    if (
        freqs_cis.dtype != torch.complex64
        or not freqs_cis.is_contiguous()
        or freqs_cis.device != x.device
        or freqs_cis.numel() != x.numel() // (x.shape[-2] * 512) * 32
    ):
        raise ValueError(
            "Q frequencies must be contiguous complex64, one row per token"
        )
    if out is not None and (
        out.shape != x.shape
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("Q output must match the contiguous input layout")
    if not x.is_cuda:
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        z = torch.view_as_complex(y[..., -64:].float().unflatten(-1, (-1, 2)))
        z = z.reshape(-1, x.shape[-2], 32) * freqs_cis.reshape(-1, 1, 32)
        y[..., -64:] = torch.view_as_real(z).reshape_as(y[..., -64:])
        if out is not None:
            out.copy_(y)
            return out
        return y
    out = torch.empty_like(x) if out is None else out
    if x.numel():
        _q_norm_rope[(x.numel() // 512,)](
            x,
            torch.view_as_real(freqs_cis),
            out,
            512,
            64,
            x.shape[-2],
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
