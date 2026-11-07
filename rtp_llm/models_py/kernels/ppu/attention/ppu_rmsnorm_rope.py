"""DSV4 RMSNorm/RoPE with the checkpoint reference's BF16 boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(
    X, W, F, O, MEAN, D: tl.constexpr, RD: tl.constexpr, H: tl.constexpr,
    WEIGHTED: tl.constexpr, EPS: tl.constexpr, INVERSE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, D)
    x = tl.load(X + row * D + cols).to(tl.float32)
    if WEIGHTED:
        inv = tl.rsqrt(tl.load(MEAN + row).to(tl.float32) + EPS)
        y = x * inv * tl.load(W + cols).to(tl.float32)
    else:
        mean = tl.load(MEAN + row).to(tl.float32)
        var = (mean + EPS).to(tl.bfloat16).to(tl.float32)
        inv = tl.rsqrt(var).to(tl.bfloat16).to(tl.float32)
        y = x * inv
    y = y.to(tl.bfloat16).to(tl.float32)
    tl.store(O + row * D + cols, y, cols < D - RD)
    pair = tl.arange(0, RD // 2)
    rc = D - RD + pair * 2
    ic = rc + 1
    real = tl.load(X + row * D + rc).to(tl.float32) * inv
    imag = tl.load(X + row * D + ic).to(tl.float32) * inv
    if WEIGHTED:
        real = real * tl.load(W + rc).to(tl.float32)
        imag = imag * tl.load(W + ic).to(tl.float32)
    # RoPE consumes the rounded RMSNorm output for both Q and KV.
    real = real.to(tl.bfloat16).to(tl.float32)
    imag = imag.to(tl.bfloat16).to(tl.float32)
    cosine = tl.load(F + row // H * RD + pair * 2)
    sine = tl.load(F + row // H * RD + pair * 2 + 1)
    if INVERSE:
        sine = -sine
    # Preserve the native complex multiply contraction order.
    tl.store(O + row * D + rc, tl.fma(real, cosine, -(imag * sine)))
    tl.store(O + row * D + ic, tl.fma(imag, cosine, real * sine))


def fused_rmsnorm_rope(
    x, weight, freqs_cis, rope_head_dim, *, eps=1e-6, inverse=False,
    out=None, inplace=False, group_heads=None,
):
    assert x.is_cuda and x.is_contiguous() and (x.dtype == torch.bfloat16) and (x.shape[-1] == 512)
    assert freqs_cis.dtype == torch.complex64 and freqs_cis.device == x.device
    assert rope_head_dim == 64
    if weight is not None:
        assert weight.shape == (512,) and weight.device == x.device and weight.is_contiguous()
        assert weight.dtype in (torch.bfloat16, torch.float32)
    assert not (inplace and out is not None)
    out = x if inplace else torch.empty_like(x) if out is None else out
    assert out.shape == x.shape and out.dtype == x.dtype and (out.device == x.device) and out.is_contiguous()
    n = x.numel() // x.shape[-1]
    if n == 0:
        return out
    f = freqs_cis.reshape(-1, rope_head_dim // 2).contiguous()
    assert n % f.shape[0] == 0
    # Match Torch's reduction order at BF16 midpoints; FP64 can round differently.
    mean = (x.float() if weight is not None else x).square().mean(-1)
    if n:
        _norm_rope[(n,)](
            x, weight, f.view(torch.float32), out, mean, 512, rope_head_dim,
            n // f.shape[0], weight is not None, eps, inverse,
            num_warps=4, enable_fp_fusion=False,
        )
    return out
