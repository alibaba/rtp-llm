"""Reference FP32 HC projection for DSpARK precision validation."""

import torch
import torch.nn.functional as F


def fp32_hc_prenorm_partials(x, weight, *, rowwise=False):
    if x.ndim != 2 or x.shape[1] != 16384 or weight.shape != (24, 16384):
        raise ValueError("FP32 HC requires [M,16384] input and [24,16384] weight")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise TypeError("FP32 HC requires BF16 input and FP32 weight")
    if any(not t.is_cuda or t.device != x.device or not t.is_contiguous() for t in (x, weight)):
        raise ValueError("FP32 HC inputs must be contiguous on one PPU")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("FP32 HC requires TF32 disabled")
    xf = x.float()
    if rowwise and x.shape[0]:
        dots = torch.cat([F.linear(row[None], weight) for row in xf])
        squares = torch.cat([row[None].square().sum(-1) for row in xf])
    else:
        dots = F.linear(xf, weight)
        squares = xf.square().sum(-1)
    return dots.unsqueeze(0), squares.unsqueeze(0)


def fp32_hc_prenorm_gemm(x, weight, out, squares, requested_splits=None):
    del requested_splits
    dots, sums = fp32_hc_prenorm_partials(x, weight)
    if out.shape != dots.shape or squares.shape != sums.shape:
        raise ValueError("FP32 HC outputs must have shapes [1,M,24] and [1,M]")
    if any(t.dtype != torch.float32 or t.device != x.device for t in (out, squares)):
        raise ValueError("FP32 HC outputs must be FP32 on the input device")
    out.copy_(dots)
    squares.copy_(sums)
    return out


def fp32_hc_rms_norm(x, weight, eps, *, rowwise=False):
    if rowwise and x.numel():
        # Native verification normalizes each committed/proposed token alone.
        # PPU's batched reduction can round differently at BF16 midpoints.
        rows = x.reshape(-1, x.shape[-1])
        return torch.cat([
            fp32_hc_rms_norm(row[None], weight, eps) for row in rows
        ]).view_as(x)
    xf = x.float()
    return (weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps))).to(x.dtype)
