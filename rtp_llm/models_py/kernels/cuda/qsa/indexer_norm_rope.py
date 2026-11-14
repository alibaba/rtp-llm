"""Exact indexer query norm and BF16 partial RoPE consumer for SM120."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(
    X,
    Gamma,
    Cos,
    Sin,
    Out,
    HEADS: tl.constexpr,
    STRIDE: tl.constexpr,
    EPS: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, 32)
    col = lane * 4
    offset = row * STRIDE + head * 128
    a = tl.load(X + offset + col).to(tl.float32)
    b = tl.load(X + offset + col + 1).to(tl.float32)
    c = tl.load(X + offset + col + 2).to(tl.float32)
    d = tl.load(X + offset + col + 3).to(tl.float32)
    inv = tl.rsqrt(tl.sum(((a * a + b * b) + c * c) + d * d, 0) / 128.0 + EPS)
    for j in tl.static_range(4):
        i = col + j
        gain = 1.0 + tl.load(Gamma + i).to(tl.float32)
        x = tl.load(X + offset + i).to(tl.float32)
        normalized = ((x * inv) * gain).to(tl.bfloat16).to(tl.float32)
        partner = tl.where(i < 32, i + 32, i - 32)
        partner = tl.where(i < 64, partner, i)
        other = tl.load(X + offset + partner).to(tl.float32)
        other_gain = 1.0 + tl.load(Gamma + partner).to(tl.float32)
        rotated = ((other * inv) * other_gain).to(tl.bfloat16).to(tl.float32)
        cosine = tl.load(Cos + row * 64 + i, i < 64, other=1.0).to(tl.float32)
        sine = tl.load(Sin + row * 64 + i, i < 64, other=0.0).to(tl.float32)
        left = (normalized * cosine).to(tl.bfloat16).to(tl.float32)
        right = (
            (rotated * tl.where(i < 32, -1.0, 1.0) * sine)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        value = tl.where(
            i < 64, (left + right).to(tl.bfloat16), normalized.to(tl.bfloat16)
        )
        tl.store(Out + (row * HEADS + head) * 128 + i, value)


def try_indexer_norm_rope(q, cos, sin, gamma, eps):
    if not (
        q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) == (12, 0)
        and q.dtype == cos.dtype == sin.dtype == torch.bfloat16
        and q.ndim in (3, 4)
        and q.shape[-1] == 128
        and q.stride(-1) == 1
        and q.stride(-2) == 128
        and (q.ndim == 3 or q.stride(0) == q.shape[1] * q.stride(1))
        and cos.is_contiguous()
        and sin.is_contiguous()
        and cos.shape == sin.shape
        and cos.numel() == q.numel() // (q.shape[-2] * 128) * 64
        and gamma.shape == (128,)
        and gamma.is_contiguous()
        and gamma.dtype in (torch.float32, torch.bfloat16)
        and q.device == gamma.device == cos.device == sin.device
    ):
        return None
    rows = q.numel() // (q.shape[-2] * 128)
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _norm_rope[(rows, q.shape[-2])](
        q,
        gamma,
        cos,
        sin,
        out,
        HEADS=q.shape[-2],
        STRIDE=q.stride(-3),
        EPS=eps,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out
