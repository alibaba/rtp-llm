"""Qwen4 sigmoid-gated GDN norm with multiple head reductions per CTA."""

import torch
import triton
import triton.language as tl

try:
    from rtp_llm.models_py.kernels.cuda.gdn_sm120.norm_gluon import flat_head_norm
    from rtp_llm.models_py.kernels.cuda.gdn_sm120.norm_quant_gluon import norm_quant
except ImportError:
    # Older platform Triton versions retain the existing implementation.
    flat_head_norm = None
    norm_quant = None


@triton.jit
def _gdn_prefill_gated_norm(
    X,
    Z,
    W,
    Y,
    N: tl.constexpr,
    ZX: tl.constexpr,
    XX: tl.constexpr,
    EPS: tl.constexpr,
    GROUPS: tl.constexpr,
    GH: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * GH + tl.arange(0, GH)
    col = tl.arange(0, 128)
    offset = head[:, None] * 128 + col[None, :]
    value = tl.load(X + row * XX + offset, head[:, None] < GROUPS, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(value * value, 1) / 128.0
    inverse = 1.0 / tl.sqrt(variance + EPS)
    weight = tl.load(W + col).to(tl.float32)
    gate = tl.load(Z + row * ZX + offset, head[:, None] < GROUPS, other=0.0).to(
        tl.float32
    )
    result = (value * inverse[:, None]) * weight[None, :]
    result *= tl.sigmoid(gate)
    tl.store(Y + row * N + offset, result, head[:, None] < GROUPS)


def _prefill_gated_norm(x, z, weight, eps=1e-6, groups_per_cta=4):
    rows, width = x.shape
    if flat_head_norm is not None and width in (1536, 3072):
        return flat_head_norm(x, z, weight, eps)
    groups = width // 128
    out = torch.empty_like(x)
    _gdn_prefill_gated_norm[(rows, triton.cdiv(groups, groups_per_cta))](
        x,
        z,
        weight,
        out,
        N=width,
        ZX=z.stride(0),
        XX=x.stride(0),
        EPS=eps,
        GROUPS=groups,
        GH=groups_per_cta,
        num_warps=groups_per_cta,
    )
    return out
