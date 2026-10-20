"""Materialize the residual and its normalized consumer in one FP8 HC producer."""

import math

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_exp.residual.gated_residual_norm_triton import (
    _norm_fp8_scale,
)


@triton.jit
def _hc_prefill_inject_norm(
    X, S, G, GAMMA, RESIDUAL, NORM, AMAX, EPS: tl.constexpr, COLLECT: tl.constexpr
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    column = tl.arange(0, 4096)
    mask = column < 2560
    offset = row * 10240 + group * 2560 + column
    x = tl.load(X + offset, mask, other=0.0).to(tl.float32)
    sub = tl.load(S + row * 2560 + column, mask, other=0.0).to(tl.float32)
    gate = tl.load(G + row * 4 + group).to(tl.float32)
    product = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[sub, gate],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    injection = product.to(tl.bfloat16).to(tl.float32)
    added = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, injection],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    residual = added.to(tl.bfloat16).to(tl.float32)
    tl.store(RESIDUAL + offset, residual, mask)
    residual = tl.where(mask, residual, 0.0)
    inverse = tl.rsqrt(tl.sum(residual * residual, 0) / 2560.0 + EPS)
    gamma = tl.load(GAMMA + group * 2560 + column, mask, other=0.0).to(tl.float32)
    result = (residual * inverse) * (1.0 + gamma)
    tl.store(NORM + offset, result, mask)
    if COLLECT:
        absolute = tl.abs(result.to(tl.bfloat16).to(tl.float32))
        absolute = tl.where(mask & (absolute == absolute), absolute, 0.0)
        tl.store(AMAX + row * 4 + group, tl.max(absolute, 0))


def _inject_grouped_norm(x, sublayer, gates, gamma, eps=1e-6):
    rows = x.numel() // 10240
    residual = torch.empty_like(x)
    norm = torch.empty_like(x)
    collect = 4093 <= rows <= 4096 or 8192 <= rows <= 12288
    partial = (
        torch.empty(rows * 4, device=x.device, dtype=torch.float32)
        if collect
        else residual
    )
    _hc_prefill_inject_norm[(rows, 4)](
        x,
        sublayer,
        gates,
        gamma,
        residual,
        norm,
        partial,
        EPS=eps,
        COLLECT=collect,
        num_warps=4,
        enable_fp_fusion=True,
    )
    scale = None
    if collect:
        scale = torch.empty(1, device=x.device, dtype=torch.float32)
        _norm_fp8_scale[(1,)](
            partial,
            scale,
            N=rows * 4,
            BLOCK=triton.next_power_of_2(rows * 4),
            num_warps=4 if rows < 4096 else 16,
        )
    return residual, norm, scale


def try_inject_grouped_norm(x, sublayer, gates, gamma, eps):
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.ndim >= 2
        and x.shape[-1] == 10240
        and sublayer.shape == (*x.shape[:-1], 2560)
        and gates.shape == (*x.shape[:-1], 4)
        and gamma.shape == (10240,)
        and 4093 <= x.numel() // 10240 <= 12288
        and all(
            t.dtype == torch.bfloat16
            and t.device == x.device
            and t.is_contiguous()
            and not t.requires_grad
            for t in (x, sublayer, gates, gamma)
        )
        and isinstance(eps, (float, int))
        and math.isfinite(eps)
        and eps >= 0
        and torch.cuda.get_device_capability(x.device) == (12, 0)
    ):
        return None
    return _inject_grouped_norm(x, sublayer, gates, gamma, eps)
