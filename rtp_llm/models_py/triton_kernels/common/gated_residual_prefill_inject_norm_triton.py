"""Materialize the residual and its normalized consumer in one HC kernel."""

import math

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.gated_residual_norm_triton import (
    _norm_fp8_scale,
)
from rtp_llm.models_py.utils.device_capability import (
    cached_device_capability,
    sm90_pdl_enabled,
)


@triton.jit
def _hc_prefill_inject_norm(
    X,
    S,
    G,
    GAMMA,
    RESIDUAL,
    NORM,
    AMAX,
    EPS: tl.constexpr,
    COLLECT: tl.constexpr,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    column = tl.arange(0, 4096)
    mask = column < 2560
    offset = row * 10240 + group * 2560 + column
    gamma = tl.load(GAMMA + group * 2560 + column, mask, other=0.0).to(tl.float32)
    x = tl.load(X + offset, mask, other=0.0).to(tl.float32)
    gate = tl.load(G + row * 4 + group).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    sub = tl.load(S + row * 2560 + column, mask, other=0.0).to(tl.float32)
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
    result = (residual * inverse) * (1.0 + gamma)
    if PDL and not COLLECT:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(NORM + offset, result, mask)
    if COLLECT:
        absolute = tl.abs(result.to(tl.bfloat16).to(tl.float32))
        absolute = tl.where(mask & (absolute == absolute), absolute, 0.0)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(AMAX + row * 4 + group, tl.max(absolute, 0))


def _inject_grouped_norm(x, sublayer, gates, gamma, eps=1e-6, *, collect_scale=True):
    rows = x.numel() // 10240
    residual = torch.empty_like(x)
    norm = torch.empty_like(x)
    sm90 = cached_device_capability(x.device) == (9, 0)
    collect = collect_scale and (
        (1 <= rows <= 32768) if sm90 else (1 <= rows <= 4096 or 8192 <= rows <= 12288)
    )
    pdl = sm90_pdl_enabled(x.device)
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
        PDL=pdl,
        num_warps=4,
        enable_fp_fusion=True,
        launch_pdl=pdl,
    )
    scale = None
    if collect:
        scale = torch.empty(1, device=x.device, dtype=torch.float32)
        block = min(65536, triton.next_power_of_2(rows * 4))
        _norm_fp8_scale[(1,)](
            partial,
            scale,
            N=rows * 4,
            BLOCK=block,
            PDL=pdl,
            num_warps=4 if rows < 4096 else 16,
            launch_pdl=pdl,
        )
    return residual, norm, scale


def try_inject_grouped_norm(x, sublayer, gates, gamma, eps, *, collect_scale=True):
    capability = cached_device_capability(x.device) if x.is_cuda else None
    rows = x.numel() // 10240 if x.ndim >= 2 and x.shape[-1] == 10240 else 0
    if capability == (12, 0):
        row_ok = 4093 <= rows <= 4096 or 8192 <= rows <= 12288
    elif capability == (9, 0):
        row_ok = 1 <= rows <= 32768
    else:
        row_ok = False
    if not (
        row_ok
        and torch.version.hip is None
        and x.ndim >= 2
        and x.shape[-1] == 10240
        and sublayer.shape == (*x.shape[:-1], 2560)
        and gates.shape == (*x.shape[:-1], 4)
        and gamma.shape == (10240,)
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
    ):
        return None
    return _inject_grouped_norm(x, sublayer, gates, gamma, eps, collect_scale=collect_scale)
