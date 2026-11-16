"""Materialize the residual and its normalized consumer in one HC producer."""

import math

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.gated_residual.norm import _norm_fp8_scale


@triton.jit
def _hc_prefill_inject_norm(
    X,
    S,
    G,
    GAMMA,
    RESIDUAL,
    NORM,
    AMAX,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    COLLECT: tl.constexpr,
):
    if HIDDEN == 8192:
        tile = tl.program_id(0)
        group = tile % 4
        row = (tile // 4).to(tl.int64)
    else:
        group = tl.program_id(0)
        row = tl.program_id(1)
    column = tl.arange(0, BLOCK)
    mask = column < HIDDEN
    offset = row * (4 * HIDDEN) + group * HIDDEN + column
    x = tl.load(X + offset, mask, other=0.0)
    sub = tl.load(S + row * HIDDEN + column, mask, other=0.0)
    gate = tl.load(G + row * 4 + group)
    if HIDDEN == 8192:
        # FP contraction is disabled for SM103. Keep both BF16 rounding points
        # visible to Triton so it can optimize the elementwise instructions.
        injection = (sub.to(tl.float32) * gate.to(tl.float32)).to(tl.bfloat16)
        residual_bf16 = (x.to(tl.float32) + injection.to(tl.float32)).to(tl.bfloat16)
        tl.store(RESIDUAL + offset, residual_bf16)
        residual = residual_bf16.to(tl.float32)
    else:
        product = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[sub.to(tl.float32), gate.to(tl.float32)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        injection = product.to(tl.bfloat16).to(tl.float32)
        added = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x.to(tl.float32), injection],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        residual = added.to(tl.bfloat16).to(tl.float32)
        tl.store(RESIDUAL + offset, residual, mask)
        residual = tl.where(mask, residual, 0.0)
    inverse = tl.rsqrt(tl.sum(residual * residual, 0) / HIDDEN + EPS)
    gamma = tl.load(GAMMA + group * HIDDEN + column, mask, other=0.0).to(tl.float32)
    result = (residual * inverse) * (1.0 + gamma)
    tl.store(NORM + offset, result, mask)
    if COLLECT:
        absolute = tl.abs(result.to(tl.bfloat16).to(tl.float32))
        absolute = tl.where(mask & (absolute == absolute), absolute, 0.0)
        tl.store(AMAX + row * 4 + group, tl.max(absolute, 0))


def _inject_grouped_norm(x, sublayer, gates, gamma, eps=1e-6):
    hidden = sublayer.shape[-1]
    rows = x.numel() // (4 * hidden)
    residual = torch.empty_like(x)
    norm = torch.empty_like(x)
    collect = hidden == 2560 and (
        7 <= rows <= 32 or 4093 <= rows <= 4096 or 8192 <= rows <= 16384
    )
    partial = (
        torch.empty(rows * 4, device=x.device, dtype=torch.float32)
        if collect
        else residual
    )
    # Adjacent groups share the sublayer row and traverse contiguous residuals.
    grid = (4 * rows,) if hidden == 8192 else (4, rows)
    _hc_prefill_inject_norm[grid](
        x,
        sublayer,
        gates,
        gamma,
        residual,
        norm,
        partial,
        HIDDEN=hidden,
        BLOCK=triton.next_power_of_2(hidden),
        EPS=eps,
        COLLECT=collect,
        # Keep the SM103 standalone RMSNorm's reduction and rounding order.
        num_warps=4,
        enable_fp_fusion=hidden != 8192,
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
        and x.shape[-1] in (10240, 32768)
        and sublayer.shape == (*x.shape[:-1], x.shape[-1] // 4)
        and gates.shape == (*x.shape[:-1], 4)
        and gamma.shape == (x.shape[-1],)
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
    width = x.shape[-1]
    rows = x.numel() // width
    capability = torch.cuda.get_device_capability(x.device)
    if width == 32768:
        if capability != (10, 3) or rows == 0:
            return None
    elif capability != (12, 0) or not (1 <= rows <= 32 or 4093 <= rows <= 16384):
        return None
    return _inject_grouped_norm(x, sublayer, gates, gamma, eps)
