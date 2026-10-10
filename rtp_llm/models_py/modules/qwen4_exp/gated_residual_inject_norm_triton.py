"""Fuse residual writeback and the next grouped RMSNorm without losing BF16 rounding."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_exp.gated_residual_inject_triton import (
    is_supported as inject_is_supported,
)
from rtp_llm.models_py.modules.qwen4_exp.gated_residual_norm_triton import (
    is_supported as norm_is_supported,
)


@triton.jit
def _inject_norm_kernel(
    hyper_ptr,
    sublayer_ptr,
    gate_ptr,
    gamma_ptr,
    combined_ptr,
    normed_ptr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    branch = tl.program_id(1)
    column = tl.arange(0, BLOCK)
    mask = column < HIDDEN
    offset = row * (4 * HIDDEN) + branch * HIDDEN + column
    hyper = tl.load(hyper_ptr + offset, mask, other=0).to(tl.float32)
    sublayer = tl.load(sublayer_ptr + row * HIDDEN + column, mask, other=0).to(
        tl.float32
    )
    gate = tl.load(gate_ptr + row * 4 + branch).to(tl.float32)
    # The two materialized BF16 rounding points belong to the residual contract.
    injection = tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;",
        constraints="=h,f",
        args=[sublayer * gate],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    ).to(tl.float32)
    combined = tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;",
        constraints="=h,f",
        args=[hyper + injection],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )
    values = tl.where(mask, combined.to(tl.float32), 0.0)
    gamma = tl.load(gamma_ptr + branch * HIDDEN + column, mask, other=0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / HIDDEN + EPS)
    normalized = values * inv_rms * (1.0 + gamma)
    tl.store(combined_ptr + offset, combined, mask)
    tl.store(normed_ptr + offset, normalized, mask)


def is_supported(hyper, sublayer, gates, gamma, group_size):
    return (
        inject_is_supported(hyper, sublayer, gates)
        and norm_is_supported(hyper, gamma, group_size)
        and group_size == sublayer.shape[-1]
        and torch.cuda.get_device_capability(hyper.device)[0] >= 8
    )


def inject_and_grouped_rms_norm(hyper, sublayer, gates, gamma, group_size, eps):
    if not is_supported(hyper, sublayer, gates, gamma, group_size):
        raise ValueError("unsupported Qwen4 fused injection/grouped RMSNorm layout")
    combined = torch.empty_like(hyper)
    normed = torch.empty_like(hyper)
    rows = hyper.numel() // hyper.shape[-1]
    if rows:
        _inject_norm_kernel[(rows, 4)](
            hyper,
            sublayer,
            gates,
            gamma,
            combined,
            normed,
            HIDDEN=group_size,
            EPS=eps,
            BLOCK=triton.next_power_of_2(group_size),
            num_warps=4,
            # Match the production norm's FP32 contraction policy. The explicit
            # BF16 casts above still preserve both residual rounding points.
            enable_fp_fusion=True,
        )
    return combined, normed
