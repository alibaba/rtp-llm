"""Shared MoE output addition preserving the original two BF16 roundings."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.kimi_kda.cached_launch import CachedLaunch


@triton.jit
def _add_moe_output(R, S, P, O, size, HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    r = tl.load(R + offsets, mask, other=0).to(tl.float32)
    s = tl.load(S + offsets, mask, other=0).to(tl.float32)
    first = (r + s).to(tl.bfloat16).to(tl.float32)
    if HAS_RESIDUAL:
        p = tl.load(P + offsets, mask, other=0).to(tl.float32)
        first += p
    tl.store(O + offsets, first, mask)


_launch_add_moe_output = CachedLaunch(_add_moe_output)


def add_moe_output(routed, shared, residual=None):
    """Add contiguous CUDA BF16 MoE outputs, preserving both BF16 roundings."""
    if residual is not None and residual.shape != routed.shape:
        raise ValueError("MoE residual must have the output shape")
    out = torch.empty_like(routed)
    size = out.numel()
    if size:
        _launch_add_moe_output(
            ((size + 1023) // 1024, 1, 1),
            (routed, shared, routed if residual is None else residual, out),
            (size, residual is not None, 1024),
        )
    return out
