"""Shared MoE output addition preserving the original two BF16 roundings."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.kimi_kda.cached_launch import CachedLaunch


@triton.jit
def _add_moe_output(R, S, P, O, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    r = tl.load(R + offsets, mask, other=0).to(tl.float32)
    s = tl.load(S + offsets, mask, other=0).to(tl.float32)
    p = tl.load(P + offsets, mask, other=0).to(tl.float32)
    first = (r + s).to(tl.bfloat16).to(tl.float32)
    tl.store(O + offsets, first + p, mask)


_launch_add_moe_output = CachedLaunch(_add_moe_output)


def add_moe_output(routed, shared, residual=None, *, optimize=False):
    """A supplied residual is always included, even on the reference path."""
    if residual is not None and residual.shape != routed.shape:
        raise ValueError("MoE residual must have the output shape")
    if (
        residual is not None
        and optimize
        and routed.is_cuda
        and routed.dtype == shared.dtype == residual.dtype == torch.bfloat16
        and routed.device == shared.device == residual.device
        and shared.shape == routed.shape
        and routed.is_contiguous()
        and shared.is_contiguous()
        and residual.is_contiguous()
    ):
        out = torch.empty_like(routed)
        size = out.numel()
        if size:
            _launch_add_moe_output(
                ((size + 1023) // 1024, 1, 1),
                (routed, shared, residual, out),
                (size, 1024),
            )
        return out
    out = routed + shared
    return out if residual is None else out + residual
