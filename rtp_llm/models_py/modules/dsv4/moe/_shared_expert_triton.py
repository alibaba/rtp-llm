"""Triton helpers for DSV4 shared expert output combine."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _add_cast_kernel(routed_ptr, shared_ptr, out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        routed = tl.load(routed_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        shared = tl.load(shared_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, routed + shared, mask=mask)


def fused_add_cast_bf16(routed: torch.Tensor, shared: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    if triton is None:
        return (routed.float() + shared.float()).to(out_dtype)
    if not (routed.is_cuda and shared.is_cuda):
        return (routed.float() + shared.float()).to(out_dtype)
    if routed.shape != shared.shape:
        raise ValueError(f"shape mismatch: routed={routed.shape}, shared={shared.shape}")
    out = torch.empty_like(routed, dtype=out_dtype)
    n = routed.numel()
    if n == 0:
        return out
    block = 1024
    _add_cast_kernel[(triton.cdiv(n, block),)](
        routed.contiguous(),
        shared.contiguous(),
        out,
        n,
        BLOCK=block,
        num_warps=4,
    )
    return out
