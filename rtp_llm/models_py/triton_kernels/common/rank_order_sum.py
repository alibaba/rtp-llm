"""Fixed-order sum of per-rank partials for batch-invariant collectives."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _rank_order_sum_kernel(src, out, N, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    """Reduce WORLD partials per element in ascending rank order.

    One FP32 accumulator per element, rounded once on store, so the result
    matches an eager `acc = p0.float(); acc.add_(p1); ...; out.copy_(acc)`
    chain. BLOCK and WORLD are constants and N is unspecialized, so an element's
    arithmetic does not depend on the message size.
    """
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offs < N
    # A 64-bit row stride: rank * N overflows int32 past 2^31 / WORLD elements.
    stride = N.to(tl.int64)
    acc = tl.load(src + offs, mask, other=0.0).to(tl.float32)
    for rank in range(1, WORLD):
        acc += tl.load(src + rank * stride + offs, mask, other=0.0).to(tl.float32)
    tl.store(out + offs, acc.to(out.dtype.element_ty), mask)


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def supports(contributions: torch.Tensor, out: torch.Tensor) -> bool:
    """Whether the fused kernel may replace the eager rank-order chain.

    FP64 is excluded because the eager path accumulates at FP64 rather than
    FP32. A non-contiguous `out` is excluded because the kernel flattens it;
    the eager `out.copy_(acc)` handles that case, so fall back instead of
    raising.
    """
    return (
        contributions.is_cuda
        and out.is_cuda
        and torch.version.hip is None
        and out.device == contributions.device
        and contributions.dtype == out.dtype
        and contributions.dtype in _SUPPORTED_DTYPES
        and contributions.is_contiguous()
        and out.is_contiguous()
        and contributions.ndim == 2
        and contributions.shape[1] == out.numel()
    )


def rank_order_sum(contributions: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Sum `contributions[world, n]` into `out`, ascending rank order."""
    world, n = contributions.shape
    if n == 0:
        return out
    _rank_order_sum_kernel[(triton.cdiv(n, 1024),)](
        contributions,
        out.view(-1),
        n,
        WORLD=world,
        BLOCK=1024,
        num_warps=4,
    )
    return out
