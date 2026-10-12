"""BF16 Gemma4 RMSNorm candidate; the existing eager implementation is its oracle.

No GEMM, attention core or weight transform is implemented here.  Call sites
must explicitly enable this candidate after local correctness/performance checks.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_kernel(
    X,
    W,
    Y,
    STATS,
    TOKEN_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
    SAVE_STATS: tl.constexpr,
    REDUCE_WIDTH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    offset = (row // HEADS) * TOKEN_STRIDE + (row % HEADS) * HEAD_STRIDE
    values = tl.load(X + offset + col, mask=col < D, other=0).to(tl.float32)
    # Match the installed float32 ATen mean topology. Each logical lane
    # accumulates four strided components before the block/warp tree. This
    # preserves rounding boundaries without materializing square/cast tensors.
    lane = tl.arange(0, REDUCE_WIDTH)
    acc0 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    acc1 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    acc2 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    acc3 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    for index in range(tl.cdiv(D, REDUCE_WIDTH * 4)):
        base = (lane + index * REDUCE_WIDTH) * 4
        v0 = tl.load(X + offset + base, mask=base < D, other=0).to(tl.float32)
        v1 = tl.load(X + offset + base + 1, mask=base + 1 < D, other=0).to(tl.float32)
        v2 = tl.load(X + offset + base + 2, mask=base + 2 < D, other=0).to(tl.float32)
        v3 = tl.load(X + offset + base + 3, mask=base + 3 < D, other=0).to(tl.float32)
        acc0 = acc0 + v0 * v0
        acc1 = acc1 + v1 * v1
        acc2 = acc2 + v2 * v2
        acc3 = acc3 + v3 * v3
    total = ((acc0 + acc1) + acc2) + acc3
    if REDUCE_WIDTH > 32:
        total = tl.sum(tl.reshape(total, (REDUCE_WIDTH // 32, 32)), axis=0)
    variance = tl.sum(total, axis=0) * (1.0 / D)
    inverse = tl.rsqrt(variance + EPS)
    normalized = values * inverse
    if HAS_WEIGHT:
        weight = tl.load(W + col, mask=col < D, other=0).to(tl.float32)
        normalized = normalized * weight
    tl.store(Y + row * D + col, normalized, mask=col < D)
    if SAVE_STATS:
        tl.store(STATS + row * 2, variance)
        tl.store(STATS + row * 2 + 1, inverse)


def is_supported(x: torch.Tensor, weight: torch.Tensor | None = None) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() in (2, 3)
        and x.numel() > 0
        and x.stride(-1) == 1
        and 128 <= x.shape[-1] <= 16384
        and x.shape[-1] % 4 == 0
        and all(stride > 0 for stride in x.stride())
        and (
            weight is None
            or (
                weight.device == x.device
                and weight.dtype in (torch.bfloat16, torch.float32)
                and weight.shape == (x.shape[-1],)
                and weight.is_contiguous()
            )
        )
    )


def norm(
    x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6
) -> torch.Tensor | None:
    """Return a new contiguous output, or None for the caller's original path."""
    if not is_supported(x, weight) or eps <= 0:
        return None
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    heads = x.shape[1] if x.dim() == 3 else 1
    reduce_width = _reference_reduce_width(x.shape[-1], x.numel() // x.shape[-1])
    _norm_kernel[(x.numel() // x.shape[-1],)](
        x,
        weight if weight is not None else x,
        output,
        output,
        x.stride(0),
        x.stride(1) if x.dim() == 3 else 0,
        heads,
        x.shape[-1],
        eps,
        weight is not None,
        triton.next_power_of_2(x.shape[-1]),
        False,
        reduce_width,
        num_warps=4 if x.shape[-1] <= 2048 else 8,
        enable_fp_fusion=False,
    )
    return output


def _reference_reduce_width(dim: int, rows: int) -> int:
    """Float32 contiguous last-axis reduction policy in installed Torch2.11."""
    input_pow2 = min(512, 1 << ((dim // 4).bit_length() - 1))
    output_pow2 = min(512, 1 << (rows.bit_length() - 1))
    width = min(input_pow2, 32)
    height = min(output_pow2, 512 // width)
    return min(input_pow2, 512 // height)


def norm_with_statistics(
    x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6
):
    """Diagnostic invocation of the same arithmetic; never used by model forward."""
    if not is_supported(x, weight):
        raise ValueError("Unsupported norm diagnostic input")
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    statistics = torch.empty(
        (x.numel() // x.shape[-1], 2), dtype=torch.float32, device=x.device
    )
    heads = x.shape[1] if x.dim() == 3 else 1
    _norm_kernel[(x.numel() // x.shape[-1],)](
        x,
        weight if weight is not None else x,
        output,
        statistics,
        x.stride(0),
        x.stride(1) if x.dim() == 3 else 0,
        heads,
        x.shape[-1],
        eps,
        weight is not None,
        triton.next_power_of_2(x.shape[-1]),
        True,
        _reference_reduce_width(x.shape[-1], x.numel() // x.shape[-1]),
        num_warps=4 if x.shape[-1] <= 2048 else 8,
        enable_fp_fusion=False,
    )
    return output, statistics
