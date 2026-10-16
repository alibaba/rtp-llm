"""Optional integer postprocessing for the FP8 indexer."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _sentinel_kernel(
    X,
    Y,
    N,
    W: tl.constexpr,
    RS: tl.constexpr,
    CS: tl.constexpr,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + p // W * RS + p % W * CS, p < N, other=-1)
    tl.store(Y + p, tl.where(x >= 0, x, SENTINEL), p < N)


@triton.jit
def _finish_topk_kernel(
    X,
    OFF,
    OUT,
    N,
    W: tl.constexpr,
    RS: tl.constexpr,
    CS: tl.constexpr,
    OS: tl.constexpr,
    ORS: tl.constexpr,
    OCS: tl.constexpr,
    CANONICAL: tl.constexpr,
    SENTINEL: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = p // W, p % W
    x = tl.load(X + row * RS + col * CS, p < N, other=-1)
    off = tl.load(OFF + row * OS, p < N, other=0)
    if CANONICAL:
        x = tl.where(x == SENTINEL, -1, x)
    if WIDE:
        value = x.to(tl.int64) + off.to(tl.int64)
    else:
        value = x.to(tl.int32) + off.to(tl.int32)
    tl.store(OUT + row * ORS + col * OCS, tl.where(x >= 0, value, x), p < N)


def topk_is_supported(
    indices: torch.Tensor, offsets: torch.Tensor, out: torch.Tensor
) -> bool:
    return (
        indices.is_cuda
        and offsets.device == indices.device == out.device
        and indices.ndim == out.ndim == 2
        and indices.shape == out.shape
        and offsets.ndim == 1
        and offsets.shape[0] == indices.shape[0]
        and all(t.dtype in (torch.int32, torch.int64) for t in (indices, offsets, out))
        and out.stride(1) == 1
        and out.stride(0) >= out.shape[1]
        and out.untyped_storage().data_ptr() != indices.untyped_storage().data_ptr()
        and out.untyped_storage().data_ptr() != offsets.untyped_storage().data_ptr()
    )


def try_write_topk(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor,
    canonicalize: bool,
) -> bool:
    """Write the original Torch canonicalization/offset result into a disjoint view."""
    if os.environ.get("RTP_INDEXER_FUSED_TOPK", "0") != "1" or not topk_is_supported(
        indices, offsets, out
    ):
        return False
    if not indices.numel():
        return True
    sentinel = torch.iinfo(indices.dtype).max
    if canonicalize:
        sortable = torch.empty(
            indices.shape, device=indices.device, dtype=indices.dtype
        )
        _sentinel_kernel[(triton.cdiv(indices.numel(), 256),)](
            indices,
            sortable,
            indices.numel(),
            indices.shape[1],
            indices.stride(0),
            indices.stride(1),
            sentinel,
            BLOCK=256,
        )
        indices = torch.sort(sortable, dim=-1).values
    _finish_topk_kernel[(triton.cdiv(indices.numel(), 256),)](
        indices,
        offsets,
        out,
        indices.numel(),
        indices.shape[1],
        indices.stride(0),
        indices.stride(1),
        offsets.stride(0),
        out.stride(0),
        out.stride(1),
        canonicalize,
        sentinel,
        indices.dtype == torch.int64 or offsets.dtype == torch.int64,
        BLOCK=256,
    )
    return True
