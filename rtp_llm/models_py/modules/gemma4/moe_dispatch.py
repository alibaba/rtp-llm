"""Experimental MoE dispatch preserving the caller's original sort permutation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_dispatch_kernel(X, PERM, OUT, INVERSE, ROWS, WIDTH: tl.constexpr,
                          XS: tl.constexpr, TOP_K: tl.constexpr,
                          BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // WIDTH, index % WIDTH
    original = tl.load(PERM + row, mask=row < ROWS, other=0)
    token = original // TOP_K
    value = tl.load(X + token * XS + col, mask=row < ROWS, other=0)
    tl.store(OUT + index, value, mask=row < ROWS)
    tl.store(INVERSE + original, row, mask=(row < ROWS) & (col == 0))


@triton.jit
def _pack_dispatch_rows_kernel(X, PERM, OUT, INVERSE, ROWS,
                               WIDTH: tl.constexpr, XS: tl.constexpr,
                               TOP_K: tl.constexpr, RM: tl.constexpr,
                               CN: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * RM + tl.arange(0, RM)
    col = tl.program_id(1).to(tl.int64) * CN + tl.arange(0, CN)
    original = tl.load(PERM + row, mask=row < ROWS, other=0)
    token = original // TOP_K
    value = tl.load(
        X + token[:, None] * XS + col[None, :],
        mask=(row[:, None] < ROWS) & (col[None, :] < WIDTH), other=0,
    )
    tl.store(OUT + row[:, None] * WIDTH + col[None, :], value,
             mask=(row[:, None] < ROWS) & (col[None, :] < WIDTH))
    if tl.program_id(1) == 0:
        tl.store(INVERSE + original, row, mask=row < ROWS)


@triton.jit
def _expert_offsets_kernel(IDS, PERM, OFFSETS, ROWS,
                           EXPERTS: tl.constexpr, ITERATIONS: tl.constexpr,
                           BLOCK: tl.constexpr):
    expert = tl.arange(0, BLOCK)
    low = tl.full((BLOCK,), 0, tl.int64)
    high = tl.full((BLOCK,), ROWS, tl.int64)
    for _ in range(ITERATIONS):
        active = (low < high) & (expert < EXPERTS)
        middle = (low + high) // 2
        original = tl.load(PERM + middle, mask=active, other=0)
        value = tl.load(IDS + original, mask=active, other=0)
        lower = value <= expert
        low = tl.where(active & lower, middle + 1, low)
        high = tl.where(active & ~lower, middle, high)
    tl.store(OFFSETS + expert, low.to(tl.int32), mask=expert < EXPERTS)


def prepare_dispatch(x, indices, permutation, experts, pack_block=1024, pack_tile=None):
    if (
        not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 2
        or x.numel() == 0 or x.stride(1) != 1 or x.stride(0) <= 0
        or indices.device != x.device or indices.dtype != torch.int64
        or indices.dim() != 2 or not indices.is_contiguous()
        or indices.shape[0] != x.shape[0] or indices.shape[1] <= 0
        or permutation.device != x.device or permutation.dtype != torch.int64
        or permutation.dim() != 1 or not permutation.is_contiguous()
        or permutation.numel() != indices.numel() or indices.numel() > 2**24
        or not isinstance(experts, int) or not 1 <= experts <= 1024
        or pack_block not in (1024, 2048, 4096, 8192, 16384)
        or (pack_tile is not None and pack_tile not in ((1, 2048), (1, 4096), (4, 256), (4, 1024), (8, 512)))
    ):
        return None
    rows, width, top_k = indices.numel(), x.shape[1], indices.shape[1]
    dispatched = torch.empty((rows, width), device=x.device, dtype=x.dtype)
    inverse = torch.empty_like(permutation)
    offsets = torch.empty((experts,), device=x.device, dtype=torch.int32)
    if pack_tile is None:
        _pack_dispatch_kernel[(triton.cdiv(rows * width, pack_block),)](
            x, permutation, dispatched, inverse, rows, width,
            x.stride(0), top_k, pack_block, num_warps=4 if pack_block <= 4096 else 8,
        )
    else:
        rm, cn = pack_tile
        _pack_dispatch_rows_kernel[(triton.cdiv(rows, rm), triton.cdiv(width, cn))](
            x, permutation, dispatched, inverse, rows, width,
            x.stride(0), top_k, rm, cn,
            num_warps=4 if rm * cn <= 2048 else 8,
        )
    _expert_offsets_kernel[(1,)](
        indices, permutation, offsets, rows, experts,
        (rows + 1).bit_length(), triton.next_power_of_2(experts),
        num_warps=4 if experts <= 128 else 8,
    )
    return dispatched, inverse, offsets
