"""M890P request-local BF16 WoA arithmetic, shared across eight Q3/Q4 requests.

Native policy: cyclic K32 across three FP32 planes, ordered sum, BF16 output.
Only the frozen G8/N1024/K4096 B8 shape is qualified by component replay.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def partial_kernel(
    X,
    W,
    P,
    ROWS: tl.constexpr,
    Q: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 32,
):
    col = tl.program_id(0) * BN + tl.arange(0, BN)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = tl.arange(0, BM)
    # Native ELF: offset 0x628 = tile_id << 5; 0x788 advances tile_id
    # by split_count. Observed G8 gridZ24 gives 3 cyclic planes.
    begin = split * BK
    end = 4096
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(begin, end, 3 * BK):
        k = block + tl.arange(0, BK)
        lhs = tl.load(
            X + row[:, None] * 32768 + group * 4096 + k[None, :],
            (row[:, None] < ROWS) & (k[None, :] < end),
            0,
        )
        rhs = tl.load(
            W + group * 4194304 + col[None, :] * 4096 + k[:, None], k[:, None] < end, 0
        )
        acc = tl.dot(lhs, rhs, acc)
    tl.store(
        P + ((split * ROWS + row[:, None]) * 8 + group) * 1024 + col[None, :],
        acc,
        row[:, None] < ROWS,
    )


@triton.jit
def reduce_kernel(P, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr = 256):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p0 = tl.load(P + pos, pos < TOTAL, 0)
    p1 = tl.load(P + TOTAL + pos, pos < TOTAL, 0)
    p2 = tl.load(P + 2 * TOTAL + pos, pos < TOTAL, 0)
    value = (p0 + p1) + p2
    tl.store(Y + pos, value.to(tl.bfloat16), pos < TOTAL)


def request_grid_eligible(x, weight, output, *, prefill, request_local):
    return (
        request_local
        and not prefill
        and x.ndim == 4
        and tuple(x.shape) in ((8, 3, 8, 4096), (8, 4, 8, 4096))
        and tuple(weight.shape) == (8, 1024, 4096)
        and tuple(output.shape) == (8, x.shape[1], 8, 1024)
        and x.dtype == weight.dtype == output.dtype == torch.bfloat16
        and x.device == weight.device == output.device
        and x.is_contiguous()
        and weight.is_contiguous()
        and output.is_contiguous()
    )


def request_grid_woa(x, weight, output):
    rows = x.shape[0] * x.shape[1]
    # Allocated in the graph pool during capture; no input/weight data is cached.
    partials = torch.empty((3, rows, 8, 1024), device=x.device, dtype=torch.float32)
    partial_kernel[(16, 8, 3)](x, weight, partials, rows, x.shape[1], num_warps=4)
    reduce_kernel[(triton.cdiv(rows * 8192, 256),)](
        partials, output, rows * 8192, num_warps=4, enable_fp_fusion=False
    )
    return output
