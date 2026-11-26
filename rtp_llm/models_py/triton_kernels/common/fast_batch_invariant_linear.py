"""Faster batch-invariant BF16 linear for Blackwell (SM100/SM103).

Invariance argument
-------------------
A GEMM is batch-invariant when, for every output element, the set and the order
of the partial products accumulated into it do not depend on M.

The repo kernel enforces this by pinning BM=16, BN=64, BK=64 and traversing K
once per CTA. Pinning BM is *not* required: BM and BN only decide which CTA owns
an output element, not the order in which that element's products are summed.
What must stay fixed is

  1. BK and the ascending K traversal (no autotuned/M-dependent BK),
  2. the number of K splits (fixed, so the partial-combine tree is fixed),
  3. the tile/warp shape, so Triton keeps emitting the same MMA instruction
     family -- an m16n8k16 `mma.sync` tile and a `tcgen05` tile need not round
     identically, so the config must not be selected from M.

Everything here is a compile-time constant per (N, K); only the grid depends on
M. That buys large tiles for prefill and a fixed split-K for skinny decode
shapes while keeping bit-identical output across batch sizes.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _bi_gemm(
    x,
    weight,
    bias,
    output,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_WN: tl.constexpr,
    STRIDE_WK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)

    # Fixed, equal-sized K ranges. Every element of this split accumulates the
    # same BK chunks in the same ascending order for any M.
    chunks: tl.constexpr = (K + BK - 1) // BK
    per_split: tl.constexpr = (chunks + SPLIT_K - 1) // SPLIT_K
    split = tl.program_id(2)
    first = split * per_split
    for chunk in range(first, tl.minimum(first + per_split, chunks)):
        inner = chunk * BK + ks
        a = tl.load(
            x + rows[:, None] * K + inner[None, :],
            (rows[:, None] < M) & (inner[None, :] < K),
            other=0,
        )
        b = tl.load(
            weight + inner[:, None] * STRIDE_WK + cols[None, :] * STRIDE_WN,
            (inner[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    if HAS_BIAS and SPLIT_K == 1:
        acc += tl.load(bias + cols, cols < N, other=0).to(tl.float32)[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    if SPLIT_K == 1:
        store = acc if OUT_FP32 else acc.to(tl.bfloat16)
        tl.store(output + rows[:, None] * N + cols[None, :], store, mask)
    else:
        # A 248320-wide LM head makes M*N exceed int32 well before M*N*SPLIT_K.
        base = split.to(tl.int64) * M.to(tl.int64) * N
        tl.store(output + base + rows[:, None] * N + cols[None, :], acc, mask)


@triton.jit
def _bi_combine(
    partial,
    bias,
    output,
    M,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    elements = M.to(tl.int64) * N
    mask = offs < elements
    # Ascending split order, one FP32 accumulator per element.
    acc = tl.load(partial + offs, mask, other=0.0)
    for split in range(1, SPLIT_K):
        acc += tl.load(partial + split * elements + offs, mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias + (offs % N), mask, other=0.0).to(tl.float32)
    tl.store(output + offs, acc if OUT_FP32 else acc.to(tl.bfloat16), mask)


# (N, K) -> (BM, BN, BK, SPLIT_K, num_warps, num_stages); BK is pinned at 64 so
# the K traversal matches the existing BI reference tile.
DEFAULT = (128, 128, 64, 1, 8, 3)


def bi_gemm(x, weight, bias=None, output_fp32=False, config=None):
    n, k = weight.shape
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or x.size(-1) != k:
        raise ValueError("bi_gemm requires compatible BF16 input and weight")
    bm, bn, bk, split_k, warps, stages = config or DEFAULT
    out_dtype = torch.float32 if output_fp32 else torch.bfloat16
    output = torch.empty((*x.shape[:-1], n), dtype=out_dtype, device=x.device)
    if output.numel() == 0:
        return output
    rows = x.reshape(-1, k).contiguous()
    m = rows.size(0)
    bias = bias.contiguous() if bias is not None else None

    if split_k == 1:
        _bi_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn), 1)](
            rows, weight, bias, output, m, n, k, *weight.stride(),
            bm, bn, bk, 1, bias is not None, output_fp32,
            num_warps=warps, num_stages=stages,
        )
        return output

    # Split-K materializes SPLIT_K FP32 copies of the output. A wide LM head
    # would need tens of GiB, so the caller must keep split-K for skinny N.
    if split_k * m * n * 4 > 1 << 31:
        raise ValueError(
            f"split-K partials would need {split_k * m * n * 4 >> 20} MiB for "
            f"M={m} N={n} SPLIT_K={split_k}"
        )
    partial = torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
    _bi_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn), split_k)](
        rows, weight, bias, partial, m, n, k, *weight.stride(),
        bm, bn, bk, split_k, bias is not None, output_fp32,
        num_warps=warps, num_stages=stages,
    )
    _bi_combine[(triton.cdiv(m * n, 1024),)](
        partial, bias, output, m, n, split_k, 1024, bias is not None, output_fp32,
        num_warps=4,
    )
    return output
