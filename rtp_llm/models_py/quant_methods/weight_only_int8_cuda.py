"""Tile-local W8A16 GEMM for the CUDA mixed-GEMM INT8 checkpoint layout."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mixed_w8a16(
    X, W, S, B, Y, M, K: tl.constexpr, N: tl.constexpr, BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    tile = tl.program_id(0)
    m = (tile // (N // BN)) * BM + tl.arange(0, BM)
    n = (tile % (N // BN)) * BN + tl.arange(0, BN)
    scale = tl.load(S + n)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(K // BK):
        k = block * BK + tl.arange(0, BK)
        x = tl.load(X + m[:, None].to(tl.int64) * K + k[None, :],
                    mask=m[:, None] < M, other=0)
        # Invert cuda_groupwise_layout(bits=8, arch>=75): row permutation
        # in each MMA group of 16, paired columns interleaved per 64 rows,
        # then the middle two bytes of each four-byte word exchanged.
        lane = k % 16
        row = k - lane + (lane % 8 // 2) * 4 + (lane // 8) * 2 + lane % 2
        byte = (row % 64 // 4) * 4 + (row % 2) * 2 + row % 4 // 2
        off = ((n // 2).to(tl.int64) * (2 * K))[None, :]
        off = off + (row // 64 * 128 + byte)[:, None] + (n % 2 * 64)[None, :]
        q = tl.load(W + off).to(tl.int32) - 128
        w = (q.to(tl.float32) * scale[None, :]).to(X.dtype.element_ty)
        acc = tl.dot(x, w, acc)
    if BIAS:
        acc += tl.load(B + n)[None, :]
    tl.store(Y + m[:, None].to(tl.int64) * N + n[None, :], acc,
             mask=m[:, None] < M)


def mixed_w8a16(x, weight, scale, *, bias=None):
    """Caller provides padded contiguous activations and unsigned packed bytes."""
    m, k = x.shape
    n = scale.numel()
    out = x.new_empty((m, n))
    if m:
        with torch.cuda.device(x.device):
            _mixed_w8a16[(triton.cdiv(m, 16) * (n // 32),)](
                x, weight, scale, bias if bias is not None else scale, out,
                m, k, n, bias is not None, 16, 32, 32,
            )
    return out
