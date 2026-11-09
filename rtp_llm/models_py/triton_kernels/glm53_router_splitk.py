"""Opt-in split-K projection of two BF16 components of FP32 router weights."""

import torch
import triton
import triton.language as tl


@triton.jit
def gemm(
    X,
    W0,
    W1,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    split = tl.program_id(2)
    a0 = tl.zeros((BM, BN), tl.float32)
    a1 = tl.zeros((BM, BN), tl.float32)
    for j in range(tl.cdiv(K, BK * SK)):
        k = (split * tl.cdiv(K, BK * SK) + j) * BK + kk
        x = tl.load(
            X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0
        )
        w0 = tl.load(
            W0 + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0
        )
        w1 = tl.load(
            W1 + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0
        )
        a0 = tl.dot(x, w0, a0)
        a1 = tl.dot(x, w1, a1)
    tl.store(
        P + split * M * N + m[:, None] * N + n[None, :],
        a0 + a1,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def combine(P, Y, M: tl.constexpr, N: tl.constexpr, SK: tl.constexpr, B: tl.constexpr):
    x = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, SK)
    v = tl.load(P + s[:, None] * M * N + x[None, :], x[None, :] < M * N, 0)
    tl.store(Y + x, tl.sum(v, 0), x < M * N)


def operation(x, w0, w1, config, y, p):
    bm, bn, bk, sk, nw, ns = config
    m, n = x.shape[0], w0.shape[0]
    gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn), sk)](
        x, w0, w1, p, m, n, 4096, bm, bn, bk, sk, num_warps=nw, num_stages=ns
    )
    if sk == 1:
        y.copy_(p[0])
    else:
        combine[(triton.cdiv(m * n, 256),)](p, y, m, n, sk, 256, num_warps=4)
    return y


def router(x, weights):
    w0, w1, _ = weights
    m, n = x.shape[0], w0.shape[0]
    config = (16 if m == 48 else 32, 64, 64, 8, 4, 3)
    partial = torch.empty((8, m, n), device=x.device, dtype=torch.float32)
    output = torch.empty((m, n), device=x.device, dtype=torch.float32)
    return operation(x, w0, w1, config, output, partial)
