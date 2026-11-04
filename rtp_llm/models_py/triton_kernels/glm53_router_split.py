"""FP32 router weights represented by three BF16 components, without TF32."""

import torch, triton, triton.language as tl


@triton.jit
def split_router_kernel(
    X,
    W0,
    W1,
    W2,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PARTS: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc0 = tl.zeros((BM, BN), tl.float32)
    acc1 = tl.zeros((BM, BN), tl.float32)
    acc2 = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        mask = (n[None, :] < N) & (k[:, None] < K)
        x = tl.load(
            X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0
        )
        b0 = tl.load(W0 + n[None, :] * K + k[:, None], mask, 0)
        b1 = tl.load(W1 + n[None, :] * K + k[:, None], mask, 0)
        acc0 = tl.dot(x, b0, acc0)
        acc1 = tl.dot(x, b1, acc1)
        if PARTS == 3:
            b2 = tl.load(W2 + n[None, :] * K + k[:, None], mask, 0)
            acc2 = tl.dot(x, b2, acc2)
    y = acc0 + (acc1 + acc2)
    tl.store(Y + m[:, None] * N + n[None, :], y, (m[:, None] < M) & (n[None, :] < N))


def split_weights(w):
    first = w.to(torch.bfloat16)
    rem = w - first.float()
    second = rem.to(torch.bfloat16)
    third = (rem - second.float()).to(torch.bfloat16)
    return first, second, third


def router(x, weights, config=(16, 32, 64, 4, 3), parts=3):
    first, second, third = weights
    bm, bn, bk, warps, stages = config
    y = torch.empty((x.shape[0], first.shape[0]), device=x.device, dtype=torch.float32)
    split_router_kernel[(triton.cdiv(x.shape[0], bm), triton.cdiv(first.shape[0], bn))](
        x,
        first,
        second,
        third,
        y,
        x.shape[0],
        first.shape[0],
        x.shape[1],
        bm,
        bn,
        bk,
        parts,
        num_warps=warps,
        num_stages=stages,
    )
    return y
