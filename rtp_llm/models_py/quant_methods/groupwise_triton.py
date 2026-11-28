"""CUDA/ROCm affine weight-only projection: decode a tile, not a floating model."""

import torch
import triton
import triton.language as tl


@triton.jit
def _project(
    X,
    W,
    Z,
    S,
    IDS,
    R,
    Y,
    K: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    BITS: tl.constexpr,
    TOP: tl.constexpr,
    WEIGHTED: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    route = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    expert = tl.load(IDS + route).to(tl.int64)
    dst = Y + route.to(tl.int64) * N + n
    if expert < 0:
        tl.store(dst, 0.0, mask=n < N)
        return
    row = (route // TOP).to(tl.int64)
    acc = tl.full((BN,), 0.0, tl.float32)
    for block in range(triton.cdiv(K, BK)):
        k = block * BK + tl.arange(0, BK)
        x = tl.load(X + row * K + k, mask=k < K, other=0.0).to(tl.float32)
        valid = (n[:, None] < N) & (k[None, :] < K)
        if BITS == 4:
            byte = tl.load(
                W + (expert * N + n[:, None]) * (K // 2) + k[None, :] // 2,
                valid,
                other=0,
            )
            q = (byte >> ((k[None, :] % 2) * 4)) & 15
            zp = tl.load(
                Z + (expert * (N // 2) + n[:, None] // 2) * (K // G) + k[None, :] // G,
                valid,
                other=0,
            )
            zero = (zp >> ((n[:, None] % 2) * 4)) & 15
        else:
            q = tl.load(W + (expert * N + n[:, None]) * K + k[None, :], valid, other=0)
            zero = tl.load(
                Z + (expert * N + n[:, None]) * (K // G) + k[None, :] // G,
                valid,
                other=0,
            )
        scale = tl.load(
            S + (expert * N + n[:, None]) * (K // G) + k[None, :] // G, valid, other=0.0
        )
        w = ((q.to(tl.float32) - zero.to(tl.float32)) * scale).to(X.dtype.element_ty)
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    if WEIGHTED:
        acc *= tl.load(R + route)
    tl.store(dst, acc, mask=n < N)


def project(state, x, ids, weights, input_top_k, multiply_weights):
    k, n = state.layout.input_size, state.layout.output_size
    out = x.new_empty((*ids.shape, n))
    if ids.numel():
        if x.shape[0] * input_top_k != ids.numel():
            raise ValueError("Integer projection route/activation count mismatch")
        with torch.cuda.device(x.device):
            _project[(ids.numel(), triton.cdiv(n, 16))](
                x.contiguous(),
                state.weight,
                state.zero,
                state.scale,
                ids.contiguous(),
                weights.contiguous(),
                out,
                k,
                n,
                state.layout.group,
                state.runtime_bits,
                input_top_k,
                multiply_weights,
                16,
                64,
            )
    return out
