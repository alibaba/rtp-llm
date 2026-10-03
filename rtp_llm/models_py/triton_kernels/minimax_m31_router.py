"""Fixed-reduction FP32 routing for M3.1 decode and target verification."""

import torch
import triton
import triton.language as tl


@triton.jit
def _router_partials(
    X,
    W,
    P,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    split = tl.program_id(2)
    offsets = tl.arange(0, 128)
    acc = tl.zeros((16, 32), tl.float32)
    # Eight fixed partitions and a fixed accumulation order, independent of M.
    for step in range(0, tl.cdiv(K, 128 * 8)):
        ks = (step * 8 + split) * 128 + offsets
        x = tl.load(
            X + rows[:, None] * K + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + ks[:, None] * S1 + cols[None, :] * S0,
            (ks[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(x, w, acc, input_precision="tf32x3")
    tl.store(
        P + split * M * N + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _router_reduce(P, Y, SIZE: tl.constexpr):
    elements = tl.program_id(0) * 256 + tl.arange(0, 256)
    splits = tl.arange(0, 8)
    partials = tl.load(
        P + splits[:, None] * SIZE + elements[None, :],
        elements[None, :] < SIZE,
        other=0,
    )
    tl.store(Y + elements, tl.sum(partials, axis=0), elements < SIZE)


def minimax_m31_router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute [M,H] @ [E,H].T with FP32 accumulation and fixed row math.

    No shared mutable workspace: eager temporaries are stream ordered and
    capture temporaries belong to each Graph's private pool. Prefill stays on
    its original router; this entry point only serves the M3.1 decode path.
    """
    if (
        x.ndim != 2
        or weight.ndim != 2
        or not x.is_cuda
        or not weight.is_cuda
        or x.device != weight.device
        or x.dtype != torch.float32
        or weight.dtype != torch.float32
        or not x.is_contiguous()
        or x.shape[1] != weight.shape[1]
    ):
        raise ValueError(
            "M3.1 router requires contiguous CUDA FP32 rows and FP32 [E,H] weights"
        )
    m, k = x.shape
    n = weight.shape[0]
    if (n, k) != (128, 6144):
        raise ValueError(
            "M3.1 fixed router requires the validated [128,6144] checkpoint gate"
        )
    output = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if m == 0:
        return output
    partial = torch.empty((8, m, n), device=x.device, dtype=torch.float32)
    _router_partials[(triton.cdiv(m, 16), triton.cdiv(n, 32), 8)](
        x, weight, partial, m, k, n, *weight.stride(), num_warps=4
    )
    _router_reduce[(triton.cdiv(m * n, 256),)](partial, output, m * n, num_warps=4)
    return output
