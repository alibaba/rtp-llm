"""BF16 matmul with a fixed K traversal, following vLLM's batch-invariant matmul."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _linear(
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
    HAS_BIAS: tl.constexpr,
):
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)

    # Each output element is owned by one CTA. M only changes grid size and masks.
    for start in range(0, K, BK):
        inner = start + ks
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
    if HAS_BIAS:
        acc += tl.load(bias + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        output + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def batch_invariant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_fp32: bool = False,
) -> torch.Tensor:
    """Multiply BF16 activations by a BF16 [out_features, in_features] weight."""
    n, k = weight.shape
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or x.size(-1) != k:
        raise ValueError(
            "batch-invariant linear requires compatible BF16 input and weight"
        )
    output = torch.empty(
        (*x.shape[:-1], n),
        dtype=torch.float32 if output_fp32 else torch.bfloat16,
        device=x.device,
    )
    if output.numel() == 0:
        return output

    rows = x.reshape(-1, k).contiguous()
    bias = bias.contiguous() if bias is not None else None
    m = rows.size(0)
    bm, bn, bk = 16, 16 if n <= 16 else 64, 64
    _linear[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        rows,
        weight,
        bias,
        output,
        m,
        n,
        k,
        *weight.stride(),
        bm,
        bn,
        bk,
        bias is not None,
        num_warps=4,
        num_stages=3,
    )
    return output
