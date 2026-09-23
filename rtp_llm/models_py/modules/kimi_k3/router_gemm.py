"""Fixed-tile router GEMM with compensated FP32 accumulation for K3."""

import torch
import triton
import triton.language as tl


@triton.jit
def _router_gemm(
    x, weight, output,
    rows: tl.constexpr, experts: tl.constexpr, hidden: tl.constexpr,
    x_stride: tl.constexpr, w_stride_0: tl.constexpr, w_stride_1: tl.constexpr,
    block_m: tl.constexpr = 16, block_n: tl.constexpr = 64,
    block_k: tl.constexpr = 32,
):
    row = tl.program_id(0) * block_m + tl.arange(0, block_m)
    col = tl.program_id(1) * block_n + tl.arange(0, block_n)
    ks = tl.arange(0, block_k)
    accumulated = tl.zeros((block_m, block_n), tl.float32)
    correction = tl.zeros((block_m, block_n), tl.float32)
    for group in range(tl.cdiv(hidden, 256)):
        partial = tl.zeros((block_m, block_n), tl.float32)
        for tile in range(8):
            k = group * 256 + tile * block_k + ks
            a = tl.load(
                x + row[:, None] * x_stride + k[None, :],
                (row[:, None] < rows) & (k[None, :] < hidden), other=0,
            )
            b = tl.load(
                weight + k[:, None] * w_stride_0 + col[None, :] * w_stride_1,
                (k[:, None] < hidden) & (col[None, :] < experts), other=0,
            ).to(tl.bfloat16)
            partial = tl.dot(a, b, partial)
        residual = partial - correction
        updated = accumulated + residual
        correction = (updated - accumulated) - residual
        accumulated = updated
    tl.store(
        output + row[:, None] * experts + col[None, :], accumulated,
        (row[:, None] < rows) & (col[None, :] < experts),
    )


def router_gemm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project BF16 hidden states using FP32 weights proven BF16-exact at load."""
    if (hidden.ndim != 2 or weight.ndim != 2
            or hidden.shape[1] != weight.shape[0]):
        raise ValueError("K3 router expects [tokens, hidden] @ [hidden, experts]")
    if (not hidden.is_cuda or hidden.device != weight.device
            or hidden.dtype != torch.bfloat16 or weight.dtype != torch.float32):
        raise ValueError("K3 router GEMM requires BF16 input and FP32 CUDA weights")
    if hidden.stride(1) != 1:
        hidden = hidden.contiguous()
    rows, width = hidden.shape
    experts = weight.shape[1]
    output = torch.empty((rows, experts), device=hidden.device, dtype=torch.float32)
    if rows == 0 or experts == 0:
        return output
    _router_gemm[(triton.cdiv(rows, 16), triton.cdiv(experts, 64))](
        hidden, weight, output, rows, experts, width,
        hidden.stride(0), weight.stride(0), weight.stride(1),
        num_warps=4, num_stages=3, enable_fp_fusion=False,
    )
    return output
