"""Host-mapped WKV projection for guest-GPU Engram decode.

The DSV4.1 Engram WKV checkpoint tensor must stay in host RAM.  During decode
the weight is read directly from CUDA-mapped pinned host pages by this kernel,
so no persistent device copy is created and the launch is CUDA-graph safe.
Prefill still uses a bounded transient device staging copy because its large
row count makes host reads uneconomical.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _wkv_host_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        a = tl.load(
            a_ptr + rows[:, None] * K + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < K),
            other=0.0,
            cache_modifier=".cg",
        )
        b = tl.load(
            b_ptr + cols[:, None] * K + k[None, :],
            mask=(cols[:, None] < N) & (k[None, :] < K),
            other=0.0,
            cache_modifier=".cg",
        )
        acc += tl.dot(a, tl.trans(b), input_precision="ieee")
    out = acc.to(tl.bfloat16)
    tl.store(
        c_ptr + rows[:, None] * N + cols[None, :],
        out,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def host_wkv(rows: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return ``rows @ weight.T`` with weight in CUDA-mapped host memory."""
    if rows.ndim != 2 or weight.ndim != 2:
        raise ValueError("host_wkv expects 2D tensors")
    m, k = rows.shape
    n, wk = weight.shape
    if k != wk:
        raise ValueError(f"host_wkv shape mismatch: {rows.shape} vs {weight.shape}")
    rows = rows.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=rows.device)
    if m == 0 or n == 0:
        return out
    block_m = 64
    block_n = 64
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _wkv_host_kernel[grid](
        rows,
        weight,
        out,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    return out
