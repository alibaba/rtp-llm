"""Fused MoE top-k output reduce Triton kernel (one launch per call).

Opt r9 ⑥ tail-clear lane, item 4a: the fused MoE executor's final
``out.view(M, top_k, K).sum(dim=1)`` was the decode-window residual
``reduce_kernel<BFloat16>`` — 41.8 ms / 3750x (= 30 MoE layers x ~125
forwards). ATen's strided mid-dim reduction over bf16 runs at ~1 TB/s on
the [M*top_k, K] token-major layout; this kernel reads the same rows with
coalesced [BLOCK_K] tiles, accumulates in fp32 (the same acc type ATen
uses for bf16 sums) and writes the [M, K] bf16 result — bf16-ulp class vs
the torch reduce (order-of-accumulation only), several times faster.

The input is the fused GEMM2 output exactly as the executor produces it:
``out`` [M * top_k, K] contiguous, row ``t * top_k + k`` holding expert k's
contribution for token t (the GEMM's sorted-token scatter already restored
token-major order).
"""

import torch
import triton
import triton.language as tl

_BLOCK_K = 1024
_NUM_WARPS = 8


@triton.jit
def _moe_sum_kernel(
    src_ptr,  # [M * TOP_K, K] contiguous
    dst_ptr,  # [M, K] out, same dtype
    M,
    K,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_k = tl.program_id(axis=1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = offs_k < K
    base = src_ptr + pid_m * TOP_K * K
    acc = tl.zeros([BLOCK_K], tl.float32)
    for k in range(TOP_K):
        row = base + k * K + offs_k
        acc += tl.load(row, mask=kmask, other=0.0).to(tl.float32)
    out = dst_ptr + pid_m * K + offs_k
    tl.store(out, acc.to(dst_ptr.dtype.element_ty), mask=kmask)


def moe_sum_reduce(out: torch.Tensor, M: int, top_k: int) -> torch.Tensor:
    """Sum ``out`` [M * top_k, K] over the top_k axis -> [M, K].

    One launch, fp32 accumulation, bf16/fp16/fp32 inputs; matches
    ``out.view(M, top_k, K).sum(dim=1)`` modulo the accumulation order
    (bf16-ulp class). The caller owns the gate and the torch fallback.
    """
    if out.dim() != 2:
        raise ValueError("moe_sum_reduce needs a 2-D [M*top_k, K] tensor")
    if not out.is_cuda:
        raise ValueError("moe_sum_reduce requires CUDA tensors")
    if int(top_k) < 1 or int(M) < 1:
        raise ValueError("moe_sum_reduce needs positive M and top_k")
    K = out.shape[1]
    if out.numel() != M * top_k * K:
        raise ValueError("moe_sum_reduce shape mismatch")
    if not out.is_contiguous():
        raise ValueError("moe_sum_reduce needs a contiguous out tensor")
    if out.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("moe_sum_reduce needs bf16/fp16/fp32 inputs")
    if K < 1:
        raise ValueError("moe_sum_reduce needs a non-empty last dim")
    final = torch.empty(M, K, dtype=out.dtype, device=out.device)
    grid = (M, triton.cdiv(K, _BLOCK_K))
    _moe_sum_kernel[grid](
        out,
        final,
        M,
        K,
        TOP_K=int(top_k),
        BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
    )
    return final
