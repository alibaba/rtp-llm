"""Fused QK-RMSNorm kernel — single launch for both Q and K heads.

Replaces two separate flashinfer.norm.rmsnorm calls (one for Q, one for K)
with a single Triton kernel.  Q heads use q_weight and K heads use k_weight;
the weights are pre-concatenated into a combined_weight tensor at init time.

Grid: (T * total_heads,) where total_heads = head_num + kv_head_num.
Each program handles one (token, head) pair of size_per_head elements.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qk_rmsnorm_kernel(
    qkv_ptr,
    weight_ptr,
    T,
    total_heads,
    size_per_head: tl.constexpr,
    eps,
    stride_qkv_t,
    stride_w_h,
):
    pid = tl.program_id(0).to(tl.int64)
    token_id = pid // total_heads
    head_id = pid % total_heads

    base = qkv_ptr + token_id * stride_qkv_t + head_id * size_per_head
    w_base = weight_ptr + head_id * stride_w_h

    offs = tl.arange(0, size_per_head)
    x = tl.load(base + offs).to(tl.float32)
    w = tl.load(w_base + offs).to(tl.float32)

    # Match flashinfer.norm.rmsnorm bit-exact: it computes rsqrt via the
    # PTX rsqrt intrinsic (one rounding), not ``1/sqrt`` (two roundings:
    # sqrt + reciprocal). Without this swap the result diverges by 1 fp32
    # ULP per element, accumulating across 36 layers for Qwen3.5 and
    # eventually shifting sampler output (verified via two-server diff).
    var = tl.sum(x * x) / size_per_head
    rrms = tl.rsqrt(var + eps)
    result = x * rrms * w

    tl.store(base + offs, result.to(tl.bfloat16))


def fused_qk_rmsnorm_triton(
    qkv: torch.Tensor,
    combined_weight: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    size_per_head: int,
    eps: float,
) -> None:
    """In-place fused QK RMSNorm on the qkv tensor.

    Args:
        qkv: [T, total_heads, size_per_head] — Q and K portions are normalized.
        combined_weight: [total_heads, size_per_head] — precomputed weights.
        head_num: number of Q heads.
        kv_head_num: number of KV heads.
        size_per_head: dimension per head (must be power of 2).
        eps: epsilon for numerical stability.
    """
    T = qkv.shape[0]
    total_heads = head_num + kv_head_num
    if T == 0:
        return

    grid = (T * total_heads,)
    _fused_qk_rmsnorm_kernel[grid](
        qkv,
        combined_weight,
        T,
        total_heads,
        size_per_head,
        eps,
        qkv.stride(0),
        combined_weight.stride(0),
    )
