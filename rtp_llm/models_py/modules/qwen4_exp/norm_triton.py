"""Graph-capturable Qwen4 Q/K RMSNorm with FP32 ``1 + gamma`` gains."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rmsnorm_kernel(
    qkv_ptr,
    q_gamma_ptr,
    k_gamma_ptr,
    ROW_STRIDE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, D)
    x = tl.load(qkv_ptr + row * ROW_STRIDE + head * D + offsets).to(tl.float32)
    q_gamma = tl.load(q_gamma_ptr + offsets).to(tl.float32)
    k_gamma = tl.load(k_gamma_ptr + offsets).to(tl.float32)
    gamma = tl.where(head < Q_HEADS, q_gamma, k_gamma)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
    output = x * scale * (1.0 + gamma)
    tl.store(qkv_ptr + row * ROW_STRIDE + head * D + offsets, output)


def is_supported(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> bool:
    return (
        qkv.is_cuda
        and torch.version.hip is None
        and qkv.dtype == torch.bfloat16
        and qkv.dim() == 2
        and qkv.is_contiguous()
        and q_gamma.dtype == torch.bfloat16
        and k_gamma.dtype == torch.bfloat16
        and q_gamma.device == qkv.device
        and k_gamma.device == qkv.device
        and q_gamma.shape == (head_dim,)
        and k_gamma.shape == (head_dim,)
        and q_gamma.is_contiguous()
        and k_gamma.is_contiguous()
        and q_heads > 0
        and kv_heads > 0
        and head_dim >= 32
        and head_dim <= 1024
        and head_dim & (head_dim - 1) == 0
        and qkv.shape[1] == (q_heads + 2 * kv_heads) * head_dim
    )


def fused_qk_rmsnorm_(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    eps: float,
) -> torch.Tensor:
    """Normalize Q and K in place, leaving V unchanged."""
    if not is_supported(qkv, q_gamma, k_gamma, q_heads, kv_heads, head_dim):
        raise ValueError("unsupported Qwen4 fused Q/K RMSNorm geometry or dtype")
    _qk_rmsnorm_kernel[(qkv.shape[0], q_heads + kv_heads)](
        qkv,
        q_gamma,
        k_gamma,
        ROW_STRIDE=qkv.stride(0),
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        D=head_dim,
        EPS=eps,
        num_warps=4,
    )
    return qkv
