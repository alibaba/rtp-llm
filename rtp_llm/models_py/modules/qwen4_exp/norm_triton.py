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


@triton.jit
def _small_qk_rmsnorm_kernel(
    QKV, QG, KG, ROWS: tl.constexpr, STRIDE: tl.constexpr, EPS: tl.constexpr
):
    row, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, 32)
    column = lane * 4
    base = row * STRIDE + head * 256
    a0 = tl.load(QKV + base + column).to(tl.float32)
    a1 = tl.load(QKV + base + column + 1).to(tl.float32)
    a2 = tl.load(QKV + base + column + 2).to(tl.float32)
    a3 = tl.load(QKV + base + column + 3).to(tl.float32)
    b0 = tl.load(QKV + base + column + 128).to(tl.float32)
    b1 = tl.load(QKV + base + column + 129).to(tl.float32)
    b2 = tl.load(QKV + base + column + 130).to(tl.float32)
    b3 = tl.load(QKV + base + column + 131).to(tl.float32)
    s0, s1, s2, s3 = a0 * a0, a1 * a1, a2 * a2, a3 * a3
    t0, t1, t2, t3 = b0 * b0, b1 * b1, b2 * b2, b3 * b3
    # CUDA mean uses 64 reduction lanes for fewer than 16 outputs, otherwise
    # 32. Q and K are separate reductions with three and one output per row.
    wide = ((head < 3) & (ROWS * 3 < 16)) | ((head >= 3) & (ROWS < 16))
    sum_wide = (((s0 + s1) + s2) + s3) + (((t0 + t1) + t2) + t3)
    sum_narrow = (((s0 + t0) + (s1 + t1)) + (s2 + t2)) + (s3 + t3)
    variance = tl.sum(tl.where(wide, sum_wide, sum_narrow), 0) / 256
    scale = tl.rsqrt(variance + EPS)
    for j in tl.static_range(4):
        for half in tl.static_range(2):
            rd = column + j + half * 128
            x = tl.load(QKV + base + rd).to(tl.float32)
            qg = tl.load(QG + rd).to(tl.float32)
            kg = tl.load(KG + rd).to(tl.float32)
            g = tl.where(head < 3, qg, kg)
            tl.store(QKV + base + rd, (x * scale) * (1.0 + g))


def is_small_qk_supported(qkv, q_gamma, k_gamma, q_heads, kv_heads, head_dim):
    return (
        qkv.is_cuda
        and torch.version.hip is None
        and qkv.dtype == torch.bfloat16
        and qkv.dim() == 2
        and qkv.is_contiguous()
        and 1 <= qkv.shape[0] <= 32
        and qkv.shape[1] == 1280
        and (q_heads, kv_heads, head_dim) == (3, 1, 256)
        and q_gamma.device == k_gamma.device == qkv.device
        and q_gamma.dtype in (torch.bfloat16, torch.float32)
        and k_gamma.dtype in (torch.bfloat16, torch.float32)
        and q_gamma.shape == k_gamma.shape == (256,)
        and q_gamma.is_contiguous()
        and k_gamma.is_contiguous()
    )


def small_qk_rmsnorm_(qkv, q_gamma, k_gamma, q_heads, kv_heads, head_dim, eps):
    if not is_small_qk_supported(qkv, q_gamma, k_gamma, q_heads, kv_heads, head_dim):
        raise ValueError("unsupported small Qwen4 Q/K RMSNorm layout")
    _small_qk_rmsnorm_kernel[(qkv.shape[0], 4)](
        qkv,
        q_gamma,
        k_gamma,
        ROWS=qkv.shape[0],
        STRIDE=qkv.stride(0),
        EPS=eps,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return qkv
