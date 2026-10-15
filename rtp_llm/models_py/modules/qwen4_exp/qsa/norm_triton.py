"""Graph-capturable Qwen4 Q/K RMSNorm with FP32 ``1 + gamma`` gains."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _indexer_rmsnorm(
    X,
    GAMMA,
    OUT,
    HEADS: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, 32)
    column = lane * 4
    offset = row // HEADS * TOKEN_STRIDE + row % HEADS * HEAD_STRIDE
    a = tl.load(X + offset + column).to(tl.float32)
    b = tl.load(X + offset + column + 1).to(tl.float32)
    c = tl.load(X + offset + column + 2).to(tl.float32)
    d = tl.load(X + offset + column + 3).to(tl.float32)
    variance = tl.sum(((a * a + b * b) + c * c) + d * d, 0) / 128.0
    inverse = tl.rsqrt(variance + EPS)
    for j in tl.static_range(4):
        value = tl.load(X + offset + column + j).to(tl.float32)
        gain = 1.0 + tl.load(GAMMA + column + j).to(tl.float32)
        tl.store(OUT + row * 128 + column + j, (value * inverse) * gain)


def try_indexer_rmsnorm(x, gamma, eps):
    if not (
        x.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(x.device) == (12, 0)
        and x.dtype == torch.bfloat16
        and x.ndim in (2, 3, 4)
        and x.shape[-1] == 128
        and 1 <= x.numel() // 128 <= 128
        and x.stride(-1) == 1
        and all(stride > 0 for stride in x.stride())
        and (x.ndim != 4 or x.stride(0) == x.shape[1] * x.stride(1))
        and gamma.shape == (128,)
        and gamma.dtype in (torch.bfloat16, torch.float32)
        and gamma.device == x.device
        and gamma.is_contiguous()
        and not (x.requires_grad or gamma.requires_grad)
        and isinstance(eps, (float, int))
        and math.isfinite(eps)
        and eps >= 0
    ):
        return None
    heads = x.shape[-2] if x.ndim > 2 else 1
    token_stride = x.stride(-3) if x.ndim > 2 else x.stride(0)
    head_stride = x.stride(-2) if x.ndim > 2 else 128
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _indexer_rmsnorm[(x.numel() // 128,)](
        x,
        gamma,
        out,
        HEADS=heads,
        TOKEN_STRIDE=token_stride,
        HEAD_STRIDE=head_stride,
        EPS=eps,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out


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
