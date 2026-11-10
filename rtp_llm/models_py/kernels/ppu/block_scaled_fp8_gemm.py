"""FP8 GEMM with a separate scale and FP32 add for every K128 dot.

This arithmetic is deliberately distinct from a GEMM that folds the block
scales into a longer dot or sums split-K partials. Small decode batches use
smaller M/N tiles without changing the order of those additions.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _block_scaled_fp8_gemm(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_id = pid // (32 * num_pid_n)
    first_pid_m = group_id * 32
    group_size_m = tl.minimum(num_pid_m - first_pid_m, 32)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % (32 * num_pid_n) // group_size_m
    rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    cols = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    kk = tl.arange(0, 128)
    a_ptr = A + rows[:, None] * K + kk[None, :]
    b_ptr = B + kk[:, None] + cols[None, :] * K
    as_ptr = As + rows * (K // 128)
    bs_ptr = Bs + (cols // 128) * (K // 128)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(K // 128):
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        a_scale = tl.load(as_ptr)
        b_scale = tl.load(bs_ptr)
        acc += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        a_ptr += 128
        b_ptr += 128
        # The scale block and dot both cover exactly 128 K values. Keeping
        # these constants avoids a dynamic modulo in every K iteration.
        as_ptr += 1
        bs_ptr += 1
    out_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        C + out_rows[:, None] * N + out_cols[None, :],
        acc.to(tl.bfloat16),
        (out_rows[:, None] < M) & (out_cols[None, :] < N),
    )


# Large-M baseline: retain the original dynamic parameters and pipeline.
# Constant-folding the scale step changed code generation and regressed
# prefill throughput on ZW-M890P, despite identical numerical results.
@triton.jit
def _prefill_block_scaled_fp8_gemm(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)
        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k
    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _tiles(m: int, n: int, k: int):
    # Offline ZW-M890P measurements; no autotuning during graph capture.
    # Prefill retains the reference's original 64x128 tile.
    if m > 64:
        return 64, 128, 3
    if n <= 128:
        return 16, 64, 3
    if n >= 8192 and m > 16:
        return (64 if m > 32 else 32), 128, 3
    if m > 32:
        return 32, 64, 3
    if k >= 16384 and m <= 16:
        return 16, 32, 4
    return 16, 64, 3


def fp8_block_scaled_matmul(
    q: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    weight_scales: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multiply row-major E4M3 matrices with row-major FP32 block scales."""
    if q.ndim != 2 or weight.ndim != 2:
        raise ValueError("block-scaled FP8 GEMM expects 2D inputs")
    m, k = q.shape
    n = weight.shape[0]
    if not n or not k or weight.shape[1] != k or k % 128:
        raise ValueError(
            "block-scaled FP8 GEMM requires positive N/K and K divisible by 128"
        )
    if (
        q.dtype != torch.float8_e4m3fn
        or weight.dtype != torch.float8_e4m3fn
        or not q.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("block-scaled FP8 GEMM requires contiguous E4M3 inputs")
    if (
        scales.dtype != torch.float32
        or weight_scales.dtype != torch.float32
        or scales.shape != (m, k // 128)
        or weight_scales.shape != (triton.cdiv(n, 128), k // 128)
        or not scales.is_contiguous()
        or not weight_scales.is_contiguous()
    ):
        raise ValueError(
            "block-scaled FP8 GEMM requires row-major FP32 K128/N128 scales"
        )
    if any(t.device != q.device for t in (weight, scales, weight_scales)):
        raise ValueError("block-scaled FP8 GEMM inputs must be on the same device")
    if out is None:
        out = torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
    elif (
        out.shape != (m, n)
        or out.dtype != torch.bfloat16
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError("block-scaled FP8 GEMM output must be contiguous BF16 [M, N]")
    if m > 64:
        _prefill_block_scaled_fp8_gemm[(triton.cdiv(m, 64) * triton.cdiv(n, 128),)](
            q,
            weight,
            out,
            scales,
            weight_scales,
            m,
            n,
            k,
            128,
            128,
            q.stride(0),
            q.stride(1),
            weight.stride(1),
            weight.stride(0),
            out.stride(0),
            out.stride(1),
            scales.stride(0),
            scales.stride(1),
            weight_scales.stride(1),
            weight_scales.stride(0),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=32,
            needs_masking=False,
            num_warps=4,
            num_stages=3,
        )
    elif m:
        bm, bn, stages = _tiles(m, n, k)
        _block_scaled_fp8_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            q,
            weight,
            out,
            scales,
            weight_scales,
            m,
            n,
            k,
            bm,
            bn,
            num_warps=4,
            num_stages=stages,
        )
    return out
