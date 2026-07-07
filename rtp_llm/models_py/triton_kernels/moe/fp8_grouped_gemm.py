"""Triton FP8 blockwise batched grouped GEMM for SM120 (consumer Blackwell).

Single batched kernel launch that processes all experts in parallel,
replacing the per-expert CUTLASS loop.

Tensor layouts:
  A:    [E, max_T, K]        float8_e4m3fn  (tokens, padded to alignment per expert)
  A_sf: [E, max_T, K//128]  float32         (blockwise row-major activation scales)
  B:    [E, N, K]            float8_e4m3fn  (weights stored as [E, N, K])
  B_sf: [E, N//128, K//128] float32         (blockwise weight scales)
  C:    [E, max_T, N]        bfloat16        (output)

Scale granularity matches BLOCK_K = BLOCK_N = 128, so each K-loop step
multiplies by exactly one A_sf scalar and one B_sf scalar.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_blockwise_batched_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_sf_ptr,
    b_sf_ptr,
    expert_num_tokens_ptr,
    max_num_tokens,
    N,
    K,
    # A strides: [E, max_T, K]
    stride_ae: tl.int64,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    # B strides: [E, N, K]
    stride_be: tl.int64,
    stride_bn: tl.int64,
    stride_bk: tl.int64,
    # C strides: [E, max_T, N]
    stride_ce: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    # A_sf strides: [E, max_T, K//128]
    stride_a_sf_e: tl.int64,
    stride_a_sf_m: tl.int64,
    stride_a_sf_k: tl.int64,
    # B_sf strides: [E, N//128, K//128]
    stride_b_sf_e: tl.int64,
    stride_b_sf_n: tl.int64,
    stride_b_sf_k: tl.int64,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens_ptr + expert_id)
    if e_num_tokens == 0:
        return

    pid_mn = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    cta_m_start = pid_m * BLOCK_M
    cta_n_start = pid_n * BLOCK_N
    if cta_m_start >= e_num_tokens:
        return

    cta_m_size = tl.minimum(BLOCK_M, e_num_tokens - cta_m_start)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < cta_m_size

    # A: [E, max_T, K] — row-major
    a_base = a_ptr + expert_id * stride_ae + cta_m_start * stride_am
    a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    # B: [E, N, K] — for NT GEMM, B[n, k], stride along n = K, stride along k = 1
    b_base = b_ptr + expert_id * stride_be + cta_n_start * stride_bn
    b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # A_sf: [E, max_T, K//128]
    a_sf_base = a_sf_ptr + expert_id * stride_a_sf_e + cta_m_start * stride_a_sf_m
    # shape: [BLOCK_M, 1] base — K-group index added per loop step
    a_sf_ptrs = a_sf_base + offs_m[:, None] * stride_a_sf_m

    # B_sf: [E, N//128, K//128]
    b_sf_base = b_sf_ptr + expert_id * stride_b_sf_e + pid_n * stride_b_sf_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        # Native FP8 tensor-core MMA on SM120 (tcgen05.mma), accumulates in f32.
        tile_acc = tl.dot(a, b)

        # A scales: [BLOCK_M, 1] — per-row scale for this K-group
        a_scales = tl.load(
            a_sf_ptrs + k * stride_a_sf_k,
            mask=mask_m[:, None],
            other=1.0,
        )
        # B scale: scalar for this (N-group, K-group)
        b_scale = tl.load(b_sf_base + k * stride_b_sf_k)

        acc += tile_acc * a_scales * b_scale

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_base = (
        c_ptr
        + expert_id * stride_ce
        + cta_m_start * stride_cm
        + cta_n_start * stride_cn
    )
    c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & (offs_n[None, :] < N - cta_n_start)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def invoke_sm120_fp8_grouped_gemm(
    A: torch.Tensor,  # [E, max_T, K]  float8_e4m3fn
    A_sf: torch.Tensor,  # [E, max_T, K//128]  float32
    B: torch.Tensor,  # [E, N, K]  float8_e4m3fn
    B_sf: torch.Tensor,  # [E, N//128, K//128]  float32
    expert_num_tokens: torch.Tensor,  # [E]  int32
    C: torch.Tensor,  # [E, max_T, N]  bfloat16  (pre-allocated output)
    BLOCK_M: int = 16,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
) -> None:
    """Launch the FP8 blockwise batched grouped GEMM Triton kernel.

    Computes C[e] = A[e] @ B[e].T * A_sf[e] * B_sf[e] for each expert e,
    using blockwise scales with granularity (128, 128).
    Only processes rows up to expert_num_tokens[e] per expert; remaining rows
    in the padded allocation are left untouched.
    """
    E = A.size(0)
    max_T = A.size(1)
    K = A.size(2)
    N = B.size(1)

    assert A.dtype == torch.float8_e4m3fn, f"A must be float8_e4m3fn, got {A.dtype}"
    assert B.dtype == torch.float8_e4m3fn, f"B must be float8_e4m3fn, got {B.dtype}"
    assert A_sf.dtype == torch.float32, f"A_sf must be float32, got {A_sf.dtype}"
    assert B_sf.dtype == torch.float32, f"B_sf must be float32, got {B_sf.dtype}"
    assert C.dtype == torch.bfloat16, f"C must be bfloat16, got {C.dtype}"
    assert B.size(2) == K, f"B K-dim mismatch: B.size(2)={B.size(2)} vs K={K}"
    assert A_sf.shape == (
        E,
        max_T,
        K // BLOCK_K,
    ), f"A_sf shape mismatch: {A_sf.shape} vs expected {(E, max_T, K // BLOCK_K)}"
    assert B_sf.shape == (
        E,
        N // BLOCK_N,
        K // BLOCK_K,
    ), f"B_sf shape mismatch: {B_sf.shape} vs expected {(E, N // BLOCK_N, K // BLOCK_K)}"
    assert C.shape == (
        E,
        max_T,
        N,
    ), f"C shape mismatch: {C.shape} vs expected {(E, max_T, N)}"

    grid = (
        E,
        triton.cdiv(max_T, BLOCK_M) * triton.cdiv(N, BLOCK_N),
    )

    _fp8_blockwise_batched_kernel[grid](
        A,
        B,
        C,
        A_sf,
        B_sf,
        expert_num_tokens,
        max_T,
        N,
        K,
        # A strides
        A.stride(0),
        A.stride(1),
        A.stride(2),
        # B strides: [E, N, K]
        B.stride(0),
        B.stride(1),
        B.stride(2),
        # C strides
        C.stride(0),
        C.stride(1),
        C.stride(2),
        # A_sf strides
        A_sf.stride(0),
        A_sf.stride(1),
        A_sf.stride(2),
        # B_sf strides
        B_sf.stride(0),
        B_sf.stride(1),
        B_sf.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
