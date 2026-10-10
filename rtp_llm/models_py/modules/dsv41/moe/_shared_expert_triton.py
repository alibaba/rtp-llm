"""Triton helpers for DSV4 shared expert quantization and output combine."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["M", "out_scale_stride_k"])
    def _bf16_to_fp8_packed_ue8m0_kernel(
        x_ptr,
        out_q_ptr,
        out_scale_ptr,
        M,
        x_stride_m,
        out_q_stride_m,
        out_scale_stride_k,
        clamp_eps: tl.constexpr,
        fp8_min: tl.constexpr,
        fp8_max: tl.constexpr,
        N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_pack = tl.program_id(0).to(tl.int64)
        pid_m = tl.program_id(1).to(tl.int64)
        m_offset = pid_m * BLOCK_M
        if m_offset >= M:
            return

        offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
        offs_n = tl.arange(0, GROUP_SIZE)
        row_mask = (m_offset + offs_m) < M
        base_in = (m_offset + offs_m[:, None]) * x_stride_m
        base_out = (m_offset + offs_m[:, None]) * out_q_stride_m

        packed_scale = tl.zeros((BLOCK_M,), dtype=tl.int32)
        for pack_idx in tl.static_range(4):
            group_id = pid_pack * 4 + pack_idx
            if group_id < NUM_GROUPS:
                n_offset = group_id * GROUP_SIZE
                cols = n_offset + offs_n
                mask = row_mask[:, None] & (cols[None, :] < N)
                x = tl.load(x_ptr + base_in + cols[None, :], mask=mask, other=0.0).to(
                    tl.float32
                )
                x_is_finite = tl.abs(x) < float("inf")
                x = tl.where(x_is_finite, x, 0.0)
                absmax = tl.max(tl.abs(x), axis=1)
                scale_raw = tl.maximum(absmax, clamp_eps) / fp8_max
                exponent = tl.ceil(tl.log2(scale_raw))
                scale = tl.math.exp2(exponent)
                q = tl.clamp(x / scale[:, None], fp8_min, fp8_max)
                tl.store(
                    out_q_ptr + base_out + cols[None, :],
                    q.to(out_q_ptr.dtype.element_ty),
                    mask=mask,
                )
                exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
                packed_scale = packed_scale | (exponent_biased << (pack_idx * 8))

        scale_ptrs = out_scale_ptr + pid_pack * out_scale_stride_k + m_offset + offs_m
        tl.store(scale_ptrs, packed_scale, mask=row_mask)

    @triton.jit(do_not_specialize=["M"])
    def _add_cast_kernel(
        routed_ptr,
        shared_ptr,
        out_ptr,
        M,
        N: tl.constexpr,
        routed_stride_m: tl.constexpr,
        routed_stride_n: tl.constexpr,
        shared_stride_m: tl.constexpr,
        shared_stride_n: tl.constexpr,
        out_stride_m: tl.constexpr,
        out_stride_n: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < (M * N)
        rows = offs // N
        cols = offs - rows * N
        routed = tl.load(
            routed_ptr + rows * routed_stride_m + cols * routed_stride_n,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        shared = tl.load(
            shared_ptr + rows * shared_stride_m + cols * shared_stride_n,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            out_ptr + rows * out_stride_m + cols * out_stride_n,
            routed + shared,
            mask=mask,
        )


def quant_bf16_fp8_packed_ue8m0(
    x: torch.Tensor,
    out_q: torch.Tensor,
    out_scale: torch.Tensor,
    group_size: int = 128,
    eps: float = 1.0e-4,
) -> None:
    M, N = x.shape
    expected_scale_cols = (N // group_size + 3) // 4
    if M == 0:
        return
    finfo = torch.finfo(torch.float8_e4m3fn)
    block_m = 8
    grid = (expected_scale_cols, triton.cdiv(M, block_m))
    _bf16_to_fp8_packed_ue8m0_kernel[grid](
        x,
        out_q,
        out_scale,
        M,
        x.stride(0),
        out_q.stride(0),
        out_scale.stride(1),
        eps,
        finfo.min,
        finfo.max,
        N=N,
        GROUP_SIZE=group_size,
        NUM_GROUPS=N // group_size,
        BLOCK_M=block_m,
        num_warps=max(4, group_size // 32),
        num_stages=2,
    )


def fused_moe_epilogue(
    routed: torch.Tensor,
    shared: torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(routed.shape, dtype=out_dtype, device=routed.device)
    M, N = routed.shape
    if M * N == 0:
        return out
    block = 1024
    n_elements = M * N
    _add_cast_kernel[(triton.cdiv(n_elements, block),)](
        routed,
        shared,
        out,
        M,
        N,
        routed.stride(0),
        routed.stride(1),
        shared.stride(0),
        shared.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK=block,
        num_warps=4,
    )
    return out


def fused_add_cast_bf16(
    routed: torch.Tensor,
    shared: torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return fused_moe_epilogue(routed, shared, out_dtype, out=out)
