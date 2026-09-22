"""Triton MegaMoE input packer.

This fuses the hot pre-Mega chain: BF16 activation -> FP8 E4M3, packed UE8M0
group-32 scales, and router tensor copies into DeepGEMM's symmetric-memory
dispatch buffer.  It mirrors ``_per_token_cast_to_fp8_packed_ue8m0`` but writes
directly into the final buffer instead of materializing temporary tensors.
"""

from __future__ import annotations

import os

import torch

from .input_pack_options import mask_pack_routes, validate_pack_options

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["M"])
    def _pack_x_kernel(
        x_ptr,
        out_fp8_ptr,
        out_sf_ptr,
        M,
        N: tl.constexpr,
        x_stride_m: tl.constexpr,
        out_stride_m: tl.constexpr,
        sf_stride_m: tl.constexpr,
        eps: tl.constexpr,
        fp8_max: tl.constexpr,
    ):
        pid_m = tl.program_id(0).to(tl.int64)
        pid_blk = tl.program_id(1)
        offs = tl.arange(0, 128)
        col = pid_blk * 128 + offs
        mask = (pid_m < M) & (col < N)
        x = tl.load(x_ptr + pid_m * x_stride_m + col, mask=mask, other=0.0).to(
            tl.float32
        )
        x_2d = tl.reshape(tl.abs(x), (4, 32))
        block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
        scale = tl.math.exp2(tl.ceil(tl.log2(block_absmax / fp8_max)))
        scale_exp = tl.reshape(
            tl.broadcast_to(tl.reshape(scale, (4, 1)), (4, 32)),
            (128,),
        )
        q = tl.clamp(x / scale_exp, -fp8_max, fp8_max).to(tl.float8e4nv)
        tl.store(out_fp8_ptr + pid_m * out_stride_m + col, q, mask=mask)

        scale_bits = scale.to(tl.int32, bitcast=True)
        group_offsets = tl.arange(0, 4)
        ue8m0 = (scale_bits >> 23) & 0xFF
        packed = tl.sum(ue8m0 << (group_offsets * 8))
        tl.store(out_sf_ptr + pid_m * sf_stride_m + pid_blk, packed, mask=pid_m < M)

    @triton.jit(do_not_specialize=["M"])
    def _pack_router_kernel(
        weights_ptr,
        indices_ptr,
        out_weights_ptr,
        out_indices_ptr,
        M,
        K: tl.constexpr,
        weights_stride_m: tl.constexpr,
        indices_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_K)
        mask = (pid < M) & (offs < K)
        w = tl.load(
            weights_ptr + pid * weights_stride_m + offs, mask=mask, other=0.0
        ).to(tl.float32)
        idx = tl.load(
            indices_ptr + pid * indices_stride_m + offs, mask=mask, other=0
        ).to(tl.int64)
        tl.store(out_weights_ptr + pid * out_weights_stride_m + offs, w, mask=mask)
        tl.store(out_indices_ptr + pid * out_indices_stride_m + offs, idx, mask=mask)

    @triton.jit(do_not_specialize=["M"])
    def _pack_mega_moe_inputs_optimized_kernel(
        x_ptr,
        weights_ptr,
        indices_ptr,
        out_fp8_ptr,
        out_sf_ptr,
        out_weights_ptr,
        out_indices_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        x_stride_m: tl.constexpr,
        weights_stride_m: tl.constexpr,
        indices_stride_m: tl.constexpr,
        out_stride_m: tl.constexpr,
        sf_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        eps: tl.constexpr,
        fp8_max: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        valid_mask_ptr,
        valid_count,
        HAS_COUNT: tl.constexpr,
        HAS_MASK: tl.constexpr,
        shared_ptr,
        shared_out_ptr,
        SHARED_D: tl.constexpr,
        shared_stride: tl.constexpr,
        shared_out_stride: tl.constexpr,
    ):
        pid_m_blk = tl.program_id(0).to(tl.int64)
        pid_blk = tl.program_id(1)

        offs_m = pid_m_blk * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        offs_32 = tl.arange(0, 32)
        row_mask = offs_m < M
        packed = tl.zeros((BLOCK_M,), dtype=tl.int32)

        for pack_idx in tl.static_range(4):
            cols = pid_blk * 128 + pack_idx * 32 + offs_32
            mask = row_mask[:, None] & (cols[None, :] < N)
            x = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            block_absmax = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
            scale_raw = block_absmax / fp8_max
            scale_raw_bits = scale_raw.to(tl.int32, bitcast=True)
            exp = ((scale_raw_bits >> 23) & 0xFF) + ((scale_raw_bits & 0x7FFFFF) != 0)
            exp = tl.minimum(tl.maximum(exp, 1), 254)
            scale_bits = exp << 23
            scale = scale_bits.to(tl.float32, bitcast=True)

            q = tl.clamp(x / scale[:, None], -fp8_max, fp8_max).to(tl.float8e4nv)
            tl.store(
                out_fp8_ptr + offs_m[:, None] * out_stride_m + cols[None, :],
                q,
                mask=mask,
            )
            packed = packed | (exp << (pack_idx * 8))

        tl.store(
            out_sf_ptr + offs_m * sf_stride_m + pid_blk,
            packed,
            mask=row_mask,
        )

        if pid_blk == 0:
            router_offs = tl.arange(0, BLOCK_K)
            router_mask = row_mask[:, None] & (router_offs[None, :] < K)
            w = tl.load(
                weights_ptr + offs_m[:, None] * weights_stride_m + router_offs[None, :],
                mask=router_mask,
                other=0.0,
            ).to(tl.float32)
            idx = tl.load(
                indices_ptr + offs_m[:, None] * indices_stride_m + router_offs[None, :],
                mask=router_mask,
                other=0,
            ).to(tl.int64)
            if HAS_COUNT or HAS_MASK:
                valid = offs_m < valid_count
                if HAS_MASK:
                    valid = valid & (
                        tl.load(valid_mask_ptr + offs_m, row_mask, other=0) != 0
                    )
                w = tl.where(valid[:, None], w, 0.0)
                idx = tl.where(valid[:, None], idx, 0)
            tl.store(
                out_weights_ptr
                + offs_m[:, None] * out_weights_stride_m
                + router_offs[None, :],
                w,
                mask=router_mask,
            )
            tl.store(
                out_indices_ptr
                + offs_m[:, None] * out_indices_stride_m
                + router_offs[None, :],
                idx,
                mask=router_mask,
            )

        if SHARED_D > 0:
            for chunk in tl.static_range((SHARED_D + N - 1) // N):
                shared_cols = chunk * N + pid_blk * 128 + tl.arange(0, 128)
                shared_mask = row_mask[:, None] & (shared_cols[None, :] < SHARED_D)
                shared = tl.load(
                    shared_ptr + offs_m[:, None] * shared_stride + shared_cols[None, :],
                    shared_mask,
                    other=0,
                )
                tl.store(
                    shared_out_ptr
                    + offs_m[:, None] * shared_out_stride
                    + shared_cols[None, :],
                    shared,
                    shared_mask,
                )


def _validate_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_sf: torch.Tensor,
) -> tuple[int, int, int]:
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if not x.is_cuda:
        raise RuntimeError("fused MegaMoE input packer requires CUDA tensors")
    if x.dim() != 2:
        raise ValueError(f"x must be [T,D], got {tuple(x.shape)}")
    if weights.shape != indices.shape:
        raise ValueError("weights and indices must have identical [T,topk] shape")
    if weights.dtype != torch.float32:
        raise ValueError(f"weights must be float32, got {weights.dtype}")
    if indices.dtype != torch.int64:
        raise ValueError(f"indices must be int64, got {indices.dtype}")
    T, D = x.shape
    if D % 128 != 0:
        raise ValueError(f"fused MegaMoE packer requires D % 128 == 0, got D={D}")
    if out_sf.shape[1] != D // 128:
        raise ValueError(
            f"out_sf shape mismatch: expected second dim {D // 128}, got {out_sf.shape}"
        )
    return T, D, weights.shape[1]


def fused_pack_mega_moe_inputs_legacy(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    *,
    valid_token_count=None,
    valid_token_mask=None,
    shared_input=None,
    shared_out=None,
) -> None:
    T, D, topk = _validate_inputs(x, weights, indices, out_sf)
    validate_pack_options(
        T, valid_token_count, valid_token_mask, shared_input, shared_out
    )
    indices, weights = mask_pack_routes(
        indices, weights, valid_token_count, valid_token_mask
    )
    if T == 0:
        return
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    grid_x = (T, triton.cdiv(D, 128))
    _pack_x_kernel[grid_x](
        x,
        out_fp8,
        out_sf,
        T,
        D,
        x.stride(0),
        out_fp8.stride(0),
        out_sf.stride(0),
        1.0e-4,
        fp8_max,
        num_warps=4,
    )
    block_k = triton.next_power_of_2(topk)
    _pack_router_kernel[(T,)](
        weights,
        indices,
        out_weights,
        out_indices,
        T,
        topk,
        weights.stride(0),
        indices.stride(0),
        out_weights.stride(0),
        out_indices.stride(0),
        BLOCK_K=block_k,
        num_warps=1,
    )
    if shared_input is not None:
        shared_out[:T].copy_(shared_input)


def fused_pack_mega_moe_inputs_optimized(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    *,
    valid_token_count=None,
    valid_token_mask=None,
    shared_input=None,
    shared_out=None,
) -> None:
    T, D, topk = _validate_inputs(x, weights, indices, out_sf)
    validate_pack_options(
        T, valid_token_count, valid_token_mask, shared_input, shared_out
    )
    if shared_input is not None and (
        shared_input.device != x.device
        or shared_input.dtype != torch.bfloat16
        or shared_input.stride(1) != 1
        or shared_out.stride(1) != 1
    ):
        raise ValueError(
            "fused shared staging requires same-device BF16 tensors with contiguous columns"
        )
    if valid_token_mask is not None:
        valid_token_mask = valid_token_mask.to(
            device=x.device, dtype=torch.bool
        ).contiguous()
    if T == 0:
        return
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    block_k = triton.next_power_of_2(topk)
    block_m_env = os.environ.get("DSV4_MEGA_MOE_PACK_BLOCK_M")
    block_m = int(block_m_env) if block_m_env is not None else (8 if T >= 2048 else 2)
    if block_m not in (1, 2, 4, 8):
        raise ValueError(
            f"invalid DSV4_MEGA_MOE_PACK_BLOCK_M={block_m}; expected 1, 2, 4, or 8"
        )
    grid = (triton.cdiv(T, block_m), triton.cdiv(D, 128))
    _pack_mega_moe_inputs_optimized_kernel[grid](
        x,
        weights,
        indices,
        out_fp8,
        out_sf,
        out_weights,
        out_indices,
        T,
        D,
        topk,
        x.stride(0),
        weights.stride(0),
        indices.stride(0),
        out_fp8.stride(0),
        out_sf.stride(0),
        out_weights.stride(0),
        out_indices.stride(0),
        1.0e-4,
        fp8_max,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        valid_mask_ptr=valid_token_mask,
        valid_count=T if valid_token_count is None else valid_token_count,
        HAS_COUNT=valid_token_count is not None,
        HAS_MASK=valid_token_mask is not None,
        shared_ptr=shared_input,
        shared_out_ptr=shared_out,
        SHARED_D=0 if shared_input is None else shared_input.shape[1],
        shared_stride=0 if shared_input is None else shared_input.stride(0),
        shared_out_stride=0 if shared_out is None else shared_out.stride(0),
        num_warps=4,
    )


def fused_pack_mega_moe_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    *,
    valid_token_count=None,
    valid_token_mask=None,
    shared_input=None,
    shared_out=None,
) -> None:
    impl = os.environ.get("DSV4_MEGA_MOE_INPUT_PACKER_IMPL", "optimized").lower()
    if impl == "legacy":
        pack = fused_pack_mega_moe_inputs_legacy
    elif impl == "optimized":
        pack = fused_pack_mega_moe_inputs_optimized
    else:
        raise ValueError(
            f"invalid DSV4_MEGA_MOE_INPUT_PACKER_IMPL={impl!r}; expected legacy|optimized"
        )
    return pack(
        x,
        weights,
        indices,
        out_fp8,
        out_sf,
        out_indices,
        out_weights,
        valid_token_count=valid_token_count,
        valid_token_mask=valid_token_mask,
        shared_input=shared_input,
        shared_out=shared_out,
    )
