"""Fused BF16-to-NVFP4 input packing for DeepGEMM MegaMoE."""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _cast_ue4m3_nearest(value):
        # Include E4M3 subnormals, matching DeepGEMM's scale cast. Clamping
        # to the minimum normal (1/64) also changes FP4 values in tiny groups.
        value = tl.minimum(tl.maximum(value, 1.0 / 512.0), 448.0)
        fp8 = value.to(tl.float8e4nv, fp_downcast_rounding="rtne")
        return fp8.to(tl.uint8, bitcast=True).to(tl.int32), fp8.to(tl.float32)

    @triton.jit
    def _e2m1_code(value):
        abs_value = tl.minimum(tl.abs(value), 6.0)
        # E2M1 round-to-nearest-even midpoint decisions, matching torchao.
        code = (abs_value > 0.25).to(tl.int32)
        code += (abs_value >= 0.75).to(tl.int32)
        code += (abs_value > 1.25).to(tl.int32)
        code += (abs_value >= 1.75).to(tl.int32)
        code += (abs_value > 2.5).to(tl.int32)
        code += (abs_value >= 3.5).to(tl.int32)
        code += (abs_value > 5.0).to(tl.int32)
        sign = (value.to(tl.int32, bitcast=True) >> 31) & 1
        return code | (sign.to(tl.int32) << 3)

    @triton.jit(do_not_specialize=["M"])
    def _row_gsf_kernel(
        x_ptr,
        out_gsf_ptr,
        M,
        D: tl.constexpr,
        x_stride_m: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_D)
        mask = (row < M) & (cols < D)
        values = tl.load(x_ptr + row * x_stride_m + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        row_amax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-30)
        tl.store(out_gsf_ptr + row, row_amax / (6.0 * 448.0), mask=row < M)

    @triton.jit(do_not_specialize=["M"])
    def _pack_nvfp4_inputs_kernel(
        x_ptr,
        weights_ptr,
        indices_ptr,
        out_fp4_ptr,
        out_sf_ptr,
        out_gsf_ptr,
        out_indices_ptr,
        out_weights_ptr,
        M,
        D: tl.constexpr,
        TOPK: tl.constexpr,
        x_stride_m: tl.constexpr,
        weights_stride_m: tl.constexpr,
        indices_stride_m: tl.constexpr,
        out_fp4_stride_m: tl.constexpr,
        out_sf_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
    ):
        row_block = tl.program_id(0).to(tl.int64)
        dim_block = tl.program_id(1)
        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        row_mask = rows < M
        gsf = tl.load(out_gsf_ptr + rows, mask=row_mask, other=1.0)
        packed_sf = tl.zeros((BLOCK_M,), dtype=tl.int32)

        for scale_group in tl.static_range(4):
            group_cols = dim_block * 64 + scale_group * 16 + tl.arange(0, 16)
            group_mask = row_mask[:, None] & (group_cols[None, :] < D)
            group_values = tl.load(
                x_ptr + rows[:, None] * x_stride_m + group_cols[None, :],
                mask=group_mask,
                other=0.0,
            ).to(tl.float32)
            group_amax = tl.max(tl.abs(group_values), axis=1)
            sf_code, sf_value = _cast_ue4m3_nearest(tl.div_rn(group_amax / 6.0, gsf))
            packed_sf = packed_sf | (sf_code << (scale_group * 8))
            scale_inv = tl.div_rn(tl.div_rn(1.0, gsf), sf_value)

            # Reuse the loaded group; even features occupy the low nibble.
            codes = _e2m1_code(
                tl.maximum(tl.minimum(group_values * scale_inv[:, None], 6.0), -6.0)
            )
            code0, code1 = tl.split(tl.reshape(codes, (BLOCK_M, 8, 2)))
            packed_fp4 = code0 | (code1 << 4)
            out_cols = dim_block * 32 + scale_group * 8 + tl.arange(0, 8)
            tl.store(
                out_fp4_ptr + rows[:, None] * out_fp4_stride_m + out_cols[None, :],
                packed_fp4,
                mask=row_mask[:, None],
            )

        tl.store(
            out_sf_ptr + rows * out_sf_stride_m + dim_block,
            packed_sf,
            mask=row_mask,
        )

        if dim_block == 0:
            router_cols = tl.arange(0, BLOCK_TOPK)
            router_mask = row_mask[:, None] & (router_cols[None, :] < TOPK)
            router_weights = tl.load(
                weights_ptr + rows[:, None] * weights_stride_m + router_cols[None, :],
                mask=router_mask,
                other=0.0,
            ).to(tl.float32)
            router_indices = tl.load(
                indices_ptr + rows[:, None] * indices_stride_m + router_cols[None, :],
                mask=router_mask,
                other=-1,
            ).to(tl.int64)
            tl.store(
                out_weights_ptr
                + rows[:, None] * out_weights_stride_m
                + router_cols[None, :],
                router_weights,
                mask=router_mask,
            )
            tl.store(
                out_indices_ptr
                + rows[:, None] * out_indices_stride_m
                + router_cols[None, :],
                router_indices,
                mask=router_mask,
            )

    @triton.jit(do_not_specialize=["M"])
    def _pack_nvfp4_inputs_vector_kernel(
        x_ptr,
        weights_ptr,
        indices_ptr,
        out_fp4_ptr,
        out_sf_ptr,
        out_gsf_ptr,
        out_indices_ptr,
        out_weights_ptr,
        M,
        D: tl.constexpr,
        TOPK: tl.constexpr,
        x_stride_m: tl.constexpr,
        weights_stride_m: tl.constexpr,
        indices_stride_m: tl.constexpr,
        out_fp4_stride_m: tl.constexpr,
        out_sf_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
    ):
        # Decode four independent block scales together, preserving div_rn,
        # E4M3 rounding and the E2M1 ladder of the original packer.
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(
            tl.int64
        )
        dim_block = tl.program_id(1)
        cols = dim_block * 64 + tl.arange(0, 64)
        row_mask = rows < M
        values = tl.load(
            x_ptr + rows[:, None] * x_stride_m + cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        groups = tl.reshape(values, (BLOCK_M, 4, 16))
        gsf = tl.load(out_gsf_ptr + rows, mask=row_mask, other=1.0)
        group_amax = tl.max(tl.abs(groups), axis=2)
        sf_code, sf_value = _cast_ue4m3_nearest(
            tl.div_rn(group_amax / 6.0, gsf[:, None])
        )
        scale_inv = tl.div_rn(tl.div_rn(1.0, gsf[:, None]), sf_value)
        codes = _e2m1_code(
            tl.maximum(tl.minimum(groups * scale_inv[:, :, None], 6.0), -6.0)
        )
        code0, code1 = tl.split(tl.reshape(codes, (BLOCK_M, 32, 2)))
        tl.store(
            out_fp4_ptr
            + rows[:, None] * out_fp4_stride_m
            + dim_block * 32
            + tl.arange(0, 32)[None, :],
            code0 | (code1 << 4),
            mask=row_mask[:, None],
        )
        # Positive finite E4M3 codes are <=126; four disjoint bytes fit int32.
        packed_sf = tl.sum(sf_code << (tl.arange(0, 4)[None, :] * 8), axis=1)
        tl.store(
            out_sf_ptr + rows * out_sf_stride_m + dim_block, packed_sf, mask=row_mask
        )
        if dim_block == 0:
            route = tl.arange(0, BLOCK_TOPK)
            route_mask = row_mask[:, None] & (route[None, :] < TOPK)
            weights = tl.load(
                weights_ptr + rows[:, None] * weights_stride_m + route[None, :],
                mask=route_mask,
                other=0.0,
            ).to(tl.float32)
            indices = tl.load(
                indices_ptr + rows[:, None] * indices_stride_m + route[None, :],
                mask=route_mask,
                other=-1,
            ).to(tl.int64)
            tl.store(
                out_weights_ptr + rows[:, None] * out_weights_stride_m + route[None, :],
                weights,
                mask=route_mask,
            )
            tl.store(
                out_indices_ptr + rows[:, None] * out_indices_stride_m + route[None, :],
                indices,
                mask=route_mask,
            )


def _validate_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp4: torch.Tensor,
    out_sf: torch.Tensor,
    out_gsf: torch.Tensor,
) -> tuple[int, int, int]:
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 2:
        raise ValueError("x must be a CUDA BF16 [T,D] tensor")
    if weights.dtype != torch.float32 or indices.dtype != torch.int64:
        raise ValueError("weights/indices must be float32/int64")
    if weights.shape != indices.shape or weights.size(0) != x.size(0):
        raise ValueError("weights and indices must match x on [T,topk]")
    tokens, hidden = x.shape
    if hidden % 128 != 0:
        raise ValueError(f"NVFP4 MegaMoE packer requires D % 128 == 0, got {hidden}")
    if out_fp4.shape != (tokens, hidden // 2):
        raise ValueError(f"out_fp4 shape mismatch: {tuple(out_fp4.shape)}")
    if out_sf.shape != (tokens, hidden // 64):
        raise ValueError(f"out_sf shape mismatch: {tuple(out_sf.shape)}")
    if out_gsf.dtype != torch.float32 or out_gsf.shape != (tokens,):
        raise ValueError(
            "out_gsf must be float32 [T], got "
            f"dtype={out_gsf.dtype}, shape={tuple(out_gsf.shape)}"
        )
    return tokens, hidden, weights.size(1)


def fused_pack_mega_nvfp4_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp4: torch.Tensor,
    out_sf: torch.Tensor,
    out_gsf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> None:
    tokens, hidden, topk = _validate_inputs(
        x, weights, indices, out_fp4, out_sf, out_gsf
    )
    if tokens == 0:
        return
    block_m_env = os.environ.get("GLM5_MEGA_MOE_NVFP4_PACK_BLOCK_M")
    # Keep the large-chunk policy and widen the measured decode shapes.
    # Other small shapes retain tile 4; explicit overrides take precedence.
    use_wide_tile = tokens >= 1024 or (
        hidden == 6144 and topk == 4 and tokens in (80, 96, 112, 128)
    )
    block_m = (
        int(block_m_env) if block_m_env is not None else (16 if use_wide_tile else 4)
    )
    if block_m not in (1, 2, 4, 8, 16):
        raise ValueError("GLM5_MEGA_MOE_NVFP4_PACK_BLOCK_M must be one of 1,2,4,8,16")
    # Small decode tiles were measured separately from the large Prefill tile.
    # Keep other small shapes and explicit overrides on their existing path.
    # Both variants write the same buffers; there is no additional workspace.
    small_vector_tile = {25: 8, 40: 16, 80: 16}.get(tokens)
    vector_pack = (
        block_m_env is None
        and hidden == 6144
        and topk == 4
        and (tokens >= 4096 or small_vector_tile is not None)
    )
    if vector_pack:
        block_m = 64 if tokens >= 4096 else small_vector_tile
    block_topk = triton.next_power_of_2(topk)
    _row_gsf_kernel[(tokens,)](
        x,
        out_gsf,
        tokens,
        hidden,
        x.stride(0),
        BLOCK_D=triton.next_power_of_2(hidden),
        num_warps=8,
    )
    grid = (triton.cdiv(tokens, block_m), triton.cdiv(hidden, 64))
    kernel = (
        _pack_nvfp4_inputs_vector_kernel if vector_pack else _pack_nvfp4_inputs_kernel
    )
    kernel[grid](
        x,
        weights,
        indices,
        out_fp4,
        out_sf,
        out_gsf,
        out_indices,
        out_weights,
        tokens,
        hidden,
        topk,
        x.stride(0),
        weights.stride(0),
        indices.stride(0),
        out_fp4.stride(0),
        out_sf.stride(0),
        out_indices.stride(0),
        out_weights.stride(0),
        BLOCK_M=block_m,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )
