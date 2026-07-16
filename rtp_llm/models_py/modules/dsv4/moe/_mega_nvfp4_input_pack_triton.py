"""Fused BF16-to-NVFP4 input packing for DeepGEMM NVFP4 Mega MoE."""

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
    def _ceil_ue4m3(value):
        value = tl.minimum(tl.maximum(value, 0.0), 448.0)
        subnormal_code = tl.ceil(value * 512.0).to(tl.int32)
        exponent = tl.floor(tl.log2(value)).to(tl.int32)
        exponent_field = exponent + 7
        mantissa = tl.ceil((value * tl.exp2(-exponent.to(tl.float32)) - 1.0) * 8.0).to(
            tl.int32
        )
        normal_code = (exponent_field << 3) + mantissa
        code = tl.where(
            value <= 7.0 / 512.0,
            subnormal_code,
            tl.where(value < 1.0 / 64.0, 8, normal_code),
        )
        code = tl.minimum(tl.maximum(code, 0), 126)
        out_exp = (code >> 3) & 0x0F
        out_mantissa = code & 0x07
        scale = tl.where(
            out_exp == 0,
            out_mantissa.to(tl.float32) * (1.0 / 512.0),
            (1.0 + out_mantissa.to(tl.float32) / 8.0)
            * tl.exp2((out_exp - 7).to(tl.float32)),
        )
        return code, scale

    @triton.jit
    def _e2m1_code(value):
        abs_value = tl.minimum(tl.abs(value), 6.0)
        code = (abs_value > 0.25).to(tl.int32)
        code += (abs_value > 0.75).to(tl.int32)
        code += (abs_value > 1.25).to(tl.int32)
        code += (abs_value > 1.75).to(tl.int32)
        code += (abs_value > 2.5).to(tl.int32)
        code += (abs_value > 3.5).to(tl.int32)
        code += (abs_value > 5.0).to(tl.int32)
        sign = (value < 0.0) & (code != 0)
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
        # Match PyTorch's constant-division lowering in the DeepGEMM reference;
        # forcing div.rn here differs by one ULP for some BF16 row maxima.
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
            group_amax = tl.maximum(tl.max(tl.abs(group_values), axis=1), 1.0e-4)
            sf_code, sf_value = _ceil_ue4m3(tl.div_rn(group_amax, 6.0 * gsf))
            packed_sf = packed_sf | (sf_code << (scale_group * 8))
            denominator = sf_value * gsf

            for pair in tl.static_range(8):
                col0 = dim_block * 64 + scale_group * 16 + pair * 2
                col1 = col0 + 1
                value0 = tl.load(
                    x_ptr + rows * x_stride_m + col0,
                    mask=row_mask & (col0 < D),
                    other=0.0,
                ).to(tl.float32)
                value1 = tl.load(
                    x_ptr + rows * x_stride_m + col1,
                    mask=row_mask & (col1 < D),
                    other=0.0,
                ).to(tl.float32)
                code0 = _e2m1_code(tl.div_rn(value0, denominator))
                code1 = _e2m1_code(tl.div_rn(value1, denominator))
                packed_fp4 = code0 | (code1 << 4)
                out_col = dim_block * 32 + scale_group * 8 + pair
                tl.store(
                    out_fp4_ptr + rows * out_fp4_stride_m + out_col,
                    packed_fp4,
                    mask=row_mask,
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
    block_m = int(os.environ.get("DSV4_MEGA_MOE_NVFP4_PACK_BLOCK_M", "4"))
    if block_m not in (1, 2, 4, 8):
        raise ValueError("DSV4_MEGA_MOE_NVFP4_PACK_BLOCK_M must be one of 1,2,4,8")
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
    _pack_nvfp4_inputs_kernel[grid](
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
