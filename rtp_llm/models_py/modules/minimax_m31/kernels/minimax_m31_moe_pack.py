"""M3.1 fused BF16-to-NVFP4 MegaMoE input packer."""

import torch
import triton
import triton.language as tl


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


def pack_m31_nvfp4_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    buf,
) -> None:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("M3.1 MegaMoE input must be CUDA BF16 [rows,hidden]")
    rows, hidden = x.shape
    topk = buf.num_topk
    if (
        hidden != 6144
        or topk not in (4, 8)
        or weights.shape != (rows, topk)
        or indices.shape != (rows, topk)
    ):
        raise ValueError("M3.1 MegaMoE requires hidden=6144 and TopK=4 or 8")
    if weights.dtype != torch.float32 or indices.dtype != torch.int64:
        raise ValueError("M3.1 MegaMoE router requires float32 weights/int64 indices")
    if rows > buf.num_max_tokens_per_rank:
        raise ValueError("M3.1 MegaMoE input exceeds buffer capacity")
    if rows == 0:
        return
    if (
        not x.is_contiguous()
        or not weights.is_contiguous()
        or not indices.is_contiguous()
    ):
        raise ValueError("M3.1 MegaMoE input/router tensors must be contiguous")
    out_fp4 = buf.x[:rows]
    out_sf = buf.x_sf[:rows]
    out_gsf = buf.x_gsf[:rows]
    out_indices = buf.topk_idx[:rows]
    out_weights = buf.topk_weights[:rows]
    if (
        out_fp4.shape != (rows, hidden // 2)
        or out_sf.shape != (rows, hidden // 64)
        or out_indices.shape != (rows, topk)
        or out_weights.shape != (rows, topk)
    ):
        raise ValueError("M3.1 MegaMoE NVFP4 buffer has wrong geometry")
    _row_gsf_kernel[(rows,)](
        x,
        out_gsf,
        rows,
        hidden,
        x.stride(0),
        BLOCK_D=triton.next_power_of_2(hidden),
        num_warps=8,
    )
    block_m = 16 if rows >= 1024 else 4
    _pack_nvfp4_inputs_kernel[(triton.cdiv(rows, block_m), hidden // 64)](
        x,
        weights,
        indices,
        out_fp4,
        out_sf,
        out_gsf,
        out_indices,
        out_weights,
        rows,
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
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=4,
    )
