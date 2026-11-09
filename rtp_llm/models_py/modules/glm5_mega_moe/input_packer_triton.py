"""Triton MegaMoE input packer for GLM-5.

Fuses: BF16 activation -> FP8 E4M3, packed UE8M0 group-32 scales, and
router tensor copies into DeepGEMM's symmetric-memory dispatch buffer.
Ported from dsv4/moe/_mega_input_pack_triton.py.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from rtp_llm.models_py.triton_kernels.common.host_math import ceil_div, next_power_of_2

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


_COMPILED_DISPATCH = os.environ.get("GLM53_MOE_PACK_COMPILED_DISPATCH", "0") == "1"
_VECTOR_PACK = os.environ.get("GLM53_MOE_PACK_VECTOR", "0") == "1"


if triton is not None:

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
    ):
        # Large prefill rows can make row * hidden_stride approach int32 range.
        # Use int64 row offsets for all pointer arithmetic.
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

    @triton.jit
    def _or_scale_bits(a, b):
        return a | b

    @triton.jit(do_not_specialize=["M"])
    def _pack_mega_moe_inputs_vector_kernel(
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
    ):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(
            tl.int64
        )
        tile = tl.program_id(1).to(tl.int64)
        group = tl.arange(0, 4)
        cols = tile * 128 + group[:, None] * 32 + tl.arange(0, 32)[None, :]
        valid = (rows[:, None, None] < M) & (cols[None, :, :] < N)
        x = tl.load(
            x_ptr + rows[:, None, None] * x_stride_m + cols[None, :, :], valid, 0
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis=2), eps)
        raw = (amax / fp8_max).to(tl.int32, bitcast=True)
        exp = tl.minimum(tl.maximum((raw >> 23 & 255) + (raw & 8388607 != 0), 1), 254)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        quant = tl.clamp(x / scale[:, :, None], -fp8_max, fp8_max).to(tl.float8e4nv)
        tl.store(
            out_fp8_ptr + rows[:, None, None] * out_stride_m + cols[None, :, :],
            quant,
            valid,
        )
        packed = tl.reduce(exp << group[None, :] * 8, axis=1, combine_fn=_or_scale_bits)
        tl.store(out_sf_ptr + rows * sf_stride_m + tile, packed, rows < M)
        if tile == 0:
            router = tl.arange(0, BLOCK_K)
            mask = (rows[:, None] < M) & (router[None, :] < K)
            weights = tl.load(
                weights_ptr + rows[:, None] * weights_stride_m + router[None, :],
                mask,
                0,
            ).to(tl.float32)
            indices = tl.load(
                indices_ptr + rows[:, None] * indices_stride_m + router[None, :],
                mask,
                0,
            ).to(tl.int64)
            tl.store(
                out_weights_ptr
                + rows[:, None] * out_weights_stride_m
                + router[None, :],
                weights,
                mask,
            )
            tl.store(
                out_indices_ptr
                + rows[:, None] * out_indices_stride_m
                + router[None, :],
                indices,
                mask,
            )


@lru_cache(maxsize=128)
def _compiled_runner(device, dtypes, constants, grid, vector):
    # Retain code only. The public runner resolves the caller's current stream.
    # An odd M keeps the same nonspecialized int32 ABI for every row count.
    function = (
        _pack_mega_moe_inputs_vector_kernel
        if vector
        else _pack_mega_moe_inputs_optimized_kernel
    )
    kernel = function.warmup(
        *dtypes, 13, *constants, grid=grid, num_warps=1 if vector else 4
    )
    return kernel[grid]


def fused_pack_mega_moe_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> None:
    """Fused packing of MegaMoE inputs into the symm-mem buffer.

    Performs BF16 → FP8 E4M3 quantization with packed UE8M0 (group=32) scales,
    and copies router weights/indices, all in a single Triton launch.
    """
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if not x.is_cuda:
        raise RuntimeError("fused MegaMoE input packer requires CUDA tensors")
    if x.dim() != 2:
        raise ValueError(f"x must be [T,D], got {tuple(x.shape)}")

    T, D = x.shape
    topk = weights.shape[1]

    if D % 128 != 0:
        raise ValueError(f"fused MegaMoE packer requires D % 128 == 0, got D={D}")
    if T == 0:
        return

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    block_k = next_power_of_2(topk)
    block_m_env = os.environ.get("GLM5_MEGA_MOE_PACK_BLOCK_M")
    block_m = (
        int(block_m_env)
        if block_m_env is not None
        else (8 if _VECTOR_PACK or T >= 2048 else 2)
    )
    if block_m not in (1, 2, 4, 8):
        block_m = 2

    grid = (ceil_div(T, block_m), ceil_div(D, 128))
    function = (
        _pack_mega_moe_inputs_vector_kernel
        if _VECTOR_PACK
        else _pack_mega_moe_inputs_optimized_kernel
    )
    if (
        _COMPILED_DISPATCH
        and torch.version.hip is None
        and T < 2**31
        and x.device.index == torch.cuda.current_device()
        and not torch.cuda.is_current_stream_capturing()
    ):
        tensors = (x, weights, indices, out_fp8, out_sf, out_weights, out_indices)
        if all(t.device == x.device and t.data_ptr() % 16 == 0 for t in tensors):
            constants = (
                D,
                topk,
                *(t.stride(0) for t in tensors),
                1.0e-4,
                fp8_max,
                block_m,
                block_k,
            )
            _compiled_runner(
                x.device.index,
                tuple(t.dtype for t in tensors),
                constants,
                (*grid, 1),
                _VECTOR_PACK,
            )(*tensors, T, *constants)
            return
    function[grid](
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
        num_warps=1 if _VECTOR_PACK else 4,
    )
