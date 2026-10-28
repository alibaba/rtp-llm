"""Fused SM90 MegaMoE input staging.

The Hopper MegaMoE ABI consumes FP8 activations plus ordinary FP32 K128
scales.  The reference path materializes ``nan_to_num`` and quantization
temporaries before copying four tensors into the symmetric buffer.  This
kernel keeps the exact K128 scale formula while writing the final buffer
directly.
"""

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

    @triton.jit(do_not_specialize=["M"])
    def _pack_mega_moe_sm90_inputs_kernel(
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
        amax_floor: tl.constexpr,
        fp8_max: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_block = tl.program_id(0).to(tl.int64)
        k128_block = tl.program_id(1)
        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        cols = k128_block * 128 + tl.arange(0, 128)
        row_mask = rows < M
        value_mask = row_mask[:, None] & (cols[None, :] < N)

        x = tl.load(
            x_ptr + rows[:, None] * x_stride_m + cols[None, :],
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        # Match torch.nan_to_num(..., nan=0, posinf=0, neginf=0).
        x = tl.where(tl.abs(x) < float("inf"), x, 0.0)
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), amax_floor)
        scale = amax / fp8_max
        q = (x * (1.0 / scale[:, None])).to(tl.float8e4nv)

        tl.store(
            out_fp8_ptr + rows[:, None] * out_stride_m + cols[None, :],
            q,
            mask=value_mask,
        )
        tl.store(
            out_sf_ptr + rows * sf_stride_m + k128_block,
            scale,
            mask=row_mask,
        )

        # Router tensors are independent of the K128 tile.  Let the first tile
        # stage them so the complete pack remains one kernel launch.
        if k128_block == 0:
            router_cols = tl.arange(0, BLOCK_K)
            router_mask = row_mask[:, None] & (router_cols[None, :] < K)
            weights = tl.load(
                weights_ptr
                + rows[:, None] * weights_stride_m
                + router_cols[None, :],
                mask=router_mask,
                other=0.0,
            ).to(tl.float32)
            indices = tl.load(
                indices_ptr
                + rows[:, None] * indices_stride_m
                + router_cols[None, :],
                mask=router_mask,
                other=0,
            ).to(tl.int64)
            tl.store(
                out_weights_ptr
                + rows[:, None] * out_weights_stride_m
                + router_cols[None, :],
                weights,
                mask=router_mask,
            )
            tl.store(
                out_indices_ptr
                + rows[:, None] * out_indices_stride_m
                + router_cols[None, :],
                indices,
                mask=router_mask,
            )
else:  # Keep CPU-only import and host launch-contract tests patchable.
    _pack_mega_moe_sm90_inputs_kernel = None


def fused_pack_mega_moe_sm90_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> None:
    """Pack BF16/FP32 activations and router tensors into SM90 buffers."""
    if triton is None:
        raise RuntimeError("Triton is required by the fused SM90 MegaMoE packer")
    tensors = {
        "x": x,
        "weights": weights,
        "indices": indices,
        "out_fp8": out_fp8,
        "out_sf": out_sf,
        "out_indices": out_indices,
        "out_weights": out_weights,
    }
    for name, tensor in tensors.items():
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be a 2D tensor")
    tokens, hidden = x.shape
    if hidden <= 0 or hidden % 128 != 0:
        raise ValueError("SM90 MegaMoE input width must be positive and divisible by 128")
    if weights.shape != indices.shape or weights.shape[0] != tokens:
        raise ValueError("router weights/indices must have matching [tokens, topk] shapes")
    topk = weights.shape[1]
    if topk <= 0:
        raise ValueError("SM90 MegaMoE topk must be positive")
    expected = {
        "out_fp8": (tokens, hidden),
        "out_sf": (tokens, hidden // 128),
        "out_indices": (tokens, topk),
        "out_weights": (tokens, topk),
    }
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensors[name].shape)}"
            )
    expected_dtypes = {
        "out_fp8": torch.float8_e4m3fn,
        "out_sf": torch.float32,
        "out_indices": torch.int64,
        "out_weights": torch.float32,
    }
    for name, dtype in expected_dtypes.items():
        if tensors[name].dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {tensors[name].dtype}")
        if tensors[name].stride(1) != 1:
            raise ValueError(f"{name} must be row-major in its last dimension")
    if any(tensor.device != x.device for tensor in tensors.values()):
        raise ValueError("all SM90 MegaMoE pack tensors must be on the same device")
    if tokens == 0:
        return

    # The kernel only needs row strides. Match the reference packer's legal
    # non-contiguous-input semantics without adding copies for the hot path.
    x = x.contiguous()
    weights = weights.to(torch.float32).contiguous()
    indices = indices.to(torch.int64).contiguous()
    block_m_override = os.environ.get("DSV4_MEGA_MOE_SM90_PACK_BLOCK_M")
    if block_m_override is not None:
        block_m = int(block_m_override)
        if block_m <= 0 or block_m > 32 or block_m & (block_m - 1):
            raise ValueError(
                "DSV4_MEGA_MOE_SM90_PACK_BLOCK_M must be a power of two in [1, 32]"
            )
        num_warps = 4
    elif tokens < 64:
        block_m, num_warps = 4, 4
    elif tokens < 2048:
        block_m, num_warps = 16, 4
    else:
        block_m, num_warps = 32, 4
    block_k = triton.next_power_of_2(topk)
    grid = (triton.cdiv(tokens, block_m), triton.cdiv(hidden, 128))
    _pack_mega_moe_sm90_inputs_kernel[grid](
        x,
        weights,
        indices,
        out_fp8,
        out_sf,
        out_weights,
        out_indices,
        tokens,
        hidden,
        topk,
        x.stride(0),
        weights.stride(0),
        indices.stride(0),
        out_fp8.stride(0),
        out_sf.stride(0),
        out_weights.stride(0),
        out_indices.stride(0),
        1.0e-4,
        torch.finfo(torch.float8_e4m3fn).max,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
