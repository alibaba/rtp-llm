"""Triton MegaMoE input packer.

This fuses the hot pre-Mega chain: BF16 activation -> FP8 E4M3, packed UE8M0
group-32 scales, and router tensor copies into DeepGEMM's symmetric-memory
dispatch buffer.  It mirrors ``_per_token_cast_to_fp8_packed_ue8m0`` but writes
directly into the final buffer instead of materializing temporary tensors.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _pack_x_kernel(
        x_ptr,
        out_fp8_ptr,
        out_sf_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        x_stride_m: tl.constexpr,
        out_stride_m: tl.constexpr,
        sf_stride_m: tl.constexpr,
        eps: tl.constexpr,
        fp8_max: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
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

    @triton.jit
    def _pack_router_kernel(
        weights_ptr,
        indices_ptr,
        out_weights_ptr,
        out_indices_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        weights_stride_m: tl.constexpr,
        indices_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
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


def fused_pack_mega_moe_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> None:
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if not x.is_cuda:
        raise RuntimeError("fused MegaMoE input packer requires CUDA tensors")
    if x.dim() != 2:
        raise ValueError(f"x must be [T,D], got {tuple(x.shape)}")
    if weights.shape != indices.shape:
        raise ValueError("weights and indices must have identical [T,topk] shape")
    T, D = x.shape
    if T == 0:
        return
    if D % 128 != 0:
        raise ValueError(f"fused MegaMoE packer requires D % 128 == 0, got D={D}")
    if out_sf.shape[1] != D // 128:
        raise ValueError(
            f"out_sf shape mismatch: expected second dim {D // 128}, got {out_sf.shape}"
        )
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
    topk = weights.shape[1]
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
