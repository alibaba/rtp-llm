"""Fused GC MoE SE staging for Hopper.

The GLM5 FusedGCMoeSeInputPacker uses the Blackwell packed-scale ABI.
SM90 GC consumes separate FP8 activations with FP32 K128 routed scales
and K32 power-of-two shared scales. Stage both plus routing in one launch.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - triton is always present on the test hosts
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["M"])
    def _pack_sm90_inputs_kernel(
        x_ptr,
        weights_ptr,
        indices_ptr,
        out_x_ptr,
        out_xsf_ptr,
        out_weights_ptr,
        out_indices_ptr,
        out_shared_x_ptr,
        out_shared_sf_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        x_stride_m: tl.constexpr,
        weights_stride_m: tl.constexpr,
        indices_stride_m: tl.constexpr,
        out_x_stride_m: tl.constexpr,
        xsf_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        shared_x_stride_m: tl.constexpr,
        shared_sf_stride_m: tl.constexpr,
        eps: tl.constexpr,
        fp8_max: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < M
        mask = row_mask[:, None] & (offs_n[None, :] < N)
        values = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_n[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        values = tl.where(tl.abs(values) < float("inf"), values, 0.0)

        # Routed activations use ordinary per-K128 FP32 scales.
        v128 = tl.reshape(values, (BLOCK_M, BLOCK_N // 128, 128))
        amax128 = tl.maximum(tl.max(tl.abs(v128), axis=2), eps)
        sf128 = amax128 * (1.0 / fp8_max)
        q_routed = (v128 * (1.0 / sf128)[:, :, None]).to(tl.float8e4nv)
        tl.store(
            out_x_ptr + offs_m[:, None] * out_x_stride_m + offs_n[None, :],
            tl.reshape(q_routed, (BLOCK_M, BLOCK_N)),
            mask=mask,
        )
        sf_cols128 = pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)
        tl.store(
            out_xsf_ptr + offs_m[:, None] * xsf_stride_m + sf_cols128[None, :],
            sf128,
            mask=row_mask[:, None] & (sf_cols128[None, :] < N // 128),
        )

        # Shared activations use a power-of-two scale per K32 group.
        g32 = tl.reshape(values, (BLOCK_M, BLOCK_N // 32, 32))
        amax32 = tl.maximum(tl.max(tl.abs(g32), axis=2), eps)
        raw = amax32 * (1.0 / fp8_max)
        bits = raw.to(tl.int32, bitcast=True)
        exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        sf32 = (exp << 23).to(tl.float32, bitcast=True)
        q_shared = (g32 * (1.0 / sf32)[:, :, None]).to(tl.float8e4nv)
        tl.store(
            out_shared_x_ptr + offs_m[:, None] * shared_x_stride_m + offs_n[None, :],
            tl.reshape(q_shared, (BLOCK_M, BLOCK_N)),
            mask=mask,
        )
        sf_cols32 = pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
        tl.store(
            out_shared_sf_ptr + offs_m[:, None] * shared_sf_stride_m + sf_cols32[None, :],
            sf32,
            mask=row_mask[:, None] & (sf_cols32[None, :] < N // 32),
        )

        if pid_n == 0:
            router_offs = tl.arange(0, BLOCK_K)
            router_mask = row_mask[:, None] & (router_offs[None, :] < K)
            w = tl.load(
                weights_ptr + offs_m[:, None] * weights_stride_m + router_offs[None, :],
                mask=router_mask,
                other=0.0,
            )
            idx = tl.load(
                indices_ptr + offs_m[:, None] * indices_stride_m + router_offs[None, :],
                mask=router_mask,
                other=0,
            )
            tl.store(
                out_weights_ptr + offs_m[:, None] * out_weights_stride_m + router_offs[None, :],
                w,
                mask=router_mask,
            )
            tl.store(
                out_indices_ptr + offs_m[:, None] * out_indices_stride_m + router_offs[None, :],
                idx,
                mask=router_mask,
            )


def fused_pack_sm90_gc_se_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    buffer,
) -> None:
    """Write the live rows of a GC SE symmetric buffer."""
    if triton is None:
        raise RuntimeError("Triton is required by the SM90 GC MoE SE packer")
    tokens, hidden = x.shape
    if tokens == 0:
        return
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.stride(1) != 1:
        raise ValueError("SM90 GC MoE SE requires row-major CUDA BF16 input")
    topk = buffer.num_topk
    if (
        hidden != buffer.hidden
        or tokens > buffer.num_max_tokens_per_rank
        or weights.shape != (tokens, topk)
        or indices.shape != (tokens, topk)
        or weights.device != x.device
        or indices.device != x.device
        or weights.stride(1) != 1
        or indices.stride(1) != 1
    ):
        raise ValueError("SM90 GC MoE SE activation or routing shape mismatch")

    block_m, block_n = 16, 256
    grid = (triton.cdiv(tokens, block_m), triton.cdiv(hidden, block_n))
    _pack_sm90_inputs_kernel[grid](
        x,
        weights,
        indices,
        buffer.x,
        buffer.x_sf,
        buffer.topk_weights,
        buffer.topk_idx,
        buffer.shared_l1_acts,
        buffer.shared_l1_acts_sf,
        tokens,
        hidden,
        topk,
        x.stride(0),
        weights.stride(0),
        indices.stride(0),
        buffer.x.stride(0),
        buffer.x_sf.stride(0),
        buffer.topk_weights.stride(0),
        buffer.topk_idx.stride(0),
        buffer.shared_l1_acts.stride(0),
        buffer.shared_l1_acts_sf.stride(0),
        1.0e-4,
        448.0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=triton.next_power_of_2(topk),
        num_warps=4,
    )
