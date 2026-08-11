# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted from vLLM's Kimi K3 AttnRes implementation, which contains code
# adapted from flash-linear-attention under the MIT license.

"""Fused Kimi K3 attention-residual kernel."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _multi_block_attn_res_kernel(
    prefix,
    delta,
    blocks,
    norm_weight,
    projection_weight,
    output_norm_weight,
    output,
    stride_prefix_m: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    block_write_idx: tl.constexpr,
    eps: tl.constexpr,
    output_norm_eps: tl.constexpr,
    has_delta: tl.constexpr,
    write_block: tl.constexpr,
    apply_output_norm: tl.constexpr,
    block_l: tl.constexpr,
    block_d: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    hidden_offsets = tl.max_contiguous(tl.arange(0, block_d), block_d)
    hidden_mask = hidden_offsets < hidden_size
    updated_prefix = tl.load(
        prefix + token * stride_prefix_m + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    if has_delta:
        updated_prefix += tl.load(
            delta + token * stride_delta_m + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        updated_prefix = updated_prefix.to(prefix.dtype.element_ty).to(tl.float32)
        tl.store(
            prefix + token * stride_prefix_m + hidden_offsets,
            updated_prefix,
            mask=hidden_mask,
        )
    if write_block:
        tl.store(
            blocks
            + token * stride_block_m
            + block_write_idx * stride_block_r
            + hidden_offsets,
            updated_prefix,
            mask=hidden_mask,
        )

    if num_blocks == 0:
        mixed = updated_prefix
    else:
        if has_delta:
            tl.debug_barrier()
        input_projection_weight = tl.load(
            norm_weight + hidden_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32) * tl.load(
            projection_weight + hidden_offsets, mask=hidden_mask, other=0.0
        ).to(
            tl.float32
        )
        max_logit = tl.full((), -float("inf"), tl.float32)
        denominator = tl.zeros((), tl.float32)
        mixed = tl.zeros((block_d,), tl.float32)
        num_sources: tl.constexpr = num_blocks + 1
        for source_tile in range(tl.cdiv(num_sources, block_l)):
            source_offsets = source_tile * block_l + tl.arange(0, block_l)
            source_mask = source_offsets < num_sources
            is_prefix = source_offsets == num_blocks
            block_ptrs = (
                blocks
                + token * stride_block_m
                + source_offsets[:, None] * stride_block_r
                + hidden_offsets[None, :]
            )
            prefix_ptrs = (
                prefix
                + token * stride_prefix_m
                + source_offsets[:, None] * 0
                + hidden_offsets[None, :]
            )
            value_ptrs = tl.where(is_prefix[:, None], prefix_ptrs, block_ptrs)
            values = tl.load(
                value_ptrs,
                mask=source_mask[:, None] & hidden_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            inverse_rms = tl.rsqrt(
                tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
            )
            logits = (
                tl.sum(values * input_projection_weight[None, :], axis=1)
                * inverse_rms
            )
            scores = tl.where(source_mask, logits, -float("inf"))

            new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
            old_scale = tl.exp(max_logit - new_max_logit)
            source_scales = tl.exp(scores - new_max_logit)
            denominator = denominator * old_scale + tl.sum(source_scales, axis=0)
            mixed = mixed * old_scale + tl.sum(
                source_scales[:, None] * values, axis=0
            )
            max_logit = new_max_logit

        mixed /= denominator
    if apply_output_norm:
        # Preserve both eager BF16 rounding points: the AttnRes output store
        # and RMSNorm's normalized activation before the affine multiply.
        mixed = mixed.to(prefix.dtype.element_ty).to(tl.float32)
        output_inverse_rms = tl.rsqrt(
            tl.sum(tl.where(hidden_mask, mixed * mixed, 0.0), axis=0)
            * (1.0 / hidden_size)
            + output_norm_eps
        )
        normalized = (mixed * output_inverse_rms).to(prefix.dtype.element_ty)
        gamma = tl.load(
            output_norm_weight + hidden_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32)
        mixed = normalized.to(tl.float32) * gamma
    tl.store(
        output + token * stride_output_m + hidden_offsets,
        mixed,
        mask=hidden_mask,
    )


def is_kimi_k3_attn_res_supported(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    block_write_idx: int = -1,
) -> bool:
    if (
        not prefix_sum.is_cuda
        or prefix_sum.ndim != 2
        or prefix_sum.shape[0] == 0
        or prefix_sum.stride(-1) != 1
        or prefix_sum.dtype not in (torch.float16, torch.bfloat16)
    ):
        return False
    if block_residual.ndim != 3:
        return False
    active_blocks = block_residual.shape[1] if num_blocks is None else int(num_blocks)
    if (
        block_residual.shape[0] != prefix_sum.shape[0]
        or active_blocks < 0
        or active_blocks > 8
        or active_blocks > block_residual.shape[1]
        or block_write_idx < -1
        or block_write_idx >= block_residual.shape[1]
        or block_residual.shape[2] != prefix_sum.shape[1]
        or block_residual.stride(-1) != 1
        or block_residual.dtype != prefix_sum.dtype
        or block_residual.device != prefix_sum.device
    ):
        return False
    if delta is not None and (
        delta.shape != prefix_sum.shape
        or delta.stride(-1) != 1
        or delta.dtype != prefix_sum.dtype
        or delta.device != prefix_sum.device
    ):
        return False
    weights = (norm_weight, projection_weight)
    if output_norm_weight is not None:
        weights += (output_norm_weight,)
    return all(
        weight.is_cuda
        and weight.device == prefix_sum.device
        and weight.numel() == prefix_sum.shape[1]
        and weight.reshape(-1).stride(0) == 1
        for weight in weights
    )


@torch.compiler.disable
def kimi_k3_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    eps: float,
    output_norm_weight: Optional[torch.Tensor] = None,
    output_norm_eps: Optional[float] = None,
    delta: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    block_write_idx: int = -1,
) -> torch.Tensor:
    active_blocks = block_residual.shape[1] if num_blocks is None else int(num_blocks)
    if not is_kimi_k3_attn_res_supported(
        prefix_sum,
        block_residual,
        norm_weight,
        projection_weight,
        output_norm_weight,
        delta,
        active_blocks,
        block_write_idx,
    ):
        raise ValueError("unsupported input for K3 fused AttnRes")
    if output_norm_weight is not None and output_norm_eps is None:
        raise ValueError("output_norm_eps is required when output RMSNorm is fused")

    output = torch.empty_like(prefix_sum, memory_format=torch.contiguous_format)
    if prefix_sum.shape[0] >= 256 or active_blocks <= 1:
        block_l, num_warps = 1, 4
    else:
        block_l, num_warps = 4, 8
    _multi_block_attn_res_kernel[(prefix_sum.shape[0],)](
        prefix_sum,
        delta,
        block_residual,
        norm_weight,
        projection_weight.reshape(-1),
        output_norm_weight,
        output,
        prefix_sum.stride(0),
        0 if delta is None else delta.stride(0),
        block_residual.stride(0),
        block_residual.stride(1),
        output.stride(0),
        num_blocks=active_blocks,
        hidden_size=prefix_sum.shape[1],
        block_write_idx=block_write_idx,
        eps=float(eps),
        output_norm_eps=(
            0.0 if output_norm_eps is None else float(output_norm_eps)
        ),
        has_delta=delta is not None,
        write_block=block_write_idx >= 0,
        apply_output_norm=output_norm_weight is not None,
        block_l=block_l,
        block_d=triton.next_power_of_2(prefix_sum.shape[1]),
        num_warps=num_warps,
        num_stages=2,
    )
    return output


__all__ = [
    "is_kimi_k3_attn_res_supported",
    "kimi_k3_attn_res",
]
