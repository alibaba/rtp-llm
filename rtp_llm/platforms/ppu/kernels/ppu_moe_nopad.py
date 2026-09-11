"""Compact MXFP4 routing metadata for the PPU DeepGEMM nopad ABI.

The layout follows SGLang's Apache-2.0 ``deepgemm_moe_permute`` contract at
commit ``f3cdeef``: valid routes are grouped by expert, ``m_indices`` and
per-expert row counts stay on device, inverse permutation retains the original
top-k shape, and MXFP4 scales keep the DeepGEMM stride ``(1, M_sum)``.

This is an RTP-LLM shim rather than a verbatim source copy. The fixed SGLang
reference and its Apache License 2.0 text are preserved in the task evidence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from rtp_llm.platforms.ppu.kernels.moe_scale_gather import (
    gather_scale_mn_major,
    scale_gather_fused_enabled,
)


def compact_mxfp4_routes_nopad(
    packed: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    *,
    fused_scale_gather: Optional[bool] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Group fixed-size 2-D MXFP4 rows by expert without host synchronization.

    Returns ``packed_out``, ``scale_out``, ``expert_ids``, and
    ``output_index``, and authoritative per-expert row counts. The physical row
    count remains ``topk_ids.numel()`` so graph allocations are fixed. Invalid
    ``-1`` routes sort to the tail, keep ``output_index == -1``, and are
    excluded from the device-generated counts; the nopad GEMM therefore does
    not consume those unspecified tail rows.
    """

    if packed.ndim != 2 or packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise TypeError("packed activation must be contiguous uint8 [M, K/2]")
    if scale.ndim != 2 or scale.dtype != torch.uint16:
        raise TypeError("activation scale must be uint16 [M, K/64]")
    expected_scale_stride = (1, packed.shape[0])
    if scale.shape[0] != packed.shape[0] or scale.stride() != expected_scale_stride:
        raise ValueError("activation scale must use DeepGEMM mn-major stride (1, M)")
    if topk_ids.ndim != 2 or not topk_ids.dtype.is_signed:
        raise TypeError("topk_ids must be a signed rank-2 tensor")
    if topk_ids.shape[0] != packed.shape[0]:
        raise ValueError("packed activation rows must match topk token rows")
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    tensors = (packed, scale, topk_ids)
    if any(tensor.device != packed.device for tensor in tensors):
        raise ValueError(
            "packed activation, scale, route ids, and counts must share a device"
        )

    flat_ids = topk_ids.reshape(-1)
    route_count = flat_ids.numel()
    valid = (flat_ids >= 0) & (flat_ids < num_experts)
    safe_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    expert_counts.scatter_add_(0, safe_ids.to(torch.int64), valid.to(torch.int32))
    sort_key = torch.where(valid, flat_ids, torch.full_like(flat_ids, num_experts))
    permutation = torch.argsort(sort_key, stable=True)
    source_rows = torch.div(permutation, topk_ids.shape[1], rounding_mode="floor").to(
        torch.int64
    )

    packed_out = torch.index_select(packed, 0, source_rows).contiguous()
    use_fused_scale_gather = (
        scale_gather_fused_enabled()
        if fused_scale_gather is None
        else fused_scale_gather
    )
    if use_fused_scale_gather:
        # index_select builds a row-major temporary that copy_ then transposes
        # into the mn-major destination; one kernel does both, bit for bit.
        scale_out = gather_scale_mn_major(scale, source_rows)
    else:
        scale_storage = torch.empty(
            (scale.shape[1], route_count), dtype=torch.uint16, device=scale.device
        )
        scale_out = scale_storage.t()
        scale_out.copy_(torch.index_select(scale, 0, source_rows))

    permuted_ids = torch.index_select(flat_ids, 0, permutation)
    permuted_valid = torch.index_select(valid, 0, permutation)
    expert_ids = torch.where(
        permuted_valid, permuted_ids, torch.zeros_like(permuted_ids)
    ).to(torch.int32)

    compact_positions = torch.arange(
        route_count, dtype=torch.int64, device=topk_ids.device
    )
    output_index_flat = torch.empty(
        route_count, dtype=torch.int64, device=topk_ids.device
    )
    output_index_flat.scatter_(0, permutation.to(torch.int64), compact_positions)
    output_index_flat = torch.where(
        valid, output_index_flat, torch.full_like(output_index_flat, -1)
    )
    output_index = output_index_flat.view_as(topk_ids)
    return packed_out, scale_out, expert_ids, output_index, expert_counts


__all__ = ["compact_mxfp4_routes_nopad"]
