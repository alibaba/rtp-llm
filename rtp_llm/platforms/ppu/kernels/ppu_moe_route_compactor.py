"""Sort-free expert-major metadata compaction for PPU MXFP4 nopad MoE.

The kernels replace the global stable sort and its surrounding elementwise
chain. They only build the route mapping; packed activations and mn-major
scales then use wide parallel gather kernels instead of serializing all top-k
row copies inside one Triton program.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.platforms.ppu.kernels.moe_scale_gather import gather_scale_mn_major

_ROUTE_BLOCK = 256


@triton.jit
def _count_routes_kernel(
    topk_ids_ptr,
    expert_counts_ptr,
    route_count,
    num_experts: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    routes = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    in_bounds = routes < route_count
    expert = tl.load(topk_ids_ptr + routes, mask=in_bounds, other=-1).to(tl.int32)
    valid = in_bounds & (expert >= 0) & (expert < num_experts)
    safe_expert = tl.where(valid, expert, 0)
    tl.atomic_add(expert_counts_ptr + safe_expert, 1, mask=valid)


@triton.jit
def _prefix_routes_kernel(
    expert_counts_ptr,
    route_cursors_ptr,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    counts = tl.load(
        expert_counts_ptr + experts,
        mask=experts < num_experts,
        other=0,
    )
    starts = tl.cumsum(counts) - counts
    tl.store(route_cursors_ptr + experts, starts, mask=experts < num_experts)


@triton.jit
def _assign_routes_kernel(
    topk_ids_ptr,
    route_cursors_ptr,
    source_rows_ptr,
    expert_ids_ptr,
    output_index_ptr,
    route_count,
    TOPK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    routes = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    in_bounds = routes < route_count
    expert = tl.load(topk_ids_ptr + routes, mask=in_bounds, other=-1).to(tl.int32)
    valid = in_bounds & (expert >= 0) & (expert < NUM_EXPERTS)
    safe_expert = tl.where(valid, expert, 0)
    destination_i32 = tl.atomic_add(
        route_cursors_ptr + safe_expert,
        1,
        mask=valid,
    )
    destination = destination_i32.to(tl.int64)
    source_row = routes // TOPK

    tl.store(
        output_index_ptr + routes,
        tl.where(valid, destination, -1),
        mask=in_bounds,
    )
    tl.store(source_rows_ptr + destination, source_row, mask=valid)
    tl.store(expert_ids_ptr + destination, expert, mask=valid)


@torch.no_grad()
def compact_mxfp4_routes_nopad_triton(
    packed: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return the nopad grouped-GEMM layout without a global route sort."""

    token_count = int(packed.shape[0])
    topk = int(topk_ids.shape[1])
    route_count = token_count * topk
    expert_counts = torch.zeros(
        num_experts, dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.zeros(route_count, dtype=torch.int32, device=topk_ids.device)
    # Invalid routes occupy an ignored tail. Row zero keeps the wide gathers
    # in bounds without a device-to-host read of the valid route count.
    source_rows = torch.zeros(route_count, dtype=torch.int64, device=topk_ids.device)
    output_index = torch.empty_like(topk_ids, dtype=torch.int64)

    if route_count:
        route_cursors = torch.empty_like(expert_counts)
        route_grid = (triton.cdiv(route_count, _ROUTE_BLOCK),)
        _count_routes_kernel[route_grid](
            topk_ids,
            expert_counts,
            route_count,
            num_experts=num_experts,
            BLOCK_R=_ROUTE_BLOCK,
            num_warps=4,
        )
        _prefix_routes_kernel[(1,)](
            expert_counts,
            route_cursors,
            num_experts=num_experts,
            BLOCK_E=triton.next_power_of_2(num_experts),
            num_warps=4,
        )
        _assign_routes_kernel[route_grid](
            topk_ids,
            route_cursors,
            source_rows,
            expert_ids,
            output_index,
            route_count,
            TOPK=topk,
            NUM_EXPERTS=num_experts,
            BLOCK_R=_ROUTE_BLOCK,
            num_warps=4,
        )

    packed_out = torch.index_select(packed, 0, source_rows).contiguous()
    scale_out = gather_scale_mn_major(scale, source_rows)
    return packed_out, scale_out, expert_ids, output_index, expert_counts


__all__ = ["compact_mxfp4_routes_nopad_triton"]
