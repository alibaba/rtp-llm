"""Stable, fixed-capacity route alignment for small Qwen4 decode batches."""

import torch
import triton
import triton.language as tl


@triton.jit
def _offsets_and_padding(
    ids,
    offsets,
    sorted_ids,
    experts,
    padded,
    ROUTES: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTE_TILE: tl.constexpr,
    SLOT_TILE: tl.constexpr,
    MAX_SLOTS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
):
    r = tl.arange(0, ROUTE_TILE)
    route_ids = tl.load(ids + r, mask=r < ROUTES, other=0)
    counts = tl.histogram(route_ids, EXPERTS, mask=r < ROUTES)
    padded_counts = tl.cdiv(counts, BLOCK) * BLOCK
    inclusive = tl.cumsum(padded_counts, 0)
    e = tl.arange(0, EXPERTS)
    tl.store(offsets + e, inclusive - padded_counts)
    total = tl.sum(padded_counts, 0)
    tl.store(offsets + EXPERTS, total)
    tl.store(padded, total)
    slot = tl.arange(0, SLOT_TILE)
    tl.store(sorted_ids + slot, ROUTES, mask=slot < MAX_SLOTS)
    block = tl.arange(0, BLOCK_TILE)
    tl.store(experts + block, EXPERTS, mask=block < MAX_BLOCKS)


@triton.jit
def _stable_scatter(
    ids,
    offsets,
    sorted_ids,
    experts,
    ROUTES: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTE_TILE: tl.constexpr,
    EXPERTS: tl.constexpr,
):
    route = tl.program_id(0)
    expert = tl.load(ids + route)
    if expert >= 0 and expert < EXPERTS:
        r = tl.arange(0, ROUTE_TILE)
        other = tl.load(ids + r, mask=r < route, other=-1)
        ordinal = tl.sum(((r < route) & (other == expert)).to(tl.int32), 0)
        destination = tl.load(offsets + expert) + ordinal
        tl.store(sorted_ids + destination, route)
        if ordinal % BLOCK == 0:
            tl.store(experts + destination // BLOCK, expert)


def moe_align_decode(topk_ids, block_size, num_experts):
    """Match the Torch layout, including stable order and unused sentinels.

    Route ids must be in [0, num_experts), as in the Torch helper. Only bounded
    decode shapes are supported: the stable scatter has quadratic route work.
    No host scalar read or data-dependent allocation occurs during replay.
    """
    if not (
        topk_ids.is_cuda
        and torch.version.hip is None
        and topk_ids.dim() == 2
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and 0 < topk_ids.numel() <= 320
        and num_experts > 0
        and num_experts <= 512
        and num_experts & (num_experts - 1) == 0
        and block_size in (8, 16, 32, 64)
    ):
        raise ValueError("unsupported Qwen4 decode alignment geometry")
    routes = topk_ids.numel()
    active = min(routes, num_experts)
    max_blocks = active + (routes - active) // block_size
    max_slots = max_blocks * block_size
    offsets = torch.empty(num_experts + 1, device=topk_ids.device, dtype=torch.int32)
    sorted_ids = torch.empty(max_slots, device=topk_ids.device, dtype=torch.int32)
    experts = torch.empty(max_blocks, device=topk_ids.device, dtype=torch.int32)
    padded = torch.empty(1, device=topk_ids.device, dtype=torch.int32)
    route_tile = triton.next_power_of_2(routes)
    _offsets_and_padding[(1,)](
        topk_ids,
        offsets,
        sorted_ids,
        experts,
        padded,
        ROUTES=routes,
        EXPERTS=num_experts,
        BLOCK=block_size,
        ROUTE_TILE=route_tile,
        SLOT_TILE=triton.next_power_of_2(max_slots),
        MAX_SLOTS=max_slots,
        MAX_BLOCKS=max_blocks,
        BLOCK_TILE=triton.next_power_of_2(max_blocks),
        num_warps=4,
    )
    _stable_scatter[(routes,)](
        topk_ids,
        offsets,
        sorted_ids,
        experts,
        ROUTES=routes,
        BLOCK=block_size,
        ROUTE_TILE=route_tile,
        EXPERTS=num_experts,
        num_warps=4,
    )
    return sorted_ids, experts, padded
