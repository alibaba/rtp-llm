"""Stable, fixed-capacity route alignment for small Qwen4 decode batches."""

import torch
import triton
import triton.language as tl


def sm120_decode_gemm_configs(rows, up, down):
    """Opt-in tiles for BF16 E512/K2560/inter80, with M=1 unchanged."""
    if not 1 < rows <= 32:
        return up, down
    if rows <= 8:
        up = dict(up, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, num_warps=4, num_stages=2)
    down = dict(
        down,
        BLOCK_SIZE_N=64 if rows <= 16 else 128,
        BLOCK_SIZE_K=32,
        num_warps=4,
        num_stages=2,
    )
    return up, down


@triton.jit
def _offsets_and_padding(
    ids,
    offsets,
    sorted_ids,
    experts,
    padded,
    ROUTES,
    EXPERTS: tl.constexpr,
    EXPERT_TILE: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTE_TILE: tl.constexpr,
    SLOT_TILE: tl.constexpr,
    MAX_SLOTS,
    MAX_BLOCKS,
    BLOCK_TILE: tl.constexpr,
):
    r = tl.arange(0, ROUTE_TILE)
    route_ids = tl.load(ids + r, mask=r < ROUTES, other=0)
    counts = tl.histogram(
        route_ids,
        EXPERT_TILE,
        mask=(r < ROUTES) & (route_ids >= 0) & (route_ids < EXPERTS),
    )
    padded_counts = tl.cdiv(counts, BLOCK) * BLOCK
    inclusive = tl.cumsum(padded_counts, 0)
    e = tl.arange(0, EXPERT_TILE)
    tl.store(offsets + e, inclusive - padded_counts, e < EXPERTS)
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


def moe_align_decode(topk_ids, block_size, num_experts, *, max_routes=640):
    """Match the Torch layout, including stable order and unused sentinels.

    Route ids outside [0, num_experts) are ignored. Only bounded
    decode shapes are supported: the stable scatter has quadratic route work.
    No host scalar read or data-dependent allocation occurs during replay.
    """
    if not (
        topk_ids.is_cuda
        and torch.version.hip is None
        and topk_ids.dim() == 2
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and 0 < topk_ids.numel() <= min(max_routes, 2560)
        and 0 < num_experts <= 513
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
        EXPERT_TILE=triton.next_power_of_2(num_experts),
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
        BLOCK=block_size,
        ROUTE_TILE=route_tile,
        EXPERTS=num_experts,
        num_warps=4,
    )
    return sorted_ids, experts, padded


@triton.jit
def _padded_expert_offsets(count, sorted_ids, routes, block: tl.constexpr):
    size = tl.cdiv(count, block) * block
    begin = tl.cumsum(size, 0) - size

    # Real routes are filled by the scatter. Only the partial tiles need
    # sentinel stores. GEMM exits before reading blocks beyond their totals.
    tail = tl.arange(0, block)
    slot = (begin + count)[:, None] + tail[None, :]
    tl.store(sorted_ids + slot, routes, tail[None, :] < (size - count)[:, None])
    return begin, tl.sum(size, 0)


@triton.jit
def _initialize_large_routes(
    counts,
    cursors,
    small_ids,
    dense_ids,
    totals,
    ROUTES,
    EXPERTS: tl.constexpr,
    EXPERT_TILE: tl.constexpr,
    SMALL_BLOCK: tl.constexpr,
    DENSE_BLOCK: tl.constexpr,
    DENSE_THRESHOLD: tl.constexpr,
):
    expert = tl.arange(0, EXPERT_TILE)
    count = tl.load(counts + expert, expert < EXPERTS, 0)
    dense = count >= DENSE_THRESHOLD
    small_begin, small_total = _padded_expert_offsets(
        tl.where(dense, 0, count), small_ids, ROUTES, SMALL_BLOCK
    )
    dense_begin, dense_total = _padded_expert_offsets(
        tl.where(dense, count, 0), dense_ids, ROUTES, DENSE_BLOCK
    )
    tl.store(
        cursors + expert, tl.where(dense, dense_begin, small_begin), expert < EXPERTS
    )
    tl.store(totals, small_total)
    tl.store(totals + 1, dense_total)


@triton.jit
def _scatter_large_routes(
    ids,
    counts,
    cursors,
    small_ids,
    small_experts,
    dense_ids,
    dense_experts,
    ROUTES,
    EXPERTS: tl.constexpr,
    SMALL_BLOCK: tl.constexpr,
    DENSE_BLOCK: tl.constexpr,
    DENSE_THRESHOLD: tl.constexpr,
    ROUTE_TILE: tl.constexpr,
):
    route = tl.program_id(0) * ROUTE_TILE + tl.arange(0, ROUTE_TILE)
    expert = tl.load(ids + route, route < ROUTES, -1)
    valid = (expert >= 0) & (expert < EXPERTS)
    count = tl.load(counts + expert, valid, 0)
    dense = count >= DENSE_THRESHOLD
    slot = tl.atomic_add(cursors + expert, 1, mask=valid, sem="relaxed")

    small_route = valid & ~dense
    tl.store(small_ids + slot, route, small_route)
    tl.store(
        small_experts + slot // SMALL_BLOCK,
        expert,
        small_route & (slot % SMALL_BLOCK == 0),
    )
    dense_route = valid & dense
    tl.store(dense_ids + slot, route, dense_route)
    tl.store(
        dense_experts + slot // DENSE_BLOCK,
        expert,
        dense_route & (slot % DENSE_BLOCK == 0),
    )


def moe_align_large_decode(topk_ids, counts, small_block, dense_block, dense_threshold):
    """Use separate block lists for sparse and dense experts in one graph.

    Counts and IDs describe the same routed input. Within an expert, routes can
    be placed in any order because each GEMM stores back to the original slot.
    """
    routes = topk_ids.numel()
    num_experts = counts.numel()
    options = {"device": topk_ids.device, "dtype": torch.int32}
    # To create an extra block, start a new expert or fill another tile for
    # one already active. Dense experts each require dense_threshold routes.
    active = min(routes, num_experts)
    small_capacity = active + (routes - active) // small_block
    dense_active = min(routes // dense_threshold, num_experts)
    dense_capacity = dense_active + (routes - dense_active) // dense_block
    small_ids = torch.empty(small_capacity * small_block, **options)
    small_experts = torch.empty(small_capacity, **options)
    dense_ids = torch.empty(dense_capacity * dense_block, **options)
    dense_experts = torch.empty(dense_capacity, **options)
    cursors = torch.empty(num_experts, **options)
    totals = torch.empty(2, **options)
    _initialize_large_routes[(1,)](
        counts,
        cursors,
        small_ids,
        dense_ids,
        totals,
        routes,
        num_experts,
        triton.next_power_of_2(num_experts),
        small_block,
        dense_block,
        dense_threshold,
    )
    route_tile = 128
    _scatter_large_routes[(triton.cdiv(routes, route_tile),)](
        topk_ids,
        counts,
        cursors,
        small_ids,
        small_experts,
        dense_ids,
        dense_experts,
        routes,
        num_experts,
        small_block,
        dense_block,
        dense_threshold,
        route_tile,
    )
    return (small_ids, small_experts, totals[:1]), (
        dense_ids,
        dense_experts,
        totals[1:],
    )
