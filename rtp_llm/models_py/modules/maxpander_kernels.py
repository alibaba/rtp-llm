"""GPU-only direct mapping and two-replica WF routing kernels.

No host/torch fallback; routing remains safe during CUDA graph capture.
"""

import triton
import triton.language as tl

MAX_PARTITIONS = 64
COUNT_BLOCK_SIZE = 1024
REDUCE_PARTITION_TILE = 4


@triton.jit
def _direct_remap_kernel(
    topk_ids_ptr,
    cand_phys_ptr,
    num_logical,
    numel,
    row_stride,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < numel
    addresses = offsets // TOPK * row_stride + offsets % TOPK
    logical_ids = tl.load(topk_ids_ptr + addresses, mask=in_bounds, other=-1)
    valid = in_bounds & (logical_ids >= 0) & (logical_ids < num_logical)
    safe_logical_ids = tl.where(valid, logical_ids, 0).to(tl.int32)
    physical_ids = tl.load(cand_phys_ptr + safe_logical_ids * 2, mask=valid, other=-1)
    tl.store(topk_ids_ptr + addresses, physical_ids, mask=in_bounds)


@triton.jit
def _wf_prepare_worklists(
    counts,
    experts,
    valid,
    hot_order_ptr,
    rank_counter_ptr,
    rep_count_ptr,
    refine_order_ptr,
    work_counts_ptr,
    alloc_ptr,
    num_logical,
):
    tie_break = tl.where(valid, num_logical - experts, 0).to(tl.uint64)
    keys = (counts << 32) | tie_break
    ordered = tl.sort(keys, descending=True)
    logical_ids = (num_logical - (ordered & 0xFFFFFFFF)).to(tl.int32)
    active = valid & ((ordered >> 32) > 0)
    replicas = tl.load(rep_count_ptr + logical_ids, mask=valid, other=0)
    refinable = active & (replicas == 2)
    positions = tl.cumsum(refinable.to(tl.int32)) - 1
    tl.store(hot_order_ptr + experts, logical_ids, mask=valid)
    tl.store(refine_order_ptr + positions, logical_ids, mask=refinable)
    tl.store(work_counts_ptr, tl.sum(active.to(tl.int32)))
    tl.store(work_counts_ptr + 1, tl.sum(refinable.to(tl.int32)))
    tl.store(rank_counter_ptr + experts, 0, mask=valid)
    tl.store(alloc_ptr + experts * 2, 0.0, mask=valid)
    tl.store(alloc_ptr + experts * 2 + 1, 0.0, mask=valid)


@triton.jit
def _wf_count_order_kernel(
    topk_ids_ptr,
    counts_ptr,
    hot_order_ptr,
    rank_counter_ptr,
    rep_count_ptr,
    refine_order_ptr,
    work_counts_ptr,
    alloc_ptr,
    num_logical,
    numel,
    row_stride,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    """Overwrite counts and prepare stable worklists without global count atomics."""
    tl.static_assert(TOPK > 0)
    tl.static_assert(TOPK <= BLOCK_SIZE)
    tl.static_assert(BLOCK_SIZE <= 1024)
    tl.static_assert(BLOCK_EXPERTS <= 512)
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    in_bounds = offsets < numel
    addresses = offsets // TOPK * row_stride + offsets % TOPK
    logical_ids = tl.load(topk_ids_ptr + addresses, mask=in_bounds, other=-1)
    valid_ids = in_bounds & (logical_ids >= 0) & (logical_ids < num_logical)
    safe_logical_ids = tl.where(valid_ids, logical_ids, 0).to(tl.int32)
    histogram = tl.histogram(safe_logical_ids, BLOCK_EXPERTS, mask=valid_ids)

    experts = tl.arange(0, BLOCK_EXPERTS)
    valid = experts < num_logical
    tl.store(counts_ptr + experts, histogram.to(tl.float32), mask=valid)
    counts = tl.where(valid, histogram, 0).to(tl.uint64)
    _wf_prepare_worklists(
        counts,
        experts,
        valid,
        hot_order_ptr,
        rank_counter_ptr,
        rep_count_ptr,
        refine_order_ptr,
        work_counts_ptr,
        alloc_ptr,
        num_logical,
    )


@triton.jit(do_not_specialize=["numel", "row_stride"])
def _wf_partial_count_kernel(
    topk_ids_ptr,
    partial_counts_ptr,
    num_logical,
    numel,
    row_stride,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    tl.static_assert(TOPK > 0)
    tl.static_assert(BLOCK_SIZE > 0)
    tl.static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)
    tl.static_assert(BLOCK_EXPERTS > 0)
    tl.static_assert(BLOCK_EXPERTS <= 2048)
    partition = tl.program_id(0)
    num_partitions = tl.num_programs(0)
    block_count = tl.cdiv(numel, BLOCK_SIZE)
    histogram = tl.zeros([BLOCK_EXPERTS], dtype=tl.int32)
    for block_index in range(partition, block_count, num_partitions):
        offsets = block_index.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        in_bounds = offsets < numel
        addresses = offsets // TOPK * row_stride + offsets % TOPK
        logical_ids = tl.load(topk_ids_ptr + addresses, mask=in_bounds, other=-1)
        valid_ids = in_bounds & (logical_ids >= 0) & (logical_ids < num_logical)
        safe_logical_ids = tl.where(valid_ids, logical_ids, 0).to(tl.int32)
        histogram += tl.histogram(safe_logical_ids, BLOCK_EXPERTS, mask=valid_ids)
    experts = tl.arange(0, BLOCK_EXPERTS)
    tl.store(partial_counts_ptr + partition * BLOCK_EXPERTS + experts, histogram)


@triton.jit
def _wf_reduce_partial_counts(
    partial_counts_ptr,
    num_logical,
    num_partitions,
    BLOCK_EXPERTS: tl.constexpr,
    PARTITION_TILE: tl.constexpr,
):
    tl.static_assert(BLOCK_EXPERTS > 0)
    tl.static_assert(BLOCK_EXPERTS <= 2048)
    tl.static_assert(PARTITION_TILE > 0)
    tl.static_assert(PARTITION_TILE <= 64)
    tl.static_assert((PARTITION_TILE & (PARTITION_TILE - 1)) == 0)
    experts = tl.arange(0, BLOCK_EXPERTS)
    partition_offsets = tl.arange(0, PARTITION_TILE)
    counts = tl.zeros([BLOCK_EXPERTS], dtype=tl.int32)
    for partition_start in range(0, num_partitions, PARTITION_TILE):
        partitions = partition_start + partition_offsets
        partial_counts = tl.load(
            partial_counts_ptr + partitions[:, None] * BLOCK_EXPERTS + experts[None, :],
            mask=(partitions[:, None] < num_partitions)
            & (experts[None, :] < num_logical),
            other=0,
        )
        counts += tl.sum(partial_counts, axis=0)
    return counts


@triton.jit(do_not_specialize=["num_partitions"])
def _wf_reduce_order_kernel(
    partial_counts_ptr,
    counts_ptr,
    hot_order_ptr,
    rank_counter_ptr,
    rep_count_ptr,
    refine_order_ptr,
    work_counts_ptr,
    alloc_ptr,
    num_logical,
    num_partitions,
    BLOCK_EXPERTS: tl.constexpr,
    PARTITION_TILE: tl.constexpr = 4,
):
    counts = _wf_reduce_partial_counts(
        partial_counts_ptr,
        num_logical,
        num_partitions,
        BLOCK_EXPERTS,
        PARTITION_TILE,
    )
    experts = tl.arange(0, BLOCK_EXPERTS)
    valid = experts < num_logical
    tl.store(counts_ptr + experts, counts.to(tl.float32), mask=valid)
    _wf_prepare_worklists(
        counts.to(tl.uint64),
        experts,
        valid,
        hot_order_ptr,
        rank_counter_ptr,
        rep_count_ptr,
        refine_order_ptr,
        work_counts_ptr,
        alloc_ptr,
        num_logical,
    )


@triton.jit
def _fill2(c, k, s0, s1):
    """Closed-form 2-way water-fill: pour ``c`` into replicas at loads s0,s1.

    Returns fractional allocation (a0, a1) in candidate order. For a single
    replica (k==1) all of ``c`` lands on replica 0.
    """
    gap = tl.abs(s0 - s1)
    level = (s0 + s1 + c) * 0.5
    two_a0 = tl.where(s0 <= s1, tl.minimum(c, gap), 0.0)
    two_a1 = tl.where(s0 <= s1, 0.0, tl.minimum(c, gap))
    split0 = level - s0
    split1 = level - s1
    use_split = c > gap
    a0_2 = tl.where(use_split, split0, two_a0)
    a1_2 = tl.where(use_split, split1, two_a1)
    a0 = tl.where(k == 2, a0_2, c)
    a1 = tl.where(k == 2, a1_2, 0.0)
    return a0, a1


@triton.jit
def _round2(c, k, a0, a1):
    """Largest-remainder integer rounding of (a0,a1) summing to int(c)."""
    ci = c.to(tl.int32)
    f0 = tl.floor(a0)
    f1 = tl.floor(a1)
    i0 = f0.to(tl.int32)
    i1 = f1.to(tl.int32)
    rem = ci - i0 - i1
    # rem is 0 or 1; award it to the larger fractional part, ties -> replica 0.
    frac0 = a0 - f0
    frac1 = a1 - f1
    give0 = (rem == 1) & (frac0 >= frac1)
    give1 = (rem == 1) & (frac1 > frac0)
    q0 = i0 + give0.to(tl.int32)
    q1 = i1 + give1.to(tl.int32)
    # k==1: everything on replica 0.
    q0 = tl.where(k == 2, q0, ci)
    q1 = tl.where(k == 2, q1, 0)
    return q0, q1


@triton.jit
def _wf_quota_kernel(
    counts_ptr,
    hot_order_ptr,
    cand_gpu_ptr,
    rep_count_ptr,
    quota_ptr,
    alloc_ptr,
    refine_order_ptr,
    work_counts_ptr,
    num_logical,
    MAX_PASSES: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    ranks = tl.arange(0, BLOCK_G)
    recv = tl.zeros([BLOCK_G], dtype=tl.float32)
    active_count = tl.load(work_counts_ptr)
    refine_count = tl.load(work_counts_ptr + 1)
    for position in range(active_count):
        expert = tl.load(hot_order_ptr + position)
        demand = tl.load(counts_ptr + expert)
        replicas = tl.load(rep_count_ptr + expert)
        gpu0 = tl.load(cand_gpu_ptr + expert * 2)
        gpu1 = tl.load(cand_gpu_ptr + expert * 2 + 1)
        if replicas == 1:
            allocation0, allocation1 = demand, 0.0
        else:
            load0 = tl.sum(tl.where(ranks == gpu0, recv, 0.0))
            load1 = tl.sum(tl.where(ranks == gpu1, recv, 0.0))
            allocation0, allocation1 = _fill2(demand, replicas, load0, load1)
        recv = (
            recv
            + tl.where(ranks == gpu0, allocation0, 0.0)
            + tl.where(ranks == gpu1, allocation1, 0.0)
        )
        tl.store(alloc_ptr + expert * 2, allocation0)
        tl.store(alloc_ptr + expert * 2 + 1, allocation1)

    for iteration in range(MAX_PASSES - 1):
        for position in range(refine_count):
            expert = tl.load(refine_order_ptr + position)
            demand = tl.load(counts_ptr + expert)
            gpu0 = tl.load(cand_gpu_ptr + expert * 2)
            gpu1 = tl.load(cand_gpu_ptr + expert * 2 + 1)
            old0 = tl.load(alloc_ptr + expert * 2)
            old1 = tl.load(alloc_ptr + expert * 2 + 1)
            recv = (
                recv
                - tl.where(ranks == gpu0, old0, 0.0)
                - tl.where(ranks == gpu1, old1, 0.0)
            )
            load0 = tl.sum(tl.where(ranks == gpu0, recv, 0.0))
            load1 = tl.sum(tl.where(ranks == gpu1, recv, 0.0))
            allocation0, allocation1 = _fill2(demand, 2, load0, load1)
            recv = (
                recv
                + tl.where(ranks == gpu0, allocation0, 0.0)
                + tl.where(ranks == gpu1, allocation1, 0.0)
            )
            tl.store(alloc_ptr + expert * 2, allocation0)
            tl.store(alloc_ptr + expert * 2 + 1, allocation1)

    tl.debug_barrier()
    experts = tl.arange(0, BLOCK_EXPERTS)
    valid = experts < num_logical
    demand = tl.load(counts_ptr + experts, mask=valid, other=0)
    replicas = tl.load(rep_count_ptr + experts, mask=valid, other=1)
    allocation0 = tl.load(alloc_ptr + experts * 2, mask=valid, other=0)
    allocation1 = tl.load(alloc_ptr + experts * 2 + 1, mask=valid, other=0)
    quota0, quota1 = _round2(demand, replicas, allocation0, allocation1)
    tl.store(quota_ptr + experts * 2, quota0, mask=valid)
    tl.store(quota_ptr + experts * 2 + 1, quota1, mask=valid)


@triton.jit
def _wf_scatter_kernel(
    topk_ids_ptr,
    quota_ptr,
    cand_phys_ptr,
    rank_counter_ptr,
    out_ids_ptr,
    num_logical,
    numel,
    row_stride,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count assignments only when both replica quotas are positive."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    addresses = offs // TOPK * row_stride + offs % TOPK
    logical_ids = tl.load(topk_ids_ptr + addresses, mask=mask, other=-1)
    valid = mask & (logical_ids >= 0) & (logical_ids < num_logical)
    safe_logical_ids = tl.where(valid, logical_ids, 0).to(tl.int32)
    q0 = tl.load(quota_ptr + safe_logical_ids * 2 + 0, mask=valid, other=0)
    q1 = tl.load(quota_ptr + safe_logical_ids * 2 + 1, mask=valid, other=0)
    split = valid & (q0 > 0) & (q1 > 0)
    rank = tl.atomic_add(rank_counter_ptr + safe_logical_ids, 1, mask=split)
    rank = tl.where(split, rank, 0)
    p0 = tl.load(cand_phys_ptr + safe_logical_ids * 2 + 0, mask=valid, other=-1)
    p1 = tl.load(cand_phys_ptr + safe_logical_ids * 2 + 1, mask=valid, other=-1)
    phys = tl.where(rank < q0, p0, p1)
    phys = tl.where(valid, phys, -1)
    tl.store(out_ids_ptr + addresses, phys, mask=mask)
