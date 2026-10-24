"""Frozen full-WF reference for the pre-optimization routing algorithm."""

import torch
import triton
import triton.language as tl


@triton.jit
def _wf_count_kernel(
    ids_ptr,
    counts_ptr,
    experts,
    numel,
    row_stride,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    addresses = offsets // TOPK * row_stride + offsets % TOPK
    expert = tl.load(ids_ptr + addresses, mask=offsets < numel, other=-1).to(tl.int32)
    valid = (offsets < numel) & (expert >= 0) & (expert < experts)
    tl.atomic_add(counts_ptr + tl.where(valid, expert, 0), 1.0, mask=valid)


@triton.jit
def _wf_hot_order_kernel(
    counts_ptr,
    order_ptr,
    counter_ptr,
    experts,
    BLOCK_EXPERTS: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_EXPERTS)
    valid = offsets < experts
    counts = tl.load(counts_ptr + offsets, mask=valid, other=0).to(tl.uint64)
    tie_break = tl.where(valid, experts - offsets, 0).to(tl.uint64)
    ordered = tl.sort((counts << 32) | tie_break, descending=True)
    logical = (experts - (ordered & 0xFFFFFFFF)).to(tl.int32)
    tl.store(order_ptr + offsets, logical, mask=valid)
    tl.store(counter_ptr + offsets, 0, mask=valid)


@triton.jit
def _fill2(demand, replicas, load0, load1):
    gap = tl.abs(load0 - load1)
    level = (load0 + load1 + demand) * 0.5
    two0 = tl.where(load0 <= load1, tl.minimum(demand, gap), 0.0)
    two1 = tl.where(load0 <= load1, 0.0, tl.minimum(demand, gap))
    split0 = level - load0
    split1 = level - load1
    use_split = demand > gap
    result0 = tl.where(use_split, split0, two0)
    result1 = tl.where(use_split, split1, two1)
    return tl.where(replicas == 2, result0, demand), tl.where(
        replicas == 2, result1, 0.0
    )


@triton.jit
def _round2(demand, replicas, allocation0, allocation1):
    count = (demand + 0.5).to(tl.int32)
    floor0 = tl.floor(allocation0)
    floor1 = tl.floor(allocation1)
    integer0 = floor0.to(tl.int32)
    integer1 = floor1.to(tl.int32)
    remainder = count - integer0 - integer1
    fraction0 = allocation0 - floor0
    fraction1 = allocation1 - floor1
    give0 = (remainder == 1) & (fraction0 >= fraction1)
    give1 = (remainder == 1) & (fraction1 > fraction0)
    quota0 = integer0 + give0.to(tl.int32)
    quota1 = integer1 + give1.to(tl.int32)
    return tl.where(replicas == 1, count, quota0), tl.where(replicas == 2, quota1, 0)


@triton.jit
def _wf_quota_kernel(
    counts_ptr,
    order_ptr,
    gpu_ptr,
    replicas_ptr,
    quota_ptr,
    alloc_ptr,
    experts,
    MAX_PASSES: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    ranks = tl.arange(0, BLOCK_G)
    recv = tl.zeros([BLOCK_G], dtype=tl.float32)
    for position in range(experts):
        expert = tl.load(order_ptr + position)
        demand = tl.load(counts_ptr + expert)
        replicas = tl.load(replicas_ptr + expert)
        gpu0 = tl.load(gpu_ptr + expert * 2)
        gpu1 = tl.load(gpu_ptr + expert * 2 + 1)
        load0 = tl.sum(tl.where(ranks == gpu0, recv, 0.0))
        load1 = tl.sum(tl.where(ranks == gpu1, recv, 0.0))
        allocation0, allocation1 = _fill2(demand, replicas, load0, load1)
        recv = (
            recv
            + tl.where(ranks == gpu0, allocation0, 0.0)
            + tl.where(ranks == gpu1, allocation1, 0.0)
        )
        quota0, quota1 = _round2(demand, replicas, allocation0, allocation1)
        tl.store(quota_ptr + expert * 2, quota0)
        tl.store(quota_ptr + expert * 2 + 1, quota1)
        tl.store(alloc_ptr + expert * 2, allocation0)
        tl.store(alloc_ptr + expert * 2 + 1, allocation1)
    for iteration in range(MAX_PASSES - 1):
        for position in range(experts):
            expert = tl.load(order_ptr + position)
            replicas = tl.load(replicas_ptr + expert)
            demand = tl.load(counts_ptr + expert)
            if (replicas == 2) & (demand > 0.0):
                gpu0 = tl.load(gpu_ptr + expert * 2)
                gpu1 = tl.load(gpu_ptr + expert * 2 + 1)
                old0 = tl.load(alloc_ptr + expert * 2)
                old1 = tl.load(alloc_ptr + expert * 2 + 1)
                recv = (
                    recv
                    - tl.where(ranks == gpu0, old0, 0.0)
                    - tl.where(ranks == gpu1, old1, 0.0)
                )
                load0 = tl.sum(tl.where(ranks == gpu0, recv, 0.0))
                load1 = tl.sum(tl.where(ranks == gpu1, recv, 0.0))
                allocation0, allocation1 = _fill2(demand, replicas, load0, load1)
                recv = (
                    recv
                    + tl.where(ranks == gpu0, allocation0, 0.0)
                    + tl.where(ranks == gpu1, allocation1, 0.0)
                )
                quota0, quota1 = _round2(demand, replicas, allocation0, allocation1)
                tl.store(quota_ptr + expert * 2, quota0)
                tl.store(quota_ptr + expert * 2 + 1, quota1)
                tl.store(alloc_ptr + expert * 2, allocation0)
                tl.store(alloc_ptr + expert * 2 + 1, allocation1)


@triton.jit
def _wf_scatter_kernel(
    ids_ptr,
    quota_ptr,
    physical_ptr,
    counter_ptr,
    experts,
    numel,
    row_stride,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    addresses = offsets // TOPK * row_stride + offsets % TOPK
    expert = tl.load(ids_ptr + addresses, mask=offsets < numel, other=-1).to(tl.int32)
    valid = (offsets < numel) & (expert >= 0) & (expert < experts)
    safe_expert = tl.where(valid, expert, 0)
    position = tl.atomic_add(counter_ptr + safe_expert, 1, mask=valid)
    quota = tl.load(quota_ptr + safe_expert * 2, mask=valid, other=0)
    physical0 = tl.load(physical_ptr + safe_expert * 2, mask=valid, other=-1)
    physical1 = tl.load(physical_ptr + safe_expert * 2 + 1, mask=valid, other=-1)
    physical = tl.where(position < quota, physical0, physical1)
    tl.store(ids_ptr + addresses, tl.where(valid, physical, -1), mask=offsets < numel)


class LegacyFullWF:
    """Independent original kernels, reusable buffers, and original launch geometry."""

    def __init__(
        self, log2phy, replicas, *, num_logical, num_physical, ep_size, wf_passes=8
    ):
        self.num_logical = num_logical
        self.ep_size = ep_size
        self.passes = wf_passes
        device = log2phy.device
        self.cand_phys = torch.full(
            (num_logical, 2), -1, dtype=torch.int32, device=device
        )
        self.cand_phys[:, : log2phy.size(1)].copy_(log2phy)
        self.cand_gpu = torch.where(
            self.cand_phys >= 0, self.cand_phys // (num_physical // ep_size), -1
        )
        self.rep_count = replicas.clone()
        self.counts = torch.empty(num_logical, dtype=torch.float32, device=device)
        self.hot_order = torch.empty(num_logical, dtype=torch.int32, device=device)
        self.quota = torch.empty((num_logical, 2), dtype=torch.int32, device=device)
        self.alloc = torch.empty((num_logical, 2), dtype=torch.float32, device=device)
        self.rank_counter = torch.empty(num_logical, dtype=torch.int32, device=device)

    def remap_inplace(self, logical_ids):
        numel = logical_ids.numel()
        if numel == 0:
            return logical_ids
        block = min(1024, triton.next_power_of_2(numel))
        grid = (triton.cdiv(numel, block),)
        self.counts.zero_()
        _wf_count_kernel[grid](
            logical_ids,
            self.counts,
            self.num_logical,
            numel,
            logical_ids.stride(0),
            TOPK=logical_ids.size(1),
            BLOCK_SIZE=block,
        )
        _wf_hot_order_kernel[(1,)](
            self.counts,
            self.hot_order,
            self.rank_counter,
            self.num_logical,
            BLOCK_EXPERTS=triton.next_power_of_2(self.num_logical),
        )
        _wf_quota_kernel[(1,)](
            self.counts,
            self.hot_order,
            self.cand_gpu,
            self.rep_count,
            self.quota,
            self.alloc,
            self.num_logical,
            MAX_PASSES=self.passes,
            BLOCK_G=triton.next_power_of_2(self.ep_size),
        )
        _wf_scatter_kernel[grid](
            logical_ids,
            self.quota,
            self.cand_phys,
            self.rank_counter,
            self.num_logical,
            numel,
            logical_ids.stride(0),
            TOPK=logical_ids.size(1),
            BLOCK_SIZE=block,
        )
        return logical_ids
