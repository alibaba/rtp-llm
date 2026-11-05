"""Logical attention view of block-round-robin CP cache storage."""

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.ops.compute_ops import LayerKVCache


class CPPagedCacheGeometry:
    """Materialize valid prefixes globally; commit only owner-local new tokens.

    CP query zigzag ownership and cache block-round-robin ownership are distinct.
    The temporary cache uses the ordinary kernel-page ABI consumed by all three
    CP attention algorithms. Its capacity is the current batch's actual N.
    """

    def __init__(self, configs, inputs, parallelism):
        self.base = int(configs.tokens_per_block)
        self.kernel = int(configs.kernel_tokens_per_block)
        self.size = int(parallelism.tp_size)
        self.rank = int(parallelism.tp_rank)
        if self.base % self.kernel:
            raise ValueError("CP cache base block must contain whole kernel pages")
        self.pages_per_block = self.base // self.kernel
        self.prefixes = inputs.prefix_lengths.tolist()
        self.lengths = [
            p + n
            for p, n in zip(
                self.prefixes,
                inputs.context_parallel_info.prefill_actual_input_lengths_cpu.tolist(),
            )
        ]
        table = inputs.kv_cache_kernel_block_id.cpu()
        self.total_pages = sum(
            (n + self.kernel - 1) // self.kernel for n in self.lengths
        )
        max_pages = max((n + self.kernel - 1) // self.kernel for n in self.lengths)
        self.block_ids = torch.full(
            (len(self.lengths), max_pages), -1, dtype=torch.int32
        )
        gather_ids, gather_valid, prefix_dst, prefix_src = [], [], [], []
        write_src, write_dst, write_offset = [], [], []
        stage_start = gather_start = 0
        for batch, (prefix, length) in enumerate(zip(self.prefixes, self.lengths)):
            page_count = (length + self.kernel - 1) // self.kernel
            self.block_ids[batch, :page_count] = torch.arange(
                stage_start, stage_start + page_count
            )
            # Fixed per-request payload shape across ranks, including a rank
            # with no prefix ownership. Padding is zeroed before all-gather.
            slots = (prefix + self.base * self.size - 1) // (self.base * self.size)
            for local_page in range(slots * self.pages_per_block):
                global_block = (
                    local_page // self.pages_per_block
                ) * self.size + self.rank
                global_start = (
                    global_block * self.base
                    + (local_page % self.pages_per_block) * self.kernel
                )
                valid = max(0, min(self.kernel, prefix - global_start))
                physical = int(table[batch, local_page]) if valid else 0
                if physical < 0:
                    raise ValueError("missing owner-local CP prefix page")
                gather_ids.append(physical)
                gather_valid.append(valid)
            for page in range((prefix + self.kernel - 1) // self.kernel):
                block, subpage = divmod(page, self.pages_per_block)
                owner = block % self.size
                local_page = (block // self.size) * self.pages_per_block + subpage
                prefix_dst.append(stage_start + page)
                prefix_src.append((owner, gather_start + local_page))
            for pos in range(prefix, length):
                block, in_block = divmod(pos, self.base)
                if block % self.size != self.rank:
                    continue
                local_page = (
                    block // self.size
                ) * self.pages_per_block + in_block // self.kernel
                physical = int(table[batch, local_page])
                if physical < 0:
                    raise ValueError("missing owner-local CP suffix page")
                write_src.append(stage_start + pos // self.kernel)
                write_dst.append(physical)
                write_offset.append(pos % self.kernel)
            stage_start += page_count
            gather_start += slots * self.pages_per_block
        self.gather_ids = torch.tensor(gather_ids, dtype=torch.long)
        self.gather_valid = torch.tensor(gather_valid, dtype=torch.long)
        self.prefix_dst = torch.tensor(prefix_dst, dtype=torch.long)
        self.prefix_src = torch.tensor(
            [owner * gather_start + offset for owner, offset in prefix_src],
            dtype=torch.long,
        )
        self.write_src = torch.tensor(write_src, dtype=torch.long)
        self.write_dst = torch.tensor(write_dst, dtype=torch.long)
        self.write_offset = torch.tensor(write_offset, dtype=torch.long)

    def materialize(self, cache: LayerKVCache) -> LayerKVCache:
        physical = cache.kv_cache_base
        if physical.ndim != 5 or physical.shape[3] != self.kernel:
            raise ValueError("CP FULL cache must use kernel-major HND pages")
        if cache.kv_scale_base is not None and cache.kv_scale_base.numel():
            raise ValueError("CP FULL cache staging requires unquantized KV")
        device = physical.device
        staged = torch.zeros(
            (self.total_pages, *physical.shape[1:]), dtype=physical.dtype, device=device
        )
        if self.gather_ids.numel():
            local = physical.index_select(0, self.gather_ids.to(device))
            valid = (
                torch.arange(self.kernel, device=device)[None, :]
                < self.gather_valid.to(device)[:, None]
            )
            local.masked_fill_(~valid[:, None, None, :, None], 0)
            gathered = all_gather(local.contiguous(), group=Group.TP)
            staged.index_copy_(
                0,
                self.prefix_dst.to(device),
                gathered.index_select(0, self.prefix_src.to(device)),
            )
        return LayerKVCache(staged, self.kernel, cache.layer_id, cache.tag)

    def commit_suffix(self, staged: LayerKVCache, physical: LayerKVCache) -> None:
        if not self.write_src.numel():
            return
        device = physical.kv_cache_base.device
        src = self.write_src.to(device)
        dst = self.write_dst.to(device)
        offset = self.write_offset.to(device)
        physical.kv_cache_base[dst, :, :, offset, :] = staged.kv_cache_base[
            src, :, :, offset, :
        ]


def save_full_checkpoints(cache, inputs, configs, parallelism):
    """Save completed FULL endpoints after this layer's runtime KV writes.

    Coordinator bindings are physical base-block IDs. Cache-owned references
    and empty sparse slots have negative IDs and are never writable targets.
    The runtime still contains every FULL KV token, so a checkpoint copies only
    its owner's base block up to P, excluding any tokens after P.
    """
    positions = inputs.cache_save_positions
    blocks = inputs.cache_save_block_ids
    if positions is None or blocks is None or not positions.numel():
        return
    base = int(configs.tokens_per_block)
    kernel = int(configs.kernel_tokens_per_block)
    pages = base // kernel
    cp_info = inputs.context_parallel_info
    sharded = parallelism.prefill_cp_config.kv_cache_sharded and cp_info is not None
    size = int(parallelism.tp_size) if sharded else 1
    rank = int(parallelism.tp_rank) if sharded else 0
    table = inputs.kv_cache_kernel_block_id.cpu()
    prefixes = inputs.prefix_lengths.tolist()
    suffixes = (
        cp_info.prefill_actual_input_lengths_cpu
        if cp_info is not None
        else inputs.input_lengths
    ).tolist()
    physical = cache.kv_cache_base
    for batch, (prefix, suffix) in enumerate(zip(prefixes, suffixes)):
        for endpoint, destination in zip(
            positions[batch].tolist(), blocks[batch].tolist()
        ):
            if destination <= 0 or endpoint <= prefix or endpoint > prefix + suffix:
                continue
            local_slot = ((endpoint - 1) // base) // size
            global_start = (local_slot * size + rank) * base
            valid = max(0, min(base, endpoint - global_start))
            for offset in range((valid + kernel - 1) // kernel):
                source = int(table[batch, local_slot * pages + offset])
                if source < 0:
                    raise ValueError(
                        "missing runtime FULL page for selected checkpoint"
                    )
                target = destination * pages + offset
                count = min(kernel, valid - offset * kernel)
                if source != target:
                    physical[target, :, :, :count].copy_(physical[source, :, :, :count])
