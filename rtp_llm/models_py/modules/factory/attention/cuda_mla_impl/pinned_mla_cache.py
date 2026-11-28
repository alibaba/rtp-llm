"""Token working set for one DSA shared-index group.

The caller owns the logical block allocator and the HBM indexer cache. Supply
physical *backing-store* token IDs after applying its block table. All layers
in a group use the same resident-token map, but retain independent KV bytes.
Allocator generations or explicit invalidate handle reused backing slots;
call write for newly computed KV. CPU access to backing storage must wait for
the compute stream. This component does not change the attention KV format.
"""

from contextlib import contextmanager
from copy import copy
from typing import Sequence

import torch
import triton
import triton.language as tl


def make_working_sets(kv_cache, config, previous=None):
    """Share one resident map per tagged MLA group and reuse it across graph clones."""
    if kv_cache is None or not getattr(kv_cache, "layer_count", 0):
        return None
    first = kv_cache.get_layer_cache(0, "default")
    host = getattr(first, "kv_host_base", None)
    if host is None or host.numel() == 0:
        return None

    from rtp_llm.utils.dsa_indexing import build_dsa_indexer_kv_slot_mapping

    generations = kv_cache.get_block_generations("default")
    if generations is None:
        raise RuntimeError("tiered MLA cache requires allocator block generations")
    page_size = kv_cache.get_kernel_seq_size_per_block("default")
    allocator_size = kv_cache.get_seq_size_per_block("default")
    active = kv_cache.layer_count
    layers = [kv_cache.get_layer_cache(i, "default") for i in range(active)]
    if any(
        l.kv_host_base is None or l.logical_device_blocks != first.logical_device_blocks
        for l in layers
    ):
        raise RuntimeError("tiered MLA layers must share a logical page boundary")
    key = (
        generations.data_ptr(),
        tuple((l.kv_cache_base.data_ptr(), l.kv_host_base.data_ptr()) for l in layers),
        first.logical_device_blocks,
        page_size,
        allocator_size,
    )
    if previous is not None and previous[0] == key:
        return previous

    hbm_tokens = first.logical_device_blocks * allocator_size
    resident_tokens = first.kv_cache_base.shape[0] * page_size - hbm_tokens
    mapping = build_dsa_indexer_kv_slot_mapping(config, active)
    by_group = {}
    for i, group in enumerate(mapping):
        by_group.setdefault(group, []).append(i)
    result = {}
    for members in by_group.values():
        working = PinnedMlaWorkingSet(
            [layers[i].kv_host_base for i in members],
            resident_tokens,
            page_size,
            first.kv_cache_base.device,
            block_generations=generations,
            allocator_block_size=allocator_size,
            hbm_tokens=hbm_tokens,
            hbm_cache=[layers[i].kv_cache_base for i in members],
        )
        for local, i in enumerate(members):
            result[i] = (working, local)
    return key, result


class PinnedMlaContextCache:
    """Expose compact CUDA pages to the ordinary eager context backend.

    Decode workers also run real context forwards when constructing system
    prompts. Retain their normal dense/sparse backend, and stage one compact
    layer only when the live page table contains host backing.
    """

    @classmethod
    def from_attention(cls, attention, working):
        first = working.backing[0]
        page_size = first.shape[1]
        device_pages = working.hbm_tokens // page_size
        original = attention.kv_cache_kernel_block_id
        if original is None or original.numel() == 0:
            original = attention.kv_cache_kernel_block_id_device
        if original is None or original.numel() == 0:
            return None

        table = original.cpu()
        prefix = attention.prefix_lengths
        lengths = attention.input_lengths.cpu().to(torch.int64)
        prefix = (
            prefix.cpu()
            if prefix is not None and prefix.numel()
            else torch.zeros_like(lengths)
        )
        ends = prefix + lengths
        columns = torch.arange(table.shape[1])
        live = columns < (ends[:, None] + page_size - 1) // page_size
        if not bool((live & (table >= device_pages)).any()):
            return None
        pages = torch.unique(table[live], sorted=True)
        if int(pages[0]) < 0 or int(pages[-1]) >= device_pages + first.shape[0]:
            raise ValueError("context page table exceeds the logical MLA cache")
        # Keep page zero invalid for paged sparse attention and dense padding.
        compact = torch.searchsorted(pages, table).to(table.dtype) + 1
        compact.masked_fill_(~live, 0)
        dirty = (
            live & (columns >= prefix[:, None] // page_size) & (lengths[:, None] > 0)
        )
        dirty_pages = set(table[dirty].tolist())
        return cls(
            attention, working, compact, pages.tolist(), dirty_pages, device_pages
        )

    def __init__(self, attention, working, table, pages, dirty_pages, device_pages):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.layered_prefill_cache import (
            contiguous_runs,
        )

        self.original_inputs = attention
        self.inputs = copy(attention)
        self.inputs.kv_cache_kernel_block_id = table
        self.inputs.kv_cache_kernel_block_id_device = table.to(working.device)
        staged_pages = list(enumerate(pages, start=1))
        self.device_fetch = contiguous_runs(
            [(page, index) for index, page in staged_pages if page < device_pages]
        )
        self.host_fetch = contiguous_runs(
            [
                (page - device_pages, index)
                for index, page in staged_pages
                if page >= device_pages
            ]
        )
        self.device_write = contiguous_runs(
            [
                (index, page)
                for index, page in staged_pages
                if page < device_pages and page in dirty_pages
            ]
        )
        self.host_write = contiguous_runs(
            [
                (index, page - device_pages)
                for index, page in staged_pages
                if page >= device_pages and page in dirty_pages
            ]
        )
        first = working.backing[0]
        self.staging = torch.empty(
            (len(staged_pages) + 1, *first.shape[1:]),
            dtype=first.dtype,
            device=working.device,
        )
        # Discard working-set entries for rewritten host pages. Each group is
        # invalidated once, before its first staged layer writes back.
        page_size = first.shape[1]
        host_dirty = [
            page for page in pages if page >= device_pages and page in dirty_pages
        ]
        self.dirty_slots = (
            torch.tensor(host_dirty, dtype=torch.int32, device=working.device)[:, None]
            * page_size
            + torch.arange(page_size, dtype=torch.int32, device=working.device)
        ).flatten()
        self.writer = None

    @staticmethod
    def _copy_device(source, target, runs):
        for first, offset, count in runs:
            target.narrow(0, offset, count).copy_(source.narrow(0, first, count))

    @contextmanager
    def attention(self, layer_cache, entry):
        from rtp_llm.models_py.modules.factory.attention import common
        from rtp_llm.ops.compute_ops import LayerKVCache, copy_paged_kv_cache_async

        working, group_layer = entry
        device, host = layer_cache.kv_cache_base, layer_cache.kv_host_base
        if group_layer == 0 and self.host_write:
            working.invalidate(self.dirty_slots)
        self._copy_device(device, self.staging, self.device_fetch)
        copy_paged_kv_cache_async(host, self.staging, self.host_fetch)
        yield LayerKVCache(
            self.staging,
            layer_cache.seq_size_per_block,
            layer_cache.layer_id,
            layer_cache.tag,
        )
        self._copy_device(self.staging, device, self.device_write)
        if self.host_write:
            copy_paged_kv_cache_async(self.staging, host, self.host_write)
        common.apply_write_cache_store(self.writer, self.original_inputs, layer_cache)


@triton.jit
def _protect(
    Ids,
    Slots,
    Owners,
    Map,
    Protected,
    Epoch,
    MissCount,
    Generations,
    Versions,
    BLOCK_SIZE: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    HBM_TOKENS: tl.constexpr = 0,
):
    if tl.program_id(0) == 0:
        tl.store(MissCount, 0)
    i = tl.program_id(0) * B + tl.arange(0, B)
    token = tl.load(Ids + i, i < N, other=-1)
    valid = (i < N) & (token >= HBM_TOKENS)
    slot = tl.load(Map + token, valid, other=-1)
    epoch = tl.load(Epoch)
    tl.atomic_xchg(Protected + slot, epoch, valid & (slot >= 0))
    generation = tl.load(
        Generations + token // BLOCK_SIZE, valid & (slot >= 0), other=-1
    )
    version = tl.load(Versions + slot, valid & (slot >= 0), other=-1)
    refresh = valid & ((slot < 0) | (version != generation))
    # Fresh hits already have their final physical index. Only misses need
    # ownership arbitration and a subsequent remap; -1 remains padding.
    physical = tl.where(token < HBM_TOKENS, token, slot + HBM_TOKENS)
    tl.store(Slots + i, tl.where(refresh, -2, physical), i < N)
    tl.atomic_min(Owners + token, i, refresh)


@triton.jit
def _admit(
    Ids,
    Slots,
    Owners,
    Map,
    Tags,
    Protected,
    Epoch,
    Clock,
    Misses,
    MissCount,
    Generations,
    Versions,
    BLOCK_SIZE: tl.constexpr,
    N: tl.constexpr,
    CAPACITY: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    refresh = tl.load(Slots + i, i < N, other=-1) == -2
    if tl.sum(refresh.to(tl.int32), 0) == 0:
        return
    token = tl.load(Ids + i, refresh, other=-1)
    owner = tl.load(Owners + token, refresh, other=-1)
    slot = tl.load(Map + token, refresh, other=-1)
    owned = refresh & (owner == i)
    generation = tl.load(Generations + token // BLOCK_SIZE, owned, other=-1)
    pending = owned & (slot < 0)
    # Protection already marked exactly the missing or stale selections.
    # Each owner refreshes once; no second version lookup is needed.
    count = tl.sum(owned.to(tl.int32), 0)
    start = tl.atomic_add(MissCount, count, sem="relaxed")
    offset = tl.cumsum(owned.to(tl.int32), 0) - 1
    tl.store(Misses + start + offset, i, owned)
    # Recycled backing blocks keep their GPU slot and refresh all group layers.
    # Only each missed token's owner changes its resident version.
    tl.store(Versions + slot, generation, owned & (slot >= 0))
    epoch = tl.load(Epoch)
    # Every selected slot was protected in a preceding kernel. Since capacity
    # covers all input entries, every unique miss can claim an unprotected slot.
    # Compact this CTA's new tokens once. Scan a full tile even with one miss:
    # one candidate per miss degenerates to a serial walk over protected hits.
    count = tl.sum(pending.to(tl.int32), 0)
    if count == 0:
        return
    tokens = tl.sort(tl.where(pending, token, 0x7FFFFFFF), descending=False)
    admitted = 0
    scan: tl.constexpr = min(B, CAPACITY)
    first_ticket = tl.atomic_add(Clock, count, sem="relaxed")
    while admitted < count:
        remaining = count - admitted
        ticket = first_ticket + tl.arange(0, B)
        candidate = (ticket % CAPACITY).to(tl.int32)
        available = (tl.arange(0, B) < scan) & (tl.load(Protected + candidate) != epoch)
        rank = tl.cumsum(available.to(tl.int32), 0) - 1
        attempt = available & (rank < remaining)
        previous = tl.atomic_xchg(Protected + candidate, epoch, attempt)
        claimed = attempt & (previous != epoch)
        rank = tl.cumsum(claimed.to(tl.int32), 0) - 1
        token = tl.gather(tokens, tl.where(claimed, admitted + rank, 0), 0)
        generation = tl.load(Generations + token // BLOCK_SIZE, claimed, other=-1)
        old = tl.load(Tags + candidate, claimed, other=-1)
        tl.store(Map + old, -1, claimed & (old >= 0))
        tl.store(Tags + candidate, token, claimed)
        tl.store(Versions + candidate, generation, claimed)
        tl.store(Map + token, candidate, claimed)
        claimed_count = tl.sum(claimed.to(tl.int32), 0)
        admitted += claimed_count
        # Reserve each retry tile in one operation: split cursor increments can
        # interleave and repeatedly skip the only victims. Initial reservations
        # stay compact so small admissions use the entire cache.
        if admitted < count:
            first_ticket = tl.atomic_add(Clock, scan, sem="relaxed")


@triton.jit
def _fetch(
    Host,
    Device,
    Ids,
    Slots,
    Map,
    Owners,
    Epoch,
    Misses,
    MissCount,
    WIDTH: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    REMAP: tl.constexpr,
    HBM_TOKENS: tl.constexpr = 0,
):
    count = tl.load(MissCount)
    if REMAP:
        # Admission has finished for every CTA. Fresh hits were remapped by
        # protection; an entirely hot group skips this work altogether.
        if count > 0:
            for start in range(tl.program_id(0) * 256, N, tl.num_programs(0) * 256):
                i = start + tl.arange(0, 256)
                refresh = tl.load(Slots + i, i < N, other=-1) == -2
                token = tl.load(Ids + i, refresh, other=-1)
                slot = tl.load(Map + token, refresh, other=-1)
                tl.store(Slots + i, slot + HBM_TOKENS, refresh)
        # Only the next invocation reads Epoch; all group layers share Map.
        if tl.program_id(0) == 0:
            tl.store(Epoch, tl.load(Epoch) + 1)
    # Fixed persistent grid over compact misses, including when all tokens hit.
    for entry in range(tl.program_id(0), count, tl.num_programs(0)):
        i = tl.load(Misses + entry)
        token = tl.load(Ids + i).to(tl.int64)
        slot = tl.load(Map + token).to(tl.int64)
        if REMAP:
            tl.store(Owners + token, 0x7FFFFFFF)
        x = tl.arange(0, B)
        data = tl.load(Host + (token - HBM_TOKENS) * WIDTH + x, x < WIDTH, other=0)
        tl.store(Device + (slot + HBM_TOKENS) * WIDTH + x, data, x < WIDTH)


@triton.jit
def _write(
    Host,
    Device,
    Map,
    Ids,
    Values,
    WIDTH: tl.constexpr,
    B: tl.constexpr,
    HBM_TOKENS: tl.constexpr = 0,
):
    # Widen before multiplying by WIDTH, including the Values source offset.
    i = tl.program_id(0).to(tl.int64)
    token = tl.load(Ids + i).to(tl.int64)
    if token >= 0:
        x = tl.arange(0, B)
        data = tl.load(Values + i * WIDTH + x, x < WIDTH, other=0)
        if token < HBM_TOKENS:
            tl.store(Device + token * WIDTH + x, data, x < WIDTH)
        else:
            tl.store(Host + (token - HBM_TOKENS) * WIDTH + x, data, x < WIDTH)
            slot = tl.load(Map + token).to(tl.int64)
            if slot >= 0:
                tl.store(Device + (slot + HBM_TOKENS) * WIDTH + x, data, x < WIDTH)


@triton.jit
def _invalidate(
    Ids, Map, Tags, N: tl.constexpr, B: tl.constexpr, HBM_TOKENS: tl.constexpr = 0
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    token = tl.load(Ids + i, i < N, other=-1)
    valid = (i < N) & (token >= HBM_TOKENS)
    slot = tl.load(Map + token, valid, other=-1)
    tl.store(Tags + slot, -1, valid & (slot >= 0))
    tl.store(Map + token, -1, valid)


@triton.jit
def _check_selected_bytes(
    Host,
    Device,
    Ids,
    Slots,
    Errors,
    N: tl.constexpr,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
    B: tl.constexpr,
    HBM_TOKENS: tl.constexpr = 0,
):
    for i in range(tl.program_id(0), N, tl.num_programs(0)):
        token = tl.load(Ids + i).to(tl.int64)
        slot = tl.load(Slots + i).to(tl.int64)
        bad = (token < 0) & (slot != -1)
        if (token >= 0) & (token < HBM_TOKENS):
            bad = slot != token
        if token >= HBM_TOKENS:
            valid = (slot >= HBM_TOKENS) & (slot < CAPACITY + HBM_TOKENS)
            x = tl.arange(0, B)
            expected = tl.load(
                Host + (token - HBM_TOKENS) * WIDTH + x, x < WIDTH, other=0
            )
            actual = tl.load(Device + slot * WIDTH + x, valid & (x < WIDTH), other=0)
            bad = ~valid | (tl.sum((actual != expected).to(tl.int32), 0) > 0)
        if bad:
            tl.atomic_add(Errors, 1, sem="relaxed")


class PinnedMlaWorkingSet:
    """One global, GPU-managed working set per shared-index group.

    ``backing`` contains only overflow blocks, compacted after ``hbm_tokens``.
    ``resident`` holds complete HBM blocks followed by the token working set.
    HBM selections retain their physical IDs; only overflow selections are cached.
    ``resident_tokens`` must cover the maximum flattened topk batch, rounded
    to whole pages. Metadata and resident KV have stable addresses for CUDA
    graph replay. There is no index D2H transfer or host-side admission loop.

    The executor must serialize invocations of this object. begin() protects
    the complete selected set before evicting anything; layer_cache() joins
    that layer's prefetch. Each returned tensor preserves backing dtype and
    trailing dimensions, so attention can consume paged KV and physical IDs.
    """

    def __init__(
        self,
        backing: Sequence[torch.Tensor],
        resident_tokens: int,
        page_size: int,
        device: torch.device,
        block_generations: torch.Tensor = None,
        allocator_block_size: int = None,
        hbm_tokens: int = 0,
        hbm_cache: Sequence[torch.Tensor] = None,
    ):
        if not backing or page_size <= 0 or resident_tokens <= 0:
            raise ValueError("backing, page_size and resident_tokens must be positive")
        if resident_tokens % page_size:
            raise ValueError("resident_tokens must be a multiple of page_size")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("working set requires a CUDA device")
        first = backing[0]
        if first.ndim < 3 or first.shape[1] != page_size:
            raise ValueError("backing must have paged [blocks, page_size, ...] shape")
        if hbm_tokens < 0 or hbm_tokens % page_size:
            raise ValueError("HBM capacity must contain whole pages")
        # Metadata and attention use int32 token IDs; byte offsets stay int64.
        # Validate both address spaces before allocating any CUDA tensors.
        logical_tokens = hbm_tokens + first.shape[0] * page_size
        max_tokens = torch.iinfo(torch.int32).max
        if logical_tokens > max_tokens:
            raise ValueError("logical token capacity must fit int32")
        if hbm_tokens + resident_tokens > max_tokens:
            raise ValueError("physical token capacity must fit int32")
        for tensor in backing:
            if tensor.is_cuda or not tensor.is_pinned() or not tensor.is_contiguous():
                raise ValueError("backing must be contiguous CPU pinned memory")
            if tensor.shape != first.shape or tensor.dtype != first.dtype:
                raise ValueError(
                    "shared-index group layers must have identical KV layout"
                )
        self.backing = tuple(backing)
        self.capacity = resident_tokens
        self.hbm_tokens = hbm_tokens
        self.logical_tokens = logical_tokens
        self.width = first[0, 0].numel() * first.element_size()
        shape = ((hbm_tokens + resident_tokens) // page_size, *first.shape[1:])
        self.resident = (
            tuple(hbm_cache)
            if hbm_cache is not None
            else tuple(
                torch.empty(shape, dtype=first.dtype, device=self.device)
                for _ in backing
            )
        )
        if len(self.resident) != len(backing) or any(
            tensor.shape != shape
            or tensor.dtype != first.dtype
            or tensor.device != self.device
            or not tensor.is_contiguous()
            for tensor in self.resident
        ):
            raise ValueError(
                "HBM tensors must hold complete blocks followed by the working set"
            )
        self.mapping = torch.full(
            (self.logical_tokens,), -1, dtype=torch.int32, device=self.device
        )
        self.tags = torch.full(
            (resident_tokens,), -1, dtype=torch.int32, device=self.device
        )
        self.owners = torch.full_like(self.mapping, 0x7FFFFFFF)
        self.versions = torch.full(
            (resident_tokens,), -1, dtype=torch.int64, device=self.device
        )
        self.allocator_block_size = allocator_block_size or page_size
        if (
            self.allocator_block_size <= 0
            or self.logical_tokens % self.allocator_block_size
        ):
            raise ValueError("backing capacity must contain whole allocator blocks")
        self.generations = block_generations
        if self.generations is None:
            self.generations = torch.zeros(
                (self.logical_tokens // self.allocator_block_size,),
                dtype=torch.int64,
                device=self.device,
            )
        if (
            self.generations.dtype != torch.int64
            or not self.generations.is_contiguous()
            or self.generations.numel()
            != self.logical_tokens // self.allocator_block_size
            or (
                self.generations.device != self.device
                and not self.generations.is_pinned()
            )
        ):
            raise ValueError(
                "block generations must be contiguous int64 on CUDA or pinned CPU"
            )
        # Direct pinned-host generation reads preserve the source default
        # without an additional CUDA allocation or an environment toggle.
        self.protected = torch.zeros(
            (resident_tokens,), dtype=torch.int64, device=self.device
        )
        self.epoch = torch.ones((), dtype=torch.int64, device=self.device)
        self.clock = torch.zeros((), dtype=torch.int64, device=self.device)
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.ready = tuple(torch.cuda.Event() for _ in backing)
        self.started = False

    def _check_ids(self, ids: torch.Tensor) -> None:
        if ids.device != self.mapping.device or ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("token IDs must be int32/int64 on the working set device")
        if not ids.is_contiguous():
            raise ValueError("token IDs must be contiguous")
        # Bounds are the allocator's contract: -1 padding or [0, logical_tokens).
        # Reading min/max back to the CPU here would serialize every decode step.

    def begin(self, logical_indices: torch.Tensor) -> torch.Tensor:
        """Admit a whole group's selected tokens and enqueue layerwise prefetch.

        Call after topk, before any group layer's KV write/attention. The returned
        physical indices retain the input shape, padding and duplicate entries.
        Join layer_cache() before consuming either its KV or these indices.
        """
        self._check_ids(logical_indices)
        n = logical_indices.numel()
        if n > self.capacity:
            raise ValueError("working set cannot hold this batch's flattened topk")
        # The preceding group's last layer joins all transfer work. During
        # capture, a new event on the not-yet-captured transfer stream would
        # illegally introduce a dependency on uncaptured work.
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)
        slots = torch.empty_like(logical_indices, dtype=torch.int32)
        misses = torch.empty_like(logical_indices, dtype=torch.int32)
        miss_count = torch.empty((), dtype=torch.int32, device=self.device)
        self.transfer_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.transfer_stream):
            # Admission is independent of Q projection too. Keep both metadata
            # and KV fetch off the compute stream until this layer consumes KV.
            generations = self.generations
            if n:
                grid = (triton.cdiv(n, 256),)
                _protect[grid](
                    logical_indices,
                    slots,
                    self.owners,
                    self.mapping,
                    self.protected,
                    self.epoch,
                    miss_count,
                    generations,
                    self.versions,
                    self.allocator_block_size,
                    n,
                    256,
                    self.hbm_tokens,
                )
                _admit[grid](
                    logical_indices,
                    slots,
                    self.owners,
                    self.mapping,
                    self.tags,
                    self.protected,
                    self.epoch,
                    self.clock,
                    misses,
                    miss_count,
                    generations,
                    self.versions,
                    self.allocator_block_size,
                    n,
                    self.capacity,
                    256,
                )
            for layer, (host, resident, ready) in enumerate(
                zip(self.backing, self.resident, self.ready)
            ):
                if n:
                    _fetch[(min(n, 128),)](
                        host.view(torch.uint8),
                        resident.view(torch.uint8),
                        logical_indices,
                        slots,
                        self.mapping,
                        self.owners,
                        self.epoch,
                        misses,
                        miss_count,
                        self.width,
                        triton.next_power_of_2(self.width),
                        n,
                        layer == 0,
                        self.hbm_tokens,
                    )
                ready.record(self.transfer_stream)
        for tensor in (logical_indices, slots, misses, miss_count):
            tensor.record_stream(self.transfer_stream)
        self.started = True
        self.logical_indices = logical_indices
        self.physical_indices = slots
        return slots

    def layer_cache(self, layer: int) -> torch.Tensor:
        if not self.started:
            raise RuntimeError("begin must precede layer_cache")
        consumer = torch.cuda.current_stream(self.device)
        consumer.wait_event(self.ready[layer])
        # CMP produces these indices on its Indexer stream. Keep their
        # allocation alive until the attention consumer is done with it.
        self.physical_indices.record_stream(consumer)
        return self.resident[layer]

    def wait_prefetch_complete(self) -> None:
        """Join this group's prefetch on the caller stream before running MoE.

        begin() has already enqueued every layer's fetch. Waiting for the last
        ready event preserves overlap with attention but keeps remaining host
        reads out of the following MoE. Use the caller stream: CMP and graph
        capture can run MoE on a different stream from the TopK producer.
        """
        if not self.started:
            raise RuntimeError("begin must precede wait_prefetch_complete")
        torch.cuda.current_stream(self.device).wait_event(self.ready[-1])

    def write(
        self, layer: int, logical_slots: torch.Tensor, values: torch.Tensor
    ) -> None:
        """Write through original KV bytes, refreshing any prefetched current token."""
        self._check_ids(logical_slots)
        if values.device != self.mapping.device or not values.is_contiguous():
            raise ValueError("values must be contiguous on the working set device")
        if values.dtype != self.backing[layer].dtype:
            raise ValueError("KV dtype must match backing storage")
        if values.numel() * values.element_size() != logical_slots.numel() * self.width:
            raise ValueError("one complete KV row is required per logical slot")
        if self.started:
            self.layer_cache(layer)
        if logical_slots.numel():
            _write[(logical_slots.numel(),)](
                self.backing[layer].view(torch.uint8),
                self.resident[layer].view(torch.uint8),
                self.mapping,
                logical_slots,
                values.view(torch.uint8),
                self.width,
                triton.next_power_of_2(self.width),
                self.hbm_tokens,
            )

    def invalidate(self, logical_slots: torch.Tensor) -> None:
        """Invalidate all layers before allocator reuse or external backing writes."""
        self._check_ids(logical_slots)
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)
        if logical_slots.numel():
            _invalidate[(triton.cdiv(logical_slots.numel(), 256),)](
                logical_slots,
                self.mapping,
                self.tags,
                logical_slots.numel(),
                256,
                self.hbm_tokens,
            )
