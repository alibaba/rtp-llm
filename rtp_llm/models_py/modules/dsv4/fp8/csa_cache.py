"""HBM-first CSA residency with private cross-step token caches.

CPU pages are authoritative. A fixed prefix of backing slots is mirrored in
HBM; the remaining HBM slots are partitioned into 2K-entry request caches.
Page allocation remains with the framework. Request registration is shared by
all CSA layers and runs once per model step. Prefix reuse, CP and MTP are not
supported by this initial integration.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.offload_config import CsaOffloadConfig


@triton.jit
def _begin_prefill(
    Table, Epochs, STRIDE: tl.constexpr, ROWS: tl.constexpr, BLOCKS: tl.constexpr
):
    row = tl.program_id(0)
    first = tl.load(Table + row * STRIDE)
    if (first > 0) & (first < BLOCKS):
        tl.atomic_add(Epochs + first, 1)


@triton.jit
def _register_requests(
    Table,
    Positions,
    RequestEpochs,
    Owners,
    OwnerEpochs,
    PartitionEpochs,
    Rows,
    STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
):
    slot = tl.arange(0, WIDTH)
    incoming = tl.load(Table + slot * STRIDE, slot < BATCH, other=-1)
    position = tl.load(Positions + slot, slot < BATCH, other=0)
    incoming = tl.where((position > 0) & (incoming > 0), incoming, -1)
    generation = tl.load(RequestEpochs + incoming, incoming >= 0, other=-1)
    owners = tl.load(Owners + slot, slot < CAPACITY, other=-2)
    owner_epochs = tl.load(OwnerEpochs + slot, slot < CAPACITY, other=-2)
    epochs = tl.load(PartitionEpochs + slot, slot < CAPACITY, other=0)
    live = (
        tl.sum(
            (
                (owners[:, None] == incoming[None, :])
                & (owner_epochs[:, None] == generation[None, :])
                & (incoming[None, :] >= 0)
            ).to(tl.int32),
            1,
        )
        > 0
    )
    for row in range(BATCH):
        identity = tl.sum(tl.where(slot == row, incoming, 0), 0)
        version = tl.sum(tl.where(slot == row, generation, 0), 0)
        chosen = -1
        if identity >= 0:
            existing = (
                (slot < CAPACITY) & (owners == identity) & (owner_epochs == version)
            )
            chosen = tl.min(tl.where(existing, slot, WIDTH), 0)
            if chosen == WIDTH:
                chosen = tl.min(tl.where((slot < CAPACITY) & ~live, slot, WIDTH), 0)
                epochs = tl.where(slot == chosen, epochs + 1, epochs)
                owners = tl.where(slot == chosen, identity, owners)
                owner_epochs = tl.where(slot == chosen, version, owner_epochs)
            live = live | (slot == chosen)
        tl.store(Rows + row, chosen)
    tl.store(Owners + slot, owners, slot < CAPACITY)
    tl.store(OwnerEpochs + slot, owner_epochs, slot < CAPACITY)
    tl.store(PartitionEpochs + slot, epochs, slot < CAPACITY)


class CsaRequestSlots:
    def __init__(self, blocks, capacity, device):
        self.capacity = capacity
        self.request_epochs = torch.zeros(blocks, dtype=torch.int64, device=device)
        self.owners = torch.full((capacity,), -1, dtype=torch.int64, device=device)
        self.owner_epochs = torch.full_like(self.owners, -1)
        self.epochs = torch.zeros_like(self.owners)
        self.rows = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self.bytes = sum(
            t.numel() * t.element_size()
            for t in (
                self.request_epochs,
                self.owners,
                self.owner_epochs,
                self.epochs,
                self.rows,
            )
        )

    def begin_prefill(self, block_table):
        if block_table.shape[0]:
            _begin_prefill[(block_table.shape[0],)](
                block_table,
                self.request_epochs,
                STRIDE=block_table.stride(0),
                ROWS=block_table.shape[0],
                BLOCKS=self.request_epochs.numel(),
            )

    def register(self, block_table, positions):
        batch = block_table.shape[0]
        if not 0 < batch <= self.capacity:
            raise ValueError("CSA offload batch exceeds private-cache capacity")
        _register_requests[(1,)](
            block_table,
            positions,
            self.request_epochs,
            self.owners,
            self.owner_epochs,
            self.epochs,
            self.rows,
            STRIDE=block_table.stride(0),
            BATCH=batch,
            CAPACITY=self.capacity,
            WIDTH=triton.next_power_of_2(self.capacity),
            num_warps=4,
        )


@triton.jit
def _copy_slot(
    Source,
    Destination,
    source_slot,
    dest_slot,
    ENTRIES: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
):
    byte = tl.arange(0, 1024)
    source_slot = source_slot.to(tl.int64)
    dest_slot = dest_slot.to(tl.int64)
    source_offset = source_slot // ENTRIES * SOURCE_STRIDE + tl.where(
        byte < 576,
        source_slot % ENTRIES * 576 + byte,
        ENTRIES * 576 + source_slot % ENTRIES * 8 + byte - 576,
    )
    dest_offset = dest_slot // ENTRIES * DEST_STRIDE + tl.where(
        byte < 576,
        dest_slot % ENTRIES * 576 + byte,
        ENTRIES * 576 + dest_slot % ENTRIES * 8 + byte - 576,
    )
    value = tl.load(Source + source_offset, byte < 584, other=0)
    tl.store(Destination + dest_offset, value, byte < 584)


@triton.jit
def _protect_private(
    Ids,
    Rows,
    Epochs,
    Map,
    Tags,
    Versions,
    Protected,
    Slots,
    RowMisses,
    MissCount,
    Counts,
    TOPK: tl.constexpr,
    HOT: tl.constexpr,
    RESIDENT: tl.constexpr,
    SOURCE_TOKENS: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    if row == 0:
        tl.store(MissCount, 0)
    pos = tl.arange(0, K)
    token = tl.load(Ids + row * TOPK + pos, pos < TOPK, other=-1)
    partition = tl.load(Rows + row)
    valid = (pos < TOPK) & (token >= 0) & (token < SOURCE_TOKENS) & (partition >= 0)
    native = valid & (token < RESIDENT)
    remote = valid & ~native
    hot_hit = tl.full((K,), False, tl.int1)
    hot_slot = tl.full((K,), -1, tl.int32)
    if tl.sum(remote.to(tl.int32), 0) > 0:
        offsets = tl.arange(0, HOT)
        tl.store(Protected + partition * HOT + offsets, 0)
        hot_slot = tl.load(Map + token, remote, other=-1)
        in_partition = (
            remote & (hot_slot >= partition * HOT) & (hot_slot < (partition + 1) * HOT)
        )
        epoch = tl.load(Epochs + partition)
        version = tl.load(Versions + hot_slot, in_partition, other=-1)
        tag = tl.load(Tags + hot_slot, in_partition, other=-1)
        hot_hit = in_partition & (version == epoch) & (tag == token)
        tl.debug_barrier()
        tl.atomic_xchg(Protected + hot_slot, 1, hot_hit, sem="relaxed")
    miss = remote & ~hot_hit
    slot = tl.where(native, token, tl.where(hot_hit, RESIDENT + hot_slot, -2))
    tl.store(Slots + row * TOPK + pos, tl.where(valid, slot, -1), pos < TOPK)
    count = tl.sum(miss.to(tl.int32), 0)
    tl.store(RowMisses + row, count)
    tl.store(Counts + row * 3, tl.sum(native.to(tl.int32), 0))
    tl.store(Counts + row * 3 + 1, tl.sum(hot_hit.to(tl.int32), 0))
    tl.store(Counts + row * 3 + 2, count)


@triton.jit
def _admit_private(
    Ids,
    Rows,
    Epochs,
    Map,
    Tags,
    Versions,
    Protected,
    Slots,
    RowMisses,
    Clock,
    FreeTickets,
    MissSources,
    MissDests,
    MissRows,
    MissCount,
    TOPK: tl.constexpr,
    HOT: tl.constexpr,
    RESIDENT: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    count = tl.load(RowMisses + row)
    if count == 0:
        return
    partition = tl.load(Rows + row)
    epoch = tl.load(Epochs + partition)
    cursor = tl.load(Clock + partition)
    ticket = tl.arange(0, HOT)
    candidate = partition * HOT + (cursor + ticket) % HOT
    available = tl.load(Protected + candidate) == 0
    rank = tl.cumsum(available.to(tl.int32), 0) - 1
    tl.store(FreeTickets + partition * HOT + rank, ticket, available)
    tl.debug_barrier()
    pos = tl.arange(0, K)
    miss = tl.load(Slots + row * TOPK + pos, pos < TOPK, other=-1) == -2
    rank = tl.cumsum(miss.to(tl.int32), 0) - 1
    chosen_ticket = tl.load(FreeTickets + partition * HOT + rank, miss, other=0)
    chosen = partition * HOT + (cursor + chosen_ticket) % HOT
    token = tl.load(Ids + row * TOPK + pos, miss, other=-1)
    tl.store(Tags + chosen, token, miss)
    tl.store(Versions + chosen, epoch, miss)
    tl.store(Map + token, chosen, miss)
    tl.store(Slots + row * TOPK + pos, RESIDENT + chosen, miss)
    start = tl.atomic_add(MissCount, count, sem="relaxed")
    tl.store(MissSources + start + rank, token, miss)
    tl.store(MissDests + start + rank, RESIDENT + chosen, miss)
    tl.store(MissRows + start + rank, row, miss)
    last = tl.max(tl.where(miss, chosen_ticket, 0), 0)
    tl.store(Clock + partition, (cursor + last + 1) % HOT)


@triton.jit
def _fetch_private(
    Source,
    Destination,
    MissSources,
    MissDests,
    MissRows,
    MissCount,
    DeferredWrites,
    ENTRIES: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
):
    count = tl.load(MissCount)
    for index in range(tl.program_id(0), count, tl.num_programs(0)):
        source = tl.load(MissSources + index)
        dest = tl.load(MissDests + index)
        row = tl.load(MissRows + index)
        pending = tl.load(DeferredWrites + row)
        # The main compressor has not produced this boundary entry yet.
        if source != pending:
            _copy_slot(
                Source, Destination, source, dest, ENTRIES, SOURCE_STRIDE, DEST_STRIDE
            )


@triton.jit
def _mirror_writes(
    Source,
    Destination,
    Ids,
    Map,
    Tags,
    N: tl.constexpr,
    SOURCE_TOKENS: tl.constexpr,
    RESIDENT: tl.constexpr,
    ENTRIES: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
):
    for index in range(tl.program_id(0), N, tl.num_programs(0)):
        source = tl.load(Ids + index).to(tl.int32)
        if (source >= 0) & (source < SOURCE_TOKENS):
            dest = source
            if source >= RESIDENT:
                hot = tl.load(Map + source)
                tag = tl.load(Tags + hot, hot >= 0, other=-1)
                dest = tl.where((hot >= 0) & (tag == source), RESIDENT + hot, -1)
            if dest >= 0:
                _copy_slot(
                    Source,
                    Destination,
                    source,
                    dest,
                    ENTRIES,
                    SOURCE_STRIDE,
                    DEST_STRIDE,
                )


class CsaLayerCache:
    def __init__(
        self,
        source,
        requests,
        *,
        budget_bytes,
        topk,
        hot_entries=2048,
        fetch_ctas=64,
        device="cuda",
    ):
        if source.device.type != "cpu" or not source.is_pinned():
            raise ValueError("CSA backing must be pinned CPU memory")
        if source.dtype != torch.uint8 or source.ndim != 3 or source.shape[-1] != 584:
            raise ValueError("CSA backing must use packed MODEL1 KV")
        entries = source.shape[1]
        if (
            hot_entries < topk
            or hot_entries % entries
            or hot_entries & (hot_entries - 1)
        ):
            raise ValueError(
                "private hot capacity must cover TopK and align to KV blocks"
            )
        self.source = source
        self.requests = requests
        self.topk = topk
        self.hot_entries = hot_entries
        self.fetch_ctas = fetch_ctas
        batch = requests.capacity
        hot_total = batch * hot_entries
        source_tokens = source.shape[0] * entries
        self.token_map = torch.full(
            (source_tokens,), -1, dtype=torch.int32, device=device
        )
        self.tags = torch.full((hot_total,), -1, dtype=torch.int32, device=device)
        self.versions = torch.full((hot_total,), -1, dtype=torch.int64, device=device)
        self.protected = torch.zeros(hot_total, dtype=torch.int32, device=device)
        self.free_tickets = torch.empty_like(self.protected)
        self.clock = torch.zeros(batch, dtype=torch.int32, device=device)
        self.selected = torch.empty((batch, topk), dtype=torch.int32, device=device)
        self.slots = torch.full((batch, 1, topk), -1, dtype=torch.int32, device=device)
        self.row_misses = torch.zeros(batch, dtype=torch.int32, device=device)
        self.miss_sources = torch.empty(batch * topk, dtype=torch.int32, device=device)
        self.miss_dests = torch.empty_like(self.miss_sources)
        self.miss_rows = torch.empty_like(self.miss_sources)
        self.miss_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.deferred_writes = torch.full(
            (batch,), -1, dtype=torch.int64, device=device
        )
        self.counts = torch.zeros((batch, 3), dtype=torch.int32, device=device)
        self.metadata_bytes = sum(
            t.numel() * t.element_size()
            for t in (
                self.token_map,
                self.tags,
                self.versions,
                self.protected,
                self.free_tickets,
                self.clock,
                self.selected,
                self.slots,
                self.row_misses,
                self.miss_sources,
                self.miss_dests,
                self.miss_rows,
                self.miss_count,
                self.deferred_writes,
                self.counts,
            )
        )
        self.validator = None
        if os.environ.get("DSV4_CSA_VALIDATE_BYTES", "0") == "1":
            from rtp_llm.models_py.modules.dsv4.fp8.kv_offload import CsaByteValidator

            self.validator = CsaByteValidator(device)
            self.metadata_bytes += self.validator.errors.numel() * 8
        stride = triton.cdiv(entries * 584, 576) * 576
        hot_blocks = hot_total // entries
        blocks = (budget_bytes - self.metadata_bytes) // stride
        if blocks <= hot_blocks:
            raise ValueError(
                "CSA GPU budget cannot fit private caches plus resident KV"
            )
        resident_blocks = min(blocks - hot_blocks, source.shape[0])
        self.resident_tokens = resident_blocks * entries
        self._storage = torch.zeros(
            (resident_blocks + hot_blocks, stride), dtype=torch.uint8, device=device
        )
        self.pool = self._storage.as_strided(
            (resident_blocks + hot_blocks, entries, 584), (stride, 584, 1)
        )
        self.allocated_bytes = self._storage.numel() + self.metadata_bytes
        self.stream = torch.cuda.Stream(device=device)
        self.ready = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.pending = False

    def prefetch(self, selected, deferred_writes):
        batch = selected.shape[0]
        if (
            selected.shape != (batch, self.topk)
            or not 0 < batch <= self.requests.capacity
        ):
            raise ValueError("invalid CSA TopK shape")
        self.selected[:batch].copy_(selected)
        self.deferred_writes[:batch].copy_(deferred_writes.reshape(-1)[:batch])
        current = torch.cuda.current_stream(self.pool.device)
        self.ready.record(current)
        self.stream.wait_event(self.ready)
        with torch.cuda.stream(self.stream):
            _protect_private[(batch,)](
                self.selected,
                self.requests.rows,
                self.requests.epochs,
                self.token_map,
                self.tags,
                self.versions,
                self.protected,
                self.slots,
                self.row_misses,
                self.miss_count,
                self.counts,
                TOPK=self.topk,
                HOT=self.hot_entries,
                RESIDENT=self.resident_tokens,
                SOURCE_TOKENS=self.token_map.numel(),
                K=triton.next_power_of_2(self.topk),
                num_warps=4,
            )
            _admit_private[(batch,)](
                self.selected,
                self.requests.rows,
                self.requests.epochs,
                self.token_map,
                self.tags,
                self.versions,
                self.protected,
                self.slots,
                self.row_misses,
                self.clock,
                self.free_tickets,
                self.miss_sources,
                self.miss_dests,
                self.miss_rows,
                self.miss_count,
                TOPK=self.topk,
                HOT=self.hot_entries,
                RESIDENT=self.resident_tokens,
                K=triton.next_power_of_2(self.topk),
                num_warps=4,
            )
            _fetch_private[(self.fetch_ctas,)](
                self.source,
                self.pool,
                self.miss_sources,
                self.miss_dests,
                self.miss_rows,
                self.miss_count,
                self.deferred_writes,
                ENTRIES=self.source.shape[1],
                SOURCE_STRIDE=self.source.stride(0),
                DEST_STRIDE=self.pool.stride(0),
                num_warps=4,
            )
            self.done.record(self.stream)
        self.pending = True
        return self.pool, self.slots[:batch]

    def wait(self):
        if self.pending:
            torch.cuda.current_stream(self.pool.device).wait_event(self.done)
            self.pending = False

    def validate_selection(self, batch):
        if self.validator is not None:
            self.validator.check(
                self.source, self.selected[:batch], self.pool, self.slots[:batch]
            )

    def mirror_writes(self, slots):
        self.wait()
        if slots.numel():
            _mirror_writes[(min(self.fetch_ctas, slots.numel()),)](
                self.source,
                self.pool,
                slots,
                self.token_map,
                self.tags,
                N=slots.numel(),
                SOURCE_TOKENS=self.token_map.numel(),
                RESIDENT=self.resident_tokens,
                ENTRIES=self.source.shape[1],
                SOURCE_STRIDE=self.source.stride(0),
                DEST_STRIDE=self.pool.stride(0),
                num_warps=4,
            )


def initialize_csa_offload(v4, kv_cache, max_batch_size):
    config = CsaOffloadConfig.from_env()
    if config is None or kv_cache is None:
        return
    from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache
    from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_KV

    layers = [layer.attn for layer in v4.layers if layer.attn.compress_ratio == 4]
    if not layers:
        raise ValueError("CSA offload requires at least one CSA layer")
    sources = []
    for attn in layers:
        with bind_attn_cache(attn, kv_cache):
            sources.append(attn._pool_view_3d_fp8(CSA_KV))
    if getattr(v4, "csa_offload", None) is not None:
        if layers[0].csa_offload.source.data_ptr() != sources[0].data_ptr():
            raise RuntimeError(
                "CSA backing allocation changed after cache initialization"
            )
        return
    device = v4.embed.weight.device
    requests = CsaRequestSlots(sources[0].shape[0], max_batch_size, device)
    budget = config.gpu_cache_mib * 1024**2
    layer_budget = (budget - requests.bytes) // len(layers)
    total = requests.bytes
    for attn, source in zip(layers, sources):
        cache = CsaLayerCache(
            source,
            requests,
            budget_bytes=layer_budget,
            topk=attn.indexer.index_topk,
            hot_entries=config.hot_entries,
            fetch_ctas=config.fetch_ctas,
            device=device,
        )
        attn.csa_offload = cache
        attn.compressor.csa_offload = cache
        total += cache.allocated_bytes
    v4.csa_offload = requests
    if cache.validator is not None:
        logging.info(
            "DSV4 CSA byte validation enabled for every decode layer and graph replay"
        )
    logging.info(
        "DSV4 CSA offload initialized: layers=%d batch_capacity=%d hot_entries_per_request=%d "
        "resident_entries_per_layer=%d actual_gpu_bytes=%d reserved_gpu_bytes=%d",
        len(layers),
        max_batch_size,
        config.hot_entries,
        cache.resident_tokens,
        total,
        budget,
    )
