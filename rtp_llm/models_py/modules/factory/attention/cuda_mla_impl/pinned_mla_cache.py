"""Token working set for one DSA shared-index group.

The caller owns the logical block allocator and the HBM indexer cache. Supply
physical *backing-store* token IDs after applying its block table. All layers
in a group use the same resident-token map, but retain independent KV bytes.
Allocator generations or explicit invalidate handle reused backing slots;
call write for newly computed KV. CPU access to backing storage must wait for
the compute stream. This component does not change the attention KV format.
"""

from typing import Sequence

import torch
import triton
import triton.language as tl


def build_working_sets(kv_cache, config):
    from rtp_llm.utils.dsa_indexing import build_dsa_indexer_kv_slot_mapping

    mapping = build_dsa_indexer_kv_slot_mapping(config, config.num_layers)
    groups = {}
    for layer, group in enumerate(mapping):
        groups.setdefault(group, []).append(layer)
    result = {}
    for layers in groups.values():
        backing = [kv_cache.get_layer_cache(layer).kv_cache_base for layer in layers]
        working = PinnedMlaWorkingSet(
            backing,
            kv_cache.dsa_mla_resident_tokens,
            kv_cache.kernel_seq_size_per_block,
            torch.device("cuda", torch.cuda.current_device()),
            block_generations=kv_cache.block_generations,
            allocator_block_size=kv_cache.seq_size_per_block,
            hbm_tokens=kv_cache.dsa_mla_hbm_blocks * kv_cache.seq_size_per_block,
            hbm_cache=[tensor.reshape(-1, kv_cache.kernel_seq_size_per_block, tensor.shape[-1])
                       for layer in layers for tensor in [kv_cache.mla_hbm_cache_by_layer[layer]]],
        )
        for offset, layer in enumerate(layers):
            result[layer] = (working, offset)
    return result


@triton.jit
def _protect(
    Ids, Slots, Owners, Map, Protected, Epoch, MissCount, Generations, Versions,
    BLOCK_SIZE: tl.constexpr, N: tl.constexpr, B: tl.constexpr, HBM_TOKENS: tl.constexpr = 0,
):
    if tl.program_id(0) == 0:
        tl.store(MissCount, 0)
    i = tl.program_id(0) * B + tl.arange(0, B)
    token = tl.load(Ids + i, i < N, other=-1)
    valid = (i < N) & (token >= HBM_TOKENS)
    slot = tl.load(Map + token, valid, other=-1)
    epoch = tl.load(Epoch)
    tl.atomic_xchg(Protected + slot, epoch, valid & (slot >= 0))
    generation = tl.load(Generations + token // BLOCK_SIZE, valid & (slot >= 0), other=-1)
    version = tl.load(Versions + slot, valid & (slot >= 0), other=-1)
    refresh = valid & ((slot < 0) | (version != generation))
    # Fresh hits already have their final physical index. Only misses need
    # ownership arbitration and a subsequent remap; -1 remains padding.
    physical = tl.where(token < HBM_TOKENS, token, slot + HBM_TOKENS)
    tl.store(Slots + i, tl.where(refresh, -2, physical), i < N)
    tl.atomic_min(Owners + token, i, refresh)


@triton.jit
def _admit(
    Ids, Slots, Owners, Map, Tags, Protected, Epoch, Clock, Misses, MissCount, Generations, Versions,
    BLOCK_SIZE: tl.constexpr, N: tl.constexpr, CAPACITY: tl.constexpr, B: tl.constexpr,
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
    while admitted < count:
        remaining = count - admitted
        first_ticket = tl.atomic_add(Clock, remaining, sem="relaxed")
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
        # Skip a blocked tile, but do not skip unused slots on small admissions.
        # Overlapping lookahead is safe: only the atomic winner owns a slot.
        if claimed_count == 0:
            tl.atomic_add(Clock, scan - remaining, sem="relaxed")


@triton.jit
def _fetch(
    Host, Device, Ids, Slots, Map, Owners, Epoch, Misses, MissCount,
    WIDTH: tl.constexpr, B: tl.constexpr, N: tl.constexpr, REMAP: tl.constexpr,
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
def _write(Host, Device, Map, Ids, Values, WIDTH: tl.constexpr, B: tl.constexpr,
           HBM_TOKENS: tl.constexpr = 0):
    i = tl.program_id(0)
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
def _invalidate(Ids, Map, Tags, N: tl.constexpr, B: tl.constexpr, HBM_TOKENS: tl.constexpr = 0):
    i = tl.program_id(0) * B + tl.arange(0, B)
    token = tl.load(Ids + i, i < N, other=-1)
    valid = (i < N) & (token >= HBM_TOKENS)
    slot = tl.load(Map + token, valid, other=-1)
    tl.store(Tags + slot, -1, valid & (slot >= 0))
    tl.store(Map + token, -1, valid)


@triton.jit
def _check_selected_bytes(
    Host, Device, Ids, Slots, Errors, N: tl.constexpr, CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr, B: tl.constexpr, HBM_TOKENS: tl.constexpr = 0,
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
            expected = tl.load(Host + (token - HBM_TOKENS) * WIDTH + x, x < WIDTH, other=0)
            actual = tl.load(Device + slot * WIDTH + x, valid & (x < WIDTH), other=0)
            bad = ~valid | (tl.sum((actual != expected).to(tl.int32), 0) > 0)
        if bad:
            tl.atomic_add(Errors, 1, sem="relaxed")


@triton.jit
def _gather_tiered_fp8_mla(
    Host,
    Hbm,
    Table,
    Lengths,
    Starts,
    Out,
    TABLE_STRIDE: tl.constexpr,
    TABLE_COLUMNS: tl.constexpr,
    BATCH: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    PAGE: tl.constexpr,
    HBM_TOKENS: tl.constexpr,
    LOGICAL_TOKENS: tl.constexpr,
):
    row = tl.program_id(0)
    batch_ids = tl.arange(0, BATCH_BLOCK)
    starts = tl.load(Starts + batch_ids, batch_ids < BATCH, other=0x7FFFFFFF)
    request = tl.sum((starts <= row).to(tl.int32), 0) - 1
    pos = row - tl.load(Starts + request)
    length = tl.load(Lengths + request)
    valid = (pos >= 0) & (pos < length) & (pos // PAGE < TABLE_COLUMNS)
    block = tl.load(Table + request * TABLE_STRIDE + pos // PAGE, valid, other=-1)
    token = block.to(tl.int64) * PAGE + pos % PAGE
    valid = valid & (block >= 0) & (token < LOGICAL_TOKENS)
    in_hbm = valid & (token < HBM_TOKENS)
    in_host = valid & (token >= HBM_TOKENS)
    device_row = Hbm + token * 656
    host_row = Host + (token - HBM_TOKENS) * 656
    d = tl.arange(0, 1024)
    k_mask = d < 512
    k_hbm = tl.load(device_row + d, in_hbm & k_mask, other=0)
    k_host = tl.load(host_row + d, in_host & k_mask, other=0)
    k = tl.where(in_hbm, k_hbm, k_host).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale_offset = 512 + (d // 128) * 4
    scale_hbm = tl.load(
        (device_row + scale_offset).to(tl.pointer_type(tl.float32)),
        in_hbm & k_mask,
        other=0.0,
    )
    scale_host = tl.load(
        (host_row + scale_offset).to(tl.pointer_type(tl.float32)),
        in_host & k_mask,
        other=0.0,
    )
    scale = tl.where(in_hbm, scale_hbm, scale_host)
    rope_mask = (d >= 512) & (d < 576)
    rope_offset = 528 + (d - 512) * 2
    rope_hbm = tl.load(
        (device_row + rope_offset).to(tl.pointer_type(tl.bfloat16)),
        in_hbm & rope_mask,
        other=0.0,
    ).to(tl.float32)
    rope_host = tl.load(
        (host_row + rope_offset).to(tl.pointer_type(tl.bfloat16)),
        in_host & rope_mask,
        other=0.0,
    ).to(tl.float32)
    rope = tl.where(in_hbm, rope_hbm, rope_host)
    result = tl.where(k_mask, k * scale, rope)
    tl.store(Out + row * 576 + d, result, d < 576)


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
        for tensor in backing:
            if tensor.is_cuda or not tensor.is_pinned() or not tensor.is_contiguous():
                raise ValueError("backing must be contiguous CPU pinned memory")
            if tensor.shape != first.shape or tensor.dtype != first.dtype:
                raise ValueError("shared-index group layers must have identical KV layout")
        self.backing = tuple(backing)
        self.capacity = resident_tokens
        self.hbm_tokens = hbm_tokens
        self.logical_tokens = hbm_tokens + first.shape[0] * page_size
        if hbm_tokens < 0 or hbm_tokens % page_size:
            raise ValueError("HBM capacity must contain whole pages")
        self.width = first[0, 0].numel() * first.element_size()
        shape = ((hbm_tokens + resident_tokens) // page_size, *first.shape[1:])
        self.resident = tuple(hbm_cache) if hbm_cache is not None else tuple(
            torch.empty(shape, dtype=first.dtype, device=self.device) for _ in backing
        )
        if len(self.resident) != len(backing) or any(
            tensor.shape != shape or tensor.dtype != first.dtype
            or tensor.device != self.device or not tensor.is_contiguous()
            for tensor in self.resident
        ):
            raise ValueError("HBM tensors must hold complete blocks followed by the working set")
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
        if self.allocator_block_size <= 0 or self.logical_tokens % self.allocator_block_size:
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
        self.protected = torch.zeros(
            (resident_tokens,), dtype=torch.int64, device=self.device
        )
        self.epoch = torch.ones((), dtype=torch.int64, device=self.device)
        self.clock = torch.zeros((), dtype=torch.int64, device=self.device)
        self.compute_stream = torch.cuda.current_stream(self.device)
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.ready = tuple(torch.cuda.Event() for _ in backing)
        self.started = False

    def _check_ids(self, ids: torch.Tensor) -> None:
        if ids.device != self.mapping.device or ids.dtype not in (
            torch.int32, torch.int64
        ):
            raise ValueError("token IDs must be int32/int64 on the working set device")
        if not ids.is_contiguous():
            raise ValueError("token IDs must be contiguous")
        # The executor serializes model invocations, including graph replay.
        # Capture may use a different stream from model initialization/warmup.
        self.compute_stream = torch.cuda.current_stream(self.device)
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
            self.compute_stream.wait_stream(self.transfer_stream)
        slots = torch.empty_like(logical_indices, dtype=torch.int32)
        misses = torch.empty_like(logical_indices, dtype=torch.int32)
        miss_count = torch.empty((), dtype=torch.int32, device=self.device)
        self.transfer_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.transfer_stream):
            # Admission is independent of Q projection too. Keep both metadata
            # and KV fetch off the compute stream until this layer consumes KV.
            if n:
                grid = (triton.cdiv(n, 256),)
                _protect[grid](
                    logical_indices, slots, self.owners, self.mapping, self.protected,
                    self.epoch, miss_count, self.generations, self.versions,
                    self.allocator_block_size, n, 256, self.hbm_tokens,
                )
                _admit[grid](
                    logical_indices, slots, self.owners, self.mapping, self.tags,
                    self.protected, self.epoch, self.clock, misses, miss_count,
                    self.generations, self.versions, self.allocator_block_size,
                    n, self.capacity, 256,
                )
            for layer, (host, resident, ready) in enumerate(zip(self.backing, self.resident, self.ready)):
                if n:
                    _fetch[(min(n, 128),)](
                        host.view(torch.uint8), resident.view(torch.uint8),
                        logical_indices, slots, self.mapping, self.owners, self.epoch,
                        misses, miss_count, self.width, triton.next_power_of_2(self.width),
                        n, layer == 0, self.hbm_tokens,
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
        self.compute_stream.wait_event(self.ready[layer])
        return self.resident[layer]

    def write(self, layer: int, logical_slots: torch.Tensor, values: torch.Tensor) -> None:
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
                self.mapping, logical_slots, values.view(torch.uint8),
                self.width, triton.next_power_of_2(self.width), self.hbm_tokens,
            )

    def gather_bf16(self, layer, out, block_table, lengths, starts):
        """Gather complete prefill history without admitting it into decode slots."""
        if self.width != 656 or self.backing[layer].element_size() != 1:
            raise ValueError("tiered prefill requires packed FP8 MLA rows of 656 bytes")
        if (
            out.ndim != 2
            or out.shape[1] != 576
            or out.dtype != torch.bfloat16
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "tiered prefill output must be contiguous BF16 [tokens, 576]"
            )
        batch = lengths.numel()
        if (
            batch <= 0
            or starts.numel() != batch
            or block_table.ndim != 2
            or block_table.shape[0] != batch
            or block_table.stride(1) != 1
            or any(
                t.device != self.device or t.dtype not in (torch.int32, torch.int64)
                for t in (block_table, lengths, starts)
            )
            or not lengths.is_contiguous()
            or not starts.is_contiguous()
        ):
            raise ValueError("invalid tiered prefill block table or ragged metadata")
        self.compute_stream = torch.cuda.current_stream(self.device)
        if self.started:
            self.layer_cache(layer)
        if out.shape[0]:
            _gather_tiered_fp8_mla[(out.shape[0],)](
                self.backing[layer].view(torch.uint8),
                self.resident[layer].view(torch.uint8),
                block_table,
                lengths,
                starts,
                out,
                block_table.stride(0),
                block_table.shape[1],
                batch,
                triton.next_power_of_2(batch),
                self.backing[layer].shape[1],
                self.hbm_tokens,
                self.logical_tokens,
            )
        return out

    def invalidate(self, logical_slots: torch.Tensor) -> None:
        """Invalidate all layers before allocator reuse or external backing writes."""
        self._check_ids(logical_slots)
        if not torch.cuda.is_current_stream_capturing():
            self.compute_stream.wait_stream(self.transfer_stream)
        if logical_slots.numel():
            _invalidate[(triton.cdiv(logical_slots.numel(), 256),)](
                logical_slots, self.mapping, self.tags, logical_slots.numel(), 256, self.hbm_tokens,
            )
