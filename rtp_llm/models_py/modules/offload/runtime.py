from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.offload.layout import FetchRows, LayoutOps

if TYPE_CHECKING:
    from rtp_llm.ops import OffloadStorage


@dataclass(frozen=True)
class KVWrite:
    backing_ids: torch.Tensor
    planes: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class KVReadView:
    planes: tuple[torch.Tensor, ...]
    indices: torch.Tensor


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
    BLOCK_ROWS: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    HBM_ROWS: tl.constexpr,
):
    if tl.program_id(0) == 0:
        tl.store(MissCount, 0)
    i = tl.program_id(0) * B + tl.arange(0, B)
    token = tl.load(Ids + i, i < N, other=-1)
    valid = (i < N) & (token >= HBM_ROWS)
    slot = tl.load(Map + token, valid, other=-1)
    epoch = tl.load(Epoch)
    tl.atomic_xchg(Protected + slot, epoch, valid & (slot >= 0))
    generation = tl.load(
        Generations + token // BLOCK_ROWS, valid & (slot >= 0), other=-1
    )
    version = tl.load(Versions + slot, valid & (slot >= 0), other=-1)
    refresh = valid & ((slot < 0) | (version != generation))
    physical = tl.where(token < HBM_ROWS, token, slot + HBM_ROWS)
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
    BLOCK_ROWS: tl.constexpr,
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
    generation = tl.load(Generations + token // BLOCK_ROWS, owned, other=-1)
    pending = owned & (slot < 0)
    count = tl.sum(owned.to(tl.int32), 0)
    start = tl.atomic_add(MissCount, count, sem="relaxed")
    offset = tl.cumsum(owned.to(tl.int32), 0) - 1
    tl.store(Misses + start + offset, i, owned)
    tl.store(Versions + slot, generation, owned & (slot >= 0))
    epoch = tl.load(Epoch)
    count = tl.sum(pending.to(tl.int32), 0)
    if count == 0:
        return
    tokens = tl.sort(tl.where(pending, token, 0x7FFFFFFF), descending=False)
    admitted = 0
    scan: tl.constexpr = min(B, CAPACITY)
    first_ticket = tl.atomic_add(Clock, count, sem="relaxed")
    while admitted < count:
        ticket = first_ticket + tl.arange(0, B)
        candidate = (ticket % CAPACITY).to(tl.int32)
        available = (tl.arange(0, B) < scan) & (tl.load(Protected + candidate) != epoch)
        rank = tl.cumsum(available.to(tl.int32), 0) - 1
        attempt = available & (rank < count - admitted)
        previous = tl.atomic_xchg(Protected + candidate, epoch, attempt)
        claimed = attempt & (previous != epoch)
        rank = tl.cumsum(claimed.to(tl.int32), 0) - 1
        token = tl.gather(tokens, tl.where(claimed, admitted + rank, 0), 0)
        generation = tl.load(Generations + token // BLOCK_ROWS, claimed, other=-1)
        old = tl.load(Tags + candidate, claimed, other=-1)
        tl.store(Map + old, -1, claimed & (old >= 0))
        tl.store(Tags + candidate, token, claimed)
        tl.store(Versions + candidate, generation, claimed)
        tl.store(Map + token, candidate, claimed)
        admitted += tl.sum(claimed.to(tl.int32), 0)
        # Reserve a whole retry tile atomically, including the one-victim case.
        if admitted < count:
            first_ticket = tl.atomic_add(Clock, scan, sem="relaxed")


@triton.jit
def _finish_admission(
    Ids,
    Slots,
    Map,
    Owners,
    Epoch,
    N: tl.constexpr,
    HBM_ROWS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    refresh = tl.load(Slots + i, i < N, other=-1) == -2
    token = tl.load(Ids + i, refresh, other=-1)
    slot = tl.load(Map + token, refresh, other=-1)
    owner = tl.load(Owners + token, refresh, other=-1)
    tl.store(Slots + i, slot + HBM_ROWS, refresh)
    tl.store(Owners + token, 0x7FFFFFFF, refresh & (owner == i))
    if tl.program_id(0) == 0:
        tl.store(Epoch, tl.load(Epoch) + 1)


@triton.jit
def _write_resident_ids(
    Ids,
    Out,
    Map,
    Generations,
    Versions,
    BLOCK_ROWS: tl.constexpr,
    HBM_ROWS: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    token = tl.load(Ids + i, i < N, other=-1)
    host = (i < N) & (token >= HBM_ROWS)
    slot = tl.load(Map + token, host, other=-1)
    version = tl.load(Versions + slot, host & (slot >= 0), other=-1)
    generation = tl.load(Generations + token // BLOCK_ROWS, host, other=-1)
    resident = tl.where((slot >= 0) & (version == generation), slot + HBM_ROWS, -1)
    tl.store(Out + i, tl.where(token < HBM_ROWS, token, resident), i < N)


class SparseOffloadRuntime:
    """One active selection; the engine serializes forwards and backing reuse."""

    def __init__(self, storages: tuple[OffloadStorage, ...], layout: LayoutOps):
        if not storages:
            raise ValueError("offload group must contain storage")
        first = storages[0]
        geometry = (
            first.total_blocks,
            first.hbm_blocks,
            first.rows_per_block,
            first.rows_per_page,
            first.max_selected_rows,
        )
        total, hbm, block, page, maximum = geometry
        if not (
            2 <= hbm <= total
            and block > 0
            and page > 0
            and block % page == 0
            and maximum > 0
        ):
            raise ValueError("invalid offload storage geometry")
        capacity = (maximum + page - 1) // page * page
        if max(total * block, hbm * block + capacity) > torch.iinfo(torch.int32).max:
            raise ValueError("offload row address spaces must fit int32")
        self.device = first.generations.device
        if (
            self.device.type != "cuda"
            or first.generations.dtype != torch.int64
            or first.generations.shape != (total,)
            or not first.generations.is_contiguous()
        ):
            raise ValueError("generations must be contiguous CUDA int64[N]")
        for storage in storages:
            if (
                (
                    storage.total_blocks,
                    storage.hbm_blocks,
                    storage.rows_per_block,
                    storage.rows_per_page,
                    storage.max_selected_rows,
                )
                != geometry
                or storage.tag != first.tag
                or storage.generations.data_ptr() != first.generations.data_ptr()
                or storage.generations.device != self.device
            ):
                raise ValueError(
                    "group storages must share geometry, tag and generations"
                )
            if not storage.host_planes or len(storage.host_planes) != len(
                storage.resident_planes
            ):
                raise ValueError("offload storage plane count mismatch")
            for host, resident in zip(storage.host_planes, storage.resident_planes):
                if (
                    host.device.type != "cpu"
                    or not host.is_pinned()
                    or resident.device != self.device
                    or host.ndim < 3
                    or prod(host.shape[2:]) == 0
                    or host.shape[1] != page
                    or host.shape[1:] != resident.shape[1:]
                    or host.dtype != resident.dtype
                    or host.shape[0] != (total - hbm) * block // page
                    or resident.shape[0] != (hbm * block + capacity) // page
                ):
                    raise ValueError(
                        "offload planes must be native paged pinned-host/CUDA tensors"
                    )
                for plane in (host, resident):
                    self._check_plane(plane)
        self.storages = storages
        self.layout = layout
        self.capacity = capacity
        self.maximum = maximum
        self.hbm_rows = hbm * block
        self.block_rows = block
        self.generations = first.generations
        self.mapping = torch.full(
            (total * block,), -1, dtype=torch.int32, device=self.device
        )
        self.owners = torch.full_like(self.mapping, 0x7FFFFFFF)
        self.tags = torch.full((capacity,), -1, dtype=torch.int32, device=self.device)
        self.versions = torch.full(
            (capacity,), -1, dtype=torch.int64, device=self.device
        )
        self.protected = torch.zeros_like(self.versions)
        self.epoch = torch.ones((), dtype=torch.int64, device=self.device)
        self.clock = torch.zeros_like(self.epoch)
        self._slots = torch.empty((maximum,), dtype=torch.int32, device=self.device)
        self._misses = torch.empty_like(self._slots)
        self._write_slots = torch.empty_like(self._slots)
        self._miss_count = torch.zeros((), dtype=torch.int32, device=self.device)
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.ready = tuple(torch.cuda.Event() for _ in storages)
        self._kickoff = torch.cuda.Event()
        # Materialize event handles before capture; no work depends on these records.
        for event in (*self.ready, self._kickoff):
            event.record(torch.cuda.current_stream(self.device))
        self._started = False
        self._capturing = False
        self._indices = None

    @staticmethod
    def _check_plane(plane: torch.Tensor) -> None:
        span = 1
        for size, stride in sorted(
            zip(plane.shape, plane.stride()), key=lambda axis: axis[1]
        ):
            if stride <= 0 or (size > 1 and stride < span):
                raise ValueError("native KV plane axes overlap or have invalid strides")
            span += max(size - 1, 0) * stride

    def _check_ids(self, ids: torch.Tensor) -> None:
        if (
            ids.device != self.device
            or ids.dtype != torch.int32
            or not ids.is_contiguous()
        ):
            raise ValueError(
                "backing IDs must be contiguous int32 on the storage device"
            )

    def _wait(self, storage_index: int) -> None:
        if not 0 <= storage_index < len(self.storages):
            raise IndexError("offload storage index out of range")
        # A capture must only wait for events recorded in that capture. The
        # model resets between captures; graph replay includes its own drain.
        if (
            self._started
            and self._capturing == torch.cuda.is_current_stream_capturing()
        ):
            torch.cuda.current_stream(self.device).wait_event(self.ready[storage_index])

    def begin(self, backing_ids: torch.Tensor) -> None:
        """Enqueue fetches for int32 backing IDs; -1 marks unused entries."""
        ids = backing_ids
        self._check_ids(ids)
        n = ids.numel()
        if n > self.maximum:
            raise ValueError("flattened selection exceeds max_selected_rows")
        self.drain()
        compute = torch.cuda.current_stream(self.device)
        self._kickoff.record(compute)
        self.transfer_stream.wait_event(self._kickoff)
        indices = self._slots[:n].view(ids.shape)
        with torch.cuda.stream(self.transfer_stream):
            if n:
                grid = (triton.cdiv(n, 256),)
                _protect[grid](
                    ids,
                    indices,
                    self.owners,
                    self.mapping,
                    self.protected,
                    self.epoch,
                    self._miss_count,
                    self.generations,
                    self.versions,
                    self.block_rows,
                    n,
                    256,
                    self.hbm_rows,
                )
                _admit[grid](
                    ids,
                    indices,
                    self.owners,
                    self.mapping,
                    self.tags,
                    self.protected,
                    self.epoch,
                    self.clock,
                    self._misses,
                    self._miss_count,
                    self.generations,
                    self.versions,
                    self.block_rows,
                    n,
                    self.capacity,
                    256,
                )
                _finish_admission[grid](
                    ids,
                    indices,
                    self.mapping,
                    self.owners,
                    self.epoch,
                    n,
                    self.hbm_rows,
                    256,
                )
            else:
                self._miss_count.zero_()
            for storage, ready in zip(self.storages, self.ready):
                if n:
                    self.layout.fetch(
                        storage, FetchRows(ids, indices, self._misses, self._miss_count)
                    )
                ready.record(self.transfer_stream)
        ids.record_stream(self.transfer_stream)
        self._started = True
        self._capturing = torch.cuda.is_current_stream_capturing()
        self._indices = indices

    def write(self, storage_index: int, data: KVWrite) -> None:
        self._check_ids(data.backing_ids)
        if data.backing_ids.ndim != 1:
            raise ValueError("write backing IDs must have shape [W]")
        self._wait(storage_index)
        storage = self.storages[storage_index]
        n = data.backing_ids.numel()
        if len(data.planes) != len(storage.resident_planes):
            raise ValueError("write plane count mismatch")
        for values, resident in zip(data.planes, storage.resident_planes):
            if (
                values.device != self.device
                or values.dtype != resident.dtype
                or values.shape != (n, *resident.shape[2:])
            ):
                raise ValueError("write planes must contain W native KV rows on CUDA")
            self._check_plane(values)
        # Chunk large writes using fixed scratch; selection size does not limit W.
        for start in range(0, n, self.maximum):
            stop = min(start + self.maximum, n)
            chunk = KVWrite(
                data.backing_ids[start:stop], tuple(p[start:stop] for p in data.planes)
            )
            slots = self._write_slots[: stop - start]
            _write_resident_ids[(triton.cdiv(stop - start, 256),)](
                chunk.backing_ids,
                slots,
                self.mapping,
                self.generations,
                self.versions,
                self.block_rows,
                self.hbm_rows,
                stop - start,
                256,
            )
            self.layout.store(storage, chunk, slots)

    def acquire(self, storage_index: int) -> KVReadView:
        if self._indices is None:
            raise RuntimeError("begin must precede acquire")
        self._wait(storage_index)
        return KVReadView(
            self.storages[storage_index].resident_planes,
            self._indices,
        )

    def drain(self) -> None:
        self._wait(len(self.storages) - 1)

    def reset(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("reset must run outside graph capture")
        torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)
        self.mapping.fill_(-1)
        self.owners.fill_(0x7FFFFFFF)
        self.tags.fill_(-1)
        self.versions.fill_(-1)
        self.protected.zero_()
        self.epoch.fill_(1)
        self.clock.zero_()
        self._miss_count.zero_()
        self._indices = None
        self._started = False
        self._capturing = False
