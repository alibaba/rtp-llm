from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.offload import policy as eviction
from rtp_llm.models_py.modules.offload.layout import FetchRows, LayoutOps, WriteState
from rtp_llm.models_py.modules.offload.policy import ERROR_MISSING_ADMISSION

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
    Tags,
    SlotEpochs,
    HotEpoch,
    ActiveRows,
    Protected,
    Epoch,
    MissCount,
    Generations,
    Versions,
    Error,
    TOTAL_ROWS: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HBM_ROWS: tl.constexpr,
    DYNAMIC: tl.constexpr,
):
    if tl.program_id(0) == 0:
        tl.store(MissCount, 0)
    i = tl.program_id(0) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    token = tl.load(Ids + i, i < NUM_ROWS, other=-1)
    valid = (
        (i < NUM_ROWS)
        & (token >= HBM_ROWS)
        & (token < TOTAL_ROWS)
        & (tl.load(Error) == 0)
    )
    slot = tl.load(Map + token, valid, other=-1)
    if DYNAMIC:
        mapped = valid & (slot >= 0) & (slot < tl.load(ActiveRows))
        tag = tl.load(Tags + slot, mapped, other=-1)
        allocation = tl.load(SlotEpochs + slot, mapped, other=-1)
        expected = tl.load(HotEpoch + slot, mapped, other=-1)
        mapped = mapped & (tag == token) & (allocation == expected)
        tl.store(Map + token, -1, valid & ~mapped)
        slot = tl.where(mapped, slot, -1)
    epoch = tl.load(Epoch)
    tl.atomic_xchg(Protected + slot, epoch, valid & (slot >= 0))
    generation = tl.load(
        Generations + token // ROWS_PER_BLOCK, valid & (slot >= 0), other=-1
    )
    version = tl.load(Versions + slot, valid & (slot >= 0), other=-1)
    refresh = valid & ((slot < 0) | (version != generation))
    physical = tl.where(
        (token >= 0) & (token < HBM_ROWS), token, tl.where(valid, slot + HBM_ROWS, -1)
    )
    tl.store(Slots + i, tl.where(refresh, -2, physical), i < NUM_ROWS)
    tl.atomic_min(Owners + token, i, refresh)


@triton.jit
def _finish_admission(
    Ids,
    Slots,
    Map,
    Owners,
    Epoch,
    Error,
    NUM_ROWS: tl.constexpr,
    HBM_ROWS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    refresh = tl.load(Slots + i, i < NUM_ROWS, other=-1) == -2
    token = tl.load(Ids + i, refresh, other=-1)
    slot = tl.load(Map + token, refresh, other=-1)
    owner = tl.load(Owners + token, refresh, other=-1)
    missing = refresh & (slot < 0)
    tl.atomic_or(
        Error,
        ERROR_MISSING_ADMISSION,
        tl.sum(missing.to(tl.int32), 0) > 0,
        sem="relaxed",
    )
    tl.store(
        Slots + i,
        tl.where((slot >= 0) & (tl.load(Error) == 0), slot + HBM_ROWS, -1),
        refresh,
    )
    tl.store(Owners + token, 0x7FFFFFFF, refresh & (owner == i))
    if tl.program_id(0) == 0:
        tl.store(Epoch, tl.load(Epoch) + 1)


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
            for name in ("active_hot_rows", "hot_cache_epoch"):
                expected = getattr(first, name, None)
                actual = getattr(storage, name, None)
                if (expected is None) != (actual is None) or (
                    expected is not None and expected.data_ptr() != actual.data_ptr()
                ):
                    raise ValueError(
                        "group storages must share hot-cache capacity and allocation epochs"
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
        self.active_rows = getattr(first, "active_hot_rows", None)
        self.hot_epoch = getattr(first, "hot_cache_epoch", None)
        self.dynamic = self.active_rows is not None
        if not self.dynamic and self.hot_epoch is not None:
            raise ValueError("allocation epochs require a dynamic hot cache")
        if self.dynamic and (
            self.active_rows.device != self.device
            or self.active_rows.dtype != torch.int32
            or self.active_rows.numel() != 1
            or self.hot_epoch is None
            or self.hot_epoch.device != self.device
            or self.hot_epoch.dtype != torch.int64
            or self.hot_epoch.shape != (capacity,)
        ):
            raise ValueError(
                "hot cache needs a CUDA int32 capacity and int64 allocation epochs"
            )
        self.mapping = torch.full(
            (total * block,), -1, dtype=torch.int32, device=self.device
        )
        self.owners = torch.full_like(self.mapping, 0x7FFFFFFF)
        self.tags = torch.full((capacity,), -1, dtype=torch.int32, device=self.device)
        self.versions = torch.full(
            (capacity,), -1, dtype=torch.int64, device=self.device
        )
        self.protected = torch.zeros_like(self.versions)
        # Static kernels compile out allocation checks and need no extra arrays.
        self.slot_epochs = (
            torch.full_like(self.versions, -1) if self.dynamic else self.versions
        )
        self.epoch = torch.ones((), dtype=torch.int64, device=self.device)
        self.policy = getattr(first, "eviction_policy", "rr")
        if self.policy not in ("rr", "lru"):
            raise ValueError("sparse_kv_offload_policy must be rr or lru")
        self._legacy_global = int(getattr(first, "max_requests", 0)) == 0
        self.max_requests = max(1, int(getattr(first, "max_requests", 0)))
        if maximum % self.max_requests:
            raise ValueError(
                "offload selection capacity must contain whole request quotas"
            )
        self.quota = maximum // self.max_requests
        for storage in storages:
            if (
                getattr(storage, "eviction_policy", "rr") != self.policy
                or max(1, int(getattr(storage, "max_requests", 0))) != self.max_requests
            ):
                raise ValueError(
                    "group storages must share eviction policy and request quotas"
                )
        self.clock = torch.zeros(
            (self.max_requests,), dtype=torch.int64, device=self.device
        )
        self._bound_keys = torch.zeros_like(self.clock)
        self._bound_generations = torch.full_like(self.clock, -1)
        self._seen_requests = torch.zeros_like(self.clock)
        self._kept_partitions = torch.zeros_like(self.clock)
        self._reset_partitions = torch.zeros_like(self.clock)
        self._request_parts = torch.empty(
            (self.max_requests,), dtype=torch.int32, device=self.device
        )
        self._first_requests = torch.empty_like(self._request_parts)
        self._pending_count = torch.zeros_like(self._request_parts)
        self._slots = torch.empty((maximum,), dtype=torch.int32, device=self.device)
        self._misses = torch.empty_like(self._slots)
        self._pending_positions = torch.empty_like(self._slots)
        # Share the counter allocation, avoiding another allocator-sized chunk.
        # Error is sticky across selections/replay; reset() explicitly clears it.
        # Device-side validation accumulates the ERROR_* bits from policy.py.
        self._counts = torch.zeros((2,), dtype=torch.int32, device=self.device)
        self._miss_count, self.error = self._counts[0], self._counts[1]
        if not self.dynamic:
            self.active_rows = self._miss_count
            self.hot_epoch = self.versions
        self._write_state = WriteState(
            self.mapping,
            self.generations,
            self.versions,
            self.tags,
            self.slot_epochs,
            self.hot_epoch,
            self.active_rows,
            self.error,
            self.dynamic,
        )
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

    def begin(
        self,
        backing_ids: torch.Tensor,
        *,
        request_keys: torch.Tensor | None = None,
        request_ids: torch.Tensor | None = None,
    ) -> None:
        """Deduplicate Host reads globally; evict only within the owning request.

        request_keys holds each request's first physical page ID. request_ids
        maps each query (the first selection dimension) to that request, also
        for ragged MTP verify/extend inputs. -1 selection entries are padding.
        """
        ids = backing_ids
        self._check_ids(ids)
        n = ids.numel()
        if n > self.maximum:
            raise ValueError("flattened selection exceeds max_selected_rows")
        anonymous = self._legacy_global or (
            request_keys is None and request_ids is None
        )
        if anonymous:
            if self.max_requests != 1:
                raise ValueError("request identity is required for partitioned offload")
            keys = query_requests = self._first_requests
            requests, width, key_stride = 1, max(n, 1), 1
        else:
            if request_keys is None or request_ids is None or ids.ndim < 2:
                raise ValueError(
                    "offload needs request keys and one request ID per query"
                )
            keys, query_requests = request_keys, request_ids
            if (
                keys.device != self.device
                or keys.dtype != torch.int32
                or keys.ndim != 1
                or not 0 < keys.numel() <= self.max_requests
                or keys.stride(0) <= 0
                or query_requests.device != self.device
                or query_requests.dtype != torch.int32
                or query_requests.ndim != 1
                or not query_requests.is_contiguous()
                or query_requests.numel() != ids.shape[0]
            ):
                raise ValueError("invalid offload request identity tensors")
            requests, width, key_stride = (
                keys.numel(),
                prod(ids.shape[1:]),
                keys.stride(0),
            )
            if width <= 0:
                raise ValueError("offload query selection width must be positive")
        self.drain()
        compute = torch.cuda.current_stream(self.device)
        self._kickoff.record(compute)
        self.transfer_stream.wait_event(self._kickoff)
        indices = self._slots[:n].view(ids.shape)
        with torch.cuda.stream(self.transfer_stream):
            if n:
                grid = (triton.cdiv(n, 256),)
                eviction.mark_requests[(triton.cdiv(max(n, self.max_requests), 256),)](
                    ids,
                    query_requests,
                    self._seen_requests,
                    self.epoch,
                    self._pending_count,
                    self.error,
                    NUM_ROWS=n,
                    ROWS_PER_QUERY=width,
                    NUM_REQUESTS=requests,
                    MAX_REQUESTS=self.max_requests,
                    ANONYMOUS=anonymous,
                    HBM_ROWS=self.hbm_rows,
                    TOTAL_ROWS=self.mapping.numel(),
                    TILE_SIZE=256,
                )
                eviction.match_requests[(requests,)](
                    keys,
                    self.generations,
                    self._seen_requests,
                    self._bound_keys,
                    self._bound_generations,
                    self._request_parts,
                    self._first_requests,
                    self._kept_partitions,
                    self.active_rows,
                    self.epoch,
                    self.error,
                    NUM_REQUESTS=requests,
                    MAX_REQUESTS=self.max_requests,
                    ROWS_PER_REQUEST=self.quota,
                    KEY_STRIDE=key_stride,
                    ROWS_PER_PAGE=self.storages[0].rows_per_page,
                    ROWS_PER_BLOCK=self.block_rows,
                    TOTAL_ROWS=self.mapping.numel(),
                    DYNAMIC=self.dynamic,
                    ANONYMOUS=anonymous,
                    TILE_SIZE=256,
                )
                eviction.assign_requests[(1,)](
                    keys,
                    self.generations,
                    self._bound_keys,
                    self._bound_generations,
                    self._request_parts,
                    self._first_requests,
                    self._kept_partitions,
                    self._reset_partitions,
                    self.active_rows,
                    self.epoch,
                    self.error,
                    NUM_REQUESTS=requests,
                    MAX_REQUESTS=self.max_requests,
                    ROWS_PER_REQUEST=self.quota,
                    KEY_STRIDE=key_stride,
                    ROWS_PER_PAGE=self.storages[0].rows_per_page,
                    ROWS_PER_BLOCK=self.block_rows,
                    TOTAL_ROWS=self.mapping.numel(),
                    DYNAMIC=self.dynamic,
                    ANONYMOUS=anonymous,
                    TILE_SIZE=triton.next_power_of_2(self.max_requests),
                )
                eviction.reset_partitions[(self.max_requests,)](
                    self._reset_partitions,
                    self.epoch,
                    self.tags,
                    self.mapping,
                    self.versions,
                    self.slot_epochs,
                    self.protected,
                    self.clock,
                    self.error,
                    ROWS_PER_REQUEST=self.quota,
                    DYNAMIC=self.dynamic,
                    TILE_SIZE=256,
                )
                _protect[grid](
                    ids,
                    indices,
                    self.owners,
                    self.mapping,
                    self.tags,
                    self.slot_epochs,
                    self.hot_epoch,
                    self.active_rows,
                    self.protected,
                    self.epoch,
                    self._miss_count,
                    self.generations,
                    self.versions,
                    self.error,
                    TOTAL_ROWS=self.mapping.numel(),
                    ROWS_PER_BLOCK=self.block_rows,
                    NUM_ROWS=n,
                    TILE_SIZE=256,
                    HBM_ROWS=self.hbm_rows,
                    DYNAMIC=self.dynamic,
                )
                eviction.collect_misses[grid](
                    ids,
                    indices,
                    self.owners,
                    self.mapping,
                    self._misses,
                    self._miss_count,
                    self.generations,
                    self.versions,
                    query_requests,
                    self._request_parts,
                    self._first_requests,
                    self._pending_positions,
                    self._pending_count,
                    self.error,
                    NUM_ROWS=n,
                    ROWS_PER_QUERY=width,
                    ROWS_PER_BLOCK=self.block_rows,
                    ROWS_PER_REQUEST=self.quota,
                    NUM_REQUESTS=requests,
                    ANONYMOUS=anonymous,
                    TILE_SIZE=256,
                )
                eviction.admit[(self.max_requests,)](
                    ids,
                    self._pending_positions,
                    self._pending_count,
                    self.mapping,
                    self.tags,
                    self.versions,
                    self.slot_epochs,
                    self.hot_epoch,
                    self.protected,
                    self.epoch,
                    self.clock,
                    self.generations,
                    self.error,
                    ROWS_PER_REQUEST=self.quota,
                    ROWS_PER_BLOCK=self.block_rows,
                    LRU=self.policy == "lru",
                    DYNAMIC=self.dynamic,
                    TILE_SIZE=(
                        triton.next_power_of_2(self.quota)
                        if self.policy == "lru"
                        else 256
                    ),
                    num_warps=32 if self.policy == "lru" and self.quota >= 2048 else 4,
                )
                _finish_admission[grid](
                    ids,
                    indices,
                    self.mapping,
                    self.owners,
                    self.epoch,
                    self.error,
                    NUM_ROWS=n,
                    HBM_ROWS=self.hbm_rows,
                    TILE_SIZE=256,
                )
            else:
                self._miss_count.zero_()
            for storage, ready in zip(self.storages, self.ready):
                if n:
                    self.layout.fetch(
                        storage,
                        FetchRows(
                            ids, indices, self._misses, self._miss_count, self.error
                        ),
                    )
                ready.record(self.transfer_stream)
        ids.record_stream(self.transfer_stream)
        if not anonymous:
            keys.record_stream(self.transfer_stream)
            query_requests.record_stream(self.transfer_stream)
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
        for start in range(0, n, self.maximum):
            stop = min(start + self.maximum, n)
            chunk = KVWrite(
                data.backing_ids[start:stop], tuple(p[start:stop] for p in data.planes)
            )
            self.layout.store(storage, chunk, self._write_state)

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
        self.error.zero_()
        self.mapping.fill_(-1)
        self.owners.fill_(0x7FFFFFFF)
        self.tags.fill_(-1)
        self.versions.fill_(-1)
        if self.dynamic:
            self.slot_epochs.fill_(-1)
        self.protected.zero_()
        self.epoch.fill_(1)
        self.clock.zero_()
        self._bound_keys.zero_()
        self._bound_generations.fill_(-1)
        self._seen_requests.zero_()
        self._kept_partitions.zero_()
        self._reset_partitions.zero_()
        self._pending_count.zero_()
        self._miss_count.zero_()
        self._indices = None
        self._started = False
        self._capturing = False
