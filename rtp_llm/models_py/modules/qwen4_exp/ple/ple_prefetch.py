"""Bounded PLE lookup with separate eager and native graph lifetimes."""

from dataclasses import dataclass
from weakref import ReferenceType, ref

import torch

from rtp_llm.models_py.kernels.cuda.qwen4_ple.gather import gather_local

# Failed capture/teardown must not free pinned storage still read by a graph.
_UNDRAINED_GRAPH_PREFETCHERS = []


@dataclass
class PLEPrefetch:
    rows: torch.Tensor
    ready: torch.cuda.Event
    owner: "PLEPrefetcher"

    def wait(self) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PLE prefetch consumption requires eager execution")
        stream = torch.cuda.current_stream(self.rows.device)
        stream.wait_event(self.ready)
        self.rows.record_stream(stream)
        return self.rows


@dataclass
class _GraphBank:
    ids: torch.Tensor
    rows: torch.Tensor
    stream: torch.cuda.Stream
    producer: torch.cuda.Stream
    capture_stream: torch.cuda.Stream | None = None
    warmup_finished: bool = False

    @property
    def nbytes(self) -> int:
        return self.ids.nbytes + self.rows.nbytes


@dataclass
class GraphPLEPrefetch:
    """A fork/join recorded in a native graph, without eager slot state."""

    _bank: ReferenceType[_GraphBank]
    owner: "PLEPrefetcher"
    producer: torch.cuda.Stream

    @property
    def bank(self) -> _GraphBank:
        bank = self._bank()
        if (
            bank is None
            or self.owner._graph_banks.get(tuple(bank.ids.shape)) is not bank
        ):
            raise RuntimeError("PLE graph prefetch handle is stale")
        return bank

    def wait(self) -> torch.Tensor:
        if self.owner._closed:
            raise RuntimeError("PLE prefetcher is closed")
        bank = self.bank
        stream = torch.cuda.current_stream(bank.rows.device)
        if stream != self.producer:
            raise RuntimeError("PLE graph prefetch must join its producer stream")
        stream.wait_stream(bank.stream)
        return bank.rows


class PLEPrefetcher:
    """Own pinned tables, a side stream, and bounded reusable lookup storage.

    Call finish() after queuing all consumers, including projections that read
    the looked-up rows. The budget bounds retained storage per embedding;
    replaced buffers use stream-deferred reclamation during growth.
    """

    def __init__(
        self,
        shards,
        shard_indices,
        total_shards,
        pointer_table,
        *,
        max_eager_bytes=256 * 1024 * 1024,
        scale=None,
    ):
        if max_eager_bytes <= 0:
            raise ValueError("PLE prefetch memory budget must be positive")
        self.shards = shards
        self.shard_indices = shard_indices
        self.total_shards = total_shards
        self.pointer_table = pointer_table
        self.scale = scale
        self.output_dtype = (
            torch.bfloat16
            if shards[0].dtype == torch.float8_e4m3fn
            else shards[0].dtype
        )
        self.device = pointer_table.device
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
        self._eager_slots = {}
        self._active_slots = set()
        self._graph_banks = {}
        self._max_eager_bytes = max_eager_bytes
        self._closed = False

    def start(self, ids, *, slot=0) -> PLEPrefetch | None:
        if self._closed:
            raise RuntimeError("PLE prefetcher is closed")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PLE prefetch must start outside CUDA Graph capture")
        if ids.device != self.device or ids.dtype != torch.int64:
            raise ValueError("PLE prefetch requires int64 ids on its CUDA device")
        if not isinstance(slot, int) or not 0 <= slot < 128:
            raise ValueError("PLE eager prefetch slot must be in [0, 128)")
        if slot in self._active_slots:
            raise RuntimeError("PLE prefetch slot is already active")
        capacity = max(1, 1 << (max(1, ids.numel()) - 1).bit_length())
        entry = self._eager_slots.get(slot)
        if entry is None or entry[0].shape[0] < capacity:
            needed = (
                capacity
                * self.shards[0].shape[1]
                * torch.empty((), dtype=self.output_dtype).element_size()
            )
            if needed > self._max_eager_bytes:
                return None
            previous = entry[0].nbytes if entry is not None else 0
            while self.retained_bytes - previous + needed > self._max_eager_bytes:
                victim = next(
                    (
                        key
                        for key, value in self._eager_slots.items()
                        if key != slot
                        and key not in self._active_slots
                        and value[3]
                        and value[2].query()
                    ),
                    None,
                )
                if victim is None:
                    return None
                del self._eager_slots[victim]
            entry = (
                torch.empty(
                    capacity,
                    self.shards[0].shape[1],
                    device=self.device,
                    dtype=self.output_dtype,
                ),
                torch.cuda.Event(),
                torch.cuda.Event(),
                False,
            )
            self._eager_slots[slot] = entry
        storage, ready, consumed, recorded = entry
        out = storage[: ids.numel()].view(*ids.shape, self.shards[0].shape[1])
        if recorded:
            self.stream.wait_event(consumed)
        self._active_slots.add(slot)
        producer = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(producer)
        ids.record_stream(self.stream)
        out.record_stream(self.stream)
        self.pointer_table.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            gather_local(
                ids,
                self.shards,
                self.shard_indices,
                self.total_shards,
                self.pointer_table,
                out=out,
                scale=self.scale,
            )
            ready.record(self.stream)
        return PLEPrefetch(out, ready, self)

    def start_graph(self, ids) -> GraphPLEPrefetch | None:
        """Warm up fixed storage, then capture readonly lookup on a side stream.

        Each shape has one bank; the native runner serializes its replays and
        retains the bound cache pools. No bank is evicted while graphs may use
        it. Warmup must run before capture, including the gather kernel's JIT.
        """
        if self._closed:
            raise RuntimeError("PLE prefetcher is closed")
        if ids.device != self.device or ids.dtype != torch.int64:
            raise ValueError("PLE graph prefetch requires int64 ids on its CUDA device")
        if ids.numel() == 0:
            raise ValueError("PLE graph prefetch requires non-empty ids")
        capturing = torch.cuda.is_current_stream_capturing()
        producer = torch.cuda.current_stream(self.device)
        key = tuple(ids.shape)
        bank = self._graph_banks.get(key)
        if bank is None:
            if capturing:
                raise RuntimeError(
                    "PLE graph prefetch shape must be warmed before capture"
                )
            needed = ids.nbytes + (
                ids.numel()
                * self.shards[0].shape[1]
                * torch.empty((), dtype=self.output_dtype).element_size()
            )
            if self.retained_bytes + needed > self._max_eager_bytes:
                return None
            bank = _GraphBank(
                torch.empty(ids.shape, device=self.device, dtype=torch.int64),
                torch.empty(
                    (*ids.shape, self.shards[0].shape[1]),
                    device=self.device,
                    dtype=self.output_dtype,
                ),
                torch.cuda.Stream(device=self.device),
                producer,
            )
            self._graph_banks[key] = bank
        if capturing:
            if not bank.warmup_finished:
                raise RuntimeError(
                    "PLE graph prefetch must finish warmup before capture"
                )
            if bank.capture_stream is not None and producer != bank.capture_stream:
                raise RuntimeError(
                    "PLE graph prefetch bank has a different producer stream"
                )
            bank.producer = producer
            bank.capture_stream = producer
        else:
            if bank.capture_stream is not None:
                raise RuntimeError(
                    "PLE graph prefetch must clear captured banks before warmup"
                )
            if producer != bank.producer and not bank.warmup_finished:
                raise RuntimeError(
                    "PLE graph prefetch bank has a different producer stream"
                )
            bank.producer = producer
            bank.warmup_finished = False
        bank.ids.copy_(ids)
        bank.stream.wait_stream(producer)
        with torch.cuda.stream(bank.stream):
            gather_local(
                bank.ids,
                self.shards,
                self.shard_indices,
                self.total_shards,
                self.pointer_table,
                out=bank.rows,
                scale=self.scale,
            )
        return GraphPLEPrefetch(ref(bank), self, producer)

    def finish_graph_warmup(self) -> None:
        """Prove completion before warmup hands banks to the capture stream."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PLE graph prefetch warmup must finish outside capture")
        for bank in self._graph_banks.values():
            if bank.capture_stream is None:
                bank.producer.synchronize()
                bank.stream.synchronize()
                bank.warmup_finished = True

    def clear_graph_banks(self, *, clear_eager=False) -> None:
        """Call after destroying native graphs, including temporary budget graphs."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "PLE graph banks must clear after CUDA Graph capture ends"
            )
        if clear_eager and self._active_slots:
            raise RuntimeError("PLE prefetch must finish before clearing eager slots")
        if self._graph_banks or (clear_eager and self._eager_slots):
            # Replays may have used a different launch stream. No owners are
            # released if draining fails, including after a failed capture.
            torch.cuda.synchronize(self.device)
            self._graph_banks.clear()
            if clear_eager:
                self._eager_slots.clear()

    def remaining_budget_bytes(self) -> int:
        # Growth replaces power-of-two buffers. Stream-deferred old storage
        # can coexist with the new retained cap, so cover both generations.
        return max(0, 2 * self._max_eager_bytes - self.retained_bytes)

    def finish(self) -> None:
        """Protect eager storage until every queued consumer is finished."""
        stream = torch.cuda.current_stream(self.device)
        for slot in self._active_slots:
            storage, ready, consumed, _ = self._eager_slots[slot]
            stream.wait_event(ready)
            consumed.record(stream)
            self._eager_slots[slot] = (storage, ready, consumed, True)
        self._active_slots.clear()

    @property
    def retained_eager_bytes(self) -> int:
        return sum(entry[0].nbytes for entry in self._eager_slots.values())

    @property
    def retained_graph_bytes(self) -> int:
        return sum(bank.nbytes for bank in self._graph_banks.values())

    @property
    def retained_bytes(self) -> int:
        return self.retained_eager_bytes + self.retained_graph_bytes

    def close(self) -> None:
        """Drain UVA reads before releasing their pinned CPU owners."""
        if not self._closed:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "PLE prefetcher must close after CUDA Graph capture ends"
                )
            self.clear_graph_banks()
            if self._active_slots:
                self.finish()
            self.stream.synchronize()
            for _, _, consumed, recorded in self._eager_slots.values():
                if recorded:
                    consumed.synchronize()
            self._eager_slots.clear()
            self.shards = []
            self.pointer_table = None
            self._closed = True

    def __del__(self):
        if hasattr(self, "stream"):
            try:
                self.close()
            except Exception:
                quarantine = globals().get("_UNDRAINED_GRAPH_PREFETCHERS")
                if quarantine is not None and getattr(self, "_graph_banks", None):
                    quarantine.append(self)
