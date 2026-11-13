from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import uuid
from bisect import bisect_left
from collections import OrderedDict
from concurrent.futures import CancelledError
from itertools import chain, islice
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


class _PoolCapacityError(RuntimeError):
    pass


def _wait_event(event, timeout, cancellation_event):
    if cancellation_event is None:
        return event.wait(timeout=timeout)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if cancellation_event.is_set():
            raise CancelledError("ViT result wait was cancelled")
        if event.is_set():
            return True
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return False
        event.wait(timeout=0.05 if remaining is None else min(0.05, remaining))


class _PoolBlock:
    __slots__ = ("offset", "size")

    def __init__(self, offset: int, size: int):
        self.offset = offset
        self.size = size


class _TensorMemoryPool:
    """One fixed torch allocation with an internal aligned free list."""

    def __init__(self, capacity_bytes: int, device: torch.device):
        self.capacity_bytes = capacity_bytes
        self.device = torch.device(device)
        self._alignment = 8
        self._lock = threading.Lock()
        with cuda_graph_gate.operation():
            self._storage = (
                torch.empty(capacity_bytes, dtype=torch.uint8, device=self.device)
                if capacity_bytes > 0
                else None
            )
        self._free: List[Tuple[int, int]] = (
            [(0, capacity_bytes)] if capacity_bytes > 0 else []
        )
        self._pending: List[Tuple[_PoolBlock, List[Any]]] = []
        self._used_bytes = 0
        # Bytes dropped because they are smaller than the allocation alignment
        # and can never satisfy an aligned request.
        self._wasted_bytes = 0

    @staticmethod
    def _align(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    def _insert_free_locked(self, offset: int, size: int) -> None:
        if size < self._alignment:
            # A sub-alignment fragment can never satisfy an aligned
            # allocation. Keeping it in the free list only makes every
            # first-fit scan longer, so account it as wasted instead.
            self._wasted_bytes += size
            return
        # The list stays address-ordered and fully coalesced. Only the two
        # neighbors can touch a newly freed block; list insertion is still O(n).
        index = bisect_left(self._free, (offset,))
        end = index
        if index and sum(self._free[index - 1]) == offset:
            index -= 1
            offset, previous_size = self._free[index]
            size += previous_size
        if end < len(self._free) and offset + size == self._free[end][0]:
            size += self._free[end][1]
            end += 1
        self._free[index:end] = [(offset, size)]

    def _reclaim_ready_locked(self) -> None:
        waiting = []
        for block, events in self._pending:
            if all(event.query() for event in events):
                self._used_bytes -= block.size
                self._insert_free_locked(block.offset, block.size)
            else:
                waiting.append((block, events))
        self._pending = waiting

    @cuda_graph_gate.operation()
    def reserve(self, requested_bytes: int) -> Optional[_PoolBlock]:
        if requested_bytes == 0:
            return None
        # Align the allocation size as well as the offset. Aligning only the
        # offset leaves a misaligned suffix that the next allocation must
        # re-align, producing an ever-growing population of unusable
        # sub-alignment fragments (observed as ~128k free-list entries).
        size = self._align(requested_bytes, self._alignment)
        if size > self.capacity_bytes:
            # Tiny pools can be smaller than one aligned allocation. Fall back
            # to the exact request so they remain usable; the block end may be
            # unaligned, but there is no room for another allocation anyway.
            size = requested_bytes
        if size > self.capacity_bytes:
            raise _PoolCapacityError(
                f"{self.device.type} pool request {size} exceeds capacity "
                f"{self.capacity_bytes}"
            )
        while True:
            pending_to_wait = None
            with self._lock:
                self._reclaim_ready_locked()
                for index, (offset, free_size) in enumerate(self._free):
                    aligned_offset = self._align(offset, self._alignment)
                    prefix_size = aligned_offset - offset
                    if free_size - prefix_size < size:
                        continue
                    block = _PoolBlock(aligned_offset, size)
                    suffix_offset = aligned_offset + size
                    suffix_size = offset + free_size - suffix_offset
                    # Splitting one free block preserves order and cannot
                    # introduce adjacent free blocks outside this slice.
                    parts = (
                        (offset, prefix_size),
                        (suffix_offset, suffix_size),
                    )
                    self._free[index : index + 1] = [
                        (part_offset, part_size)
                        for part_offset, part_size in parts
                        if part_size >= self._alignment
                    ]
                    self._wasted_bytes += sum(
                        part_size
                        for _, part_size in parts
                        if 0 < part_size < self._alignment
                    )
                    self._used_bytes += size
                    return block
                if self._pending:
                    pending_to_wait = self._pending.pop(0)
            if pending_to_wait is None:
                raise _PoolCapacityError(
                    f"{self.device.type} pool has no contiguous block for {size} bytes"
                )
            block, events = pending_to_wait
            for event in events:
                event.synchronize()
            with self._lock:
                self._used_bytes -= block.size
                self._insert_free_locked(block.offset, block.size)

    @cuda_graph_gate.operation()
    def release(self, block: Optional[_PoolBlock], events: List[Any]) -> None:
        if block is None:
            return
        with self._lock:
            if events and not all(event.query() for event in events):
                self._pending.append((block, events))
            else:
                self._used_bytes -= block.size
                self._insert_free_locked(block.offset, block.size)

    def tensor_view(
        self,
        block: _PoolBlock,
        relative_offset: int,
        nbytes: int,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> torch.Tensor:
        byte_view = self._storage[
            block.offset + relative_offset : block.offset + relative_offset + nbytes
        ]
        return byte_view.view(dtype).view(shape)

    @cuda_graph_gate.operation()
    def stats(self) -> Dict[str, int]:
        with self._lock:
            self._reclaim_ready_locked()
            pending_bytes = sum(block.size for block, _ in self._pending)
            return {
                "capacity_bytes": self.capacity_bytes,
                "used_bytes": self._used_bytes,
                "pending_bytes": pending_bytes,
                "wasted_bytes": self._wasted_bytes,
                "free_bytes": (
                    self.capacity_bytes - self._used_bytes - self._wasted_bytes
                ),
                "largest_free_block_bytes": max(
                    (
                        max(
                            0,
                            size - (self._align(offset, self._alignment) - offset),
                        )
                        for offset, size in self._free
                    ),
                    default=0,
                ),
            }


class _PoolReservation:
    """Own a pool block until all copies reading it have completed."""

    def __init__(self, pool: _TensorMemoryPool, block: Optional[_PoolBlock]):
        self.pool = pool
        self.block = block
        self._events: List[Any] = []
        self._lock = threading.Lock()
        self._released = False

    @cuda_graph_gate.operation()
    def add_events(self, events: List[Any]) -> None:
        if not events:
            return
        with self._lock:
            if self._released:
                for event in events:
                    event.synchronize()
                return
            self._events = [event for event in self._events if not event.query()]
            self._events.extend(events)

    @cuda_graph_gate.operation()
    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            block, events = self.block, self._events
            self.block, self._events = None, []
        self.pool.release(block, events)

    def __del__(self):
        try:
            self.release()
        except Exception:
            # Interpreter shutdown may tear down torch/CUDA before cache owners.
            pass


def _current_cuda_streams(value: Any) -> List[Any]:
    devices = set()
    _map_tensors(value, lambda tensor: devices.add(tensor.device))
    return [
        torch.cuda.current_stream(device) for device in devices if device.type == "cuda"
    ]


@cuda_graph_gate.operation()
def _record_cuda_events(value: Any) -> List[Any]:
    devices = set()
    _map_tensors(value, lambda tensor: devices.add(tensor.device))
    events = []
    for device in devices:
        if device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            events.append(event)
    return events


def _pool_result(
    result: Any,
    devices: Any,
    gpu_pool: Optional[_TensorMemoryPool],
    cpu_pool: Optional[_TensorMemoryPool],
) -> Tuple[Any, List[_PoolReservation], List[Any], int, int]:
    """Compact a tensor tree into fixed pools and preserve repeated objects."""

    specs: Dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.device, int]] = {}
    grouped: Dict[_TensorMemoryPool, List[Tuple[int, torch.device]]] = {}

    def collect(value: Any, target: Any) -> None:
        if isinstance(value, torch.Tensor):
            device = torch.device(target)
            pool = cpu_pool if device.type == "cpu" else gpu_pool
            if pool is None:
                raise _PoolCapacityError(f"no {device.type} pool configured")
            if device.type == "cuda" and device != pool.device:
                raise _PoolCapacityError(
                    f"cache pool is on {pool.device}, result targets {device}"
                )
            identity = (id(value), device)
            if identity not in specs:
                specs[identity] = (
                    value,
                    device,
                    value.numel() * value.element_size(),
                )
                grouped.setdefault(pool, []).append(identity)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                collect(item, target[index])
            return
        if isinstance(value, dict):
            for key, item in value.items():
                collect(item, target[key])

    reservations: Dict[_TensorMemoryPool, _PoolReservation] = {}
    offsets: Dict[Tuple[int, torch.device], int] = {}
    copied: Dict[Tuple[int, torch.device], torch.Tensor] = {}
    gpu_bytes = cpu_bytes = 0
    try:
        collect(result, devices)
        for pool, identities in grouped.items():
            total = 0
            for identity in identities:
                tensor, _, nbytes = specs[identity]
                total = _TensorMemoryPool._align(total, max(1, tensor.element_size()))
                offsets[identity] = total
                total += nbytes
            block = pool.reserve(total)
            reservation = _PoolReservation(pool, block)
            reservations[pool] = reservation
            if pool.device.type == "cuda":
                gpu_bytes += total
            else:
                cpu_bytes += total
            for identity in identities:
                tensor, device, nbytes = specs[identity]
                with cuda_graph_gate.operation():
                    if nbytes == 0:
                        destination = torch.empty(
                            tensor.shape, dtype=tensor.dtype, device=device
                        )
                    else:
                        destination = pool.tensor_view(
                            block,
                            offsets[identity],
                            nbytes,
                            tensor.dtype,
                            tensor.shape,
                        )
                    destination.copy_(tensor.detach(), non_blocking=False)
                copied[identity] = destination

        def rebuild(value: Any, target: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return copied[(id(value), torch.device(target))]
            if isinstance(value, tuple):
                return tuple(rebuild(item, target[i]) for i, item in enumerate(value))
            if isinstance(value, list):
                return [rebuild(item, target[i]) for i, item in enumerate(value)]
            if isinstance(value, dict):
                return {key: rebuild(item, target[key]) for key, item in value.items()}
            return value

        pooled = rebuild(result, devices)
        events = _record_cuda_events(pooled)
        owners = list(reservations.values())
        for owner in owners:
            owner.add_events(events)
        return pooled, owners, events, gpu_bytes, cpu_bytes
    except Exception:
        for reservation in reservations.values():
            reservation.release()
        raise
    finally:
        # Recursive collectors/rebuilders form closure cycles. Neither producer
        # tensors nor pool views should wait for cyclic GC to release ownership.
        specs.clear()
        copied.clear()


@cuda_graph_gate.operation()
def _copy_pooled_result_to_devices(result: Any, devices: Any) -> Any:
    copied = {}

    def visit(value: Any, target: Any) -> Any:
        if isinstance(value, torch.Tensor):
            device = torch.device(target)
            identity = (id(value), device)
            if identity not in copied:
                copied[identity] = value.detach().to(
                    device=device,
                    non_blocking=False,
                    copy=True,
                    memory_format=torch.contiguous_format,
                )
            return copied[identity]
        if isinstance(value, tuple):
            if target is None:
                return tuple(visit(item, None) for item in value)
            return tuple(visit(item, target[i]) for i, item in enumerate(value))
        if isinstance(value, list):
            if target is None:
                return [visit(item, None) for item in value]
            return [visit(item, target[i]) for i, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: visit(item, target[key]) for key, item in value.items()}
        return value

    try:
        return visit(result, devices)
    finally:
        # The recursive visitor forms a closure cycle. Drop its tensor owners
        # immediately, including when copying one of the later leaves fails.
        copied.clear()


def _map_tensors(value: Any, fn: Callable[[torch.Tensor], Any]) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    return value


def _compact_tensor_tier_bytes(result: Any, devices: Any) -> Tuple[int, int]:
    """Bytes required when tensor views are compacted into the cache pools."""

    sizes = [0, 0]
    seen = set()

    def visit(value: Any, target: Any) -> None:
        if isinstance(value, torch.Tensor):
            device = torch.device(target)
            identity = (id(value), device)
            if identity not in seen:
                seen.add(identity)
                tier = device.type == "cpu"
                sizes[tier] = _TensorMemoryPool._align(
                    sizes[tier], max(1, value.element_size())
                )
                sizes[tier] += value.numel() * value.element_size()
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, target[index])
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, target[key])

    visit(result, devices)
    return tuple(sizes)


@cuda_graph_gate.operation()
def _copy_result_to_devices(result: Any, devices: Any) -> Any:
    """Copy without mutating a result that an active request may still hold."""
    copied = {}

    def visit(value: Any, target: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.device == target:
                return value
            identity = (id(value), target)
            if identity not in copied:
                copied[identity] = value.detach().to(
                    device=target,
                    non_blocking=False,
                    memory_format=torch.contiguous_format,
                )
            return copied[identity]
        if isinstance(value, tuple):
            if target is None:
                return tuple(visit(item, None) for item in value)
            return tuple(visit(item, target[i]) for i, item in enumerate(value))
        if isinstance(value, list):
            if target is None:
                return [visit(item, None) for item in value]
            return [visit(item, target[i]) for i, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: visit(item, target[key]) for key, item in value.items()}
        return value

    try:
        return visit(result, devices)
    finally:
        # The recursive visitor forms a closure cycle. Drop its tensor owners
        # immediately, including when copying one of the later leaves fails.
        copied.clear()


def _embedding_output_tokens(result: Any) -> int:
    """Count tokens in the primary embedding, excluding position metadata."""

    def output_tokens(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 0
            if value.ndim >= 2 and value.shape[-1] > 0:
                return value.numel() // value.shape[-1]
            # A one-dimensional embedding is one token vector, not a sequence
            # of scalar tokens. Keep this consistent with the request-level
            # embedding length metric.
            return 1
        if isinstance(value, (list, tuple)):
            return sum(output_tokens(item) for item in value)
        if isinstance(value, dict):
            return sum(output_tokens(item) for item in value.values())
        return 0

    primary = result[0] if isinstance(result, tuple) and result else result
    return output_tokens(primary)


class MMEmbeddingCacheEntry:
    """Three-state embedding cache entry.

    States: PENDING -> COMPLETE or ERROR. GreenNet uses a separate event so its
    verdict can be consumed before the ViT result is ready.
    """

    def __init__(
        self,
        on_complete: Optional[Callable[["MMEmbeddingCacheEntry", Any], None]] = None,
        on_fail: Optional[Callable[["MMEmbeddingCacheEntry", Exception], None]] = None,
        on_read: Optional[
            Callable[["MMEmbeddingCacheEntry", Optional[float]], Any]
        ] = None,
    ):
        self._event = threading.Event()
        self._state_lock = threading.Lock()
        # Claim completion once; _event publishes it after callbacks finish.
        self._terminal = False
        self._on_complete = on_complete
        self._on_fail = on_fail
        self._on_read = on_read
        self.result: Optional[Any] = None
        self.error: Optional[Exception] = None
        self._pending_before_publish: Optional[Callable[[], None]] = None
        self.generation = uuid.uuid4().hex
        # Error telemetry may be observed from several places (the producer
        # callback, a cache waiter, and an RPC handler). Keep one atomic claim
        # on the entry so a failed result contributes one error-QPS sample.
        self._error_reported = False
        self.charge_tokens = 0
        self.charge_bytes = 0
        self.charge_gpu_bytes = 0
        self.charge_cpu_bytes = 0
        self.tier: Optional[str] = None
        self.original_devices: Any = None
        self.offloaded = False
        self.ready_events: List[Any] = []
        self.producer_streams: List[Any] = []
        self.pool_owners: List[_PoolReservation] = []
        self.storage_lock = threading.Lock()
        # A follower can wait for the async producer's verdict, including a
        # KVCM approval, instead of auditing the same in-flight URL again.
        self.expects_greennet = False
        self._greennet_event = threading.Event()
        self._greennet_verdict: Optional[GreenNetVerdict] = None
        self._greennet_error: Optional[Exception] = None
        self._greennet_checked = False
        self._on_shared_approval = None

    @property
    def has_read_callback(self) -> bool:
        """Whether reads must use the cache-owned transfer callback."""
        return self._on_read is not None

    def claim_error_report(self) -> bool:
        """Claim the single error-telemetry sample for this cache entry."""
        with self._state_lock:
            if self._error_reported:
                return False
            self._error_reported = True
            return True

    def wait(
        self,
        timeout: Optional[float] = None,
        *,
        wait_on_current_stream: bool = False,
        cancellation_event: Optional[threading.Event] = None,
    ) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        self.wait_ready(timeout, cancellation_event=cancellation_event)
        if self._on_read is not None:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            return self._on_read(self, remaining)
        # Cache-free async entries publish GPU work without synchronizing the
        # scheduler. The consumer waits on the producer's streams before using
        # the result, including when its own current stream is different.
        for producer in self.producer_streams:
            if wait_on_current_stream:
                with cuda_graph_gate.operation():
                    consumer = torch.cuda.current_stream(producer.device)
                    if consumer != producer:
                        consumer.wait_stream(producer)
            else:
                with cuda_graph_gate.operation():
                    producer.synchronize()
        return self.result

    def wait_ready(
        self,
        timeout: Optional[float] = None,
        *,
        cancellation_event: Optional[threading.Event] = None,
    ) -> None:
        """Wait for completion without promoting an offloaded embedding."""
        if not _wait_event(self._event, timeout, cancellation_event):
            raise TimeoutError("Waiting for embedding result timed out")
        if self.error is not None:
            raise self.error

    def complete(
        self,
        result: Any,
        *,
        before_publish: Optional[Callable[[], None]] = None,
        defer: bool = False,
    ) -> bool:
        with self._state_lock:
            if self._terminal:
                return False
            self._terminal = True
            self.result = result
            self._pending_before_publish = before_publish
        if defer:
            # Publish the result to waiters immediately; the cache insertion is
            # best-effort and runs on the cache's publication thread. A failed
            # insertion must never turn a successful computation into an error.
            self._event.set()
            try:
                if self._on_complete is not None:
                    self._on_complete(self, result)
            except Exception as error:
                logging.warning("ViT cache publication enqueue failed: %s", error)
                try:
                    if self._on_fail is not None:
                        self._on_fail(self, error)
                except Exception:
                    logging.exception("Failed to remove unqueued ViT cache entry")
            return True
        try:
            if self._on_complete is not None:
                self._on_complete(self, result)
            # Publish dependent metadata only after storage succeeds, but
            # before any waiter can observe ready. Duplicate completion never
            # executes this hook. Exceptions use the same rollback as storage.
            if before_publish is not None:
                before_publish()
        except Exception as error:
            # fail() cannot take over this already-claimed transition. Roll it
            # back here so waiters never see a failed insertion as a cache hit.
            self.error = error
            with self.storage_lock:
                self.result = None
                self.original_devices = None
                self.ready_events = []
                self.producer_streams = []
                owners, self.pool_owners = self.pool_owners, []
            try:
                if self._on_fail is not None:
                    self._on_fail(self, error)
            finally:
                for owner in owners:
                    owner.release()
            raise
        finally:
            self._event.set()
        return True

    def fail(self, error: Exception) -> bool:
        with self._state_lock:
            if self._terminal:
                return False
            self._terminal = True
            self.error = error
        try:
            if self._on_fail is not None:
                self._on_fail(self, error)
        finally:
            self._event.set()
        return True

    @property
    def is_done(self) -> bool:
        return self._event.is_set()

    def on_shared_approval(self, callback: Callable[[GreenNetVerdict], None]) -> None:
        """Register once for an unchecked remote hit, including a racing audit."""
        with self._state_lock:
            verdict = self._greennet_verdict
            if self._greennet_error is not None:
                return
            if not (self._greennet_checked and verdict and verdict.passed):
                self._on_shared_approval = callback
                return
        callback(verdict)

    def set_greennet_verdict(
        self, verdict: GreenNetVerdict, *, checked: bool = True
    ) -> None:
        with self._state_lock:
            # A request that skips GreenNet may publish an unchecked embedding
            # while another request independently inspects the same URL. Do not
            # let the unchecked publisher downgrade that approved cache entry.
            if self._greennet_checked and not checked:
                return
            self._greennet_verdict = verdict
            self._greennet_checked = checked
            self._greennet_event.set()
            callback = (
                self._on_shared_approval
                if checked and verdict.passed and self._greennet_error is None
                else None
            )
            if callback is not None:
                self._on_shared_approval = None
        if callback is not None:
            callback(verdict)

    def greennet_snapshot(self) -> Tuple[Optional[GreenNetVerdict], bool]:
        with self._state_lock:
            if self._greennet_error is not None:
                return None, False
            return self._greennet_verdict, self._greennet_checked

    def set_greennet_error(self, error: Exception) -> None:
        """Wake inspection waiters without losing a typed preprocessing error."""
        with self._state_lock:
            self._greennet_error = error
            self._greennet_event.set()

    @property
    def greennet_passed(self) -> bool:
        with self._state_lock:
            return (
                self._greennet_event.is_set()
                and self._greennet_error is None
                and self._greennet_checked
                and self._greennet_verdict is not None
                and self._greennet_verdict.passed
            )

    def wait_greennet(
        self,
        timeout: Optional[float] = None,
        *,
        cancellation_event: Optional[threading.Event] = None,
    ) -> GreenNetVerdict:
        if not _wait_event(self._greennet_event, timeout, cancellation_event):
            raise TimeoutError("Waiting for greennet verdict timed out")
        with self._state_lock:
            if self._greennet_error is not None:
                raise self._greennet_error
            return self._greennet_verdict

    @property
    def is_greennet_decided(self) -> bool:
        return self._greennet_event.is_set()


class _HashCacheValue:
    # All lifecycle fields use the owning hash cache's index lock. A reader
    # lease keeps a retired value alive until its clone completes.
    __slots__ = (
        "hashes",
        "generation",
        "charge_bytes",
        "owners",
        "greennet_passed",
        "readers",
        "retired",
    )

    def __init__(self, generation, charge_bytes, greennet_passed):
        self.hashes = []
        self.generation = generation
        self.charge_bytes = charge_bytes
        self.owners = []
        self.greennet_passed = bool(greennet_passed)
        self.readers = 0
        self.retired = False


class MMHashKeyCache:
    """Byte-bounded CPU tensor cache for multimodal feature-hash token ids.

    Charge tensor storage plus Python key/value metadata. The latter is an
    estimate (not process RSS), but bounds even entries with very few hashes.
    Metadata probes do not update recency; real embedding reuse does.
    """

    _ENTRY_OVERHEAD = sys.getsizeof(OrderedDict([(None, None)])) - sys.getsizeof(
        OrderedDict()
    )

    def __init__(self, max_bytes: int):
        if max_bytes < 0:
            raise ValueError("hash cache max_bytes must be non-negative")
        self._lock = threading.Lock()
        self._drained = threading.Condition(self._lock)
        self._entries = OrderedDict()
        self._pending = {}
        self._max_bytes = max_bytes
        self._resident_bytes = 0
        self._building_bytes = 0
        self._retired_bytes = 0
        self.instance_id = uuid.uuid4().hex

    @property
    def enabled(self) -> bool:
        return self._max_bytes > 0

    def put(
        self,
        cache_key: str,
        feature_hashes: List[torch.Tensor],
        generation: Optional[str] = None,
        greennet_passed: bool = False,
    ) -> None:
        self._put(cache_key, feature_hashes, generation, greennet_passed)

    def put_for_entry(
        self,
        cache_key: str,
        feature_hashes: List[torch.Tensor],
        entry: MMEmbeddingCacheEntry,
        embedding_cache: "MMEmbeddingCache",
    ) -> None:
        """Publish once per live generation; never replace it with a stale writer."""
        self._put(
            cache_key,
            feature_hashes,
            entry.generation,
            entry.greennet_passed,
            entry=entry,
            embedding_cache=embedding_cache,
        )

    def _used_bytes_locked(self) -> int:
        return self._resident_bytes + self._building_bytes + self._retired_bytes

    def _put(
        self,
        cache_key: str,
        feature_hashes: List[torch.Tensor],
        generation: Optional[str],
        greennet_passed: bool,
        *,
        entry: Optional[MMEmbeddingCacheEntry] = None,
        embedding_cache: Optional["MMEmbeddingCache"] = None,
    ) -> None:
        if not cache_key or not self.enabled:
            return
        generation = generation or ""
        # Estimate the final owned int32 value before allocating/copying it.
        # Staging is bounded by the in-flight charge.
        charge_bytes = (
            sum(h.numel() * 4 for h in feature_hashes)
            + sys.getsizeof(cache_key)
            + sys.getsizeof(generation)
            + sys.getsizeof([None for _ in feature_hashes])
            + sum(sys.getsizeof(h) for h in feature_hashes)
            + sys.getsizeof(_HashCacheValue(generation, 0, False))
            + sys.getsizeof(0)
            + self._ENTRY_OVERHEAD
        )

        def current(action):
            # The only nested index order is embedding -> hash, and these
            # callbacks only update metadata: no tensor work or waits.
            if entry is None:
                return action()
            return embedding_cache._with_current_entry(cache_key, entry, action)

        retired = []
        value = _HashCacheValue(generation, charge_bytes, greennet_passed)

        def begin():
            with self._lock:
                if not self.enabled:
                    return False
                if entry is not None:
                    existing = self._pending.get(cache_key) or self._entries.get(
                        cache_key
                    )
                    if existing is not None and existing.generation == generation:
                        existing.greennet_passed |= bool(greennet_passed)
                        return False
                # A replacement supersedes even a still-building older value.
                self._pending.pop(cache_key, None)
                self._remove_locked(cache_key, retired)
                if charge_bytes > self._max_bytes:
                    return False  # Oversized writes do not evict other keys.
                reclaimable = sum(item.charge_bytes for item in retired)
                while (
                    self._used_bytes_locked() + charge_bytes - reclaimable
                    > self._max_bytes
                    and self._entries
                ):
                    reclaimable += self._remove_locked(
                        next(iter(self._entries)), retired
                    )
                self._pending[cache_key] = value
                return True

        started = current(begin)
        self._release_retired(retired)
        if not started:
            return

        def reserve():
            with self._lock:
                if self._pending.get(cache_key) is not value:
                    return False
                if self._used_bytes_locked() + charge_bytes > self._max_bytes:
                    return False  # Pinned readers/builders still own this budget.
                self._building_bytes += charge_bytes
                return True

        if not current(reserve):
            return
        published = False
        try:
            # Hash tensors are a few KB. A fixed first-fit pool only adds a
            # linear free-list scan to every publication; store the CPU tensors
            # directly and let the byte-bounded LRU own eviction.
            with cuda_graph_gate.operation():
                value.hashes = [
                    h.detach().to(device="cpu", dtype=torch.int32).clone()
                    for h in feature_hashes
                ]
            value.owners = []

            def publish():
                with self._lock:
                    if (
                        self._pending.get(cache_key) is not value
                        or self._used_bytes_locked() > self._max_bytes
                    ):
                        return False
                    self._pending.pop(cache_key)
                    self._entries[cache_key] = value
                    self._building_bytes -= charge_bytes
                    self._resident_bytes += charge_bytes
                    return True

            published = bool(current(publish))
        finally:
            if not published:
                with self._lock:
                    if self._pending.get(cache_key) is value:
                        self._pending.pop(cache_key)
                    self._building_bytes -= charge_bytes
                    self._drained.notify_all()

    def _remove_locked(self, cache_key: str, retired: List[_HashCacheValue]) -> int:
        value = self._entries.pop(cache_key, None)
        if value is not None:
            self._resident_bytes -= value.charge_bytes
            self._retired_bytes += value.charge_bytes
            value.retired = True
            if not value.readers:
                retired.append(value)
                return value.charge_bytes
        return 0

    def _release_retired(self, values: List[_HashCacheValue]) -> None:
        for value in values:
            for owner in value.owners:
                owner.release()
            with self._lock:
                self._retired_bytes -= value.charge_bytes
                self._drained.notify_all()

    def _snapshot(self, cache_key, generation=None, *, touch=False):
        with self._lock:
            value = self._entries.get(cache_key)
            if value is None or (
                generation is not None and value.generation != generation
            ):
                return None
            if touch:
                self._entries.move_to_end(cache_key)
            value.readers += 1
            approval = value.greennet_passed
        try:
            # Same copy as before, now protected by a lease rather than holding
            # the global index lock. Pool views still never escape this cache.
            return (
                [tensor.clone() for tensor in value.hashes],
                value.generation,
                approval,
            )
        finally:
            with self._lock:
                value.readers -= 1
                retired = value.retired and value.readers == 0
            if retired:
                self._release_retired([value])

    def get(
        self, cache_key: str, generation: Optional[str] = None
    ) -> Optional[List[torch.Tensor]]:
        value = self._snapshot(cache_key, generation, touch=True)
        return None if value is None else value[0]

    def greennet_passed(self, cache_key: str) -> bool:
        """An approval survives embedding eviction, but never hash eviction."""
        with self._lock:
            value = self._entries.get(cache_key)
            return value is not None and value.greennet_passed

    def mark_greennet_passed(self, cache_key: str, generation: str) -> bool:
        """Attach a fresh GreenNet approval to the matching hash-cache entry."""
        with self._lock:
            value = self._entries.get(cache_key)
            if value is None or value.generation != generation:
                return False
            value.greennet_passed = True
            return True

    def contains(self, cache_key: str) -> bool:
        with self._lock:
            return cache_key in self._entries

    def keys(self, limit: Optional[int] = None) -> List[str]:
        with self._lock:
            if limit is None:
                return list(self._entries.keys())
            # Keep directory responses bounded without a count-based cache
            # eviction limit or copying every resident key into a snapshot.
            return list(islice(reversed(self._entries), limit))[::-1]

    def metadata_keys(self) -> List[str]:
        """Compatibility name used by the ViT cache HTTP endpoint."""
        return self.keys()

    def clear(self) -> None:
        retired = []
        with self._lock:
            self._pending.clear()
            for key in list(self._entries):
                self._remove_locked(key, retired)
        self._release_retired(retired)

    def resize(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("hash cache max_bytes must be non-negative")
        retired = []
        with self._drained:
            self._pending.clear()
            self._max_bytes = max_bytes
            reclaimable = 0
            while self._entries and (
                self._used_bytes_locked() - reclaimable > max_bytes
            ):
                reclaimable += self._remove_locked(next(iter(self._entries)), retired)
        self._release_retired(retired)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "resident_entries": len(self._entries),
                "resident_bytes": self._resident_bytes,
                "building_bytes": self._building_bytes,
                "retired_bytes": self._retired_bytes,
                "max_bytes": self._max_bytes,
            }

    def metadata(
        self,
        keys: List[str],
        embedding_cache: "MMEmbeddingCache",
        *,
        binary_hashes: bool = False,
    ) -> Dict[str, Any]:
        results = []
        total_rows = 0
        tiers = embedding_cache.resident_tiers(keys)
        for key in keys:
            value = self._snapshot(key)
            hash_hit = (
                value is not None
                and bool(value[0])
                and all(tensor.numel() > 0 for tensor in value[0])
            )
            result = {
                "key": key,
                "hit": hash_hit,
                "hash_hit": hash_hit,
                "greennet_passed": bool(hash_hit and value[2]),
                "embedding_hit": key in tiers,
                "embedding_tier": tiers.get(key),
            }
            results.append(result)
            if not hash_hit:
                continue
            hashes = value[0]
            split_size = [hash_tensor.numel() for hash_tensor in hashes]
            total_rows += sum(split_size)
            if total_rows > 1048576:
                raise ValueError("multimodal metadata response exceeds row limit")
            result.update(
                {
                    "split_size": split_size,
                    "feature_hashes": (
                        b"".join(
                            tensor.numpy().astype("<i4", copy=False).tobytes()
                            for tensor in hashes
                        )
                        if binary_hashes
                        else [
                            value
                            for hash_tensor in hashes
                            for value in hash_tensor.tolist()
                        ]
                    ),
                    "entry_generation": value[1],
                }
            )
        return {
            "worker_instance": self.instance_id,
            "entries": results,
        }


class _CacheLifetimeMetrics:
    """Estimate cache turnover from admissions, with O(1) shared state.

    Start at the first admission, then emit on the first admission at least
    60s after the window start. Use the actual elapsed time, including idle
    gaps; no timer or per-entry history is needed. No ingress means no sample.
    """

    def __init__(self, capacity_bytes: int):
        self._lock = threading.Lock()
        self._capacity_bytes = capacity_bytes
        self._window_start: Optional[float] = None
        self._admitted_bytes = 0

    def resize(self, capacity_bytes: int) -> None:
        with self._lock:
            self._capacity_bytes = capacity_bytes
            self._window_start = None
            self._admitted_bytes = 0

    def record_admission(self, size_bytes: int) -> None:
        if size_bytes <= 0:
            return
        with self._lock:
            if self._capacity_bytes <= 0:
                return
            now = time.monotonic()
            self._admitted_bytes += size_bytes
            if self._window_start is None:
                self._window_start = now
                return
            elapsed = now - self._window_start
            if elapsed < 60.0:
                return
            lifetime_seconds = self._capacity_bytes * elapsed / self._admitted_bytes
            self._window_start = now
            self._admitted_bytes = 0
        kmonitor.report(
            GaugeMetrics.VIT_EMBEDDING_CACHE_ESTIMATED_LIFETIME_SECONDS_METRIC,
            lifetime_seconds,
        )


class MMEmbeddingCache:
    """Two-tier fixed-pool LRU shared by sync and async embedding paths.

    GPU victims spill to CPU; CPU victims leave the cache. Reads restore each
    tensor's original device and promote when it fits. Transfers are serialized
    separately from the index lock: hot reads and metadata probes never wait for
    another entry's copy. Active callers keep their own tensor references.
    """

    def __init__(
        self,
        gpu_max_bytes: int,
        cpu_max_bytes: int,
        report_metrics: bool = False,
        on_cpu_evict: Optional[Callable[[str, MMEmbeddingCacheEntry], None]] = None,
        async_publication: bool = False,
        publish_queue_size: int = 1024,
        lifetime_metrics: Optional[_CacheLifetimeMetrics] = None,
        on_gpu_evict: Optional[Callable[[str, MMEmbeddingCacheEntry], None]] = None,
    ):
        self._on_cpu_evict = on_cpu_evict
        self._on_gpu_evict = on_gpu_evict
        self._validate_limits(gpu_max_bytes, cpu_max_bytes)
        self._lock = threading.Lock()
        # All residency mutations take this lock before _lock. Copies never
        # hold _lock, and only one cache transfer can allocate staging at a time.
        self._transfer_lock = threading.Lock()
        self._entries: "OrderedDict[str, MMEmbeddingCacheEntry]" = OrderedDict()
        self._gpu_lru: "OrderedDict[str, MMEmbeddingCacheEntry]" = OrderedDict()
        self._cpu_lru: "OrderedDict[str, MMEmbeddingCacheEntry]" = OrderedDict()
        self._gpu_max_bytes = gpu_max_bytes
        self._cpu_max_bytes = cpu_max_bytes
        gpu_device = (
            torch.device("cuda", torch.cuda.current_device())
            if gpu_max_bytes > 0
            else torch.device("cuda")
        )
        self._gpu_pool = _TensorMemoryPool(gpu_max_bytes, gpu_device)
        self._cpu_pool = _TensorMemoryPool(cpu_max_bytes, torch.device("cpu"))
        self._report_metrics_enabled = report_metrics
        self._owns_lifetime_metrics = lifetime_metrics is None
        self._lifetime_metrics = lifetime_metrics or (
            _CacheLifetimeMetrics(gpu_max_bytes + cpu_max_bytes)
            if report_metrics
            else None
        )
        self._resident_tokens = 0
        self._gpu_bytes = 0
        self._cpu_bytes = 0
        self.instance_id = uuid.uuid4().hex
        self._stats: Dict[str, int] = {
            "hit": 0,
            "gpu_hit": 0,
            "cpu_hit": 0,
            "miss": 0,
            "inflight_dedup": 0,
            "eviction": 0,
            "demotion": 0,
            "promotion": 0,
            "transfer_error": 0,
            "publish_drop": 0,
        }
        self._publish_queue: Optional["queue.Queue"] = None
        self._publish_thread: Optional[threading.Thread] = None
        self._publish_pending = 0
        self._publish_cond = threading.Condition()
        if async_publication:
            self._publish_queue = queue.Queue(maxsize=max(1, publish_queue_size))
            self._publish_thread = threading.Thread(
                target=self._publish_worker,
                name="mm-cache-publish",
                daemon=True,
            )
            self._publish_thread.start()

    @staticmethod
    def _validate_limits(gpu_max_bytes: int, cpu_max_bytes: int) -> None:
        if gpu_max_bytes < 0 or cpu_max_bytes < 0:
            raise ValueError("embedding cache byte limits must be non-negative")

    @property
    def enabled(self) -> bool:
        return self._gpu_max_bytes > 0 or self._cpu_max_bytes > 0

    def _new_entry(self, cache_key: str) -> MMEmbeddingCacheEntry:
        return MMEmbeddingCacheEntry(
            on_complete=lambda entry, result: self._on_complete(
                cache_key, entry, result
            ),
            on_fail=lambda entry, error: self._on_fail(cache_key, entry, error),
            on_read=lambda entry, timeout: self._read_entry(cache_key, entry, timeout),
        )

    def try_acquire(self, cache_key: str) -> Tuple[str, MMEmbeddingCacheEntry]:
        with self._lock:
            if not self.enabled:
                self._stats["miss"] += 1
                state, entry = "miss", MMEmbeddingCacheEntry()
            elif cache_key in self._entries:
                entry = self._entries[cache_key]
                self._entries.move_to_end(cache_key)
                if entry.is_done:
                    self._stats["hit"] += 1
                    if entry.tier is not None:
                        self._stats[entry.tier + "_hit"] += 1
                        lru = self._gpu_lru if entry.tier == "gpu" else self._cpu_lru
                        lru.move_to_end(cache_key)
                    state = "complete"
                else:
                    self._stats["inflight_dedup"] += 1
                    state = "in_progress"
            else:
                entry = self._new_entry(cache_key)
                self._entries[cache_key] = entry
                self._stats["miss"] += 1
                state = "miss"
        self._report_access_metric(state)
        return state, entry

    def try_acquire_lazy(
        self, key_factory: Callable[[], str]
    ) -> Tuple[str, str, MMEmbeddingCacheEntry]:
        """Avoid constructing an unused input key when residency is disabled.

        The disabled branch still creates a distinct, unregistered entry and
        reports the same miss as try_acquire. Its empty key is never published
        into the index, including if a concurrent resize enables the cache.
        Async request identity must continue to use an eagerly computed key.
        """
        with self._lock:
            disabled = not self.enabled
            if disabled:
                self._stats["miss"] += 1
                entry = MMEmbeddingCacheEntry()
        if disabled:
            self._report_access_metric("miss")
            return "", "miss", entry
        # Do not serialize a potentially large URL hash under the cache lock.
        # try_acquire rechecks capacity, preserving concurrent resize semantics.
        key = key_factory()
        state, entry = self.try_acquire(key)
        return key, state, entry

    @staticmethod
    def _record_ready_events(result: Any) -> List[Any]:
        return _record_cuda_events(result)

    @staticmethod
    @cuda_graph_gate.operation()
    def _ready_result(result: Any, events: List[Any]) -> Any:
        # The producer may use a different thread/stream. Wait for its writes
        # before a D2H copy or exposing the value to the request's consumer.
        for event in events:
            event.synchronize()

        def record(tensor: torch.Tensor) -> None:
            if tensor.device.type == "cuda":
                tensor.record_stream(torch.cuda.current_stream(tensor.device))

        _map_tensors(result, record)
        return result

    def _uncharge_locked(self, cache_key: str, entry: MMEmbeddingCacheEntry) -> None:
        self._gpu_lru.pop(cache_key, None)
        self._cpu_lru.pop(cache_key, None)
        self._gpu_bytes -= entry.charge_gpu_bytes
        self._cpu_bytes -= entry.charge_cpu_bytes
        self._resident_tokens -= entry.charge_tokens
        entry.charge_gpu_bytes = entry.charge_cpu_bytes = entry.charge_bytes = 0
        entry.charge_tokens = 0
        entry.tier = None

    def _detach_pool_storage(
        self, entry: MMEmbeddingCacheEntry, *, to_cpu: bool = False
    ) -> None:
        """Move an evicted entry out of the pool for an outstanding waiter."""

        with entry.storage_lock:
            owners = list(entry.pool_owners)
            if not owners:
                return
            try:
                self._ready_result(entry.result, entry.ready_events)
                # Stage GPU-only writeback directly out of the retiring pool.
                # Outstanding readers still restore entry.original_devices.
                devices = _map_tensors(
                    entry.result, lambda t: torch.device("cpu") if to_cpu else t.device
                )
                detached = _copy_pooled_result_to_devices(entry.result, devices)
                events = self._record_ready_events(detached)
                for owner in owners:
                    owner.add_events(events)
                entry.result = detached
                entry.ready_events = events
                entry.pool_owners = []
            except Exception:
                # Keeping the owner is safe; it only delays reuse of this pool
                # block until the outstanding entry reference is released.
                logging.warning(
                    "Failed to detach evicted ViT cache entry from pool",
                    exc_info=True,
                )
                return
        for owner in owners:
            owner.release()

    def _unlink_entry_locked(self, cache_key: str, eviction: bool = False):
        """Index-only removal. Caller owns transfer lock through finalization."""
        entry = self._entries.pop(cache_key)
        tier = entry.tier
        self._uncharge_locked(cache_key, entry)
        if eviction:
            self._stats["eviction"] += 1
        return cache_key, entry, tier, eviction

    def _finish_removal(self, removal) -> None:
        cache_key, entry, tier, eviction = removal
        callback = None
        if eviction and entry.is_done:
            if tier == "cpu":
                callback = self._on_cpu_evict
            elif tier == "gpu" and self._cpu_max_bytes == 0:
                callback = self._on_gpu_evict
        # Copies, event waits, pool releases and remote admission must not
        # block unrelated index probes or embedding -> hash publication.
        self._detach_pool_storage(entry, to_cpu=tier == "gpu" and callback is not None)
        if callback is not None:
            try:
                callback(cache_key, entry)
            except Exception:
                logging.warning(
                    "Skipping failed remote cache eviction admission", exc_info=True
                )
        if eviction:
            if self._report_metrics_enabled:
                kmonitor.report(AccMetrics.VIT_EMBEDDING_CACHE_EVICTION_QPS_METRIC, 1)

    def _remove_entry(self, cache_key, expected=None, *, eviction=False) -> bool:
        """Caller owns _transfer_lock, never the index lock."""
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None or (expected is not None and entry is not expected):
                return False
            removal = self._unlink_entry_locked(cache_key, eviction)
        self._finish_removal(removal)
        return True

    def _admit_locked(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        gpu_bytes: int,
        cpu_bytes: int,
        charge_tokens: int,
    ) -> None:
        entry.charge_tokens = charge_tokens
        entry.charge_gpu_bytes = gpu_bytes
        entry.charge_cpu_bytes = cpu_bytes
        entry.charge_bytes = gpu_bytes + cpu_bytes
        entry.tier = "gpu" if gpu_bytes else "cpu"
        self._gpu_bytes += gpu_bytes
        self._cpu_bytes += cpu_bytes
        self._resident_tokens += entry.charge_tokens
        lru = self._gpu_lru if gpu_bytes else self._cpu_lru
        lru[cache_key] = entry
        self._entries.move_to_end(cache_key)

    def _make_cpu_room(self, required_bytes: int) -> None:
        while True:
            with self._lock:
                if self._cpu_bytes + required_bytes <= self._cpu_max_bytes:
                    return
                # CPU tensors attached to hot GPU entries also count toward the
                # CPU budget. Drop a hot entry only after all cold victims.
                lru = self._cpu_lru or self._gpu_lru
                removal = self._unlink_entry_locked(next(iter(lru)), eviction=True)
            self._finish_removal(removal)

    def _make_cpu_pool_room(self, required_bytes: int) -> None:
        while required_bytes:
            stats = self._cpu_pool.stats()
            if (
                stats["largest_free_block_bytes"] >= required_bytes
                or stats["pending_bytes"] > 0
            ):
                return
            with self._lock:
                lru = self._cpu_lru or self._gpu_lru
                if not lru:
                    return
                removal = self._unlink_entry_locked(next(iter(lru)), eviction=True)
            self._finish_removal(removal)

    def _make_gpu_pool_room(self, required_bytes: int) -> None:
        while required_bytes:
            stats = self._gpu_pool.stats()
            if (
                stats["largest_free_block_bytes"] >= required_bytes
                or stats["pending_bytes"] > 0
            ):
                return
            if not self._evict_gpu_lru():
                return

    def _store_pooled(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        result: Any,
        devices: Any,
        offloaded: bool,
        source_events: List[Any],
    ) -> bool:
        """Pack and admit an uncharged entry; caller owns _transfer_lock."""
        required_gpu, required_cpu = _compact_tensor_tier_bytes(result, devices)
        if (
            required_gpu > self._gpu_max_bytes
            or required_cpu > self._cpu_max_bytes
            or (required_gpu == 0 and required_cpu == 0)
        ):
            self._remove_entry(cache_key, entry, eviction=True)
            return False
        if required_gpu:
            self._make_gpu_room(required_gpu)
        if required_cpu:
            self._make_cpu_room(required_cpu)
        if required_gpu:
            self._make_gpu_pool_room(required_gpu)
        if required_cpu:
            self._make_cpu_pool_room(required_cpu)
        try:
            self._ready_result(result, source_events)
            for attempt in range(2):
                try:
                    (
                        pooled_result,
                        owners,
                        events,
                        gpu_bytes,
                        cpu_bytes,
                    ) = _pool_result(result, devices, self._gpu_pool, self._cpu_pool)
                    break
                except _PoolCapacityError:
                    if attempt:
                        raise
                    # reserve() has drained completed/pending releases. If the
                    # remaining failure is fragmentation, evict more LRU blocks
                    # and retry once against the coalesced free list.
                    if required_gpu:
                        self._make_gpu_pool_room(required_gpu)
                    if required_cpu:
                        self._make_cpu_pool_room(required_cpu)
        except _PoolCapacityError:
            logging.warning(
                "ViT cache pool has no reusable block for %s; bypassing cache",
                cache_key,
            )
            with self._lock:
                if self._entries.get(cache_key) is entry:
                    self._stats["transfer_error"] += 1
            self._remove_entry(cache_key, entry, eviction=True)
            return False
        except Exception:
            # Cache insertion is optional. Preserve the successful computation
            # for existing waiters when a pool copy fails.
            logging.warning(
                "ViT cache pool copy failed for %s", cache_key, exc_info=True
            )
            with self._lock:
                if self._entries.get(cache_key) is entry:
                    self._stats["transfer_error"] += 1
            self._remove_entry(cache_key, entry, eviction=True)
            return False
        with self._lock:
            indexed = self._entries.get(cache_key) is entry
        if not indexed:
            for owner in owners:
                owner.release()
            return False
        # Residency mutation is serialized by _transfer_lock. A request may
        # hold storage_lock while cloning; do not wait for it under _lock.
        with entry.storage_lock:
            old_owners = entry.pool_owners
            for owner in old_owners:
                owner.add_events(events)
            entry.result = pooled_result
            entry.offloaded = offloaded
            entry.ready_events = events
            entry.pool_owners = owners
        for owner in old_owners:
            owner.release()
        with self._lock:
            self._admit_locked(
                cache_key,
                entry,
                gpu_bytes,
                cpu_bytes,
                _embedding_output_tokens(result),
            )
        return True

    def _store_cpu(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        result: Any,
        events: List[Any],
    ) -> None:
        devices = _map_tensors(result, lambda _: torch.device("cpu"))
        if self._store_pooled(cache_key, entry, result, devices, True, events):
            with self._lock:
                self._stats["demotion"] += 1

    def _make_gpu_room(self, required_bytes: int) -> None:
        # The caller holds _transfer_lock across all changes to residency.
        while self._gpu_bytes + required_bytes > self._gpu_max_bytes:
            self._evict_gpu_lru()

    def _evict_gpu_lru(self) -> bool:
        with self._lock:
            if not self._gpu_lru:
                return False
            key, entry = next(iter(self._gpu_lru.items()))
            if self._cpu_max_bytes == 0:
                removal = self._unlink_entry_locked(key, eviction=True)
            else:
                self._uncharge_locked(key, entry)
                removal = None
        if removal is not None:
            self._finish_removal(removal)
        else:
            self._store_cpu(key, entry, entry.result, entry.ready_events)
        return True

    def _on_complete(
        self, cache_key: str, entry: MMEmbeddingCacheEntry, result: Any
    ) -> None:
        if self._publish_queue is not None:
            self._enqueue_publication(
                cache_key, entry, result, entry._pending_before_publish
            )
            return
        self._on_complete_sync(cache_key, entry, result)

    def _drop_pending_entry(self, cache_key: str, entry: MMEmbeddingCacheEntry) -> None:
        """Remove an unqueued PENDING entry without taking the transfer lock."""
        with self._lock:
            if self._entries.get(cache_key) is entry:
                self._entries.pop(cache_key, None)

    def _enqueue_publication(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        result: Any,
        before_publish: Optional[Callable[[], None]],
    ) -> None:
        with self._publish_cond:
            if self._publish_pending >= self._publish_queue.maxsize:
                drop = True
            else:
                self._publish_pending += 1
                drop = False
        if drop:
            with self._lock:
                self._stats["publish_drop"] += 1
            logging.warning("ViT cache publication queue full; dropping %s", cache_key)
            self._drop_pending_entry(cache_key, entry)
            return
        try:
            self._publish_queue.put_nowait((cache_key, entry, result, before_publish))
        except queue.Full:
            with self._publish_cond:
                self._publish_pending -= 1
                self._publish_cond.notify_all()
            with self._lock:
                self._stats["publish_drop"] += 1
            logging.warning("ViT cache publication queue full; dropping %s", cache_key)
            self._drop_pending_entry(cache_key, entry)

    def _publish_worker(self) -> None:
        while True:
            item = self._publish_queue.get()
            if item is None:
                break
            cache_key, entry, result, before_publish = item
            try:
                self._on_complete_sync(cache_key, entry, result)
                if before_publish is not None:
                    before_publish()
            except Exception:
                logging.warning(
                    "ViT cache publication failed for %s", cache_key, exc_info=True
                )
                self._remove_if_same(cache_key, entry)
            finally:
                with self._publish_cond:
                    self._publish_pending -= 1
                    self._publish_cond.notify_all()

    def flush(self) -> None:
        """Wait for queued cache publications to finish (tests/shutdown)."""
        if self._publish_queue is None:
            return
        with self._publish_cond:
            while self._publish_pending:
                self._publish_cond.wait()

    def close(self) -> None:
        """Stop the publication thread after draining queued work."""
        if self._publish_queue is None:
            return
        self._publish_queue.put(None)
        if self._publish_thread is not None:
            self._publish_thread.join(timeout=10)
        self._publish_queue = None
        self._publish_thread = None

    def _on_complete_sync(
        self, cache_key: str, entry: MMEmbeddingCacheEntry, result: Any
    ) -> None:
        entry.original_devices = _map_tensors(result, lambda tensor: tensor.device)
        entry.ready_events = self._record_ready_events(result)
        gpu_bytes, cpu_bytes = _compact_tensor_tier_bytes(
            result, entry.original_devices
        )
        with self._transfer_lock:
            with self._lock:
                if self._entries.get(cache_key) is not entry:
                    return
            if not self.enabled:
                self._remove_entry(cache_key, entry)
                return
            if gpu_bytes > self._gpu_max_bytes:
                self._store_cpu(cache_key, entry, result, entry.ready_events)
            elif cpu_bytes > self._cpu_max_bytes or (
                gpu_bytes == 0 and self._cpu_max_bytes == 0
            ):
                self._remove_entry(cache_key, entry)
            else:
                self._store_pooled(
                    cache_key,
                    entry,
                    result,
                    entry.original_devices,
                    False,
                    entry.ready_events,
                )
            # Completion runs once per generation, unlike _store_pooled which
            # also handles tier moves. Snapshot before releasing the transfer
            # lock: a concurrent eviction must not erase this admission.
            admitted_bytes = entry.charge_bytes
        self._report_admission(admitted_bytes)
        self._report_current_residency()

    def _materialize_entry(self, entry: MMEmbeddingCacheEntry, devices: Any) -> Any:
        with entry.storage_lock:
            result = entry.result
            events = entry.ready_events
            owners = list(entry.pool_owners)
            if devices is None:
                devices = _map_tensors(result, lambda tensor: tensor.device)
            if not owners:
                self._ready_result(result, events)
                entry.ready_events = []
                current_devices = _map_tensors(result, lambda tensor: tensor.device)
                return (
                    _copy_result_to_devices(result, devices)
                    if (current_devices != devices)
                    else result
                )
            self._ready_result(result, events)
            entry.ready_events = []
            copied = _copy_pooled_result_to_devices(result, devices)
            read_events = self._record_ready_events(copied)
            for event in read_events:
                with cuda_graph_gate.operation():
                    event.synchronize()
            return copied

    def _read_entry(
        self, cache_key: str, entry: MMEmbeddingCacheEntry, timeout: Optional[float]
    ) -> Any:
        with entry.storage_lock:
            offloaded = entry.offloaded
        if not offloaded:
            return self._materialize_entry(entry, entry.original_devices)
        # A CPU hit is promoted only on consumption, never on a routing probe.
        acquired = self._transfer_lock.acquire(
            timeout=-1 if timeout is None else timeout
        )
        if not acquired:
            raise TimeoutError("Waiting for embedding cache transfer timed out")
        try:
            with entry.storage_lock:
                offloaded = entry.offloaded
            with self._lock:
                indexed = self._entries.get(cache_key) is entry
            if not offloaded:
                return self._materialize_entry(entry, entry.original_devices)
            with entry.storage_lock:
                result = entry.result
            gpu_bytes, cpu_bytes = _compact_tensor_tier_bytes(
                result, entry.original_devices
            )
            retain = (
                indexed
                and self._gpu_max_bytes > 0
                and gpu_bytes <= self._gpu_max_bytes
                and cpu_bytes <= self._cpu_max_bytes
            )
            try:
                restored = self._materialize_entry(entry, entry.original_devices)
            except Exception:
                with self._lock:
                    self._stats["transfer_error"] += 1
                raise
            if retain:
                # The restored tensor is now an independent request-owned copy.
                # Drop the old CPU-pool reservation before making room for GPU
                # promotion; a GPU victim may need that same CPU slot.
                with entry.storage_lock:
                    old_owners = entry.pool_owners
                    restored_events = self._record_ready_events(restored)
                    for owner in old_owners:
                        owner.add_events(restored_events)
                    entry.result = restored
                    entry.ready_events = restored_events
                    entry.pool_owners = []
                for owner in old_owners:
                    owner.release()
                with self._lock:
                    self._uncharge_locked(cache_key, entry)
                promoted = self._store_pooled(
                    cache_key,
                    entry,
                    restored,
                    entry.original_devices,
                    False,
                    self._record_ready_events(restored),
                )
                if promoted:
                    with self._lock:
                        self._stats["promotion"] += 1
                else:
                    with entry.storage_lock:
                        entry.offloaded = False
            return restored
        finally:
            self._transfer_lock.release()
            self._report_current_residency()

    def _on_fail(
        self, cache_key: str, entry: MMEmbeddingCacheEntry, error: Exception
    ) -> None:
        self._remove_if_same(cache_key, entry)

    def _remove_if_same(self, cache_key: str, expected: MMEmbeddingCacheEntry) -> bool:
        with self._transfer_lock:
            if not self._remove_entry(cache_key, expected):
                return False
        self._report_current_residency()
        return True

    def complete(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        result: Any,
        *,
        before_publish: Optional[Callable[[], None]] = None,
    ) -> bool:
        # Prepare reader-visible metadata BEFORE the entry publishes its event.
        # The deferred (async publication) path wakes waiters before the
        # publication thread runs _on_complete_sync; without this a waiter can
        # observe original_devices=None and crash in _copy_result_to_devices.
        entry.original_devices = _map_tensors(result, lambda tensor: tensor.device)
        entry.ready_events = self._record_ready_events(result)
        return entry.complete(
            result,
            before_publish=before_publish,
            defer=self._publish_queue is not None,
        )

    def fail(
        self, cache_key: str, entry: MMEmbeddingCacheEntry, error: Exception
    ) -> bool:
        return entry.fail(error)

    def remove(self, cache_key: str) -> None:
        with self._lock:
            entry = self._entries.get(cache_key)
        if entry is not None:
            self._remove_if_same(cache_key, entry)

    def peek(self, cache_key: str) -> Optional[MMEmbeddingCacheEntry]:
        with self._lock:
            return self._entries.get(cache_key)

    def _with_current_entry(self, cache_key, expected, action):
        """Validate and publish sidecar metadata atomically with index removal.

        action may take the hash index lock, but must not copy tensors, touch
        pools, or call back into this cache. Never acquire these in reverse.
        """
        with self._lock:
            if self._entries.get(cache_key) is expected and expected.error is None:
                return action()
        return None

    def resident_tiers(
        self, keys: Optional[List[str]] = None, limit: Optional[int] = None
    ) -> Dict[str, str]:
        """Snapshot completed residency without touching LRU or moving tensors."""
        if limit is not None and limit <= 0:
            return {}
        with self._lock:
            candidates = (
                keys
                if keys is not None
                else chain(reversed(self._gpu_lru), reversed(self._cpu_lru))
            )
            tiers = {}
            for key in candidates:
                entry = self._entries.get(key)
                if (
                    entry is not None
                    and entry._event.is_set()
                    and entry.error is None
                    and entry.tier in ("gpu", "cpu")
                ):
                    tiers[key] = entry.tier
                    if limit is not None and len(tiers) >= limit:
                        break
            return tiers

    def resize(self, gpu_max_bytes: int, cpu_max_bytes: int) -> None:
        self._validate_limits(gpu_max_bytes, cpu_max_bytes)
        with self._transfer_lock:
            grow_gpu = gpu_max_bytes > self._gpu_pool.capacity_bytes
            grow_cpu = cpu_max_bytes > self._cpu_pool.capacity_bytes
            if grow_gpu or grow_cpu:
                # Pool sizes are fixed. Runtime growth is rare and cannot move
                # live pool views safely, so discard indexed values first.
                with self._lock:
                    removals = [
                        self._unlink_entry_locked(key, eviction=True)
                        for key in list(self._entries)
                    ]
                for removal in removals:
                    self._finish_removal(removal)
                if grow_gpu:
                    self._gpu_pool = _TensorMemoryPool(
                        gpu_max_bytes,
                        torch.device("cuda", torch.cuda.current_device()),
                    )
                if grow_cpu:
                    self._cpu_pool = _TensorMemoryPool(
                        cpu_max_bytes, torch.device("cpu")
                    )
            with self._lock:
                self._gpu_max_bytes = gpu_max_bytes
                self._cpu_max_bytes = cpu_max_bytes
            self._make_gpu_room(0)
            self._make_cpu_room(0)
            with self._lock:
                removals = (
                    [
                        self._unlink_entry_locked(key, eviction=True)
                        for key in list(self._cpu_lru)
                    ]
                    if cpu_max_bytes == 0
                    else []
                )
            for removal in removals:
                self._finish_removal(removal)
        if self._owns_lifetime_metrics and self._lifetime_metrics is not None:
            self._lifetime_metrics.resize(gpu_max_bytes + cpu_max_bytes)
        self._report_current_residency()

    def clear(self, error: Optional[Exception] = None) -> None:
        with self._transfer_lock:
            with self._lock:
                entries = list(self._entries.values())
                removals = [
                    self._unlink_entry_locked(key) for key in list(self._entries)
                ]
            for removal in removals:
                self._finish_removal(removal)
        if error is not None:
            for entry in entries:
                if not entry.is_done:
                    entry.fail(error)
        self._report_current_residency()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            stats = {
                **self._stats,
                "resident_entries": len(self._gpu_lru) + len(self._cpu_lru),
                "resident_tokens": self._resident_tokens,
                "resident_bytes": self._gpu_bytes + self._cpu_bytes,
                "gpu_resident_entries": len(self._gpu_lru),
                "cpu_resident_entries": len(self._cpu_lru),
                "gpu_resident_bytes": self._gpu_bytes,
                "cpu_resident_bytes": self._cpu_bytes,
                "gpu_max_bytes": self._gpu_max_bytes,
                "cpu_max_bytes": self._cpu_max_bytes,
                "pending_entries": sum(
                    1 for entry in self._entries.values() if not entry.is_done
                ),
            }
        gpu_pool = self._gpu_pool.stats()
        cpu_pool = self._cpu_pool.stats()
        stats.update({f"gpu_pool_{key}": value for key, value in gpu_pool.items()})
        stats.update({f"cpu_pool_{key}": value for key, value in cpu_pool.items()})
        return stats

    def _report_current_residency(self) -> None:
        with self._lock:
            tokens = self._resident_tokens
            size_bytes = self._gpu_bytes + self._cpu_bytes
        self._report_resident_metrics(tokens, size_bytes)

    def _report_admission(self, size_bytes: int) -> None:
        if not self._report_metrics_enabled or self._lifetime_metrics is None:
            return
        try:
            self._lifetime_metrics.record_admission(size_bytes)
        except Exception:
            logging.exception("Failed to report ViT cache estimated lifetime")

    def _report_access_metric(self, state: str) -> None:
        if not self._report_metrics_enabled:
            return
        metric = {
            "miss": AccMetrics.VIT_EMBEDDING_CACHE_MISS_QPS_METRIC,
            "complete": AccMetrics.VIT_EMBEDDING_CACHE_HIT_QPS_METRIC,
            "in_progress": AccMetrics.VIT_EMBEDDING_CACHE_INFLIGHT_QPS_METRIC,
        }[state]
        kmonitor.report(metric, 1)

    def _report_resident_metrics(self, tokens: int, size_bytes: int) -> None:
        if not self._report_metrics_enabled:
            return
        kmonitor.report(GaugeMetrics.VIT_EMBEDDING_CACHE_TOKENS_METRIC, tokens)
        kmonitor.report(GaugeMetrics.VIT_EMBEDDING_CACHE_BYTES_METRIC, size_bytes)


def _split_budget(total: int, shards: int) -> List[int]:
    """Split a byte budget into `shards` parts whose sum is exactly `total`."""
    base, remainder = divmod(total, shards)
    return [base + (1 if index < remainder else 0) for index in range(shards)]


class ShardedMMEmbeddingCache:
    """Route embedding-cache keys across independent cache shards.

    Each shard owns its own index lock, transfer lock, LRU and fixed pools.
    Sharding bounds both the critical section and the first-fit free-list scan
    by the shard's entry count instead of the whole cache.
    """

    def __init__(
        self,
        gpu_max_bytes: int,
        cpu_max_bytes: int,
        *,
        shards: int = 8,
        report_metrics: bool = False,
        on_cpu_evict: Optional[Callable[[str, MMEmbeddingCacheEntry], None]] = None,
        async_publication: bool = False,
        on_gpu_evict: Optional[Callable[[str, MMEmbeddingCacheEntry], None]] = None,
    ):
        if shards < 1:
            raise ValueError("embedding cache shards must be positive")
        self._lifetime_metrics = (
            _CacheLifetimeMetrics(gpu_max_bytes + cpu_max_bytes)
            if report_metrics
            else None
        )
        self._shards = [
            MMEmbeddingCache(
                gpu_bytes,
                cpu_bytes,
                report_metrics=report_metrics,
                on_cpu_evict=on_cpu_evict,
                on_gpu_evict=on_gpu_evict,
                async_publication=async_publication,
                lifetime_metrics=self._lifetime_metrics,
            )
            for gpu_bytes, cpu_bytes in zip(
                _split_budget(gpu_max_bytes, shards),
                _split_budget(cpu_max_bytes, shards),
            )
        ]
        self.instance_id = uuid.uuid4().hex

    @property
    def enabled(self) -> bool:
        return any(shard.enabled for shard in self._shards)

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    def _shard(self, cache_key: str) -> MMEmbeddingCache:
        return self._shards[hash(cache_key) % len(self._shards)]

    def try_acquire(self, cache_key: str) -> Tuple[str, MMEmbeddingCacheEntry]:
        return self._shard(cache_key).try_acquire(cache_key)

    def try_acquire_lazy(
        self, key_factory: Callable[[], str]
    ) -> Tuple[str, str, MMEmbeddingCacheEntry]:
        if not self.enabled:
            return "", "miss", MMEmbeddingCacheEntry()
        key = key_factory()
        state, entry = self._shard(key).try_acquire(key)
        return key, state, entry

    def complete(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        result: Any,
        *,
        before_publish: Optional[Callable[[], None]] = None,
    ) -> bool:
        return self._shard(cache_key).complete(
            cache_key, entry, result, before_publish=before_publish
        )

    def fail(
        self, cache_key: str, entry: MMEmbeddingCacheEntry, error: Exception
    ) -> bool:
        return self._shard(cache_key).fail(cache_key, entry, error)

    def remove(self, cache_key: str) -> None:
        self._shard(cache_key).remove(cache_key)

    def peek(self, cache_key: str) -> Optional[MMEmbeddingCacheEntry]:
        return self._shard(cache_key).peek(cache_key)

    def _with_current_entry(self, cache_key, expected, action):
        return self._shard(cache_key)._with_current_entry(cache_key, expected, action)

    def resident_tiers(
        self, keys: Optional[List[str]] = None, limit: Optional[int] = None
    ) -> Dict[str, str]:
        if limit is not None and limit <= 0:
            return {}
        if keys is not None:
            tiers: Dict[str, str] = {}
            for key in keys:
                tier = self._shard(key).resident_tiers([key]).get(key)
                if tier is not None:
                    tiers[key] = tier
                    if limit is not None and len(tiers) >= limit:
                        break
            return tiers
        tiers = {}
        for shard in self._shards:
            tiers.update(shard.resident_tiers(limit=limit))
            if limit is not None and len(tiers) >= limit:
                break
        return tiers

    def clear(self, error: Optional[Exception] = None) -> None:
        for shard in self._shards:
            shard.clear(error)

    def flush(self) -> None:
        for shard in self._shards:
            shard.flush()

    def close(self) -> None:
        for shard in self._shards:
            shard.close()

    def resize(self, gpu_max_bytes: int, cpu_max_bytes: int) -> None:
        MMEmbeddingCache._validate_limits(gpu_max_bytes, cpu_max_bytes)
        for shard, gpu_bytes, cpu_bytes in zip(
            self._shards,
            _split_budget(gpu_max_bytes, len(self._shards)),
            _split_budget(cpu_max_bytes, len(self._shards)),
        ):
            shard.resize(gpu_bytes, cpu_bytes)
        if self._lifetime_metrics is not None:
            self._lifetime_metrics.resize(gpu_max_bytes + cpu_max_bytes)

    def stats(self) -> Dict[str, int]:
        per_shard = [shard.stats() for shard in self._shards]
        stats: Dict[str, int] = {}
        for shard_stats in per_shard:
            for key, value in shard_stats.items():
                if key.endswith("largest_free_block_bytes"):
                    stats[key] = max(stats.get(key, 0), value)
                elif isinstance(value, (int, float)):
                    stats[key] = stats.get(key, 0) + value
                else:
                    stats.setdefault(key, value)
        stats["shards"] = len(self._shards)
        return stats


# Compatibility alias for internal callers that imported the previous class.
MMEmbeddingAsyncCache = MMEmbeddingCache
