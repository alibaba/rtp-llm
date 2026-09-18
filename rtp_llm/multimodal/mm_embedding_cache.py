from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
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
    __slots__ = ("offset", "size", "requested_bytes")

    def __init__(self, offset: int, size: int, requested_bytes: int):
        self.offset = offset
        self.size = size
        self.requested_bytes = requested_bytes


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

    @staticmethod
    def _align(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    def _insert_free_locked(self, offset: int, size: int) -> None:
        if size == 0:
            return
        self._free.append((offset, size))
        self._free.sort()
        merged = []
        for current_offset, current_size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == current_offset:
                previous_offset, previous_size = merged[-1]
                merged[-1] = (previous_offset, previous_size + current_size)
            else:
                merged.append((current_offset, current_size))
        self._free = merged

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
                    block = _PoolBlock(aligned_offset, size, requested_bytes)
                    self._free.pop(index)
                    self._insert_free_locked(offset, prefix_size)
                    suffix_offset = aligned_offset + size
                    suffix_size = offset + free_size - suffix_offset
                    self._insert_free_locked(suffix_offset, suffix_size)
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
                "free_bytes": self.capacity_bytes - self._used_bytes,
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

    collect(result, devices)
    reservations: Dict[_TensorMemoryPool, _PoolReservation] = {}
    offsets: Dict[Tuple[int, torch.device], int] = {}
    copied: Dict[Tuple[int, torch.device], torch.Tensor] = {}
    gpu_bytes = cpu_bytes = 0
    try:
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
        response = pooled, owners, events, gpu_bytes, cpu_bytes
        # The recursive local collector owns a closure cycle. Do not let that
        # cycle retain producer tensors until a later cyclic-GC pass.
        specs.clear()
        return response
    except Exception:
        specs.clear()
        for reservation in reservations.values():
            reservation.release()
        raise


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
            return tuple(visit(item, target[i]) for i, item in enumerate(value))
        if isinstance(value, list):
            return [visit(item, target[i]) for i, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: visit(item, target[key]) for key, item in value.items()}
        return value

    return visit(result, devices)


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


def _tensor_tier_bytes(result: Any, devices: Any = None) -> Tuple[int, int]:
    """Actual retained (GPU, CPU) bytes, or compact transfer destination bytes."""
    sizes = [0, 0]
    seen = set()

    def visit(value: Any, target: Any) -> None:
        if isinstance(value, torch.Tensor):
            device = value.device if target is None else target
            if device == value.device:
                storage = value.untyped_storage()
                identity = (device, storage.data_ptr())
                size = storage.nbytes()
            else:
                # Transfers compact views, preserving repeated tensor objects.
                identity = (device, id(value))
                size = value.numel() * value.element_size()
            if identity not in seen:
                seen.add(identity)
                sizes[device.type == "cpu"] += size
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(item, target[i] if target is not None else None)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, target[key] if target is not None else None)

    visit(result, devices)
    return tuple(sizes)


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
            return tuple(visit(item, target[i]) for i, item in enumerate(value))
        if isinstance(value, list):
            return [visit(item, target[i]) for i, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: visit(item, target[key]) for key, item in value.items()}
        return value

    return visit(result, devices)


def _embedding_result_cost(result: Any) -> Tuple[int, int]:
    """Return ``(output_tokens, tensor_bytes)`` retained by one cache value."""

    seen_storages = set()

    def tensor_bytes(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            # A slice keeps its entire backing allocation alive. Counting its
            # logical numel would undercharge batched outputs retained as views.
            storage = value.untyped_storage()
            storage_id = (value.device, storage.data_ptr())
            if storage_id in seen_storages:
                return 0
            seen_storages.add(storage_id)
            return storage.nbytes()
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(item) for item in value)
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        return 0

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
    return output_tokens(primary), tensor_bytes(result)


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
        self._greennet_event = threading.Event()
        self._greennet_verdict: Optional[GreenNetVerdict] = None
        self._greennet_checked = False

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
        self, result: Any, feature_hashes: Optional[List[torch.Tensor]] = None
    ) -> bool:
        with self._state_lock:
            if self._terminal:
                return False
            self._terminal = True
            self.result = result
        try:
            if self._on_complete is not None:
                self._on_complete(self, result)
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

    def ready_metadata(self) -> Optional[List[torch.Tensor]]:
        # Feature hashes are owned by MMHashKeyCache. Keep this compatibility
        # method so older internal callers treat an embedding entry as having
        # no inline metadata.
        return None

    def set_greennet_verdict(
        self, verdict: GreenNetVerdict, *, checked: bool = True
    ) -> None:
        self._greennet_verdict = verdict
        self._greennet_checked = checked
        self._greennet_event.set()

    @property
    def greennet_passed(self) -> bool:
        return (
            self.is_greennet_decided
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
        return self._greennet_verdict

    @property
    def is_greennet_decided(self) -> bool:
        return self._greennet_event.is_set()


class MMHashKeyCache:
    """Byte-bounded CPU-pool cache for multimodal feature-hash token ids.

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
        self._entries = OrderedDict()
        self._max_bytes = max_bytes
        self._resident_bytes = 0
        self._pool = _TensorMemoryPool(max_bytes, torch.device("cpu"))
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
        if not cache_key or not self.enabled:
            return
        # Reject oversized values before making CPU copies. Replacing a key
        # must also discard its previous generation, even when the new value
        # cannot fit. Do not evict unrelated entries for such a value.
        tensor_bytes = sum(h.numel() * 4 for h in feature_hashes)
        pool_bytes = tensor_bytes
        if pool_bytes > self._max_bytes:
            with self._lock:
                self._remove_locked(cache_key)
            return
        # Hashes are small CPU int32 token-id tensors. Store an owned CPU copy
        # so the sidecar does not retain embedding storage or a GPU tensor.
        with cuda_graph_gate.operation():
            hashes = [
                hash_tensor.detach().to(device="cpu", dtype=torch.int32).clone()
                for hash_tensor in feature_hashes
            ]
        generation = generation or ""
        charge_bytes = (
            pool_bytes
            + sys.getsizeof(cache_key)
            + sys.getsizeof(generation)
            + sys.getsizeof(hashes)
            + sum(sys.getsizeof(h) for h in hashes)
            + sys.getsizeof((hashes, generation, 0, [], False))
            + sys.getsizeof(0)
            + self._ENTRY_OVERHEAD
        )
        with self._lock:
            self._remove_locked(cache_key)
            if charge_bytes > self._max_bytes:
                return
            # Evict before insertion so resident accounting never exceeds the
            # budget, including when a key is replaced by a larger value.
            while self._resident_bytes + charge_bytes > self._max_bytes:
                self._remove_locked(next(iter(self._entries)))
            devices = _map_tensors(hashes, lambda _: torch.device("cpu"))
            try:
                pooled, owners, _, _, _ = _pool_result(
                    hashes, devices, None, self._pool
                )
            except _PoolCapacityError:
                logging.warning(
                    "Hash-key cache CPU pool is full; bypassing key %s", cache_key
                )
                return
            self._entries[cache_key] = (
                pooled,
                generation,
                charge_bytes,
                owners,
                bool(greennet_passed),
            )
            self._resident_bytes += charge_bytes

    def _remove_locked(self, cache_key: str) -> None:
        value = self._entries.pop(cache_key, None)
        if value is not None:
            self._resident_bytes -= value[2]
            for owner in value[3]:
                owner.release()

    def get(
        self, cache_key: str, generation: Optional[str] = None
    ) -> Optional[List[torch.Tensor]]:
        with self._lock:
            value = self._entries.get(cache_key)
            if value is None or (generation is not None and value[1] != generation):
                return None
            self._entries.move_to_end(cache_key)
            # Pool views never escape the cache; otherwise eviction could reuse
            # their storage while an active request is still reading it.
            return [tensor.clone() for tensor in value[0]]

    def greennet_passed(self, cache_key: str) -> bool:
        """An approval survives embedding eviction, but never hash eviction."""
        with self._lock:
            value = self._entries.get(cache_key)
            return value is not None and value[4]

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
        with self._lock:
            for key in list(self._entries):
                self._remove_locked(key)

    def resize(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("hash cache max_bytes must be non-negative")
        with self._lock:
            if max_bytes > self._pool.capacity_bytes:
                for key in list(self._entries):
                    self._remove_locked(key)
                self._pool = _TensorMemoryPool(max_bytes, torch.device("cpu"))
            self._max_bytes = max_bytes
            while self._resident_bytes > max_bytes:
                self._remove_locked(next(iter(self._entries)))

    def stats(self) -> Dict[str, int]:
        with self._lock:
            stats = {
                "resident_entries": len(self._entries),
                "resident_bytes": self._resident_bytes,
                "max_bytes": self._max_bytes,
            }
            pool_stats = self._pool.stats()
            stats.update({f"pool_{key}": value for key, value in pool_stats.items()})
            return stats

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
            with self._lock:
                value = self._entries.get(key)
                if value is not None:
                    value = (
                        [tensor.clone() for tensor in value[0]],
                        value[1],
                        value[2],
                        [],
                        value[4],
                    )
            hash_hit = (
                value is not None
                and bool(value[0])
                and all(tensor.numel() > 0 for tensor in value[0])
            )
            result = {
                "key": key,
                "hit": hash_hit,
                "hash_hit": hash_hit,
                "greennet_passed": bool(hash_hit and value[4]),
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
            "feature_hash_version": 1,
            "entries": results,
        }


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
    ):
        self._on_cpu_evict = on_cpu_evict
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
        }

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

    def _detach_pool_storage(self, entry: MMEmbeddingCacheEntry) -> None:
        """Move an evicted entry out of the pool for an outstanding waiter."""

        with entry.storage_lock:
            owners = list(entry.pool_owners)
            if not owners:
                return
            try:
                self._ready_result(entry.result, entry.ready_events)
                current_devices = _map_tensors(
                    entry.result, lambda tensor: tensor.device
                )
                detached = _copy_pooled_result_to_devices(entry.result, current_devices)
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

    def _remove_entry_locked(self, cache_key: str, eviction: bool = False) -> None:
        entry = self._entries.pop(cache_key)
        was_cpu = entry.tier == "cpu"
        self._uncharge_locked(cache_key, entry)
        self._detach_pool_storage(entry)
        if eviction and was_cpu and self._on_cpu_evict is not None and entry.is_done:
            try:
                self._on_cpu_evict(cache_key, entry)
            except Exception:
                logging.warning(
                    "Skipping failed remote cache eviction admission", exc_info=True
                )
        if eviction:
            self._stats["eviction"] += 1
            if self._report_metrics_enabled:
                kmonitor.report(AccMetrics.VIT_EMBEDDING_CACHE_EVICTION_QPS_METRIC, 1)

    def _admit_locked(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        result: Any,
        offloaded: bool,
        events: List[Any],
        owners: List[_PoolReservation],
        gpu_bytes: int,
        cpu_bytes: int,
        charge_tokens: int,
    ) -> None:
        with entry.storage_lock:
            old_owners = entry.pool_owners
            # New pool copies may still be reading the old pool blocks.
            for owner in old_owners:
                owner.add_events(events)
            entry.result = result
            entry.offloaded = offloaded
            entry.ready_events = events
            entry.pool_owners = owners
        for owner in old_owners:
            owner.release()
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
        with self._lock:
            while self._cpu_bytes + required_bytes > self._cpu_max_bytes:
                # CPU tensors attached to hot GPU entries also count toward the
                # CPU budget. Drop a hot entry only after all cold victims.
                lru = self._cpu_lru or self._gpu_lru
                self._remove_entry_locked(next(iter(lru)), eviction=True)

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
                self._remove_entry_locked(next(iter(lru)), eviction=True)

    def _make_gpu_pool_room(self, required_bytes: int) -> None:
        while required_bytes:
            stats = self._gpu_pool.stats()
            if (
                stats["largest_free_block_bytes"] >= required_bytes
                or stats["pending_bytes"] > 0
            ):
                return
            with self._lock:
                if not self._gpu_lru:
                    return
                cache_key, entry = next(iter(self._gpu_lru.items()))
                result, events = entry.result, entry.ready_events
                self._uncharge_locked(cache_key, entry)
            self._store_cpu(cache_key, entry, result, events)

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
            with self._lock:
                if self._entries.get(cache_key) is entry:
                    self._remove_entry_locked(cache_key, eviction=True)
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
                    self._remove_entry_locked(cache_key, eviction=True)
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
                    self._remove_entry_locked(cache_key, eviction=True)
            return False
        with self._lock:
            if self._entries.get(cache_key) is not entry:
                for owner in owners:
                    owner.release()
                return False
            self._admit_locked(
                cache_key,
                entry,
                pooled_result,
                offloaded,
                events,
                owners,
                gpu_bytes,
                cpu_bytes,
                _embedding_result_cost(result)[0],
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
        while True:
            with self._lock:
                if self._gpu_bytes + required_bytes <= self._gpu_max_bytes:
                    return
                cache_key, entry = next(iter(self._gpu_lru.items()))
                result, events = entry.result, entry.ready_events
                self._uncharge_locked(cache_key, entry)
            self._store_cpu(cache_key, entry, result, events)
            # Do not retain the previous victim's GPU allocation during the
            # next transfer (unless a caller independently holds that value).
            del result

    def _on_complete(
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
                    self._remove_entry_locked(cache_key)
                    return
            if gpu_bytes > self._gpu_max_bytes:
                self._store_cpu(cache_key, entry, result, entry.ready_events)
            elif cpu_bytes > self._cpu_max_bytes or (
                gpu_bytes == 0 and self._cpu_max_bytes == 0
            ):
                with self._lock:
                    self._remove_entry_locked(cache_key)
            else:
                self._store_pooled(
                    cache_key,
                    entry,
                    result,
                    entry.original_devices,
                    False,
                    entry.ready_events,
                )
        self._report_current_residency()

    def _materialize_entry(self, entry: MMEmbeddingCacheEntry, devices: Any) -> Any:
        with entry.storage_lock:
            result = entry.result
            events = entry.ready_events
            owners = list(entry.pool_owners)
            if not owners:
                self._ready_result(result, events)
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
        with self._transfer_lock, self._lock:
            if self._entries.get(cache_key) is not expected:
                return False
            self._remove_entry_locked(cache_key)
        self._report_current_residency()
        return True

    def complete(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        result: Any,
        feature_hashes: Optional[List[torch.Tensor]] = None,
    ) -> bool:
        return entry.complete(result, feature_hashes)

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

    def metadata_keys(self) -> List[str]:
        with self._lock:
            entries = list(self._entries.items())
        return [key for key, entry in entries if entry.ready_metadata() is not None]

    def metadata(self, keys: List[str]) -> Dict[str, Any]:
        results = []
        total_rows = 0
        for key in keys:
            entry = self.peek(key)
            hashes = entry.ready_metadata() if entry is not None else None
            if hashes is None:
                results.append({"key": key, "hit": False})
                continue
            split_size = [h.numel() for h in hashes]
            total_rows += sum(split_size)
            if total_rows > 1048576:
                raise ValueError("multimodal metadata response exceeds row limit")
            results.append(
                {
                    "key": key,
                    "hit": True,
                    "split_size": split_size,
                    "feature_hashes": [v for h in hashes for v in h.tolist()],
                    "entry_generation": entry.generation,
                }
            )
        return {
            "worker_instance": self.instance_id,
            "feature_hash_version": 1,
            "entries": results,
        }

    def resize(self, gpu_max_bytes: int, cpu_max_bytes: int) -> None:
        self._validate_limits(gpu_max_bytes, cpu_max_bytes)
        with self._transfer_lock:
            grow_gpu = gpu_max_bytes > self._gpu_pool.capacity_bytes
            grow_cpu = cpu_max_bytes > self._cpu_pool.capacity_bytes
            if grow_gpu or grow_cpu:
                # Pool sizes are fixed. Runtime growth is rare and cannot move
                # live pool views safely, so discard indexed values first.
                with self._lock:
                    for key in list(self._entries):
                        self._remove_entry_locked(key, eviction=True)
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
                if cpu_max_bytes == 0:
                    for key in list(self._cpu_lru):
                        self._remove_entry_locked(key, eviction=True)
        self._report_current_residency()

    def clear(self, error: Optional[Exception] = None) -> None:
        with self._transfer_lock, self._lock:
            entries = list(self._entries.values())
            for key in list(self._entries):
                self._remove_entry_locked(key)
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


# Compatibility alias for internal callers that imported the previous class.
MMEmbeddingAsyncCache = MMEmbeddingCache
