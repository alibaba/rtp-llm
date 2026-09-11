from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from collections import OrderedDict
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict


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
        self._greennet_event = threading.Event()
        self._greennet_verdict: Optional[GreenNetVerdict] = None

    def claim_error_report(self) -> bool:
        """Claim the single error-telemetry sample for this cache entry."""
        with self._state_lock:
            if self._error_reported:
                return False
            self._error_reported = True
            return True

    def wait(self, timeout: Optional[float] = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        if not self._event.wait(timeout=timeout):
            raise TimeoutError("Waiting for embedding result timed out")
        if self.error is not None:
            raise self.error
        if self._on_read is not None:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            return self._on_read(self, remaining)
        return self.result

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
        return self._terminal

    def ready_metadata(self) -> Optional[List[torch.Tensor]]:
        # Feature hashes are owned by MMHashKeyCache. Keep this compatibility
        # method so older internal callers treat an embedding entry as having
        # no inline metadata.
        return None

    def set_greennet_verdict(self, verdict: GreenNetVerdict) -> None:
        self._greennet_verdict = verdict
        self._greennet_event.set()

    def wait_greennet(self, timeout: Optional[float] = None) -> GreenNetVerdict:
        if not self._greennet_event.wait(timeout=timeout):
            raise TimeoutError("Waiting for greennet verdict timed out")
        return self._greennet_verdict

    @property
    def is_greennet_decided(self) -> bool:
        return self._greennet_event.is_set()


class MMHashKeyCache:
    """Byte-bounded CPU cache for multimodal keys and feature-hash token ids.

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
        self._entries: "OrderedDict[str, Tuple[List[torch.Tensor], str, int]]" = (
            OrderedDict()
        )
        self._max_bytes = max_bytes
        self._resident_bytes = 0
        self.instance_id = uuid.uuid4().hex

    @property
    def enabled(self) -> bool:
        return self._max_bytes > 0

    def put(
        self,
        cache_key: str,
        feature_hashes: List[torch.Tensor],
        generation: Optional[str] = None,
    ) -> None:
        if not cache_key or not self.enabled:
            return
        # Reject oversized values before making CPU copies. Replacing a key
        # must also discard its previous generation, even when the new value
        # cannot fit. Do not evict unrelated entries for such a value.
        tensor_bytes = sum(h.numel() * 4 for h in feature_hashes)
        if tensor_bytes > self._max_bytes:
            with self._lock:
                self._remove_locked(cache_key)
            return
        # Hashes are small CPU int32 token-id tensors. Store an owned CPU copy
        # so the sidecar does not retain embedding storage or a GPU tensor.
        hashes = [
            hash_tensor.detach().to(device="cpu", dtype=torch.int32).clone()
            for hash_tensor in feature_hashes
        ]
        generation = generation or ""
        charge_bytes = (
            tensor_bytes
            + sys.getsizeof(cache_key)
            + sys.getsizeof(generation)
            + sys.getsizeof(hashes)
            + sum(sys.getsizeof(h) for h in hashes)
            + sys.getsizeof((hashes, generation, 0))
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
            self._entries[cache_key] = (hashes, generation, charge_bytes)
            self._resident_bytes += charge_bytes

    def _remove_locked(self, cache_key: str) -> None:
        value = self._entries.pop(cache_key, None)
        if value is not None:
            self._resident_bytes -= value[2]

    def get(
        self, cache_key: str, generation: Optional[str] = None
    ) -> Optional[List[torch.Tensor]]:
        with self._lock:
            value = self._entries.get(cache_key)
            if value is None or (generation is not None and value[1] != generation):
                return None
            self._entries.move_to_end(cache_key)
            return value[0]

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
            self._entries.clear()
            self._resident_bytes = 0

    def resize(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("hash cache max_bytes must be non-negative")
        with self._lock:
            self._max_bytes = max_bytes
            while self._resident_bytes > max_bytes:
                self._remove_locked(next(iter(self._entries)))

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "resident_entries": len(self._entries),
                "resident_bytes": self._resident_bytes,
                "max_bytes": self._max_bytes,
            }

    def metadata(
        self, keys: List[str], embedding_cache: "MMEmbeddingCache"
    ) -> Dict[str, Any]:
        results = []
        total_rows = 0
        for key in keys:
            entry = embedding_cache.peek(key)
            with self._lock:
                value = self._entries.get(key)
            if (
                entry is None
                or not entry.is_done
                or entry.error is not None
                or value is None
                or value[1] != entry.generation
            ):
                results.append({"key": key, "hit": False})
                continue
            hashes = value[0]
            split_size = [hash_tensor.numel() for hash_tensor in hashes]
            total_rows += sum(split_size)
            if total_rows > 1048576:
                raise ValueError("multimodal metadata response exceeds row limit")
            results.append(
                {
                    "key": key,
                    "hit": True,
                    "split_size": split_size,
                    "feature_hashes": [
                        value
                        for hash_tensor in hashes
                        for value in hash_tensor.tolist()
                    ],
                    "entry_generation": entry.generation,
                }
            )
        return {
            "worker_instance": self.instance_id,
            "feature_hash_version": 1,
            "entries": results,
        }


class MMEmbeddingCache:
    """Two-tier byte-bounded LRU shared by sync and async embedding paths.

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
    ):
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
        devices = set()
        _map_tensors(result, lambda tensor: devices.add(tensor.device))
        events = []
        for device in devices:
            if device.type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                events.append(event)
        return events

    @staticmethod
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

    def _remove_entry_locked(self, cache_key: str, eviction: bool = False) -> None:
        entry = self._entries.pop(cache_key)
        self._uncharge_locked(cache_key, entry)
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
    ) -> None:
        gpu_bytes, cpu_bytes = _tensor_tier_bytes(result)
        entry.result = result
        entry.offloaded = offloaded
        entry.ready_events = events
        entry.charge_tokens = _embedding_result_cost(result)[0]
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

    def _store_cpu(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        result: Any,
        events: List[Any],
    ) -> None:
        """Store an uncharged entry; caller owns _transfer_lock."""
        devices = _map_tensors(result, lambda _: torch.device("cpu"))
        _, required_bytes = _tensor_tier_bytes(result, devices)
        if self._cpu_max_bytes == 0 or required_bytes > self._cpu_max_bytes:
            with self._lock:
                self._remove_entry_locked(cache_key, eviction=True)
            return
        self._make_cpu_room(required_bytes)
        try:
            self._ready_result(result, events)
            cpu_result = _copy_result_to_devices(result, devices)
        except Exception:
            # Cache insertion is optional. Preserve the successful computation
            # for existing waiters when host allocation/copy fails.
            logging.warning(
                "ViT cache CPU spill failed for %s", cache_key, exc_info=True
            )
            with self._lock:
                self._stats["transfer_error"] += 1
                self._remove_entry_locked(cache_key, eviction=True)
            return
        with self._lock:
            self._admit_locked(cache_key, entry, cpu_result, True, [])
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
        gpu_bytes, cpu_bytes = _tensor_tier_bytes(result)
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
                self._make_gpu_room(gpu_bytes)
                self._make_cpu_room(cpu_bytes)
                with self._lock:
                    self._admit_locked(
                        cache_key, entry, result, False, entry.ready_events
                    )
        self._report_current_residency()

    def _read_entry(
        self, cache_key: str, entry: MMEmbeddingCacheEntry, timeout: Optional[float]
    ) -> Any:
        with self._lock:
            result, events, offloaded = (
                entry.result,
                entry.ready_events,
                entry.offloaded,
            )
        if not offloaded:
            return self._ready_result(result, events)
        # A CPU hit is promoted only on consumption, never on a routing probe.
        acquired = self._transfer_lock.acquire(
            timeout=-1 if timeout is None else timeout
        )
        if not acquired:
            raise TimeoutError("Waiting for embedding cache transfer timed out")
        try:
            with self._lock:
                result, events = entry.result, entry.ready_events
                offloaded = entry.offloaded
                indexed = self._entries.get(cache_key) is entry
            if not offloaded:
                return self._ready_result(result, events)
            gpu_bytes, cpu_bytes = _tensor_tier_bytes(result, entry.original_devices)
            retain = (
                indexed
                and self._gpu_max_bytes > 0
                and gpu_bytes <= self._gpu_max_bytes
                and cpu_bytes <= self._cpu_max_bytes
            )
            if retain:
                with self._lock:
                    self._uncharge_locked(cache_key, entry)
                self._make_gpu_room(gpu_bytes)
                self._make_cpu_room(cpu_bytes)
            try:
                restored = _copy_result_to_devices(result, entry.original_devices)
            except Exception:
                with self._lock:
                    self._stats["transfer_error"] += 1
                if retain:
                    # Keep the CPU value and generation available for retry.
                    self._make_cpu_room(_tensor_tier_bytes(result)[1])
                    with self._lock:
                        self._admit_locked(cache_key, entry, result, True, [])
                raise
            if retain:
                with self._lock:
                    self._admit_locked(cache_key, entry, restored, False, [])
                    self._stats["promotion"] += 1
            return self._ready_result(restored, [])
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
            return {
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
