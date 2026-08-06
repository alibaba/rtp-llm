from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict


def _embedding_result_cost(result: Any) -> Tuple[int, int]:
    """Return ``(output_tokens, tensor_bytes)`` retained by one cache value."""

    seen_tensors = set()

    def tensor_bytes(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            if tensor_id in seen_tensors:
                return 0
            seen_tensors.add(tensor_id)
            return value.numel() * value.element_size()
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
            return value.numel()
        if isinstance(value, (list, tuple)):
            return sum(output_tokens(item) for item in value)
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
    ):
        self._event = threading.Event()
        self._state_lock = threading.Lock()
        self._terminal = False
        self._on_complete = on_complete
        self._on_fail = on_fail
        self.result: Optional[Any] = None
        self.error: Optional[Exception] = None
        self.charge_tokens = 0
        self.charge_bytes = 0
        self._greennet_event = threading.Event()
        self._greennet_verdict: Optional[GreenNetVerdict] = None

    def wait(self, timeout: Optional[float] = None) -> Any:
        if not self._event.wait(timeout=timeout):
            raise TimeoutError("Waiting for embedding result timed out")
        if self.error is not None:
            raise self.error
        return self.result

    def complete(self, result: Any) -> bool:
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


class MMEmbeddingCache:
    """Concurrent weighted LRU shared by sync and async embedding paths.

    ``max_size`` preserves the old entry-count behavior for models that cannot
    expose an output-token budget. When ``max_bytes`` is available, completed
    entries are charged by their actual retained tensor bytes and the entry
    count is no longer the eviction limit. PENDING entries may temporarily
    exceed either limit so concurrent misses can still be deduplicated.
    """

    def __init__(
        self,
        max_size: int = 10,
        max_bytes: Optional[int] = None,
        report_metrics: bool = False,
    ):
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, MMEmbeddingCacheEntry]" = OrderedDict()
        self._max_size = max_size
        self._max_bytes = max_bytes
        self._report_metrics_enabled = report_metrics
        self._resident_tokens = 0
        self._resident_bytes = 0
        self._stats: Dict[str, int] = {
            "hit": 0,
            "miss": 0,
            "inflight_dedup": 0,
            "eviction": 0,
        }

    @property
    def enabled(self) -> bool:
        return self._max_size > 0 and (self._max_bytes is None or self._max_bytes > 0)

    def _new_entry(self, cache_key: str) -> MMEmbeddingCacheEntry:
        return MMEmbeddingCacheEntry(
            on_complete=lambda entry, result: self._on_complete(
                cache_key, entry, result
            ),
            on_fail=lambda entry, error: self._on_fail(cache_key, entry, error),
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
                    state = "complete"
                else:
                    self._stats["inflight_dedup"] += 1
                    state = "in_progress"
            else:
                entry = self._new_entry(cache_key)
                self._entries[cache_key] = entry
                self._stats["miss"] += 1
                self._evict_locked()
                state = "miss"
        self._report_access_metric(state)
        return state, entry

    def _on_complete(
        self, cache_key: str, entry: MMEmbeddingCacheEntry, result: Any
    ) -> None:
        charge_tokens, charge_bytes = _embedding_result_cost(result)
        with self._lock:
            if self._entries.get(cache_key) is not entry:
                return
            entry.charge_tokens = charge_tokens
            entry.charge_bytes = charge_bytes
            self._resident_tokens += charge_tokens
            self._resident_bytes += charge_bytes
            self._entries.move_to_end(cache_key)
            self._evict_locked()
            resident_tokens = self._resident_tokens
            resident_bytes = self._resident_bytes
        self._report_resident_metrics(resident_tokens, resident_bytes)

    def _on_fail(
        self, cache_key: str, entry: MMEmbeddingCacheEntry, error: Exception
    ) -> None:
        self._remove_if_same(cache_key, entry)

    def _remove_entry_locked(self, cache_key: str) -> None:
        entry = self._entries.pop(cache_key)
        self._resident_tokens -= entry.charge_tokens
        self._resident_bytes -= entry.charge_bytes

    def _evict_locked(self) -> None:
        while True:
            if self._max_bytes is not None:
                over_limit = self._resident_bytes > self._max_bytes
            else:
                over_limit = len(self._entries) > self._max_size
            if not over_limit:
                return

            evict_key = next(
                (key for key, entry in self._entries.items() if entry.is_done),
                None,
            )
            if evict_key is None:
                return
            self._remove_entry_locked(evict_key)
            self._stats["eviction"] += 1
            if self._report_metrics_enabled:
                kmonitor.report(AccMetrics.VIT_EMBEDDING_CACHE_EVICTION_QPS_METRIC, 1)

    def _remove_if_same(self, cache_key: str, expected: MMEmbeddingCacheEntry) -> bool:
        with self._lock:
            if self._entries.get(cache_key) is not expected:
                return False
            self._remove_entry_locked(cache_key)
            resident_tokens = self._resident_tokens
            resident_bytes = self._resident_bytes
        self._report_resident_metrics(resident_tokens, resident_bytes)
        return True

    def complete(
        self, cache_key: str, entry: MMEmbeddingCacheEntry, result: Any
    ) -> bool:
        return entry.complete(result)

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

    def resize(self, max_size: int, max_bytes: Optional[int] = None) -> None:
        with self._lock:
            self._max_size = max_size
            self._max_bytes = max_bytes
            self._evict_locked()
            resident_tokens = self._resident_tokens
            resident_bytes = self._resident_bytes
        self._report_resident_metrics(resident_tokens, resident_bytes)

    def clear(self, error: Optional[Exception] = None) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            self._resident_tokens = 0
            self._resident_bytes = 0
        if error is not None:
            for entry in entries:
                if not entry.is_done:
                    entry.fail(error)
        self._report_resident_metrics(0, 0)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                **self._stats,
                "resident_entries": sum(
                    1 for entry in self._entries.values() if entry.is_done
                ),
                "resident_tokens": self._resident_tokens,
                "resident_bytes": self._resident_bytes,
                "pending_entries": sum(
                    1 for entry in self._entries.values() if not entry.is_done
                ),
            }

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
