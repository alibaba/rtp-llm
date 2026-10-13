import logging
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, Optional


DEFAULT_TIME_WINDOW_MS = 30 * 60 * 1000
CACHE_HIT_TIME_WINDOW_MS_ENV = "CACHE_HIT_TIME_WINDOW_MS"


@dataclass(frozen=True)
class RecentCacheKeySnapshot:
    time_window_ms: int
    request_occurrences: int
    request_hit_occurrences: int
    request_hit_ratio: float
    retained_occurrences: int
    retained_unique_cache_keys: int


class RecentCacheKeyWindow:
    def __init__(
        self,
        time_window_ms: Optional[int] = None,
        now_ms_supplier: Optional[Callable[[], int]] = None,
    ) -> None:
        if time_window_ms is None:
            time_window_ms = self._resolve_env_window_ms()
        self.time_window_ms = self._normalize_time_window_ms(time_window_ms)
        self._now_ms_supplier = now_ms_supplier or (lambda: int(time.time() * 1000))
        self._window_entries: Deque[tuple[int, Dict[int, int]]] = deque()
        self._cache_key_counts: Counter[int] = Counter()
        self._retained_occurrences = 0

    def record(self, cache_keys: Optional[Iterable[int]]) -> RecentCacheKeySnapshot:
        now_ms = self._now_ms_supplier()
        self._evict_expired(now_ms)

        if not cache_keys:
            return self._snapshot(0, 0)

        entry_counts: Counter[int] = Counter()
        request_occurrences = 0
        request_hit_occurrences = 0
        for cache_key in cache_keys:
            if cache_key is None:
                continue
            request_occurrences += 1
            if cache_key in self._cache_key_counts:
                request_hit_occurrences += 1
            entry_counts[int(cache_key)] += 1

        if not entry_counts:
            return self._snapshot(0, 0)

        self._window_entries.append((now_ms, dict(entry_counts)))
        self._cache_key_counts.update(entry_counts)
        self._retained_occurrences += sum(entry_counts.values())
        return self._snapshot(request_occurrences, request_hit_occurrences)

    def snapshot(self) -> RecentCacheKeySnapshot:
        self._evict_expired(self._now_ms_supplier())
        return self._snapshot(0, 0)

    def _evict_expired(self, now_ms: int) -> None:
        expire_before_or_at = now_ms - self.time_window_ms
        while self._window_entries:
            timestamp_ms, entry_counts = self._window_entries[0]
            if timestamp_ms > expire_before_or_at:
                return
            self._window_entries.popleft()
            for cache_key, expired_count in entry_counts.items():
                current_count = self._cache_key_counts.get(cache_key, 0)
                if current_count <= expired_count:
                    self._cache_key_counts.pop(cache_key, None)
                    self._retained_occurrences -= current_count
                else:
                    self._cache_key_counts[cache_key] = current_count - expired_count
                    self._retained_occurrences -= expired_count

    def _snapshot(
        self, request_occurrences: int, request_hit_occurrences: int
    ) -> RecentCacheKeySnapshot:
        hit_ratio = (
            request_hit_occurrences / request_occurrences
            if request_occurrences > 0
            else 0.0
        )
        return RecentCacheKeySnapshot(
            time_window_ms=self.time_window_ms,
            request_occurrences=request_occurrences,
            request_hit_occurrences=request_hit_occurrences,
            request_hit_ratio=hit_ratio,
            retained_occurrences=self._retained_occurrences,
            retained_unique_cache_keys=len(self._cache_key_counts),
        )

    @staticmethod
    def _resolve_env_window_ms() -> int:
        value = os.environ.get(CACHE_HIT_TIME_WINDOW_MS_ENV)
        if value is None:
            return DEFAULT_TIME_WINDOW_MS
        try:
            return int(value)
        except ValueError:
            logging.warning(
                "Invalid %s=%s, fallback to default: %s",
                CACHE_HIT_TIME_WINDOW_MS_ENV,
                value,
                DEFAULT_TIME_WINDOW_MS,
            )
            return DEFAULT_TIME_WINDOW_MS

    @staticmethod
    def _normalize_time_window_ms(candidate_ms: int) -> int:
        if candidate_ms > 0:
            return candidate_ms
        logging.warning(
            "Invalid cache hit time window: %s, fallback to default: %s",
            candidate_ms,
            DEFAULT_TIME_WINDOW_MS,
        )
        return DEFAULT_TIME_WINDOW_MS
