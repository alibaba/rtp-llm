from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheSnapshot:
    """One complete cache-key view collected from all configured RTP workers."""

    keys: frozenset[int]
    block_size: int
    version: int


@dataclass(frozen=True)
class CacheDiff:
    """KVCM mutations required to converge from the acknowledged baseline."""

    added: tuple[int, ...]
    removed: tuple[int, ...]

    @property
    def empty(self) -> bool:
        return not self.added and not self.removed


class CacheDiffTracker:
    """Track the KVCM-acknowledged key set and debounce removals.

    ``plan`` observes a full RTP snapshot but never mutates the acknowledged
    baseline. The caller must invoke ``commit`` only after every KVCM request
    for the returned diff succeeds. This makes a failed report retryable.
    """

    def __init__(self, deletion_confirmations: int = 2) -> None:
        if deletion_confirmations < 1:
            raise ValueError("deletion_confirmations must be >= 1")
        self._deletion_confirmations = deletion_confirmations
        self._acknowledged: set[int] = set()
        self._possibly_reported: set[int] = set()
        self._missing_counts: dict[int, int] = {}

    @property
    def acknowledged_keys(self) -> frozenset[int]:
        return frozenset(self._acknowledged)

    def plan(self, observed: frozenset[int], *, force_full_add: bool = False) -> CacheDiff:
        for key in observed:
            self._missing_counts.pop(key, None)

        deletion_domain = self._acknowledged | self._possibly_reported
        for key in deletion_domain - observed:
            self._missing_counts[key] = self._missing_counts.get(key, 0) + 1

        added = observed if force_full_add else observed - self._acknowledged
        removed = {
            key
            for key, count in self._missing_counts.items()
            if key in deletion_domain and count >= self._deletion_confirmations
        }
        return CacheDiff(tuple(sorted(added)), tuple(sorted(removed)))

    def mark_uncertain(self, diff: CacheDiff) -> None:
        """Remember adds that may have succeeded before a batch-group failure."""

        self._possibly_reported.update(set(diff.added) - self._acknowledged)

    def commit(self, diff: CacheDiff) -> None:
        self._acknowledged.update(diff.added)
        self._acknowledged.difference_update(diff.removed)
        self._possibly_reported.difference_update(diff.added)
        self._possibly_reported.difference_update(diff.removed)
        for key in diff.removed:
            self._missing_counts.pop(key, None)

    def reset(self) -> None:
        self._acknowledged.clear()
        self._possibly_reported.clear()
        self._missing_counts.clear()
