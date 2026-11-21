"""Python request orchestration for the native local EMB resident cache."""

import logging
import weakref

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

from .mm_embedding_cache import (
    MMEmbeddingCache,
    MMEmbeddingCacheEntry,
    _current_cuda_streams,
    _embedding_output_tokens,
    _SIGNALED_EVENT,
)
from .greennet_hook import GreenNetVerdict


class _NativeRequestEntry(MMEmbeddingCacheEntry):
    """A pending producer or transient request handle for a native generation."""

    def __init__(self, generation, cell, on_complete, on_fail, on_read, ready=False):
        super().__init__(on_complete, on_fail, on_read, generation=generation)
        self._native_cell = cell
        self._native_published = ready
        if ready:
            self._terminal = True
            self._event = _SIGNALED_EVENT

    @property
    def remote_embedding_origin(self):
        origin = self._native_cell.audit()[-1]
        if not origin:
            raise AttributeError("embedding origin has not been recorded")
        return origin

    @remote_embedding_origin.setter
    def remote_embedding_origin(self, origin):
        self._native_cell.set_origin(origin)

    def set_greennet_verdict(self, verdict, *, checked=True):
        cell = self._native_cell
        cell.set_verdict(
            checked,
            verdict is not None,
            verdict.passed if verdict is not None else False,
            verdict.code if verdict is not None else 0,
            verdict.message if verdict is not None else "",
        )
        super().set_greennet_verdict(verdict, checked=checked)

    def _sync_verdict(self):
        decided, checked, present, passed, code, message, _ = self._native_cell.audit()
        if decided:
            verdict = GreenNetVerdict(passed, code, message) if present else None
            super().set_greennet_verdict(verdict, checked=checked)

    @property
    def greennet_passed(self):
        if self._native_published:
            _, checked, present, passed, *_ = self._native_cell.audit()
            return checked and present and passed
        return super().greennet_passed

    def inspection_decision(self):
        if self._native_published:
            self._sync_verdict()
        return super().inspection_decision()

    def greennet_snapshot(self):
        if self._native_published:
            self._sync_verdict()
        return super().greennet_snapshot()


class NativeMMEmbeddingCache(MMEmbeddingCache):
    """Own Python pending/inspection state through each native generation."""

    def __init__(
        self,
        gpu_max_bytes,
        cpu_max_bytes,
        *,
        report_metrics=False,
        async_publication=False,
        lifetime_metrics=None,
        publish_queue_size=1024,
    ):
        from librtp_compute_ops import EmbLocalCacheShard, record_emb_local_ready

        self._validate_limits(gpu_max_bytes, cpu_max_bytes)
        # Share the existing request/publish coordinator. Zero capacities here
        # leave the legacy pools unallocated.
        super().__init__(
            0,
            0,
            report_metrics=report_metrics,
            lifetime_metrics=lifetime_metrics,
        )
        self._gpu_max_bytes = gpu_max_bytes
        self._cpu_max_bytes = cpu_max_bytes
        self._native_closed = False
        self._pending_entries = {}
        self._native_device = (
            torch.cuda.current_device()
            if gpu_max_bytes or torch.cuda.is_available()
            else -1
        )
        with cuda_graph_gate.operation():
            self._core = EmbLocalCacheShard(
                gpu_max_bytes, cpu_max_bytes, self._native_device
            )
        self._record_native_ready = record_emb_local_ready
        if async_publication:
            self._start_publication_thread(publish_queue_size)
        if self._owns_lifetime_metrics and self._lifetime_metrics is not None:
            self._lifetime_metrics.resize(gpu_max_bytes + cpu_max_bytes)

    def _new_entry(self, cache_key, cell, ready=False):
        # Only pending work has a canonical Python entry. A resident
        # generation holds the native record and immutable topology.
        owner = weakref.ref(self)

        def on_read(entry, timeout):
            shard = owner()
            if shard is None:
                raise RuntimeError("embedding cache owner was released")
            return shard._read_entry(cache_key, entry, timeout)

        generation = f"{self.instance_id}-{cell.generation}"
        if ready:
            return _NativeRequestEntry(generation, cell, None, None, on_read, True)

        def on_complete(entry, result):
            shard = owner()
            if shard is not None:
                shard._on_complete(cache_key, entry, result)

        def on_fail(entry, error):
            shard = owner()
            if shard is not None:
                shard._on_fail(cache_key, entry, error)

        return _NativeRequestEntry(generation, cell, on_complete, on_fail, on_read)

    def try_acquire(self, cache_key):
        with self._lock:
            cell, tier = (
                self._core.claim(cache_key)
                if self.enabled and not self._native_closed
                else (None, "")
            )
            if cell is None:
                entry = MMEmbeddingCacheEntry()
                state = "miss"
            elif tier:
                entry = self._new_entry(cache_key, cell, ready=True)
                entry.tier = tier
                if self._report_metrics_enabled:
                    entry._cache_hit_implementation = "native"
                state = "complete"
                self._stats[tier + "_hit"] += 1
            else:
                entry = self._pending_entries.get(cell)
                if entry is None:
                    entry = self._new_entry(cache_key, cell)
                    self._pending_entries[cell] = entry
                    state = "miss"
                else:
                    state = "complete" if entry.is_done else "in_progress"
            self._stats[
                {"complete": "hit", "in_progress": "inflight_dedup", "miss": "miss"}[
                    state
                ]
            ] += 1
        self._report_access_metric(state)
        return state, entry

    def complete(self, cache_key, entry, result, *, before_publish=None):
        if not hasattr(entry, "_native_cell"):
            return super().complete(cache_key, entry, result, before_publish=before_publish)
        with cuda_graph_gate.operation():
            streams = [
                (stream.cuda_stream, stream.device.index)
                for stream in _current_cuda_streams(result)
            ]
            ready = self._record_native_ready(streams) if streams else []
        return entry.complete(
            result,
            before_publish=before_publish,
            defer=self._publish_queue is not None,
            prepared_storage=(None, ready),
        )

    def _stream(self):
        device = self._native_device
        if device < 0:
            return 0, device
        return torch.cuda.current_stream(device).cuda_stream, device

    def _on_complete_sync(self, cache_key, entry, result):
        cell = getattr(entry, "_native_cell", None)
        if cell is None:
            return
        published = False
        charged = 0
        evictions = demoted = 0
        try:
            ready = entry.ready_events
            with cuda_graph_gate.operation():
                published, charged, evictions, demoted = self._core.publish(
                    cache_key, cell, result, _embedding_output_tokens(result), ready,
                    *self._stream()
                )
            if published:
                with entry.storage_lock:
                    entry._native_published = True
                    entry.result = None
                    entry.ready_events = []
        except Exception:
            with self._lock:
                self._stats["transfer_error"] += 1
            logging.warning("ViT native cache insertion failed for %s", cache_key, exc_info=True)
        with self._lock:
            if not published:
                self._core.erase(cache_key, cell)
            self._pending_entries.pop(cell, None)
        self._record_mutations(evictions, demoted)
        self._report_admission(charged)
        self._report_current_residency()

    def _read_entry(self, cache_key, entry, timeout):
        cell = entry._native_cell
        with cuda_graph_gate.operation():
            stream = self._stream()
            result, promoted, evicted, demoted, promotion_failed = self._core.read(
                cache_key, cell, *stream
            )
            with entry.storage_lock:
                published = entry._native_published
                fallback = entry.result
                ready = entry.ready_events
            if result is None and published:
                # The publisher replaced the staging result after the first
                # lookup. The cell still owns the acquired generation.
                result, promoted, evicted, demoted, promotion_failed = self._core.read(
                    cache_key, cell, *stream
                )
            if result is None:
                for event in ready:
                    event.wait()
                return fallback
        if promoted or evicted or demoted or promotion_failed:
            with self._lock:
                self._stats["promotion"] += int(promoted)
                self._stats["transfer_error"] += int(promotion_failed)
            self._record_mutations(evicted, demoted)
            self._report_current_residency()
        return result

    def _drop_pending_entry(self, cache_key, entry):
        self._remove_if_same(cache_key, entry)

    def _remove_if_same(self, cache_key, expected):
        cell = getattr(expected, "_native_cell", None)
        if cell is None:
            return False
        with self._lock:
            removed = self._core.erase(cache_key, cell)
            self._pending_entries.pop(cell, None)
        if removed:
            self._report_current_residency()
        return removed

    def remove(self, cache_key):
        entry = self.peek(cache_key)
        if entry is not None:
            self._remove_if_same(cache_key, entry)

    def peek(self, cache_key):
        with self._lock:
            cell, tier = self._core.peek(cache_key)
            if tier:
                return self._new_entry(cache_key, cell, ready=True)
            return self._pending_entries.get(cell)

    def is_current(self, cache_key, entry):
        cell = getattr(entry, "_native_cell", None)
        return cell is not None and self._core.is_current(cache_key, cell)

    def _with_current_entry(self, cache_key, expected, action):
        cell = getattr(expected, "_native_cell", None)
        if cell is None or expected.error is not None:
            return None
        return self._core.with_current(cache_key, cell, action)

    def resident_tiers(self, keys=None, limit=None):
        return self._core.resident_tiers(keys, -1 if limit is None else max(0, limit))

    def resize(self, gpu_max_bytes, cpu_max_bytes):
        self._validate_limits(gpu_max_bytes, cpu_max_bytes)
        with cuda_graph_gate.operation():
            evicted, demoted, cleared_through = self._core.resize(
                gpu_max_bytes, cpu_max_bytes, *self._stream()
            )
        with self._lock:
            self._gpu_max_bytes = gpu_max_bytes
            self._cpu_max_bytes = cpu_max_bytes
            if cleared_through:
                self._pending_entries = {
                    cell: entry
                    for cell, entry in self._pending_entries.items()
                    if cell.generation > cleared_through
                }
        self._record_mutations(evicted, demoted)
        if self._owns_lifetime_metrics and self._lifetime_metrics is not None:
            self._lifetime_metrics.resize(gpu_max_bytes + cpu_max_bytes)
        self._report_current_residency()

    def clear(self, error=None):
        with self._lock:
            self._core.clear()
            pending, self._pending_entries = self._pending_entries, {}
        if error is not None:
            for entry in pending.values():
                if not entry.is_done:
                    entry.fail(error)
        self._report_current_residency()

    def close(self):
        with self._lock:
            if self._native_closed:
                return
            self._native_closed = True
        self.clear(RuntimeError("embedding cache closed"))
        with cuda_graph_gate.operation():
            self._core.close()
        super().close()

    def _record_mutations(self, evicted, demoted):
        if evicted or demoted:
            with self._lock:
                self._stats["eviction"] += evicted
                self._stats["demotion"] += demoted
        if evicted and self._report_metrics_enabled:
            kmonitor.report(AccMetrics.VIT_EMBEDDING_CACHE_EVICTION_QPS_METRIC, evicted)

    def stats(self):
        with self._lock:
            stats = {
                **self._stats,
                "gpu_max_bytes": self._gpu_max_bytes,
                "cpu_max_bytes": self._cpu_max_bytes,
            }
        stats.update(self._core.stats())
        return stats

    def _report_current_residency(self):
        if self._report_metrics_enabled:
            tokens, nbytes = self._core.resident_charge()
            self._report_resident_metrics(tokens, nbytes)
