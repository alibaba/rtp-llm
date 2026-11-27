"""Python request orchestration for the native local EMB resident cache."""

import logging
import uuid
import weakref

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

from .greennet_hook import GreenNetVerdict
from .mm_embedding_cache import (
    _SIGNALED_EVENT,
    MMEmbeddingCache,
    MMEmbeddingCacheEntry,
    MMHashKeyCache,
    _current_cuda_streams,
    _embedding_output_tokens,
)


class NativeMMHashKeyCache(MMHashKeyCache):
    """Route metadata requests through one native, byte-bounded hash index."""

    def __init__(self, max_bytes):
        from librtp_compute_ops import EmbLocalHashCache

        if max_bytes < 0:
            raise ValueError("hash cache max_bytes must be non-negative")
        self._hash_core = EmbLocalHashCache(max_bytes)
        self.instance_id = uuid.uuid4().hex

    @property
    def enabled(self):
        return self._hash_core.enabled()

    def put(self, cache_key, feature_hashes, generation=None, greennet_passed=False):
        with cuda_graph_gate.operation():
            self._hash_core.put(
                cache_key, feature_hashes, generation or "", bool(greennet_passed)
            )

    def put_for_entry(self, cache_key, feature_hashes, entry, embedding_cache):
        if entry.error is not None:
            return
        shard = embedding_cache._shard(cache_key)
        with cuda_graph_gate.operation():
            self._hash_core.put_for_entry(
                shard._core,
                entry._native_cell,
                cache_key,
                feature_hashes,
                entry.generation,
                entry.greennet_passed,
            )

    def _snapshot(self, cache_key, generation=None, *, touch=False):
        return self._hash_core.snapshot(cache_key, generation, touch)

    def _metadata_snapshots(self, keys):
        return zip(keys, self._hash_core.metadata(keys, self._MAX_METADATA_ROWS))

    def greennet_passed(self, cache_key):
        return self._hash_core.approved(cache_key)

    def mark_greennet_passed(self, cache_key, generation):
        return self._hash_core.mark_approved(cache_key, generation)

    def contains(self, cache_key):
        return self._hash_core.contains(cache_key)

    def keys(self, limit=None):
        if limit is not None and limit < 0:
            raise ValueError("hash cache key limit must be non-negative")
        return self._hash_core.keys(limit)

    def clear(self):
        self._hash_core.clear()

    def resize(self, max_bytes):
        if max_bytes < 0:
            raise ValueError("hash cache max_bytes must be non-negative")
        self._hash_core.resize(max_bytes)

    def stats(self):
        entries, resident, building, retired, capacity = self._hash_core.stats()
        return {
            "resident_entries": entries,
            "resident_bytes": resident,
            "building_bytes": building,
            "retired_bytes": retired,
            "max_bytes": capacity,
        }


class _NativeRequestEntry(MMEmbeddingCacheEntry):
    """A pending producer or transient request handle for a native generation."""

    def __init__(
        self, generation, cell, on_complete, on_fail, on_read, *, ready, track_evictions
    ):
        super().__init__(on_complete, on_fail, on_read, generation=generation)
        self._native_cell = cell
        self._native_published = ready
        self._track_evictions = track_evictions
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

    def set_greennet_error(self, error):
        if self._track_evictions:
            self._native_cell.set_audit_error()
        super().set_greennet_error(error)

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
        on_cpu_evict=None,
        on_gpu_evict=None,
        eviction_snapshot_max_bytes=None,
    ):
        from librtp_compute_ops import EmbLocalCacheShard, record_emb_local_ready

        self._validate_limits(gpu_max_bytes, cpu_max_bytes)
        self._native_device = (
            torch.cuda.current_device()
            if gpu_max_bytes or torch.cuda.is_available()
            else -1
        )
        self._notify_evictions = on_cpu_evict is not None or on_gpu_evict is not None
        core_options = ()
        if self._notify_evictions:
            core_options = (
                on_cpu_evict is not None,
                on_gpu_evict is not None,
                64 * 1024**2
                if eviction_snapshot_max_bytes is None
                else eviction_snapshot_max_bytes,
            )
        with cuda_graph_gate.operation():
            self._core = EmbLocalCacheShard(
                gpu_max_bytes, cpu_max_bytes, self._native_device, *core_options
            )
        # Share the existing request/publish coordinator. Zero capacities here
        # leave the legacy pools unallocated.
        super().__init__(
            0,
            0,
            report_metrics=report_metrics,
            lifetime_metrics=lifetime_metrics,
            on_cpu_evict=on_cpu_evict,
            on_gpu_evict=on_gpu_evict,
        )
        self._gpu_max_bytes = gpu_max_bytes
        self._cpu_max_bytes = cpu_max_bytes
        self._native_closed = False
        self._pending_entries = {}
        self._record_native_ready = record_emb_local_ready
        if async_publication:
            self._start_publication_thread(publish_queue_size)
        if self._owns_lifetime_metrics and self._lifetime_metrics is not None:
            self._lifetime_metrics.resize(gpu_max_bytes + cpu_max_bytes)

    def _new_entry(self, cache_key, cell, generation, ready=False):
        # Only pending work has a canonical Python entry. A resident
        # generation holds the native record and immutable topology.
        owner = weakref.ref(self)

        def on_read(entry, timeout):
            shard = owner()
            if shard is None:
                raise RuntimeError("embedding cache owner was released")
            return shard._read_entry(cache_key, entry, timeout)

        generation = f"{self.instance_id}-{generation}"
        if ready:
            return _NativeRequestEntry(
                generation, cell, None, None, on_read,
                ready=True, track_evictions=self._notify_evictions,
            )

        def on_complete(entry, result):
            shard = owner()
            if shard is not None:
                shard._on_complete(cache_key, entry, result)

        def on_fail(entry, error):
            shard = owner()
            if shard is not None:
                shard._on_fail(cache_key, entry, error)

        entry = _NativeRequestEntry(
            generation, cell, on_complete, on_fail, on_read,
            ready=False, track_evictions=self._notify_evictions,
        )
        entry.capture_compute_streams = True
        return entry

    def try_acquire(self, cache_key):
        with self._lock:
            cell, tier, generation = (
                self._core.claim(cache_key)
                if self.enabled and not self._native_closed
                else (None, "", 0)
            )
            if cell is None:
                entry = MMEmbeddingCacheEntry()
                state = "miss"
            elif tier:
                entry = self._new_entry(cache_key, cell, generation, ready=True)
                entry.tier = tier
                if self._report_metrics_enabled:
                    entry._cache_hit_implementation = "native"
                state = "complete"
                self._stats[tier + "_hit"] += 1
            else:
                entry = self._pending_entries.get(cell)
                if entry is None:
                    entry = self._new_entry(cache_key, cell, generation)
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
            return super().complete(
                cache_key, entry, result, before_publish=before_publish
            )
        with cuda_graph_gate.operation():
            # Deferred inspection can publish from a different thread.
            producers = entry.producer_streams or _current_cuda_streams(result)
            streams = [
                (stream.cuda_stream, stream.device.index) for stream in producers
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

    def _deliver_evictions(self, snapshots):
        # Called after the native mutation and outside cache locks. A callback
        # may retain its value, raise, or immediately re-enter the same shard.
        for snapshot in snapshots:
            try:
                (
                    key, remote_key, generation, tier, result, origin,
                    present, checked, passed, code, message,
                ) = snapshot.unpack()
                entry = MMEmbeddingCacheEntry(
                    generation=f"{self.instance_id}-{generation}"
                )
                entry.tier = tier
                entry.remote_cache_context = {"key": remote_key, "trace_id": None}
                entry.remote_embedding_origin = origin
                entry.set_greennet_verdict(
                    GreenNetVerdict(passed, code, message) if present else None,
                    checked=checked,
                )
                entry.complete(result)
                callback = self._on_gpu_evict if tier == "gpu" else self._on_cpu_evict
                callback(key, entry)
            except Exception:
                logging.warning(
                    "ViT native cache eviction callback failed", exc_info=True
                )

    def _on_complete_sync(self, cache_key, entry, result):
        cell = getattr(entry, "_native_cell", None)
        if cell is None:
            return
        published = False
        charged = 0
        evictions = demoted = 0
        snapshots = ()
        try:
            ready = entry.ready_events
            if self._notify_evictions:
                try:
                    context = getattr(entry, "remote_cache_context", None)
                    if context and isinstance(context.get("key"), str):
                        cell.set_remote_key(context["key"])
                except Exception:
                    logging.warning(
                        "ViT native cache eviction metadata skipped", exc_info=True
                    )
            with cuda_graph_gate.operation():
                outcome = self._core.publish(
                    cache_key,
                    cell,
                    result,
                    _embedding_output_tokens(result),
                    ready,
                    *self._stream(),
                )
                if len(outcome) == 5:
                    snapshots = outcome[4]
                    outcome = outcome[:4]
                published, charged, evictions, demoted = outcome
            if published:
                with entry.storage_lock:
                    entry._native_published = True
                    entry.result = None
                    entry.ready_events = []
        except Exception:
            with self._lock:
                self._stats["transfer_error"] += 1
            logging.warning(
                "ViT native cache insertion failed for %s", cache_key, exc_info=True
            )
        with self._lock:
            if not published:
                self._core.erase(cache_key, cell)
            self._pending_entries.pop(cell, None)
        self._record_mutations(evictions, demoted)
        if snapshots:
            self._deliver_evictions(snapshots)
        self._report_admission(charged)
        self._report_current_residency()

    def _read_entry(self, cache_key, entry, timeout):
        cell = entry._native_cell
        with cuda_graph_gate.operation():
            stream = self._stream()
            outcome = self._core.read(cache_key, cell, *stream)
            with entry.storage_lock:
                published = entry._native_published
                fallback = entry.result
                ready = entry.ready_events
            if outcome[0] is None and published:
                # The publisher replaced the staging result after the first
                # lookup. The cell still owns the acquired generation.
                outcome = self._core.read(cache_key, cell, *stream)
            if outcome[0] is None:
                for event in ready:
                    event.wait()
                return fallback
        if len(outcome) == 6:
            self._deliver_evictions(outcome[5])
            outcome = outcome[:5]
        result, promoted, evicted, demoted, promotion_failed = outcome
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
            cell, tier, generation = self._core.peek(cache_key)
            if tier:
                return self._new_entry(cache_key, cell, generation, ready=True)
            return self._pending_entries.get(cell)

    def is_current(self, cache_key, entry):
        cell = getattr(entry, "_native_cell", None)
        return cell is not None and self._core.is_current(cache_key, cell)

    def _with_current_entry(self, cache_key, expected, action):
        cell = getattr(expected, "_native_cell", None)
        if cell is None or expected.error is not None:
            return None
        return self._core.with_current(cache_key, cell, action)

    def resident_tier(self, key, *, touch=False):
        return self._core.resident_tier(key, touch)

    def resident_tiers(self, keys=None, limit=None):
        return self._core.resident_tiers(keys, -1 if limit is None else max(0, limit))

    def resize(self, gpu_max_bytes, cpu_max_bytes):
        self._validate_limits(gpu_max_bytes, cpu_max_bytes)
        with cuda_graph_gate.operation():
            outcome = self._core.resize(
                gpu_max_bytes, cpu_max_bytes, *self._stream()
            )
        snapshots = outcome[3] if len(outcome) == 4 else ()
        evicted, demoted, cleared_through = outcome[:3]
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
        if snapshots:
            self._deliver_evictions(snapshots)
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

    def resident_tokens(self) -> int:
        return self._core.resident_charge()[0]

    def _report_current_residency(self):
        if self._report_metrics_enabled:
            _, nbytes, entries = self._core.resident_charge()
            self._report_resident_metrics(nbytes, entries)
