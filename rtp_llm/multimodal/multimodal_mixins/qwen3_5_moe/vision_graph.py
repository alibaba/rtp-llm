"""Bounded exact-grid CUDA graphs for Qwen3.5 vision inference."""

import logging
import os
import threading
from collections import OrderedDict

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics

logger = logging.getLogger(__name__)


class VisionGraphCache:
    def __init__(
        self,
        visual,
        enabled=None,
        max_entries=None,
        max_patches=None,
        allow_multi_media=None,
        max_cached_bytes=None,
    ):
        self.visual = visual
        self.enabled = (
            (os.environ.get("QWEN35_VIT_CUDA_GRAPH", "1") == "1")
            if enabled is None
            else enabled
        )
        self.max_entries = (
            int(os.environ.get("QWEN35_VIT_GRAPH_MAX_ENTRIES", "4"))
            if max_entries is None
            else max_entries
        )
        self.max_patches = (
            int(os.environ.get("QWEN35_VIT_GRAPH_MAX_PATCHES", "4096"))
            if max_patches is None
            else max_patches
        )
        self.allow_multi_media = (
            os.environ.get("QWEN35_VIT_GRAPH_MULTI_MEDIA", "0") == "1"
            if allow_multi_media is None
            else allow_multi_media
        )
        # A retained-memory budget is mandatory for the experimental packed
        # path. Zero preserves the existing single-media behavior, not an
        # unbounded packed-graph cache. max_patches still limits each capture.
        self.max_cached_bytes = (
            int(os.environ.get("QWEN35_VIT_GRAPH_MAX_CACHED_BYTES", "0"))
            if max_cached_bytes is None
            else max_cached_bytes
        )
        if min(self.max_entries, self.max_patches, self.max_cached_bytes) < 0:
            raise ValueError("vision graph limits must be non-negative")
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._seen = OrderedDict()
        self._entry_bytes = {}
        self._stats = dict(hit=0, miss=0, capture=0, fallback=0)

    def stats(self):
        with self._lock:
            return dict(
                self._stats,
                entries=len(self._entries),
                cached_bytes=(
                    sum(self._entry_bytes.values()) if self.max_cached_bytes else None
                ),
                max_cached_bytes=self.max_cached_bytes,
            )

    @staticmethod
    def _tensor_storage_bytes(*values):
        storages = {}

        def visit(value):
            if isinstance(value, torch.Tensor) and value.is_cuda:
                storage = value.untyped_storage()
                storages[(value.device, storage.data_ptr())] = storage.nbytes()
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        for value in values:
            visit(value)
        return sum(storages.values())

    @staticmethod
    def _graph_pool_bytes(graph, device):
        # Count reserved private-pool segments, not just live output tensors:
        # intermediate activations remain reserved for subsequent replays.
        pool = tuple(graph.pool())
        segments = [
            s
            for s in torch.cuda.memory_snapshot()
            if s["device"] == device.index
            and tuple(s.get("segment_pool_id", ())) == pool
        ]
        if not segments:
            raise RuntimeError("CUDA graph private-pool memory accounting unavailable")
        return sum(s["total_size"] for s in segments)

    def _evict_oldest(self):
        signature, victim = self._entries.popitem(last=False)
        # Do not release captured storage while replay/clone is in flight.
        victim[-1].synchronize()
        self._entry_bytes.pop(signature, None)

    def _report(self, event):
        self._stats[event] += 1
        # The M3 dashboard uses separate counters; keep the tagged counter too.
        metric = {
            "hit": AccMetrics.VIT_CUDA_GRAPH_HIT_QPS_METRIC,
            "miss": AccMetrics.VIT_CUDA_GRAPH_MISS_QPS_METRIC,
            "capture": AccMetrics.VIT_CUDA_GRAPH_CAPTURE_QPS_METRIC,
            "fallback": AccMetrics.VIT_CUDA_GRAPH_FALLBACK_QPS_METRIC,
        }[event]
        try:
            kmonitor.report(metric, 1)
            if event in ("hit", "capture"):
                kmonitor.report(GaugeMetrics.VIT_CUDA_GRAPH_PADDING_RATIO_METRIC, 0)
            kmonitor.report(
                AccMetrics.VIT_GRAPH_EVENT_QPS_METRIC,
                1,
                {"event": event, "model": "qwen35"},
            )
        except Exception:
            logger.warning("Failed to report ViT CUDA graph metrics", exc_info=True)

    @torch.inference_mode()
    def run(self, pixels, grid, **kwargs):
        def eager():
            return self.visual(pixels, grid_thw=grid, **kwargs).pooler_output

        # Count every explicit bypass so eager-only workloads remain visible.
        if (
            not self.enabled
            or not self.max_entries
            or (
                grid.shape[0] != 1
                and not (self.allow_multi_media and self.max_cached_bytes > 0)
            )
        ):
            with self._lock:
                self._report("fallback")
            return eager()
        if (
            not pixels.is_cuda
            or pixels.shape[0] > self.max_patches
            or torch.cuda.is_current_stream_capturing()
        ):
            with self._lock:
                self._report("fallback")
            return eager()
        signature = (
            pixels.device,
            pixels.dtype,
            tuple(pixels.shape),
            tuple(tuple(row) for row in grid.cpu().tolist()),
        )
        with self._lock:
            entry = self._entries.get(signature)
            if entry is not None:
                self._entries.move_to_end(signature)
                self._report("hit")
                return self._replay(entry, pixels)
            self._report("miss")
            seen = self._seen.get(signature, 0)
            if seen < 0:
                self._report("fallback")
                return eager()
            self._seen[signature] = seen + 1
            self._seen.move_to_end(signature)
            while len(self._seen) > 256:
                self._seen.popitem(last=False)
            if seen == 0:
                return eager()
            try:
                metadata = self.visual.prepare_graph_metadata(grid, pixels)
                if metadata["attention_backend"] not in ("fa4", "flash_attention_2"):
                    self._seen[signature] = -1
                    self._report("fallback")
                    return eager()
                if self.max_cached_bytes and (
                    self._tensor_storage_bytes(metadata)
                    + pixels.numel() * pixels.element_size()
                    > self.max_cached_bytes
                ):
                    self._seen[signature] = -1
                    self._report("fallback")
                    return eager()
                stream = torch.cuda.current_stream(pixels.device)
                capture_stream = torch.cuda.Stream(device=pixels.device)
                capture_stream.wait_stream(stream)
                static_input = torch.empty_like(pixels)
                with torch.cuda.stream(capture_stream):
                    static_input.copy_(pixels)
                    for _ in range(2):
                        self.visual(
                            static_input,
                            grid_thw=grid,
                            _graph_metadata=metadata,
                            **kwargs,
                        )
                stream.wait_stream(capture_stream)
                graph = torch.cuda.CUDAGraph()
                # PDFusion runs LLM transfers/pinned allocations on other CPU
                # threads with independent streams and buffers. A global
                # capture would also restrict those threads. Keep capture
                # safety checks on the ViT thread without restricting LLM I/O;
                # the stream/event dependencies below still protect ViT data.
                with torch.cuda.graph(
                    graph, stream=capture_stream, capture_error_mode="thread_local"
                ):
                    output = self.visual(
                        static_input, grid_thw=grid, _graph_metadata=metadata, **kwargs
                    ).pooler_output
                stream.wait_stream(capture_stream)
                event = torch.cuda.Event()
                event.record(capture_stream)
                entry = (graph, static_input, output, metadata, event)
                entry_bytes = 0
                if self.max_cached_bytes:
                    try:
                        entry_bytes = self._graph_pool_bytes(graph, pixels.device)
                    except RuntimeError:
                        # A runtime without private-pool accounting must not
                        # keep an unaccounted graph alive during eager fallback.
                        event.synchronize()
                        graph.reset()
                        del entry, graph, static_input, output, metadata
                        raise
                    entry_bytes += self._tensor_storage_bytes(static_input, metadata)
                    if entry_bytes > self.max_cached_bytes:
                        event.synchronize()
                        graph.reset()
                        del entry, graph, static_input, output, metadata
                        self._seen[signature] = -1
                        self._report("fallback")
                        return eager()
                    while self._entries and (
                        sum(self._entry_bytes.values()) + entry_bytes
                        > self.max_cached_bytes
                    ):
                        self._evict_oldest()
                self._entries[signature] = entry
                self._entry_bytes[signature] = entry_bytes
                while len(self._entries) > self.max_entries:
                    self._evict_oldest()
                self._report("capture")
                return self._replay(entry, pixels)
            except RuntimeError as error:
                self._seen[signature] = -1
                self._report("fallback")
                logger.warning(
                    "Qwen3.5 ViT graph capture failed; eager fallback: %s", error
                )
                return eager()

    @staticmethod
    def _replay(entry, pixels):
        graph, static_input, output, metadata, event = entry
        stream = torch.cuda.current_stream(pixels.device)
        stream.wait_event(event)
        static_input.copy_(pixels)
        graph.replay()
        result = output.clone()
        event.record(stream)
        return result
