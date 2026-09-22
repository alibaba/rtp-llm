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
    def __init__(self, visual, enabled=None, max_entries=None, max_patches=None):
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
        if self.max_entries < 0 or self.max_patches < 0:
            raise ValueError("vision graph limits must be non-negative")
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._seen = OrderedDict()
        self._stats = dict(hit=0, miss=0, capture=0, fallback=0)

    def stats(self):
        with self._lock:
            return dict(self._stats, entries=len(self._entries))

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
        if not self.enabled or not self.max_entries or grid.shape[0] != 1:
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
                            **kwargs
                        )
                stream.wait_stream(capture_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    output = self.visual(
                        static_input, grid_thw=grid, _graph_metadata=metadata, **kwargs
                    ).pooler_output
                stream.wait_stream(capture_stream)
                event = torch.cuda.Event()
                event.record(capture_stream)
                entry = (graph, static_input, output, metadata, event)
                self._entries[signature] = entry
                while len(self._entries) > self.max_entries:
                    _, victim = self._entries.popitem(last=False)
                    # Do not release captured storage while replay/clone is in flight.
                    victim[-1].synchronize()
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
