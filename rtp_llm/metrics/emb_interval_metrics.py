"""Interval telemetry for the remote embedding cache's hot path."""

import time
from bisect import bisect_left
from collections import defaultdict
from threading import Lock

from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_base import (
    MetricBase,
    MetricDataPoint,
)

# Milliseconds. Export cumulative bucket counts within each interval only.
BOUNDS_MS = (
    0.125,
    0.25,
    0.5,
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    75,
    100,
    150,
    200,
    300,
    500,
    750,
    1000,
    1500,
    2500,
    5000,
)
BUCKET_LABELS = tuple(map(str, BOUNDS_MS)) + ("+Inf",)
PREFIX = "py_rtp_vit_remote_embedding_cache_"
STAGE_RT = PREFIX + "stage_rt_ms"
QUEUE_RT = PREFIX + "queue_wait_ms"
OPERATION_QPS = PREFIX + "operation_qps"


def outcome_rates(counts, tags, interval):
    return [
        MetricDataPoint(
            sum(counts.values()) / interval, dict(tags, status="total"), OPERATION_QPS
        ),
        *(
            MetricDataPoint(
                counts[status] / interval, dict(tags, status=status), OPERATION_QPS
            )
            for status in ("success", "error", "skipped")
        ),
    ]


class EmbOutcomeMetric(MetricBase):
    """Record one outcome and derive the legacy totals at flush time."""

    def __init__(self, name, tags):
        super().__init__(name, tags)
        self._lock = Lock()
        self._counts = defaultdict(lambda: defaultdict(int))
        self._last = time.monotonic()

    def report(self, value, tags):
        key = tags["operation"], tags["stage"]
        status = tags["status"]
        with self._lock:
            self._counts[key][status] += value

    def fetch_reported_data(self):
        with self._lock:
            now = time.monotonic()
            interval = now - self._last
            if interval <= 0:
                return []
            self._last = now
            counts = self._counts
            self._counts = defaultdict(lambda: defaultdict(int))
        return [
            point
            for (operation, stage), statuses in counts.items()
            for point in outcome_rates(
                statuses, dict(self.tags, operation=operation, stage=stage), interval
            )
        ]


class _Latency:
    __slots__ = ("count", "total", "maximum", "buckets")

    def __init__(self, histogram):
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.buckets = [0] * len(BUCKET_LABELS) if histogram else None

    def add(self, milliseconds):
        self.count += 1
        self.total += milliseconds
        if milliseconds > self.maximum:
            self.maximum = milliseconds
        if self.buckets is not None:
            self.buckets[bisect_left(BOUNDS_MS, milliseconds)] += 1


class EmbLatencyMetric(MetricBase):
    def __init__(self, name, tags):
        super().__init__(name, tags)
        self._is_stage = name == STAGE_RT
        self._is_queue = name == QUEUE_RT
        self._tag_names = (
            ("operation", "stage", "status")
            if self._is_stage
            else ("operation",) if self._is_queue else ("status",)
        )
        self._lock = Lock()
        self._samples = {}
        self._last = time.monotonic()

    def report(self, value, tags):
        # These tags come from the fixed operation/stage/status call sites.
        # No point, tag merge or formatter runs on the per-request path.
        if self._is_stage:
            key = tags["operation"], tags["stage"], tags["status"]
        elif self._is_queue:
            key = (tags["operation"],)
        else:
            key = (tags["status"],)
        with self._lock:
            entry = self._samples.get(key)
            if entry is None:
                histogram = key == ("read",) or key[:2] == ("read", "metadata")
                entry = self._samples[key] = _Latency(histogram)
            entry.add(value)

    def fetch_reported_data(self):
        with self._lock:
            now = time.monotonic()
            interval = now - self._last
            if interval <= 0:
                return []
            self._last = now
            samples, self._samples = self._samples, {}

        points = []
        stage_counts = defaultdict(lambda: defaultdict(int))
        for key, entry in samples.items():
            tags = dict(self.tags, **dict(zip(self._tag_names, key)))
            points.append(MetricDataPoint(entry.total / entry.count, tags))
            points.append(MetricDataPoint(entry.count, tags, self.name + "_count"))
            points.append(MetricDataPoint(entry.total, tags, self.name + "_sum"))
            points.append(MetricDataPoint(entry.maximum, tags, self.name + "_max"))
            if entry.buckets is not None:
                running = 0
                for label, count in zip(BUCKET_LABELS, entry.buckets):
                    running += count
                    points.append(
                        MetricDataPoint(
                            running, dict(tags, le=label), self.name + "_bucket"
                        )
                    )
            if self._is_stage:
                operation, stage, status = key
                outcome = status if status in ("success", "skipped") else "error"
                stage_counts[operation, stage][outcome] += entry.count
                points.append(
                    MetricDataPoint(entry.count / interval, tags, PREFIX + "stage_qps")
                )

        # Existing dashboards consume these views of the same observations.
        for (operation, stage), counts in stage_counts.items():
            tags = dict(self.tags, operation=operation, stage=stage)
            points.append(
                MetricDataPoint(
                    counts["error"] / interval, tags, PREFIX + "stage_error_qps"
                )
            )
            points.extend(
                outcome_rates(
                    counts, dict(tags, operation=operation + "_stage"), interval
                )
            )
        return points
