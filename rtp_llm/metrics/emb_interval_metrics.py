"""Interval telemetry for embedding cache hot paths."""

import time
from bisect import bisect_left
from collections import defaultdict
from threading import Lock
from weakref import WeakSet

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
GDR_LOCK_HOLD_RT = PREFIX + "gdr_lock_hold_ms"
OPERATION_QPS = PREFIX + "operation_qps"
LOCAL_HIT_RT = "py_rtp_vit_embedding_cache_hit_rt_ms"
LOCAL_HIT_BOUNDS_MS = (
    0.025,
    0.05,
    0.075,
    0.1,
    0.125,
    0.25,
    0.5,
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    500,
    1000,
)
LOCAL_HIT_BUCKET_LABELS = tuple(map(str, LOCAL_HIT_BOUNDS_MS)) + ("+Inf",)
OBJECT_RT_BOUNDS_MS = (10, 25, 50, 75, 100, 150, 250, 500, 1000, 3000)
OBJECT_RT_BUCKET_LABELS = tuple(map(str, OBJECT_RT_BOUNDS_MS)) + ("+Inf",)
OBJECT_RT_OUTCOMES = {PREFIX + "read_rt": ("hit",), PREFIX + "write_rt": ("success",)}


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
            # Keep emitting zero for previously active series. Monitoring
            # averages over reported points and would otherwise ignore idle time.
            self._counts = defaultdict(
                lambda: defaultdict(int),
                ((key, defaultdict(int)) for key in counts),
            )
        return [
            point
            for (operation, stage), statuses in counts.items()
            for point in outcome_rates(
                statuses, dict(self.tags, operation=operation, stage=stage), interval
            )
        ]


class EmbCacheTokensMetric(MetricBase):
    """Sample total resident tokens across live shards on every KMonitor flush."""

    def __init__(self, name, tags):
        super().__init__(name, tags)
        self._shards = WeakSet()
        self._lock = Lock()

    def track(self, shard):
        with self._lock:
            self._shards.add(shard)

    def untrack(self, shard):
        with self._lock:
            self._shards.discard(shard)

    def fetch_reported_data(self):
        with self._lock:
            shards = tuple(self._shards)
        if not shards:
            return []
        tokens = sum(shard.resident_tokens() for shard in shards)
        return [MetricDataPoint(tokens, self.tags)]


class EmbInflightMetric(MetricBase):
    """Record an in-process balance and its high-water mark per flush."""

    def __init__(self, name, tags):
        super().__init__(name, tags)
        self._lock = Lock()
        self._current = defaultdict(int)
        self._peak = defaultdict(int)

    def report(self, value, tags):
        stage = tags["stage"]
        with self._lock:
            current = self._current[stage] + value
            self._current[stage] = current
            if current > self._peak[stage]:
                self._peak[stage] = current

    def fetch_reported_data(self):
        with self._lock:
            samples = [
                (stage, current, self._peak[stage])
                for stage, current in self._current.items()
            ]
            self._peak.update(self._current)
        points = []
        for stage, current, peak in samples:
            tags = dict(self.tags, stage=stage)
            points.append(MetricDataPoint(current, tags))
            points.append(MetricDataPoint(peak, tags, self.name + "_peak"))
        return points


class EmbSampleMetric(MetricBase):
    """Aggregate remote EMB gauges without queuing one point per request."""

    def __init__(self, name, tags, *, arena=False):
        super().__init__(name, tags)
        self._tag_name = "stage" if arena else "operation"
        self._arena = arena
        self._lock = Lock()
        self._samples = {}

    def report(self, value, tags):
        key = tags[self._tag_name]
        with self._lock:
            if self._arena:
                previous = self._samples.get(key, value)
                # Capacity is reported as a maximum. Keep the lowest observed
                # free values so brief fragmentation remains visible.
                self._samples[key] = (
                    max(previous, value) if key == "total" else min(previous, value)
                )
            else:
                sample = self._samples.get(key)
                if sample is None:
                    self._samples[key] = [value, 1]
                else:
                    sample[0] += value
                    sample[1] += 1

    def fetch_reported_data(self):
        with self._lock:
            samples, self._samples = self._samples, {}
        return [
            MetricDataPoint(
                sample if self._arena else sample[0] / sample[1],
                dict(self.tags, **{self._tag_name: key}),
            )
            for key, sample in samples.items()
        ]


class _Latency:
    __slots__ = ("count", "total", "maximum", "bounds", "buckets")

    def __init__(self, bounds):
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.bounds = bounds
        self.buckets = [0] * (len(bounds) + 1) if bounds else None

    def add(self, milliseconds):
        self.count += 1
        self.total += milliseconds
        if milliseconds > self.maximum:
            self.maximum = milliseconds
        if self.buckets is not None:
            self.buckets[bisect_left(self.bounds, milliseconds)] += 1


class EmbLatencyMetric(MetricBase):
    def __init__(self, name, tags):
        super().__init__(name, tags)
        self._is_stage = name == STAGE_RT
        self._is_queue = name == QUEUE_RT
        self._is_gdr_lock_hold = name == GDR_LOCK_HOLD_RT
        self._is_local_hit = name == LOCAL_HIT_RT
        self._object_outcome = OBJECT_RT_OUTCOMES.get(name)
        if self._is_stage:
            self._tag_names = ("operation", "stage", "status")
        elif self._is_queue:
            self._tag_names = ("operation",)
        elif self._is_gdr_lock_hold:
            self._tag_names = ("operation", "phase")
        elif self._is_local_hit:
            self._tag_names = ("tier", "implementation")
        else:
            self._tag_names = ("status",)
        self._lock = Lock()
        self._samples = {}
        self._stage_keys = set()
        self._last = time.monotonic()

    def report(self, value, tags):
        # These tags come from the fixed operation/stage/status call sites.
        # No point, tag merge or formatter runs on the per-request path.
        if self._is_stage:
            key = tags["operation"], tags["stage"], tags["status"]
        elif self._is_queue:
            key = (tags["operation"],)
        elif self._is_gdr_lock_hold:
            key = tags["operation"], tags["phase"]
        elif self._is_local_hit:
            key = tags["tier"], tags["implementation"]
        else:
            key = (tags["status"],)
        with self._lock:
            entry = self._samples.get(key)
            if entry is None:
                if self._is_local_hit:
                    bounds = LOCAL_HIT_BOUNDS_MS
                elif key == self._object_outcome:
                    bounds = OBJECT_RT_BOUNDS_MS
                elif self._is_stage and key[:2] in (
                    ("read", "native_restore_header"),
                    ("read", "d2h_header"),
                ):
                    bounds = BOUNDS_MS
                elif key in (("read",), ("read_transfer",)) or key[:2] == (
                    "read",
                    "metadata",
                ):
                    bounds = BOUNDS_MS
                else:
                    bounds = ()
                entry = self._samples[key] = _Latency(bounds)
            entry.add(value)

    def fetch_reported_data(self):
        with self._lock:
            now = time.monotonic()
            interval = now - self._last
            if interval <= 0:
                return []
            self._last = now
            samples, self._samples = self._samples, {}
            if self._is_stage:
                self._stage_keys.update(samples)
            stage_keys = tuple(self._stage_keys)

        points = []
        for key, entry in samples.items():
            tags = dict(self.tags, **dict(zip(self._tag_names, key)))
            points.append(MetricDataPoint(entry.total / entry.count, tags))
            points.append(MetricDataPoint(entry.count, tags, self.name + "_count"))
            points.append(MetricDataPoint(entry.total, tags, self.name + "_sum"))
            points.append(MetricDataPoint(entry.maximum, tags, self.name + "_max"))
            if entry.buckets is not None:
                running = 0
                if self._is_local_hit:
                    labels = LOCAL_HIT_BUCKET_LABELS
                elif key == self._object_outcome:
                    labels = OBJECT_RT_BUCKET_LABELS
                else:
                    labels = BUCKET_LABELS
                for label, count in zip(labels, entry.buckets):
                    running += count
                    points.append(
                        MetricDataPoint(
                            running, dict(tags, le=label), self.name + "_bucket"
                        )
                    )

        stage_counts = defaultdict(lambda: defaultdict(int))
        for key in stage_keys:
            operation, stage, status = key
            sample = samples.get(key)
            count = sample.count if sample is not None else 0
            outcome = status if status in ("success", "skipped") else "error"
            stage_counts[operation, stage][outcome] += count
            tags = dict(self.tags, operation=operation, stage=stage, status=status)
            points.append(MetricDataPoint(count / interval, tags, PREFIX + "stage_qps"))

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
