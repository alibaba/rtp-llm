"""Explicit-bucket histograms reported through KMonitor's scalar transport."""

import math
import os
import uuid
from bisect import bisect_left
from functools import lru_cache
from threading import Lock

from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_base import (
    MetricBase,
    MetricDataPoint,
)


@lru_cache(maxsize=1)
def _process_instance_id(pid):
    # Key by PID to regenerate after fork; UUID distinguishes PID reuse/restarts.
    return uuid.uuid4().hex


class HistogramMetric(MetricBase):
    """Cumulative nonnegative observations, never reset by a reporting cycle.

    Emit numeric _bucket (le tag), _sum and _count components via Flume.
    Receivers must retain process_instance_id and treat them as cumulative
    snapshots, not add successive snapshots as new observations.
    """

    def __init__(self, name, tags, buckets):
        super().__init__(name, dict(tags))
        self.buckets = tuple(float(bound) for bound in buckets)
        if (
            not self.buckets
            or any(not math.isfinite(b) or b < 0 for b in self.buckets)
            or any(a >= b for a, b in zip(self.buckets, self.buckets[1:]))
        ):
            raise ValueError(
                "histogram buckets must be finite, nonnegative and increasing"
            )
        if "le" in tags or "process_instance_id" in tags:
            raise ValueError("le and process_instance_id are reserved histogram tags")
        self.lock = Lock()
        self.series = {}

    def report(self, value=1, tags=None):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("histogram observations must be finite and nonnegative")
        tags = tags or {}
        if "le" in tags or "process_instance_id" in tags:
            raise ValueError("le and process_instance_id are reserved histogram tags")
        report_tags = dict(self.tags)
        report_tags.update(tags)
        report_tags["process_instance_id"] = _process_instance_id(os.getpid())
        key = tuple(sorted(report_tags.items()))
        with self.lock:
            counts, total = self.series.get(key, ([0] * (len(self.buckets) + 1), 0.0))
            counts[bisect_left(self.buckets, value)] += 1
            self.series[key] = (counts, total + value)

    def fetch_reported_data(self):
        points = []
        with self.lock:
            for key, (counts, total) in self.series.items():
                tags = dict(key)
                cumulative = 0
                for bound, count in zip(self.buckets + (math.inf,), counts):
                    cumulative += count
                    bucket_tags = dict(
                        tags, le="+Inf" if math.isinf(bound) else str(bound)
                    )
                    points.append(
                        MetricDataPoint(cumulative, bucket_tags, self.name + "_bucket")
                    )
                points.append(MetricDataPoint(total, dict(tags), self.name + "_sum"))
                points.append(
                    MetricDataPoint(cumulative, dict(tags), self.name + "_count")
                )
        return points
