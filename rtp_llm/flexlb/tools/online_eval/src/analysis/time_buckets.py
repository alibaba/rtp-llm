"""Explicit bucket boundaries, independent of a report's display origin."""

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeBuckets:
    origin_epoch_s: float
    width_s: float

    def __post_init__(self):
        if any(type(v) not in (int, float) or not math.isfinite(v)
               for v in (self.origin_epoch_s, self.width_s)) or self.width_s <= 0:
            raise ValueError("time buckets require a finite origin and positive width")

    def index(self, epoch_ms):
        if type(epoch_ms) not in (int, float) or not math.isfinite(epoch_ms):
            raise ValueError("bucket timestamp must be finite epoch milliseconds")
        return math.floor((epoch_ms - self.origin_epoch_s * 1000) / (self.width_s * 1000))

    def epoch_s(self, index):
        return self.origin_epoch_s + index * self.width_s

    def group(self, rows, *, timestamp_ms):
        buckets = defaultdict(list)
        for row in rows:
            buckets[self.index(timestamp_ms(row))].append(row)
        return buckets
