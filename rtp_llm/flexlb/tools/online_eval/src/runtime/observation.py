"""Clock anchors, bounded samples and polling independent of case measurement."""

import json
import math
from dataclasses import dataclass


def capture_limits(data, path):
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping")
    fields = {"max_samples", "max_bytes"}
    extra = set(data) - fields
    if extra:
        raise ValueError(f"{path}: unknown configuration fields {sorted(extra)}")
    missing = fields - set(data)
    if missing:
        raise ValueError(f"missing YAML parameter {path}.{sorted(missing)[0]}")
    for field in data:
        if type(data[field]) is not int or data[field] < 1:
            raise ValueError(f"{path}.{field}: positive integer required")
    return dict(data)


class SampleBudget:
    def __init__(self, limits):
        self.limits = capture_limits(limits, "capture")
        self.count = self.bytes = 0

    def append(self, row):
        encoded = json.dumps(row, separators=(",", ":"), allow_nan=False)
        return self.append_encoded(encoded)

    def append_encoded(self, encoded):
        size = len(encoded.encode()) + 1
        if self.count + 1 > self.limits["max_samples"] or self.bytes + size > self.limits["max_bytes"]:
            raise ValueError("observation sample/byte budget exceeded; evidence is incomplete")
        self.count += 1
        self.bytes += size
        return encoded


@dataclass(frozen=True)
class ObservationClock:
    origin_epoch_s: float
    origin_monotonic_s: float

    def stamp(self, monotonic_s):
        elapsed = monotonic_s - self.origin_monotonic_s
        return dict(elapsed_s=elapsed, epoch_s=self.origin_epoch_s + elapsed,
                    monotonic_s=self.origin_monotonic_s + elapsed)

    def to_dict(self):
        return dict(origin_epoch_s=self.origin_epoch_s,
                    origin_monotonic_s=self.origin_monotonic_s,
                    epoch_policy="anchored_monotonic")


def poll_samples(deadline, interval_s, sample, *, until=None, immediate=False):
    """Preserve the last sample at a boundary; all waiting uses the stage clock."""
    if type(interval_s) not in (int, float) or not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("poll interval must be finite and positive")
    first = True
    while first or until is None or deadline.clock() < until:
        deadline.check()
        if not (first and immediate):
            wait = interval_s if until is None else min(interval_s, max(0, until - deadline.clock()))
            deadline.sleep(wait)
        first = False
        yield sample()


@dataclass(frozen=True)
class FrozenSnapshot:
    """Serialized canonical payload makes returned dictionaries independent deep copies."""

    encoded: str

    def to_dict(self):
        return json.loads(self.encoded)
