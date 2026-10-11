"""Clock anchors, bounded samples and polling independent of case measurement."""

import json
import math
from dataclasses import dataclass

from input_contract import mapping_fields


def capture_limits(data, path):
    mapping_fields(data, {"max_samples", "max_bytes"}, path,
                   required={"max_samples", "max_bytes"})
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

    @classmethod
    def start(cls, ctx):
        event = ctx.record_event("observation_start")
        return cls(event["epoch_s"], event["monotonic_s"])

    def stamp(self, ctx):
        elapsed = ctx.clock() - self.origin_monotonic_s
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


def evidence_origin(evidence):
    """Live evidence has an explicit anchor; historical imports have one legal adapter."""
    if "clock" in evidence:
        value = evidence["clock"]["origin_epoch_s"]
    elif "observation_origin_epoch_s" in evidence:
        value = evidence["observation_origin_epoch_s"]
    else:
        rows = evidence["samples"]
        if not rows:
            raise ValueError("historical evidence lacks a clock anchor and samples")
        value = rows[0]["epoch_s"] - rows[0]["t"]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("evidence origin must be finite epoch seconds")
    return value


def verdict_status(verdict):
    return {"INVALID": "ERROR", "PASS": "PASS", "FAIL": "FAIL"}[verdict]
