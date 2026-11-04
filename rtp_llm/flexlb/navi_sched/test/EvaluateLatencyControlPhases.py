#!/usr/bin/env python3
"""Score latency replay against phases of the master's requested replay QPS."""

import argparse
import bisect
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from EvaluateLatencyReplay import read_trace


CONTROL_UPDATE = re.compile(
    r"new qps \[([^]]+)\].*update time \[(\d+)\]")
PHASES = ("rising", "falling", "steady", "unknown")


def parse_control_update(line):
    match = CONTROL_UPDATE.search(line)
    if not match:
        return None
    qps = float(match.group(1))
    time_us = int(match.group(2))
    if not math.isfinite(qps) or qps <= 0 or time_us <= 0:
        raise ValueError("invalid replay QPS update")
    return time_us, qps


def read_control_updates(paths):
    updates = []
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                update = parse_control_update(line)
                if update:
                    updates.append(update)
    updates.sort()
    if not updates:
        raise ValueError("no replay QPS updates in control logs")
    return updates


def qps_at(time_us, update_times, update_qps, max_age_us):
    index = bisect.bisect_right(update_times, time_us) - 1
    if index < 0 or time_us - update_times[index] > max_age_us:
        return None
    return update_qps[index]


def classify(event_time_us, actual_latency_ms, update_times, update_qps,
             window_us, threshold_qps, max_age_us):
    if not math.isfinite(actual_latency_ms) or actual_latency_ms < 0:
        return "unknown"
    admission_time_us = round(event_time_us - actual_latency_ms * 1000.0)
    current = qps_at(admission_time_us, update_times, update_qps, max_age_us)
    previous = qps_at(admission_time_us - window_us, update_times, update_qps,
                      max_age_us)
    if current is None or previous is None:
        return "unknown"
    change = current - previous
    if change >= threshold_qps:
        return "rising"
    if change <= -threshold_qps:
        return "falling"
    return "steady"


def percentile(values, fraction):
    return sorted(values)[int(fraction * (len(values) - 1))]


def summarize(pairs):
    if not pairs:
        return {"count": 0}
    fixed = [old - actual for actual, old, _ in pairs]
    candidate = [new - actual for actual, _, new in pairs]

    def metrics(errors):
        absolute = [abs(value) for value in errors]
        positive = [value for value in errors if value > 0]
        return {
            "mae_ms": sum(absolute) / len(absolute),
            "bias_ms": sum(errors) / len(errors),
            "p95_absolute_ms": percentile(absolute, 0.95),
            "p95_overprediction_ms": percentile(positive, 0.95)
            if positive else None,
        }

    return {"count": len(pairs), "fixed": metrics(fixed),
            "candidate": metrics(candidate)}


def evaluate(baseline, candidate, updates, window_us, threshold_qps,
             max_age_us, tolerance_ms=0.1):
    update_times = [item[0] for item in updates]
    update_qps = [item[1] for item in updates]
    groups = defaultdict(list)
    coverage = defaultdict(int)
    for key, old in baseline.items():
        architecture = "MI" if key[0].startswith("mi-") or "MI308" in key[0] else "RTX"
        if (not all(math.isfinite(value) for value in old[:5]) or
                abs(old[1] - old[3]) > tolerance_ms or
                abs(old[2] - old[4]) > tolerance_ms):
            coverage[architecture + "_baseline_mismatch"] += 1
            continue
        new = candidate.get(key)
        if new is None or not math.isfinite(new[2]):
            coverage[architecture + "_missing_candidate"] += 1
            continue
        phase = classify(old[6], old[0], update_times, update_qps, window_us,
                         threshold_qps, max_age_us)
        groups[(architecture, phase)].append((old[0], old[2], new[2]))
        coverage[architecture + "_matched"] += 1
    return {
        "control_updates": len(updates),
        "coverage": dict(coverage),
        "phases": {architecture: {
            phase: summarize(groups[(architecture, phase)]) for phase in PHASES}
            for architecture in ("MI", "RTX")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("control_log", type=Path, nargs="+")
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--threshold-qps", type=float, default=10.0)
    parser.add_argument("--max-update-age-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if (not all(math.isfinite(value) and value > 0 for value in
                (args.window_seconds, args.threshold_qps,
                 args.max_update_age_seconds))):
        parser.error("window, threshold, and maximum update age must be positive")
    updates = read_control_updates(args.control_log)
    baseline, _, _ = read_trace(args.baseline)
    candidate, _, _ = read_trace(args.candidate)
    result = evaluate(baseline, candidate, updates,
                      round(args.window_seconds * 1000000), args.threshold_qps,
                      round(args.max_update_age_seconds * 1000000))
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
