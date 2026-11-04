#!/usr/bin/env python3
"""Compare baseline and repeated-update replay on causal pressure segments."""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from EvaluateLatencyReplay import read_trace
from RepeatRisingReplaySamples import WINDOW_US, rising_pressure


def sample_segments(directory):
    segments = {}
    for path in directory.glob("*.jsonl"):
        if not (path.stem.startswith("mi-") or "MI308" in path.stem):
            continue
        observed = deque()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event["type"] != "sample":
                    continue
                now = event["event_time_us"]
                features = event["features"]
                observed.append((now, features[6]))
                while observed and observed[0][0] < now - 2 * WINDOW_US:
                    observed.popleft()
                low = features[1] <= 4 and features[6] <= 4
                high = features[6] >= 33
                rising = high and rising_pressure(observed, now, 8)
                segments[(path.stem, event["sequence"])] = (
                    "low" if low else "high_rising" if rising else
                    "high_other" if high else "middle")
    return segments


def summarize(rows):
    if not rows:
        return {"n": 0}
    values = np.asarray(rows)
    actual, old_base, new_base, old_final, new_final = values.T
    result = {"n": len(rows)}
    for name, predicted in (("base_before", old_base), ("base_after", new_base),
                            ("final_before", old_final), ("final_after", new_final)):
        error = actual - predicted
        result[name] = {
            "mae_ms": float(np.mean(np.abs(error))),
            "bias_ms": float(np.mean(error)),
            "p95_abs_ms": float(np.percentile(np.abs(error), 95)),
            "relative100": float(np.mean(np.abs(error) / np.maximum(actual, 100.0))),
        }
    return result


def evaluate(baseline_path, candidate_path, samples_dir, prefix_only=False):
    baseline, baseline_zero, update_drift = read_trace(baseline_path)
    candidate, candidate_zero, _ = read_trace(candidate_path)
    segments = sample_segments(samples_dir)
    cutoffs = {}
    if prefix_only:
        for (host, _), old in baseline.items():
            if abs(old[1] - old[3]) > 0.1 or abs(old[2] - old[4]) > 0.1:
                cutoffs[host] = min(cutoffs.get(host, old[6]), old[6])
    groups = defaultdict(list)
    skipped = {"baseline_mismatch": 0, "missing_candidate": 0,
               "after_first_mismatch": 0}
    for key, old in baseline.items():
        if key not in segments:
            continue
        if old[6] >= cutoffs.get(key[0], float("inf")):
            skipped["after_first_mismatch"] += 1
            continue
        if abs(old[1] - old[3]) > 0.1 or abs(old[2] - old[4]) > 0.1:
            skipped["baseline_mismatch"] += 1
            continue
        new = candidate.get(key)
        if new is None:
            skipped["missing_candidate"] += 1
            continue
        row = (old[0], old[1], new[1], old[2], new[2])
        groups["all"].append(row)
        groups[segments[key]].append(row)
        groups["host:" + key[0]].append(row)
        groups["host:" + key[0] + ":" + segments[key]].append(row)
    zero = []
    sample_hosts = {host for host, _ in segments}
    for key, old in baseline_zero.items():
        drift = update_drift.get(key)
        if (key not in candidate_zero or key[0] not in sample_hosts or
                old[2] >= cutoffs.get(key[0], float("inf")) or
                drift is None or drift[0] > 0.1 or drift[1] > 0.1):
            continue
        new = candidate_zero[key]
        zero.append((old[0], new[0], old[1], new[1]))
    zero_array = np.asarray(zero) if zero else np.empty((0, 4))
    zero_summary = {"n": len(zero),
                    "base_drift_mae_ms": float(np.mean(np.abs(
                        zero_array[:, 1] - zero_array[:, 0]))) if zero else None,
                    "base_drift_p95_ms": float(np.percentile(np.abs(
                        zero_array[:, 1] - zero_array[:, 0]), 95)) if zero else None,
                    "final_drift_mae_ms": float(np.mean(np.abs(
                        zero_array[:, 3] - zero_array[:, 2]))) if zero else None,
                    "final_drift_p95_ms": float(np.percentile(np.abs(
                        zero_array[:, 3] - zero_array[:, 2]), 95)) if zero else None}
    return {"coverage": skipped, "prefix_cutoffs_us": cutoffs,
            "segments": {name: summarize(rows) for name, rows in groups.items()},
            "zero": zero_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("samples_dir", type=Path)
    parser.add_argument("--prefix-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.baseline, args.candidate, args.samples_dir,
                              args.prefix_only),
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
