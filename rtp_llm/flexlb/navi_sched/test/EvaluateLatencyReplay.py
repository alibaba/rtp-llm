#!/usr/bin/env python3
"""Compare counterfactual replay predictions only where baseline matches logs."""

import argparse
import collections
import json
import math
import statistics
from pathlib import Path


def read_trace(path):
    predictions = {}
    zero_points = {}
    update_drift = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            kind = parts[0]
            if kind not in ("P", "U", "Z"):
                continue
            key = (parts[1], int(parts[4]))
            if kind == "P":
                predictions[key] = (
                    tuple(float(parts[index]) for index in (6, 8, 9, 11, 12)) +
                    (int(parts[7]), int(parts[5]))
                )
            elif kind == "U":
                update_drift[key] = (float(parts[6]), float(parts[7]))
            else:
                zero_points[key] = (float(parts[6]), float(parts[7]), int(parts[5]))
    return predictions, zero_points, update_drift


def read_pressure_samples(directory):
    samples = {}
    for path in Path(directory).glob("*.jsonl"):
        host = path.stem
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("type") != "sample":
                    continue
                features = event.get("features", [])
                if len(features) < 7:
                    raise ValueError(f"missing pressure features: {path}")
                samples[(host, event["sequence"])] = (features[1], features[6])
    return samples


def percentile(values, fraction):
    return sorted(values)[int(fraction * (len(values) - 1))]


def report(label, pairs):
    if not pairs:
        return
    before = [pair[0] for pair in pairs]
    after = [pair[1] for pair in pairs]
    print(label, "n", len(pairs),
          "MAE", round(statistics.mean(before), 3), round(statistics.mean(after), 3),
          "P95", round(percentile(before, .95), 3), round(percentile(after, .95), 3),
          "P99", round(percentile(before, .99), 3), round(percentile(after, .99), 3))


def report_zero(label, pairs):
    before = [pair[0] for pair in pairs]
    after = [pair[1] for pair in pairs]
    drift = [abs(pair[1] - pair[0]) for pair in pairs]
    print("zero_" + label, "n", len(pairs),
          "mean", round(statistics.mean(before), 3), round(statistics.mean(after), 3),
          "P95", round(percentile(before, .95), 3), round(percentile(after, .95), 3),
          "drift_MAE", round(statistics.mean(drift), 3),
          "drift_P95", round(percentile(drift, .95), 3))


def evaluate(baseline_path, candidate_path, tolerance, samples_dir=None,
             before_event_time_us=None, prefix_drift_ms=None):
    baseline, baseline_zero, update_drift = read_trace(baseline_path)
    candidate, candidate_zero, _ = read_trace(candidate_path)
    pressure_samples = read_pressure_samples(samples_dir) if samples_dir else None
    prefix_cutoffs = {}
    if prefix_drift_ms is not None:
        for (host, _), prediction in baseline.items():
            if (abs(prediction[1] - prediction[3]) > prefix_drift_ms or
                    abs(prediction[2] - prediction[4]) > prefix_drift_ms):
                prefix_cutoffs[host] = min(prefix_cutoffs.get(host, prediction[6]),
                                           prediction[6])
    counts = collections.Counter()
    groups = collections.defaultdict(list)
    zeros = collections.defaultdict(list)
    for key, old in baseline.items():
        host, _ = key
        arch = "MI" if "MI308" in host or host.startswith("mi-") else "RTX"
        if before_event_time_us is not None and old[6] >= before_event_time_us:
            counts[arch + "_outside_window"] += 1
            continue
        if old[6] >= prefix_cutoffs.get(host, math.inf):
            counts[arch + "_outside_prefix"] += 1
            continue
        if (not all(math.isfinite(value) for value in old[:5]) or
                abs(old[1] - old[3]) > tolerance or
                abs(old[2] - old[4]) > tolerance):
            counts[arch + "_baseline_mismatch"] += 1
            continue
        new = candidate.get(key)
        if new is None:
            counts[arch + "_missing_candidate"] += 1
            continue
        if not all(math.isfinite(value) for value in new[:5]):
            counts[arch + "_invalid_candidate"] += 1
            continue
        counts[arch + "_matched"] += 1
        pair = (abs(old[2] - old[0]), abs(new[2] - old[0]))
        groups[arch].append(pair)
        groups[arch + ("_idle" if old[5] == 0 else "_queued")].append(pair)
        groups[host].append(pair)
        if pressure_samples is not None:
            pressure = pressure_samples.get(key)
            if pressure is None:
                counts[arch + "_pressure_missing"] += 1
            elif pressure[0] <= 1 and pressure[1] == 0:
                groups[arch + "_low_concurrency"].append(pair)
            elif pressure[0] >= 9:
                groups[arch + "_high_concurrency"].append(pair)
            else:
                groups[arch + "_mid_concurrency"].append(pair)
    for key, old in baseline_zero.items():
        host, _ = key
        arch = "MI" if "MI308" in host or host.startswith("mi-") else "RTX"
        if before_event_time_us is not None and old[2] >= before_event_time_us:
            counts[arch + "_zero_outside_window"] += 1
            continue
        if old[2] >= prefix_cutoffs.get(host, math.inf):
            counts[arch + "_zero_outside_prefix"] += 1
            continue
        drift = update_drift.get(key)
        if drift is None or drift[0] > tolerance or drift[1] > tolerance:
            counts[arch + "_zero_unverified"] += 1
            continue
        new = candidate_zero.get(key)
        if new is None:
            counts[arch + "_zero_missing"] += 1
            continue
        if not all(math.isfinite(value) for value in old + new):
            counts[arch + "_zero_invalid"] += 1
            continue
        counts[arch + "_zero_matched"] += 1
        zeros[arch + "_base"].append((old[0], new[0]))
        zeros[arch + "_final"].append((old[1], new[1]))
    print("coverage", dict(counts))
    if prefix_drift_ms is not None:
        print("prefix_cutoffs_us", dict(sorted(prefix_cutoffs.items())))
    for label, pairs in sorted(groups.items()):
        report(label, pairs)
    for label, pairs in sorted(zeros.items()):
        report_zero(label, pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--baseline-tolerance-ms", type=float, default=0.1)
    parser.add_argument("--samples-dir", type=Path,
                        help="Replay JSONL directory for running/waiting pressure groups")
    parser.add_argument("--before-event-time-us", type=int,
                        help="Exclude events at or after this timestamp")
    parser.add_argument("--prefix-drift-ms", type=float,
                        help="Stop each node at its first larger baseline/log mismatch")
    args = parser.parse_args()
    if not math.isfinite(args.baseline_tolerance_ms) or args.baseline_tolerance_ms < 0:
        parser.error("baseline tolerance must be finite and nonnegative")
    if args.prefix_drift_ms is not None and (not math.isfinite(args.prefix_drift_ms) or
                                             args.prefix_drift_ms < args.baseline_tolerance_ms):
        parser.error("prefix drift must be finite and at least baseline tolerance")
    evaluate(args.baseline, args.candidate, args.baseline_tolerance_ms,
             args.samples_dir, args.before_event_time_us, args.prefix_drift_ms)


if __name__ == "__main__":
    main()
