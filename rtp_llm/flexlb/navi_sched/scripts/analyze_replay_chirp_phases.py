#!/usr/bin/env python3
"""Compare paired replay predictions by intended chirp pressure and direction."""

import argparse
import collections
import json
import math
from pathlib import Path

from drive_replay_chirp import wave


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def summarize(pairs):
    def metrics(errors):
        positive = [error for error in errors if error > 0]
        return {
            "mae_ms": sum(map(abs, errors)) / len(errors),
            "bias_ms": sum(errors) / len(errors),
            "over_p95_ms": percentile(positive, 0.95),
        }

    baseline = [old - target for target, old, _ in pairs]
    candidate = [new - target for target, _, new in pairs]
    return {"count": len(pairs), "baseline": metrics(baseline),
            "candidate": metrics(candidate)}


def read_predictions(path, start_us, end_us, check_online):
    predictions = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if fields[0] != "P":
                continue
            target = float(fields[6])
            ready_us = int(fields[5]) - round(target * 1000)
            if not start_us <= ready_us < end_us:
                continue
            predicted = float(fields[9])
            if check_online and abs(predicted - float(fields[12])) > 0.1:
                raise ValueError("baseline differs from online at " + fields[1] +
                                 ":" + fields[4])
            key = (fields[1], int(fields[4]))
            if key in predictions:
                raise ValueError("duplicate prediction: " + str(key))
            predictions[key] = (ready_us, target, predicted)
    return predictions


def classify(ready_us, start_us, minimum, maximum, base_period, fast_period,
             warmup_cycles, sweep_duration, lookback_seconds, threshold_qps):
    elapsed = warmup_cycles * base_period + (ready_us - start_us) / 1e6
    qps, _, _ = wave(elapsed, minimum, maximum, base_period, fast_period,
                     warmup_cycles, sweep_duration)
    earlier, _, _ = wave(elapsed - lookback_seconds, minimum, maximum,
                         base_period, fast_period, warmup_cycles, sweep_duration)
    delta = qps - earlier
    direction = ("rising" if delta > threshold_qps else
                 "falling" if delta < -threshold_qps else "turn")
    pressure = "low" if qps < 100 else "mid" if qps < 250 else "high"
    return pressure, direction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--end-us", type=int, required=True)
    parser.add_argument("--min-qps", type=float, default=0.0)
    parser.add_argument("--max-qps", type=float, default=400.0)
    parser.add_argument("--base-period-seconds", type=float, default=300.0)
    parser.add_argument("--fast-period-seconds", type=float, default=45.0)
    parser.add_argument("--warmup-cycles", type=int, default=3)
    parser.add_argument("--sweep-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--lookback-seconds", type=float, default=5.0)
    parser.add_argument("--threshold-qps", type=float, default=5.0)
    args = parser.parse_args()
    if (args.end_us <= args.start_us or args.min_qps < 0 or
            args.max_qps <= args.min_qps or args.warmup_cycles < 1 or
            any(not math.isfinite(value) or value <= 0 for value in
                (args.base_period_seconds, args.fast_period_seconds,
                 args.sweep_duration_seconds, args.lookback_seconds,
                 args.threshold_qps)) or
            args.fast_period_seconds >= args.base_period_seconds):
        parser.error("invalid chirp parameters")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    host_arch = {}
    for node in manifest["nodes"]:
        deployment = node["deployment"].upper()
        host = node["ip"].replace(".", "_")
        if "MI308" in deployment:
            host_arch[host] = "MI"
        elif "RTX_PRO_5000" in deployment:
            host_arch[host] = "RTX"
    baseline = read_predictions(args.baseline, args.start_us, args.end_us, True)
    candidate = read_predictions(args.candidate, args.start_us, args.end_us, False)
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate prediction keys differ")
    groups = collections.defaultdict(list)
    for (host, sequence), (ready_us, target, old) in baseline.items():
        if host not in host_arch:
            raise ValueError("unknown replay host: " + host)
        new_ready_us, new_target, new = candidate[(host, sequence)]
        if new_ready_us != ready_us or abs(new_target - target) > 1e-6:
            raise ValueError("candidate event differs from baseline")
        pressure, direction = classify(
            ready_us, args.start_us, args.min_qps, args.max_qps,
            args.base_period_seconds, args.fast_period_seconds,
            args.warmup_cycles, args.sweep_duration_seconds,
            args.lookback_seconds, args.threshold_qps)
        arch = host_arch[host]
        for key in (arch, f"{arch}:{pressure}", f"{arch}:{direction}",
                    f"{arch}:{pressure}:{direction}"):
            groups[key].append((target, old, new))
    print(json.dumps({"count": len(baseline), "basis": "intended_qps",
                      "groups": {key: summarize(pairs)
                                 for key, pairs in sorted(groups.items())}},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
