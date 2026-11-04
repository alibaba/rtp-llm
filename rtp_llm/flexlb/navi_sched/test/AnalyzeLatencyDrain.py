#!/usr/bin/env python3
"""Compare original Master predictions across an arrival-only replay cutoff.

Ready time is inferred from completion time minus the logged execution target.
This is an observational comparison, not a replay of a counterfactual scheduler.
All active non-PD model inputs participate in neighbor matching.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from AnalyzeLatencyFeatureNeighbors import (
    FEATURES, compact_sample, parse_time, read_samples, relative_differences,
)


def ready_us(row):
    return row["event_time_us"] - row["target_ms"] * 1000


def summary(rows):
    if not rows:
        return {"count": 0}
    actual = np.array([r["target_ms"] for r in rows], dtype=float)
    predicted = np.array([r["master_prediction_ms"] for r in rows], dtype=float)
    error = predicted - actual
    return {
        "count": len(rows),
        "actual_mean_ms": float(actual.mean()),
        "master_mean_ms": float(predicted.mean()),
        "signed_error_mean_ms": float(error.mean()),
        "signed_error_median_ms": float(np.median(error)),
        "mae_ms": float(np.abs(error).mean()),
        "mape_pct": float((np.abs(error) / np.maximum(actual, 1)).mean() * 100),
        "overprediction_fraction": float((error > 0).mean()),
        "queue_mean_ms": float(np.mean([r["backend_queue_ms"] for r in rows])),
        "features_mean": {k: float(np.mean([r[k] for r in rows])) for k in FEATURES},
    }


def analyze(rows, steady_start_us, cut_us, relative_guard=0.05, distance_limit=0.2):
    if steady_start_us >= cut_us - 240_000_000:
        raise ValueError("need at least four minutes before cutoff")
    if not 0 <= relative_guard <= 1 or distance_limit < 0:
        raise ValueError("invalid matching limits")
    controls = [r for r in rows if
                steady_start_us + 180_000_000 <= ready_us(r) < cut_us - 120_000_000
                and r["event_time_us"] < cut_us - 60_000_000]
    tail = [r for r in rows if cut_us - 60_000_000 <= ready_us(r) <= cut_us
            and r["event_time_us"] > cut_us + 2_000_000]
    intervals = [(-120, -60), (-60, -30), (-30, -10), (-10, 0)]
    groups = {f"{low}:{high}": summary([
        r for r in rows if cut_us + low * 1e6 <= ready_us(r) < cut_us + high * 1e6
    ]) for low, high in intervals}
    pairs = []
    if controls and tail:
        x = np.asarray([[r[k] for k in FEATURES] for r in controls])
        scale = np.std(x, axis=0)
        scale[scale == 0] = 1.0
        for r in tail:
            y = np.array([r[k] for k in FEATURES])
            rel = relative_differences(x, y).max(axis=1)
            distance = np.linalg.norm((x - y) / scale, axis=1)
            distance[rel > relative_guard] = np.inf
            index = int(np.argmin(distance))
            if distance[index] > distance_limit:
                continue
            c = controls[index]
            pairs.append({
                "control": compact_sample(c), "tail": compact_sample(r),
                "distance": float(distance[index]), "max_relative_difference": float(rel[index]),
                "actual_change_ms": r["target_ms"] - c["target_ms"],
                "signed_error_change_ms":
                    (r["master_prediction_ms"] - r["target_ms"])
                    - (c["master_prediction_ms"] - c["target_ms"]),
            })
    return {
        "steady": summary(controls), "tail_finishing_after_cut": summary(tail),
        "ready_seconds_relative_to_cut": groups,
        "matching": {"features": FEATURES, "relative_guard": relative_guard,
                     "distance_limit": distance_limit,
                     "scale_population": "steady controls only",
                     "pair_count": len(pairs),
                     "unique_controls": len({p["control"]["sequence"] for p in pairs}),
                     "examples": pairs[:20]},
        "samples_ready_after_cut_plus_2s": sum(ready_us(r) > cut_us + 2e6 for r in rows),
        "last_completion_us": max((r["event_time_us"] for r in rows), default=None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--steady-start", required=True)
    parser.add_argument("--cut", required=True)
    parser.add_argument("--relative-guard", type=float, default=0.05)
    parser.add_argument("--distance-limit", type=float, default=0.2)
    args = parser.parse_args()
    start, cut = parse_time(args.steady_start), parse_time(args.cut)
    print(json.dumps({
        "steady_start_us": start, "cut_us": cut,
        "error_sign": "prediction - actual; positive means overprediction",
        "ready_time": "completion - target (Agent admission, not Master ingress)",
        "nodes": {str(p): analyze(read_samples(p), start, cut,
                                  args.relative_guard, args.distance_limit) for p in args.logs},
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
