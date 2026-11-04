#!/usr/bin/env python3
"""Audit same-host, non-PD latency samples for conflicting feature neighbors.

Reads one process lifetime's training log, writes a bounded JSON report to stdout.
Distances support standardized Euclidean and per-feature relative maximum.
This is a diagnostic, not a proof of causality or an online counterfactual.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import gzip
import json
from pathlib import Path
import re

import numpy as np


FEATURES = (
    "running_batch_size", "running_hit_tokens", "running_miss_tokens",
    "running_miss_square_sum", "running_hit_miss_sum", "waiting_batch_size",
    "waiting_hit_tokens", "waiting_miss_tokens", "waiting_miss_square_sum",
    "waiting_hit_miss_sum", "preceding_request_count", "running_elapsed_time_ms",
)
NUMERIC = re.compile(r"\b([a-zA-Z_][a-zA-Z_0-9]*)=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b")
CST = timezone(timedelta(hours=8))


def read_samples(path):
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    seen = set()
    with opener(path, "rt") as stream:
        for line in stream:
            if "[navi-latency-training-sample]" not in line:
                continue
            row = {key: float(value) if any(c in value for c in ".eE") else int(value)
                   for key, value in NUMERIC.findall(line)}
            for key in (*FEATURES, "sequence", "batch_id", "event_time_us", "target_ms"):
                if key not in row:
                    raise ValueError("sample missing " + key)
            if row.get("capacity_split") != 1 or row.get("unified_sample") != 1:
                raise ValueError("only unified capacity-split samples are supported")
            if row["sequence"] in seen:
                raise ValueError("duplicate sequence: split process lifetimes before analysis")
            values = np.asarray([row[key] for key in FEATURES] + [row["target_ms"]])
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("nonfinite or negative sample")
            seen.add(row["sequence"])
            rows.append(row)
    return sorted(rows, key=lambda row: row["event_time_us"])


def relative_differences(values, reference):
    return np.abs(values - reference) / np.maximum(1.0, np.maximum(np.abs(values), np.abs(reference)))


def compact_sample(row):
    keys = ("sequence", "batch_id", "event_time_us", "target_ms", "master_prediction_ms",
            "base_prediction_before_ms", "prediction_before_ms", "backend_queue_ms",
            "nonlinear_u_before", "nonlinear_r_before", "revision_before")
    result = {key: row[key] for key in keys if key in row}
    result["finish_cst"] = datetime.fromtimestamp(row["event_time_us"] / 1e6, CST).isoformat()
    result["inferred_ready_cst"] = datetime.fromtimestamp(
        row["event_time_us"] / 1e6 - row["target_ms"] / 1e3, CST).isoformat()
    result["features"] = {key: row[key] for key in FEATURES}
    return result


def pressure_phases(rows, window_s=30, flat_ratio=0.05):
    """Classify local pressure using only earlier inferred admissions.

    This is the trend of preceding_request_count, not the command waveform or
    the latency target. A missing historical window is explicitly unknown.
    """
    if window_s <= 0 or not 0 <= flat_ratio < 1:
        raise ValueError("invalid pressure trend settings")
    ready = np.asarray([r["event_time_us"] - r["target_ms"] * 1000 for r in rows])
    pressure = np.asarray([r["preceding_request_count"] for r in rows], dtype=float)
    result = []
    for i, current in enumerate(ready):
        earlier = pressure[(ready >= current - window_s * 1e6) &
                           (ready <= current - window_s * 0.5e6)]
        if len(earlier) < 3:
            result.append({"phase": "unknown", "reference_count": len(earlier)})
            continue
        baseline = float(np.median(earlier))
        change = (pressure[i] - baseline) / max(1.0, abs(baseline), abs(pressure[i]))
        phase = "rising" if change > flat_ratio else "falling" if change < -flat_ratio else "flat"
        result.append({"phase": phase, "reference_count": len(earlier),
                       "reference_pressure": baseline, "current_pressure": float(pressure[i]),
                       "relative_change": float(change)})
    return result


def analyze(rows, focus_start_us, focus_end_us, thresholds=(0.02, 0.05, 0.1, 0.2),
            min_separation_s=60, limit=10, metric="standardized_l2",
            max_relative_difference=None, opposite_pressure_phase=False,
            pressure_window_s=30, pressure_flat_ratio=0.05):
    if not rows:
        raise ValueError("no samples")
    x = np.asarray([[row[key] for key in FEATURES] for row in rows], dtype=float)
    times = np.asarray([row["event_time_us"] for row in rows], dtype=np.int64)
    ready_times = times - np.asarray([row["target_ms"] * 1000 for row in rows])
    focus = np.flatnonzero((times >= focus_start_us) & (times < focus_end_us))
    focus_set = set(focus)
    phases = pressure_phases(rows, pressure_window_s, pressure_flat_ratio)
    phase_names = np.asarray([p["phase"] for p in phases])
    scales = np.std(x, axis=0)
    scales[scales == 0] = 1.0
    if metric not in ("relative_max", "standardized_l2"):
        raise ValueError("unknown distance metric")
    pairs = {threshold: {} for threshold in thresholds}
    nearest = []
    nearest_pairs = []
    for i in focus:
        relative = relative_differences(x, x[i]).max(axis=1)
        distance = (relative
                    if metric == "relative_max" else np.linalg.norm((x - x[i]) / scales, axis=1))
        if max_relative_difference is not None:
            distance[relative > max_relative_difference] = np.inf
        # Exclude neighboring completions and neighboring admissions separately.
        distance[(np.abs(times - times[i]) < min_separation_s * 1e6) |
                 (np.abs(ready_times - ready_times[i]) < min_separation_s * 1e6)] = np.inf
        if opposite_pressure_phase:
            opposite = {"rising": "falling", "falling": "rising"}.get(phase_names[i])
            distance[phase_names != opposite] = np.inf
        distance[i] = np.inf
        best = int(np.argmin(distance))
        if np.isfinite(distance[best]):
            nearest.append(float(distance[best]))
            nearest_pairs.append((int(i), best, float(distance[best])))
        for threshold in thresholds:
            for j in np.flatnonzero(distance <= threshold):
                pair = tuple(sorted((int(i), int(j))))
                pairs[threshold][pair] = float(distance[j])
    output = {}
    for threshold, found in pairs.items():
        severe = []
        ratios, gaps, matched = [], [], set()
        for (i, j), distance in found.items():
            left, right = rows[i], rows[j]
            gap = abs(left["target_ms"] - right["target_ms"])
            ratio = max(left["target_ms"], right["target_ms"]) / max(1.0, min(left["target_ms"], right["target_ms"]))
            ratios.append(ratio)
            gaps.append(gap)
            matched.update((i, j))
            if ratio >= 1.5 and gap >= 5000:
                severe.append((gap, ratio, distance, i, j))
        examples = []
        used = set()
        disjoint_count = 0
        for gap, ratio, distance, i, j in sorted(severe, reverse=True):
            if i in used or j in used:
                continue
            used.update((i, j))
            disjoint_count += 1
            if len(examples) >= limit:
                continue
            examples.append({"target_gap_ms": gap, "target_ratio": ratio,
                             "vector_distance": distance,
                             "max_relative_feature_difference": float(
                                 relative_differences(x[i], x[j]).max()),
                             "per_feature_relative_difference": dict(zip(
                                 FEATURES, relative_differences(x[i], x[j]).tolist())),
                             "inferred_admission_separation_s": float(abs(ready_times[i] - ready_times[j]) / 1e6),
                             "left_pressure_phase": phases[i], "right_pressure_phase": phases[j],
                             "left": compact_sample(rows[i]), "right": compact_sample(rows[j])})
        severe_focus = {i for item in severe for i in item[-2:] if i in focus_set}
        near = [(i, j) for i, j, distance in nearest_pairs if distance <= threshold]
        nearest_severe_count = sum(
            abs(rows[i]["target_ms"] - rows[j]["target_ms"]) >= 5000 and
            max(rows[i]["target_ms"], rows[j]["target_ms"]) >=
            1.5 * max(1.0, min(rows[i]["target_ms"], rows[j]["target_ms"]))
            for i, j in near)
        output[str(threshold)] = {
            "pair_count": len(found), "focus_samples_with_neighbor": len(focus_set & matched),
            "severe_pair_count": len(severe), "focus_samples_with_severe_neighbor": len(severe_focus),
            "nearest_neighbor_samples": len(near), "nearest_neighbor_severe_samples": nearest_severe_count,
            "greedy_disjoint_severe_pair_count": disjoint_count,
            "target_gap_median_ms": float(np.median(gaps)) if gaps else None,
            "target_gap_p95_ms": float(np.percentile(gaps, 95)) if gaps else None,
            "target_ratio_median": float(np.median(ratios)) if ratios else None,
            "examples": examples,
        }
    minutes = defaultdict(list)
    for row in rows:
        if focus_start_us <= row["event_time_us"] < focus_end_us:
            minute = datetime.fromtimestamp(row["event_time_us"] / 1e6, CST).strftime("%Y-%m-%d %H:%M")
            minutes[minute].append(row)
    summaries = {}
    for minute, samples in minutes.items():
        keys = ("target_ms", "master_prediction_ms", "prediction_before_ms",
                "base_prediction_before_ms", "backend_queue_ms", "revision_before")
        summaries[minute] = {"count": len(samples), **{
            key: float(np.mean([row[key] for row in samples])) for key in keys
            if all(key in row for row in samples)}}
    return {"sample_count": len(rows), "focus_count": len(focus), "features": FEATURES,
            "metric": metric,
            "distance": ("max(abs(a-b)/max(1,abs(a),abs(b)))" if metric == "relative_max"
                         else "sqrt(sum(((a-b)/std(feature))**2)); scale from comparison corpus"),
            "feature_std": dict(zip(FEATURES, scales.tolist())),
            "max_relative_difference_guard": max_relative_difference,
            "examples_selection": "descending target gap, no sample reused within a threshold",
            "non_pd_batch_count_feature": "disabled, excluded from distance",
            "min_finish_and_ready_separation_s": min_separation_s,
            "opposite_pressure_phase_required": opposite_pressure_phase,
            "pressure_phase_definition": "preceding_request_count vs median of earlier admissions 0.5-1 window ago",
            "pressure_window_s": pressure_window_s, "pressure_flat_ratio": pressure_flat_ratio,
            "independence": "time separation alone does not exclude shared long-lived decode predecessors",
            "severe_definition": "target ratio >=1.5 AND absolute target gap >=5000ms",
            "nearest_distance_median": float(np.median(nearest)) if nearest else None,
            "thresholds": output, "finish_minute_summary": summaries}


def parse_time(text):
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError("time must include timezone")
    return int(value.timestamp() * 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--focus-start", required=True)
    parser.add_argument("--focus-end", required=True)
    parser.add_argument("--min-separation-s", type=float, default=60)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--metric", choices=("relative_max", "standardized_l2"),
                        default="standardized_l2")
    parser.add_argument("--history-start", help="optional inclusive completion-time bound")
    parser.add_argument("--history-end", help="optional exclusive completion-time bound")
    parser.add_argument("--max-relative-difference", type=float,
                        help="optional per-component guard in addition to vector distance")
    parser.add_argument("--opposite-pressure-phase", action="store_true",
                        help="only pair rising and falling local preceding-request counts")
    parser.add_argument("--pressure-window-s", type=float, default=30)
    parser.add_argument("--pressure-flat-ratio", type=float, default=0.05)
    args = parser.parse_args()
    start, end = parse_time(args.focus_start), parse_time(args.focus_end)
    if end <= start or args.min_separation_s <= 0 or not 0 <= args.limit <= 100:
        parser.error("invalid interval, separation or example limit")
    rows = read_samples(args.log)
    if args.history_start:
        history_start = parse_time(args.history_start)
        rows = [row for row in rows if row["event_time_us"] >= history_start]
    if args.history_end:
        history_end = parse_time(args.history_end)
        rows = [row for row in rows if row["event_time_us"] < history_end]
    if args.max_relative_difference is not None and not 0 <= args.max_relative_difference <= 1:
        parser.error("relative-difference guard must be in [0,1]")
    if args.pressure_window_s <= 0 or not 0 <= args.pressure_flat_ratio < 1:
        parser.error("invalid pressure trend settings")
    result = analyze(rows, start, end, min_separation_s=args.min_separation_s,
                     limit=args.limit, metric=args.metric,
                     max_relative_difference=args.max_relative_difference,
                     opposite_pressure_phase=args.opposite_pressure_phase,
                     pressure_window_s=args.pressure_window_s,
                     pressure_flat_ratio=args.pressure_flat_ratio)
    print(json.dumps({"input": str(args.log), **result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
