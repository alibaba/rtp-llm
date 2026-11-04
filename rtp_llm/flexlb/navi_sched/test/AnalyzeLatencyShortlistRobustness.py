#!/usr/bin/env python3
"""Audit paired latency candidates by replay phase, host, and cycle."""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


CONTROL_UPDATE = re.compile(
    r"new qps \[([^]]+)\].*update time \[(\d+)\]"
)


def architecture(host):
    lowered = host.lower()
    return "MI308" if "mi308" in lowered or lowered.startswith("mi-") else "RTX5000"


def target_qps(elapsed_seconds, minimum, maximum, period_seconds):
    midpoint = (minimum + maximum) / 2.0
    amplitude = (maximum - minimum) / 2.0
    return midpoint - amplitude * math.cos(
        2.0 * math.pi * elapsed_seconds / period_seconds
    )


def read_control_updates(paths):
    updates = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                match = CONTROL_UPDATE.search(line)
                if not match:
                    continue
                qps = float(match.group(1))
                time_us = int(match.group(2))
                if not math.isfinite(qps) or qps <= 0.0 or time_us <= 0:
                    raise ValueError("invalid replay control update")
                updates.append((time_us, qps))
    updates.sort()
    if not updates:
        raise ValueError("no replay control updates")
    return updates


def fit_control_origin(updates, approximate_start_us, duration_seconds,
                       minimum_qps, maximum_qps, period_seconds,
                       step_seconds=1.0):
    margin_us = round(max(10.0, step_seconds * 3.0) * 1_000_000)
    end_us = approximate_start_us + round(duration_seconds * 1_000_000)
    selected = [
        item for item in updates
        if approximate_start_us - margin_us <= item[0] <= end_us + margin_us
    ]
    if len(selected) < 3:
        raise ValueError("insufficient replay control updates in experiment window")

    midpoint = (minimum_qps + maximum_qps) / 2.0
    amplitude = (maximum_qps - minimum_qps) / 2.0
    angular = 2.0 * math.pi / (period_seconds * 1_000_000.0)
    first_time = selected[0][0]
    phase = np.asarray([(time_us - first_time) * angular
                        for time_us, _ in selected])
    normalized = np.asarray([(qps - midpoint) / amplitude
                             for _, qps in selected])
    design = np.column_stack((np.cos(phase), np.sin(phase)))
    cosine, sine = np.linalg.lstsq(design, normalized, rcond=None)[0]
    relative_origin = math.atan2(-sine, -cosine) / angular
    period_us = period_seconds * 1_000_000.0
    relative_origin -= round(relative_origin / period_us) * period_us
    initial_origin = first_time + relative_origin

    step_us = step_seconds * 1_000_000.0
    ticks = np.rint(
        (np.asarray([item[0] for item in selected]) - initial_origin) / step_us
    ).astype(np.int64)
    origins = np.asarray([item[0] for item in selected]) - ticks * step_us
    origin_us = float(np.median(origins))
    ticks = np.rint(
        (np.asarray([item[0] for item in selected]) - origin_us) / step_us
    ).astype(np.int64)

    expected_last_tick = math.ceil(duration_seconds / step_seconds) - 1
    expected_ticks = set(range(1, expected_last_tick + 1))
    observed_ticks = set(int(value) for value in ticks)
    qps_errors = [
        abs(qps - target_qps(tick * step_seconds, minimum_qps,
                             maximum_qps, period_seconds))
        for tick, (_, qps) in zip(ticks, selected)
    ]
    timing_errors = [
        time_us - (origin_us + tick * step_us)
        for tick, (time_us, _) in zip(ticks, selected)
    ]
    gaps = [right[0] - left[0]
            for left, right in zip(selected, selected[1:])]
    return {
        "origin_us": round(origin_us),
        "origin_offset_from_approximate_start_ms": (
            origin_us - approximate_start_us
        ) / 1000.0,
        "update_count": len(selected),
        "first_tick": int(np.min(ticks)),
        "last_tick": int(np.max(ticks)),
        "missing_ticks": sorted(expected_ticks - observed_ticks),
        "unexpected_ticks": sorted(observed_ticks - expected_ticks),
        "duplicate_tick_count": len(ticks) - len(observed_ticks),
        "target_qps_max_abs_error": max(qps_errors),
        "timing_jitter_p95_ms": float(np.quantile(np.abs(timing_errors), 0.95)) / 1000.0,
        "timing_jitter_max_ms": max(map(abs, timing_errors)) / 1000.0,
        "maximum_update_gap_ms": max(gaps) / 1000.0,
        "updates_with_gap_over_two_steps": sum(
            gap > 2.0 * step_us for gap in gaps
        ),
    }


def read_trace(path, start_us, end_us, origin_us, period_seconds,
               phase_seconds):
    records = {}
    period_us = round(period_seconds * 1_000_000)
    phase_us = round(phase_seconds * 1_000_000)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if fields[0] != "P":
                continue
            host = fields[1]
            sequence = int(fields[4])
            event_time_us = int(fields[5])
            target_ms = float(fields[6])
            ready_time_us = event_time_us - round(target_ms * 1000.0)
            if not start_us <= ready_time_us < end_us:
                continue
            if not all(math.isfinite(float(fields[index]))
                       for index in (6, 8, 9, 11, 12)):
                raise ValueError(f"non-finite prediction in {path}: {host}/{sequence}")
            key = (host, sequence)
            if key in records:
                raise ValueError(f"duplicate prediction in {path}: {host}/{sequence}")
            position_us = (ready_time_us - origin_us) % period_us
            records[key] = {
                "host": host,
                "architecture": architecture(host),
                "sequence": sequence,
                "event_time_us": event_time_us,
                "ready_time_us": ready_time_us,
                "target_ms": target_ms,
                "base_ms": float(fields[8]),
                "prediction_ms": float(fields[9]),
                "logged_base_ms": float(fields[11]),
                "logged_prediction_ms": float(fields[12]),
                "cycle": (ready_time_us - origin_us) // period_us,
                "phase_bin": position_us // phase_us,
                "trend": "rising" if position_us < period_us / 2 else "falling",
            }
    return records


def percentile(values, fraction):
    return float(np.quantile(values, fraction)) if values else None


def error_metrics(rows, prediction_field):
    if not rows:
        return {"count": 0}
    target = np.asarray([row["target_ms"] for row in rows], dtype=np.float64)
    prediction = np.asarray(
        [row[prediction_field] for row in rows], dtype=np.float64
    )
    error = prediction - target
    absolute = np.abs(error)
    return {
        "count": len(rows),
        "mae_ms": float(np.mean(absolute)),
        "wape_pct": float(np.sum(absolute) / np.sum(target) * 100.0),
        "bias_ms": float(np.mean(error)),
        "p95_abs_ms": percentile(absolute.tolist(), 0.95),
        "underprediction_mae_ms": float(np.mean(np.maximum(-error, 0.0))),
        "overprediction_mae_ms": float(np.mean(np.maximum(error, 0.0))),
    }


def paired_metrics(rows):
    baseline = error_metrics(rows, "baseline_prediction_ms")
    candidate = error_metrics(rows, "candidate_prediction_ms")
    if not rows:
        return {"count": 0}
    delta = candidate["mae_ms"] - baseline["mae_ms"]
    return {
        "count": len(rows),
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline_mae_ms": delta,
        "mae_change_pct": delta / baseline["mae_ms"] * 100.0,
    }


def pair_candidate(baseline, candidate, tolerance_ms=0.1):
    coverage = defaultdict(int)
    rows = []
    for key, old in baseline.items():
        arch = old["architecture"]
        if (abs(old["base_ms"] - old["logged_base_ms"]) > tolerance_ms or
                abs(old["prediction_ms"] - old["logged_prediction_ms"]) > tolerance_ms):
            coverage[arch + "_baseline_log_mismatch"] += 1
            continue
        new = candidate.get(key)
        if new is None:
            coverage[arch + "_missing_candidate"] += 1
            continue
        if (new["event_time_us"] != old["event_time_us"] or
                abs(new["target_ms"] - old["target_ms"]) > 1e-9):
            coverage[arch + "_candidate_identity_mismatch"] += 1
            continue
        row = dict(old)
        row["baseline_prediction_ms"] = old["prediction_ms"]
        row["candidate_prediction_ms"] = new["prediction_ms"]
        rows.append(row)
        coverage[arch + "_matched"] += 1
    baseline_keys = set(baseline)
    for key, row in candidate.items():
        if key not in baseline_keys:
            coverage[row["architecture"] + "_extra_candidate"] += 1
    return rows, dict(sorted(coverage.items()))


def block_bootstrap(rows, repetitions=20000, seed=20261011):
    blocks = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    for row in rows:
        key = (row["host"], row["cycle"])
        blocks[key] += (
            1.0,
            abs(row["baseline_prediction_ms"] - row["target_ms"]),
            abs(row["candidate_prediction_ms"] - row["target_ms"]),
        )
    if not blocks:
        return {"block_count": 0, "sample_count": 0}
    values = np.asarray([blocks[key] for key in sorted(blocks)], dtype=np.float64)
    rng = np.random.RandomState(seed)
    selections = rng.randint(0, len(values), size=(repetitions, len(values)))
    sampled = np.sum(values[selections], axis=1)
    differences = (sampled[:, 2] - sampled[:, 1]) / sampled[:, 0]
    observed = (np.sum(values[:, 2]) - np.sum(values[:, 1])) / np.sum(values[:, 0])
    return {
        "block_count": len(values),
        "sample_count": int(np.sum(values[:, 0])),
        "candidate_minus_baseline_mae_ms": float(observed),
        "confidence_95_ms": np.quantile(differences, [0.025, 0.975]).tolist(),
        "probability_candidate_improves": float(np.mean(differences < 0.0)),
    }


def group_report(rows, repetitions, seed):
    hosts = defaultdict(list)
    cycles = defaultdict(list)
    phases = defaultdict(list)
    trends = defaultdict(list)
    for row in rows:
        hosts[row["host"]].append(row)
        cycles[str(row["cycle"])].append(row)
        phases[str(row["phase_bin"])].append(row)
        trends[row["trend"]].append(row)
    host_metrics = {key: paired_metrics(value) for key, value in sorted(hosts.items())}
    cycle_metrics = {key: paired_metrics(value)
                     for key, value in sorted(cycles.items(), key=lambda item: int(item[0]))}
    return {
        "all": paired_metrics(rows),
        "trends": {key: paired_metrics(trends[key])
                   for key in ("rising", "falling")},
        "phase_bins": {key: paired_metrics(value)
                       for key, value in sorted(phases.items(), key=lambda item: int(item[0]))},
        "hosts": host_metrics,
        "cycles": cycle_metrics,
        "improved_hosts": sum(value["mae_change_pct"] < 0.0
                              for value in host_metrics.values()),
        "total_hosts": len(host_metrics),
        "improved_cycles": sum(value["mae_change_pct"] < 0.0
                               for value in cycle_metrics.values()),
        "total_cycles": len(cycle_metrics),
        "worst_host_change_pct": max(
            (value["mae_change_pct"] for value in host_metrics.values()),
            default=None,
        ),
        "worst_cycle_change_pct": max(
            (value["mae_change_pct"] for value in cycle_metrics.values()),
            default=None,
        ),
        "host_cycle_bootstrap": block_bootstrap(rows, repetitions, seed),
    }


def analyze_candidate(baseline, candidate, repetitions, seed):
    rows, coverage = pair_candidate(baseline, candidate)
    architectures = {}
    for arch in ("MI308", "RTX5000"):
        selected = [row for row in rows if row["architecture"] == arch]
        excluding_first_two = [row for row in selected if row["cycle"] not in (0, 1)]
        architectures[arch] = {
            "full": group_report(selected, repetitions, seed),
            "excluding_cycles_0_1": group_report(
                excluding_first_two, repetitions, seed + 1
            ),
        }
    return {"coverage": coverage, "architectures": architectures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--control-log", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approximate-start-us", type=int, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--min-qps", type=float, required=True)
    parser.add_argument("--max-qps", type=float, required=True)
    parser.add_argument("--period-seconds", type=float, required=True)
    parser.add_argument("--phase-seconds", type=float, default=30.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    args = parser.parse_args()
    if (args.duration_seconds <= 0.0 or args.period_seconds <= 0.0 or
            args.phase_seconds <= 0.0 or args.bootstrap_repetitions < 1 or
            not 0.0 < args.min_qps <= args.max_qps):
        parser.error("invalid QPS, duration, period, phase, or repetitions")
    if args.period_seconds % args.phase_seconds != 0.0:
        parser.error("period must be an integer multiple of phase")

    control = fit_control_origin(
        read_control_updates(args.control_log), args.approximate_start_us,
        args.duration_seconds, args.min_qps, args.max_qps,
        args.period_seconds,
    )
    start_us = control["origin_us"]
    end_us = start_us + round(args.duration_seconds * 1_000_000)
    baseline = read_trace(
        args.baseline, start_us, end_us, start_us, args.period_seconds,
        args.phase_seconds,
    )
    candidates = {}
    for index, path in enumerate(args.candidates):
        candidate = read_trace(
            path, start_us, end_us, start_us, args.period_seconds,
            args.phase_seconds,
        )
        candidates[path.stem] = analyze_candidate(
            baseline, candidate, args.bootstrap_repetitions,
            20261011 + index * 100,
        )
    report = {
        "control": control,
        "score_start_us": start_us,
        "score_end_us": end_us,
        "score_time_basis": "ready",
        "baseline": str(args.baseline),
        "baseline_record_count": len(baseline),
        "candidates": candidates,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
