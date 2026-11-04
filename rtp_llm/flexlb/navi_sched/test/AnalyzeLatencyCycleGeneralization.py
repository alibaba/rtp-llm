#!/usr/bin/env python3
"""Compare static formula capacity and cross-cycle generalization by host."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from AnalyzeLatencyPhaseLeastSquares import (
    PARAMETER_SCALES,
    fit,
    load_host,
    prediction_and_jacobian,
    unpack,
)


def score(target, prediction):
    if not len(target):
        return None
    error = prediction - target
    return {
        "n": int(len(target)),
        "target_sum_ms": float(np.sum(target)),
        "absolute_error_sum_ms": float(np.sum(np.abs(error))),
        "wape_pct": float(np.sum(np.abs(error)) / np.sum(target) * 100.0),
        "mae_ms": float(np.mean(np.abs(error))),
        "bias_prediction_minus_target_ms": float(np.mean(error)),
        "p95_absolute_error_ms": float(np.percentile(np.abs(error), 95)),
    }


def feature_neighbor_support(inputs, target, ready_time, cycle, anomaly_cycle):
    normal = cycle != anomaly_cycle
    anomaly = ~normal
    normal_inputs = inputs[normal]
    anomaly_inputs = inputs[anomaly]
    scale = np.std(normal_inputs, axis=0)
    active = scale > 1e-12
    if np.sum(normal) < 2 or not np.any(active) or not np.any(anomaly):
        raise ValueError("insufficient samples for feature-neighbor analysis")
    center = np.mean(normal_inputs[:, active], axis=0)
    scale = scale[active]
    normalized_normal = (normal_inputs[:, active] - center) / scale
    normalized_anomaly = (anomaly_inputs[:, active] - center) / scale
    divisor = np.sqrt(np.sum(active))
    normal_distances = cdist(normalized_normal, normalized_normal) / divisor
    np.fill_diagonal(normal_distances, np.inf)
    normal_nearest = np.min(normal_distances, axis=1)
    cross_distances = cdist(normalized_anomaly, normalized_normal) / divisor
    nearest_position = np.argmin(cross_distances, axis=1)
    nearest_distance = cross_distances[
        np.arange(len(normalized_anomaly)), nearest_position]
    normal_indexes = np.flatnonzero(normal)
    anomaly_indexes = np.flatnonzero(anomaly)
    nearest_indexes = normal_indexes[nearest_position]
    target_gap = target[anomaly_indexes] - target[nearest_indexes]
    target_ratio = (
        np.maximum(target[anomaly_indexes], target[nearest_indexes]) /
        np.maximum(
            1.0,
            np.minimum(target[anomaly_indexes], target[nearest_indexes]),
        )
    )
    severe = (np.abs(target_gap) >= 5000.0) & (target_ratio >= 1.5)
    normal_p95 = float(np.percentile(normal_nearest, 95))
    supported = nearest_distance <= normal_p95
    supported_severe = supported & severe
    examples = []
    for position in np.argsort(np.abs(target_gap))[::-1][:5]:
        left = anomaly_indexes[position]
        right = nearest_indexes[position]
        examples.append({
            "anomaly_ready_time_us": int(ready_time[left]),
            "normal_ready_time_us": int(ready_time[right]),
            "distance": float(nearest_distance[position]),
            "anomaly_target_ms": float(target[left]),
            "normal_target_ms": float(target[right]),
            "anomaly_minus_normal_target_ms": float(target_gap[position]),
            "target_ratio": float(target_ratio[position]),
            "inside_normal_p95_support": bool(supported[position]),
        })
    supported_severe_examples = []
    supported_severe_positions = np.flatnonzero(supported_severe)
    order = supported_severe_positions[
        np.argsort(np.abs(target_gap[supported_severe_positions]))[::-1]
    ]
    for position in order[:5]:
        left = anomaly_indexes[position]
        right = nearest_indexes[position]
        supported_severe_examples.append({
            "anomaly_ready_time_us": int(ready_time[left]),
            "normal_ready_time_us": int(ready_time[right]),
            "distance": float(nearest_distance[position]),
            "anomaly_target_ms": float(target[left]),
            "normal_target_ms": float(target[right]),
            "anomaly_minus_normal_target_ms": float(target_gap[position]),
            "target_ratio": float(target_ratio[position]),
        })
    return {
        "active_feature_count": int(np.sum(active)),
        "normal_count": int(np.sum(normal)),
        "anomaly_count": int(np.sum(anomaly)),
        "normal_leave_one_out_distance_p50": float(np.percentile(
            normal_nearest, 50)),
        "normal_leave_one_out_distance_p95": normal_p95,
        "anomaly_to_normal_distance_p50": float(np.percentile(
            nearest_distance, 50)),
        "anomaly_to_normal_distance_p95": float(np.percentile(
            nearest_distance, 95)),
        "anomaly_outside_normal_p95_count": int(np.sum(
            nearest_distance > normal_p95)),
        "supported_anomaly_count": int(np.sum(supported)),
        "severe_target_conflict_count": int(np.sum(severe)),
        "supported_severe_target_conflict_count": int(np.sum(
            supported_severe)),
        "target_gap_absolute_p50_ms": float(np.percentile(
            np.abs(target_gap), 50)),
        "target_gap_absolute_p95_ms": float(np.percentile(
            np.abs(target_gap), 95)),
        "target_gap_signed_mean_ms": float(np.mean(target_gap)),
        "examples": examples,
        "supported_severe_examples": supported_severe_examples,
    }


def fit_model(inputs, target, starts, train, loss, constrained):
    attempts = [fit(inputs[train], target[train], [start], loss, "none",
                    enforce_nonnegative_waiting=constrained)
                for start in starts]
    valid = [attempt for attempt in attempts
             if attempt.success and attempt.constraint_violation <= 1e-7]
    result = min(valid or attempts,
                 key=lambda candidate: (candidate.constraint_violation,
                                        candidate.cost))
    prediction = prediction_and_jacobian(inputs, result.x, "none")[0]
    train_prediction, train_jacobian = prediction_and_jacobian(
        inputs[train], result.x, "none")
    if loss == "log1p":
        train_jacobian = train_jacobian / (1.0 + train_prediction[:, None])
    column_norm = np.linalg.norm(train_jacobian, axis=0)
    active_columns = column_norm > 1e-10
    normalized_jacobian = (
        train_jacobian[:, active_columns] / column_norm[active_columns]
    )
    singular_values = np.linalg.svd(
        normalized_jacobian, compute_uv=False)
    identifiable = singular_values[singular_values > 1e-8]
    parameters = unpack(result.x)
    effective = parameters * PARAMETER_SCALES[:17]
    waiting_terms = inputs[:, 7:12] * effective[None, 10:15]
    waiting_cost = np.sum(waiting_terms, axis=1)
    waiting_tolerance = 1e-12 * np.maximum(
        1.0, np.sum(np.abs(waiting_terms), axis=1))
    waiting_unsafe = waiting_cost < -waiting_tolerance
    train_waiting_cost = waiting_cost[train]
    return {
        "train_count": int(np.sum(train)),
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "optimality": float(result.optimality),
        "constraint_violation": float(result.constraint_violation),
        "active_jacobian_columns": int(np.sum(active_columns)),
        "normalized_jacobian_rank": int(len(identifiable)),
        "normalized_jacobian_min_singular": (
            float(identifiable[-1]) if len(identifiable) else 0.0),
        "normalized_jacobian_condition_number": (
            float(identifiable[0] / identifiable[-1])
            if len(identifiable) else None),
        "attempts": [
            {
                "success": bool(attempt.success),
                "cost": float(attempt.cost),
                "nfev": int(attempt.nfev),
                "optimality": float(attempt.optimality),
                "constraint_violation": float(attempt.constraint_violation),
            }
            for attempt in attempts
        ],
        "parameters": parameters.tolist(),
        "minimum_train_waiting_linear_cost": float(np.min(train_waiting_cost)),
        "negative_train_waiting_cost_count": int(np.sum(
            waiting_unsafe[train])),
        "minimum_all_waiting_linear_cost": float(np.min(waiting_cost)),
        "negative_all_waiting_cost_count": int(np.sum(waiting_unsafe)),
    }, prediction


def evaluate_host(job):
    (path, start_us, cycle_count, anomaly_cycle, losses, constrained,
     fit_start_count) = job
    (inputs, target, event_time, online_base, online_final, starts,
     check) = load_host(path, fit_start_count)
    ready_time = event_time - np.rint(target * 1000.0).astype(np.int64)
    selected = ((ready_time >= start_us) &
                (ready_time < start_us + cycle_count * 300_000_000))
    inputs = inputs[selected]
    target = target[selected]
    ready_time = ready_time[selected]
    online_base = online_base[selected]
    online_final = online_final[selected]
    cycle = ((ready_time - start_us) // 300_000_000).astype(int)
    if len(target) < cycle_count * 40:
        raise ValueError("insufficient selected samples: " + path.name)
    masks = {"all": np.ones(len(target), dtype=bool),
             "normal": cycle != anomaly_cycle,
             "anomaly": cycle == anomaly_cycle}
    for index in range(cycle_count):
        masks["cycle_{}".format(index)] = cycle == index
    evaluations = {
        "online_base": {name: score(target[mask], online_base[mask])
                        for name, mask in masks.items()},
        "online_final": {name: score(target[mask], online_final[mask])
                         for name, mask in masks.items()},
    }
    fitted = {}
    train_masks = {"global": masks["all"],
                   "normal_only": masks["normal"],
                   "anomaly_only": masks["anomaly"]}
    for index in range(cycle_count):
        train_masks["own_cycle_{}".format(index)] = masks["cycle_{}".format(index)]
        train_masks["leave_cycle_{}_out".format(index)] = ~masks["cycle_{}".format(index)]
    for loss in losses:
        fitted[loss] = {}
        for name, train in train_masks.items():
            details, prediction = fit_model(
                inputs, target, starts, train, loss, constrained)
            details["evaluations"] = {
                segment: score(target[mask], prediction[mask])
                for segment, mask in masks.items()
            }
            fitted[loss][name] = details
    architecture = "MI308" if "MI308" in path.name else "RTX5000"
    return {
        "host": path.stem,
        "architecture": architecture,
        "formula_check_max_ms": check,
        "sample_count": int(len(target)),
        "cycle_counts": {str(index): int(np.sum(cycle == index))
                         for index in range(cycle_count)},
        "feature_neighbor_support": feature_neighbor_support(
            inputs, target, ready_time, cycle, anomaly_cycle),
        "evaluations": evaluations,
        "fits": fitted,
    }


def aggregate(hosts):
    result = {}
    for architecture in ("MI308", "RTX5000"):
        selected = [host for host in hosts if host["architecture"] == architecture]
        architecture_result = {}
        keys = [("online", "online_base"), ("online", "online_final")]
        if selected:
            for loss in selected[0]["fits"]:
                for model in selected[0]["fits"][loss]:
                    keys.append((loss, model))
        for loss, model in keys:
            source = (lambda host: host["evaluations"][model]
                      if loss == "online" else host["fits"][loss][model]["evaluations"])
            segments = {}
            for segment in selected[0]["evaluations"]["online_final"] if selected else ():
                metrics = [source(host)[segment] for host in selected]
                count = sum(metric["n"] for metric in metrics)
                target_sum = sum(metric["target_sum_ms"] for metric in metrics)
                absolute_error_sum = sum(metric["absolute_error_sum_ms"] for metric in metrics)
                segments[segment] = {
                    "n": count,
                    "wape_pct": absolute_error_sum / target_sum * 100.0,
                    "mae_ms": absolute_error_sum / count,
                    "bias_prediction_minus_target_ms": sum(
                        metric["bias_prediction_minus_target_ms"] * metric["n"]
                        for metric in metrics) / count,
                }
            architecture_result[loss + "/" + model] = segments
        result[architecture] = architecture_result
    return result


def aggregate_parameter_shifts(hosts):
    result = {}
    for architecture in ("MI308", "RTX5000"):
        selected = [host for host in hosts
                    if host["architecture"] == architecture]
        losses = selected[0]["fits"] if selected else ()
        result[architecture] = {}
        for loss in losses:
            normal = np.asarray([
                host["fits"][loss]["normal_only"]["parameters"]
                for host in selected
            ]) * PARAMETER_SCALES[:17]
            anomaly = np.asarray([
                host["fits"][loss]["anomaly_only"]["parameters"]
                for host in selected
            ]) * PARAMETER_SCALES[:17]
            absolute_delta = np.abs(anomaly - normal)
            baseline = np.mean(np.abs(normal), axis=0)
            relative = np.divide(
                absolute_delta,
                np.maximum(np.abs(normal), 1e-12),
            )
            result[architecture][loss] = [
                {
                    "index": index,
                    "baseline_absolute_mean": float(baseline[index]),
                    "absolute_delta_mean": float(np.mean(
                        absolute_delta[:, index])),
                    "absolute_delta_over_baseline_absolute_mean": (
                        float(np.mean(absolute_delta[:, index]) /
                              baseline[index])
                        if baseline[index] > 1e-12 else None
                    ),
                    "per_host_relative_delta_p50": float(np.percentile(
                        relative[:, index], 50)),
                    "per_host_relative_delta_p95": float(np.percentile(
                        relative[:, index], 95)),
                }
                for index in range(17)
            ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--cycle-count", type=int, default=6)
    parser.add_argument("--anomaly-cycle", type=int, default=1)
    parser.add_argument("--losses", nargs="+", choices=("raw", "log1p"),
                        default=("raw", "log1p"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--constrained", action="store_true",
                        help="Enforce the online nonnegative waiting-cost rule")
    parser.add_argument("--fit-start-count", type=int, default=3)
    args = parser.parse_args()
    paths = sorted(path for path in args.input_dir.glob("*.jsonl")
                   if "MI308" in path.name or "RTX_PRO_5000" in path.name)
    jobs = [(path, args.start_us, args.cycle_count, args.anomaly_cycle,
             tuple(args.losses), args.constrained, args.fit_start_count)
            for path in paths]
    if args.workers == 1:
        hosts = [evaluate_host(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(evaluate_host, jobs))
    print(json.dumps({"input_dir": str(args.input_dir),
                      "start_us": args.start_us,
                      "cycle_count": args.cycle_count,
                      "anomaly_cycle": args.anomaly_cycle,
                      "constrained": args.constrained,
                      "fit_start_count": args.fit_start_count,
                      "hosts": hosts,
                      "aggregate": aggregate(hosts),
                      "parameter_shifts": aggregate_parameter_shifts(hosts)},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
