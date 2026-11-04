#!/usr/bin/env python3
"""Fit the deployed shape plus a learnable square of its inner input."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, LOWER, PARAMETER_SCALES, load_host, metrics, shape,
)
from CompareLatencyGlobalLeastSquares import aggregate_metrics, segment_masks


def inner_input(inputs, parameters):
    physical = parameters * PARAMETER_SCALES[:17]
    return (inputs[:, :7] @ physical[:7] +
            inputs[:, 7:12] @ physical[10:15] +
            inputs[:, 13] * physical[16]) / 320.0


def augmented_prediction_and_jacobian(inputs, values, square_scale):
    parameters = np.zeros(17)
    parameters[ACTIVE] = values[:len(ACTIVE)]
    physical = parameters * PARAMETER_SCALES[:17]
    weight = values[-1]
    inner = inner_input(inputs, parameters)
    zero_inner = physical[0] / 320.0
    curvature = physical[8]
    total_prompt = inputs[:, 2] + inputs[:, 3] + inputs[:, 8] + inputs[:, 9]
    square = (inner * inner - zero_inner * zero_inner) / (square_scale ** 2)
    raw = (physical[7] +
           1500.0 * (shape(inner, curvature) -
                     shape(np.asarray([zero_inner]), curvature)[0]) +
           physical[9] * total_prompt + physical[15] * inputs[:, 12] +
           weight * square)
    root = np.sqrt(inner * inner + curvature)
    zero_root = np.sqrt(zero_inner * zero_inner + curvature)
    slope = (1500.0 * (1.0 + inner / root) +
             2.0 * weight * inner / (square_scale ** 2)) / 320.0
    zero_slope = (1500.0 * (1.0 + zero_inner / zero_root) +
                  2.0 * weight * zero_inner / (square_scale ** 2)) / 320.0
    gradient = np.zeros((len(inputs), 17))
    gradient[:, :7] = slope[:, None] * inputs[:, :7]
    gradient[:, 0] -= zero_slope
    gradient[:, 7] = 1.0
    gradient[:, 8] = 750.0 * (1.0 / root - 1.0 / zero_root)
    gradient[:, 9] = total_prompt
    gradient[:, 10:15] = slope[:, None] * inputs[:, 7:12]
    gradient[:, 15] = inputs[:, 12]
    gradient[:, 16] = slope * inputs[:, 13]
    gradient *= PARAMETER_SCALES[:17]
    gradient[raw < 0.0] = 0.0
    square[raw < 0.0] = 0.0
    return np.maximum(raw, 0.0), np.column_stack((gradient[:, ACTIVE], square))


def fit_augmented(inputs, target, initial_parameters, loss, nonnegative_weight=False):
    initial_parameters = np.asarray(initial_parameters)
    initial_inner = inner_input(inputs, initial_parameters)
    square_scale = max(float(np.percentile(np.abs(initial_inner), 95)), 1.0)
    initial = np.r_[initial_parameters[ACTIVE], 0.0]
    lower = np.r_[LOWER, 0.0 if nonnegative_weight else -np.inf]
    upper = np.full(len(initial), np.inf)
    cache = [None, None, None]

    def evaluate(values):
        if cache[0] is None or not np.array_equal(values, cache[0]):
            predicted, jacobian = augmented_prediction_and_jacobian(
                inputs, values, square_scale)
            cache[:] = [values.copy(), predicted, jacobian]
        return cache[1], cache[2]

    def residual(values):
        predicted, _ = evaluate(values)
        if loss == "raw":
            return predicted - target
        return np.log1p(predicted) - np.log1p(target)

    def jacobian(values):
        predicted, derivative = evaluate(values)
        if loss == "raw":
            return derivative
        return derivative / (1.0 + predicted[:, None])

    baseline_cost = float(np.sum(residual(initial) ** 2) / 2.0)
    best = None
    total_nfev = 0
    starts = (0.0, 1.0) if nonnegative_weight else (0.0, 1.0, -1.0)
    for weight in starts:
        start = initial.copy()
        start[-1] = weight
        start = np.clip(start, lower, upper)
        result = least_squares(residual, start, jac=jacobian,
                               bounds=(lower, upper), x_scale="jac",
                               max_nfev=1200, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        total_nfev += result.nfev
        if best is None or result.cost < best.cost:
            best = result
    for _ in range(3):
        if best.success:
            break
        refined = least_squares(residual, best.x, jac=jacobian,
                                bounds=(lower, upper), x_scale="jac",
                                max_nfev=1200, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        total_nfev += refined.nfev
        if refined.cost < best.cost:
            best = refined
        else:
            break
    return best, square_scale, baseline_cost, total_nfev


def evaluate_host(path, loss, baseline, nonnegative_weight=False):
    inputs, target, _, online_base, online_final, _, check = load_host(path)
    result, square_scale, baseline_cost, nfev = fit_augmented(
        inputs, target, baseline["fit_parameters"], loss, nonnegative_weight)
    predictions = {"static": augmented_prediction_and_jacobian(
        inputs, result.x, square_scale)[0],
        "online_base": online_base, "online_final": online_final}
    segments = {
        name: {**{predictor: metrics(target[mask], prediction[mask])
                  for predictor, prediction in predictions.items()},
               "mean_running": float(np.mean(inputs[mask, 1])) if mask.any() else None,
               "mean_waiting": float(np.mean(inputs[mask, 7])) if mask.any() else None}
        for name, mask in segment_masks(inputs, target).items()
    }
    parameters = np.zeros(17)
    parameters[ACTIVE] = result.x[:len(ACTIVE)]
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    return {"host": path.stem, "architecture": architecture, "sample_count": len(target),
            "loss": loss, "formula_check_max_ms": check,
            "fit_success": bool(result.success), "fit_nfev": nfev,
            "baseline_cost": baseline_cost, "fit_cost": float(result.cost),
            "fit_parameters": parameters.tolist(),
            "square_scale": square_scale, "square_weight": float(result.x[-1]),
            "square_weight_per_u2": float(result.x[-1] / square_scale ** 2),
            "full_window": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--loss", choices=("log1p", "raw"), default="log1p")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--nonnegative-weight", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    baseline_data = json.loads(args.baseline_json.read_text())
    if baseline_data["loss"] != args.loss:
        parser.error("baseline and candidate losses differ")
    baseline = {host["host"]: host for host in baseline_data["hosts"]}
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if set(path.stem for path in paths) != set(baseline):
        parser.error("baseline hosts do not match input hosts")
    if args.workers == 1:
        hosts = [evaluate_host(path, args.loss, baseline[path.stem], args.nonnegative_weight)
                 for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(
                evaluate_host_args, ((path, args.loss, baseline[path.stem], args.nonnegative_weight)
                                     for path in paths)))
    aggregate = {architecture: {"full_window": aggregate_metrics(hosts, architecture)}
                 for architecture in ("MI", "RTX")
                 if any(host["architecture"] == architecture for host in hosts)}
    print(json.dumps({"input_dir": str(args.input_dir), "loss": args.loss,
                      "hosts": hosts, "aggregate": aggregate}, separators=(",", ":")))


def evaluate_host_args(args):
    return evaluate_host(*args)


if __name__ == "__main__":
    main()
