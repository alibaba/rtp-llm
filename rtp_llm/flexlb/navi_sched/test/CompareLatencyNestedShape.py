#!/usr/bin/env python3
"""Compare the deployed F(u, K) with the same shape nested twice."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from AnalyzeLatencyPhaseLeastSquares import ACTIVE, LOWER, PARAMETER_SCALES, load_host, metrics, shape
from CompareLatencyGlobalLeastSquares import aggregate_metrics, segment_masks


def nested_shape_and_derivatives(value, curvature):
    inner = shape(value, curvature)
    inner_root = np.sqrt(value * value + curvature)
    outer = shape(inner, curvature)
    outer_root = np.sqrt(inner * inner + curvature)
    inner_slope = inner / inner_root
    outer_slope = outer / outer_root
    return (outer, outer_slope * inner_slope,
            outer_slope / (2.0 * inner_root) + 1.0 / (2.0 * outer_root))


def nested_prediction_and_jacobian(inputs, active):
    parameters = np.zeros(17)
    parameters[ACTIVE] = active * PARAMETER_SCALES[ACTIVE]
    inner_input = (inputs[:, :7] @ parameters[:7] +
                   inputs[:, 7:12] @ parameters[10:15] +
                   inputs[:, 13] * parameters[16]) / 320.0
    zero_input = parameters[0] / 320.0
    curvature = parameters[8]
    nested, slope, curvature_slope = nested_shape_and_derivatives(
        inner_input, curvature)
    zero_nested, zero_slope, zero_curvature_slope = nested_shape_and_derivatives(
        np.asarray([zero_input]), curvature)
    total_prompt = inputs[:, 2] + inputs[:, 3] + inputs[:, 8] + inputs[:, 9]
    raw = (parameters[7] + 1500.0 * (nested - zero_nested[0]) +
           parameters[9] * total_prompt + parameters[15] * inputs[:, 12])
    gradient = np.zeros((len(inputs), 17))
    scaled_slope = 1500.0 * slope / 320.0
    gradient[:, :7] = scaled_slope[:, None] * inputs[:, :7]
    gradient[:, 0] -= 1500.0 * zero_slope[0] / 320.0
    gradient[:, 7] = 1.0
    gradient[:, 8] = 1500.0 * (curvature_slope - zero_curvature_slope[0])
    gradient[:, 9] = total_prompt
    gradient[:, 10:15] = scaled_slope[:, None] * inputs[:, 7:12]
    gradient[:, 15] = inputs[:, 12]
    gradient[:, 16] = scaled_slope * inputs[:, 13]
    gradient *= PARAMETER_SCALES[:17]
    gradient[raw < 0.0] = 0.0
    return np.maximum(raw, 0.0), gradient[:, ACTIVE]


def fit_nested(inputs, target, baseline_parameters, loss):
    baseline_parameters = np.asarray(baseline_parameters)
    cache = [None, None, None]

    def evaluate(values):
        if cache[0] is None or not np.array_equal(values, cache[0]):
            prediction, gradient = nested_prediction_and_jacobian(inputs, values)
            cache[:] = [values.copy(), prediction, gradient]
        return cache[1], cache[2]

    def residual(values):
        prediction, _ = evaluate(values)
        if loss == "raw":
            return prediction - target
        return np.log1p(prediction) - np.log1p(target)

    def jacobian(values):
        prediction, gradient = evaluate(values)
        if loss == "raw":
            return gradient
        return gradient / (1.0 + prediction[:, None])

    best = None
    total_nfev = 0
    for factor in (1.0, 0.5, 0.25):
        initial = baseline_parameters.copy()
        initial[:7] *= factor
        initial[10:15] *= factor
        initial[16] *= factor
        initial[8] *= factor * factor
        initial[ACTIVE] = np.maximum(initial[ACTIVE], LOWER)
        result = least_squares(residual, initial[ACTIVE], jac=jacobian,
                               bounds=(LOWER, np.inf), x_scale="jac",
                               max_nfev=1200, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        total_nfev += result.nfev
        if best is None or result.cost < best.cost:
            best = result
    for _ in range(3):
        if best.success:
            break
        refined = least_squares(residual, best.x, jac=jacobian,
                                bounds=(LOWER, np.inf), x_scale="jac",
                                max_nfev=1200, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        total_nfev += refined.nfev
        if refined.cost < best.cost:
            best = refined
        else:
            break
    return best, total_nfev


def evaluate_host(path, loss, baseline):
    inputs, target, _, online_base, online_final, _, check = load_host(path)
    result, nfev = fit_nested(inputs, target, baseline["fit_parameters"], loss)
    predictions = {"static": nested_prediction_and_jacobian(inputs, result.x)[0],
                   "online_base": online_base, "online_final": online_final}
    segments = {
        name: {**{predictor: metrics(target[mask], prediction[mask])
                  for predictor, prediction in predictions.items()},
               "mean_running": float(np.mean(inputs[mask, 1])) if mask.any() else None,
               "mean_waiting": float(np.mean(inputs[mask, 7])) if mask.any() else None}
        for name, mask in segment_masks(inputs, target).items()
    }
    parameters = np.zeros(17)
    parameters[ACTIVE] = result.x
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    return {"host": path.stem, "architecture": architecture,
            "sample_count": len(target), "loss": loss,
            "formula_check_max_ms": check, "fit_success": bool(result.success),
            "fit_nfev": nfev, "fit_cost": float(result.cost),
            "fit_parameters": parameters.tolist(), "full_window": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--loss", choices=("log1p", "raw"), default="log1p")
    parser.add_argument("--workers", type=int, default=1)
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
        hosts = [evaluate_host(path, args.loss, baseline[path.stem]) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(
                evaluate_host_args, ((path, args.loss, baseline[path.stem])
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
