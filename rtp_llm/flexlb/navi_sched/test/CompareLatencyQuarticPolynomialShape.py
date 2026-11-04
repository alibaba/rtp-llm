#!/usr/bin/env python3
"""Fit u + sqrt(w1*u^4 + w2*u^3 + w3*u^2 + w4*u + k)."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from AnalyzeLatencyPhaseLeastSquares import ACTIVE, LOWER, PARAMETER_SCALES, load_host, metrics
from CompareLatencyAugmentedSquareShape import inner_input
from CompareLatencyGlobalLeastSquares import aggregate_metrics, segment_masks


def polynomial_shape_and_derivatives(value, curvature, coefficients, scale):
    a, b, c, d, e = coefficients
    z = value / scale
    square = a * z * z + b * z + c
    line = d * z + e
    root = np.sqrt(scale * scale * (square * square + line * line) + curvature)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = value[positive] + root[positive]
    output[~positive] = (scale * scale * (
        square[~positive] ** 2 + line[~positive] ** 2 - z[~positive] ** 2) +
        curvature) / (root[~positive] - value[~positive])
    slope = 1.0 + scale * (square * (2.0 * a * z + b) + line * d) / root
    derivative = scale * scale / root[:, None] * np.column_stack((
        square * z * z, square * z, square, line * z, line))
    return output, slope, 1.0 / (2.0 * root), derivative


def polynomial_prediction_and_jacobian(inputs, values, scale):
    parameters = np.zeros(17)
    parameters[ACTIVE] = values[:len(ACTIVE)] * PARAMETER_SCALES[ACTIVE]
    coefficients = values[len(ACTIVE):]
    inner = inner_input(inputs, parameters / PARAMETER_SCALES[:17])
    zero_inner = parameters[0] / 320.0
    nonlinear, slope, d_curvature, d_coefficients = (
        polynomial_shape_and_derivatives(inner, parameters[8], coefficients, scale))
    zero, zero_slope, zero_d_curvature, zero_d_coefficients = (
        polynomial_shape_and_derivatives(
            np.asarray([zero_inner]), parameters[8], coefficients, scale))
    total_prompt = inputs[:, 2] + inputs[:, 3] + inputs[:, 8] + inputs[:, 9]
    raw = (parameters[7] + 1500.0 * (nonlinear - zero[0]) +
           parameters[9] * total_prompt + parameters[15] * inputs[:, 12])
    gradient = np.zeros((len(inputs), 17))
    scaled_slope = 1500.0 * slope / 320.0
    gradient[:, :7] = scaled_slope[:, None] * inputs[:, :7]
    gradient[:, 0] -= 1500.0 * zero_slope[0] / 320.0
    gradient[:, 7] = 1.0
    gradient[:, 8] = 1500.0 * (d_curvature - zero_d_curvature[0])
    gradient[:, 9] = total_prompt
    gradient[:, 10:15] = scaled_slope[:, None] * inputs[:, 7:12]
    gradient[:, 15] = inputs[:, 12]
    gradient[:, 16] = scaled_slope * inputs[:, 13]
    gradient *= PARAMETER_SCALES[:17]
    jacobian = np.column_stack((
        gradient[:, ACTIVE], 1500.0 * (d_coefficients - zero_d_coefficients)))
    jacobian[raw < 0.0] = 0.0
    return np.maximum(raw, 0.0), jacobian


def physical_coefficients(coefficients, curvature, scale):
    a, b, c, d, e = coefficients
    return [a * a / (scale * scale), 2.0 * a * b / scale,
            b * b + 2.0 * a * c + d * d,
            2.0 * scale * (b * c + d * e),
            scale * scale * (c * c + e * e) + curvature]


def fit_polynomial(inputs, target, baseline_parameters, even, loss,
                   max_zero_shift_ms=None):
    baseline_parameters = np.asarray(baseline_parameters)
    scale = max(float(np.percentile(np.abs(inner_input(inputs, baseline_parameters)), 95)), 1.0)
    baseline_start = np.r_[baseline_parameters[ACTIVE], 0.0, 1.0, 0.0, 0.0, 0.0]
    even_parameters = np.asarray(even["fit_parameters"])
    even_start = np.r_[even_parameters[ACTIVE],
                       np.sqrt(max(even["quartic_weight_per_u4"], 0.0)) * scale,
                       0.0, 0.0, np.sqrt(max(even["quadratic_weight"], 0.0)), 0.0]
    lower = np.r_[LOWER, np.full(5, -np.inf)]
    upper = np.full(len(baseline_start), np.inf)
    if max_zero_shift_ms is not None:
        zero_index = int(np.where(ACTIVE == 7)[0][0])
        zero_shift = max_zero_shift_ms / PARAMETER_SCALES[7]
        lower[zero_index] = max(lower[zero_index], baseline_start[zero_index] - zero_shift)
        upper[zero_index] = baseline_start[zero_index] + zero_shift
    cache = [None, None, None]

    def evaluate(values):
        if cache[0] is None or not np.array_equal(values, cache[0]):
            prediction, jacobian = polynomial_prediction_and_jacobian(inputs, values, scale)
            cache[:] = [values.copy(), prediction, jacobian]
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

    baseline_cost = float(np.sum(residual(baseline_start) ** 2) / 2.0)
    even_cost = float(np.sum(residual(even_start) ** 2) / 2.0)
    best = None
    total_nfev = 0
    starts = (baseline_start, even_start,
              np.r_[baseline_parameters[ACTIVE], 0.1, 1.0, 0.1, 0.1, 0.0],
              np.r_[even_parameters[ACTIVE], even_start[-5] + 0.1,
                    0.1, 0.0, even_start[-2], 0.1])
    for start in starts:
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
                                max_nfev=5000, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        total_nfev += refined.nfev
        if refined.cost < best.cost:
            best = refined
        else:
            break
    return best, scale, baseline_cost, even_cost, total_nfev


def evaluate_host(path, loss, baseline, even, max_zero_shift_ms=None):
    inputs, target, _, online_base, online_final, _, check = load_host(path)
    result, scale, baseline_cost, even_cost, nfev = fit_polynomial(
        inputs, target, baseline["fit_parameters"], even, loss, max_zero_shift_ms)
    predictions = {"static": polynomial_prediction_and_jacobian(inputs, result.x, scale)[0],
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
            "baseline_cost": baseline_cost, "even_cost": even_cost,
            "fit_cost": float(result.cost), "fit_parameters": parameters.tolist(),
            "scale": scale, "sos_coefficients": result.x[len(ACTIVE):].tolist(),
            "polynomial_coefficients": physical_coefficients(
                result.x[len(ACTIVE):], parameters[8] * PARAMETER_SCALES[8], scale),
            "full_window": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--even-json", type=Path, required=True)
    parser.add_argument("--loss", choices=("log1p", "raw"), default="log1p")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-zero-shift-ms", type=float)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_zero_shift_ms is not None and args.max_zero_shift_ms <= 0:
        parser.error("--max-zero-shift-ms must be positive")
    baseline_data = json.loads(args.baseline_json.read_text())
    even_data = json.loads(args.even_json.read_text())
    if baseline_data["loss"] != args.loss or even_data["loss"] != args.loss:
        parser.error("baseline, even model, and candidate losses differ")
    baseline = {host["host"]: host for host in baseline_data["hosts"]}
    even = {host["host"]: host for host in even_data["hosts"]}
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if set(path.stem for path in paths) != set(baseline) or set(baseline) != set(even):
        parser.error("host sets differ between input and baselines")
    if args.workers == 1:
        hosts = [evaluate_host(path, args.loss, baseline[path.stem], even[path.stem],
                               args.max_zero_shift_ms)
                 for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(
                evaluate_host_args, ((path, args.loss, baseline[path.stem], even[path.stem],
                                     args.max_zero_shift_ms)
                                     for path in paths)))
    aggregate = {architecture: {"full_window": aggregate_metrics(hosts, architecture)}
                 for architecture in ("MI", "RTX")
                 if any(host["architecture"] == architecture for host in hosts)}
    print(json.dumps({"input_dir": str(args.input_dir), "loss": args.loss,
                      "max_zero_shift_ms": args.max_zero_shift_ms,
                      "hosts": hosts, "aggregate": aggregate}, separators=(",", ":")))


def evaluate_host_args(args):
    return evaluate_host(*args)


if __name__ == "__main__":
    main()
