#!/usr/bin/env python3
"""Fit u + sqrt(w1*u^4 + w2*u^2 + K) against the deployed shape."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from AnalyzeLatencyPhaseLeastSquares import ACTIVE, LOWER, PARAMETER_SCALES, load_host, metrics
from CompareLatencyAugmentedSquareShape import inner_input
from CompareLatencyGlobalLeastSquares import aggregate_metrics, segment_masks


def quartic_shape_and_derivatives(value, curvature, quartic, quadratic, scale):
    squared = value * value
    fourth_scaled = squared * (value / scale) ** 2
    root = np.sqrt(quartic * fourth_scaled + quadratic * squared + curvature)
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = value[positive] + root[positive]
    result[~positive] = (quartic * fourth_scaled[~positive] +
                         (quadratic - 1.0) * squared[~positive] + curvature) / (
                             root[~positive] - value[~positive])
    slope = 1.0 + (2.0 * quartic * value * squared / (scale * scale) +
                   quadratic * value) / root
    return result, slope, 1.0 / (2.0 * root), fourth_scaled / (2.0 * root), squared / (2.0 * root)


def quartic_prediction_and_jacobian(inputs, values, scale):
    parameters = np.zeros(17)
    parameters[ACTIVE] = values[:len(ACTIVE)] * PARAMETER_SCALES[ACTIVE]
    quartic, quadratic = values[-2:]
    inner = inner_input(inputs, parameters / PARAMETER_SCALES[:17])
    zero_inner = parameters[0] / 320.0
    curvature = parameters[8]
    nonlinear, slope, d_curvature, d_quartic, d_quadratic = (
        quartic_shape_and_derivatives(inner, curvature, quartic, quadratic, scale))
    zero, zero_slope, zero_d_curvature, zero_d_quartic, zero_d_quadratic = (
        quartic_shape_and_derivatives(
            np.asarray([zero_inner]), curvature, quartic, quadratic, scale))
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
    extra_gradient = 1500.0 * np.column_stack((
        d_quartic - zero_d_quartic[0], d_quadratic - zero_d_quadratic[0]))
    jacobian = np.column_stack((gradient[:, ACTIVE], extra_gradient))
    jacobian[raw < 0.0] = 0.0
    return np.maximum(raw, 0.0), jacobian


def fit_quartic(inputs, target, baseline_parameters, loss, max_zero_shift_ms=None,
                disable_quartic=False):
    baseline_parameters = np.asarray(baseline_parameters)
    scale = max(float(np.percentile(np.abs(inner_input(inputs, baseline_parameters)), 95)), 1.0)
    initial = (np.r_[baseline_parameters[ACTIVE], 1.0] if disable_quartic else
               np.r_[baseline_parameters[ACTIVE], 0.0, 1.0])
    lower = np.r_[LOWER, 0.0] if disable_quartic else np.r_[LOWER, 0.0, 0.0]
    upper = np.full(len(initial), np.inf)
    if max_zero_shift_ms is not None:
        zero_index = int(np.where(ACTIVE == 7)[0][0])
        zero_shift = max_zero_shift_ms / PARAMETER_SCALES[7]
        lower[zero_index] = max(lower[zero_index], initial[zero_index] - zero_shift)
        upper[zero_index] = initial[zero_index] + zero_shift
    cache = [None, None, None]

    def evaluate(values):
        if cache[0] is None or not np.array_equal(values, cache[0]):
            full_values = (np.r_[values[:len(ACTIVE)], 0.0, values[-1]]
                           if disable_quartic else values)
            prediction, jacobian = quartic_prediction_and_jacobian(inputs, full_values, scale)
            if disable_quartic:
                jacobian = np.column_stack((jacobian[:, :len(ACTIVE)], jacobian[:, -1]))
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

    baseline_cost = float(np.sum(residual(initial) ** 2) / 2.0)
    best = None
    total_nfev = 0
    starts = ((0.0, 1.0), (0.0, 0.5), (0.0, 2.0)) if disable_quartic else (
        (0.0, 1.0), (0.01, 1.0), (0.1, 1.0), (0.0, 2.0))
    for quartic, quadratic in starts:
        start = initial.copy()
        if disable_quartic:
            start[-1] = quadratic
        else:
            start[-2:] = quartic, quadratic
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
    if best.cost < 1e-8:
        refined = least_squares(residual, best.x, jac=jacobian,
                                bounds=(lower, upper), x_scale="jac",
                                max_nfev=2000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        total_nfev += refined.nfev
        if refined.cost < best.cost:
            best = refined
    return best, scale, baseline_cost, total_nfev


def evaluate_host(path, loss, baseline, max_zero_shift_ms=None, disable_quartic=False):
    inputs, target, _, online_base, online_final, _, check = load_host(path)
    result, scale, baseline_cost, nfev = fit_quartic(
        inputs, target, baseline["fit_parameters"], loss, max_zero_shift_ms,
        disable_quartic)
    fitted_values = (np.r_[result.x[:len(ACTIVE)], 0.0, result.x[-1]]
                     if disable_quartic else result.x)
    predictions = {"static": quartic_prediction_and_jacobian(inputs, fitted_values, scale)[0],
                   "online_base": online_base, "online_final": online_final}
    segments = {
        name: {**{predictor: metrics(target[mask], prediction[mask])
                  for predictor, prediction in predictions.items()},
               "mean_running": float(np.mean(inputs[mask, 1])) if mask.any() else None,
               "mean_waiting": float(np.mean(inputs[mask, 7])) if mask.any() else None}
        for name, mask in segment_masks(inputs, target).items()
    }
    parameters = np.zeros(17)
    parameters[ACTIVE] = fitted_values[:len(ACTIVE)]
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    return {"host": path.stem, "architecture": architecture, "sample_count": len(target),
            "loss": loss, "formula_check_max_ms": check,
            "fit_success": bool(result.success), "fit_nfev": nfev,
            "baseline_cost": baseline_cost, "fit_cost": float(result.cost),
            "fit_parameters": parameters.tolist(), "quartic_scale": scale,
            "quartic_weight_per_u4": float(fitted_values[-2] / scale ** 2),
            "quadratic_weight": float(fitted_values[-1]), "full_window": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--loss", choices=("log1p", "raw"), default="log1p")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-zero-shift-ms", type=float)
    parser.add_argument("--disable-quartic", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_zero_shift_ms is not None and args.max_zero_shift_ms <= 0:
        parser.error("--max-zero-shift-ms must be positive")
    baseline_data = json.loads(args.baseline_json.read_text())
    if baseline_data["loss"] != args.loss:
        parser.error("baseline and candidate losses differ")
    baseline = {host["host"]: host for host in baseline_data["hosts"]}
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if set(path.stem for path in paths) != set(baseline):
        parser.error("baseline hosts do not match input hosts")
    if args.workers == 1:
        hosts = [evaluate_host(path, args.loss, baseline[path.stem], args.max_zero_shift_ms,
                               args.disable_quartic)
                 for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(
                evaluate_host_args, ((path, args.loss, baseline[path.stem], args.max_zero_shift_ms,
                                     args.disable_quartic)
                                     for path in paths)))
    aggregate = {architecture: {"full_window": aggregate_metrics(hosts, architecture)}
                 for architecture in ("MI", "RTX")
                 if any(host["architecture"] == architecture for host in hosts)}
    print(json.dumps({"input_dir": str(args.input_dir), "loss": args.loss,
                      "max_zero_shift_ms": args.max_zero_shift_ms,
                      "disable_quartic": args.disable_quartic,
                      "hosts": hosts, "aggregate": aggregate}, separators=(",", ":")))


def evaluate_host_args(args):
    return evaluate_host(*args)


if __name__ == "__main__":
    main()
