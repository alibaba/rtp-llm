#!/usr/bin/env python3
"""Replace the deployed nonlinear shape with the square of its inner input."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from AnalyzeLatencyPhaseLeastSquares import PARAMETER_SCALES, load_host, metrics
from CompareLatencyGlobalLeastSquares import aggregate_metrics, segment_masks


ACTIVE = np.asarray([index for index in range(17) if index not in (6, 8)])
INNER = np.asarray([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 16])
LOWER = np.full(len(ACTIVE), -np.inf)
for index, bound in ((7, 0.0), (9, 0.0), (15, 1.0)):
    LOWER[np.where(ACTIVE == index)[0][0]] = bound / PARAMETER_SCALES[index]


def square_prediction_and_jacobian(inputs, values):
    parameters = np.zeros(17)
    parameters[ACTIVE] = values * PARAMETER_SCALES[ACTIVE]
    inner = (inputs[:, :6] @ parameters[:6] +
             inputs[:, 7:12] @ parameters[10:15] +
             inputs[:, 13] * parameters[16]) / 320.0
    zero_inner = parameters[0] / 320.0
    total_prompt = inputs[:, 2] + inputs[:, 3] + inputs[:, 8] + inputs[:, 9]
    raw = (parameters[7] + 1500.0 * (inner * inner - zero_inner * zero_inner) +
           parameters[9] * total_prompt + parameters[15] * inputs[:, 12])
    gradient = np.zeros((len(inputs), 17))
    slope = 3000.0 * inner / 320.0
    gradient[:, :6] = slope[:, None] * inputs[:, :6]
    gradient[:, 0] -= 3000.0 * zero_inner / 320.0
    gradient[:, 7] = 1.0
    gradient[:, 9] = total_prompt
    gradient[:, 10:15] = slope[:, None] * inputs[:, 7:12]
    gradient[:, 15] = inputs[:, 12]
    gradient[:, 16] = slope * inputs[:, 13]
    gradient *= PARAMETER_SCALES[:17]
    gradient[raw < 0.0] = 0.0
    return np.maximum(raw, 0.0), gradient[:, ACTIVE]


def fit_square(inputs, target, starts, loss):
    best = None
    total_nfev = 0
    inner_scales = (0.1, 0.25, 0.5, 1.0)
    cache = [None, None, None]

    def evaluate(values):
        if cache[0] is None or not np.array_equal(values, cache[0]):
            predicted, jacobian = square_prediction_and_jacobian(inputs, values)
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

    def optimize(initial):
        return least_squares(residual, initial, jac=jacobian,
                             bounds=(LOWER, np.full(len(ACTIVE), np.inf)),
                             x_scale="jac", max_nfev=1200,
                             ftol=1e-8, xtol=1e-8, gtol=1e-8)

    for parameters in starts:
        for inner_scale in inner_scales:
            initial = np.asarray(parameters, dtype=float).copy()
            initial[INNER] *= inner_scale
            initial = np.maximum(initial[ACTIVE], LOWER + 1e-9)
            result = optimize(initial)
            total_nfev += result.nfev
            if best is None or result.cost < best.cost:
                best = result
    for _ in range(3):
        if best.success:
            break
        refined = optimize(best.x)
        total_nfev += refined.nfev
        if refined.cost < best.cost:
            best = refined
        else:
            break
    return best, total_nfev


def evaluate_host(path, loss):
    inputs, target, _, online_base, online_final, starts, check = load_host(path)
    result, nfev = fit_square(inputs, target, starts, loss)
    predictions = {"static": square_prediction_and_jacobian(inputs, result.x)[0],
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
    return {"host": path.stem, "architecture": architecture, "sample_count": len(target),
            "loss": loss, "formula_check_max_ms": check,
            "fit_success": bool(result.success), "fit_nfev": nfev,
            "fit_cost": float(result.cost), "fit_parameters": parameters.tolist(),
            "full_window": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--loss", choices=("log1p", "raw"), default="log1p")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if args.workers == 1:
        hosts = [evaluate_host(path, args.loss) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(partial(evaluate_host, loss=args.loss), paths))
    aggregate = {architecture: {"full_window": aggregate_metrics(hosts, architecture)}
                 for architecture in ("MI", "RTX")
                 if any(host["architecture"] == architecture for host in hosts)}
    print(json.dumps({"input_dir": str(args.input_dir), "loss": args.loss,
                      "hosts": hosts, "aggregate": aggregate}, separators=(",", ":")))


if __name__ == "__main__":
    main()
