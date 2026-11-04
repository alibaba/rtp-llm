#!/usr/bin/env python3
"""Test a learnable two-width transition against the deployed latency shape."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, LOWER, PARAMETER_SCALES, load_host, metrics,
    prediction_and_jacobian as baseline_prediction_and_jacobian, shape, unpack,
)
from CompareLatencyFixedCurvature import segment_masks


MAX_LOG_WIDTH_RATIO = np.log(1000.0)


def prediction_and_jacobian(inputs, values):
    p = unpack(values[:len(ACTIVE)]) * PARAMETER_SCALES[:17]
    log_ratio = values[-1]
    ratio_squared = np.exp(2.0 * log_ratio)
    curvature = p[8]
    broad_curvature = curvature * ratio_squared
    linear = (inputs[:, :7] @ p[:7] + inputs[:, 7:12] @ p[10:15] +
              inputs[:, 13] * p[16]) / 320.0
    zero_linear = p[0] / 320.0
    zero = np.full(len(linear), zero_linear)
    root = np.sqrt(linear * linear + curvature)
    broad_root = np.sqrt(linear * linear + broad_curvature)
    zero_root = np.sqrt(zero_linear * zero_linear + curvature)
    broad_zero_root = np.sqrt(zero_linear * zero_linear + broad_curvature)
    prompt = inputs[:, 2] + inputs[:, 3] + inputs[:, 8] + inputs[:, 9]
    nonlinear = 0.5 * (shape(linear, curvature) - shape(zero, curvature) +
                       shape(linear, broad_curvature) - shape(zero, broad_curvature))
    raw = p[7] + 1500.0 * nonlinear + p[9] * prompt + p[15] * inputs[:, 12]
    jac = np.zeros((len(inputs), 18))
    slope = 1500.0 * (1.0 + 0.5 * linear / root +
                      0.5 * linear / broad_root) / 320.0
    zero_slope = 1500.0 * (1.0 + 0.5 * zero_linear / zero_root +
                           0.5 * zero_linear / broad_zero_root) / 320.0
    jac[:, :7] = slope[:, None] * inputs[:, :7]
    jac[:, 0] -= zero_slope
    jac[:, 7] = 1.0
    jac[:, 8] = 375.0 * (1.0 / root - 1.0 / zero_root +
                         ratio_squared / broad_root -
                         ratio_squared / broad_zero_root)
    jac[:, 9] = prompt
    jac[:, 10:15] = slope[:, None] * inputs[:, 7:12]
    jac[:, 15] = inputs[:, 12]
    jac[:, 16] = slope * inputs[:, 13]
    jac[:, 17] = 750.0 * broad_curvature * (
        1.0 / broad_root - 1.0 / broad_zero_root)
    jac[:, :17] *= PARAMETER_SCALES[:17]
    jac[raw < 0] = 0.0
    return np.maximum(raw, 0.0), jac[:, list(ACTIVE) + [17]]


def fit_model(inputs, target, starts, loss, max_nfev, learn_width):
    lower = (np.r_[LOWER, 0.0] if learn_width else LOWER)
    upper = (np.r_[np.full(len(ACTIVE), np.inf), MAX_LOG_WIDTH_RATIO]
             if learn_width else np.full(len(ACTIVE), np.inf))
    best = None
    for start in starts:
        initial = np.clip(start, lower + 1e-9, upper - 1e-9)

        def evaluate(values):
            return (prediction_and_jacobian(inputs, values) if learn_width
                    else baseline_prediction_and_jacobian(inputs, values))

        def residual(values):
            prediction = evaluate(values)[0]
            return (np.log1p(prediction) - np.log1p(target)
                    if loss == "log1p" else prediction - target)

        def jacobian(values):
            prediction, jac = evaluate(values)
            return jac / (1.0 + prediction[:, None]) if loss == "log1p" else jac

        result = least_squares(residual, initial, jac=jacobian,
                               bounds=(lower, upper), x_scale="jac",
                               max_nfev=max_nfev, ftol=1e-7, xtol=1e-7,
                               gtol=1e-7)
        if best is None or result.cost < best.cost:
            best = result
    return best


def evaluate_host(path, loss, split_mode, max_nfev):
    inputs, target, event_time, _, online_final, starts, check = load_host(path)
    held = ((event_time // 120_000_000) % 4 == 3 if split_mode == "blocks"
            else event_time >= np.quantile(event_time, 0.75))
    train = ~held
    masks = segment_masks(inputs, target, event_time)
    initial = np.asarray(starts[0])[ACTIVE]
    # Identical r=1 predictions do not imply an identifiable width gradient.
    baseline = fit_model(inputs[train], target[train], [initial], loss,
                         max_nfev, False)
    starts_width = [np.r_[baseline.x, np.log(ratio)]
                    for ratio in (1.0, 2.0, 4.0, 16.0)]
    wide = fit_model(inputs[train], target[train], starts_width, loss,
                     max_nfev, True)
    experiments = {}
    for label, result in (("baseline", baseline), ("learned_width", wide)):
        prediction = (prediction_and_jacobian(inputs, result.x)[0]
                      if label == "learned_width" else
                      baseline_prediction_and_jacobian(inputs, result.x)[0])
        experiments[label] = {
            "parameters": unpack(result.x[:len(ACTIVE)]).tolist(),
            "width_ratio": float(np.exp(result.x[-1])) if label == "learned_width" else 1.0,
            "cost": float(result.cost), "success": bool(result.success),
            "nfev": int(result.nfev),
            "metrics": {split + "_" + name: metrics(target[split_mask & mask],
                                                     prediction[split_mask & mask])
                        for split, split_mask in (("train", train), ("held", held))
                        for name, mask in masks.items()},
        }
    return {"host": path.stem, "sample_count": len(target),
            "held_count": int(held.sum()), "formula_check_max_ms": check,
            "experiments": experiments,
            "online_held": {name: metrics(target[held & mask], online_final[held & mask])
                            for name, mask in masks.items()}}


def aggregate(hosts):
    result = {}
    for label in ("baseline", "learned_width"):
        result[label] = {}
        for segment in hosts[0]["experiments"][label]["metrics"]:
            records = [host["experiments"][label]["metrics"][segment]
                       for host in hosts]
            records = [record for record in records if record is not None]
            n = sum(record["n"] for record in records)
            result[label][segment] = {
                "n": n,
                **{field: sum(record[field] * record["n"] for record in records) / n
                   for field in ("mae_ms", "bias_ms", "mean_relative_error")}
            } if n else {"n": 0}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--loss", choices=("log1p", "raw"), default="log1p")
    parser.add_argument("--split-mode", choices=("blocks", "forward"), default="blocks")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=800)
    parser.add_argument("--hosts", nargs="*")
    args = parser.parse_args()
    paths = sorted(path for path in args.input_dir.glob("*.jsonl")
                   if path.stem.startswith("mi-") or "MI308" in path.stem)
    if args.hosts:
        paths = [path for path in paths if path.stem in args.hosts]
    if not paths or args.workers < 1:
        parser.error("select MI hosts and positive worker count")
    if args.workers == 1:
        hosts = [evaluate_host(path, args.loss, args.split_mode, args.max_nfev)
                 for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(evaluate_host, paths,
                                      [args.loss] * len(paths),
                                      [args.split_mode] * len(paths),
                                      [args.max_nfev] * len(paths)))
    print(json.dumps({"loss": args.loss, "split_mode": args.split_mode,
                      "hosts": hosts, "aggregate": aggregate(hosts)},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
