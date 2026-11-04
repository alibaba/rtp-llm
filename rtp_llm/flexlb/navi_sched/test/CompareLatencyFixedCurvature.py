#!/usr/bin/env python3
"""Refit the deployed latency formula with fixed curvature on time-held MI samples."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, LOWER, PARAMETER_SCALES, load_host, metrics, prediction_and_jacobian, pressure_masks,
    unpack,
)


CURVATURE_COLUMN = int(np.flatnonzero(ACTIVE == 8)[0])
FREE_COLUMNS = np.asarray([i for i in range(len(ACTIVE)) if i != CURVATURE_COLUMN])
CURVATURES = (0.01, 0.05, 0.1, 0.25, 1.0, 10.0, 100.0, 300.0,
              900.0, 3000.0, 10000.0)
SHAPES = {"convex": None, "concave_05": 0.5, "concave_09": 0.9}


def full_parameters(values, curvature):
    if curvature is None:
        return values
    full = np.empty(len(ACTIVE))
    full[FREE_COLUMNS] = values
    full[CURVATURE_COLUMN] = curvature / PARAMETER_SCALES[8]
    return full


def model_prediction_and_jacobian(inputs, active, shape_kind):
    if shape_kind == "convex":
        return prediction_and_jacobian(inputs, active)
    c = SHAPES[shape_kind]
    p = unpack(active) * PARAMETER_SCALES[:17]
    linear = (inputs[:, :7] @ p[:7] + inputs[:, 7:12] @ p[10:15] +
              inputs[:, 13] * p[16]) / 320.0
    zero_linear = p[0] / 320.0
    root = np.sqrt(linear * linear + p[8])
    zero_root = np.sqrt(zero_linear * zero_linear + p[8])
    prompt = inputs[:, 2] + inputs[:, 3] + inputs[:, 8] + inputs[:, 9]
    raw = (p[7] + 1500.0 * (linear - zero_linear - c * (root - zero_root)) +
           p[9] * prompt + p[15] * inputs[:, 12])
    jac = np.zeros((len(inputs), 17))
    slope = 1500.0 * (1.0 - c * linear / root) / 320.0
    jac[:, :7] = slope[:, None] * inputs[:, :7]
    jac[:, 0] -= 1500.0 * (1.0 - c * zero_linear / zero_root) / 320.0
    jac[:, 7] = 1.0
    jac[:, 8] = -750.0 * c * (1.0 / root - 1.0 / zero_root)
    jac[:, 9] = prompt
    jac[:, 10:15] = slope[:, None] * inputs[:, 7:12]
    jac[:, 15] = inputs[:, 12]
    jac[:, 16] = slope * inputs[:, 13]
    jac *= PARAMETER_SCALES[:17]
    jac[raw < 0] = 0.0
    return np.maximum(raw, 0.0), jac[:, ACTIVE]


def fit_curvature(inputs, target, starts, curvature, loss, max_nfev=400,
                  shape_kind="convex"):
    columns = np.arange(len(ACTIVE)) if curvature is None else FREE_COLUMNS
    lower = LOWER[columns]
    best = None
    for start in starts:
        initial = np.maximum(np.asarray(start)[columns], lower + 1e-9)

        def evaluate(values):
            return model_prediction_and_jacobian(
                inputs, full_parameters(values, curvature), shape_kind)

        def residual(values):
            prediction = evaluate(values)[0]
            return (np.log1p(prediction) - np.log1p(target)
                    if loss == "log1p" else prediction - target)

        def jacobian(values):
            prediction, jac = evaluate(values)
            selected = jac[:, columns]
            return selected / (1.0 + prediction[:, None]) if loss == "log1p" else selected

        result = least_squares(residual, initial, jac=jacobian,
                               bounds=(lower, np.full(len(columns), np.inf)),
                               x_scale="jac", max_nfev=max_nfev,
                               ftol=1e-7, xtol=1e-7, gtol=1e-7)
        if best is None or result.cost < best.cost:
            best = result
    return best


def segment_masks(inputs, target, event_time):
    low, middle, high = pressure_masks(inputs)
    # Compare adjacent 30-second blocks; the threshold rejects near-flat noise.
    blocks = event_time // 30_000_000
    block_means = {int(block): float(np.mean(inputs[blocks == block, 7]))
                   for block in np.unique(blocks)}
    rising = np.array([block_means.get(int(block), 0) -
                       block_means.get(int(block) - 1, block_means[int(block)]) >= 4
                       for block in blocks])
    return {"all": np.ones(len(target), bool), "low": low, "middle": middle,
            "high": high, "high_rising": high & rising,
            "high_other": high & ~rising, "short": target < 256}


def evaluate_host(path, loss, curvatures=CURVATURES, max_nfev=400,
                  split_mode="blocks", shape_kind="convex"):
    inputs, target, event_time, online_base, online_final, starts, check = load_host(path)
    held = ((event_time // 120_000_000) % 4 == 3 if split_mode == "blocks"
            else event_time >= np.quantile(event_time, 0.75))
    train = ~held
    masks = segment_masks(inputs, target, event_time)
    start = np.asarray(starts[0])[ACTIVE]
    free = fit_curvature(inputs[train], target[train], [start], None, loss,
                         max_nfev, shape_kind)
    fitted = {}
    for curvature in curvatures:
        fitted[str(curvature)] = fit_curvature(
            inputs[train], target[train], [start, free.x],
            curvature, loss, max_nfev, shape_kind)
    free = fit_curvature(inputs[train], target[train],
                         [start, free.x] + [full_parameters(result.x, k)
                                            for k, result in zip(curvatures, fitted.values())],
                         None, loss, max_nfev, shape_kind)
    experiments = {}
    for label, curvature in [("free", None)] + [(str(k), k) for k in curvatures]:
        result = free if curvature is None else fitted[label]
        parameters = full_parameters(result.x, curvature)
        predicted = model_prediction_and_jacobian(inputs, parameters, shape_kind)[0]
        experiments[label] = {
            "curvature": float(parameters[CURVATURE_COLUMN] * PARAMETER_SCALES[8]),
            "cost": float(result.cost), "success": bool(result.success),
            "nfev": int(result.nfev),
            "metrics": {split + "_" + name: metrics(target[split_mask & mask],
                                                     predicted[split_mask & mask])
                        for split, split_mask in (("train", train), ("held", held))
                        for name, mask in masks.items()},
        }
    return {"host": path.stem, "loss": loss, "split_mode": split_mode,
            "shape": shape_kind,
            "sample_count": len(target),
            "held_count": int(held.sum()), "formula_check_max_ms": check,
            "online_held": {name: metrics(target[held & mask], online_final[held & mask])
                            for name, mask in masks.items()},
            "experiments": experiments}


def aggregate(hosts):
    result = {}
    for label in hosts[0]["experiments"]:
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
    parser.add_argument("--hosts", nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=400)
    parser.add_argument("--split-mode", choices=("blocks", "forward"), default="blocks")
    parser.add_argument("--shape", choices=tuple(SHAPES), default="convex")
    parser.add_argument("--curvatures", type=float, nargs="*", default=CURVATURES)
    args = parser.parse_args()
    paths = sorted(path for path in args.input_dir.glob("*.jsonl")
                   if path.stem.startswith("mi-") or "MI308" in path.stem)
    if args.hosts:
        paths = [path for path in paths if path.stem in args.hosts]
    if not paths or args.workers < 1 or any(k <= 0 for k in args.curvatures):
        parser.error("select MI hosts, positive worker count and positive curvatures")
    if args.workers == 1:
        hosts = [evaluate_host(path, args.loss, args.curvatures, args.max_nfev,
                               args.split_mode, args.shape)
                 for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(evaluate_host, paths,
                                      [args.loss] * len(paths),
                                      [args.curvatures] * len(paths),
                                      [args.max_nfev] * len(paths),
                                      [args.split_mode] * len(paths),
                                      [args.shape] * len(paths)))
    print(json.dumps({"loss": args.loss, "curvatures": args.curvatures,
                      "split_mode": args.split_mode, "shape": args.shape,
                      "hosts": hosts, "aggregate": aggregate(hosts)},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
