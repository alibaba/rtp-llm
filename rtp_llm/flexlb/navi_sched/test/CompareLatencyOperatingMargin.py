#!/usr/bin/env python3
"""Test whether repelling u/sqrt(k) from zero improves held-out latency fits."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, LOWER, PARAMETER_SCALES, load_host, metrics, prediction_and_jacobian,
    pressure_masks, unpack,
)


def operating_ratio(inputs, parameters):
    full = unpack(parameters) * PARAMETER_SCALES[:17]
    u = (inputs[:, :7] @ full[:7] + inputs[:, 7:12] @ full[10:15] +
         inputs[:, 13] * full[16]) / 320.0
    return u / np.sqrt(full[8])


def margin_residual_and_jacobian(inputs, parameters, margin, strength, mode):
    full = unpack(parameters) * PARAMETER_SCALES[:17]
    curvature = full[8]
    ratio = operating_ratio(inputs, parameters)
    multiplier = np.sqrt(strength)
    if mode == "center_repulsion":
        active = np.abs(ratio) < margin
        residual = multiplier * np.maximum(0.0, margin - np.abs(ratio))
        derivative = -multiplier * np.sign(ratio) * active
    else:
        high = pressure_masks(inputs)[2]
        active = high & ((ratio < 0.0) | (ratio > margin))
        residual = multiplier * np.where(
            high, np.where(ratio < 0.0, ratio,
                           np.maximum(0.0, ratio - margin)), 0.0)
        derivative = multiplier * active

    du = np.zeros((len(inputs), 17))
    du[:, :7] = inputs[:, :7] / 320.0
    du[:, 10:15] = inputs[:, 7:12] / 320.0
    du[:, 16] = inputs[:, 13] / 320.0
    du *= PARAMETER_SCALES[:17]
    jacobian = derivative[:, None] * du[:, ACTIVE] / np.sqrt(curvature)
    curvature_column = int(np.flatnonzero(ACTIVE == 8)[0])
    jacobian[:, curvature_column] += derivative * (-ratio / (2.0 * curvature)) * PARAMETER_SCALES[8]
    return residual, jacobian


def fit(inputs, target, initial, margin, strength, max_nfev, mode):
    initial = np.maximum(np.asarray(initial, dtype=np.float64)[ACTIVE], LOWER + 1e-9)

    def evaluate(parameters):
        prediction, jacobian = prediction_and_jacobian(inputs, parameters)
        residual = np.log1p(prediction) - np.log1p(target)
        jacobian = jacobian / (1.0 + prediction[:, None])
        if strength:
            penalty, penalty_jacobian = margin_residual_and_jacobian(
                inputs, parameters, margin, strength, mode)
            residual = np.concatenate((residual, penalty))
            jacobian = np.vstack((jacobian, penalty_jacobian))
        return residual, jacobian

    cache = [None, None]

    def cached(parameters):
        if cache[0] is None or not np.array_equal(parameters, cache[0]):
            cache[:] = [parameters.copy(), evaluate(parameters)]
        return cache[1]

    return least_squares(
        lambda parameters: cached(parameters)[0], initial,
        jac=lambda parameters: cached(parameters)[1],
        bounds=(LOWER, np.full(len(ACTIVE), np.inf)),
        x_scale="jac", max_nfev=max_nfev,
        ftol=1e-7, xtol=1e-7, gtol=1e-7,
    )


def ratio_summary(ratio):
    if not len(ratio):
        return None
    return {
        "count": int(len(ratio)),
        "negative_fraction": float(np.mean(ratio < 0)),
        "transition_fraction": float(np.mean(np.abs(ratio) <= 3)),
        "median": float(np.median(ratio)),
        "p10": float(np.percentile(ratio, 10)),
        "p90": float(np.percentile(ratio, 90)),
    }


def evaluate_host(path, margin, strengths, max_nfev, mode):
    inputs, target, event_time, _, online_final, starts, check = load_host(path)
    held = (event_time // 30_000_000) % 4 == 3
    train = ~held
    low, middle, high = pressure_masks(inputs)
    masks = {"all": np.ones(len(target), dtype=bool),
             "low": low, "middle": middle, "high": high}
    baseline = min(
        (fit(inputs[train], target[train], start, margin, 0.0, max_nfev, mode)
         for start in starts),
        key=lambda result: result.cost,
    )
    records = {}
    for strength in (0.0, *strengths):
        result = (baseline if strength == 0 else min(
            (fit(inputs[train], target[train], start, margin,
                 strength, max_nfev, mode)
             for start in [unpack(baseline.x), *starts]),
            key=lambda candidate: candidate.cost,
        ))
        prediction = prediction_and_jacobian(inputs, result.x)[0]
        ratio = operating_ratio(inputs, result.x)
        records[str(strength)] = {
            "success": bool(result.success), "nfev": int(result.nfev),
            "curvature": float(unpack(result.x)[8] * PARAMETER_SCALES[8]),
            "metrics": {name: metrics(target[held & mask], prediction[held & mask])
                        for name, mask in masks.items()},
            "training_metrics": metrics(target[train], prediction[train]),
            "ratio": {name: ratio_summary(ratio[mask]) for name, mask in masks.items()},
        }
    return {"host": path.stem, "sample_count": len(target),
            "formula_check_max_ms": check, "margin": margin, "mode": mode,
            "online_held": {name: metrics(target[held & mask], online_final[held & mask])
                            for name, mask in masks.items()},
            "experiments": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--hosts", nargs="*")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--mode", choices=("center_repulsion", "high_band"),
                        default="center_repulsion")
    parser.add_argument("--strengths", type=float, nargs="*",
                        default=(0.001, 0.01, 0.1, 1.0))
    parser.add_argument("--max-nfev", type=int, default=180)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.margin <= 0 or any(value <= 0 for value in args.strengths):
        parser.error("margin and strengths must be positive")
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if args.hosts:
        paths = [path for path in paths if path.stem in args.hosts]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(
            evaluate_host, paths, [args.margin] * len(paths),
            [args.strengths] * len(paths), [args.max_nfev] * len(paths),
            [args.mode] * len(paths)))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
