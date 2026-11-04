#!/usr/bin/env python3
"""Fit the same scheduling features without the nonlinear latency shape."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from AnalyzeLatencyPhaseLeastSquares import load_host, metrics
from CompareLatencyGlobalLeastSquares import aggregate_metrics, segment_masks


def design_matrix(inputs):
    # Batch count is inactive in the deployed formula; total prompt duplicates
    # existing token columns, so these are the independent linear inputs.
    return np.column_stack((inputs[:, :6], inputs[:, 7:14]))


def fit_design(design, target, loss):
    scale = np.maximum(np.sqrt(np.mean(design * design, axis=0)), 1e-12)
    normalized = design / scale
    raw_coefficients, _, rank, _ = np.linalg.lstsq(normalized, target, rcond=None)
    if loss == "raw_ols":
        return raw_coefficients / scale, True, 0, int(rank)

    starts = [raw_coefficients]
    constant = np.zeros(normalized.shape[1])
    constant[0] = np.median(target)
    starts.append(constant)
    best = None
    for start in starts:
        def evaluate(coefficients):
            unbounded = normalized @ coefficients
            return np.maximum(unbounded, 0.0), unbounded

        def residual(coefficients):
            predicted, _ = evaluate(coefficients)
            if loss == "raw":
                return predicted - target
            return np.log1p(predicted) - np.log1p(target)

        def jacobian(coefficients):
            predicted, unbounded = evaluate(coefficients)
            positive = unbounded > 0.0
            return normalized * (positive if loss == "raw" else
                                 positive / (1.0 + predicted))[:, None]

        result = least_squares(residual, start, jac=jacobian, x_scale="jac",
                               max_nfev=1200, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        if best is None or result.cost < best.cost:
            best = result
    return best.x / scale, bool(best.success), int(best.nfev), int(rank)


def fit_linear(inputs, target, loss):
    return fit_design(design_matrix(inputs), target, loss)


def evaluate_host(path, loss):
    inputs, target, _, online_base, online_final, _, check = load_host(path)
    design = design_matrix(inputs)
    coefficients, success, nfev, rank = fit_design(design, target, loss)
    unbounded = design @ coefficients
    predictions = {"static": np.maximum(unbounded, 0.0),
                   "online_base": online_base, "online_final": online_final}
    segments = {
        name: {**{predictor: metrics(target[mask], prediction[mask])
                  for predictor, prediction in predictions.items()},
               "mean_running": float(np.mean(inputs[mask, 1])) if mask.any() else None,
               "mean_waiting": float(np.mean(inputs[mask, 7])) if mask.any() else None}
        for name, mask in segment_masks(inputs, target).items()
    }
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    return {"host": path.stem, "architecture": architecture, "sample_count": len(target),
            "loss": loss, "formula_check_max_ms": check,
            "fit_success": success,
            "fit_nfev": nfev, "design_rank": rank,
            "unbounded_negative_count": int(np.sum(unbounded < 0.0)),
            "fit_coefficients": coefficients.tolist(), "full_window": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--loss", choices=("log1p", "raw", "raw_ols"),
                        default="log1p")
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
