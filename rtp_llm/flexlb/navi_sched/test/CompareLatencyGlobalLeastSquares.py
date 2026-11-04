#!/usr/bin/env python3
"""Compare full-window static least squares with the logged online trajectory."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from AnalyzeLatencyPhaseLeastSquares import (
    fit, load_host, metrics, prediction_and_jacobian, pressure_masks, unpack,
)


WAITING_BINS = (("waiting_0", 0, 0), ("waiting_1_4", 1, 4),
                ("waiting_5_16", 5, 16), ("waiting_17_32", 17, 32),
                ("waiting_33_64", 33, 64), ("waiting_65_128", 65, 128),
                ("waiting_129_256", 129, 256),
                ("waiting_257_plus", 257, float("inf")))
SEGMENTS = ("all", "low", "middle", "high", "short") + tuple(
    name for name, _, _ in WAITING_BINS)
PREDICTORS = ("static", "online_base", "online_final")


def segment_masks(inputs, target):
    low, middle, high = pressure_masks(inputs)
    masks = {"all": np.ones(len(target), bool), "low": low,
             "middle": middle, "high": high, "short": target < 256}
    waiting = inputs[:, 7]
    masks.update({name: (waiting >= lower) & (waiting <= upper)
                  for name, lower, upper in WAITING_BINS})
    return masks


def evaluate_host(path, loss):
    inputs, target, _, online_base, online_final, starts, check = load_host(path)
    result = fit(inputs, target, starts, loss, "none")
    total_nfev = result.nfev
    for _ in range(3):
        if result.success:
            break
        refined = fit(inputs, target, [unpack(result.x)], loss, "none")
        total_nfev += refined.nfev
        if refined.cost < result.cost:
            result = refined
        else:
            break
    static = prediction_and_jacobian(inputs, result.x)[0]
    predictions = {"static": static, "online_base": online_base,
                   "online_final": online_final}
    all_metrics = {
        name: {**{predictor: metrics(target[mask], predictions[predictor][mask])
                  for predictor in PREDICTORS},
               "mean_running": float(np.mean(inputs[mask, 1])) if mask.any() else None,
               "mean_waiting": float(np.mean(inputs[mask, 7])) if mask.any() else None}
        for name, mask in segment_masks(inputs, target).items()
    }
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    return {"host": path.stem, "architecture": architecture,
            "sample_count": int(len(target)), "loss": loss,
            "formula_check_max_ms": check,
            "fit_success": bool(result.success), "fit_nfev": int(total_nfev),
            "fit_parameters": unpack(result.x).tolist(),
            "full_window": all_metrics}


def aggregate_metrics(hosts, architecture):
    selected = [host for host in hosts if host["architecture"] == architecture]
    summary = {"hosts": len(selected), "failed_full_fits": sum(
        not host["fit_success"] for host in selected)}
    for segment in SEGMENTS:
        records = [host["full_window"][segment] for host in selected]
        summary[segment] = {}
        segment_count = sum(record["static"]["n"] for record in records
                            if record["static"] is not None)
        summary[segment]["mean_running"] = (
            sum(record["mean_running"] * record["static"]["n"] for record in records
                if record["static"] is not None) / segment_count
            if segment_count else None)
        summary[segment]["mean_waiting"] = (
            sum(record["mean_waiting"] * record["static"]["n"] for record in records
                if record["static"] is not None) / segment_count
            if segment_count else None)
        for predictor in PREDICTORS:
            present = [record[predictor] for record in records
                       if record[predictor] is not None]
            sample_count = sum(record["n"] for record in present)
            if not sample_count:
                summary[segment][predictor] = {"n": 0, "mae_ms": None,
                                               "signed_error_ms": None,
                                               "overprediction_contribution_ms": None,
                                               "underprediction_contribution_ms": None}
                continue
            mae_ms = sum(record["mae_ms"] * record["n"] for record in present) / sample_count
            signed_error_ms = -sum(record["bias_ms"] * record["n"]
                                   for record in present) / sample_count
            summary[segment][predictor] = {
                "n": sample_count,
                "mae_ms": mae_ms,
                "signed_error_ms": signed_error_ms,
                "overprediction_contribution_ms": (mae_ms + signed_error_ms) / 2,
                "underprediction_contribution_ms": (mae_ms - signed_error_ms) / 2,
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--loss", choices=("log1p", "raw"), default="log1p")
    parser.add_argument("--hosts", nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    selected_paths = sorted(args.input_dir.glob("*.jsonl"))
    if args.hosts:
        selected_paths = [path for path in selected_paths if path.stem in args.hosts]
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers == 1:
        hosts = [evaluate_host(path, args.loss) for path in selected_paths]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(partial(evaluate_host, loss=args.loss), selected_paths))
    aggregate = {architecture: {"full_window": aggregate_metrics(
        hosts, architecture)}
        for architecture in ("MI", "RTX") if any(
            host["architecture"] == architecture for host in hosts)}
    print(json.dumps({"input_dir": str(args.input_dir), "loss": args.loss,
                      "hosts": hosts,
                      "aggregate": aggregate}, separators=(",", ":")))


if __name__ == "__main__":
    main()
