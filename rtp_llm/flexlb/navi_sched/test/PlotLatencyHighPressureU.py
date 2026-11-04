#!/usr/bin/env python3
"""Fit high-pressure samples only, then plot observed latency against fitted u."""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from AnalyzeLatencyPhaseLeastSquares import ACTIVE, PARAMETER_SCALES, fit, load_host, metrics, shape, unpack
from CompareLatencyAugmentedSquareShape import inner_input


def operating_masks(u, curvature):
    normalized = u / np.sqrt(curvature)
    return {"negative_far": normalized < -3.0,
            "transition": np.abs(normalized) <= 3.0,
            "positive_far": normalized > 3.0}


def evaluate_host(path, baselines):
    inputs, target, _, _, _, online_starts, check = load_host(path)
    high = inputs[:, 7] >= 33
    high_inputs = inputs[high]
    high_target = target[high]
    if len(high_target) < 100:
        raise ValueError(f"too few high-pressure samples: {path}")
    starts = online_starts + [row["fit_parameters"] for row in baselines]
    results = [fit(high_inputs, high_target, [start], "raw", "none")
               for start in starts]
    best = min(results, key=lambda result: result.cost)
    total_nfev = sum(result.nfev for result in results)
    for _ in range(3):
        if best.success:
            break
        refined = fit(high_inputs, high_target, [unpack(best.x)], "raw", "none")
        total_nfev += refined.nfev
        if refined.cost < best.cost:
            best = refined
        else:
            break
    parameters = unpack(best.x)
    physical = parameters * PARAMETER_SCALES[:17]
    u = inner_input(high_inputs, parameters)
    u0 = physical[0] / 320.0
    prompt = high_inputs[:, 2] + high_inputs[:, 3] + high_inputs[:, 8] + high_inputs[:, 9]
    linear_other = physical[7] + physical[9] * prompt + physical[15] * high_inputs[:, 12]
    adjusted = high_target - linear_other
    nonlinear = 1500.0 * (shape(u, physical[8]) -
                          shape(np.asarray([u0]), physical[8])[0])
    prediction = np.maximum(linear_other + nonlinear, 0.0)
    operating_regions = {
        name: {"count": int(np.count_nonzero(selected)),
               "metrics": metrics(high_target[selected], prediction[selected])}
        for name, selected in operating_masks(u, physical[8]).items()
    }
    order = np.argsort(u)
    binned_curve = []
    oracle_metrics = {}
    for bin_count in (30, 100):
        raw_prediction = np.empty(len(u))
        adjusted_prediction = np.empty(len(u))
        bins = [indices for indices in np.array_split(order, bin_count) if len(indices)]
        for indices in bins:
            raw_prediction[indices] = np.median(high_target[indices])
            adjusted_prediction[indices] = (
                np.median(adjusted[indices]) + linear_other[indices])
            if bin_count == 30:
                binned_curve.append({
                    "u_median": float(np.median(u[indices])),
                    "target_median_ms": float(np.median(high_target[indices])),
                    "target_p10_ms": float(np.percentile(high_target[indices], 10)),
                    "target_p90_ms": float(np.percentile(high_target[indices], 90)),
                    "adjusted_median_ms": float(np.median(adjusted[indices])),
                    "adjusted_p10_ms": float(np.percentile(adjusted[indices], 10)),
                    "adjusted_p90_ms": float(np.percentile(adjusted[indices], 90)),
                    "formula_median_ms": float(np.median(nonlinear[indices])),
                })
        oracle_metrics[str(bin_count)] = {
            "raw_target": metrics(high_target, raw_prediction),
            "adjusted_target": metrics(high_target, np.maximum(adjusted_prediction, 0.0)),
        }
    comparable = [result for result in results if result.cost <= best.cost * 1.01]
    rank_correlation = [spearmanr(u, inner_input(high_inputs, unpack(result.x))).statistic
                        for result in comparable]
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    summary = {"host": path.stem, "architecture": architecture,
               "high_sample_count": len(high_target), "loss": "raw",
               "fit_cost": float(best.cost), "fit_success": bool(best.success),
               "fit_nfev": total_nfev, "fit_parameters": parameters.tolist(),
               "formula_check_max_ms": check, "metrics": metrics(high_target, prediction),
               "operating_regions": operating_regions,
               "binned_oracle": oracle_metrics, "binned_curve": binned_curve,
               "u_percentiles": np.percentile(u, [1, 5, 50, 95, 99]).tolist(),
               "target_percentiles_ms": np.percentile(high_target, [5, 50, 95]).tolist(),
               "near_optimal_starts": len(comparable),
               "minimum_u_rank_correlation": (float(np.min(rank_correlation))
                                              if rank_correlation else None)}
    plot_data = {"u": u, "target": high_target, "adjusted": adjusted,
                 "nonlinear": nonlinear}
    return summary, plot_data


def display_host(host):
    parts = re.findall(r"\d+", host)
    return ".".join(parts[-4:]) if len(parts) >= 4 else host


def draw_group(label, architecture, rows, output):
    figure, axes = plt.subplots(5, 2, figsize=(16, 22), layout="constrained")
    for axis, (summary, data) in zip(axes.flat, rows):
        u = data["u"]
        target = data["target"] / 1000.0
        adjusted = data["adjusted"] / 1000.0
        nonlinear = data["nonlinear"] / 1000.0
        order = np.argsort(u)
        bins = [indices for indices in np.array_split(order, 30) if len(indices)]
        center = np.array([np.median(u[indices]) for indices in bins])
        raw_median = np.array([np.median(target[indices]) for indices in bins])
        adjusted_median = np.array([np.median(adjusted[indices]) for indices in bins])
        lower = np.array([np.percentile(adjusted[indices], 10) for indices in bins])
        upper = np.array([np.percentile(adjusted[indices], 90) for indices in bins])
        sampled = order[::max(1, len(order) // 1500)]
        curvature = summary["fit_parameters"][8] * PARAMETER_SCALES[8]
        boundary = 3.0 * np.sqrt(curvature)
        axis.axvspan(-boundary, boundary, color="#e4bb63", alpha=0.12, zorder=0,
                     label="Nonlinear transition" if axis is axes.flat[0] else None)
        axis.scatter(u[sampled], target[sampled], s=3, alpha=0.08,
                     color="#66747c", rasterized=True)
        axis.scatter(center, raw_median, color="#66747c", s=18,
                     label="Observed latency" if axis is axes.flat[0] else None)
        axis.vlines(center, lower, upper, color="#1d857c", alpha=0.35, lw=2)
        axis.scatter(center, adjusted_median, color="#1d857c", s=18,
                     label="Observed minus other terms" if axis is axes.flat[0] else None)
        axis.plot(u[order], nonlinear[order], color="#d04f36", lw=1.6, ls="--",
                  label="Fitted F(u)" if axis is axes.flat[0] else None)
        threshold = max(1.0, np.percentile(np.abs(u), 50) * 0.01)
        axis.set_xscale("symlog", linthresh=threshold)
        axis.set_xlim(np.percentile(u, 1), np.percentile(u, 99))
        axis.set_title(f"{display_host(summary['host'])} | MAE {summary['metrics']['mae_ms']/1000:.1f}s"
                       f" | K {curvature:.3g}")
        axis.grid(alpha=0.2)
        axis.set_xlabel("Fitted u (sorted left to right)")
        axis.set_ylabel("Latency / contribution (s)")
    for axis in list(axes.flat)[len(rows):]:
        axis.set_visible(False)
    figure.suptitle(f"{label} / {architecture}: high-pressure fit (waiting >= 33)")
    axes.flat[0].legend(loc="upper left", fontsize=7, framealpha=0.85)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--raw-baseline", type=Path, required=True)
    parser.add_argument("--log-baseline", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    raw = {host["host"]: host for host in json.loads(args.raw_baseline.read_text())["hosts"]}
    log = {host["host"]: host for host in json.loads(args.log_baseline.read_text())["hosts"]}
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if set(path.stem for path in paths) != set(raw) or set(raw) != set(log):
        parser.error("host sets differ between input and baselines")
    work = [(path, (raw[path.stem], log[path.stem])) for path in paths]
    if args.workers == 1:
        evaluated = [evaluate_host(*item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            evaluated = list(executor.map(evaluate_host_args, work))
    for architecture in ("MI", "RTX"):
        rows = [(summary, plot_data) for summary, plot_data in evaluated
                if summary["architecture"] == architecture]
        draw_group(args.label, architecture, rows,
                   args.output_prefix.with_name(
                       f"{args.output_prefix.name}-{architecture}.png"))
    print(json.dumps({"label": args.label, "input_dir": str(args.input_dir),
                      "hosts": [summary for summary, _ in evaluated]},
                     separators=(",", ":")))


def evaluate_host_args(args):
    return evaluate_host(*args)


if __name__ == "__main__":
    main()
