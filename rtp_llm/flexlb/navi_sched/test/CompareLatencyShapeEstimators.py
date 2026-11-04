#!/usr/bin/env python3
"""Estimate the outer latency curve with u fixed from high-pressure fitting."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from AnalyzeLatencyPhaseLeastSquares import ACTIVE, PARAMETER_SCALES, load_host, metrics, prediction_and_jacobian
from CompareLatencyAugmentedSquareShape import inner_input
from PlotLatencyHighPressureU import operating_masks


def fit_shape(u, adjusted_target, train, method):
    if method == "isotonic":
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(u[train], adjusted_target[train])
        return model.predict(u)
    scale = max(float(np.percentile(np.abs(u[train]), 95)), 1.0)
    transformed = np.arcsinh(u / scale).reshape(-1, 1)
    spline = SplineTransformer(n_knots=12, degree=3, knots="quantile",
                               extrapolation="linear", include_bias=False)
    basis = spline.fit_transform(transformed[train])
    model = Ridge(alpha=1.0).fit(basis, adjusted_target[train])
    return model.predict(spline.transform(transformed))


def evaluate_host(args):
    path, fitted = args
    inputs, target, event_time, _, _, _, _ = load_host(path)
    high = inputs[:, 7] >= 33
    x = inputs[high]
    y = target[high]
    time = event_time[high]
    parameters = np.asarray(fitted["fit_parameters"])
    u = inner_input(x, parameters)
    physical = parameters * PARAMETER_SCALES[:17]
    prompt = x[:, 2] + x[:, 3] + x[:, 8] + x[:, 9]
    other = physical[7] + physical[9] * prompt + physical[15] * x[:, 12]
    adjusted = y - other
    regions = operating_masks(u, physical[8])
    original = prediction_and_jacobian(x, parameters[ACTIVE], "none")[0]
    count = len(y)
    random = np.random.default_rng(17)
    random_train = np.zeros(count, dtype=bool)
    random_train[random.permutation(count)[:count * 4 // 5]] = True
    time_train = np.zeros(count, dtype=bool)
    time_train[np.argsort(time)[:count * 4 // 5]] = True
    experiments = {}
    for split, train in (("all_fit", np.ones(count, dtype=bool)),
                         ("random_split", random_train), ("time_split", time_train)):
        experiments[split] = {}
        for method in ("isotonic", "spline"):
            predicted = np.maximum(other + fit_shape(u, adjusted, train, method), 0.0)
            held = ~train
            experiments[split][method] = {
                "train": metrics(y[train], predicted[train]),
                "held": metrics(y[held], predicted[held]) if held.any() else None,
                "regions": {name: metrics(y[selected], predicted[selected])
                            for name, selected in regions.items()} if split == "all_fit" else None,
            }
    return {"host": fitted["host"], "architecture": fitted["architecture"],
            "high_sample_count": count, "original": metrics(y, original),
            "experiments": experiments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--high-fit-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    fitted = json.loads(args.high_fit_json.read_text())
    by_host = {row["host"]: row for row in fitted["hosts"]}
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if set(path.stem for path in paths) != set(by_host):
        parser.error("host sets differ between input and high-pressure fit")
    work = [(path, by_host[path.stem]) for path in paths]
    if args.workers == 1:
        hosts = [evaluate_host(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(evaluate_host, work))
    print(json.dumps({"input_dir": str(args.input_dir),
                      "u_source": str(args.high_fit_json), "hosts": hosts},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
