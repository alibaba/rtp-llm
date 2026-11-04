#!/usr/bin/env python3
"""Compare additive splines with interacting trees on existing latency features."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from AnalyzeLatencyPhaseLeastSquares import PARAMETER_SCALES, load_host, metrics
from CompareLatencyAugmentedSquareShape import inner_input
from CompareLatencyGlobalLeastSquares import segment_masks
from PlotLatencyHighPressureU import operating_masks


def fit_predict(inputs, target, train, method):
    if method == "interacting_tree":
        model = HistGradientBoostingRegressor(
            max_iter=300, max_leaf_nodes=31, min_samples_leaf=10,
            learning_rate=0.05, l2_regularization=1e-3,
            early_stopping=False, random_state=17)
        model.fit(inputs[train], np.log1p(target[train]))
        transformed = model.predict(inputs)
    else:
        columns = np.flatnonzero(np.ptp(inputs[train], axis=0) > 1e-12)
        spline = SplineTransformer(n_knots=7, degree=3, knots="quantile",
                                   extrapolation="linear", include_bias=False)
        basis = spline.fit_transform(inputs[train][:, columns])
        model = Ridge(alpha=1.0).fit(basis, np.log1p(target[train]))
        transformed = model.predict(spline.transform(inputs[:, columns]))
    return np.maximum(np.expm1(transformed), 0.0)


def score(inputs, target, prediction, selected, operating):
    masks = segment_masks(inputs, target)
    selected_segments = ("all", "low", "middle", "high", "short")
    result = {name: metrics(target[selected & masks[name]],
                            prediction[selected & masks[name]])
              for name in selected_segments}
    result["high_operating"] = {
        name: metrics(target[selected & masks["high"] & region],
                      prediction[selected & masks["high"] & region])
        for name, region in operating.items()
    }
    return result


def evaluate_host(args):
    path, high_fit = args
    inputs, target, event_time, _, _, _, check = load_host(path)
    count = len(target)
    if count < 100:
        raise ValueError(f"too few samples: {path}")
    parameters = np.asarray(high_fit["fit_parameters"])
    operating = operating_masks(inner_input(inputs, parameters), parameters[8] * PARAMETER_SCALES[8])
    random = np.random.default_rng(17)
    random_train = np.zeros(count, dtype=bool)
    random_train[random.permutation(count)[:count * 4 // 5]] = True
    time_train = np.zeros(count, dtype=bool)
    time_train[np.argsort(event_time)[:count * 4 // 5]] = True
    experiments = {}
    for split, train in (("all_fit", np.ones(count, dtype=bool)),
                         ("random_split", random_train), ("time_split", time_train)):
        experiments[split] = {}
        for method in ("additive_spline", "interacting_tree"):
            prediction = fit_predict(inputs, target, train, method)
            experiments[split][method] = {
                "train": score(inputs, target, prediction, train, operating),
                "held": score(inputs, target, prediction, ~train, operating)
                if not train.all() else None,
            }
    return {"host": path.stem, "architecture": high_fit["architecture"],
            "sample_count": count, "formula_check_max_ms": check,
            "experiments": experiments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--high-fit-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    fitted = {host["host"]: host for host in
              json.loads(args.high_fit_json.read_text())["hosts"]}
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if set(path.stem for path in paths) != set(fitted):
        parser.error("host sets differ between input and high-pressure fit")
    work = [(path, fitted[path.stem]) for path in paths]
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
