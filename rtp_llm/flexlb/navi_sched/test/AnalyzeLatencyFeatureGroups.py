#!/usr/bin/env python3
"""Measure group-level feature sensitivity on held high-pressure samples."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from AnalyzeLatencyPhaseLeastSquares import PARAMETER_SCALES, load_host
from CompareLatencyAugmentedSquareShape import inner_input
from PlotLatencyHighPressureU import operating_masks


GROUPS = {
    "running_count": (1,),
    "running_tokens": (2, 3, 4, 5),
    "waiting_count": (7,),
    "waiting_tokens": (8, 9, 10, 11),
    "preceding_count": (12,),
    "running_elapsed": (13,),
}


def evaluate_host(args):
    path, high_fit = args
    inputs, target, _, _, _, _, _ = load_host(path)
    count = len(target)
    random = np.random.default_rng(17)
    train = np.zeros(count, dtype=bool)
    train[random.permutation(count)[:count * 4 // 5]] = True
    high_held = (~train) & (inputs[:, 7] >= 33)
    model = HistGradientBoostingRegressor(
        max_iter=300, max_leaf_nodes=31, min_samples_leaf=10,
        learning_rate=0.05, l2_regularization=1e-3,
        early_stopping=False, random_state=17)
    model.fit(inputs[train], np.log1p(target[train]))
    x = inputs[high_held]
    y = target[high_held]
    parameters = np.asarray(high_fit["fit_parameters"])
    regions = operating_masks(inner_input(x, parameters), parameters[8] * PARAMETER_SCALES[8])

    def score(values):
        prediction = np.maximum(np.expm1(model.predict(values)), 0.0)
        return {name: float(np.mean(np.abs(prediction[selected] - y[selected])))
                if selected.any() else None
                for name, selected in {"all": np.ones(len(y), dtype=bool),
                                       **regions}.items()}

    baseline = score(x)
    changed = {}
    for name, columns in GROUPS.items():
        trials = []
        for repeat in range(3):
            permuted = x.copy()
            order = np.random.default_rng(200 + repeat).permutation(len(x))
            permuted[:, columns] = x[order][:, columns]
            trials.append(score(permuted))
        changed[name] = {region: float(np.mean([trial[region] for trial in trials]))
                         if baseline[region] is not None else None
                         for region in baseline}
    return {"host": path.stem, "architecture": high_fit["architecture"],
            "high_held_count": len(y),
            "region_counts": {name: int(np.count_nonzero(selected))
                              for name, selected in regions.items()},
            "baseline_mae_ms": baseline, "permuted_mae_ms": changed}


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
    print(json.dumps({"input_dir": str(args.input_dir), "groups": GROUPS,
                      "hosts": hosts}, separators=(",", ":")))


if __name__ == "__main__":
    main()
