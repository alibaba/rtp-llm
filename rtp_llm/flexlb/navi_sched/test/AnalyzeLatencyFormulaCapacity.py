#!/usr/bin/env python3
"""Compare the deployed base formula with a flexible fit on identical inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from AnalyzeLatencyPhaseLeastSquares import load_host, metrics, pressure_masks


def evaluate(path):
    inputs, target, event_time, logged_base, logged_final, _, check = load_host(path)
    held = (event_time // 120_000_000) % 4 == 3
    low, middle, high = pressure_masks(inputs)
    model = HistGradientBoostingRegressor(
        max_iter=180, max_leaf_nodes=15, min_samples_leaf=30,
        learning_rate=0.06, l2_regularization=10.0, random_state=0)
    model.fit(inputs[~held], np.log1p(target[~held]))
    predicted = np.expm1(model.predict(inputs[held]))
    predicted = np.maximum(predicted, 0.0)
    outputs = {}
    for name, mask in (("all", held), ("low", held & low),
                       ("middle", held & middle), ("high", held & high),
                       ("short", held & (target < 256))):
        outputs[name] = {
            "flexible": metrics(target[mask], predicted[mask[held]]),
            "online_base": metrics(target[mask], logged_base[mask]),
            "online_final": metrics(target[mask], logged_final[mask]),
        }
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    return {"host": path.stem, "architecture": architecture,
            "formula_check_max_ms": check, "evaluations": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()
    for path in sorted(args.input_dir.glob("*.jsonl")):
        print(json.dumps(evaluate(path), separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
