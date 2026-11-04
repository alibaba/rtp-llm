#!/usr/bin/env python3
"""Test whether feature-to-latency mappings transfer between collection windows."""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from AnalyzeLatencyPhaseLeastSquares import PARAMETER_SCALES, load_host, metrics
from CompareLatencyAugmentedSquareShape import inner_input
from CompareLatencyGlobalLeastSquares import segment_masks
from PlotLatencyHighPressureU import operating_masks


def host_key(host, architecture):
    address = tuple(re.findall(r"\d+", host)[-4:])
    return architecture, address


def score(inputs, target, prediction, high_fit):
    masks = segment_masks(inputs, target)
    parameters = np.asarray(high_fit["fit_parameters"])
    operating = operating_masks(inner_input(inputs, parameters), parameters[8] * PARAMETER_SCALES[8])
    result = {name: metrics(target[masks[name]], prediction[masks[name]])
              for name in ("all", "low", "middle", "high", "short")}
    result["high_operating"] = {
        name: metrics(target[masks["high"] & selected],
                      prediction[masks["high"] & selected])
        for name, selected in operating.items()}
    return result


def evaluate_pair(args):
    a_path, b_path, a_fit, b_fit = args
    a_inputs, a_target, *_ = load_host(a_path)
    b_inputs, b_target, *_ = load_host(b_path)
    directions = {}
    for name, train_x, train_y, test_x, test_y, train_fit, test_fit in (
        ("A_to_B", a_inputs, a_target, b_inputs, b_target, a_fit, b_fit),
        ("B_to_A", b_inputs, b_target, a_inputs, a_target, b_fit, a_fit),
    ):
        model = HistGradientBoostingRegressor(
            max_iter=300, max_leaf_nodes=31, min_samples_leaf=10,
            learning_rate=0.05, l2_regularization=1e-3,
            early_stopping=False, random_state=17)
        model.fit(train_x, np.log1p(train_y))
        train_prediction = np.maximum(np.expm1(model.predict(train_x)), 0.0)
        test_prediction = np.maximum(np.expm1(model.predict(test_x)), 0.0)
        directions[name] = {
            "train": score(train_x, train_y, train_prediction, train_fit),
            "test": score(test_x, test_y, test_prediction, test_fit),
        }
    return {"architecture": a_fit["architecture"], "a_host": a_path.stem,
            "b_host": b_path.stem, "directions": directions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-source", type=Path, required=True)
    parser.add_argument("--b-source", type=Path, required=True)
    parser.add_argument("--a-high-fit", type=Path, required=True)
    parser.add_argument("--b-high-fit", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    a_fit = {host_key(row["host"], row["architecture"]): row for row in
             json.loads(args.a_high_fit.read_text())["hosts"]}
    b_fit = {host_key(row["host"], row["architecture"]): row for row in
             json.loads(args.b_high_fit.read_text())["hosts"]}
    a_paths = {key: args.a_source / (row["host"] + ".jsonl")
               for key, row in a_fit.items()}
    b_paths = {key: args.b_source / (row["host"] + ".jsonl")
               for key, row in b_fit.items()}
    keys = sorted(set(a_paths) & set(b_paths))
    if not keys:
        parser.error("no shared node addresses")
    work = [(a_paths[key], b_paths[key], a_fit[key], b_fit[key]) for key in keys]
    if args.workers == 1:
        pairs = [evaluate_pair(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pairs = list(executor.map(evaluate_pair, work))
    print(json.dumps({"paired_nodes": len(keys), "pairs": pairs},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
