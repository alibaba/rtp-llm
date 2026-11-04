#!/usr/bin/env python3
"""Block-bootstrap paired held-out errors across nodes sharing traffic."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from AnalyzeLatencyPhaseLeastSquares import (
    load_host, prediction_and_jacobian, pressure_masks,
)


BLOCK_US = 120_000_000
SEGMENTS = ("all", "low", "high", "short")
PREDICTORS = ("online_base", "online_final")


def block_comparison(blocks, repetitions=10000, seed=20261007):
    ordered = np.asarray([blocks[key] for key in sorted(blocks)], dtype=np.float64)
    counts = ordered[:, 0]
    static = ordered[:, 1]
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(ordered), size=(repetitions, len(ordered)))
    selected_counts = np.sum(counts[samples], axis=1)
    results = {"block_count": len(ordered), "sample_count": int(np.sum(counts))}
    for index, predictor in enumerate(PREDICTORS, start=2):
        difference = static - ordered[:, index]
        bootstrapped = np.sum(difference[samples], axis=1) / selected_counts
        results[predictor] = {
            "static_minus_online_mae_ms": float(np.sum(difference) / np.sum(counts)),
            "confidence_95_ms": np.percentile(bootstrapped, [2.5, 97.5]).tolist(),
            "static_worse_blocks": int(np.sum(difference > 0)),
            "static_better_blocks": int(np.sum(difference < 0)),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("held_model", type=Path)
    args = parser.parse_args()
    held_models = {}
    with args.held_model.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            held_models[row["host"]] = row["models"]["all"]["values"]
    groups = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    for path in sorted(args.input_dir.glob("*.jsonl")):
        if path.stem not in held_models:
            raise ValueError(f"missing held model: {path.stem}")
        inputs, target, event_time, base, final, _, _ = load_host(path)
        static = prediction_and_jacobian(
            inputs, np.asarray(held_models[path.stem]))[0]
        low, _, high = pressure_masks(inputs)
        block_number = event_time // BLOCK_US
        held = block_number % 4 == 3
        masks = {"all": held, "low": held & low, "high": held & high,
                 "short": held & (target < 256)}
        architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
        for segment, mask in masks.items():
            for block in np.unique(block_number[mask]):
                selected = mask & (block_number == block)
                groups[(architecture, segment)][int(block)] += np.array((
                    np.sum(selected), np.sum(np.abs(target[selected] - static[selected])),
                    np.sum(np.abs(target[selected] - base[selected])),
                    np.sum(np.abs(target[selected] - final[selected]))))
    output = {architecture: {segment: block_comparison(
        groups[(architecture, segment)]) for segment in SEGMENTS}
        for architecture in ("MI", "RTX")}
    print(json.dumps(output, separators=(",", ":")))


if __name__ == "__main__":
    main()
