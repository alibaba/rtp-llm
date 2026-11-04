#!/usr/bin/env python3
"""Compare matched 16-parameter latency replay predictions by hardware."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_predictions(path):
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if fields[0] == "P":
                key = (fields[1], int(fields[4]))
                if key in result:
                    raise ValueError(f"duplicate prediction: {key}")
                result[key] = (
                    int(fields[5]), float(fields[6]), float(fields[8]),
                    float(fields[9]),
                )
    return result


def summarize(baseline, candidate, hardware):
    if baseline.keys() != candidate.keys():
        raise ValueError("replays do not contain the same samples")
    groups = defaultdict(list)
    for key, original in baseline.items():
        other = candidate[key]
        if original[:2] != other[:2]:
            raise ValueError(f"sample changed: {key}")
        target = original[1]
        for name in ("all", hardware[key[0]], key[0]):
            groups[name].append((target, original[2], original[3],
                                 other[2], other[3]))
    for name in ("all", "MI308", "RTX5000"):
        rows = np.asarray(groups[name], dtype=np.float64)
        target = rows[:, 0]
        base_before = rows[:, 1] - target
        final_before = rows[:, 2] - target
        base_after = rows[:, 3] - target
        final_after = rows[:, 4] - target
        print(name, "n", len(rows),
              "base_MAE", np.mean(np.abs(base_before)),
              np.mean(np.abs(base_after)),
              "final_MAE", np.mean(np.abs(final_before)),
              np.mean(np.abs(final_after)),
              "base_bias", np.mean(base_before), np.mean(base_after),
              "final_bias", np.mean(final_before), np.mean(final_after),
              "final_P95_abs", np.quantile(np.abs(final_before), .95),
              np.quantile(np.abs(final_after), .95),
              "final_improved_fraction",
              np.mean(np.abs(final_after) < np.abs(final_before)))
    for name in sorted(groups):
        if name in ("all", "MI308", "RTX5000"):
            continue
        rows = np.asarray(groups[name], dtype=np.float64)
        print("host", name, "n", len(rows), "final_MAE",
              np.mean(np.abs(rows[:, 2] - rows[:, 0])),
              np.mean(np.abs(rows[:, 4] - rows[:, 0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    nodes = json.loads(args.manifest.read_text())["nodes"]
    hardware = {
        node["ip"].replace(".", "_"):
            "MI308" if "MI308" in node["deployment"] else "RTX5000"
        for node in nodes
    }
    summarize(read_predictions(args.baseline),
              read_predictions(args.candidate), hardware)


if __name__ == "__main__":
    main()
