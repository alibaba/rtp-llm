#!/usr/bin/env python3
"""Run matched latency-scale replay ablations and summarize complete traces."""

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np


RATES = (0.25, 0.5, 1, 2, 4, 8)
DEFAULT_STANDARDS = ("prediction", "loss", "log_abs", "mse_loss")
STANDARDS = {
    "prediction": ("prediction", "prediction-history-2"),
    "loss": ("loss-damped", "loss-damped-history"),
    "log_abs": ("log-abs", "log-abs-history"),
    "mse_loss": ("mse", "mse-history"),
    "mse_round2": ("mse-round2", "mse-round2-history"),
    "mse_round3": ("mse-round3", "mse-round3-history"),
}
EXPECTED_PREDICTIONS = 54625
MAX_WORKERS = 6
MAX_GRID_POINTS = 64


def predictions(path):
    rows = {}
    with path.open() as stream:
        for line in stream:
            fields = line.split("\t")
            if fields[0] != "P":
                continue
            key = (fields[1], int(fields[4]))
            if key in rows:
                raise ValueError(f"duplicate prediction {key} in {path}")
            rows[key] = (int(fields[5]), float(fields[6]),
                         float(fields[8]), float(fields[9]))
    if len(rows) != EXPECTED_PREDICTIONS:
        raise ValueError(f"incomplete trace {path}: {len(rows)} predictions")
    return rows


def replay(root, output, standard, rate, library_path):
    binary, history = STANDARDS[standard]
    path = output / f"{standard}-lr-{rate}.tsv"
    command = [str(root / f"{binary}-replay"), str(root / history),
               str(path), "4", "20261010", str(rate), "0.8", "all",
               "1", "parameter_rates:17=0,18=0"]
    env = dict(os.environ, LD_LIBRARY_PATH=library_path)
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode:
        return standard, rate, path, result.stderr[-2000:] or f"exit {result.returncode}"
    return standard, rate, path, None


def metrics(baseline, candidate, hardware):
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate sample keys differ")
    result = {}
    for group in ("all", "MI308", "RTX5000"):
        selected = [key for key in baseline if group == "all" or
                    hardware[key[0]] == group]
        if not selected:
            raise ValueError(f"no samples in {group}")
        original = np.asarray([baseline[key] for key in selected])
        updated = np.asarray([candidate[key] for key in selected])
        if not np.array_equal(original[:, :2], updated[:, :2]):
            raise ValueError(f"changed target or timestamp in {group}")
        target = original[:, 1]
        result[group] = {"n": len(selected)}
        for label, column in (("base", 2), ("final", 3)):
            error = updated[:, column] - target
            result[group][f"{label}_mae_ms"] = float(np.abs(error).mean())
            result[group][f"{label}_bias_ms"] = float(error.mean())
            result[group][f"{label}_p95_abs_ms"] = float(
                np.quantile(np.abs(error), 0.95))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--rates", default=",".join(str(rate) for rate in RATES))
    parser.add_argument("--standards", default=",".join(DEFAULT_STANDARDS))
    parser.add_argument("--summary-name", default="summary.json")
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"workers must be between 1 and {MAX_WORKERS}")
    rates = tuple(float(value) for value in args.rates.split(","))
    standards = tuple(args.standards.split(","))
    if (not rates or len(set(rates)) != len(rates) or
            any(not math.isfinite(rate) or rate <= 0 for rate in rates) or
            not standards or len(set(standards)) != len(standards) or
            len(rates) * len(standards) > MAX_GRID_POINTS or
            any(standard not in STANDARDS for standard in standards) or
            Path(args.summary_name).name != args.summary_name):
        parser.error("invalid rates, standards, or summary name")
    output = args.root / "scale-grid"
    output.mkdir(exist_ok=True)
    hardware = {node["ip"].replace(".", "_"):
                "MI308" if "MI308" in node["deployment"] else "RTX5000"
                for node in json.loads(args.manifest.read_text())["nodes"]}
    baseline = predictions(args.root / "d351-baseline-lr1.tsv")
    library_path = str(Path(__file__).parents[1] /
                       "release/navi_sched_cpp-0.1.0/lib")
    grid = [(standard, rate) for standard in standards for rate in rates]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(replay, args.root, output, standard, rate,
                            library_path): (standard, rate)
                for standard, rate in grid}
        for job in concurrent.futures.as_completed(jobs):
            standard, rate, path, error = job.result()
            row = {"standard": standard, "learning_rate_scale": rate}
            if error:
                row["error"] = error
            else:
                try:
                    row["metrics"] = metrics(baseline, predictions(path), hardware)
                except (ValueError, OSError) as exc:
                    row["error"] = str(exc)
            results.append(row)
            print(json.dumps(row), flush=True)
    results.sort(key=lambda row: (row["standard"], row["learning_rate_scale"]))
    (output / args.summary_name).write_text(json.dumps(results, indent=2) + "\n")
    if any("error" in row for row in results):
        raise SystemExit(f"some grid points failed; see scale-grid/{args.summary_name}")


if __name__ == "__main__":
    main()
