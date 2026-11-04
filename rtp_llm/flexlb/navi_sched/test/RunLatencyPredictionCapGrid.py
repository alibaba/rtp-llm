#!/usr/bin/env python3
"""Ablate exact per-parameter prediction-step caps without model scales."""

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

from RunLatencyScaleGrid import metrics, predictions


CAPS = (None, 0.0025, 0.005, 0.01, 0.02, 0.05)
RATES = (8, 16, 32, 64)
MAX_WORKERS = 6
MAX_GRID_POINTS = 32
EXPECTED_UPDATES = 54740


def audit_cap(path, cap):
    updates = 0
    violations = 0
    saturated = 0
    maximum_ratio = 0.0
    with path.open() as stream:
        for line in stream:
            fields = line.split("\t")
            if fields[0] != "C":
                continue
            updates += 1
            target = float(fields[4])
            for value in fields[6:]:
                shift = abs(float(value))
                if shift > cap * target + 1e-6:
                    violations += 1
                if target > 0:
                    ratio = shift / target
                    maximum_ratio = max(maximum_ratio, ratio)
                    saturated += abs(ratio - cap) < 1e-7
    if updates != EXPECTED_UPDATES or violations:
        raise ValueError(f"cap audit failed: {updates} updates, "
                         f"{violations} violations in {path}")
    return {"updates": updates, "violations": violations,
            "saturated_parameter_steps": saturated,
            "maximum_shift_to_target_ratio": maximum_ratio}


def run_point(root, history, output, cap, rate, library_path):
    label = "none" if cap is None else f"{cap * 100:g}pct"
    path = output / f"cap-{label}-lr-{rate}.tsv"
    temporary = output / f"cap-{label}-lr-{rate}.partial.tsv"
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    env = dict(os.environ, LD_LIBRARY_PATH=library_path)
    if cap is None:
        env.pop("NAVI_REPLAY_PARAMETER_PREDICTION_CAP_RATIO", None)
    else:
        env["NAVI_REPLAY_PARAMETER_PREDICTION_CAP_RATIO"] = str(cap)
    command = [str(root / "no-scale-cap-replay"), str(history), str(temporary),
               "4", "20261010", str(rate), "0.8", "1,1,1,1,1,1,0,0", "1"]
    try:
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:] or f"exit {result.returncode}")
        rows = predictions(temporary)
        cap_audit = audit_cap(temporary, cap) if cap is not None else None
        temporary.replace(path)
        return label, rate, rows, cap_audit, None
    except (OSError, RuntimeError, ValueError) as exc:
        if temporary.exists():
            temporary.unlink()
        return label, rate, None, None, str(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("history", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS or len(CAPS) * len(RATES) > MAX_GRID_POINTS:
        parser.error("invalid grid size or worker count")
    output = args.root / "no-scale-cap-grid"
    output.mkdir(exist_ok=True)
    hardware = {node["ip"].replace(".", "_"):
                "MI308" if "MI308" in node["deployment"] else "RTX5000"
                for node in json.loads(args.manifest.read_text())["nodes"]}
    baseline = predictions(args.root / "original-no-decode.tsv")
    library_path = str(Path(__file__).parents[1] /
                       "release/navi_sched_cpp-0.1.0/lib")
    summary = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_point, args.root, args.history, output, cap,
                            rate, library_path) for cap in CAPS for rate in RATES]
        for job in concurrent.futures.as_completed(jobs):
            label, rate, rows, cap_audit, error = job.result()
            result = {"cap": label, "learning_rate_scale": rate}
            if error:
                result["error"] = error
            else:
                result["metrics"] = metrics(baseline, rows, hardware)
                if cap_audit:
                    result["cap_audit"] = cap_audit
            summary.append(result)
            print(json.dumps(result), flush=True)
    summary.sort(key=lambda row: (row["cap"], row["learning_rate_scale"]))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if any("error" in row for row in summary):
        raise SystemExit("some cap grid points failed")


if __name__ == "__main__":
    main()
