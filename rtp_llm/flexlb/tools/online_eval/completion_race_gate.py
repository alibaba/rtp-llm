#!/usr/bin/env python3
"""Repeat length_mixed without diagnostic polling; verdicts come from artifacts."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INSTANCE = "balance_distribution::length_mixed::batch-window"
CLEAN_STAGES = {f"wave{i}_master_clean" for i in range(1, 6)}


def _number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("missing or invalid scheduler/time evidence")
    return value


def classify_result(path: Path) -> dict:
    """PASS, ZOMBIE or INVALID. A missing/failed run must never become green."""
    result = json.loads(path.read_text())
    if (
        result.get("id") != INSTANCE
        or not result.get("cleanup")
        or any(c.get("status") != "PASS" for c in result["cleanup"])
    ):
        raise ValueError("wrong instance or incomplete cleanup")
    stages = result["stages"]
    clean = [s for s in stages if s["id"] in CLEAN_STAGES]
    if {s["id"] for s in clean} != CLEAN_STAGES or len(clean) != 5:
        raise ValueError("missing/duplicate clean stages")
    zombies = []
    evidence = []
    for stage in clean:
        if stage["status"] == "BLOCKED":
            continue
        artifacts = [
            Path(p)
            for p in stage.get("artifacts", [])
            if Path(p).name.startswith("balance-master-clean-")
        ]
        if not artifacts and stage["status"] == "TIMEOUT":
            # A timed-out action raises before returning StageOutput. Its finally
            # block still saves samples, but result.json cannot list that artifact.
            # Match the unique local sample window; never borrow another instance.
            start, end = _number(stage["started_s"]), _number(stage["finished_s"])
            for candidate in path.parent.glob("balance-master-clean-*.json"):
                samples = json.loads(candidate.read_text())
                if (
                    samples
                    and start
                    <= _number(samples[0]["time_s"])
                    <= _number(samples[-1]["time_s"])
                    <= end
                ):
                    artifacts.append(candidate)
        if len(artifacts) != 1:
            raise ValueError("missing or ambiguous clean samples")
        artifact = artifacts[0]
        if not artifact.is_file():
            matches = list(path.parent.rglob(artifact.name))
            if len(matches) != 1:
                raise ValueError("clean artifact unavailable")
            artifact = matches[0]
        samples = json.loads(artifact.read_text())
        if not samples:
            raise ValueError("empty clean samples")
        values = [_number(s["raw"].get("scheduler_inflight")) for s in samples]
        times = [_number(s["time_s"]) for s in samples]
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("non-monotonic clean samples")
        evidence.append(str(artifact))
        if stage["status"] == "PASS":
            if values[-1] != 0:
                raise ValueError("PASS contradicts final scheduler count")
        elif stage["status"] == "TIMEOUT":
            # The original stage deadline defines the 30 s business window.
            # Samples arrive at 0.5 s intervals; they need not land exactly at 30 s.
            start, end = _number(stage["started_s"]), _number(stage["finished_s"])
            covered = (
                end - start >= 30
                and len(times) >= 2
                and 0 <= times[0] - start <= 1.5
                and 0 <= end - times[-1] <= 1.5
                and all(0 <= b - a <= 1.5 for a, b in zip(times, times[1:]))
            )
            if not covered or not all(v > 0 for v in values):
                raise ValueError("timeout is not a fully observed scheduler zombie")
            zombies.append(stage["id"])
        else:
            raise ValueError("unexpected clean-stage failure")
    other_failures = [
        s["id"]
        for s in stages
        if s["status"] not in ("PASS", "BLOCKED") and s["id"] not in zombies
    ]
    if other_failures:
        raise ValueError(f"other failures: {other_failures}")
    if zombies:
        if result["status"] != "TIMEOUT":
            raise ValueError("zombie stage/result mismatch")
        return dict(verdict="ZOMBIE", zombie_stages=zombies, artifacts=evidence)
    if result["status"] != "PASS" or any(s["status"] != "PASS" for s in stages):
        raise ValueError("incomplete execution")
    return dict(verdict="PASS", zombie_stages=[], artifacts=evidence)


def analyze_round(folder: Path, rc: int) -> dict:
    try:
        if type(rc) is not int:
            raise ValueError("missing runner exit code")
        results = list(folder.rglob("result.json"))
        if len(results) != 1:
            raise ValueError("expected exactly one raw result")
        answer = classify_result(results[0])
        if (answer["verdict"] == "PASS") != (rc == 0):
            raise ValueError("runner exit/result mismatch")
        return dict(answer, result=str(results[0]), runner_exit_code=rc)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(verdict="INVALID", error=str(error), runner_exit_code=rc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analyze-dir",
        type=Path,
        help="Rejudge an archived gate run without executing or polling services",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--port-base", type=int, default=os.environ.get("FLEXLB_PORT_BASE")
    )
    parser.add_argument(
        "--port-block",
        type=int,
        help="Size of this run's leased port block",
    )
    args = parser.parse_args()
    if args.analyze_dir is not None:
        old = json.loads((args.analyze_dir / "summary.json").read_text())
        rows = old["results"]
        if len(rows) < 20 or sorted(r["round"] for r in rows) != list(
            range(1, len(rows) + 1)
        ):
            parser.error("archive must contain at least 20 unique consecutive rounds")
        args.out_dir = args.out_dir.resolve()
        args.out_dir.mkdir(parents=True, exist_ok=False)
        results = []
        for row in rows:
            folder = args.analyze_dir.resolve() / f"round-{row['round']:03d}"
            result = dict(
                round=row["round"],
                lane=row["lane"],
                case_id=INSTANCE,
                directory=str(folder),
                **analyze_round(folder / "cases", row.get("runner_exit_code")),
            )
            results.append(result)
            print(json.dumps(result), flush=True)
        return write_summary(args.out_dir, results, old["lanes"])
    if (
        args.rounds < 20
        or not 1 <= args.lanes <= args.rounds
        or args.port_base is None
        or args.port_block is None
        or args.port_block < 1
    ):
        parser.error("require rounds >= 20, valid lanes and leased ports")
    if args.lanes * 6 > 100 or 100 + args.lanes * 200 > args.port_block:
        parser.error(
            "leased block cannot contain disjoint master and mock lane windows"
        )
    args.out_dir = args.out_dir.resolve()
    args.source_root = args.source_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows, lock = [], threading.Lock()

    def lane_work(lane):
        for index in range(lane, args.rounds, args.lanes):
            folder = args.out_dir / f"round-{index + 1:03d}"
            folder.mkdir()
            env = dict(
                os.environ, PYTHONUNBUFFERED="1", MOCK_STATUS_SNAPSHOT_LOG="false"
            )
            env.update(
                FLEXLB_FT_PARALLEL_MASTER_BASE=str(args.port_base + lane * 6),
                FLEXLB_FT_PARALLEL_MOCK_BASE=str(args.port_base + 100 + lane * 200),
                FLEXLB_EVAL_PROTO_OUT=str(folder / "proto"),
                FLEXLB_FT_INSTANCE_TIMING_BASELINE=str(folder / "timing.json"),
            )
            cmd = [
                sys.executable,
                str(args.source_root / "parallel_runner.py"),
                "--profile",
                "batch-window",
                "--parallel",
                "1",
                "--mock-stride",
                "200",
                "--instances",
                INSTANCE,
                "--out-dir",
                str(folder / "cases"),
            ]
            try:
                dry = subprocess.run(
                    cmd + ["--dry-run"],
                    cwd=args.source_root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                (folder / "plan-output.txt").write_text(dry.stdout + dry.stderr)
                if dry.returncode:
                    raise ValueError("runner plan failed")
                plan = json.loads(dry.stdout[dry.stdout.index("{") :])
                ports = [
                    p
                    for l in plan["lanes"]
                    for iv in l["intervals"]
                    for p in range(iv["first"], iv["last"] + 1)
                ]
                if (
                    not ports
                    or len(set(ports)) != len(ports)
                    or min(ports) < args.port_base
                    or max(ports) >= args.port_base + args.port_block
                ):
                    raise ValueError("runner plan exceeds leased ports")
                with (folder / "runner.log").open("w") as log:
                    run = subprocess.run(
                        cmd,
                        cwd=args.source_root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                answer = analyze_round(folder / "cases", run.returncode)
            except (OSError, ValueError, KeyError, TypeError) as error:
                answer = dict(verdict="INVALID", error=str(error))
            row = dict(
                round=index + 1,
                lane=lane,
                case_id=INSTANCE,
                directory=str(folder),
                **answer,
            )
            with lock:
                rows.append(row)
                rows.sort(key=lambda r: r["round"])
                (args.out_dir / "progress.json").write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)

    # Each worker owns one port window and runs its assigned rounds serially.
    with ThreadPoolExecutor(max_workers=args.lanes) as pool:
        list(pool.map(lane_work, range(args.lanes)))
    return write_summary(args.out_dir, rows, args.lanes)


def write_summary(out_dir, rows, lanes):
    summary = dict(
        case_id=INSTANCE,
        rounds=len(rows),
        lanes=lanes,
        zombie_count=sum(r["verdict"] == "ZOMBIE" for r in rows),
        invalid_count=sum(r["verdict"] == "INVALID" for r in rows),
        passed=sum(r["verdict"] == "PASS" for r in rows),
        results=rows,
    )
    summary["exit_code"] = (
        2 if summary["invalid_count"] else 1 if summary["zombie_count"] else 0
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
