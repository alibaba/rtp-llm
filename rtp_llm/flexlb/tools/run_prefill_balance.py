#!/usr/bin/env python3
"""Reproduce the Prefill router experiment; compile Java modules first (see docs/BUILD_AND_TEST.md)."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=["full", "sweep", "steady", "controls", "burst", "matched"],
        default="full",
    )
    parser.add_argument(
        "--java", default=str(Path(os.environ.get("JAVA_HOME", "/usr")) / "bin/java")
    )
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    if args.profile == "sweep":
        for window in (5000, 10000, 30000, 60000, 120000):
            for cap in (70000, 140000, 280000):
                trials.append(
                    (
                        f"w{window}-c{cap}",
                        window,
                        cap,
                        False,
                        600,
                        ".8",
                        "CACHE_WINDOW",
                        "CONVERSATION",
                    )
                )
    elif args.profile == "matched":
        trials.extend(
            [
                (
                    "matched-steady",
                    10000,
                    140000,
                    False,
                    2400,
                    ".8",
                    "SHORT_TTFT_CAPPED",
                    "CONVERSATION",
                ),
                (
                    "matched-controls",
                    10000,
                    140000,
                    False,
                    180,
                    ".1,.5,.8,1,1.5",
                    "SHORT_TTFT_CAPPED",
                    "WARM_REPLAY,HOTSPOT",
                ),
            ]
        )
    elif args.profile == "burst":
        trials.append(
            (
                "burst",
                10000,
                140000,
                False,
                2400,
                ".8",
                "SHORT_TTFT,CACHE_ONLY,CACHE_WINDOW",
                "COLD_BURST,COLD_BURST_OPEN",
            )
        )
    elif args.profile == "controls":
        trials.append(
            (
                "controls",
                10000,
                140000,
                False,
                180,
                ".1,.5,.8,1,1.5",
                "SHORT_TTFT,CACHE_WINDOW",
                "WARM_REPLAY,HOTSPOT",
            )
        )
    elif args.profile == "steady":
        for reorder in (False, True):
            trials.append(
                (
                    f"reorder-{reorder}",
                    10000,
                    140000,
                    reorder,
                    2400,
                    ".8",
                    "SHORT_TTFT,CACHE_ONLY,CACHE_WINDOW",
                    "CONVERSATION",
                )
            )
    else:
        trials.append(
            (
                "full",
                10000,
                140000,
                False,
                300,
                ".1,.3,.5,.8,1,1.2,1.5",
                "SHORT_TTFT,CACHE_ONLY,CACHE_WINDOW",
                "REPLAY,CONVERSATION,COLD_BURST,TTL_GAP",
            )
        )

    def run(trial):
        name, window, cap, reorder, seconds, loads, policies, scenes = trial
        out = args.out_dir / name
        command = [
            args.java,
            "-Xmx3g",
            f"-Dflexlb.sim.loads={loads}",
            f"-Dflexlb.sim.policies={policies}",
            "-cp",
            "flexlb-sync/target/classes:flexlb-common/target/classes",
            "tools/PrefillBalanceSimulation.java",
            str(out.resolve()),
            str(5 if name == "matched-steady" else args.seeds),
            str(seconds),
            str(window),
            str(cap),
            str(reorder).lower(),
            scenes,
        ]
        start = time.monotonic()
        with (args.out_dir / f"{name}.log").open("w") as log:
            completed = subprocess.run(
                command, cwd=root, stdout=log, stderr=subprocess.STDOUT
            )
        result = dict(
            name=name,
            command=command,
            returncode=completed.returncode,
            elapsed_seconds=time.monotonic() - start,
        )
        print(json.dumps(result), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=3 if args.profile == "sweep" else 1
    ) as pool:
        results = list(pool.map(run, trials))
    (args.out_dir / "runs.json").write_text(json.dumps(results, indent=2))
    if any(result["returncode"] for result in results):
        raise SystemExit("simulation failed; inspect trial logs")


if __name__ == "__main__":
    main()
