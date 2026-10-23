#!/usr/bin/env python3
"""Run deterministic scale-out and burst recovery experiments; build sync/common first."""

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
        "--profile", choices=["tune", "full", "acceptance"], default="full"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rescue-limit", type=int, default=32768)
    parser.add_argument(
        "--java", default=str(Path(os.environ.get("JAVA_HOME", "/usr")) / "bin/java")
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    properties = {}
    if args.profile == "acceptance":
        trials = []
        for hit in (0.0, 0.5, 0.9):
            loads = (0.2, 0.5, 0.8, 1.0, 1.2, 1.5) if hit == 0.9 else (0.2, 0.8)
            for load in loads:
                for scene, target in (
                    ("SCALE_OUT", 150),
                    ("SCALE_IN", 80),
                    ("SCALE_IN", 50),
                ):
                    name = f"{scene.lower()}-{target}-load-{load}-hit-{hit}"
                    trials.append(
                        (
                            name,
                            args.rescue_limit,
                            args.seeds,
                            60,
                            f"{load},{load}",
                            "CACHE_WINDOW,ELASTIC,ADAPTIVE,SESSION",
                            scene,
                        )
                    )
                    properties[name] = [
                        f"-Dflexlb.sim.hit={hit}",
                        "-Dflexlb.sim.resizeAt=.5",
                        (
                            f"-Dflexlb.sim.scaleIn={target}"
                            if scene == "SCALE_IN"
                            else f"-Dflexlb.sim.scaleOut={target}"
                        ),
                    ]
        for scene in ("RECOVERY_BURST", "RAMP", "SCALE_CYCLE"):
            name = scene.lower()
            trials.append(
                (
                    name,
                    args.rescue_limit,
                    args.seeds,
                    60,
                    ".8,.8" if scene == "RECOVERY_BURST" else ".2,.5,.8,1,1.2,1.5",
                    "CACHE_WINDOW,ELASTIC,ADAPTIVE,SESSION",
                    scene,
                )
            )
    elif args.profile == "tune":
        trials = [
            (
                f"limit-{limit}",
                limit,
                1,
                120,
                ".2,.5,.8,1.2,1.5,1.5",
                "CACHE_WINDOW,ELASTIC",
                "SCALE_OUT,HOTSPOT_SCALE_OUT",
            )
            for limit in (32768, 65536, 131072)
        ]
    else:
        trials = [
            (
                scene.lower(),
                args.rescue_limit,
                args.seeds,
                300,
                ".2,.5,.8,1.2,1.5,1.5",
                "CACHE_WINDOW,ELASTIC,ADAPTIVE",
                scene,
            )
            for scene in ("SCALE_OUT", "HOTSPOT_SCALE_OUT", "RAMP")
        ]
        trials.append(
            (
                "burst",
                args.rescue_limit,
                args.seeds,
                1800,
                ".8",
                "CACHE_WINDOW,ELASTIC,ADAPTIVE",
                "RECOVERY_BURST",
            )
        )

    def run(trial):
        name, limit, seeds, seconds, loads, policies, scenes = trial
        command = [
            args.java,
            "-Xmx3g",
            f"-Dflexlb.sim.loads={loads}",
            f"-Dflexlb.sim.policies={policies}",
            f"-Dflexlb.sim.rescueLimit={limit}",
            *properties.get(name, []),
            *(
                ["-Dflexlb.sim.rescueWaitMs=4000"]
                if args.profile == "acceptance"
                else []
            ),
            "-cp",
            "flexlb-sync/target/classes:flexlb-common/target/classes",
            "tools/PrefillElasticSimulation.java",
            str((args.out_dir / name).resolve()),
            str(seeds),
            str(seconds),
            "10000",
            "140000",
            "false",
            scenes,
        ]
        start = time.monotonic()
        with (args.out_dir / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=root, stdout=log, stderr=subprocess.STDOUT
            )
        data = dict(
            name=name,
            command=command,
            returncode=result.returncode,
            elapsed_seconds=time.monotonic() - start,
        )
        print(json.dumps(data), flush=True)
        return data

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, trials))
    (args.out_dir / "runs.json").write_text(json.dumps(results, indent=2))
    if any(r["returncode"] for r in results):
        raise SystemExit("elastic simulation failed; inspect logs")


if __name__ == "__main__":
    main()
