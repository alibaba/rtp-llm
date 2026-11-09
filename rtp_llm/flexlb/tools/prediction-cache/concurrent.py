#!/usr/bin/env python3
"""Identical contended workload against two compiled revisions; three alternating JVM forks."""
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

root = pathlib.Path(__file__).resolve().parents[2]
baseline, out = map(pathlib.Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
cp = (
    ET.parse(
        root
        / "flexlb-sync/target/surefire-reports/TEST-org.flexlb.balance.prediction.IncrementalPredictionTest.xml"
    )
    .find(".//property[@name='java.class.path']")
    .attrib["value"]
)
java = pathlib.Path(os.environ["JAVA_HOME"]) / "bin"
formula = root / "flexlb-sync/src/test/resources/prediction/deepseek-v4.txt"
for variant, directory in [("before", baseline), ("after", root)]:
    dest = out / variant
    dest.mkdir(exist_ok=True)
    runtime = cp.replace(str(root), str(directory.resolve()))
    (dest / "classpath.txt").write_text(runtime)
    subprocess.run(
        [
            str(java / "javac"),
            "-proc:none",
            "-cp",
            runtime,
            "-d",
            str(dest),
            str(pathlib.Path(__file__).with_name("ConcurrentBench.java")),
        ],
        check=True,
    )
for fork in range(3):
    for variant in ["before", "after"] if fork % 2 == 0 else ["after", "before"]:
        dest = out / variant
        with (out / f"{variant}-{fork}.txt").open("w") as log:
            subprocess.run(
                [
                    str(java / "java"),
                    "-Xms1g",
                    "-Xmx1g",
                    "-cp",
                    str(dest) + os.pathsep + (dest / "classpath.txt").read_text(),
                    "ConcurrentBench",
                    str(formula),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"{variant}-{fork} complete", flush=True)
results = {}
for variant in ["before", "after"]:
    samples = {}
    for log in out.glob(f"{variant}-*.txt"):
        for line in log.read_text().splitlines():
            m = re.fullmatch(
                r"(\w+) t=(\d+) qps=([\d.]+) cpu_ns=([\d.]+) bytes=([\d.]+) p99_us=([\d.]+) blocked_us=([\d.]+) blocks=([\d.]+)",
                line,
            )
            if m:
                samples.setdefault(m[1] + "_" + m[2], []).append(
                    list(map(float, m.groups()[2:]))
                )
    assert len(samples) == 10 and all(len(v) == 6 for v in samples.values())
    results[variant] = {
        k: dict(
            zip(
                ["qps", "cpu_ns", "bytes", "p99_us", "blocked_us", "blocks"],
                map(statistics.median, zip(*v)),
            )
        )
        for k, v in samples.items()
    }
(out / "summary.json").write_text(json.dumps(results, indent=2))
for name, b in results["before"].items():
    a = results["after"][name]
    print(
        name,
        "throughput",
        round(a["qps"] / b["qps"], 2),
        "cpu_reduction",
        round(1 - a["cpu_ns"] / b["cpu_ns"], 3),
        "p99_ratio",
        round(a["p99_us"] / b["p99_us"], 2),
        "blocked_us",
        a["blocked_us"],
    )
