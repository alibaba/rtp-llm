#!/usr/bin/env python3
"""l1_mock_reference.py — mock-side theoretical-value calculator (Python wrapper).

Runs the thin Java main {@code org.flexlb.mockengine.L1MockReferenceMain}
(lives in flexlb-mock-engine test sources, same package as
MockPerformanceModel) so the L1 reference values come from the EXACT
production mock timing formulas — prefillMs() / decodeMs() /
decodeStepDelayMs() — with zero formula duplication. No network, no
running engine: this is a pure local computation.

The wrapper resolves the flexlb Maven root, ensures the test classes are
compiled, and invokes the main via the exec-maven-plugin with test-scope
classpath:

  ./mvnw -q -pl flexlb-mock-engine -am test-compile
  ./mvnw -q -pl flexlb-mock-engine exec:java \
      -Dexec.mainClass=org.flexlb.mockengine.L1MockReferenceMain \
      -Dexec.classpathScope=test \
      -Dexec.args="--plan <plan> [--performance <perf>] [--master <cfg>] --out <ref>"

Pass the SAME performance/master JSON the remote mock deployment uses when
comparing against a real run; omit both to get the built-in defaults
(sleep_scale=1, production DSv4 prefill fit + decode fit 19.5+0.175x,
2.6 tokens/step).

Usage:
  python3 l1_mock_reference.py --plan /tmp/l1/l1_grid_plan.json \\
      --out /tmp/l1/l1_mock_reference.json \\
      [--performance data/config/performance_dsv4.json] \\
      [--master data/config/master_fixed_window.json] \\
      [--flexlb-root ../../..] [--skip-compile]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_FLEXLB_ROOT = Path(__file__).resolve().parents[2]  # rtp_llm/flexlb (mvnw root)
MAIN_CLASS = "org.flexlb.mockengine.L1MockReferenceMain"


def _find_mvnw(flexlb_root: Path) -> list[str]:
    """Prefer the wrapper, fall back to a system mvn."""
    wrapper = flexlb_root / "mvnw"
    if wrapper.exists():
        return [str(wrapper)]
    if shutil.which("mvn"):
        return ["mvn"]
    print(
        "ERROR: no mvnw in %s and no system mvn on PATH" % flexlb_root, file=sys.stderr
    )
    sys.exit(2)


def _run(cmd: list[str], flexlb_root: Path) -> None:
    print("+ " + " ".join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, cwd=flexlb_root)
    if result.returncode != 0:
        print(
            f"ERROR: command failed with exit code {result.returncode}: "
            f"{' '.join(cmd)}",
            file=sys.stderr,
        )
        sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="l1_mock_reference.py",
        description="Compute L1 mock theoretical values via the "
        "MockPerformanceModel Java formulas (local, no engine).",
    )
    parser.add_argument(
        "--plan", required=True, help="l1_grid_plan.json from l1_grid_runner.py plan"
    )
    parser.add_argument(
        "--out", required=True, help="output l1_mock_reference.json path"
    )
    parser.add_argument(
        "--performance",
        default=None,
        help="performance JSON the mock under test runs "
        "with (omit for built-in defaults)",
    )
    parser.add_argument(
        "--master",
        default=None,
        help="master config JSON carrying the FLEXLB_CONFIG "
        "prefill FORMULA estimator (omit for the "
        "built-in DSv4 production fit)",
    )
    parser.add_argument(
        "--flexlb-root",
        default=str(DEFAULT_FLEXLB_ROOT),
        help="flexlb Maven root (default: auto-detected " "relative to this script)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="skip test-compile (fast re-runs after the " "first successful compile)",
    )
    args = parser.parse_args(argv)

    flexlb_root = Path(args.flexlb_root).resolve()
    if not (flexlb_root / "flexlb-mock-engine" / "pom.xml").exists():
        print(
            f"ERROR: {flexlb_root} does not look like the flexlb root "
            f"(flexlb-mock-engine/pom.xml missing)",
            file=sys.stderr,
        )
        return 2
    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"ERROR: plan file not found: {plan_path}", file=sys.stderr)
        return 2
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mvn = _find_mvnw(flexlb_root)

    if not args.skip_compile:
        _run(
            mvn + ["-q", "-pl", "flexlb-mock-engine", "-am", "test-compile"],
            flexlb_root,
        )

    java_args = ["--plan", str(plan_path), "--out", str(out_path)]
    if args.performance:
        perf = Path(args.performance)
        if not perf.exists():
            print(f"ERROR: performance file not found: {perf}", file=sys.stderr)
            return 2
        java_args += ["--performance", str(perf)]
    if args.master:
        master = Path(args.master)
        if not master.exists():
            print(f"ERROR: master config file not found: {master}", file=sys.stderr)
            return 2
        java_args += ["--master", str(master)]

    _run(
        mvn
        + [
            "-q",
            "-pl",
            "flexlb-mock-engine",
            "exec:java",
            f"-Dexec.mainClass={MAIN_CLASS}",
            "-Dexec.classpathScope=test",
            f"-Dexec.args={' '.join(java_args)}",
        ],
        flexlb_root,
    )

    if not out_path.exists():
        print(f"ERROR: expected output was not produced: {out_path}", file=sys.stderr)
        return 1
    print(f"mock reference ready: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
