#!/usr/bin/env python3
"""FlexLB unified functional + chaos test runner.

Usage:
    python3 flexlb_functional_tests.py --suite smoke --mode batch
    python3 flexlb_functional_tests.py --list
    python3 flexlb_functional_tests.py --filter T1 --mode batch
    python3 flexlb_functional_tests.py --suite all --json results.json
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Allow running from online_eval/ directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flexlb_ft.chaos_cases import CHAOS_CASES
from flexlb_ft.context import SMOKE_LABEL_PERF, CaseContext, CaseDef
from flexlb_ft.harness import EnvManager
from flexlb_ft.smoke_cases import SMOKE_CASES

ALL_CASES: list[CaseDef] = SMOKE_CASES + CHAOS_CASES


def main():
    parser = argparse.ArgumentParser(description="FlexLB functional test runner")
    parser.add_argument("--suite", choices=["smoke", "chaos", "all"], default="all")
    parser.add_argument("--mode", choices=["batch", "direct", "queue"], default="batch")
    parser.add_argument("--filter", default=None, help="substring filter on case name")
    parser.add_argument("--json", default=None, help="write JSON results to path")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument(
        "--keep", action="store_true", help="keep env running after tests"
    )
    args = parser.parse_args()

    # Filter cases
    cases = ALL_CASES
    if args.suite != "all":
        cases = [c for c in cases if c.suite == args.suite]
    if args.filter:
        cases = [c for c in cases if args.filter.lower() in c.name.lower()]
    # Mode filter
    cases = [c for c in cases if c.modes is None or args.mode in c.modes]

    if args.list:
        print(f"{'NAME':<40} {'SUITE':<8} {'MODES':<20} {'SOURCE'}")
        print("-" * 90)
        for c in cases:
            modes = ",".join(c.modes) if c.modes else "all"
            print(f"{c.name:<40} {c.suite:<8} {modes:<20} {c.source}")
        print(f"\nTotal: {len(cases)} cases")
        return 0

    if not cases:
        print("No cases match filters.", file=sys.stderr)
        return 1

    # Setup
    run_root = Path(f"/tmp/flexlb_ft_{int(time.time())}")
    run_root.mkdir(parents=True, exist_ok=True)
    env_mgr = EnvManager(run_root)
    ctx = CaseContext(
        env_mgr,
        args.mode,
        run_root,
        log_fn=lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}"),
    )

    results = []
    passed_count = 0
    failed_count = 0

    print(f"\n{'='*60}")
    print(f" FlexLB Functional Tests — suite={args.suite} mode={args.mode}")
    print(f" {len(cases)} cases, run_root={run_root}")
    print(f"{'='*60}\n")

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.name} ... ", end="", flush=True)
        t0 = time.monotonic()
        try:
            # Fresh id range per case (dedup table in a reused master).
            CaseContext._case_seq += 1
            ctx.case_seq = CaseContext._case_seq
            ok, detail = case.fn(ctx)
        except Exception as e:
            ok, detail = False, f"EXCEPTION: {e}\n{traceback.format_exc()}"
        duration_ms = int((time.monotonic() - t0) * 1000)
        status = "PASS" if ok else "FAIL"
        results.append(
            {
                "suite": case.suite,
                "name": case.name,
                "mode": args.mode,
                "status": status,
                "duration_ms": duration_ms,
                "detail": detail if not ok else "",
            }
        )
        if ok:
            passed_count += 1
            print(f"PASS ({duration_ms}ms)")
        else:
            failed_count += 1
            print(f"FAIL ({duration_ms}ms)")
            if detail:
                for line in str(detail).split("\n")[:5]:
                    print(f"    {line}")

    # Teardown
    if not args.keep:
        ctx.close()
        env_mgr.teardown()

    # Summary
    print(f"\n{'='*60}")
    print(f" Results: {passed_count} PASS / {failed_count} FAIL / {len(cases)} total")
    print(f"{'='*60}\n")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"JSON written to {args.json}")

    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
