#!/usr/bin/env python3
"""FlexLB mock engine CASE test runner (场景测试).

Terminology (unified 2026-09, suite reorg task #85):

  * mock engine CASE test (场景测试) — THIS runner: flexlb_functional_tests.py
    plus the flexlb_ft/cases/ category modules.  Each case boots a small
    mock cluster via EnvManager and pins one behavioural contract per
    scenario.
  * mock engine STRESS test (压测) — the online_eval load pipeline
    (run_online_eval.sh + the eval/analysis scripts): QPS / ramp /
    duration load profiles and time-series analysis.  A separate
    lineage; do not mix the terms.

The legacy "e2e test" / "chaos test" suite wording is retired: fault
injection is a MECHANISM inside case tests (the engine_fault / status /
direct categories), not a suite name.

Nine scenario categories (flexlb_ft/cases/, one contract theme per
module — 75 cases total):

    cancel 13 | status 19 | kv 15 | balance 6 | elastic 8
    engine_fault 7 | master 3 | admission 3 | direct 1

Usage:
    python3 flexlb_functional_tests.py --category all --profile batch-window
    python3 flexlb_functional_tests.py --list
    python3 flexlb_functional_tests.py --filter cancel_t1 --profile single-nonbatch
    python3 flexlb_functional_tests.py --category kv --json results.json
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Allow running from online_eval/ directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flexlb_ft.cases import (
    ADMISSION_CASES,
    BALANCE_CASES,
    CANCEL_CASES,
    DIRECT_CASES,
    ELASTIC_CASES,
    ENGINE_FAULT_CASES,
    KV_CASES,
    MASTER_CASES,
    STATUS_CASES,
)
from flexlb_ft.context import CaseContext, CaseDef
from flexlb_ft.grade import GRADES, VERDICT_LABELS, GradeReport, overall_verdict
from flexlb_ft.harness import PROFILE_CAPS, PROFILES, EnvManager

# Task #85 (category reorg): the nine cases/ modules register into their
# own CATEGORY_CASES lists; the runner concatenates them in the canonical
# category order below.
ALL_CASES: list[CaseDef] = (
    CANCEL_CASES
    + STATUS_CASES
    + KV_CASES
    + BALANCE_CASES
    + ELASTIC_CASES
    + ENGINE_FAULT_CASES
    + MASTER_CASES
    + ADMISSION_CASES
    + DIRECT_CASES
)

# CLI spelling (kebab-case) -> CaseDef.category (python identifier).
CATEGORY_ALIASES = {"engine-fault": "engine_fault"}


def main():
    parser = argparse.ArgumentParser(
        description="FlexLB mock engine case test runner (场景测试)"
    )
    parser.add_argument(
        "--category",
        choices=[
            "all",
            "cancel",
            "status",
            "kv",
            "balance",
            "elastic",
            "engine-fault",
            "master",
            "admission",
            "direct",
        ],
        default="all",
        help="scenario category (one of the nine flexlb_ft/cases/ modules)",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES),
        default="batch-window",
        help="scheduling profile (scheduler.ordering.decision.dispatcher axes)",
    )
    parser.add_argument("--filter", default=None, help="substring filter on case name")
    parser.add_argument("--json", default=None, help="write JSON results to path")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument(
        "--grade",
        choices=list(GRADES),
        default="normal",
        help=(
            "assertion grade: strict uses tight bounds (达到=优异), normal "
            "standard bounds (达到=良好), loose floor bounds (最宽但仍能判不可用). "
            "Exceeding the running grade's bound fails the case; the achieved "
            "grade is recorded per case and rolled up into a run verdict."
        ),
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep env running after tests"
    )
    args = parser.parse_args()

    category = CATEGORY_ALIASES.get(args.category, args.category)

    # Filter cases
    cases = ALL_CASES
    if category != "all":
        cases = [c for c in cases if c.category == category]
    if args.filter:
        cases = [c for c in cases if args.filter.lower() in c.name.lower()]
    # Profile filter: explicit profile list, then semantic requirements
    # (CaseDef.requires must be covered by the profile's capability set).
    caps = PROFILE_CAPS[args.profile]
    cases = [
        c
        for c in cases
        if (c.profiles is None or args.profile in c.profiles)
        and (not c.requires or set(c.requires) <= caps)
    ]

    if args.list:
        print(
            f"{'NAME':<40} {'CATEGORY':<14} {'PROFILES':<20} {'REQUIRES':<24} {'SOURCE'}"
        )
        print("-" * 120)
        for c in cases:
            profiles = ",".join(c.profiles) if c.profiles else "all"
            requires = ",".join(c.requires) if c.requires else "-"
            print(
                f"{c.name:<40} {c.category:<14} {profiles:<20} {requires:<24} {c.source}"
            )
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
        args.profile,
        run_root,
        log_fn=lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}"),
        grade=args.grade,
    )

    results = []
    passed_count = 0
    failed_count = 0

    print(f"\n{'='*60}")
    print(
        f" FlexLB Case Tests — category={args.category} "
        f"profile={args.profile} grade={args.grade}"
    )
    print(f" {len(cases)} cases, run_root={run_root}")
    print(f"{'='*60}\n")

    graded_achieved: list[str] = []  # achieved grade per graded case (verdict roll-up)

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.name} ... ", end="", flush=True)
        t0 = time.monotonic()
        report: GradeReport | None = None
        try:
            # Fresh id range per case (dedup table in a reused master).
            CaseContext._case_seq += 1
            ctx.case_seq = CaseContext._case_seq
            outcome = case.fn(ctx)
            if len(outcome) == 3:
                ok, detail, report = outcome
            else:
                ok, detail = outcome
        except Exception as e:
            ok, detail = False, f"EXCEPTION: {e}\n{traceback.format_exc()}"
        duration_ms = int((time.monotonic() - t0) * 1000)
        status = "PASS" if ok else "FAIL"
        achieved = report.achieved if report is not None else None
        if report is not None:
            graded_achieved.append(report.achieved)
        results.append(
            {
                "category": case.category,
                "name": case.name,
                "profile": args.profile,
                "status": status,
                "duration_ms": duration_ms,
                "detail": detail if not ok else "",
                **({"grade": report.to_dict()} if report is not None else {}),
            }
        )
        grade_note = f" [{achieved}]" if achieved else ""
        if ok:
            passed_count += 1
            print(f"PASS{grade_note} ({duration_ms}ms)")
        else:
            failed_count += 1
            print(f"FAIL{grade_note} ({duration_ms}ms)")
            if detail:
                for line in str(detail).split("\n")[:5]:
                    print(f"    {line}")
            if report is not None and report.results:
                print(f"    grades: {report.summary()}")

    # Teardown
    if not args.keep:
        ctx.close()
        env_mgr.teardown()

    # Summary
    print(f"\n{'='*60}")
    print(f" Results: {passed_count} PASS / {failed_count} FAIL / {len(cases)} total")
    verdict = overall_verdict(graded_achieved)
    if verdict is not None:
        print(
            f" Overall grade: {verdict} ({VERDICT_LABELS[verdict]}) — "
            f"run grade={args.grade}, graded cases={len(graded_achieved)} "
            f"(all strict=优异 / all ≥normal=良好 / any beyond loose=不可用)"
        )
    print(f"{'='*60}\n")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"JSON written to {args.json}")

    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
