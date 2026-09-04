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
injection is a MECHANISM inside case tests (the engine_fault / status
categories and the master category's direct-path case), not a suite
name.

Nine scenario categories (flexlb_ft/cases/, one contract theme per
module; cases self-register per module, so category totals move with
in-flight category work — verify the current count with
`python3 flexlb_functional_tests.py --list`):

    cancel | status | kv | balance | elastic
    engine_fault | master | admission | priority

(cancel_stream_break_decode_autonomous requires generate_stream, so the
batch-window --list shows one fewer cancel row — profile filtering, not
a missing case.)

Outcome classification (task #101 expected-fail mechanism): every case is
normal or a declared-finding probe (``@case(..., expected_fail=True)``).

    PASS               normal case passed (contract-pass)
    FAIL               normal case failed → gates the exit code (1)
    FINDING-CONFIRMED  expected_fail probe failed as predicted — the
                       declared finding stands (exit 0)
    FINDING-RESOLVED   expected_fail probe unexpectedly passed — the
                       finding was FIXED; counted and reported for mark
                       review (exit 0)

Verdict roll-up and exit code consume ONLY normal cases: a declared
finding never renders the suite verdict unusable, so CI can gate on the
contract cases while findings are tracked as first-class outcomes.

Usage:
    python3 flexlb_functional_tests.py --category all --profile batch-window
    python3 flexlb_functional_tests.py --list
    python3 flexlb_functional_tests.py --filter cancel_basic --profile single-nonbatch
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
    ELASTIC_CASES,
    ENGINE_FAULT_CASES,
    KV_CASES,
    MASTER_CASES,
    PRIORITY_CASES,
    STATUS_CASES,
)
from flexlb_ft.context import CaseContext, CaseDef
from flexlb_ft.grade import GRADES, VERDICT_LABELS, GradeReport, overall_verdict
from flexlb_ft.harness import PROFILE_CAPS, PROFILES, EnvManager

# Task #85 (category reorg): the nine cases/ modules register into their
# own CATEGORY_CASES lists; the runner concatenates them in the canonical
# category order below (priority after admission — the 2026-09 intake3-
# rebuild migration, PRIORITY-axis case-layer JSON injection).
ALL_CASES: list[CaseDef] = (
    CANCEL_CASES
    + STATUS_CASES
    + KV_CASES
    + BALANCE_CASES
    + ELASTIC_CASES
    + ENGINE_FAULT_CASES
    + MASTER_CASES
    + ADMISSION_CASES
    + PRIORITY_CASES
)

# CLI spelling (kebab-case) -> CaseDef.category (python identifier).
CATEGORY_ALIASES = {"engine-fault": "engine_fault"}

# ── Three-way outcome classification (task #101 expected-fail) ─────────────

STATUS_PASS = "PASS"  # normal case passed (contract-pass)
STATUS_FAIL = "FAIL"  # normal case failed → exit 1
# Declared-finding probe failed as predicted — the finding stands.
STATUS_FINDING_CONFIRMED = "FINDING-CONFIRMED"
# Declared-finding probe unexpectedly passed — the finding was fixed;
# reported for mark review, never silently absorbed.
STATUS_FINDING_RESOLVED = "FINDING-RESOLVED"


def classify_outcome(expected_fail: bool, ok: bool) -> str:
    """Three-way classification of one case outcome (task #101).

    Normal cases: PASS / FAIL.  Declared-finding probes (expected_fail —
    contract written per the CORRECT behaviour, current master known not
    to satisfy it): FINDING-CONFIRMED when they fail as predicted,
    FINDING-RESOLVED when they unexpectedly pass.  Neither finding class
    enters failed_count / the verdict roll-up / the exit code.
    """
    if not expected_fail:
        return STATUS_PASS if ok else STATUS_FAIL
    return STATUS_FINDING_RESOLVED if ok else STATUS_FINDING_CONFIRMED


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
            "priority",
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
    parser.add_argument(
        "--cases",
        default=None,
        help=(
            "comma-separated exact case names to run — takes priority over "
            "the --category/--filter selection (profile filtering still "
            "applies); unknown names exit 2"
        ),
    )
    parser.add_argument("--json", default=None, help="write JSON results to path")
    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "override the run root directory (the parallel orchestrator "
            "isolates sibling lanes this way; default keeps the "
            "/tmp/flexlb_ft_<epoch> layout)"
        ),
    )
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

    # Filter cases.  --cases (exact-name list — the parallel orchestrator
    # passes each case-shard lane's slice this way) wins over the
    # --category/--filter selection; profile filtering below still applies.
    cases = ALL_CASES
    if args.cases:
        wanted = [n.strip() for n in args.cases.split(",") if n.strip()]
        by_name = {c.name: c for c in ALL_CASES}
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            print(
                f"error: unknown --cases entries: {unknown} "
                "(run --list for valid names)",
                file=sys.stderr,
            )
            return 2
        # dict.fromkeys: order-preserving dedup of the requested names.
        cases = [by_name[n] for n in dict.fromkeys(wanted)]
    else:
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
            f"{'NAME':<40} {'CATEGORY':<14} {'PROFILES':<20} {'REQUIRES':<24} "
            f"{'FINDING':<17} {'SOURCE'}"
        )
        print("-" * 140)
        for c in cases:
            profiles = ",".join(c.profiles) if c.profiles else "all"
            requires = ",".join(c.requires) if c.requires else "-"
            finding = "expected-fail" if c.expected_fail else "-"
            print(
                f"{c.name:<40} {c.category:<14} {profiles:<20} {requires:<24} "
                f"{finding:<17} {c.source}"
            )
        n_findings = sum(1 for c in cases if c.expected_fail)
        print(f"\nTotal: {len(cases)} cases ({n_findings} expected-fail probes)")
        return 0

    if not cases:
        print("No cases match filters.", file=sys.stderr)
        return 1

    # Setup.  --run-root (parallel_runner.py passes a per-lane dir) keeps
    # sibling lanes from sharing the same second-derived directory — two
    # lanes started in the same wall-clock second would otherwise merge
    # their env<N>_<label> dirs and interleave mock/master logs.
    run_root = (
        Path(args.run_root)
        if args.run_root
        else Path(f"/tmp/flexlb_ft_{int(time.time())}")
    )
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
    passed_count = 0  # contract-pass (normal cases only)
    failed_count = 0  # normal-case failures — the ONLY exit-code gate
    finding_confirmed = 0
    finding_resolved = 0

    print(f"\n{'='*60}")
    print(
        f" FlexLB Case Tests — category={args.category} "
        f"profile={args.profile} grade={args.grade}"
    )
    print(f" {len(cases)} cases, run_root={run_root}")
    print(f"{'='*60}\n")

    graded_achieved: list[str] = []  # achieved grade per NORMAL graded case
    # (verdict roll-up — task #101: an expected_fail graded case's achieved
    # is finding evidence, not suite quality; see grade.overall_verdict)

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
        status = classify_outcome(case.expected_fail, ok)
        achieved = report.achieved if report is not None else None
        # Verdict roll-up takes ONLY normal graded cases (task #101): an
        # expected_fail graded case's achieved (e.g. kv_storm_hot_churn's
        # band failure) is finding evidence, not suite quality.
        if report is not None and not case.expected_fail:
            graded_achieved.append(report.achieved)
        results.append(
            {
                "category": case.category,
                "name": case.name,
                "profile": args.profile,
                "status": status,
                "expected_fail": case.expected_fail,
                "duration_ms": duration_ms,
                "detail": detail if not ok else "",
                **({"grade": report.to_dict()} if report is not None else {}),
            }
        )
        grade_note = f" [{achieved}]" if achieved else ""
        if status == STATUS_PASS:
            passed_count += 1
            print(f"PASS{grade_note} ({duration_ms}ms)")
        elif status == STATUS_FAIL:
            failed_count += 1
            print(f"FAIL{grade_note} ({duration_ms}ms)")
            if detail:
                for line in str(detail).split("\n")[:5]:
                    print(f"    {line}")
            if report is not None and report.results:
                print(f"    grades: {report.summary()}")
        elif status == STATUS_FINDING_CONFIRMED:
            finding_confirmed += 1
            print(f"FINDING-CONFIRMED{grade_note} ({duration_ms}ms)")
            # The failing detail IS the finding evidence — surface it.
            if detail:
                for line in str(detail).split("\n")[:5]:
                    print(f"    {line}")
            if report is not None and report.results:
                print(f"    grades: {report.summary()}")
        else:  # STATUS_FINDING_RESOLVED
            finding_resolved += 1
            print(f"FINDING-RESOLVED{grade_note} ({duration_ms}ms)")
            print(
                "    probe unexpectedly PASSED — the declared finding looks "
                "fixed; review the expected_fail mark"
            )

    # Teardown
    if not args.keep:
        ctx.close()
        env_mgr.teardown()

    # Summary
    print(f"\n{'='*60}")
    print(
        f" Results: {passed_count} PASS / {failed_count} FAIL / "
        f"{finding_confirmed} finding-confirmed / "
        f"{finding_resolved} finding-resolved / {len(cases)} total"
    )
    if finding_confirmed or finding_resolved:
        print(
            f" Findings: {finding_confirmed} confirmed (expected-fail probes "
            f"failed as declared), {finding_resolved} resolved (probes "
            f"unexpectedly PASSED — review the expected_fail marks)"
        )
    verdict = overall_verdict(graded_achieved)
    if verdict is not None:
        print(
            f" Overall grade: {verdict} ({VERDICT_LABELS[verdict]}) — "
            f"run grade={args.grade}, graded cases={len(graded_achieved)} "
            f"(all strict=优异 / all ≥normal=良好 / any beyond loose=不可用)"
        )
    print(f"{'='*60}\n")

    if args.json:
        # JSON payload (task #101): summary block first (CI reads the counts
        # and the exit code without scanning the case rows), then the
        # per-case rows — each row carries expected_fail plus the four-way
        # status (PASS / FAIL / FINDING-CONFIRMED / FINDING-RESOLVED).
        exit_code = 1 if failed_count > 0 else 0
        payload = {
            "summary": {
                "total": len(cases),
                "passed": passed_count,
                "failed": failed_count,
                "finding_confirmed": finding_confirmed,
                "finding_resolved": finding_resolved,
                "verdict": verdict,
                "exit_code": exit_code,
            },
            "cases": results,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"JSON written to {args.json}")

    # Exit code (task #101): only NORMAL-case failures gate CI.  A pure
    # finding-confirmed run exits 0 — findings are the suite's product,
    # not an unstable verdict; finding-resolved runs also exit 0 (flagged
    # in the summary / JSON for mark review instead).
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
