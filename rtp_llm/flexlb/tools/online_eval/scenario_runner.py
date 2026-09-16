#!/usr/bin/env python3
"""Configure Python cases from YAML and execute them inside a parent-owned port lane."""

import argparse
import hashlib
import json
import signal
import sys
import threading
from pathlib import Path

from flexlb_test_framework.scenario import (
    ScenarioError,
    compile_scenarios,
    load_scenarios,
)
from flexlb_test_framework.suites import classify
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.compiler import plan_counts
from flexlb_test_framework.scenario.lease import validate_lease

LIST_FIELDS = (
    "id",
    "scenario_id",
    "variant_id",
    "profile",
    "grade",
    "effective_axes",
    "effective_capabilities",
    "category",
    "tags",
    "requires",
    "source_path",
    "source",
    "estimated_duration_s",
    "resource_budget",
)


def instance_directory(instance_id):
    # JVM -Xlog uses ':' as a delimiter even when the argv is shell-quoted.
    # The public ID remains unchanged; only its filesystem storage key differs.
    return "instance-" + hashlib.sha256(instance_id.encode("utf-8")).hexdigest()


def inventory(plans):
    rows = []
    for plan in plans:
        row = {key: plan[key] for key in LIST_FIELDS}
        row["test_kind"] = plan.get("test_kind", "functional")
        if "implementation" in plan:
            row["implementation"] = plan["implementation"]
        row["execution"] = {
            key: plan["execution"][key] for key in ("timeout_s", "cleanup_timeout_s")
        }
        rows.append(row)
    return dict(schema_version=1, counts=plan_counts(plans), instances=rows)


def select(plans, exact_ids):
    if exact_ids is None:
        if not plans:
            raise ScenarioError("no matching instances")
        return plans
    ids = exact_ids.split(",")
    known = {plan["id"]: plan for plan in plans}
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ScenarioError("instance selection is empty or contains duplicates")
    missing = set(ids) - set(known)
    if missing:
        raise ScenarioError(f"unknown exact instance IDs: {sorted(missing)}")
    return [known[value] for value in ids]


def summarize(rows):
    failed = sum(
        row["status"] in ("FAIL", "ERROR", "TIMEOUT")
        or any(c["status"] != "PASS" for c in row["cleanup"])
        for row in rows
    )
    return dict(
        passed_count=sum(r["status"] == "PASS" for r in rows),
        failed_count=failed,
        finding_confirmed=sum(r["status"] == "FINDING-CONFIRMED" for r in rows),
        finding_resolved=sum(r["status"] == "FINDING-RESOLVED" for r in rows),
        exit_code=1 if failed else 0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="scenario file or directory")
    parser.add_argument("--profile")
    parser.add_argument(
        "--suite", choices=("functional", "workload", "all"), default="all"
    )
    parser.add_argument(
        "--grade", choices=("strict", "normal", "loose"), default="normal"
    )
    parser.add_argument("--instances", help="comma separated exact instance IDs")
    parser.add_argument("--list-json", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--lease-json", type=Path)
    args = parser.parse_args(argv)
    try:
        registry = handlers()
        plans = select(
            classify(
                compile_scenarios(
                    load_scenarios(args.source),
                    args.profile,
                    registry,
                    grade=args.grade,
                ),
                args.suite,
            ),
            args.instances,
        )
        if args.list_json:
            print(json.dumps(inventory(plans), indent=2, allow_nan=False))
            return 0
        if args.out_dir is None or args.lease_json is None:
            raise ScenarioError(
                "execution requires --out-dir and --lease-json from the parent port planner"
            )
        if ":" in str(args.out_dir.resolve()):
            raise ScenarioError(
                "Java mock output directory cannot contain a colon (JVM -Xlog delimiter)"
            )
        leases = [validate_lease(args.lease_json, p["resource_budget"]) for p in plans]
    except (ScenarioError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Keep --list-json free of gRPC/proto/process initialization.
    from flexlb_test_framework.scenario.backend import JavaMockBackend
    from flexlb_test_framework.scenario.runtime import StageTimeout, execute_instance

    from flexlb_test_framework.workload.runtime import execute_workload

    cancelled = threading.Event()

    def cancel(signum, frame):
        cancelled.set()
        # Synchronous stage I/O must unwind promptly; cleanup has a distinct
        # deadline and its own cancel/join handling. A repeated TERM during
        # cleanup can fail that cleanup, which remains visible as ERROR.
        handler = signal.getsignal(signal.SIGALRM)
        if callable(handler) and getattr(handler, "__name__", "") == "expire":
            raise StageTimeout(f"child interrupted by signal {signum}")

    previous = {
        sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for index, (plan, lease) in enumerate(zip(plans, leases)):
            rows.append(
                (
                    execute_workload
                    if plan["test_kind"] == "workload"
                    else execute_instance
                )(
                    plan,
                    JavaMockBackend(lease),
                    registry,
                    args.out_dir / "instances" / instance_directory(plan["id"]),
                    cancelled=cancelled,
                    enforce_deadlines=True,
                )
            )
            payload = dict(schema_version=1, summary=summarize(rows), instances=rows)
            target = args.out_dir / "scenarios.json"
            temporary = args.out_dir / "scenarios.json.tmp"
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            temporary.replace(target)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return summarize(rows)["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
