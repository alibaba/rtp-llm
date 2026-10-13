"""Parallel orchestration over scenario_runner's compiled instance protocol v1.

Compilation and stage execution belong to the child. This module coordinates
selection, lane scheduling, child ownership, validated aggregation and delivery.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
import queue
from concurrent.futures import ThreadPoolExecutor
from artifacts.json_io import write_json
from pathlib import Path


from reporting.view_config import DEFAULT_VIEW
from scenario.instance_plan import (
    InstancePlanError,
    parse_catalog,
    plan_instances,
    order_instances,
    select_instances,
)
from runtime.resource_plan import ResourcePlanError, plan_lane_leases
from runtime import port_windows

SCENARIO_RUNNER = Path(__file__).resolve().parents[2] / "scripts/commands/list_cases.py"
from runtime.outcome import TERMINAL_STATUSES as STATUSES
from scenario.result_validation import error_result, read_results
from scenario.timing_store import timing_path, read_timings, write_timings


class ChildProcesses:
    """Own only this run's child sessions; stop admission before terminating them."""

    def __init__(self):
        self.lock = threading.Lock()
        self.processes = {}
        self.stopping = False

    def run(self, command, env, log_path, *, timeout=None, cleanup_timeout=10):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if self.stopping:
                raise RuntimeError("suite interrupted before child launch")
            with open(log_path, "wb") as output:
                child = subprocess.Popen(
                    command,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self.processes[child] = cleanup_timeout
        settled = False
        try:
            try:
                code = child.wait(timeout=timeout)
                settled = code >= 0
                return code
            except subprocess.TimeoutExpired:
                self._signal(child, signal.SIGTERM)
                self._wait_or_kill(child, cleanup_timeout)
                return 124
        finally:
            with self.lock:
                self.processes.pop(child, None)
            if settled and not self.stopping:
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    self._signal(child, signal.SIGKILL)
                    raise RuntimeError("runner exited with live descendant processes")

    @staticmethod
    def _signal(child, sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    def _wait_or_kill(self, child, timeout):
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._signal(child, signal.SIGKILL)
            child.wait(timeout=5)

    def stop(self):
        with self.lock:
            self.stopping = True
            children = dict(self.processes)
        for child in children:
            self._signal(child, signal.SIGTERM)
        deadline = time.monotonic() + max(children.values(), default=0)
        for child in children:
            self._wait_or_kill(child, max(0, deadline - time.monotonic()))


def _catalog(args):
    from scenario.selection_cli import selection_arguments
    instances = []
    if not SCENARIO_RUNNER.is_file():
        raise InstancePlanError(
            "list_cases.py is unavailable; install the scenario compiler/runner before selecting YAML"
        )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCENARIO_RUNNER),
            "--source",
            str(Path(args.case_dir).resolve()),
            "--profile",
            args.profile,
            "--grade",
            args.grade,
            *selection_arguments(args),
            *(["--experiment-params", args.experiment_params] if getattr(args, "experiment_params", None) else []),
            "--list-json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode:
        raise InstancePlanError(
            f"scenario list failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
        )
    try:
        payload = json.loads(proc.stdout)
    except ValueError as exc:
        raise InstancePlanError("scenario list did not return JSON") from exc
    compiled = parse_catalog(payload, source="yaml", profile=args.profile)
    if any((instance.metadata.get("grade") != args.grade for instance in compiled)):
        raise InstancePlanError("scenario list grade does not match requested grade")
    instances.extend(compiled)
    return instances


def _run_segment(index, part, group, lease, args, out, children):
    directory = out / f"lane{index}"
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(lease.child_env())
    results = []
    segments = []
    segment = directory / f"part{part}-yaml"
    segment.mkdir(parents=True, exist_ok=True)
    result_path = segment / "scenarios.json"
    lease_path = segment / "lease.json"
    write_json(lease_path, {"lease_schema_version": 1, **lease.to_manifest()})
    try:
        from scenario.execution_plan import freeze_plan
        plan_path = segment / "execution-plan.json"
        plan_sha = freeze_plan(plan_path, [instance.compiled for instance in group],
                               dependencies=group[0].dependencies)
        command = [
            sys.executable,
            str(SCENARIO_RUNNER),
            "--plan", str(plan_path), "--plan-sha256", plan_sha,
            "--source",
            str(Path(args.case_dir).resolve()),
            "--profile",
            args.profile,
            "--grade",
            args.grade,
            "--instances",
            ",".join((instance.id for instance in group)),
            "--out-dir",
            str(segment),
            "--lease-json",
            str(lease_path),
        ]
        if getattr(args, "retain_tsdb", False):
            command.append("--retain-tsdb")
        execution = [instance.metadata["execution"] for instance in group]
        timeout = (
            sum((item["timeout_s"] + item["cleanup_timeout_s"]
                 + item["finalize_timeout_s"] + item["report_timeout_s"]
                 + item["archive_timeout_s"] for item in execution))
            + 30
            if execution
            else None
        )
        cleanup_timeout = (
            max((item["cleanup_timeout_s"] for item in execution), default=0) + 15
        )
        rc = children.run(
            command,
            env,
            segment / "runner.log",
            timeout=timeout,
            cleanup_timeout=cleanup_timeout,
        )
        rows = read_results(result_path, group)
    except Exception as exc:
        rc = 1
        rows = [error_result(instance, f"child launch failed: {exc}") for instance in group]
    results.extend(({**row, "lane": index} for row in rows))
    segments.append(
        {
            "source": "yaml",
            "instance_ids": [instance.id for instance in group],
            "exit_code": rc,
            "watchdog_timeout": rc == 124,
            "result": str(result_path),
            "log": str(segment / "runner.log"),
        }
    )
    return {"lane": index, "segments": segments, "instances": results}



def _run_lane(index, lane, lease, args, out, children, dispatch=None):
    """One lease, one child per instance; failed cleanup retires this lane."""
    from scenario.lane_safety import cleanup_failure
    from runtime.port_windows import busy_ports
    results, segments = [], []
    pending = iter(lane)
    failure = None
    while True:
        if children.stopping:
            failure = "suite interrupted before dispatch"
            break
        try:
            instance = dispatch.get_nowait() if dispatch is not None else next(pending)
        except (queue.Empty, StopIteration):
            break
        busy = busy_ports(lease.ports())
        if busy:
            failure = "lane ports still occupied: " + str(busy)
            results.append(dict(error_result(instance, failure), not_started=True, lane=index))
            break
        part = _run_segment(index, len(segments), [instance], lease, args, out, children)
        results.extend(part["instances"])
        segments.extend(part["segments"])
        row = part["instances"][0]
        failure = cleanup_failure(row)
        if part["segments"][0]["exit_code"] and row["status"] in {"PASS", "FINDING-CONFIRMED", "FINDING-RESOLVED"}:
            failure = "child exit code disagrees with successful execution result"
        if part["segments"][0]["watchdog_timeout"]:
            failure = "child watchdog expired; lane cannot be reused"
        busy = busy_ports(lease.ports())
        if busy:
            failure = "lane ports not released: " + str(busy)
        if failure:
            break
    if failure and dispatch is None:
        results.extend(dict(error_result(instance, failure), not_started=True, lane=index)
                       for instance in pending)
    return {"lane": index, "segments": segments, "instances": results,
            "reusable": failure is None, "retirement_reason": failure}


def _aggregate(lanes, instances, args, elapsed):
    rows = [row for lane in lanes for row in lane["instances"]]
    order = {instance.id: i for (i, instance) in enumerate(instances)}
    rows.sort(key=lambda row: order[row["id"]])
    counts = {
        status: sum((row["status"] == status for row in rows)) for status in STATUSES
    }
    failed = counts["FAIL"] + counts["ERROR"] + counts["TIMEOUT"]
    rc = int(
        bool(
            failed
            or any(lane.get("retirement_reason") for lane in lanes)
            or any((part["exit_code"] for lane in lanes for part in lane["segments"]))
        )
    )
    return {
        "run_summary_schema_version": 1,
        "source": args.source,
        "summary": {
            "total": len(rows),
            "passed": counts["PASS"],
            "failed": failed,
            "errors": counts["ERROR"],
            "timeouts": counts["TIMEOUT"],
            "blocked": sum(bool(row.get("not_started")) for row in rows),
            "finding_confirmed": counts["FINDING-CONFIRMED"],
            "finding_resolved": counts["FINDING-RESOLVED"],
            "exit_code": rc,
            "profile": args.profile,
            "grade": args.grade,
            "parallel": len(lanes),
            "retired_lanes": sum(bool(lane.get("retirement_reason")) for lane in lanes),
            "wall_time_s": round(elapsed, 3),
        },
        "instances": rows,
        "cases": [{**row, "name": row["id"]} for row in rows],
        "lanes": [
            {key: value for (key, value) in lane.items() if key != "instances"}
            for lane in lanes
        ],
    }


def _print_planned_reports(instances):
    for instance in instances:
        if instance.metadata.get("test_kind") == "workload":
            names = instance.metadata.get("reports")
            if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
                names = [DEFAULT_VIEW]
            print(f"  planned reports {instance.id}: {', '.join(names)}")


def _print_generated_reports(rows):
    for row in rows:
        if row.get('artifacts'):
            print('experiment artifacts: ' + json.dumps(dict(instance=row['id'],
                artifacts=row['artifacts']), ensure_ascii=False))
        if row.get("test_kind") != "workload":
            continue
        workload = row.get("workload")
        reports = workload.get("reports") if isinstance(workload, dict) else None
        print("workload reports: " + json.dumps({
            "instance": row["id"], "status": row["status"],
            "reports": reports if isinstance(reports, dict) else None,
        }, ensure_ascii=False))


def run_structured(args: argparse.Namespace) -> int:
    """Keep window -> output lock -> child list -> lane execution ordering."""
    port_windows.resolve_port_bases(args)
    stamp = uuid.uuid4().hex
    out = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else Path(f"/tmp/flexlb_ft_instances_{stamp}")
    )
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        port_windows.acquire_run_lock(out)
    try:
        instances = select_instances(
            _catalog(args), exact_ids=args.instances, categories=args.categories
        )
        timings = read_timings(
            Path(args.timing_json) if args.timing_json else timing_path()
        )
        lanes = plan_instances(instances, args.parallel, timings)
        scheduling = getattr(args, "scheduling", "dynamic")
        if scheduling not in {"static", "dynamic"}:
            raise InstancePlanError("unknown scheduling policy")
        ordered = order_instances(instances, timings)
        budgets = ([[instance.budget for instance in instances]] * len(lanes)
                   if scheduling == "dynamic" else
                   [[instance.budget for instance in lane] for lane in lanes])
        leases = plan_lane_leases(
            budgets,
            master_base=port_windows.master_base(),
            mock_base=port_windows.mock_base(),
            mock_stride=args.mock_stride,
        )
        for i, lease in enumerate(leases):
            if list(lease.ports()) != port_windows.lane_ports(
                i, args.mock_stride, port_windows.master_base(), port_windows.mock_base()
            ):
                raise ResourcePlanError("resource planner and port preflight disagree")
            if lease.child_env() != port_windows.lane_env(i, args.mock_stride):
                raise ResourcePlanError(
                    "resource planner and child environment disagree"
                )
        from mode_profiles import master_mode_for_profile, resolve_mode
        runtime_plan = resolve_mode("scenario", master_mode_for_profile(args.profile))
        manifest = {
            "runner_plan_schema_version": 1,
            "experiment_params": json.loads(getattr(args, "experiment_params", None) or "{}"),
            "source": args.source,
            "profile": args.profile,
            "scheduling": scheduling,
            "retain_tsdb": bool(getattr(args, "retain_tsdb", False)),
            "dispatch_queue": [instance.id for instance in ordered] if scheduling == "dynamic" else [],
            "runtime_plan": runtime_plan,
            "grade": args.grade,
            "port_provenance": args.port_provenance,
            "instances": [instance.metadata for instance in instances],
            "lanes": [
                {
                    **lease.to_manifest(),
                    "instance_ids": [instance.id for instance in lane] if scheduling == "static" else [],
                }
                for (lease, lane) in zip(leases, lanes)
            ],
        }
    except (InstancePlanError, ResourcePlanError, subprocess.TimeoutExpired) as exc:
        print(f"error: instance planning failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"instance plan: source={args.source} profile={args.profile} instances={len(instances)} lanes={len(lanes)} out_dir={out}"
    )
    print("  scheduling: " + scheduling)
    if scheduling == "dynamic":
        print("  dispatch queue: " + ", ".join(instance.id for instance in ordered))
    else:
        for i, lane in enumerate(lanes):
            print(f"  lane {i}: " + ", ".join((instance.id for instance in lane)))
    _print_planned_reports(instances)
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return 0
    write_json(out / "manifest.json", manifest)
    started = time.monotonic()
    children = ChildProcesses()
    pool = ThreadPoolExecutor(max_workers=len(lanes))
    dispatch = queue.Queue() if scheduling == "dynamic" else None
    if dispatch is not None:
        for instance in ordered:
            dispatch.put(instance)
    futures = [
        pool.submit(_run_lane, i, lane, leases[i], args, out, children, dispatch)
        for (i, lane) in enumerate(lanes)
    ]
    interrupted = False
    try:
        results = [future.result() for future in futures]
    except KeyboardInterrupt:
        interrupted = True
        children.stop()
        results = [future.result() for future in futures]
    finally:
        pool.shutdown(wait=True)
    if dispatch is not None:
        reasons = "; ".join(row["retirement_reason"] for row in results
                            if row.get("retirement_reason"))
        while True:
            try:
                instance = dispatch.get_nowait()
            except queue.Empty:
                break
            blocked = error_result(instance, "no reusable lane remains: " + reasons)
            results[0]["instances"].append(dict(blocked, not_started=True, lane=None))
    payload = _aggregate(results, instances, args, time.monotonic() - started)
    payload["scheduling"] = scheduling
    payload["runtime_plan"] = runtime_plan
    if interrupted:
        payload["summary"]["exit_code"] = 130
        payload["summary"]["interrupted"] = True
    target = Path(args.json).resolve() if args.json else out / "aggregate.json"
    write_json(target, payload)
    _print_generated_reports(payload["instances"])
    if getattr(args, "archive", None):
        from artifacts.archive import create_archive

        sources = {"run": out}
        if target.parent != out:
            sources["aggregate"] = target
        archive_kind = "scenario" if any(i.metadata.get("test_kind") == "workload" for i in instances) else "case"
        create_archive(
            Path(args.archive), sources, kind=archive_kind,
            status="incomplete" if interrupted else "complete",
            metadata={"profile": args.profile, "grade": args.grade,
                      "exit_code": payload["summary"]["exit_code"],
                      "instance_count": payload["summary"]["total"]},
        )
        print(f"experiment archive: {args.archive}")
    try:
        write_timings(payload["instances"])
    except OSError as exc:
        print(f"warning: instance timing update failed: {exc}", file=sys.stderr)
    print(f"instance results: {target} exit_code={payload['summary']['exit_code']}")
    return payload["summary"]["exit_code"]
