"""Parallel orchestration over scenario_runner's compiled instance protocol v1.

Compilation and stage execution belong to scenario_runner. This module owns
selection, lane assignment, port reservations and result completeness only.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .instance_plan import (
    InstancePlanError,
    parse_catalog,
    plan_instances,
    select_instances,
)
from .resource_plan import ResourcePlanError, plan_lane_leases

SCENARIO_RUNNER = Path(__file__).resolve().parents[1] / "scenario_runner.py"
STATUSES = {"PASS", "FAIL", "ERROR", "TIMEOUT", "FINDING-CONFIRMED", "FINDING-RESOLVED"}


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
        try:
            try:
                return child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._signal(child, signal.SIGTERM)
                self._wait_or_kill(child, cleanup_timeout)
                return 124
        finally:
            with self.lock:
                self.processes.pop(child, None)

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
    instances = []
    if not SCENARIO_RUNNER.is_file():
        raise InstancePlanError(
            "scenario_runner.py is unavailable; install the scenario compiler/runner before selecting YAML"
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


def _timing_path():
    return Path(
        os.environ.get(
            "FLEXLB_FT_INSTANCE_TIMING_BASELINE",
            "/tmp/flexlb_ft_instance_timing_v1.json",
        )
    )


def _read_timings(path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1 or not isinstance(
            data.get("instances"), list
        ):
            raise ValueError("expected versioned instance timings")
        timings = {}
        for row in data["instances"]:
            duration = row.get("duration_ms")
            if (
                isinstance(row.get("id"), str)
                and type(duration) in (int, float)
                and math.isfinite(duration)
                and (duration > 0)
            ):
                timings[row["id"]] = duration / 1000
        return timings
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(
            f"warning: cannot use instance timings {path}: {exc}; using static estimates",
            file=sys.stderr,
        )
        return {}


def _write_timings(rows):
    path = _timing_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = _read_timings(path)
        for row in rows:
            if (
                row["status"] not in {"ERROR", "TIMEOUT"}
                and type(row.get("duration_ms")) in (int, float)
                and math.isfinite(row["duration_ms"])
                and (row["duration_ms"] > 0)
            ):
                merged[row["id"]] = row["duration_ms"] / 1000
        doc = {
            "schema_version": 1,
            "instances": [
                {"id": key, "duration_ms": value * 1000}
                for (key, value) in sorted(merged.items())
            ],
        }
        _write_json(path, doc)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _error(instance, message):
    return {
        **instance.metadata,
        "status": "ERROR",
        "duration_ms": 0,
        "stages": [],
        "cleanup": [],
        "error": message,
    }


def _execution_issues(row):
    """Validate nested evidence before accepting a child's success classification."""
    issues = []
    (stages, cleanup) = (row.get("stages", []), row.get("cleanup", []))
    if not isinstance(stages, list) or not isinstance(cleanup, list):
        return ["stages and cleanup must be arrays"]
    for item in cleanup:
        if (
            not isinstance(item, dict)
            or item.get("status") != "PASS"
            or item.get("error")
        ):
            issues.append("cleanup did not complete successfully")
    failed_checks = set()
    green = row["status"] in {"PASS", "FINDING-CONFIRMED", "FINDING-RESOLVED"}
    for stage in stages:
        if not isinstance(stage, dict):
            issues.append("stage must be an object")
            continue
        status = stage.get("status")
        if status not in {"PASS", "FAIL", "BLOCKED", "SKIP"} or stage.get("error"):
            issues.append("stage has an execution error or unknown status")
        checks = stage.get("checks", [])
        if not isinstance(checks, list):
            issues.append("stage checks must be an array")
            continue
        failures = []
        for check in checks:
            if (
                not isinstance(check, dict)
                or check.get("status") not in {"PASS", "FAIL"}
                or check.get("error")
            ):
                issues.append("check has an execution error or unknown status")
                continue
            if check["status"] == "FAIL":
                failures.append(check)
                failed_checks.add(f"{stage.get('id')}.{check.get('id')}")
        if status == "PASS" and failures:
            issues.append("PASS stage contains a failed check")
        if status in {"BLOCKED", "SKIP"} and (checks or green):
            issues.append("unexecuted stage contradicts successful instance")
        if green and status == "FAIL" and (not failures):
            issues.append("failed stage has no failed check evidence")
    if row["status"] in {"PASS", "FINDING-RESOLVED"} and failed_checks:
        issues.append("successful instance contains failed checks")
    if row["status"] == "FINDING-CONFIRMED":
        confirmed = row.get("finding_confirmed", [])
        if (
            not isinstance(confirmed, list)
            or any((not isinstance(x, str) for x in confirmed))
            or (not failed_checks)
            or (set(confirmed) != failed_checks)
        ):
            issues.append(
                "finding classification does not match explicit failed checks"
            )
    return issues


def _read_results(path, group):
    """Missing, duplicated, unexpected or malformed rows cannot become a green run."""
    expected = {instance.id: instance for instance in group}
    try:
        data = json.loads(path.read_text())
        if (
            type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
            or (not isinstance(data.get("instances"), list))
        ):
            raise ValueError("unsupported scenario result schema")
        rows = data["instances"]
        if not isinstance(rows, list):
            raise ValueError("result rows must be an array")
        by_id = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("result row must be an object")
            identity = row.get("id")
            if identity not in expected or identity in by_id:
                raise ValueError(f"unexpected or duplicate result: {identity!r}")
            if row.get("status") not in STATUSES:
                raise ValueError(f"unsupported result status: {row.get('status')!r}")
            if row.get("profile") != expected[identity].profile:
                raise ValueError(f"result profile mismatch: {identity}")
            if expected[identity].metadata.get("grade") is not None:
                if row.get("grade") != expected[identity].metadata["grade"]:
                    raise ValueError(f"result grade mismatch: {identity}")
            row = dict(row)
            failures = _execution_issues(row)
            if (row.get("error") or failures) and row["status"] not in {
                "ERROR",
                "TIMEOUT",
            }:
                row["original_status"] = row["status"]
                row["status"] = "ERROR"
                row["error"] = (
                    row.get("error")
                    or "; ".join(failures)
                    or "stage/cleanup error cannot be classified as a finding"
                )
            if row["status"] in {"PASS", "FINDING-CONFIRMED", "FINDING-RESOLVED"}:
                stages = row.get("stages", [])
                executed_checks = [
                    check
                    for stage in stages
                    if isinstance(stage, dict)
                    and stage.get("status") in {"PASS", "FAIL"}
                    for check in stage.get("checks", [])
                    if isinstance(check, dict)
                    and check.get("status") in {"PASS", "FAIL"}
                ]
                if not executed_checks:
                    row["original_status"] = row["status"]
                    row["status"] = "ERROR"
                    row["error"] = "successful instance has no executed checks"
            by_id[identity] = {**row, **expected[identity].metadata, "id": identity}
        return [
            by_id.get(identity) or _error(instance, "missing instance result")
            for (identity, instance) in expected.items()
        ]
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [
            _error(instance, f"invalid child result {path}: {exc}")
            for instance in group
        ]


def _run_lane(index, lane, lease, args, out, children):
    directory = out / f"lane{index}"
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(lease.child_env())
    results = []
    segments = []
    group = list(lane)
    segment = directory / "part0-yaml"
    segment.mkdir(parents=True, exist_ok=True)
    result_path = segment / "scenarios.json"
    lease_path = segment / "lease.json"
    _write_json(lease_path, {"schema_version": 1, **lease.to_manifest()})
    command = [
        sys.executable,
        str(SCENARIO_RUNNER),
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
    try:
        execution = [instance.metadata["execution"] for instance in group]
        timeout = (
            sum((item["timeout_s"] + item["cleanup_timeout_s"] for item in execution))
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
        rows = _read_results(result_path, group)
    except Exception as exc:
        rc = 1
        rows = [_error(instance, f"child launch failed: {exc}") for instance in group]
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
            or any((part["exit_code"] for lane in lanes for part in lane["segments"]))
        )
    )
    return {
        "schema_version": 1,
        "source": args.source,
        "summary": {
            "total": len(rows),
            "passed": counts["PASS"],
            "failed": failed,
            "errors": counts["ERROR"],
            "timeouts": counts["TIMEOUT"],
            "finding_confirmed": counts["FINDING-CONFIRMED"],
            "finding_resolved": counts["FINDING-RESOLVED"],
            "exit_code": rc,
            "profile": args.profile,
            "grade": args.grade,
            "parallel": len(lanes),
            "wall_time_s": round(elapsed, 3),
        },
        "instances": rows,
        "cases": [{**row, "name": row["id"]} for row in rows],
        "lanes": [
            {key: value for (key, value) in lane.items() if key != "instances"}
            for lane in lanes
        ],
    }


def run_structured(args: argparse.Namespace, ports) -> int:
    """Keep window -> output lock -> child list -> lane execution ordering."""
    ports._resolve_port_bases(args)
    stamp = uuid.uuid4().hex
    out = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else Path(f"/tmp/flexlb_ft_instances_{stamp}")
    )
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        ports._acquire_run_lock(out)
    try:
        instances = select_instances(
            _catalog(args), exact_ids=args.instances, categories=args.categories
        )
        timings = _read_timings(
            Path(args.timing_json) if args.timing_json else _timing_path()
        )
        lanes = plan_instances(instances, args.parallel, timings)
        leases = plan_lane_leases(
            [[instance.budget for instance in lane] for lane in lanes],
            master_base=ports._master_base(),
            mock_base=ports._mock_base(),
            mock_stride=args.mock_stride,
        )
        for i, lease in enumerate(leases):
            if list(lease.ports()) != ports._lane_ports(
                i, args.mock_stride, ports._master_base(), ports._mock_base()
            ):
                raise ResourcePlanError("resource planner and port preflight disagree")
            if lease.child_env() != ports.lane_env(i, args.mock_stride):
                raise ResourcePlanError(
                    "resource planner and child environment disagree"
                )
        manifest = {
            "schema_version": 1,
            "source": args.source,
            "profile": args.profile,
            "grade": args.grade,
            "port_provenance": args.port_provenance,
            "instances": [instance.metadata for instance in instances],
            "lanes": [
                {
                    **lease.to_manifest(),
                    "instance_ids": [instance.id for instance in lane],
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
    for i, lane in enumerate(lanes):
        print(f"  lane {i}: " + ", ".join((instance.id for instance in lane)))
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return 0
    _write_json(out / "manifest.json", manifest)
    started = time.monotonic()
    children = ChildProcesses()
    pool = ThreadPoolExecutor(max_workers=len(lanes))
    futures = [
        pool.submit(_run_lane, i, lane, leases[i], args, out, children)
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
    payload = _aggregate(results, instances, args, time.monotonic() - started)
    if interrupted:
        payload["summary"]["exit_code"] = 130
        payload["summary"]["interrupted"] = True
    target = Path(args.json).resolve() if args.json else out / "aggregate.json"
    _write_json(target, payload)
    try:
        _write_timings(payload["instances"])
    except OSError as exc:
        print(f"warning: instance timing update failed: {exc}", file=sys.stderr)
    print(f"instance results: {target} exit_code={payload['summary']['exit_code']}")
    return payload["summary"]["exit_code"]
