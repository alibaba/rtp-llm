"""Independent, bounded Kineto control and conservative workload validation."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict
from pathlib import Path

from ..analysis.kineto import (
    validate_disk_native_evidence,
    validate_remote_native_evidence,
)


class ProfileController:
    """Operations must honor their transport deadlines/cancellation event.

    Daemon isolation also bounds the supervisor when an injected operation violates
    that contract. Python cannot forcibly terminate such an operation safely.
    """

    def __init__(
        self,
        start_capture,
        wait_capture,
        stop_capture,
        *,
        start_timeout_s=5,
        wait_timeout_s=120,
        stop_timeout_s=2,
    ):
        self.operations = dict(
            start=start_capture, wait=wait_capture, stop=stop_capture
        )
        self.timeouts = dict(
            start=start_timeout_s, wait=wait_timeout_s, stop=stop_timeout_s
        )
        if any(not math.isfinite(v) or v <= 0 for v in self.timeouts.values()):
            raise ValueError("profile operation deadlines must be finite and positive")
        self.cancel = threading.Event()
        self.log = []
        self.lock = threading.RLock()
        self.future = Future()
        self.stop_future = None
        self.started = False
        self.start_settled = threading.Event()
        self.stopped_during_start = False

    def _call(self, operation):
        future = Future()
        entry = dict(operation=operation, started_s=time.monotonic())
        with self.lock:
            self.log.append(entry)

        def invoke():
            try:
                value = self.operations[operation]()
                future.set_result(value)
            except BaseException as exc:
                future.set_exception(exc)

        threading.Thread(
            target=invoke, name="pressure-profile-" + operation, daemon=True
        ).start()
        try:
            value = future.result(self.timeouts[operation])
            entry.update(acknowledged_s=time.monotonic())
            if operation != "wait":
                entry["result"] = value
            return value
        except BaseException as exc:
            entry.update(failed_s=time.monotonic(), error=repr(exc))
            raise

    def start(self):
        with self.lock:
            if self.started:
                raise RuntimeError("profile controller already started")
            self.started = True

        def run():
            result = {}
            errors = []
            try:
                try:
                    if self.cancel.is_set():
                        raise RuntimeError("profile_cancelled_before_start")
                    self._call("start")
                finally:
                    self.start_settled.set()
                if self.cancel.is_set():
                    raise RuntimeError("profile_cancelled_during_start")
                result = self._call("wait") or {}
            except BaseException as exc:
                errors.append(repr(exc))
            finally:
                self.request_stop()
                try:
                    self.stop_future.result(self.timeouts["stop"] + 0.1)
                except BaseException as exc:
                    errors.append(repr(exc))
                if self.stopped_during_start:
                    try:
                        self._call("stop")
                    except BaseException as exc:
                        errors.append(repr(exc))
            if errors:
                result.update(trace_valid=False, control_errors=errors)
            result["control_log"] = list(self.log)
            self.future.set_result(result)

        threading.Thread(
            target=run, name="pressure-profile-controller", daemon=True
        ).start()

    def request_stop(self):
        self.cancel.set()
        with self.lock:
            if self.stop_future is not None:
                return
            self.stopped_during_start = not self.start_settled.is_set()
            self.stop_future = Future()

        def stop():
            try:
                self.stop_future.set_result(self._call("stop"))
            except BaseException as exc:
                self.stop_future.set_exception(exc)

        threading.Thread(
            target=stop, name="pressure-profile-cleanup", daemon=True
        ).start()

    def done(self):
        return self.future.done()

    def result(self, timeout_s):
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("invalid profile result deadline")
        try:
            return self.future.result(timeout_s)
        except FutureTimeoutError as exc:
            self.request_stop()
            raise TimeoutError("profile result deadline expired") from exc


def _analyze_profile(request, deadline):
    """Exec a fresh interpreter; timeout kills and reaps it before returning."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return dict(ranks={}, reasons=["profile_analysis_timeout"])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", __package__.rsplit(".", 1)[0] + ".analysis.worker"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=remaining,
            check=True,
            env=environment,
        )
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return dict(ranks={}, reasons=["profile_analysis_timeout"])
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        return dict(ranks={}, reasons=["profile_analysis_failed"], error=repr(exc))


def analyze_kineto_files(paths, steps, requested_steps, deadline):
    return _analyze_profile(
        dict(
            backend="kineto", paths=paths, steps=steps, requested_steps=requested_steps
        ),
        deadline,
    )


def analyze_nsys_file(path, steps, output, deadline):
    return _analyze_profile(
        dict(backend="nsys", path=str(path), steps=steps, output=str(output)), deadline
    )


class KinetoCapture:
    """PressureSession adapter. Supervisor hooks only copy bounded evidence.

    Transport callbacks are injected with explicit deadlines. Online readiness
    never parses traces; isolated final validation follows step drain.
    """

    def __init__(
        self,
        *,
        start_capture,
        read_status,
        stop_capture,
        trace_name,
        directory,
        ranks,
        instances,
        session_id,
        timeout_s,
        requested_steps=6,
        state_capacity=4096,
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trace_name):
            raise ValueError("invalid continuous trace name")
        self.trace_name, self.directory = trace_name, Path(directory)
        self.ranks, self.instances = tuple(ranks), dict(instances)
        self.session_id, self.requested_steps = session_id, requested_steps
        self.read_status, self.timeout_s = read_status, timeout_s
        self.steps = {rank: OrderedDict() for rank in ranks}
        self.offsets, self.statuses = {}, {}
        self.remote_counters = {rank: deque(maxlen=state_capacity) for rank in ranks}
        self.state_capacity, self.evictions = state_capacity, 0
        self.errors = set()
        self.started = False
        self.controller = ProfileController(
            start_capture,
            self._wait,
            stop_capture,
            start_timeout_s=5,
            wait_timeout_s=timeout_s + 2,
            stop_timeout_s=2,
        )

    def consume(self, event):
        if event.get("event") != "pressure":
            return
        raw = event.get("raw", {})
        for page in raw.get("response", {}).get("ranks", []):
            rank = int(page.get("world_rank", 0))
            if rank not in self.steps:
                self.errors.add("unexpected_step_rank")
                continue
            if page.get("service_instance") != self.instances[rank]:
                self.errors.add("step_instance_changed")
                continue
            if (
                page.get("remote_runtime")
                and not page.get("error")
                and not raw.get("diagnostics_error")
            ):
                # Both the sampled counter and the native capture bounds are on this rank's clock.
                stamp = int(page.get("sampled_monotonic_ns", 0))
                if stamp:
                    point = (
                        stamp,
                        stamp + int(page.get("collect_duration_ns", 0)),
                        int(page.get("remote_read_payload_bytes", 0)),
                        int(page.get("remote_write_acknowledged_payload_bytes", 0)),
                    )
                    points = self.remote_counters[rank]
                    if points and (
                        point[0] < points[-1][0]
                        or point[2] < points[-1][2]
                        or point[3] < points[-1][3]
                        or point[0] == points[-1][0]
                        and point != points[-1]
                    ):
                        self.errors.add("remote_capture_counter_regressed")
                    elif not points or points[-1] != point:
                        points.append(point)
            bounds = raw.get("rank_offset_bounds_s", {}).get(str(rank))
            if bounds is not None:
                self.offsets[rank] = bounds
            status = page.get("profile", {})
            if status.get("trace_name") != self.trace_name:
                status = self.statuses.get(rank, {})
            begin, end = int(status.get("capture_begin_ns", 0)), int(
                status.get("capture_end_ns", 0)
            )
            for record in page.get("steps", []):
                if record.get("session_id") != self.session_id:
                    self.errors.add("step_session_changed")
                    continue
                entered = int(record.get("forward_enter_ns", 0))
                if (begin and entered < begin) or (end and entered > end):
                    continue
                step_id = int(record.get("step_id", 0))
                values = dict(
                    step_id=step_id,
                    context_batch=int(record.get("context_count", 0)),
                    decode_batch=int(record.get("decode_count", 0)),
                    complete=record.get("completed", False),
                    begin_ns=int(record.get("preparation_begin_ns", 0)),
                    forward_enter_ns=int(record.get("forward_enter_ns", 0)),
                    end_ns=int(record.get("end_ns", 0)),
                )
                old = self.steps[rank].get(step_id)
                if old and old["complete"] and not values["complete"]:
                    continue
                self.steps[rank][step_id] = values
                if len(self.steps[rank]) > self.state_capacity:
                    self.steps[rank].popitem(last=False)
                    self.evictions += 1

    def start(self, now):
        self.started = True
        self.started_s = now
        self.deadline = time.monotonic() + self.timeout_s
        for records in self.steps.values():
            records.clear()
        self.evictions = 0
        self.controller.start()

    def pending(self, now):
        if self.started and now >= self.started_s + self.timeout_s:
            self.controller.request_stop()
            return False
        return self.started and not self.controller.done()

    def stop(self, reason, now):
        if self.started:
            self.controller.request_stop()

    def _wait(self):
        deadline = self.deadline
        cancel_deadline = None
        paths, last_error = {}, None
        while time.monotonic() < deadline:
            if self.controller.cancel.is_set():
                if cancel_deadline is None:
                    cancel_deadline = time.monotonic() + 2
                if time.monotonic() >= cancel_deadline:
                    break
            try:
                raw = self.read_status(min(1, max(0.01, deadline - time.monotonic())))
                pages = raw.get("response", {}).get("ranks", [])
                actual = [int(page.get("world_rank", 0)) for page in pages]
                if sorted(actual) != sorted(self.ranks):
                    raise ValueError("missing_or_duplicate_profile_rank")
                for page in pages:
                    rank = int(page.get("world_rank", 0))
                    status = page.get("profile", {})
                    self.statuses[rank] = status
                    if (
                        page.get("service_instance") != self.instances[rank]
                        or status.get("trace_name") != self.trace_name
                    ):
                        raise ValueError("profile_identity_mismatch")
                    if page.get("error") or not status.get("supported"):
                        raise ValueError("profile_rank_error")
                    if status.get("collection_error") or status.get("export_error"):
                        raise ValueError("profile_collection_or_export_error")
                    if not (
                        status.get("collection_complete")
                        and status.get("export_complete")
                        and status.get("state") == "finished"
                    ):
                        continue
                    if not status.get("file_name"):
                        raise ValueError("profile_artifact_identity_missing")
                    path = Path(status["file_name"])
                    if not path.is_absolute():
                        path = self.directory / path
                    if str(path.resolve()) in {
                        p for r, p in paths.items() if r != rank
                    }:
                        raise ValueError("duplicate_rank_trace")
                    path = path.resolve()
                    if not path.is_file() or path.stat().st_size == 0:
                        raise OSError("profile_artifact_not_ready")
                    paths[rank] = str(path)
                if len(paths) == len(self.ranks):
                    return dict(trace_files=paths)
            except (OSError, ValueError) as exc:
                last_error = repr(exc)
                if isinstance(exc, ValueError):
                    return dict(
                        trace_valid=False,
                        reasons=[str(exc)],
                        trace_files=paths,
                    )
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(0.05)
        return dict(
            trace_valid=False,
            reasons=["profile_trace_timeout"],
            last_error=last_error,
            trace_files=paths,
        )

    def _remote_window(self, rank):
        status = self.statuses[rank]
        begin, end = int(status["capture_begin_ns"]), int(status["capture_end_ns"])
        points = self.remote_counters[rank]
        before = next((p for p in reversed(points) if p[1] <= begin), None)
        after = next((p for p in points if p[0] >= end), None)
        if before is None or after is None:
            return {"reasons": ["remote_capture_boundary_unobserved"]}
        inner = [p for p in points if begin <= p[0] and p[1] <= end]
        result = {
            "reasons": [],
            "uncertain_byte_counts": [],
            "outer_bounds_ns": [before[0], after[1]],
            "inner_bounds_ns": [inner[0][0], inner[-1][1]] if inner else [],
        }
        for index, kind in ((2, "read_bytes"), (3, "write_bytes")):
            outer = after[index] - before[index]
            confirmed = inner[-1][index] - inner[0][index] if len(inner) >= 2 else 0
            result[kind] = confirmed
            result["possible_" + kind] = outer
            if outer != confirmed:
                result["uncertain_byte_counts"].append(kind)
        return result

    def finish(self, coverage):
        if not self.started:
            return dict(
                trace_valid=False, reasons=["profile_not_started"], pressure_valid=False
            )
        try:
            result = self.controller.result(5)
        except TimeoutError:
            result = dict(
                trace_valid=False,
                reasons=["profile_cleanup_timeout"],
                control_log=list(self.controller.log),
            )
        paths = result.get("trace_files", {})
        rank_steps, windows = {}, []
        reasons = set(result.get("reasons", [])) | set(self.errors)
        for rank in self.ranks:
            status = self.statuses.get(rank, {})
            begin, end = int(status.get("capture_begin_ns", 0)), int(
                status.get("capture_end_ns", 0)
            )
            bounds = self.offsets.get(rank)
            if begin <= 0 or end <= begin or not bounds:
                reasons.add("missing_capture_clock_boundaries")
            else:
                windows.append((begin / 1e9 + bounds[0], end / 1e9 + bounds[1]))
            if rank not in paths:
                reasons.add("missing_rank_trace")
                continue
            steps = [
                dict(
                    step,
                    profile_boundary=(step["begin_ns"] < begin or step["end_ns"] > end),
                )
                for step in self.steps[rank].values()
                if begin <= step["forward_enter_ns"] < end
            ]
            rank_steps[str(rank)] = steps
        analysis_started = time.monotonic()
        analysis = analyze_kineto_files(
            paths, rank_steps, self.requested_steps, self.deadline
        )
        analysis["remaining_budget_s"] = max(0, self.deadline - analysis_started)
        analysis["wall_elapsed_s"] = time.monotonic() - analysis_started
        rank_results = analysis.get("ranks", {})
        reasons.update(analysis.get("reasons", []))
        if self.evictions:
            reasons.add("profile_step_history_overflow")
        if result.get("control_errors"):
            reasons.add("profile_control_failed")
        pressure = None
        if len(windows) == len(self.ranks) and coverage is not None:
            # Conservatively cover every rank's entire capture including flush;
            # all target FULL groups must hold across their union in client time.
            start, end = min(w[0] for w in windows), max(w[1] for w in windows)
            pressure = dict(
                start_s=start, end_s=end, **asdict(coverage.assess(start, end))
            )
            if coverage.config.evidence_policy == "tree_then_mock_disk":
                summary = coverage.mock_cache_summary(start, end)
                for owner, counts in summary["owners"].items():
                    scopes = rank_results.get(owner, {}).get("native_cache_scopes", {})
                    reasons.update(
                        validate_disk_native_evidence(
                            scopes,
                            disk_read_bytes=counts["disk_read_bytes"],
                            successful_disk=counts["successful_disk_blocks"],
                            completed_writes=counts["disk_write_completed_records"],
                        )
                    )
                pressure["mock_cache"] = summary
            if coverage.config.evidence_policy == "tree_then_mock_remote":
                # Final drain stays in the epoch summary. A scope obligation here
                # requires counter movement strictly bracketed by this rank's trace.
                captured = {}
                for rank in self.ranks:
                    counts = self._remote_window(rank)
                    captured[str(rank)] = counts
                    if not counts["reasons"]:
                        scopes = rank_results.get(str(rank), {}).get(
                            "native_cache_scopes", {}
                        )
                        missing = validate_remote_native_evidence(
                            scopes,
                            read_bytes=counts["read_bytes"],
                            write_bytes=counts["write_bytes"],
                        )
                        possible_missing = validate_remote_native_evidence(
                            scopes,
                            read_bytes=counts["possible_read_bytes"],
                            write_bytes=counts["possible_write_bytes"],
                        )
                        counts["reasons"].extend(missing)
                        if set(possible_missing).difference(missing):
                            counts["reasons"].append(
                                "remote_capture_boundary_ambiguous"
                            )
                    reasons.update(counts["reasons"])
                pressure["mock_remote"] = {"ranks": captured}
        valid = (
            result.get("trace_valid", True)
            and not reasons
            and len(rank_results) == len(self.ranks)
            and all(r["trace_valid"] for r in rank_results.values())
        )
        result.update(
            trace_valid=bool(valid),
            reasons=sorted(reasons),
            ranks=rank_results,
            statuses={str(k): v for k, v in self.statuses.items()},
            pressure=pressure,
            pressure_valid=pressure["valid"] if pressure else False,
            trace_name=self.trace_name,
            state_evictions=self.evictions,
            analysis=analysis,
        )
        result["partial_files"] = list(paths.values())
        return result


class NsightCapture(KinetoCapture):
    """Duration window with the same bounded sidecar ingestion as Kineto.

    The client command envelope is conservative pressure evidence, not an exact
    GPU activity boundary. SQLite timestamps never participate in that envelope.
    """

    def __init__(
        self,
        *,
        command,
        duration_s,
        trace_name,
        directory,
        ranks,
        instances,
        session_id,
        timeout_s,
        command_timeout_s,
    ):
        if duration_s is None or not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("positive finite Nsight duration required")
        if not math.isfinite(command_timeout_s) or command_timeout_s <= 0:
            raise ValueError("positive finite Nsight command timeout required")
        self.duration_s = min(duration_s, timeout_s)
        self.command = command
        self.output = (Path(directory) / "nsys" / trace_name).resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.report = self.output.with_suffix(".nsys-rep")
        self.sqlite = self.output.with_suffix(".sqlite")
        if self.report.exists() or self.sqlite.exists():
            raise FileExistsError(self.output)
        self.capture_begin_s = self.capture_end_s = None
        self.capture_ready_s = None
        self.command_record = dict(backend="nsys", trace_name=trace_name)
        super().__init__(
            start_capture=self._start_capture,
            read_status=None,
            stop_capture=self._stop_capture,
            trace_name=trace_name,
            directory=directory,
            ranks=ranks,
            instances=instances,
            session_id=session_id,
            timeout_s=timeout_s,
            requested_steps=0,
        )
        # stop includes report flushing and a separate SQLite export, both with
        # their own subprocess deadline. Sender workers continue throughout.
        self.controller.timeouts.update(
            start=command_timeout_s + 1, stop=2 * command_timeout_s + 2
        )

    def _start_capture(self):
        self.capture_begin_s = time.monotonic()
        self.command(
            self.command_record,
            "start",
            "--sample=process-tree",
            "--cpuctxsw=process-tree",
            "--output=" + str(self.output),
        )
        self.capture_ready_s = time.monotonic()

    def consume(self, event):
        if event.get("event") != "pressure":
            return
        raw = event.get("raw", {})
        pages = []
        for page in raw.get("response", {}).get("ranks", []):
            bounds = raw.get("rank_offset_bounds_s", {}).get(
                str(int(page.get("world_rank", 0)))
            )
            records = []
            for step in page.get("steps", []):
                entered = int(step.get("forward_enter_ns", 0)) / 1e9
                if (
                    bounds
                    and self.capture_begin_s is not None
                    and entered + bounds[1] < self.capture_begin_s
                ):
                    continue
                if (
                    bounds
                    and self.capture_end_s is not None
                    and entered + bounds[0] > self.capture_end_s
                ):
                    continue
                records.append(step)
            pages.append(dict(page, steps=records))
        super().consume(
            dict(
                event,
                raw=dict(raw, response=dict(raw.get("response", {}), ranks=pages)),
            )
        )

    def _wait(self):
        self.controller.cancel.wait(
            max(
                0,
                min(self.capture_ready_s + self.duration_s, self.deadline)
                - time.monotonic(),
            )
        )
        return {}

    def _stop_capture(self):
        self.command(self.command_record, "stop")
        self.capture_end_s = time.monotonic()
        if not self.report.is_file() or self.report.stat().st_size == 0:
            raise RuntimeError("nsys stop did not produce a nonempty report")
        self.command(
            self.command_record,
            "export",
            "--type=sqlite",
            "--lazy=false",
            "--output=" + str(self.sqlite),
            str(self.report),
        )
        if not self.sqlite.is_file() or self.sqlite.stat().st_size == 0:
            raise RuntimeError("nsys export did not produce a nonempty SQLite file")

    def pending(self, now):
        # Stop at the capture deadline but keep cleanup/export visible as pending
        # until the session's own support deadline bounds further traffic.
        if self.started and now >= self.started_s + self.timeout_s:
            self.controller.request_stop()
        return self.started and not self.controller.done()

    def finish(self, coverage):
        if not self.started:
            return dict(
                trace_valid=False, pressure_valid=False, reasons=["profile_not_started"]
            )
        try:
            result = self.controller.result(5)
        except TimeoutError:
            result = dict(
                trace_valid=False,
                reasons=["profile_cleanup_timeout"],
                control_log=list(self.controller.log),
            )
        reasons = set(result.get("reasons", [])) | self.errors
        if result.get("control_errors"):
            reasons.add("profile_control_failed")
        if self.evictions:
            reasons.add("profile_step_history_overflow")
        snapshot = {
            rank: list(records.values()) for rank, records in self.steps.items()
        }
        analysis_started = time.monotonic()
        normalized = analyze_nsys_file(
            self.sqlite,
            snapshot,
            self.output.with_suffix(".normalized.json"),
            self.deadline,
        )
        normalized.setdefault("trace_valid", False)
        normalized["analysis_remaining_budget_s"] = max(
            0, self.deadline - analysis_started
        )
        normalized["analysis_wall_elapsed_s"] = time.monotonic() - analysis_started
        reasons.update(normalized.get("reasons", []))
        pressure = None
        if (
            self.capture_begin_s is not None
            and self.capture_end_s is not None
            and coverage is not None
        ):
            pressure = dict(
                start_s=self.capture_begin_s,
                end_s=self.capture_end_s,
                bounds_kind="client_start_call_to_stop_ack",
                **asdict(coverage.assess(self.capture_begin_s, self.capture_end_s)),
            )
        else:
            reasons.add("missing_capture_operation_bounds")
        result.update(normalized)
        result.update(
            trace_valid=normalized["trace_valid"] is True
            and not reasons
            and result.get("trace_valid", True),
            reasons=sorted(reasons),
            pressure=pressure,
            pressure_valid=pressure["valid"] if pressure else False,
            trace_name=self.trace_name,
            backend="nsys",
            capture_ready_s=self.capture_ready_s,
            duration_s=self.duration_s,
            operation_timeout_s=dict(self.controller.timeouts),
            nsys_commands=list(self.command_record.get("nsys_commands", [])),
            trace_files=[str(self.report), str(self.sqlite)],
            partial_files=[
                str(p) for p in sorted(self.output.parent.glob(self.trace_name + "*"))
            ],
            state_evictions=self.evictions,
        )
        return result
