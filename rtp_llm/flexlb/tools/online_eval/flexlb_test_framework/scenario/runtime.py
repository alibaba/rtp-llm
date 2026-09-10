"""Ordered stages with authentic handles, cooperative deadlines and LIFO cleanup."""

import copy
import json
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .compiler import OUTPUTS
from .contracts import CheckResult, ResourceHandle, StageOutput


class StageTimeout(TimeoutError):
    pass


@contextmanager
def interruptible(deadline, enabled=False):
    """Main-thread POSIX guard for synchronous backend calls, never a future timeout."""
    if not enabled:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("signal deadline enforcement requires the main thread")
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    entered = time.monotonic()

    def expire(signum, frame):
        raise StageTimeout("stage wall-clock deadline expired")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline.remaining()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer[0] > 0:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.001, previous_timer[0] - (time.monotonic() - entered)),
                previous_timer[1],
            )


class Deadline:
    def __init__(
        self, expires_at, clock=time.monotonic, sleeper=time.sleep, cancelled=None
    ):
        self.expires_at = expires_at
        self.clock = clock
        self.sleeper = sleeper
        self.cancelled = cancelled or threading.Event()

    def remaining(self):
        remaining = self.expires_at - self.clock()
        if self.cancelled.is_set() or remaining <= 0:
            raise StageTimeout("deadline expired or operation cancelled")
        return remaining

    def check(self):
        if self.cancelled.is_set() or self.clock() >= self.expires_at:
            raise StageTimeout("deadline expired or operation cancelled")

    def sleep(self, seconds):
        target = self.clock() + seconds
        while True:
            now = self.clock()
            if self.cancelled.is_set() or now >= self.expires_at:
                raise StageTimeout("deadline expired or operation cancelled")
            if now >= target:
                break
            self.sleeper(min(0.1, target - now, self.expires_at - now))
        self.check()


class RuntimeContext:
    def __init__(
        self, instance, backend, artifact_dir, clock, sleeper, enforce_deadlines=False
    ):
        self.instance = instance
        self.backend = backend
        self.artifact_dir = Path(artifact_dir)
        self.clock, self.sleeper = clock, sleeper
        self.env_epoch = 0
        self.env = self.ops = None
        self.outputs = {}
        self._resources = {}
        self._cleanup = []
        self.cleanup_results = []
        self.enforce_deadlines = enforce_deadlines

    def add_cleanup(self, name, callback):
        self._cleanup.append((name, callback))

    def register_resource(self, kind, value, cleanup=None, historical=False):
        handle = ResourceHandle(
            kind, f"resource_{len(self._resources) + 1}", self.env_epoch
        ).to_dict()
        self._resources[handle["id"]] = (handle, value, historical)
        if cleanup is not None:
            self.add_cleanup(handle["id"], cleanup)
        return dict(handle)

    def resolve(self, value):
        if isinstance(value, dict) and "$ref" in value:
            if set(value) != {"$ref"}:
                raise ValueError("reference cannot contain extra fields")
            parts = value["$ref"].split(".")
            if len(parts) != 4 or parts[0] != "stages" or parts[2] != "output":
                raise ValueError("invalid stage reference")
            return self.outputs[parts[1]][parts[3]]
        return value

    def resource(self, value, kind, allow_stale=False):
        handle = self.resolve(value)
        if not isinstance(handle, dict) or set(handle) != {"kind", "id", "env_epoch"}:
            raise ValueError("expected authentic resource handle")
        record = self._resources.get(handle["id"])
        if record is None or record[0] != handle or handle["kind"] != kind:
            raise ValueError("unknown or forged resource handle")
        if handle["env_epoch"] != self.env_epoch and not (allow_stale and record[2]):
            raise ValueError("stale environment epoch")
        return record[1]

    def cleanup(self, budget_s, retain_failed=False):
        # Instance expiry does not consume the separately reserved cleanup time.
        deadline = Deadline(self.clock() + budget_s, self.clock, self.sleeper)
        results, retry = [], []
        while self._cleanup:
            name, callback = self._cleanup.pop()
            started = self.clock()
            try:
                # Even an expired callback gets the opportunity to cancel its
                # owned calls/processes before checking its remaining join time.
                if self.clock() < deadline.expires_at:
                    with interruptible(deadline, self.enforce_deadlines):
                        callback(deadline)
                else:
                    callback(deadline)
                deadline.check()
                status, error = "PASS", None
            except Exception as exc:
                status = "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR"
                error = f"{type(exc).__name__}: {exc}"
                if retain_failed:
                    retry.append((name, callback))
            results.append(
                {
                    "id": name,
                    "status": status,
                    "error": error,
                    "duration_ms": int((self.clock() - started) * 1000),
                }
            )
        # A failed intermediate teardown must remain reachable by the final
        # separately budgeted cleanup; do not retry it in this same loop.
        self._cleanup.extend(reversed(retry))
        self.cleanup_results.extend(results)
        return results


def _core_action(action, ctx, params, deadline):
    if action == "setup":
        ctx.env_epoch += 1
        ctx.add_cleanup("environment", lambda d: ctx.backend.teardown(ctx, d))
        ctx.env, ctx.ops = ctx.backend.setup(ctx, ctx.instance["environment"], deadline)
        handle = ctx.register_resource("environment", ctx.env)
        return StageOutput({"environment": handle})
    if action == "request":
        # Backend registers the request set before the first RPC so a partial
        # submission failure still leaves every started request discoverable.
        handle = ctx.backend.start_requests(ctx, params, deadline)
        ctx.resource(handle, "requests")
        return StageOutput({"requests": handle, "count": params["count"]})
    if action in ("wait", "cancel"):
        requests = ctx.resource(params["requests"], "requests")
        if action == "wait":
            return StageOutput(ctx.backend.wait_requests(ctx, requests, deadline))
        return StageOutput(
            {"issued": ctx.backend.cancel_requests(ctx, requests, deadline)}
        )
    if action == "check":
        actual = ctx.resolve(params["actual"])
        expected = params["expected"]
        op = params["op"]
        passed = (
            actual == expected
            if op == "eq"
            else actual <= expected if op == "le" else actual >= expected
        )
        return StageOutput(
            {"passed": passed},
            [
                CheckResult(
                    "comparison",
                    "PASS" if passed else "FAIL",
                    actual=actual,
                    expected=expected,
                )
            ],
        )
    if action == "teardown":
        results = ctx.cleanup(deadline.remaining())
        if any(row["status"] != "PASS" for row in results):
            raise RuntimeError("explicit teardown cleanup failed")
        ctx.env_epoch += 1
        ctx.env = ctx.ops = None
        return StageOutput({"clean": True})
    raise ValueError(f"unknown core action {action}")


def _validate_output(ctx, result, expected, check_ids):
    if not isinstance(result, StageOutput) or set(result.output) != set(expected):
        raise ValueError("adapter output fields do not match declared schema")
    for key, kind in expected.items():
        value = result.output[key]
        primitive = {
            "boolean": (bool,),
            "integer": (int,),
            "number": (int, float),
            "string": (str,),
        }.get(kind)
        if primitive is not None:
            if type(value) not in primitive:
                raise ValueError(f"output {key} expected {kind}")
        else:
            ctx.resource(value, kind, allow_stale=True)
    ids = [c.id for c in result.checks]
    if len(set(ids)) != len(ids) or set(ids) != set(check_ids):
        raise ValueError("adapter checks do not match declared check IDs")
    if any(c.status not in ("PASS", "FAIL", "ERROR", "SKIP") for c in result.checks):
        raise ValueError("invalid check status")
    json.dumps(asdict(result), allow_nan=False)


def execute_instance(
    instance,
    backend,
    handlers=None,
    artifact_dir=".",
    clock=time.monotonic,
    sleeper=time.sleep,
    cancelled=None,
    enforce_deadlines=False,
):
    """Execute a compiled plan. Adapters must enforce deadlines in actual work."""
    handlers = dict(handlers or {})
    ctx = RuntimeContext(
        instance, backend, artifact_dir, clock, sleeper, enforce_deadlines
    )
    ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
    started = clock()
    end = started + instance["execution"]["timeout_s"]
    ctx.instance_deadline_s = end
    rows, finding_failures, finding_passes = [], [], []
    terminal_status, primary_error = "PASS", None
    try:
        for spec in instance["stages"]:
            t0 = clock()
            row = {
                "id": spec["id"],
                "action": spec["action"],
                "status": "BLOCKED",
                "output": {},
                "checks": [],
                "artifacts": [],
                "error": None,
                "duration_ms": 0,
                "started_s": t0,
            }
            if terminal_status != "PASS":
                rows.append(row)
                continue
            deadline = Deadline(
                min(end, t0 + spec["timeout_s"]), clock, sleeper, cancelled
            )
            try:
                deadline.check()
                action = spec["action"]
                with interruptible(deadline, enforce_deadlines):
                    if action in handlers:
                        descriptor = handlers[action]
                        result = descriptor.execute(ctx, spec["params"], deadline)
                        outputs = descriptor.outputs
                    else:
                        result = _core_action(action, ctx, spec["params"], deadline)
                        outputs = OUTPUTS[action]
                deadline.check()
                _validate_output(ctx, result, outputs, spec.get("check_ids", []))
                ctx.outputs[spec["id"]] = result.output
                row.update(
                    output=result.output,
                    checks=[asdict(c) for c in result.checks],
                    artifacts=result.artifacts,
                    status="PASS",
                )
                for check in result.checks:
                    if check.status == "SKIP":
                        continue
                    qualified = spec["id"] + "." + check.id
                    if check.status == "ERROR":
                        row["status"] = terminal_status = "ERROR"
                        primary_error = check.detail or "check evidence error"
                    elif qualified in instance["findings"]:
                        (
                            finding_failures
                            if check.status == "FAIL"
                            else finding_passes
                        ).append(qualified)
                        if check.status == "FAIL" and row["status"] == "PASS":
                            row["status"] = "FAIL"
                    elif check.status == "FAIL":
                        if row["status"] != "ERROR":
                            row["status"] = terminal_status = "FAIL"
                # Known findings do not prevent additional independent checks.
            except Exception as exc:
                terminal_status = (
                    "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR"
                )
                primary_error = f"{type(exc).__name__}: {exc}"
                row.update(status=terminal_status, error=primary_error)
            row["duration_ms"] = int((clock() - t0) * 1000)
            row["finished_s"] = clock()
            rows.append(row)
    finally:
        ctx.cleanup(instance["execution"]["cleanup_timeout_s"])
        cleanup = ctx.cleanup_results
    if any(c["status"] != "PASS" for c in cleanup):
        if terminal_status == "PASS":
            terminal_status = "ERROR"
        primary_error = primary_error or "cleanup failed"
    if terminal_status == "PASS":
        if not any(row["checks"] for row in rows):
            terminal_status, primary_error = "ERROR", "no checks executed"
        elif finding_failures:
            terminal_status = "FINDING-CONFIRMED"
        elif finding_passes:
            terminal_status = "FINDING-RESOLVED"
    result = {
        key: instance[key]
        for key in (
            "id",
            "scenario_id",
            "variant_id",
            "profile",
            "category",
            "source",
        )
    }
    result.update(
        status=terminal_status,
        grade=instance.get("grade", "normal"),
        effective_axes=instance.get("effective_axes"),
        effective_capabilities=instance.get("effective_capabilities"),
        stages=rows,
        cleanup=cleanup,
        error=primary_error,
        duration_ms=int((clock() - started) * 1000),
        finding_confirmed=finding_failures,
        finding_resolved=finding_passes,
        source_path=instance["source_path"],
        resource_budget=instance["resource_budget"],
        resolved_config=instance["environment"]["resolved_config"],
        clock_domain="monotonic",
    )
    if "implementation" in instance:
        result["implementation"] = copy.deepcopy(instance["implementation"])
    (ctx.artifact_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result
