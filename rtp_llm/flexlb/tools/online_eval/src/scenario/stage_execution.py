"""Execute ordered stages, validate adapter contracts and preserve findings."""

import json
from dataclasses import asdict
from scenario.stage_compiler import OUTPUTS
from scenario.contracts import StageOutput
from analysis.checks import CHECK_STATUSES
from runtime.deadline import Deadline, interruptible


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
        from analysis.checks import evaluate

        actual = ctx.resolve(params["actual"])
        result = evaluate("comparison", actual, params["op"], params["expected"])
        return StageOutput(
            {"passed": result.status == "PASS"}, [result],
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
        if kind == "nullable_number":
            import math
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"output {key} expected finite number or null")
            if value is None and not any(c.status == "ERROR" for c in result.checks):
                raise ValueError(f"output {key} can be null only with an evidence error")
        elif primitive is not None:
            if type(value) not in primitive:
                raise ValueError(f"output {key} expected {kind}")
        else:
            ctx.resource(value, kind, allow_stale=True)
    ids = [c.id for c in result.checks]
    if len(set(ids)) != len(ids) or set(ids) != set(check_ids):
        raise ValueError("adapter checks do not match declared check IDs")
    if any(c.status not in CHECK_STATUSES for c in result.checks):
        raise ValueError("invalid check status")
    json.dumps(asdict(result), allow_nan=False)


class StageExecutor:
    def run(self, ctx, handlers, started, end, cancelled, policy):
        instance = ctx.instance
        clock, sleeper = ctx.clock, ctx.sleeper
        enforce_deadlines = ctx.enforce_deadlines
        _policy = policy
        rows, finding_failures, finding_passes = [], [], []
        terminal_status, primary_error = "PASS", None
        blocked = False
        if _policy is not None:
            _policy.attach(ctx)
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
                if blocked:
                    rows.append(row)
                    continue
                deadline = Deadline(
                    min(end, t0 + spec["timeout_s"]), clock, sleeper, cancelled
                )
                try:
                    deadline.check()
                    action = spec["action"]
                    if _policy is not None:
                        _policy.before_stage(ctx, spec)
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
                        if check.status in {"SKIP", "WARNING"}:
                            continue
                        qualified = spec["id"] + "." + check.id
                        if check.status == "ERROR":
                            row["status"] = terminal_status = "ERROR"
                            primary_error = check.detail or "check evidence error"
                            blocked = True
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
                                blocked = blocked or not (
                                    _policy is not None
                                    and _policy.continue_after_failure(spec)
                                )
                    if _policy is not None:
                        _policy.after_stage(ctx, spec, row)
                    # Known findings do not prevent additional independent checks.
                except Exception as exc:
                    terminal_status = (
                        "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR"
                    )
                    primary_error = f"{type(exc).__name__}: {exc}"
                    row.update(status=terminal_status, error=primary_error)
                    blocked = True
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
        return rows, terminal_status, primary_error, finding_failures, finding_passes, cleanup
