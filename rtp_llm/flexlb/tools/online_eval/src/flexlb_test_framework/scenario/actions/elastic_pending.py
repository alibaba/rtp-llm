"""Explicit pending-wave construction and client-visible removal outcomes."""

import json
import threading
import time

from ..contracts import CheckResult, StageHandler, StageOutput


def validate_empty(params, plan):
    from .elastic import _validate

    return _validate(params, plan, set())


def wave_validate(params, plan):
    from .elastic import _validate

    keys = {
        "count",
        "input_len",
        "output_len",
        "spacing_s",
        "ready_timeout_s",
        "min_victim_routed",
        "min_pending",
        "request_timeout_s",
        "schedule_timeout_s",
        "stream_timeout_s",
    }
    p = _validate(params, plan, keys, keys)
    for k in keys:
        if type(p[k]) not in (int, float) or not 0 < p[k] <= 120:
            if k != "input_len" or type(p[k]) is not int or not 1 <= p[k] <= 65536:
                raise ValueError("invalid pending construction parameter: " + k)
    for k in ("count", "input_len", "output_len", "min_victim_routed", "min_pending"):
        if type(p[k]) is not int:
            raise ValueError("pending counts must be integers")
    return p


def wave(ctx, params, deadline):
    from .elastic import RecordedRequests, _snapshot

    initial = _snapshot(ctx, deadline)
    addresses = {
        name: initial[name]["grpc_addr"] for name in ("prefill-0", "prefill-1")
    }
    baseline_completed = initial["prefill-0"].get("completed")
    if type(baseline_completed) is not int or baseline_completed < 0:
        raise ValueError("pending construction lacks victim completed counter")
    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    records.pending_events = []
    records.pending_addresses = addresses
    records.pending_timed_out = []
    path = ctx.artifact_dir / f"elastic-pending-wave-{time.time_ns()}.json"

    def finish(d):
        records.cancel_active("pending_cleanup")
        try:
            for event in records.pending_events:
                if not event.wait(max(0, d.remaining())):
                    raise TimeoutError("pending request consumer did not exit")
        finally:
            path.with_name(path.stem + "-cleanup.json").write_text(
                json.dumps(records.snapshot_records(), indent=2)
            )

    handle = ctx.register_resource("requests", records, cleanup=finish)
    for _ in range(params["count"]):
        deadline.check()
        rid = ctx.ops.next_request_id()
        record = records.issue(rid, ctx.clock)
        event = threading.Event()
        records.pending_events.append(event)

        def run(record=record, event=event):
            try:
                records.run(
                    record,
                    dict(
                        input_len=params["input_len"],
                        output_len=params["output_len"],
                        block_keys=[
                            record["wire_request_id"] * 100 + j for j in range(3)
                        ],
                    ),
                    timeout_s=params["request_timeout_s"],
                    schedule_timeout_s=params["schedule_timeout_s"],
                    stream_timeout_s=params["stream_timeout_s"],
                )
            finally:
                event.set()

        try:
            threading.Thread(
                target=run, name=f"elastic-pending-{rid}", daemon=True
            ).start()
        except BaseException:
            event.set()
            raise
        # Keep enqueue windows distinct without waiting for a parked Schedule.
        deadline.sleep(params["spacing_s"])
    ready_end = min(deadline.expires_at, ctx.clock() + params["ready_timeout_s"])
    while True:
        rows = records.snapshot_records()
        victim_rows = [
            r for r in rows if r.get("prefill_addr") == addresses["prefill-0"]
        ]
        pending_rows = [
            r
            for r in rows
            if r["schedule"]["started_s"] is not None
            and r["schedule"]["ended_s"] is None
            and r["consumer_exit_s"] is None
        ]
        if (
            len(victim_rows) >= params["min_victim_routed"]
            and len(pending_rows) >= params["min_pending"]
        ):
            break
        if ctx.clock() >= ready_end:
            break
        deadline.sleep(0.01)
    snap = _snapshot(ctx, deadline)["prefill-0"]
    completed = snap.get("completed")
    if type(completed) is not int or completed < baseline_completed:
        raise ValueError("pending construction lacks monotonic completed counter")
    counters = [snap.get(k) for k in ("waiting", "running")]
    if any(type(v) is not int or v < 0 for v in counters):
        raise ValueError("pending construction lacks victim waiting/running counters")
    rows = records.snapshot_records()
    victim_rows = [r for r in rows if r.get("prefill_addr") == addresses["prefill-0"]]
    pending_rows = [
        r
        for r in rows
        if r["schedule"]["started_s"] is not None
        and r["schedule"]["ended_s"] is None
        and r["consumer_exit_s"] is None
    ]
    exited = [
        r["wire_request_id"]
        for r in victim_rows
        if r["transport_terminal_s"] is not None
    ]
    estimate = len(pending_rows)
    evidence = dict(
        victim_routed=len(victim_rows),
        engine_waiting_running=counters,
        pending_estimate=estimate,
        terminal_before_removal=exited,
        engine_completed_delta=completed - baseline_completed,
        inference="Schedule started but has not returned; engine counts are diagnostic only",
        pending_request_ids=[r["wire_request_id"] for r in pending_rows],
        snapshot=snap,
        records=records.snapshot_records(),
    )
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        output=dict(requests=handle),
        checks=[
            CheckResult(
                "victim_routed",
                "PASS" if len(victim_rows) >= params["min_victim_routed"] else "FAIL",
                actual=len(victim_rows),
                expected=params["min_victim_routed"],
            ),
            CheckResult(
                "pending_nonempty",
                "PASS" if estimate >= params["min_pending"] else "FAIL",
                actual=estimate,
                expected=params["min_pending"],
                evidence=evidence,
            ),
            CheckResult(
                "no_completed_interference",
                "PASS" if not exited and completed == baseline_completed else "FAIL",
                actual=dict(
                    client_terminal=exited,
                    engine_completed_delta=completed - baseline_completed,
                ),
            ),
        ],
        artifacts=[str(path)],
    )


def collect_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"requests", "mutation"}, {"requests", "mutation"})
    plan.reference(p["requests"], "requests")
    plan.reference(p["mutation"], "snapshot")
    return p


def classify(record, removed_s, timed_out=False):
    from .elastic import request_success

    terminal = record["transport_terminal_s"]
    latency = terminal - removed_s if terminal is not None else None
    explicit_error = (
        record["business_error_code"] not in (None, 0)
        or record["stream"]["status"] not in (None, "OK")
        or (
            record["schedule"]["status"] == "REJECTED"
            and bool(record["schedule"]["error"])
        )
    )
    if timed_out or terminal is None or record["cancel"]["requested_s"] is not None:
        kind = "hang"
    elif request_success(record):
        kind = "completed"
    elif explicit_error:
        kind = "error"
    else:
        kind = "empty"
    shape = (
        "completed"
        if kind == "completed"
        else (
            "fast_fail"
            if kind == "error" and latency <= 5
            else (
                "stale_window_fail"
                if kind == "error" and latency <= 16
                else "slow_fail" if kind == "error" else "no_terminal"
            )
        )
    )
    return dict(
        shape=shape,
        rid=record["wire_request_id"],
        kind=kind,
        latency_s=latency,
        visible_within_40s=kind in ("completed", "error")
        and latency is not None
        and latency <= 40,
    )


def collect(ctx, params, deadline):
    from .elastic import completeness

    records = ctx.resource(params["requests"], "requests")
    mutation = ctx.resource(params["mutation"], "snapshot")
    removed_s = mutation["started_s"]
    # Preserve the old per-fired-stream 45s collection wait and independent
    # terminal timestamps. A collector cancellation is never a visible success.
    for record, event in zip(records.snapshot_records(), records.pending_events):
        if not event.wait(min(45, max(0, deadline.remaining()))):
            records.pending_timed_out.append(record["wire_request_id"])
            with records._lock:
                active = records._calls.get(record["wire_request_id"])
                if active:
                    records._cancel_call(*active, "pending_collection_timeout")
            if not event.wait(min(5, max(0, deadline.remaining()))):
                raise TimeoutError("pending collector lacks consumer exit proof")
    rows = records.snapshot_records()
    outcomes = [
        dict(
            classify(r, removed_s, r["wire_request_id"] in records.pending_timed_out),
            route=r.get("prefill_addr"),
        )
        for r in rows
    ]
    result = dict(
        summary=completeness(rows),
        outcomes=outcomes,
        victim_addr=records.pending_addresses["prefill-0"],
        removed_s=removed_s,
        collected_s=ctx.clock(),
        records=rows,
    )
    path = ctx.artifact_dir / f"elastic-pending-outcomes-{time.time_ns()}.json"
    path.write_text(json.dumps(result, indent=2))
    return StageOutput(
        output=dict(
            result=ctx.register_resource("snapshot", result, historical=True),
            summary=ctx.register_resource(
                "snapshot", result["summary"], historical=True
            ),
        ),
        artifacts=[str(path)],
    )


def visible_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"result"}, {"result"})
    plan.reference(p["result"], "snapshot")
    return p


def visible(ctx, params, deadline):
    deadline.check()
    result = ctx.resource(params["result"], "snapshot")
    victim = [o for o in result["outcomes"] if o["route"] == result["victim_addr"]]
    return StageOutput(
        checks=[
            CheckResult(
                "victim_visible_terminal",
                (
                    "PASS"
                    if victim and all(o["visible_within_40s"] for o in victim)
                    else "FAIL"
                ),
                actual=victim,
                expected="every victim-routed request completed or explicit error within 40s of remove start",
            )
        ]
    )


def remove(ctx, params, deadline):
    """Preserve the 60s graceful drain and 95s HTTP client cap."""
    from .elastic import _engine_identity, _snapshot
    from .elastic_concurrent import mutation_http

    path = ctx.artifact_dir / f"elastic-pending-remove-{time.time_ns()}.json"
    evidence = dict(operation="remove", complete=False)
    try:
        before = _snapshot(ctx, deadline)
        victim = "prefill-0"
        role, port = _engine_identity(before[victim])
        if role != "prefill":
            raise ValueError("pending victim must be a prefill")
        body = dict(engine=victim, mode="graceful", drain_timeout_ms=60000)
        # Timestamp immediately before the actual scale-in call, not snapshot I/O.
        evidence.update(before=before, request=body, started_s=ctx.clock())
        response = mutation_http(ctx.ops, "remove_engine", deadline, body)
        evidence["response"] = response
        after = _snapshot(ctx, deadline)
        evidence["after"] = after
        if not isinstance(response, dict) or not (
            response.get("status") == "ok"
            and response.get("action") == "removed"
            and response.get("engine") == victim
            and response.get("port") == port
            and response.get("mode") == "graceful"
            and type(response.get("drained")) is bool
            and victim not in after
        ):
            raise ValueError("pending remove identity/ack/snapshot mismatch")
        evidence.update(complete=True, engine=victim, role=role, port=port)
        return StageOutput(
            output=dict(
                port=port,
                mutation=ctx.register_resource("snapshot", evidence, historical=True),
            ),
            checks=[
                CheckResult(
                    "membership", "PASS", actual=dict(engine=victim, present=False)
                )
            ],
            artifacts=[str(path)],
        )
    except BaseException as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence["ended_s"] = ctx.clock()
        path.write_text(json.dumps(evidence, indent=2))


def accounting(ctx, params, deadline):
    from .elastic_lifecycle import accounting_window

    # Called after collection in the explicit YAML. The 50s clock starts here.
    return accounting_window(ctx, deadline, 50)


def recovery_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"concurrency"})
    if "concurrency" in p and (
        type(p["concurrency"]) is not int or not 1 <= p["concurrency"] <= 10
    ):
        raise ValueError("recovery concurrency must be an integer in [1,10]")
    return p


def recovery(ctx, params, deadline):
    from .elastic_lifecycle import bounded_batch

    return bounded_batch(
        ctx,
        20,
        deadline,
        min_success_rate=0.95,
        concurrency=params.get("concurrency", 10),
    )


def batch_path_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"requests"}, {"requests"})
    plan.reference(p["requests"], "requests")
    return p


def batch_path(ctx, params, deadline):
    """FetchResponse follows enqueued_by_master; it does not prove EnqueueBatch ran."""
    deadline.check()
    rows = ctx.resource(params["requests"], "requests").snapshot_records()
    admitted = [
        r for r in rows if r["schedule"]["status"] == "OK" and r["prefill_addr"]
    ]
    methods = {str(r["wire_request_id"]): r["stream"]["method"] for r in admitted}
    return StageOutput(
        checks=[
            CheckResult(
                "batch_fetch",
                (
                    "PASS"
                    if methods and all(v == "FetchResponse" for v in methods.values())
                    else "FAIL"
                ),
                actual=methods,
                expected="every admitted stream uses the batch FetchResponse path",
            )
        ]
    )


HANDLERS = [
    StageHandler(
        "elastic_pending_batch_path",
        batch_path_validate,
        batch_path,
        {},
        checks=frozenset({"batch_fetch"}),
    ),
    StageHandler(
        "elastic_pending_remove",
        validate_empty,
        remove,
        {"port": "integer", "mutation": "snapshot"},
        checks=frozenset({"membership"}),
    ),
    StageHandler(
        "elastic_pending_accounting",
        validate_empty,
        accounting,
        {},
        checks=frozenset({"scheduler", "prefill_batches", "decode_load"}),
    ),
    StageHandler(
        "elastic_pending_recovery",
        recovery_validate,
        recovery,
        {"result": "snapshot"},
        checks=frozenset({"complete", "success_rate"}),
    ),
    StageHandler(
        "elastic_pending_wave",
        wave_validate,
        wave,
        {"requests": "requests"},
        checks=frozenset(
            {"victim_routed", "pending_nonempty", "no_completed_interference"}
        ),
    ),
    StageHandler(
        "elastic_pending_collect",
        collect_validate,
        collect,
        {"result": "snapshot", "summary": "snapshot"},
    ),
    StageHandler(
        "elastic_pending_visible",
        visible_validate,
        visible,
        {},
        checks=frozenset({"victim_visible_terminal"}),
    ),
]


def terminal_validate(params, plan):
    from .elastic import _validate

    p = _validate(
        params, plan, {"result", "terminal_bound_s"}, {"result", "terminal_bound_s"}
    )
    plan.reference(p["result"], "snapshot")
    if (
        type(p["terminal_bound_s"]) not in (int, float)
        or not 0 < p["terminal_bound_s"] <= 120
    ):
        raise ValueError("pending terminal bound must be in (0,120]")
    return p


def terminal_contract(result, bound_s):
    from .elastic import request_success

    bad = []
    removed_s = result["removed_s"]
    for record in result["records"]:
        end = record.get("transport_terminal_s")
        outcome = classify(record, removed_s)
        # An error that predates the removal is not explained by the stale window.
        valid = (
            record.get("consumer_exit_s") is not None
            and end is not None
            and record["cancel"]["requested_s"] is None
            and outcome["kind"] in {"completed", "error"}
            and end - removed_s <= bound_s
            and record["consumer_exit_s"] - removed_s <= bound_s
            and (request_success(record) or end >= removed_s)
        )
        if not valid:
            bad.append(record["wire_request_id"])
    return bool(result["records"]) and not bad, bad


def terminal(ctx, params, deadline):
    deadline.check()
    result = ctx.resource(params["result"], "snapshot")
    good, bad = terminal_contract(result, params["terminal_bound_s"])
    return StageOutput(
        checks=[
            CheckResult(
                "all_terminal",
                "PASS" if good else "FAIL",
                actual=bad,
                expected="all issued requests have an uncancelled bounded terminal; pre-remove errors fail",
                evidence=result,
            )
        ]
    )


HANDLERS.append(
    StageHandler(
        "elastic_pending_terminal",
        terminal_validate,
        terminal,
        {},
        checks=frozenset({"all_terminal"}),
    )
)
