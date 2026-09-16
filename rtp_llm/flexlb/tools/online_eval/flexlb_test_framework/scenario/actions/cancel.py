"""Explicit cancellation RPCs, bounded observations and independent owner checks."""

from __future__ import annotations

import threading

from ..contracts import CheckResult, StageHandler, StageOutput
from . import status_protocol as status
from .elastic import request_success


class CancelRequests(status.StatusRequests):
    """Reuse the core Schedule/stream driver, with explicit frontend stream opening."""

    def __init__(self, ctx, params):
        self.auto_consume = params["consume"] == "immediate"
        super().__init__(ctx, dict(params, consume="immediate"))
        for child in self.children:
            if params.get("unique_block_keys"):
                child.params["block_keys"] = [
                    child.prepared["wire_request_id"] * 100 + 1
                ]
            original = child._start_consumer

            def gated(entry, end, start=original):
                if self.auto_consume:
                    start(entry, end)

            child._start_consumer = gated
            submit = child.submit

            def submitted(deadline, original_submit=submit, owned=child):
                try:
                    return original_submit(deadline)
                except Exception as error:
                    # Cancelling an owned grpc.Future produces FutureCancelledError,
                    # not a gRPC wire status. Keep this distinct from unknown ERROR.
                    entries = owned.entries
                    if len(entries) != 1:
                        raise
                    entry, record = entries[0], entries[0]["record"]
                    call = entry["call"]
                    cancelled = getattr(call, "cancelled", None)
                    if (
                        type(error).__name__ != "FutureCancelledError"
                        or type(error).__module__.split(".")[0] != "grpc"
                        or not callable(cancelled)
                        or not cancelled()
                        or record["cancel"]["requested_s"] is None
                        or record.get("transport_terminal_s") is None
                        or record.get("consumer_exit_s") is None
                    ):
                        raise
                    owned.update(record, schedule={"status": "FUTURE_CANCELLED"})
                    owned.persist()

            child.submit = submitted

    def begin(self, deadline):
        if self.dispatched:
            raise RuntimeError("cohort was already dispatched")
        self.dispatched = True
        for index in range(min(self.params["concurrency"], len(self.children))):
            done = threading.Event()
            thread = threading.Thread(
                target=self._worker,
                args=(deadline, done, index),
                name="cancel-schedule",
                daemon=True,
            )
            self.done.append(done)
            self.threads.append(thread)
            thread.start()

    def open_streams(self, deadline):
        self.join_submission(deadline)
        self.auto_consume = True
        for child in self.children:
            for entry in child.entries:
                child._start_consumer(
                    entry, min(deadline.expires_at, self.ctx.instance_deadline_s)
                )

    def entries(self):
        return [entry for child in self.children for entry in child.entries]

    def prove_ended(self, deadline):
        for child in self.children:
            for entry in child.entries:
                if entry["record"]["stream"]["ended_s"] is not None:
                    child._await_consumer(entry, deadline)
                    self._completed_child(child, deadline)
                    state = entry["record"]["stream"]["status"]
                    if state not in {"OK", *self.params["expected_stream_statuses"]}:
                        raise RuntimeError(
                            f"unexpected stream transport source: {state}"
                        )


def validate_prepare(params, plan):
    allowed = {
        "count",
        "concurrency",
        "input_len",
        "output_len",
        "consume",
        "stream_timeout_s",
        "schedule_timeout_s",
        "priority",
        "block_keys",
        "unique_block_keys",
        "expected_stream_statuses",
        "expected_rpc_statuses",
    }
    p = status._validate(params, plan, allowed)
    for key, default, high in (
        ("count", 1, 16),
        ("concurrency", 1, 16),
        ("input_len", 2048, 1048576),
        ("output_len", 10, 10000),
    ):
        p[key] = status._number(p.get(key, default), plan, key, 1, high, True)
    p.setdefault("consume", "immediate")
    if p["consume"] not in {"immediate", "manual"}:
        raise ValueError(f"{plan.path}: consume must be immediate/manual")
    p["stream_timeout_s"] = status._number(
        p.get("stream_timeout_s", 60), plan, "stream_timeout_s", 0.05, 60
    )
    p["schedule_timeout_s"] = status._number(
        p.get("schedule_timeout_s", 30), plan, "schedule_timeout_s", 0.05, 60
    )
    if "priority" in p:
        status._number(p["priority"], plan, "priority", -(2**31), 2**31 - 1, True)
    if "block_keys" in p:
        if (
            not isinstance(p["block_keys"], list)
            or not p["block_keys"]
            or len(p["block_keys"]) > 4096
        ):
            raise ValueError(f"{plan.path}: invalid block keys")
        for value in p["block_keys"]:
            status._number(value, plan, "block_key", -(2**63), 2**63 - 1, True)
    if "unique_block_keys" in p:
        if type(p["unique_block_keys"]) is not bool or "block_keys" in p:
            raise ValueError(
                f"{plan.path}: unique_block_keys must be boolean and exclusive"
            )
    allowed = p.setdefault("expected_stream_statuses", ["CANCELLED"])
    if not isinstance(allowed, list) or any(
        value
        not in {"CANCELLED", "DEADLINE_EXCEEDED", "UNKNOWN", "INTERNAL", "UNAVAILABLE"}
        for value in allowed
    ):
        raise ValueError(f"{plan.path}: invalid expected stream statuses")
    rpc_statuses = p.setdefault("expected_rpc_statuses", [])
    if not isinstance(rpc_statuses, list) or any(
        v not in {"DEADLINE_EXCEEDED", "UNAVAILABLE", "INTERNAL", "UNKNOWN"}
        for v in rpc_statuses
    ):
        raise ValueError(f"{plan.path}: invalid explicit Schedule status allowlist")
    return p


def execute_prepare(ctx, params, deadline):
    deadline.check()
    cohort = CancelRequests(ctx, params)
    handle = ctx.register_resource(
        "requests", cohort, cleanup=cohort.cleanup, historical=True
    )
    return StageOutput({"requests": handle, "count": params["count"]})


def validate_request_ref(params, plan):
    p = status._validate(params, plan, {"requests"}, {"requests"})
    plan.reference(p["requests"], "requests")
    return p


def _cohort(ctx, reference):
    cohort = ctx.resource(reference, "requests")
    if not isinstance(cohort, CancelRequests):
        raise RuntimeError("cancel operation requires its explicitly prepared cohort")
    return cohort


def execute_dispatch(ctx, params, deadline):
    _cohort(ctx, params["requests"]).dispatch(deadline)
    return StageOutput({"requests": ctx.resolve(params["requests"])})


def execute_begin(ctx, params, deadline):
    _cohort(ctx, params["requests"]).begin(deadline)
    return StageOutput({"requests": ctx.resolve(params["requests"])})


def validate_transport(params, plan):
    p = status._validate(params, plan, {"requests", "phase"}, {"requests", "phase"})
    plan.reference(p["requests"], "requests")
    if p["phase"] not in {"schedule", "stream"}:
        raise ValueError(f"{plan.path}: transport phase must be explicit")
    return p


def execute_transport(ctx, params, deadline):
    cohort = _cohort(ctx, params["requests"])
    receipts = []
    for child in cohort.children:
        for entry in child.entries:
            deadline.check()
            record = entry["record"]
            if params["phase"] == "stream" and record["stream"]["started_s"] is None:
                raise RuntimeError("stream-break trigger has no opened stream")
            if (
                params["phase"] == "schedule"
                and record["stream"]["started_s"] is not None
            ):
                raise RuntimeError(
                    "refusing to cancel a stream through a Schedule-drop action"
                )
            started = ctx.clock()
            child.cancel("explicit_" + params["phase"] + "_transport_drop")
            receipts.append(
                {
                    "request_id": str(record["wire_request_id"]),
                    "owner": "client_transport",
                    "phase": params["phase"],
                    "started_s": started,
                    "ended_s": ctx.clock(),
                    "cancel": child.snapshot_records()[0]["cancel"],
                }
            )
    if not receipts:
        raise RuntimeError("transport-drop trigger has no issued request")
    return status._frozen(ctx, "cancel-transport", {"receipts": receipts})


def execute_open(ctx, params, deadline):
    _cohort(ctx, params["requests"]).open_streams(deadline)
    return StageOutput({"requests": ctx.resolve(params["requests"])})


def validate_rpc(params, plan):
    p = status._validate(
        params,
        plan,
        {"requests", "destination", "expected_rpc_statuses"},
        {"requests", "destination"},
    )
    plan.reference(p["requests"], "requests")
    if p["destination"] not in {"master", "prefill", "decode"}:
        raise ValueError(f"{plan.path}: cancellation owner must be explicit")
    p.setdefault("expected_rpc_statuses", [])
    if p["expected_rpc_statuses"] not in ([], ["NOT_FOUND"]):
        raise ValueError(
            f"{plan.path}: only explicit NOT_FOUND is a supported RPC alternative"
        )
    return p


def _ack_fields(ack):
    fields = {}
    for key in ("found", "status", "lifecycle", "error_code"):
        if hasattr(ack, key):
            value = getattr(ack, key)
            fields[key] = (
                value if type(value) in (bool, int, str, float) else str(value)
            )
    return fields


def execute_rpc(ctx, params, deadline):
    cohort = _cohort(ctx, params["requests"])
    records = cohort.snapshot_records()
    entries = {e["record"]["wire_request_id"]: e for e in cohort.entries()}
    receipts = []
    try:
        for record in records:
            deadline.check()
            rid, owner = record["wire_request_id"], params["destination"]
            response = entries.get(rid, {}).get("response")
            if owner == "master":
                target = ctx.ops.master_target()
                stub = ctx.ops.schedule_pb2_grpc.FlexlbServiceStub(
                    ctx.ops._channel(target)
                )
                request = ctx.ops.schedule_pb2.FlexlbCancelRequestPB(
                    request_id=rid,
                    reason=ctx.ops.schedule_pb2.CANCEL_REASON_CLIENT_CANCELLED,
                )
                if (
                    response is not None
                    and response.HasField("lifecycle")
                    and response.lifecycle.batch_id
                ):
                    request.batch_id = response.lifecycle.batch_id
            else:
                if response is None:
                    raise RuntimeError(
                        "engine cancellation requires the actual scheduled route"
                    )
                target = ctx.ops.role_addr(response, owner.upper())
                if not target:
                    raise RuntimeError("requested cancellation owner has no route")
                stub = ctx.ops.pb2_grpc.RpcServiceStub(ctx.ops._channel(target))
                request = ctx.ops.pb2.CancelRequestPB(request_id=rid)
            receipt = dict(
                request_id=str(rid),
                owner=owner,
                target=target,
                started_s=ctx.clock(),
                rpc_status=None,
                response=None,
            )
            receipts.append(receipt)
            try:
                ack = stub.Cancel(request, timeout=min(10, deadline.remaining()))
                receipt.update(rpc_status="OK", response=_ack_fields(ack))
                if owner != "master":
                    receipt["response"]["typed_not_found"] = (
                        ack.status == ctx.ops.pb2.CANCEL_STATUS_NOT_FOUND
                    )
            except Exception as error:
                code_fn = getattr(error, "code", None)
                rpc_status = (
                    getattr(code_fn(), "name", None) if callable(code_fn) else None
                )
                receipt.update(rpc_status=rpc_status or "ERROR", error=repr(error))
                if rpc_status not in params["expected_rpc_statuses"]:
                    raise
            finally:
                receipt["ended_s"] = ctx.clock()
    except Exception as error:
        status._artifact(
            ctx,
            "cancel-rpc-incomplete",
            {"env_epoch": ctx.env_epoch, "receipts": receipts, "error": repr(error)},
        )
        raise
    return status._frozen(ctx, "cancel-rpc", {"receipts": receipts})


COHORT_METRICS = {
    "issued",
    "schedule_ok",
    "schedule_cancelled",
    "decode_running",
    "typed_preempted",
    "engine_not_found",
    "fence_rejected_8429",
    "probe_accepted",
    "first_output",
    "stream_ended",
    "business_finished",
    "success",
    "engine_cancelled",
    "cancel_rpc_count",
    "rpc_ok",
    "rpc_not_found",
    "engine_clean_total",
}
METRICS = COHORT_METRICS | status.METRICS


def metric(frame, name):
    if name in {"fence_rejected_8429", "probe_accepted"}:
        rows = frame["receipts"]
        if (
            len(rows) != 1
            or rows[0]["rpc_status"] != "OK"
            or rows[0].get("ended_s") is None
        ):
            raise RuntimeError("direct Enqueue acknowledgement is incomplete")
        row = rows[0]
        if name == "probe_accepted":
            return status._count(row["successes"])
        return int(
            row["successes"] == 0
            and len(row["errors"]) == 1
            and row["errors"][0]["request_id"] == row["request_id"]
            and row["errors"][0]["error_code"] == 8429
        )
    if name == "engine_clean_total":
        return status.metric(frame, "engine_inflight") + status.metric(
            frame, "engine_leaks"
        )
    if name == "cancel_rpc_count":
        return sum(
            status._count(e["rpc_counts"]["cancel"]) for e in frame["mock"].values()
        )
    if name in {"rpc_ok", "rpc_not_found", "engine_not_found"}:
        rows = frame["receipts"]
        if not rows or any(
            r.get("ended_s") is None or r.get("rpc_status") is None for r in rows
        ):
            raise RuntimeError("cancellation RPC receipt is incomplete")
        if name == "rpc_ok":
            return sum(r["rpc_status"] == "OK" for r in rows)
        if name == "engine_not_found":
            return sum(
                r["rpc_status"] == "OK" and r["response"]["typed_not_found"] is True
                for r in rows
            )
        return sum(
            r["rpc_status"] == "NOT_FOUND"
            or (r["rpc_status"] == "OK" and r["response"].get("found") is False)
            for r in rows
        )
    if name not in COHORT_METRICS:
        return status.metric(frame, name)
    records = frame["records"]
    if not records:
        raise RuntimeError("empty cancellation cohort evidence")
    if name == "decode_running":
        engines = frame["mock"]
        decode = [e for e in engines.values() if e["role"] == "decode"]
        if not decode or any(
            not isinstance(e["request_lifecycle"], dict) for e in decode
        ):
            raise RuntimeError("missing decode lifecycle evidence")
        return sum(
            any(
                e["request_lifecycle"]
                .get(str(r["wire_request_id"]), {})
                .get("end_state")
                == "running"
                for e in decode
            )
            for r in records
        )
    if name == "engine_cancelled":
        engines = frame["mock"]
        if not engines or any(
            not isinstance(e.get("cancelled_rids"), list)
            or not isinstance(e.get("request_lifecycle"), dict)
            for e in engines.values()
        ):
            raise RuntimeError("missing engine cancellation evidence")
        cancelled = {str(rid) for e in engines.values() for rid in e["cancelled_rids"]}
        return sum(
            str(r["wire_request_id"]) in cancelled
            or any(
                e["request_lifecycle"]
                .get(str(r["wire_request_id"]), {})
                .get("end_state")
                == "cancelled"
                for e in engines.values()
            )
            for r in records
        )
    predicates = {
        "issued": lambda r: r["issued_s"] is not None,
        "schedule_ok": lambda r: r["schedule"]["status"] == "OK",
        "schedule_cancelled": lambda r: r["schedule"]["status"] == "FUTURE_CANCELLED"
        and r["cancel"]["scope"] == "client_transport"
        and r["cancel"]["acknowledged"] is True
        and r["transport_terminal_s"] is not None
        and r["consumer_exit_s"] is not None,
        "first_output": lambda r: r["stream"]["first_output_s"] is not None,
        "stream_ended": lambda r: r["stream"]["ended_s"] is not None
        and r.get("consumer_done") is True
        and r.get("consumer_completion_verified") is True
        and r["consumer_exit_s"] is not None
        and r["transport_terminal_s"] is not None,
        "typed_preempted": lambda r: r["business_error_code"] in {2, 8429},
        "business_finished": lambda r: r["business_finished"] is True,
        "success": request_success,
    }
    return sum(predicates[name](r) for r in records)


def validate_observe(params, plan):
    p = status._validate(
        params,
        plan,
        {
            "requests",
            "duration_s",
            "interval_s",
            "until",
            "since",
            "baseline",
            "include",
        },
    )
    if "requests" in p:
        plan.reference(p["requests"], "requests")
    for key in ("since", "baseline"):
        if key in p:
            plan.reference(p[key], "snapshot")
    p.setdefault("include", ["inflight", "mock"])
    if (
        not isinstance(p["include"], list)
        or not p["include"]
        or any(
            value not in {"inflight", "mock", "info", "client_records"}
            for value in p["include"]
        )
    ):
        raise ValueError(f"{plan.path}: invalid cancellation observation sources")
    if "client_records" in p["include"] and "requests" not in p:
        raise ValueError(f"{plan.path}: client records require a cohort")
    p["duration_s"] = status._number(p.get("duration_s", 0), plan, "duration_s", 0, 180)
    p["interval_s"] = status._number(
        p.get("interval_s", 0.05), plan, "interval_s", 0.01, 5
    )
    if "until" in p:
        until = status._validate(
            p["until"], plan, {"metric", "op", "value"}, {"metric", "op", "value"}
        )
        if until["metric"] not in METRICS - {
            "fingerprint",
            "rpc_ok",
            "rpc_not_found",
        } or until["op"] not in {"eq", "ge", "le"}:
            raise ValueError(f"{plan.path}: invalid cancellation stop predicate")
        status._number(until["value"], plan, "until.value", 0, 2**63 - 1)
    return p


def execute_observe(ctx, params, deadline):
    start, epoch = ctx.clock(), ctx.env_epoch
    anchor = start
    if "since" in params:
        source = ctx.resource(params["since"], "snapshot").to_dict()
        if source["env_epoch"] != epoch or not source.get("receipts"):
            raise RuntimeError("cancellation timing anchor is missing or stale")
        anchor = min(r["started_s"] for r in source["receipts"])
    baseline = None
    if "baseline" in params:
        baseline = ctx.resource(params["baseline"], "snapshot").to_dict()
        if baseline["env_epoch"] != epoch or not baseline.get("frames"):
            raise RuntimeError("missing or stale observation baseline")
    end = anchor + params["duration_s"]
    frames, size = [], 0
    try:
        while True:
            deadline.check()
            if ctx.env_epoch != epoch:
                raise RuntimeError("cancellation observation epoch changed")
            cohort = _cohort(ctx, params["requests"]) if "requests" in params else None
            if cohort:
                if cohort.done and all(event.is_set() for event in cohort.done):
                    cohort.join_submission(deadline)
                    if cohort.errors:
                        raise cohort.errors[0]
                cohort.prove_ended(deadline)
            frame = status._frame(
                ctx,
                {"include": [s for s in params["include"] if s != "client_records"]},
                deadline,
            )
            if cohort:
                frame["records"] = cohort.snapshot_records()
            frame["capture_finished_s"] = ctx.clock()
            size += len(str(frame).encode())
            if len(frames) >= 4000 or size > 32 * 1024 * 1024:
                raise RuntimeError("cancellation observation budget exceeded")
            frames.append(frame)
            until = params.get("until")
            actual = metric(frame, until["metric"]) if until else None
            if until and baseline:
                actual -= metric(baseline["frames"][-1], until["metric"])
            if until and status._compare(actual, until["op"], until["value"]):
                break
            if ctx.clock() >= end:
                break
            deadline.sleep(min(params["interval_s"], end - ctx.clock()))
    except Exception as error:
        status._artifact(
            ctx,
            "cancel-observation-incomplete",
            {"frames": frames, "error": repr(error), "env_epoch": epoch},
        )
        raise
    return status._frozen(
        ctx,
        "cancel-observation",
        {
            "frames": frames,
            "started_s": start,
            "anchor_s": anchor,
            "ended_s": ctx.clock(),
        },
    )


def validate_check(params, plan):
    p = status._validate(
        params,
        plan,
        {"snapshot", "baseline", "metric", "op", "expected", "within_s"},
        {"snapshot", "metric", "op", "expected"},
    )
    for key in ("snapshot", "baseline"):
        if key in p:
            plan.reference(p[key], "snapshot")
    if p["metric"] not in METRICS or p["op"] not in {"eq", "ge", "le"}:
        raise ValueError(f"{plan.path}: invalid cancellation check")
    if p["metric"] == "fingerprint":
        if "baseline" not in p or p["op"] != "eq" or type(p["expected"]) is not bool:
            raise ValueError(
                f"{plan.path}: fingerprint requires boolean equality against a baseline"
            )
    else:
        status._number(p["expected"], plan, "expected", -(2**63), 2**63 - 1)
    if "within_s" in p:
        status._number(p["within_s"], plan, "within_s", 0.01, 180)
    return p


def execute_check(ctx, params, deadline):
    deadline.check()
    source = ctx.resource(params["snapshot"], "snapshot").to_dict()
    if source["env_epoch"] != ctx.env_epoch:
        raise RuntimeError("stale cancellation evidence")
    frames = source.get("frames") or (
        [{"receipts": source["receipts"]}] if "receipts" in source else []
    )
    if not frames:
        raise RuntimeError("missing cancellation observation frames")
    actual = metric(frames[-1], params["metric"])
    if "baseline" in params:
        baseline = ctx.resource(params["baseline"], "snapshot").to_dict()
        if baseline["env_epoch"] != source["env_epoch"] or not baseline.get("frames"):
            raise RuntimeError("invalid cancellation baseline")
        before = metric(baseline["frames"][-1], params["metric"])
        actual = (
            actual == before if params["metric"] == "fingerprint" else actual - before
        )
    passed = status._compare(actual, params["op"], params["expected"])
    if "within_s" in params:
        elapsed = frames[-1]["capture_finished_s"] - source["anchor_s"]
        passed = passed and elapsed <= params["within_s"]
    return StageOutput(
        {"passed": bool(passed)},
        [
            CheckResult(
                "contract",
                "PASS" if passed else "FAIL",
                "cancellation protocol boundary",
                actual=actual,
                expected=params["expected"],
                evidence={
                    "metric": params["metric"],
                    "within_s": params.get("within_s"),
                },
            )
        ],
    )


HANDLERS = [
    StageHandler(
        "cancel_prepare",
        validate_prepare,
        execute_prepare,
        {"requests": "requests", "count": "integer"},
    ),
    StageHandler(
        "cancel_dispatch",
        validate_request_ref,
        execute_dispatch,
        {"requests": "requests"},
    ),
    StageHandler(
        "cancel_open", validate_request_ref, execute_open, {"requests": "requests"}
    ),
    StageHandler("cancel_rpc", validate_rpc, execute_rpc, {"snapshot": "snapshot"}),
    StageHandler(
        "cancel_observe", validate_observe, execute_observe, {"snapshot": "snapshot"}
    ),
    StageHandler(
        "cancel_check",
        validate_check,
        execute_check,
        {"passed": "boolean"},
        checks=frozenset({"contract"}),
    ),
]

HANDLERS += [
    StageHandler(
        "cancel_begin", validate_request_ref, execute_begin, {"requests": "requests"}
    ),
    StageHandler(
        "cancel_transport",
        validate_transport,
        execute_transport,
        {"snapshot": "snapshot"},
    ),
]


def validate_group(params, plan):
    p = status._validate(params, plan, {"cohorts"}, {"cohorts"})
    if not isinstance(p["cohorts"], list) or not 1 <= len(p["cohorts"]) <= 16:
        raise ValueError(f"{plan.path}: 1..16 prepared cohorts are required")
    for reference in p["cohorts"]:
        plan.reference(reference, "requests")
    return p


def execute_group(ctx, params, deadline):
    cohorts = [_cohort(ctx, reference) for reference in params["cohorts"]]
    if (
        len({id(c) for c in cohorts}) != len(cohorts)
        or sum(c.params["concurrency"] for c in cohorts) > 16
    ):
        raise RuntimeError("duplicate cohorts or aggregate concurrency exceeds 16")
    for cohort in cohorts:
        cohort.begin(deadline)
    for cohort in cohorts:
        cohort.join_submission(deadline)
        if cohort.errors:
            raise cohort.errors[0]
    return StageOutput({"started": sum(len(c.children) for c in cohorts)})


HANDLERS.append(
    StageHandler(
        "cancel_dispatch_group", validate_group, execute_group, {"started": "integer"}
    )
)


def validate_probe(params, plan):
    p = status._validate(
        params,
        plan,
        {"requests", "output_len", "attempt", "wait_port_s", "scope"},
        {"requests", "output_len", "attempt"},
    )
    plan.reference(p["requests"], "requests")
    p["scope"] = p.get("scope", "pd")
    if p["scope"] not in {"pd", "prefill_only"}:
        raise ValueError(f"{plan.path}: scope must be pd or prefill_only")
    status._number(p["output_len"], plan, "output_len", 1, 10000, True)
    status._number(p["attempt"], plan, "attempt", 1, 2, True)
    p["wait_port_s"] = status._number(
        p.get("wait_port_s", 0), plan, "wait_port_s", 0, 10
    )
    return p


def execute_probe(ctx, params, deadline):
    import socket

    cohort = _cohort(ctx, params["requests"])
    entries = cohort.entries()
    if len(entries) != 1 or entries[0]["response"] is None:
        raise RuntimeError("direct Enqueue probe requires one actual scheduled route")
    entry = entries[0]
    rid, response = entry["record"]["wire_request_id"], entry["response"]
    target = ctx.ops.prefill_addr(response)
    if not target:
        raise RuntimeError("direct Enqueue probe has no original Prefill route")
    if params["wait_port_s"]:
        host, _, port = target.rpartition(":")
        end = min(deadline.expires_at, ctx.clock() + params["wait_port_s"])
        while True:
            deadline.check()
            try:
                with socket.create_connection(
                    (host, int(port)), timeout=min(1, max(0.01, end - ctx.clock()))
                ):
                    break
            except OSError:
                if ctx.clock() >= end:
                    raise TimeoutError("original Prefill port did not become ready")
                deadline.sleep(min(0.1, end - ctx.clock()))
    inp = ctx.ops.build_generate_input(rid, output_len=params["output_len"])
    # A P-only fence probe must not reuse a surviving D generation's RID.
    # PD remains the default for probes exercising the complete admission path.
    if params.get("scope", "pd") == "pd":
        ctx.ops._copy_role_addrs(inp, response)
    request = ctx.ops.pb2.EnqueueBatchRequestPB(
        batch_id=rid * 10 + params["attempt"],
        dp_slots=[
            ctx.ops.pb2.EnqueueBatchDpSlotPB(
                dp_rank=0, requests=[ctx.ops.pb2.EnqueueBatchExternalInputPB(input=inp)]
            )
        ],
        fetch_attach_timeout_ms=30000,
    )
    receipt = {
        "request_id": str(rid),
        "target": target,
        "owner": "original_prefill",
        "scope": params.get("scope", "pd"),
        "started_s": ctx.clock(),
        "rpc_status": None,
    }
    try:
        stub = ctx.ops.pb2_grpc.RpcServiceStub(ctx.ops._channel(target))
        # One RPC only. Retrying the same request ID could admit an orphan twice.
        ack = stub.EnqueueBatch(request, timeout=min(10, deadline.remaining()))
        receipt.update(
            rpc_status="OK",
            successes=len(ack.successes),
            errors=[
                {
                    "request_id": str(e.request_id),
                    "error_code": int(e.error_info.error_code),
                    "error_message": getattr(e.error_info, "error_message", ""),
                }
                for e in ack.errors
            ],
        )
    except Exception as error:
        receipt["error"] = repr(error)
        status._artifact(ctx, "direct-enqueue-incomplete", receipt)
        raise
    finally:
        receipt["ended_s"] = ctx.clock()
    return status._frozen(ctx, "direct-enqueue", {"receipts": [receipt]})


HANDLERS.append(
    StageHandler(
        "cancel_enqueue_probe", validate_probe, execute_probe, {"snapshot": "snapshot"}
    )
)
