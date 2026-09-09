"""Explicit KV pressure, owner counters and request evidence; no legacy dispatch."""

import copy
import json
import math

from ..backend import RequestBatch
from ..contracts import CheckResult, StageHandler, StageOutput
from .elastic import request_success
from .engine_control import ENGINE_NAME, _engines, _http
from .master import _master_json


def _fields(params, allowed, required=()):
    if (
        not isinstance(params, dict)
        or set(params) - set(allowed)
        or set(required) - set(params)
    ):
        raise ValueError("invalid KV capacity parameters")
    return copy.deepcopy(params)


def _artifact(ctx, label, data):
    path = (
        ctx.artifact_dir
        / f"kv-capacity-{label}-{len(ctx.outputs)}-{len(ctx._resources)}.json"
    )
    path.write_text(json.dumps(data, indent=2) + "\n")
    return str(path)


def _target(value, plan):
    if isinstance(value, dict):
        plan.reference(value, "string")
    elif not isinstance(value, str) or not ENGINE_NAME.fullmatch(value):
        raise ValueError("explicit engine name required")


def _request_validate(params, plan):
    p = _fields(
        params,
        {
            "input_len",
            "output_len",
            "block_keys",
            "mode",
            "schedule_timeout_s",
            "stream_timeout_s",
            "expected_rpc_statuses",
        },
    )
    for k, default in (("input_len", 2048), ("output_len", 2)):
        p.setdefault(k, default)
        if type(p[k]) is not int or not 1 <= p[k] <= 1048576:
            raise ValueError("request shape outside bounds")
    for k, default in (("schedule_timeout_s", 30), ("stream_timeout_s", 15)):
        p.setdefault(k, default)
        if (
            type(p[k]) not in (int, float)
            or not math.isfinite(p[k])
            or not 0 < p[k] <= 60
        ):
            raise ValueError("request timeout outside bounds")
    if "block_keys" in p:
        keys = p["block_keys"]
        if (
            not isinstance(keys, list)
            or not 1 <= len(keys) <= 4096
            or any(type(k) is not int or not 0 <= k < 2**63 for k in keys)
        ):
            raise ValueError("invalid explicit block keys")
    p.setdefault("mode", "complete")
    p.setdefault("expected_rpc_statuses", [])
    if p["mode"] not in ("complete", "fire"):
        raise ValueError("invalid request consumption mode")
    if not isinstance(p["expected_rpc_statuses"], list) or any(
        s not in ("UNKNOWN", "INTERNAL", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED")
        for s in p["expected_rpc_statuses"]
    ):
        raise ValueError("invalid RPC status policy")
    return p


def _accept_request_error(batch, error, deadline):
    deadline.check()
    rows = batch.snapshot_records()
    if len(rows) != 1:
        raise error
    row = rows[0]
    statuses = batch._capacity_expected_rpc_statuses
    if not any(row[p]["status"] in statuses for p in ("schedule", "stream")):
        raise error
    for phase in ("schedule", "stream"):
        rpc = row[phase]
        if phase == "stream" and rpc["started_s"] is None:
            continue
        if rpc["ended_s"] is None or rpc["status"] not in ("OK", "REJECTED", *statuses):
            raise error
    if row["transport_terminal_s"] is None or row["consumer_exit_s"] is None:
        raise error
    if row["stream"]["started_s"] is not None and (
        not row.get("consumer_done") or not row.get("consumer_completion_verified")
    ):
        raise error


def request(ctx, p, deadline):
    params = {
        k: p[k]
        for k in (
            "input_len",
            "output_len",
            "block_keys",
            "schedule_timeout_s",
            "stream_timeout_s",
        )
        if k in p
    }
    batch_dispatch = ctx.instance.get("effective_axes", {}).get("dispatcher")
    if batch_dispatch not in ("BATCH", "NON_BATCH"):
        raise ValueError("resolved dispatcher axis unavailable")
    consume = (
        "deferred" if p["mode"] == "fire" and batch_dispatch == "BATCH" else "immediate"
    )
    batch = RequestBatch(ctx, dict(params, count=1, consume=consume))
    batch._capacity_expected_rpc_statuses = p["expected_rpc_statuses"]
    handle = ctx.register_resource("requests", batch, batch.cleanup)
    try:
        batch.submit(deadline)
        if p["mode"] == "complete":
            batch.wait(deadline)
    except Exception as exc:
        _accept_request_error(batch, exc, deadline)
    return StageOutput(
        {"requests": handle},
        artifacts=[_artifact(ctx, "request", batch.snapshot_records())],
    )


def _wait_validate(params, plan):
    p = _fields(params, {"requests"}, {"requests"})
    if not isinstance(p["requests"], list) or not 1 <= len(p["requests"]) <= 32:
        raise ValueError("explicit bounded cohort list required")
    for ref in p["requests"]:
        plan.reference(ref, "requests")
    return p


def wait_requests(ctx, p, deadline):
    rows = []
    for ref in p["requests"]:
        batch = ctx.resource(ref, "requests")
        try:
            batch.wait(deadline)
        except Exception as exc:
            _accept_request_error(batch, exc, deadline)
        rows.extend(batch.snapshot_records())
    return StageOutput({"count": len(rows)}, artifacts=[_artifact(ctx, "drain", rows)])


def _outcome_validate(params, plan):
    p = _fields(
        params, {"requests", "metric", "tokens", "max_s"}, {"requests", "metric"}
    )
    _wait_validate({"requests": p["requests"]}, plan)
    if p["metric"] not in (
        "success",
        "admitted",
        "typed_error",
        "bounded_errors",
        "schedule_deadline",
    ):
        raise ValueError("unknown request criterion")
    if p["metric"] in ("typed_error", "bounded_errors"):
        if (
            not isinstance(p.get("tokens"), list)
            or not p["tokens"]
            or any(not isinstance(s, str) or not s for s in p["tokens"])
        ):
            raise ValueError("explicit error tokens required")
        if (
            type(p.get("max_s")) not in (int, float)
            or not math.isfinite(p["max_s"])
            or p["max_s"] <= 0
        ):
            raise ValueError("finite failure bound required")
    return p


def _error_text(row):
    return str(
        row["schedule"]["error"]
        or row["stream"]["error"]
        or ("" if request_success(row) else "stream did not complete")
    )


def outcome(ctx, p, deadline):
    deadline.check()
    rows = [
        r
        for ref in p["requests"]
        for r in ctx.resource(ref, "requests").snapshot_records()
    ]
    if not rows:
        raise ValueError("no request evidence")
    metric = p["metric"]
    details = []
    for row in rows:
        if metric == "success":
            good = request_success(row)
        elif metric == "admitted":
            good = row["schedule"]["status"] == "OK"
        elif metric == "schedule_deadline":
            good = (
                row["schedule"]["status"] == "DEADLINE_EXCEEDED"
                and row["stream"]["started_s"] is None
            )
        else:
            text = _error_text(row).lower()
            start, end = row["schedule"]["started_s"], row["consumer_exit_s"]
            if (
                any(
                    type(v) not in (int, float) or not math.isfinite(v)
                    for v in (start, end)
                )
                or end < start
            ):
                raise ValueError("missing request elapsed time")
            tokens = list(p["tokens"])
            if (
                metric == "typed_error"
                and ctx.instance["effective_axes"]["dispatcher"] == "BATCH"
            ):
                tokens.append("enqueuebatch rejected")
            good = (
                not request_success(row)
                and end - start < p["max_s"]
                and all(t.lower() in text for t in tokens)
            )
            if metric == "bounded_errors" and request_success(row):
                good = True
        details.append(good)
    return StageOutput(
        {"passed": all(details)},
        [CheckResult("criterion", "PASS" if all(details) else "FAIL", actual=details)],
        [_artifact(ctx, "outcomes", rows)],
    )


COUNTERS = {
    "cache_blocks",
    "held_blocks",
    "referenced_blocks",
    "available_blocks",
    "cache_evictions",
    "lack_mem_rejects",
    "kv_admission_fails",
    "available_kv_tokens",
    "active_kv_tokens",
    "running",
    "waiting",
    "pending",
}


def _observe_validate(params, plan):
    p = _fields(
        params,
        {
            "targets",
            "fields",
            "duration_s",
            "interval_s",
            "until_field",
            "until_op",
            "until_value",
            "baseline",
        },
        {"targets", "fields"},
    )
    if not isinstance(p["targets"], list) or not 1 <= len(p["targets"]) <= 32:
        raise ValueError("explicit targets required")
    for target in p["targets"]:
        _target(target, plan)
    if (
        not isinstance(p["fields"], list)
        or not p["fields"]
        or any(
            f not in COUNTERS | {"cache_key_set", "request_lifecycle"}
            for f in p["fields"]
        )
    ):
        raise ValueError("unknown owner field")
    for k, d, lo, hi in [("duration_s", 0, 0, 60), ("interval_s", 0.5, 0.01, 2)]:
        p.setdefault(k, d)
        if (
            type(p[k]) not in (int, float)
            or not math.isfinite(p[k])
            or not lo <= p[k] <= hi
        ):
            raise ValueError("invalid observation bounds")
    if "baseline" in p:
        plan.reference(p["baseline"], "kv_capacity_observation")
    if "until_field" in p:
        if (
            p["until_field"] not in p["fields"]
            or p.get("until_op") not in ("eq", "ge", "le")
            or type(p.get("until_value")) is not int
        ):
            raise ValueError("invalid stopping predicate")
    return p


def _compare(a, op, b):
    return (
        a == b
        if op == "eq"
        else (
            a >= b
            if op == "ge"
            else a <= b if op == "le" else a > b if op == "gt" else a < b
        )
    )


def observe(ctx, p, deadline):
    names = [ctx.resolve(n) for n in p["targets"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate owner")
    samples = []
    until = ctx.clock() + p["duration_s"]
    base = (
        ctx.resource(p["baseline"], "kv_capacity_observation")[-1]["engines"]
        if "baseline" in p
        else None
    )
    try:
        while True:
            owners = _engines(_http(ctx.ops, "snapshot", deadline), names)
            data = {}
            for name, row in owners.items():
                data[name] = {}
                for field in p["fields"]:
                    if field == "pending" and any(
                        type(row.get(k)) is not int or row[k] < 0
                        for k in ("running", "waiting")
                    ):
                        raise ValueError("missing pending components")
                    value = (
                        row["running"] + row["waiting"]
                        if field == "pending"
                        else row.get(field)
                    )
                    if field in COUNTERS and (type(value) is not int or value < 0):
                        raise ValueError(
                            "missing/invalid owner counter " + name + "." + field
                        )
                    if field == "cache_key_set" and (
                        not isinstance(value, list)
                        or any(type(k) is not int for k in value)
                    ):
                        raise ValueError("missing key set")
                    if field == "request_lifecycle" and not isinstance(value, dict):
                        raise ValueError("missing lifecycle map")
                    data[name][field] = value
            samples.append(dict(time_s=ctx.clock(), engines=data))
            if "until_field" in p and all(
                _compare(
                    r[p["until_field"]],
                    p["until_op"],
                    p["until_value"] + (base[n][p["until_field"]] if base else 0),
                )
                for n, r in data.items()
            ):
                break
            if ctx.clock() >= until:
                break
            deadline.sleep(min(p["interval_s"], until - ctx.clock()))
    finally:
        path = _artifact(ctx, "observe", samples)
    return StageOutput(
        {
            "observation": ctx.register_resource(
                "kv_capacity_observation", samples, historical=True
            )
        },
        artifacts=[path],
    )


HANDLERS = [
    StageHandler(
        "kv_capacity_request", _request_validate, request, {"requests": "requests"}
    ),
    StageHandler(
        "kv_capacity_wait", _wait_validate, wait_requests, {"count": "integer"}
    ),
    StageHandler(
        "kv_capacity_outcome",
        _outcome_validate,
        outcome,
        {"passed": "boolean"},
        checks=frozenset({"criterion"}),
    ),
    StageHandler(
        "kv_capacity_observe",
        _observe_validate,
        observe,
        {"observation": "kv_capacity_observation"},
    ),
]


def _counter_validate(params, plan):
    p = _fields(
        params,
        {"observations", "field", "op", "expected", "stat", "baseline"},
        {"observations", "field", "op", "expected", "stat"},
    )
    if not isinstance(p["observations"], list) or not p["observations"]:
        raise ValueError("counter needs samples")
    for ref in p["observations"]:
        plan.reference(ref, "kv_capacity_observation")
    if "baseline" in p:
        plan.reference(p["baseline"], "kv_capacity_observation")
    if (
        p["field"] not in COUNTERS | {"key_count", "conservation"}
        or p["op"] not in ("eq", "ge", "le", "gt", "lt")
        or p["stat"] not in ("all", "max", "min", "latest")
    ):
        raise ValueError("invalid counter predicate")
    if type(p["expected"]) is not int:
        raise ValueError("integer counter expectation required")
    return p


def counter(ctx, p, deadline):
    deadline.check()
    samples = [
        s
        for ref in p["observations"]
        for s in ctx.resource(ref, "kv_capacity_observation")
    ]
    if not samples:
        raise ValueError("empty counter series")
    base = (
        ctx.resource(p["baseline"], "kv_capacity_observation")[-1]["engines"]
        if "baseline" in p
        else None
    )
    values = []
    for sample in samples[-1:] if p["stat"] == "latest" else samples:
        for name, row in sample["engines"].items():
            f = p["field"]
            value = (
                len(row["cache_key_set"])
                if f == "key_count"
                else (
                    row["held_blocks"]
                    + row["referenced_blocks"]
                    + row["available_blocks"]
                    - row["cache_blocks"]
                    if f == "conservation"
                    else row[f]
                )
            )
            if base is not None:
                value -= base[name][f]
            values.append(value)
    compared = (
        [max(values)]
        if p["stat"] == "max"
        else [min(values)] if p["stat"] == "min" else values
    )
    passed = all(_compare(v, p["op"], p["expected"]) for v in compared)
    return StageOutput(
        {"passed": passed},
        [
            CheckResult(
                "counter",
                "PASS" if passed else "FAIL",
                actual=compared,
                expected=p["expected"],
            )
        ],
    )


def _owner_cancel_validate(params, plan):
    p = _fields(params, {"requests"}, {"requests"})
    plan.reference(p["requests"], "requests")
    return p


def owner_cancel(ctx, p, deadline):
    batch = ctx.resource(p["requests"], "requests")
    batch.cancel_server(deadline)
    return StageOutput(
        {"issued": True},
        artifacts=[_artifact(ctx, "master-cancel", batch.snapshot_records())],
    )


def _watermark_validate(params, plan):
    return _fields(params, set())


def _decode_load(row):
    # Match the established cleanup compatibility contract (d955a8434b).
    # total_load is routing load, not reserved_total; never sum these owners.
    fields = [key for key in ("inflight_requests", "total_load") if key in row]
    if not fields or any(type(row[key]) is not int or row[key] < 0 for key in fields):
        raise ValueError("missing or malformed Decode load watermark")
    return row.get("inflight_requests") or row.get("total_load", 0)


def _watermark(ctx, deadline):
    data = _master_json(ctx, "single", "/rtp_llm/inflight_status", deadline)
    scheduler = data.get("scheduler_inflight")
    pre = data.get("prefill_endpoints")
    dec = data.get("decode_endpoints")
    if (
        type(scheduler) is not int
        or scheduler < 0
        or not isinstance(pre, list)
        or not isinstance(dec, list)
        or not pre
        or not dec
    ):
        raise ValueError("missing Master watermark owners")

    def total(rows, field):
        values = [r.get(field) for r in rows]
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("missing watermark " + field)
        return sum(values)

    return dict(
        scheduler=scheduler,
        prefill_batches=total(pre, "inflight_batches"),
        decode_load=sum(_decode_load(row) for row in dec),
        raw=data,
    )


def watermark(ctx, p, deadline):
    value = _watermark(ctx, deadline)
    return StageOutput(
        {
            "watermark": ctx.register_resource(
                "kv_capacity_watermark", value, historical=True
            )
        },
        artifacts=[_artifact(ctx, "watermark", value)],
    )


def _watermark_wait_validate(params, plan):
    p = _fields(params, {"baseline"}, {"baseline"})
    plan.reference(p["baseline"], "kv_capacity_watermark")
    return p


def watermark_wait(ctx, p, deadline):
    base = ctx.resource(p["baseline"], "kv_capacity_watermark")
    samples = []
    try:
        while True:
            current = _watermark(ctx, deadline)
            samples.append(current)
            if all(
                current[k] <= base[k]
                for k in ("scheduler", "prefill_batches", "decode_load")
            ):
                break
            deadline.sleep(0.5)
    finally:
        path = _artifact(ctx, "watermark-return", samples)
    return StageOutput(
        {"passed": True},
        [CheckResult("watermark", "PASS", actual=current, expected=base)],
        [path],
    )


def _not_delivered_validate(params, plan):
    p = _fields(params, {"requests", "observation"}, {"requests", "observation"})
    plan.reference(p["requests"], "requests")
    plan.reference(p["observation"], "kv_capacity_observation")
    return p


def not_delivered(ctx, p, deadline):
    deadline.check()
    rows = ctx.resource(p["requests"], "requests").snapshot_records()
    if len(rows) != 1:
        raise ValueError("not-delivered probe needs one request")
    rid = str(rows[0]["wire_request_id"])
    samples = ctx.resource(p["observation"], "kv_capacity_observation")
    delivered = [
        name
        for s in samples
        for name, e in s["engines"].items()
        if rid in e["request_lifecycle"]
    ]
    passed = bool(samples) and not delivered
    return StageOutput(
        {"passed": passed},
        [CheckResult("not_delivered", "PASS" if passed else "FAIL", actual=delivered)],
    )


def _pressure_validate(params, plan):
    p = _fields(params, {"targets"}, {"targets"})
    if not isinstance(p["targets"], list) or not p["targets"]:
        raise ValueError("decode targets required")
    for name in p["targets"]:
        _target(name, plan)
        if not isinstance(name, str) or not name.startswith("decode-"):
            raise ValueError("pressure must name decode owner")
    return p


def pressure(ctx, p, deadline):
    from .engine_fault import inject

    owners = _engines(_http(ctx.ops, "snapshot", deadline), p["targets"])
    handles = []
    for name, row in owners.items():
        values = [row.get(f) for f in ("available_kv_tokens", "active_kv_tokens")]
        if any(type(v) is not int or v < 0 for v in values) or sum(values) <= 0:
            raise ValueError("missing/invalid actual decode KV total")
        out = inject(
            ctx,
            dict(targets=[name], type="kv_pressure", options=dict(tokens=sum(values))),
            deadline,
        )
        handles.append(out.output["fault"])
    return StageOutput(
        {"pressure": ctx.register_resource("kv_capacity_pressure", handles)}
    )


def _clear_validate(params, plan):
    p = _fields(params, {"pressure"}, {"pressure"})
    plan.reference(p["pressure"], "kv_capacity_pressure")
    return p


def clear_pressure(ctx, p, deadline):
    from .engine_fault import clear

    for h in ctx.resource(p["pressure"], "kv_capacity_pressure"):
        clear(ctx, {"fault": h}, deadline)
    return StageOutput({"cleared": True})


HANDLERS += [
    StageHandler(
        "kv_capacity_counter",
        _counter_validate,
        counter,
        {"passed": "boolean"},
        checks=frozenset({"counter"}),
    ),
    StageHandler(
        "kv_capacity_cancel",
        _owner_cancel_validate,
        owner_cancel,
        {"issued": "boolean"},
    ),
    StageHandler(
        "kv_capacity_watermark",
        _watermark_validate,
        watermark,
        {"watermark": "kv_capacity_watermark"},
    ),
    StageHandler(
        "kv_capacity_watermark_wait",
        _watermark_wait_validate,
        watermark_wait,
        {"passed": "boolean"},
        checks=frozenset({"watermark"}),
    ),
    StageHandler(
        "kv_capacity_not_delivered",
        _not_delivered_validate,
        not_delivered,
        {"passed": "boolean"},
        checks=frozenset({"not_delivered"}),
    ),
    StageHandler(
        "kv_capacity_pressure",
        _pressure_validate,
        pressure,
        {"pressure": "kv_capacity_pressure"},
    ),
    StageHandler(
        "kv_capacity_clear", _clear_validate, clear_pressure, {"cleared": "boolean"}
    ),
]


def _other_validate(params, plan):
    p = _fields(params, {"targets", "exclude"}, {"targets", "exclude"})
    if not isinstance(p["targets"], list) or len(p["targets"]) != 2:
        raise ValueError("two prefill owners required")
    for n in p["targets"]:
        _target(n, plan)
    _target(p["exclude"], plan)
    return p


def other_engine(ctx, p, deadline):
    deadline.check()
    names = [ctx.resolve(n) for n in p["targets"]]
    excluded = ctx.resolve(p["exclude"])
    if len(set(names)) != 2 or excluded not in names:
        raise ValueError("holder outside declared pair")
    return StageOutput({"engine": next(n for n in names if n != excluded)})


def _protection_validate(params, plan):
    p = _fields(
        params,
        {"baseline", "probe", "holder", "wave", "share_bands", "ratio_bands"},
        {"baseline", "probe", "holder", "wave", "share_bands", "ratio_bands"},
    )
    for k in ("baseline", "probe"):
        plan.reference(p[k], "requests")
    _wait_validate({"requests": p["wave"]}, plan)
    _target(p["holder"], plan)
    for key in ("share_bands", "ratio_bands"):
        b = p[key]
        if (
            not isinstance(b, dict)
            or set(b) != {"strict", "normal", "loose"}
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in b.values()
            )
            or not b["strict"] <= b["normal"] <= b["loose"]
        ):
            raise ValueError("explicit ordered grade bands required")
    return p


def protection(ctx, p, deadline):
    from ...grade import GradeReport

    deadline.check()
    holder = ctx.resolve(p["holder"])
    engines = _engines(_http(ctx.ops, "snapshot", deadline), [holder])
    addr = engines[holder]["grpc_addr"]
    wave = [
        r for ref in p["wave"] for r in ctx.resource(ref, "requests").snapshot_records()
    ]
    if len(wave) != 5:
        raise ValueError("capacity conflict requires all five wave records")
    share = (
        sum(r["schedule"]["status"] == "OK" and r["prefill_addr"] == addr for r in wave)
        / 5
    )
    base = ctx.resource(p["baseline"], "requests").snapshot_records()
    probe = ctx.resource(p["probe"], "requests").snapshot_records()
    if len(base) != 1 or len(probe) != 1:
        raise ValueError("timing needs one baseline/probe")
    batch = ctx.instance["effective_axes"]["dispatcher"] == "BATCH"

    def latency(r):
        a = r["schedule"]["started_s"]
        b = r["transport_terminal_s"] if batch else r["stream"]["first_output_s"]
        if (
            not request_success(r)
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in (a, b))
            or b <= a
        ):
            return None
        return b - a

    a, b = latency(base[0]), latency(probe[0])
    ratio = b / a if a is not None and b is not None else None
    report = GradeReport(run_grade=ctx.instance["grade"])
    p5 = report.check("P5", share, bands=p["share_bands"])
    p7 = ratio is not None and report.check("P7", ratio, bands=p["ratio_bands"])
    complete = all(r["schedule"]["status"] == "OK" for r in wave) and request_success(
        probe[0]
    )
    checks = [
        CheckResult("P6", "PASS" if complete else "FAIL", actual=complete),
        CheckResult("overflow", "PASS" if share < 1 else "FAIL", actual=share),
        CheckResult(
            "P5",
            "PASS" if p5 else "FAIL",
            actual=share,
            expected=p["share_bands"][report.run_grade],
        ),
        CheckResult(
            "P7",
            "PASS" if p7 else "FAIL",
            actual=ratio,
            expected=p["ratio_bands"][report.run_grade],
        ),
    ]
    return StageOutput(
        {"passed": all(c.status == "PASS" for c in checks)},
        checks,
        [
            _artifact(
                ctx,
                "protection",
                dict(
                    wave=wave,
                    baseline=base,
                    probe=probe,
                    caliber="completion_duration" if batch else "client_ttft",
                    grade=report.run_grade,
                ),
            )
        ],
    )


def _failure_bound_validate(params, plan):
    p = _fields(
        params, {"requests", "minimum", "maximum"}, {"requests", "minimum", "maximum"}
    )
    _wait_validate({"requests": p["requests"]}, plan)
    if (
        any(type(p[k]) is not int or p[k] < 0 for k in ("minimum", "maximum"))
        or p["minimum"] > p["maximum"]
    ):
        raise ValueError("invalid failure-count bounds")
    return p


def failure_bound(ctx, p, deadline):
    deadline.check()
    rows = [
        r
        for ref in p["requests"]
        for r in ctx.resource(ref, "requests").snapshot_records()
    ]
    failures = sum(not request_success(r) for r in rows)
    passed = bool(rows) and p["minimum"] <= failures <= p["maximum"]
    return StageOutput(
        {"passed": passed},
        [
            CheckResult(
                "failure_count",
                "PASS" if passed else "FAIL",
                actual=failures,
                expected=[p["minimum"], p["maximum"]],
            )
        ],
    )


HANDLERS += [
    StageHandler(
        "kv_capacity_other", _other_validate, other_engine, {"engine": "string"}
    ),
    StageHandler(
        "kv_capacity_protection",
        _protection_validate,
        protection,
        {"passed": "boolean"},
        checks=frozenset({"P6", "overflow", "P5", "P7"}),
    ),
    StageHandler(
        "kv_capacity_failure_bound",
        _failure_bound_validate,
        failure_bound,
        {"passed": "boolean"},
        checks=frozenset({"failure_count"}),
    ),
]
