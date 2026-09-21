"""Status protocol primitives. YAML specifies the experiment and each assertion."""

from __future__ import annotations

import copy
import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ..contracts import CheckResult, StageHandler, StageOutput
from .observation import FrozenSnapshot

MAX_BYTES = 8 * 1024 * 1024
FAULT_FIELDS = {
    "enqueue_ack_partial_fail": {"k"},
    "enqueue_ack_error_code": {"code"},
    "enqueue_ack_drop": set(),
    "prefill_async_partial_fail": {"k", "code"},
    "status_suppress_rids": {"rids"},
    "status_suppress_running": set(),
    "status_suppress_finished": set(),
    "status_no_respond": set(),
    "status_version_regress": set(),
    "status_cursor_regress": {"n"},
    "status_duplicate_finished": {"n"},
    "status_zombie_running": set(),
    "status_fake_task": {"rid", "phase", "batch_id", "error_code"},
    "fetch_error": set(),
}
METRICS = {
    "master_http",
    "scheduler_tombstones",
    "cohort_prefill_completed",
    "cohort_fetch_invocations",
    "cohort_enqueued",
    "scheduler",
    "prefill_batches",
    "prefill_requests",
    "decode_total_load",
    "all_inflight",
    "cleanup_inflight",
    "fingerprint",
    "accepted",
    "prefill_accepted",
    "prefill_enqueue_rpc",
    "prefill_engine_inflight",
    "enqueue_rpc",
    "fetch_rpc",
    "engine_inflight",
    "engine_leaks",
    "alive_prefill",
    "alive_decode",
    "ttl_scheduler",
    "ttl_prefill",
    "ttl_decode",
}


def _validate(params, plan, allowed, required=()):
    if (
        not isinstance(params, dict)
        or set(params) - set(allowed)
        or set(required) - set(params)
    ):
        raise ValueError(f"{plan.path}: invalid status parameters")
    return copy.deepcopy(params)


def _number(value, plan, name, low, high, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{plan.path}.{name}: expected {low}..{high}")
    return value


def _http(ctx, server, path, deadline, body=None, allowed=(200,), text=False):
    deadline.check()
    port = getattr(
        ctx.env,
        {
            "mock": "mock_http_port",
            "master": "master_http_port",
            "metrics": "master_management_port",
        }[server],
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=min(5, deadline.remaining())
        ) as response:
            status, raw = response.status, response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise RuntimeError("status source exceeds byte budget")
    if status not in allowed:
        raise RuntimeError(f"{server}/{path} HTTP {status}: {raw[:200]!r}")
    return status, raw.decode() if text else json.loads(raw)


def _artifact(ctx, label, data):
    path = Path(ctx.artifact_dir) / f"status-{label}-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(data, indent=2, allow_nan=False))
    return str(path)


def _frozen(ctx, label, data):
    data = dict(schema_version=1, env_epoch=ctx.env_epoch, **data)
    artifact = _artifact(ctx, label, data)
    resource = FrozenSnapshot(json.dumps(data, allow_nan=False))
    handle = ctx.register_resource("snapshot", resource, historical=True)
    return StageOutput({"snapshot": handle}, artifacts=[artifact])


def _mock(ctx, deadline):
    _, data = _http(ctx, "mock", "snapshot", deadline)
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("engines"), list)
        or not data["engines"]
    ):
        raise RuntimeError("nonempty mock engine source required")
    engines = {e["name"]: e for e in data["engines"]}
    if len(engines) != len(data["engines"]):
        raise RuntimeError("duplicate mock engine names")
    return engines


def _targets(ctx, params, deadline):
    engines = _mock(ctx, deadline)
    names = sorted(
        name
        for name, e in engines.items()
        if params["role"] == "all" or e["role"].lower() == params["role"]
    )
    if params.get("selection") == "first":
        names = names[:1]
    if params.get("selection") == "landing":
        records = ctx.resource(params["requests"], "requests").snapshot_records()
        addresses = {r["prefill_addr"] for r in records if r.get("prefill_addr")}
        names = [name for name in names if engines[name]["grpc_addr"] in addresses]
    if not names:
        raise RuntimeError("status control has no real target engines")
    return names


def validate_control(params, plan):
    p = _validate(
        params,
        plan,
        {
            "role",
            "selection",
            "fault",
            "config",
            "enabled",
            "requests",
            "rids",
            "expected_http",
        },
        {"fault"},
    )
    p.setdefault("role", "prefill")
    p.setdefault("selection", "all")
    p.setdefault("enabled", True)
    if p["role"] not in {"prefill", "decode", "all"} or p["selection"] not in {
        "all",
        "first",
        "landing",
    }:
        raise ValueError(f"{plan.path}: invalid target selection")
    if p["fault"] not in FAULT_FIELDS or type(p["enabled"]) is not bool:
        raise ValueError(f"{plan.path}: unknown fault or nonboolean enabled")
    p["config"] = _validate(p.get("config", {}), plan, FAULT_FIELDS[p["fault"]])
    for key, value in p["config"].items():
        if key == "phase":
            if value not in {"RUNNING", "RECEIVED", "finished"}:
                raise ValueError(f"{plan.path}: invalid fake phase")
        elif key == "rids":
            if (
                not isinstance(value, list)
                or not value
                or len(value) > 64
                or any(type(v) is not int for v in value)
            ):
                raise ValueError(f"{plan.path}: invalid rid list")
        else:
            _number(value, plan, key, -(2**63), 2**63 - 1, True)
    if "requests" in p:
        plan.reference(p["requests"], "requests")
    if p["selection"] == "landing" and "requests" not in p:
        raise ValueError(f"{plan.path}: landing requires request evidence")
    if "rids" in p:
        if "requests" not in p or p["rids"] not in {"all", "first"}:
            raise ValueError(
                f"{plan.path}: rids requires requests and all/first selector"
            )
    p.setdefault("expected_http", [200])
    if p["expected_http"] not in ([200], [200, 400]):
        raise ValueError(
            f"{plan.path}: only explicit sentinel-channel 400 may be observed"
        )
    if p["expected_http"] == [200, 400] and not (
        p["fault"] == "status_fake_task" and p["config"].get("rid") == 0
    ):
        raise ValueError(
            f"{plan.path}: optional channel rejection is restricted to rid=0"
        )
    return p


def execute_control(ctx, params, deadline):
    epoch, env = ctx.env_epoch, ctx.env
    names = _targets(ctx, params, deadline)

    def clear(limit):
        if ctx.env_epoch != epoch or ctx.env is not env:
            raise RuntimeError(
                "refusing injection cleanup against a different environment epoch"
            )
        errors = []
        for name in names:
            try:
                _http(
                    ctx,
                    "mock",
                    "inject",
                    limit,
                    {"engine": name, "type": params["fault"], "enabled": False},
                )
            except Exception as error:
                errors.append(f"{name}: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))

    if params["enabled"]:
        ctx.add_cleanup("status-injection-" + uuid.uuid4().hex, clear)
    configs = [dict(params["config"])]
    if "rids" in params:
        records = ctx.resource(params["requests"], "requests").snapshot_records()
        ids = [r["wire_request_id"] for r in records]
        if not ids:
            raise RuntimeError("empty declared request cohort")
        if params["rids"] == "first":
            ids = ids[:1]
        configs = (
            [dict(params["config"], rids=ids)]
            if params["fault"] == "status_suppress_rids"
            else [dict(params["config"], rid=rid) for rid in ids]
        )
    receipts = []
    for name in names:
        for config in configs:
            body = dict(
                config, engine=name, type=params["fault"], enabled=params["enabled"]
            )
            status, response = _http(
                ctx, "mock", "inject", deadline, body, params["expected_http"]
            )
            receipts.append(
                dict(
                    request=body, http_status=status, response=response, at=ctx.clock()
                )
            )
    return _frozen(ctx, "control", {"receipts": receipts, "targets": names})


def validate_sample(params, plan):
    p = _validate(
        params, plan, {"include", "duration_s", "interval_s", "until", "requests"}
    )
    p.setdefault("include", ["inflight", "mock"])
    if (
        not isinstance(p["include"], list)
        or not p["include"]
        or any(
            v not in {"inflight", "mock", "info", "ttl", "debug"} for v in p["include"]
        )
    ):
        raise ValueError(f"{plan.path}: invalid source list")
    if "requests" in p:
        plan.reference(p["requests"], "requests")
    p["duration_s"] = _number(p.get("duration_s", 0), plan, "duration_s", 0, 180)
    p["interval_s"] = _number(p.get("interval_s", 0.2), plan, "interval_s", 0.05, 5)
    if "until" in p:
        p["until"] = _validate(
            p["until"], plan, {"metric", "op", "value"}, {"metric", "op", "value"}
        )
        if p["until"]["metric"] not in METRICS or p["until"]["op"] not in {
            "eq",
            "ge",
            "le",
        }:
            raise ValueError(f"{plan.path}: invalid stop condition")
        if p["until"]["metric"] == "fingerprint":
            raise ValueError(
                f"{plan.path}: fingerprint is not a numeric stop condition"
            )
        _number(p["until"]["value"], plan, "until.value", 0, 2**63 - 1)
    return p


def _count(value):
    if isinstance(value, list):
        return len(value)
    if type(value) is not int or value < 0:
        raise RuntimeError("missing/invalid owner count")
    return value


def _alive_count(info, role):
    summary = info["worker_summary"]
    if isinstance(summary, dict) and role in summary:
        return _count(summary[role]["alive"])
    # HttpLoadBalanceServer.buildWorkerSummary omits roles whose directory
    # is empty and returns null when every role is absent. This is a sparse
    # successful response, not permission to turn unavailable data into zero.
    if info.get("success") is not True or info.get("code") != 200:
        raise RuntimeError("absent role lacks a successful Master info response")
    if summary is None:
        return 0
    if not isinstance(summary, dict):
        raise RuntimeError("Master worker_summary is not a role map")
    for key, entry in summary.items():
        if not isinstance(key, str) or not key or not isinstance(entry, dict):
            raise RuntimeError("malformed Master worker role summary")
        discovered, alive = _count(entry["discovered"]), _count(entry["alive"])
        if alive > discovered:
            raise RuntimeError("Master alive count exceeds discovered count")
    return 0


def metric(frame, name):
    if name == "scheduler_tombstones":
        from ...debug_client import check_scheduler_tombstone

        total = 0
        for payload in frame["debug"]:
            rows = payload["components"]["scheduler"]["rows"]
            if len(rows) != 1:
                raise RuntimeError("issued member missing from scheduler debug source")
            if rows[0]["storage_phase"] == "TOMBSTONE":
                passed, _ = check_scheduler_tombstone(rows[0])
                total += bool(passed)
        return total
    if name.startswith("cohort_"):
        records = frame["records"]
        if not records or any(r.get("issued_s") is None for r in records):
            raise RuntimeError("cohort source is empty or undispatched")
        if name == "cohort_fetch_invocations":
            return sum(_count(r["fetch_invocations"]) for r in records)
        if name == "cohort_enqueued":
            return sum(
                r["schedule"]["status"] == "OK" and r["enqueued_by_master"] is True
                for r in records
            )
        total = 0
        for r in records:
            selected = [
                e
                for e in frame["mock"].values()
                if e["grpc_addr"] == r["prefill_addr"]
                and e["role"].lower() == "prefill"
            ]
            if len(selected) != 1:
                raise RuntimeError("cohort Prefill identity is not uniquely mapped")
            lifecycle = selected[0]["request_lifecycle"].get(str(r["wire_request_id"]))
            total += bool(
                lifecycle
                and lifecycle.get("end_state") == "completed"
                and lifecycle.get("end_ms", 0) > 0
            )
        return total
    if name == "master_http":
        return frame["master_http_status"]
    if name in {
        "scheduler",
        "prefill_batches",
        "prefill_requests",
        "decode_total_load",
        "all_inflight",
        "cleanup_inflight",
        "fingerprint",
    }:
        master = frame["inflight"]
        scheduler = _count(master["scheduler_inflight"])

        def endpoints(role):
            rows = master[role + "_endpoints"]
            if not isinstance(rows, list):
                raise RuntimeError("missing owner directory")
            return sorted(
                (
                    r["ip_port"],
                    _count(r["inflight_batches"]),
                    _count(r["inflight_requests"]),
                )
                for r in rows
            )

        p = endpoints("prefill")
        decode = master["decode_endpoints"]
        if not isinstance(decode, list):
            raise RuntimeError("missing Decode owner directory")
        fields = (
            "reserved_total",
            "master_queued",
            "confirmed_accepted",
            "confirmed_running",
            "total_load",
            "engine_load",
            "active_dispatch_permits",
            "engine_capacity_used",
        )
        d = sorted((r["ip_port"], *(_count(r[key]) for key in fields)) for r in decode)
        values = dict(
            scheduler=scheduler,
            prefill_batches=sum(r[1] for r in p),
            prefill_requests=sum(r[2] for r in p),
            decode_total_load=sum(r[5] for r in d),
        )
        values["all_inflight"] = sum(values.values())
        values["cleanup_inflight"] = (
            scheduler + values["prefill_batches"] + values["decode_total_load"]
        )
        values["fingerprint"] = [scheduler, p, d]
        return values[name]
    if name.startswith("alive_"):
        return _alive_count(frame["info"], name[6:].upper())
    if name.startswith("ttl_"):
        return frame["ttl"][name[4:]]
    engines = frame["mock"]
    if not engines:
        raise RuntimeError("missing engine evidence")
    if name.startswith("prefill_"):
        engines = {n: e for n, e in engines.items() if e["role"].lower() == "prefill"}
        if not engines:
            raise RuntimeError("missing Prefill engine evidence")
        name = name[len("prefill_") :]
    if name in {"enqueue_rpc", "fetch_rpc"}:
        key = "enqueue_batch" if name == "enqueue_rpc" else "fetch_response"
        return sum(_count(e["rpc_counts"][key]) for e in engines.values())
    if name == "engine_leaks":
        if any(type(e.get("leak_detected")) is not bool for e in engines.values()):
            raise RuntimeError("missing mock leak evidence")
        return sum(e["leak_detected"] for e in engines.values())
    key = "accepted" if name == "accepted" else "inflight"
    return sum(_count(e[key]) for e in engines.values())


def _debug_directory(capture, n_prefill, n_decode):
    from ...debug_client import DebugUnavailable

    payload = capture.payload
    if payload["endpointDirectoryTruncated"]:
        raise DebugUnavailable("required debug endpoint directory is truncated")
    capture.component("scheduler")
    capture.component("queues")
    owners = {}
    for role, count in (
        ("prefill", n_prefill),
        ("decode", n_decode),
        ("engine", n_prefill + n_decode),
    ):
        pages = [
            (key, capture.component(key))
            for key in payload["components"]
            if key.startswith(role + "/")
        ]
        if len(pages) != count:
            raise DebugUnavailable(f"required {role} directory is incomplete")
        mapped = {}
        for key, page in pages:
            metadata = page["metadata"]
            endpoint, generation = metadata.get("endpoint"), metadata.get(
                "endpoint_generation"
            )
            if (
                not endpoint
                or not generation
                or key != f"{role}/{generation}"
                or endpoint in mapped
            ):
                raise DebugUnavailable("invalid or duplicate debug owner identity")
            mapped[endpoint] = generation
        owners[role] = mapped
    if set(owners["prefill"]) & set(owners["decode"]) or owners["engine"] != dict(
        owners["prefill"], **owners["decode"]
    ):
        raise DebugUnavailable("debug owner and engine directories do not match")
    return owners


def _frame(ctx, params, deadline):
    frame = {"at": ctx.clock(), "env_epoch": ctx.env_epoch}
    if "requests" in params:
        frame["records"] = ctx.resource(
            params["requests"], "requests"
        ).snapshot_records()
    if "debug" in params["include"]:
        from ...debug_client import DebugClient, DebugUnavailable

        client = DebugClient(f"http://127.0.0.1:{ctx.env.master_http_port}")
        ids = [r["wire_request_id"] for r in frame.get("records", [])] or [None]
        frame["debug"] = []
        for rid in ids:
            deadline.check()
            client.timeout_s = min(5, deadline.remaining())
            capture = client.snapshot(
                request_id=rid, include="scheduler,queues,prefill,decode,engine"
            )
            if capture.payload["status"] != "ok":
                artifact = _artifact(ctx, "incomplete-debug", capture.payload)
                raise DebugUnavailable(
                    f"incomplete required debug source; evidence={artifact}"
                )
            directory = _debug_directory(
                capture,
                ctx.instance["environment"]["n_prefill"],
                ctx.instance["environment"]["n_decode"],
            )
            prior_directory = getattr(
                ctx, "status_debug_directory", (ctx.env_epoch, directory)
            )
            if prior_directory[0] == ctx.env_epoch and prior_directory[1] != directory:
                raise DebugUnavailable(
                    "debug endpoint generation changed during the experiment"
                )
            ctx.status_debug_directory = (ctx.env_epoch, directory)
            for component in capture.payload["components"]:
                capture.component(component)
            identity = (ctx.env_epoch, capture.payload["instanceId"])
            previous = getattr(ctx, "status_debug_identity", identity)
            if previous[0] == identity[0] and previous != identity:
                raise DebugUnavailable(
                    "Master generation changed during status experiment"
                )
            ctx.status_debug_identity = identity
            frame["debug"].append(capture.payload)
    if "inflight" in params["include"]:
        frame["master_http_status"], frame["inflight"] = _http(
            ctx, "master", "rtp_llm/inflight_status", deadline
        )
        metric(frame, "fingerprint")
    if "mock" in params["include"]:
        frame["mock"] = _mock(ctx, deadline)
    if "info" in params["include"]:
        _, frame["info"] = _http(ctx, "master", "rtp_llm/master/info", deadline, {})
    if "ttl" in params["include"]:
        from ...engine_ops import parse_prometheus_samples

        source_epoch, source_path = getattr(
            ctx, "status_metrics_source", (ctx.env_epoch, "actuator/prometheus")
        )
        if source_epoch != ctx.env_epoch:
            raise RuntimeError("metric source belongs to a different epoch")
        _, body = _http(ctx, "metrics", source_path, deadline, text=True)
        counts = {"scheduler": 0.0, "prefill": 0.0, "decode": 0.0}
        for _, labels, value in parse_prometheus_samples(
            body, "flexlb_app_flexlb_inflight_ttl_expired"
        ):
            role = labels.get("role", "").lower()
            if role in counts:
                counts[role] += value
        frame["ttl"] = counts
        frame["ttl_series_sparse"] = True
    return frame


def _compare(actual, op, expected):
    return {
        "eq": lambda: actual == expected,
        "ge": lambda: actual >= expected,
        "le": lambda: actual <= expected,
    }[op]()


def execute_sample(ctx, params, deadline):
    start, epoch, env = ctx.clock(), ctx.env_epoch, ctx.env
    end = start + params["duration_s"]
    frames, size = [], 0
    while True:
        deadline.check()
        if ctx.env_epoch != epoch or ctx.env is not env:
            raise RuntimeError("sampling epoch changed")
        try:
            frame = _frame(ctx, params, deadline)
        except Exception as error:
            _artifact(
                ctx,
                "incomplete-samples",
                {
                    "env_epoch": epoch,
                    "started_s": start,
                    "failed_s": ctx.clock(),
                    "frames": frames,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        size += len(json.dumps(frame).encode())
        if len(frames) >= 4000 or size > 32 * 1024 * 1024:
            raise RuntimeError("status observation budget exceeded")
        frames.append(frame)
        until = params.get("until")
        if until and _compare(
            metric(frame, until["metric"]), until["op"], until["value"]
        ):
            break
        if ctx.clock() >= end:
            break
        deadline.sleep(min(params["interval_s"], end - ctx.clock()))
    return _frozen(
        ctx, "samples", {"started_s": start, "ended_s": ctx.clock(), "frames": frames}
    )


def validate_check(params, plan):
    p = _validate(
        params,
        plan,
        {"snapshot", "baseline", "metric", "op", "expected", "aggregate"},
        {"snapshot", "metric", "op", "expected"},
    )
    for key in ("snapshot", "baseline"):
        if key in p:
            plan.reference(p[key], "snapshot")
    if p["metric"] not in METRICS or p["op"] not in {"eq", "ge", "le"}:
        raise ValueError(f"{plan.path}: invalid status metric comparison")
    p.setdefault("aggregate", "last")
    if p["aggregate"] not in {"last", "max", "min", "all", "stable"}:
        raise ValueError(f"{plan.path}: invalid aggregation")
    if p["metric"] == "fingerprint" and p["aggregate"] == "stable" and "baseline" in p:
        raise ValueError(
            f"{plan.path}: fingerprint stability cannot also compare a baseline; use aggregate all"
        )
    if p["aggregate"] == "stable" or p["metric"] == "fingerprint":
        if p["op"] != "eq" or type(p["expected"]) is not bool:
            raise ValueError(f"{plan.path}: fingerprint/stability compares a boolean")
        if (
            p["metric"] == "fingerprint"
            and p["aggregate"] != "stable"
            and "baseline" not in p
        ):
            raise ValueError(f"{plan.path}: fingerprint comparison needs a baseline")
    else:
        _number(p["expected"], plan, "expected", -(2**63), 2**63 - 1)
    return p


def execute_check(ctx, params, deadline):
    deadline.check()
    data = ctx.resource(params["snapshot"], "snapshot").to_dict()
    frames = data["frames"]
    if not frames or data["env_epoch"] != ctx.env_epoch:
        raise RuntimeError("empty or stale observation")
    values = [metric(f, params["metric"]) for f in frames]
    baseline = None
    if "baseline" in params:
        before = ctx.resource(params["baseline"], "snapshot").to_dict()
        if before["env_epoch"] != data["env_epoch"] or not before["frames"]:
            raise RuntimeError("invalid baseline epoch")
        baseline = metric(before["frames"][-1], params["metric"])
        if params["metric"] == "fingerprint":
            values = [value == baseline for value in values]
        else:
            values = [value - baseline for value in values]
    mode = params["aggregate"]
    actual = (
        values[-1]
        if mode == "last"
        else (
            max(values) if mode == "max" else min(values) if mode == "min" else values
        )
    )
    if mode == "stable":
        actual = all(v == values[0] for v in values)
    passed = (
        all(_compare(v, params["op"], params["expected"]) for v in values)
        if mode == "all"
        else _compare(actual, params["op"], params["expected"])
    )
    return StageOutput(
        {"passed": passed},
        [
            CheckResult(
                "contract",
                "PASS" if passed else "FAIL",
                "status protocol boundary",
                actual=actual,
                expected=params["expected"],
                evidence={"metric": params["metric"], "baseline": baseline},
            )
        ],
    )


HANDLERS = [
    StageHandler(
        "status_control", validate_control, execute_control, {"snapshot": "snapshot"}
    ),
    StageHandler(
        "status_sample", validate_sample, execute_sample, {"snapshot": "snapshot"}
    ),
    StageHandler(
        "status_check",
        validate_check,
        execute_check,
        {"passed": "boolean"},
        checks=frozenset({"contract"}),
    ),
]


def validate_prepare(params, plan):
    p = _validate(
        params,
        plan,
        {
            "count",
            "concurrency",
            "input_len",
            "output_len",
            "consume",
            "stream_timeout_s",
            "expected_rpc_statuses",
            "observe_schedule_future_terminal",
        },
    )
    for key, default, maximum in [
        ("count", 4, 64),
        ("concurrency", 4, 16),
        ("input_len", 2048, 1048576),
        ("output_len", 2, 4096),
    ]:
        p[key] = _number(p.get(key, default), plan, key, 1, maximum, True)
    p.setdefault("consume", "immediate")
    if p["consume"] not in {"immediate", "deferred"}:
        raise ValueError(f"{plan.path}: invalid consumption mode")
    if p["consume"] == "deferred":
        from flexlb_cfg import PROFILE_CAPS

        if any(
            "enqueue_batch" not in PROFILE_CAPS[profile] for profile in plan.profiles
        ):
            raise ValueError(f"{plan.path}: deferred consumption needs enqueue_batch")
    p["stream_timeout_s"] = _number(
        p.get("stream_timeout_s", 15), plan, "stream_timeout_s", 1, 120
    )
    statuses = p.setdefault("expected_rpc_statuses", [])
    if (
        not isinstance(statuses, list)
        or len(statuses) != len(set(statuses))
        or any(
            v not in {"INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED", "UNKNOWN"}
            for v in statuses
        )
    ):
        raise ValueError(f"{plan.path}: invalid explicit request RPC status allowlist")
    observe = p.get("observe_schedule_future_terminal", False)
    if type(observe) is not bool or (observe and "DEADLINE_EXCEEDED" not in statuses):
        raise ValueError(
            f"{plan.path}: Schedule terminal observation requires an explicit deadline allowlist"
        )
    return p


def _witnessed_schedule_timeout(record, allowed, enabled=False):
    if not enabled or record["schedule"].get("status") != "ERROR":
        return False
    witness = record["schedule"].get("future_terminal", {})
    return (
        witness.get("exception_type") == "FutureTimeoutError"
        and witness.get("done_before") is True
        and witness.get("done_after") is True
        and witness.get("cancelled") is False
        and witness.get("observed_s") is not None
        and witness.get("code") in allowed
        and witness.get("code") == "DEADLINE_EXCEEDED"
        and record.get("cancel", {}).get("requested_s") is None
        and record["stream"].get("started_s") is None
    )


def _terminal_records(records, allowed=(), observe_schedule_future_terminal=False):
    if not records:
        raise RuntimeError("empty request evidence")
    for record in records:
        if any(
            record.get(key) is None
            for key in ("issued_s", "consumer_exit_s", "transport_terminal_s")
        ):
            raise RuntimeError("incomplete request terminal evidence")
        if record.get("cancel", {}).get("requested_s") is not None:
            raise RuntimeError(
                "cancelled request cannot satisfy expected-fault contract"
            )
        if record["stream"].get("started_s") is not None and (
            record.get("consumer_done") is not True
            or record.get("consumer_completion_verified") is not True
        ):
            raise RuntimeError("request consumer completion has not been verified")
        for phase in ("schedule", "stream"):
            rpc = record[phase]
            if phase == "stream" and rpc.get("started_s") is None:
                if record["schedule"]["status"] == "OK":
                    raise RuntimeError("successful Schedule has no stream terminal")
                continue
            if rpc.get("ended_s") is None or rpc.get("status") is None:
                raise RuntimeError("incomplete RPC terminal evidence")
            state = rpc["status"]
            if phase == "schedule" and _witnessed_schedule_timeout(
                record, allowed, observe_schedule_future_terminal
            ):
                continue
            if state not in {"OK", "REJECTED", *allowed}:
                if state == "DEADLINE_EXCEEDED":
                    raise TimeoutError("RPC deadline is not a business terminal")
                raise RuntimeError(f"unavailable request transport evidence: {state}")


class StatusRequests:
    """Prepared IDs precede injection; bounded submissions reuse the core RPC driver."""

    def __init__(self, ctx, params):
        from ..backend import RequestBatch

        self.ctx, self.params, self.epoch = ctx, params, ctx.env_epoch
        self.children, self.threads, self.done = [], [], []
        self.errors = []
        self.cancelled = threading.Event()
        self.dispatched = False
        self.queue_lock = threading.Lock()
        self.next_index = 0
        self.exit_records = []

        class PreparedRequest(RequestBatch):
            def __init__(child):
                super().__init__(ctx, dict(params, count=1))
                child.status_done = threading.Event()
                child.status_exit_s = None
                child.prepared = super().issue(ctx.ops.next_request_id(), ctx.clock)
                child.prepared["planned_s"] = child.prepared["issued_s"]
                child.prepared["issued_s"] = None

            def issue(child, rid, clock):
                # The core submit allocates an unused ID before calling issue;
                # return the predeclared ID so injections bind before Schedule.
                child.update(child.prepared, issued_s=clock())
                return child.prepared

            def _error(child, record, phase, exc):
                super()._error(record, phase, exc)
                if phase != "schedule" or not params.get(
                    "observe_schedule_future_terminal", False
                ):
                    return
                import grpc

                if not isinstance(exc, grpc.FutureTimeoutError):
                    return
                # Observe before RequestBatch's exception cleanup cancels the
                # call. A local result() timeout alone proves no RPC terminal.
                call = child.entries[-1]["call"]
                witness = dict(
                    exception_type="FutureTimeoutError", observed_s=ctx.clock()
                )
                try:
                    witness["done_before"] = call.done()
                    if witness["done_before"] is True:
                        witness["cancelled"] = call.cancelled()
                        if witness["cancelled"] is False:
                            witness["code"] = getattr(call.code(), "name", None)
                            witness["done_after"] = call.done()
                except Exception as observation_error:
                    witness["observation_error"] = repr(observation_error)
                child.update(record, schedule={"future_terminal": witness})

            def _consume(child, entry, end):
                try:
                    super()._consume(entry, end)
                finally:
                    child.status_exit_s = ctx.clock()
                    child.status_done.set()

            def _start_consumer(child, entry, end):
                super()._start_consumer(
                    entry, min(end, ctx.clock() + params["stream_timeout_s"])
                )

        for _ in range(params["count"]):
            child = PreparedRequest()
            ctx.register_resource(
                "requests", child, cleanup=child.cleanup, historical=True
            )
            self.children.append(child)

    def snapshot_records(self):
        return [
            record for child in self.children for record in child.snapshot_records()
        ]

    def _worker(self, deadline, done, index):
        try:
            while not self.cancelled.is_set():
                with self.queue_lock:
                    if self.next_index >= len(self.children):
                        break
                    child = self.children[self.next_index]
                    self.next_index += 1
                try:
                    child.submit(deadline)
                except Exception as error:
                    deadline.check()
                    code_fn = getattr(error, "code", None)
                    state = (
                        getattr(code_fn(), "name", None) if callable(code_fn) else None
                    )
                    records = child.snapshot_records()
                    if state not in self.params["expected_rpc_statuses"] and not all(
                        _witnessed_schedule_timeout(
                            r,
                            self.params["expected_rpc_statuses"],
                            self.params.get("observe_schedule_future_terminal", False),
                        )
                        for r in records
                    ):
                        raise
                    _terminal_records(
                        records,
                        self.params["expected_rpc_statuses"],
                        self.params.get("observe_schedule_future_terminal", False),
                    )
        except Exception as error:
            with self.queue_lock:
                self.errors.append(error)
        finally:
            with self.queue_lock:
                self.exit_records.append(
                    {"worker": index, "exited_s": self.ctx.clock()}
                )
            done.set()

    def dispatch(self, deadline):
        if self.dispatched:
            raise RuntimeError("request cohort already dispatched")
        self.dispatched = True
        for index in range(min(self.params["concurrency"], len(self.children))):
            done = threading.Event()
            worker = threading.Thread(
                target=self._worker,
                args=(deadline, done, index),
                name="status-submit",
                daemon=True,
            )
            self.threads.append(worker)
            self.done.append(done)
            worker.start()
        self.join_submission(deadline)
        if self.errors:
            raise self.errors[0]

    def join_submission(self, deadline):
        for worker, done in zip(self.threads, self.done):
            while not done.is_set():
                deadline.check()
                done.wait(min(0.05, deadline.remaining()))
            worker.join(timeout=deadline.remaining())
            if worker.is_alive():
                raise TimeoutError("status submission worker has not exited")
        if len(self.exit_records) != len(self.threads):
            raise RuntimeError("status submission exit records missing")

    def wait(self, deadline):
        if not self.dispatched:
            raise RuntimeError("cannot consume an undispatched cohort")
        self.join_submission(deadline)
        if self.errors:
            raise self.errors[0]
        for child in self.children:
            try:
                child.wait(deadline)
            except (RuntimeError, TimeoutError):
                # Core wait reports recorded RPC errors. It may only be relaxed
                # after the stage budget and every independent exit proof pass.
                deadline.check()
                if not self.params["expected_rpc_statuses"]:
                    raise
                records = child.snapshot_records()
                if not any(
                    r[phase]["status"] in self.params["expected_rpc_statuses"]
                    or (
                        phase == "schedule"
                        and _witnessed_schedule_timeout(
                            r,
                            self.params["expected_rpc_statuses"],
                            self.params.get("observe_schedule_future_terminal", False),
                        )
                    )
                    for r in records
                    for phase in ("schedule", "stream")
                ):
                    raise
                self._completed_child(child, deadline)
                _terminal_records(
                    records,
                    self.params["expected_rpc_statuses"],
                    self.params.get("observe_schedule_future_terminal", False),
                )
            self._completed_child(child, deadline)
        records = self.snapshot_records()
        _terminal_records(
            records,
            self.params["expected_rpc_statuses"],
            self.params.get("observe_schedule_future_terminal", False),
        )
        from .elastic import request_success

        return {
            "completed": all(request_success(r) for r in records),
            "error_count": sum(not request_success(r) for r in records),
        }

    def _completed_child(self, child, deadline):
        for entry in child.entries:
            thread = entry["thread"]
            if thread is not None:
                while not child.status_done.is_set():
                    deadline.check()
                    child.status_done.wait(min(0.05, deadline.remaining()))
                if child.status_exit_s is None:
                    raise RuntimeError("request consumer exit record missing")
                thread.join(timeout=deadline.remaining())
                if thread.is_alive():
                    raise TimeoutError("request consumer has not exited")
        deadline.check()

    def cancel_server(self, deadline):
        return sum(child.cancel_server(deadline) for child in self.children)

    def cleanup(self, deadline):
        self.cancelled.set()
        # Cancelling the actual outstanding RPC allows submit workers to exit.
        errors = []
        for child in self.children:
            try:
                child.cleanup(deadline)
                self._completed_child(child, deadline)
            except Exception as error:
                errors.append(error)
        try:
            self.join_submission(deadline)
        except Exception as error:
            errors.append(error)
        if errors:
            raise errors[0]


def execute_prepare(ctx, params, deadline):
    deadline.check()
    requests = StatusRequests(ctx, params)
    handle = ctx.register_resource(
        "requests", requests, cleanup=requests.cleanup, historical=True
    )
    return StageOutput({"requests": handle, "count": params["count"]})


def validate_dispatch(params, plan):
    p = _validate(params, plan, {"requests"}, {"requests"})
    plan.reference(p["requests"], "requests")
    return p


def execute_dispatch(ctx, params, deadline):
    requests = ctx.resource(params["requests"], "requests")
    if not isinstance(requests, StatusRequests):
        raise RuntimeError("status dispatch requires its prepared cohort")
    requests.dispatch(deadline)
    return StageOutput({"requests": ctx.resolve(params["requests"])})


def validate_outcomes(params, plan):
    p = _validate(
        params,
        plan,
        {
            "requests",
            "success_min",
            "failure_min",
            "failure_max",
            "error_code",
            "failure_phase",
            "per_execution_batch",
            "slo_or_success",
            "timeout_or_success",
        },
        {"requests"},
    )
    plan.reference(p["requests"], "requests")
    for key in ("success_min", "failure_min", "failure_max"):
        if key in p:
            p[key] = _number(p[key], plan, key, 0, 64, True)
    if "error_code" in p:
        _number(p["error_code"], plan, "error_code", 0, 2**31 - 1, True)
    if p.get("failure_phase") not in (None, "schedule", "execution"):
        raise ValueError(
            f"{plan.path}: ACK/schedule versus execution phase must be explicit"
        )
    if any(
        type(p.get(key, False)) is not bool
        for key in ("slo_or_success", "timeout_or_success")
    ):
        raise ValueError(f"{plan.path}: invalid slo_or_success")
    if "per_execution_batch" in p:
        q = p["per_execution_batch"]
        if not isinstance(q, dict) or set(q) != {
            "fallback_success_min",
            "fallback_failure_max",
        }:
            raise ValueError("per_execution_batch requires explicit fallback bounds")
        for key, value in q.items():
            _number(value, plan, key, 0, 64, True)
    return p


def execute_outcomes(ctx, params, deadline):
    from .elastic import request_success

    deadline.check()
    records = ctx.resource(params["requests"], "requests").snapshot_records()
    if not records or any(
        r.get("issued_s") is None or r.get("consumer_exit_s") is None for r in records
    ):
        raise RuntimeError("request outcomes need a nonempty fully observed cohort")
    cohort = ctx.resource(params["requests"], "requests")
    allowed = (
        cohort.params.get("expected_rpc_statuses", [])
        if isinstance(cohort, StatusRequests)
        else []
    )
    if isinstance(cohort, StatusRequests):
        for child in cohort.children:
            cohort._completed_child(child, deadline)
    _terminal_records(records, allowed)
    failures = [r for r in records if not request_success(r)]
    success = len(records) - len(failures)
    batch_bounds = None
    if "per_execution_batch" in params:
        from .execution_evidence import partial_outcome_bounds, read_events

        batch_bounds = partial_outcome_bounds(
            read_events(ctx.env.run_dir / "engine_events.jsonl"),
            {r["wire_request_id"] for r in records},
            **params["per_execution_batch"],
        )
    success_min = (
        batch_bounds["success_min"] if batch_bounds else params.get("success_min", 0)
    )
    failure_max = (
        batch_bounds["failure_max"] if batch_bounds else params.get("failure_max", 64)
    )
    passed = (
        success >= success_min
        and len(failures) >= params.get("failure_min", 0)
        and len(failures) <= failure_max
    )
    if "error_code" in params:
        code = params["error_code"]
        passed = (
            passed
            and bool(failures)
            and all(
                r.get("business_error_code") == code
                or str(code) in str(r["schedule"].get("error"))
                for r in failures
            )
        )
    if batch_bounds is not None:
        # Injection is per execution batch; typed errors remain per request.
        passed = passed and all(
            r.get("business_error_code") == params["error_code"]
            and bool(r.get("business_error_message"))
            for r in failures
        )
    if params.get("failure_phase") == "schedule":
        passed = (
            passed
            and bool(failures)
            and all(
                r["schedule"]["status"] == "REJECTED"
                and r["stream"]["started_s"] is None
                for r in failures
            )
        )
    if params.get("failure_phase") == "execution":
        passed = (
            passed
            and bool(failures)
            and all(
                r["schedule"]["status"] == "OK"
                and r["stream"]["started_s"] is not None
                and r.get("business_error_code") not in (None, 0)
                for r in failures
            )
        )
    if params.get("slo_or_success") or params.get("timeout_or_success"):

        def legal_failure(record):
            text = (
                str(record["schedule"].get("error"))
                + " "
                + str(record["stream"].get("error"))
                + " "
                + str(record.get("business_error_code"))
                + " "
                + str(record.get("business_error_message"))
            ).lower()
            if params.get("timeout_or_success"):
                # Preserve the old _timeout_typed vocabulary, but only after
                # completed consumers and explicit transport policy are verified.
                if record["stream"]["status"] == "OK" and not record.get(
                    "business_finished"
                ):
                    text += " stream did not complete"
                return any(
                    token in text
                    for token in (
                        "deadline",
                        "timeout",
                        "timed out",
                        "not complete",
                        "expire",
                        "exhaust",
                        "8400",
                        "8511",
                        "8431",
                    )
                )
            return any(
                token in text for token in ("slo", "queue timeout", "queue deadline")
            )

        passed = passed and all(legal_failure(record) for record in failures)
    artifact = _artifact(
        ctx,
        "outcomes",
        {
            "records": records,
            "success": success,
            "failure_count": len(failures),
            "batch_bounds": batch_bounds,
        },
    )
    return StageOutput(
        {"passed": bool(passed)},
        [
            CheckResult(
                "contract",
                "PASS" if passed else "FAIL",
                "per-request status boundary",
                actual={
                    "success": success,
                    "failure_count": len(failures),
                    "batch_bounds": batch_bounds,
                },
                expected=params,
                evidence={"artifact": artifact},
            )
        ],
        artifacts=[artifact],
    )


HANDLERS += [
    StageHandler(
        "status_prepare",
        validate_prepare,
        execute_prepare,
        {"requests": "requests", "count": "integer"},
    ),
    StageHandler(
        "status_dispatch", validate_dispatch, execute_dispatch, {"requests": "requests"}
    ),
    StageHandler(
        "status_outcomes",
        validate_outcomes,
        execute_outcomes,
        {"passed": "boolean"},
        checks=frozenset({"contract"}),
    ),
]


def validate_perf(params, plan):
    p = _validate(
        params,
        plan,
        {"role", "selection", "prefill_fixed_ms", "restore_prefill_fixed_ms"},
        {"prefill_fixed_ms", "restore_prefill_fixed_ms"},
    )
    p.setdefault("role", "prefill")
    p.setdefault("selection", "all")
    if p["role"] not in {"prefill", "decode"} or p["selection"] not in {"all", "first"}:
        raise ValueError(f"{plan.path}: invalid performance target")
    for key in ("prefill_fixed_ms", "restore_prefill_fixed_ms"):
        p[key] = _number(p[key], plan, key, 0, 100000)
    return p


def execute_perf(ctx, params, deadline):
    names = _targets(ctx, params, deadline)
    epoch, env = ctx.env_epoch, ctx.env

    def restore(limit):
        if ctx.env_epoch != epoch or ctx.env is not env:
            raise RuntimeError("performance cleanup epoch changed")
        errors = []
        for name in names:
            try:
                _http(
                    ctx,
                    "mock",
                    "set_perf",
                    limit,
                    {
                        "engine": name,
                        "prefill_fixed_ms": params["restore_prefill_fixed_ms"],
                    },
                )
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("; ".join(errors))

    ctx.add_cleanup("status-perf-" + uuid.uuid4().hex, restore)
    receipts = []
    for name in names:
        body = {"engine": name, "prefill_fixed_ms": params["prefill_fixed_ms"]}
        code, response = _http(ctx, "mock", "set_perf", deadline, body)
        receipts.append({"request": body, "http_status": code, "response": response})
    return _frozen(ctx, "perf", {"receipts": receipts})


HANDLERS.append(
    StageHandler("status_perf", validate_perf, execute_perf, {"snapshot": "snapshot"})
)


def validate_metrics_ready(params, plan):
    p = _validate(params, plan, {"duration_s", "interval_s"})
    p["duration_s"] = _number(p.get("duration_s", 180), plan, "duration_s", 1, 180)
    p["interval_s"] = _number(p.get("interval_s", 2), plan, "interval_s", 0.1, 5)
    return p


def execute_metrics_ready(ctx, params, deadline):
    """Explicit old cold-exporter gate, distinct from the later event assertions."""
    start, epoch, env = ctx.clock(), ctx.env_epoch, ctx.env
    end = start + params["duration_s"]
    attempts = []
    try:
        while True:
            deadline.check()
            if ctx.env_epoch != epoch or ctx.env is not env:
                raise RuntimeError("metrics readiness epoch changed")
            for path in ("actuator/prometheus", "prometheus"):
                attempt = {"path": path, "at": ctx.clock()}
                attempts.append(attempt)
                try:
                    code, body = _http(
                        ctx,
                        "metrics",
                        path,
                        deadline,
                        allowed=(200, 404, 503),
                        text=True,
                    )
                    attempt["http_status"] = code
                    if code == 200:
                        ctx.status_metrics_source = (epoch, path)
                        return _frozen(
                            ctx,
                            "metrics-ready",
                            {"attempts": attempts, "path": path, "body": body},
                        )
                except (OSError, urllib.error.URLError) as error:
                    attempt["error"] = f"{type(error).__name__}: {error}"
                if ctx.clock() >= end:
                    raise TimeoutError("Master metric endpoint did not become ready")
            deadline.sleep(min(params["interval_s"], end - ctx.clock()))
    except Exception as error:
        _artifact(
            ctx,
            "metrics-not-ready",
            {
                "env_epoch": epoch,
                "attempts": attempts,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


HANDLERS.append(
    StageHandler(
        "status_metrics_ready",
        validate_metrics_ready,
        execute_metrics_ready,
        {"snapshot": "snapshot"},
    )
)
