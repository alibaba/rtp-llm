"""Bounded admission traffic and evidence checks; no legacy case dispatch."""

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, wait

from ..backend import RequestBatch
from ..contracts import CheckResult, StageHandler, StageOutput
from ..runtime import Deadline
from .elastic import request_success
from .engine_control import _engines, _http
from .master import _master_json


def _fields(params, allowed, required=()):
    if (
        not isinstance(params, dict)
        or set(params) - set(allowed)
        or not set(required) <= set(params)
    ):
        raise ValueError("invalid admission action fields")
    return dict(params)


def _traffic_validate(params, plan):
    p = _fields(
        params,
        {
            "count",
            "concurrency",
            "input_len",
            "output_len",
            "request_timeout_s",
            "consume",
            "keys_per_request",
            "priority",
            "schedule_timeout_s",
        },
    )
    for key, default, limit in (
        ("count", 1, 300),
        ("concurrency", 1, 32),
        ("input_len", 512, 65536),
        ("output_len", 2, 4096),
    ):
        p.setdefault(key, default)
        if type(p[key]) is not int or not 1 <= p[key] <= limit:
            raise ValueError(f"invalid bounded admission {key}")
    if "priority" in p and (
        type(p["priority"]) is not int or not -(2**31) <= p["priority"] < 2**31
    ):
        raise ValueError("priority must be an explicit int32 when present")
    p.setdefault("request_timeout_s", 20)
    p.setdefault("schedule_timeout_s", 30)
    if (
        type(p["schedule_timeout_s"]) not in (int, float)
        or not math.isfinite(p["schedule_timeout_s"])
        or not 0 < p["schedule_timeout_s"] <= 60
    ):
        raise ValueError("invalid admission schedule deadline")
    if (
        type(p["request_timeout_s"]) not in (int, float)
        or not math.isfinite(p["request_timeout_s"])
        or not 0 < p["request_timeout_s"] <= 60
    ):
        raise ValueError("invalid admission request deadline")
    p.setdefault("consume", "immediate")
    p.setdefault("keys_per_request", 0)
    if p["consume"] not in {"immediate", "deferred"}:
        raise ValueError("invalid consumer mode")
    if type(p["keys_per_request"]) is not int or not 0 <= p["keys_per_request"] <= 1024:
        raise ValueError("invalid per-request block count")
    return p


def _wave_validate(params, plan):
    p = _traffic_validate(params, plan)
    if p["consume"] != "immediate":
        raise ValueError(
            "deferred admission uses admission_fire with its enqueue_batch capability"
        )
    return p


class AdmissionWave:
    """Own submission futures plus core RequestBatch consumer completion proofs."""

    def __init__(self, ctx, params):
        self.ctx, self.params = ctx, params
        self.pool = ThreadPoolExecutor(max_workers=params["concurrency"])
        self.items = []
        self.stop = threading.Event()
        self.path = ctx.artifact_dir / f"admission-wave-{len(ctx._resources)}.json"

    def submit(self, deadline):
        p = self.params
        batch = RequestBatch(
            self.ctx,
            dict(
                count=1,
                input_len=p["input_len"],
                output_len=p["output_len"],
                consume=p.get("consume", "immediate"),
                schedule_timeout_s=p.get("schedule_timeout_s", 30),
                stream_timeout_s=p["request_timeout_s"],
            ),
        )
        if "priority" in p:
            batch.params["priority"] = p["priority"]
        if p.get("keys_per_request", 0):
            seed = self.ctx.ops.next_request_id()
            batch.params["block_keys"] = [
                seed * 2048 + i for i in range(1, p["keys_per_request"] + 1)
            ]
        item = dict(
            batch=batch, started=False, done=threading.Event(), error=None, future=None
        )
        batch.artifact = self.path.with_name(
            self.path.stem + f"-request-{len(self.items)}.json"
        )
        self.items.append(item)

        def run():
            try:
                if self.stop.is_set():
                    return
                item["started"] = True
                request_deadline = Deadline(
                    min(
                        self.ctx.instance_deadline_s,
                        self.ctx.clock()
                        + p.get("schedule_timeout_s", 30)
                        + p["request_timeout_s"],
                    ),
                    self.ctx.clock,
                    self.ctx.sleeper,
                )
                batch.submit(request_deadline)
                if p.get("consume", "immediate") == "immediate" and not p.get(
                    "submission_only"
                ):
                    batch.wait(request_deadline)
            except Exception as exc:
                item["error"] = exc
            finally:
                item["done"].set()

        item["future"] = self.pool.submit(run)

    def rows(self):
        rows = []
        for item in self.items:
            batch = item["batch"]
            for entry, record in zip(batch.entries, batch.snapshot_records()):
                response = entry.get("response")
                record["schedule_response"] = (
                    None
                    if response is None
                    else dict(
                        code=int(response.code),
                        success=bool(response.success),
                        error_message=str(response.error_message),
                    )
                )
                rows.append(record)
        return rows

    def persist(self):
        self.path.write_text(
            json.dumps(
                dict(
                    records=self.rows(),
                    jobs=[
                        dict(
                            started=x["started"],
                            submission_done=x["done"].is_set(),
                            error=repr(x["error"]) if x["error"] else None,
                            cancelled_before_start=x["future"].cancelled()
                            and not x["started"],
                        )
                        for x in self.items
                    ],
                ),
                indent=2,
            )
            + "\n"
        )

    def await_done(self, deadline):
        for item in self.items:
            future = item["future"]
            if future.cancelled() and not item["started"]:
                continue
            while not item["done"].is_set():
                item["done"].wait(min(0.1, deadline.remaining()))
            future.result(timeout=deadline.remaining())
        errors = [x["error"] for x in self.items if x["error"]]
        if errors:
            raise errors[0]
        if self.params.get("consume") == "deferred" or self.params.get(
            "submission_only"
        ):
            for item in self.items:
                if item["started"]:
                    item["batch"].wait(deadline)
        self.persist()

    def cleanup(self, deadline):
        self.stop.set()
        self.pool.shutdown(wait=False, cancel_futures=True)
        errors = []
        try:
            for item in self.items:
                item["batch"].cancel("admission_cleanup")
            live = [x["future"] for x in self.items if not x["future"].cancelled()]
            _, unfinished = wait(live, timeout=deadline.remaining())
            if unfinished:
                raise TimeoutError("admission submission workers did not exit")
            self.pool.shutdown(wait=True, cancel_futures=True)
            for item in self.items:
                if item["started"]:
                    if not item["done"].is_set():
                        raise RuntimeError(
                            "submission future lacks its independent exit signal"
                        )
                    try:
                        item["batch"].cleanup(deadline)
                    except Exception as exc:
                        errors.append(exc)
            if errors:
                raise errors[0]
        finally:
            self.persist()


def _wave(ctx, params, deadline):
    wave = AdmissionWave(ctx, params)
    handle = ctx.register_resource("admission_wave", wave, wave.cleanup)
    for _ in range(params["count"]):
        deadline.check()
        wave.submit(deadline)
    return StageOutput({"wave": handle})


def _wait_validate(params, plan):
    p = _fields(params, {"wave"}, {"wave"})
    plan.reference(p["wave"], "admission_wave")
    return p


def _wait(ctx, params, deadline):
    wave = ctx.resource(params["wave"], "admission_wave")
    wave.await_done(deadline)
    rows = wave.rows()
    return StageOutput(
        {"rows": ctx.register_resource("admission_rows", rows, historical=True)},
        artifacts=[str(wave.path)],
    )


def _occupy_validate(params, plan):
    p = _fields(params, {"targets"}, {"targets"})
    if (
        not isinstance(p["targets"], list)
        or not p["targets"]
        or any(
            not isinstance(n, str)
            or not n.startswith("prefill-")
            or not n[8:].isdigit()
            for n in p["targets"]
        )
    ):
        raise ValueError("occupancy requires explicit prefill targets")
    return p


def _occupy(ctx, params, deadline):
    wave = AdmissionWave(
        ctx,
        _traffic_validate(dict(count=50, concurrency=8, request_timeout_s=20), None),
    )
    handle = ctx.register_resource("admission_wave", wave, wave.cleanup)
    samples = []
    path = wave.path.with_name(wave.path.stem + "-occupancy.json")
    occupied = False
    try:
        for _ in range(50):
            deadline.check()
            snapshot = _engines(_http(ctx.ops, "snapshot", deadline), params["targets"])
            counts = {}
            for name, row in snapshot.items():
                if any(
                    type(row.get(k)) is not int or row[k] < 0
                    for k in ("waiting", "running")
                ):
                    raise ValueError("missing engine occupancy counters")
                counts[name] = row["waiting"] + row["running"]
            samples.append(dict(time_s=ctx.clock(), counts=counts))
            if all(n >= 1 for n in counts.values()):
                occupied = True
                break
            wave.submit(deadline)
            deadline.sleep(0.3)
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput(
        {"wave": handle},
        [
            CheckResult(
                "all_occupied",
                "PASS" if occupied else "FAIL",
                actual=samples[-1] if samples else None,
            )
        ],
        artifacts=[str(path)],
    )


METRICS = {
    "admitted_count",
    "success_count",
    "reject_count",
    "serve_error_count",
    "all_error_contains",
    "any_error_contains",
    "all_reject_code",
    "all_schedule_code",
    "schedule_latency_max",
    "schedule_latency_min",
    "deadline_reject_family",
    "await_fifo",
    "consumer_fifo",
    "await_strict_fifo",
    "latency_min",
    "latency_max",
}


def _check_validate(params, plan):
    p = _fields(
        params,
        {
            "rows",
            "metric",
            "expected",
            "op",
            "text",
            "scope",
            "min_samples",
            "case_sensitive",
        },
        {"rows", "metric", "expected", "op"},
    )
    plan.reference(p["rows"], "admission_rows")
    if p["metric"] not in METRICS or p["op"] not in {"eq", "ge", "le", "lt", "gt"}:
        raise ValueError("unsupported admission metric")
    p.setdefault("case_sensitive", False)
    if type(p["case_sensitive"]) is not bool:
        raise ValueError("case_sensitive must be boolean")
    p.setdefault("scope", "all")
    p.setdefault("min_samples", 1)
    if (
        p["scope"] not in {"all", "rejected"}
        or type(p["min_samples"]) is not int
        or p["min_samples"] < 1
    ):
        raise ValueError("admission check requires actual nonempty evidence")
    if p["metric"] in {"all_error_contains", "any_error_contains"}:
        if (
            not isinstance(p.get("text"), list)
            or not p["text"]
            or any(not isinstance(t, str) or not t for t in p["text"])
        ):
            raise ValueError("error text predicate requires nonempty tokens")
    if type(p["expected"]) not in (int, float, bool) or not math.isfinite(
        p["expected"]
    ):
        raise ValueError("expected must be finite scalar")
    return p


def _check(ctx, params, deadline):
    deadline.check()
    rows = ctx.resource(params["rows"], "admission_rows")
    rejected = [r for r in rows if r["schedule"]["status"] == "REJECTED"]
    selected = rejected if params["scope"] == "rejected" else rows
    metric = params["metric"]
    if metric == "admitted_count":
        actual = sum(r["schedule"]["status"] == "OK" for r in selected)
    elif metric == "success_count":
        actual = sum(request_success(r) for r in selected)
    elif metric == "reject_count":
        actual = len(rejected)
    elif metric == "serve_error_count":
        actual = sum(
            r["schedule"]["status"] != "REJECTED" and not request_success(r)
            for r in selected
        )
    elif metric == "deadline_reject_family":
        actual = bool(selected) and all(
            r["schedule"]["status"] == "REJECTED"
            and (
                r["schedule_response"]["code"] == 8511
                or any(
                    token in str(r["schedule_response"]["error_message"]).lower()
                    for token in (
                        "deadline",
                        "expired",
                        "exhaust",
                        "8400",
                        "8511",
                        "8431",
                    )
                )
            )
            for r in selected
        )
    elif metric in {"await_fifo", "await_strict_fifo", "consumer_fifo"}:
        field = "consumer_exit_s" if metric == "consumer_fifo" else "await_return_s"
        ends = [r.get(field) for r in selected]
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in ends):
            raise ValueError("FIFO needs actual completion timestamps")
        actual = len(ends) >= 2 and all(
            a < b if metric == "await_strict_fifo" else a <= b
            for a, b in zip(ends, ends[1:])
        )
    elif metric in {"all_reject_code", "all_schedule_code"}:
        checked = rejected if metric == "all_reject_code" else selected
        actual = bool(checked) and all(
            r["schedule_response"] is not None
            and r["schedule_response"]["code"] == params["expected"]
            for r in checked
        )
        # This metric compares actual response codes, never text substrings.
        expected = True
    elif metric in {"all_error_contains", "any_error_contains"}:
        match = all if metric == "all_error_contains" else any
        normalize = (
            (lambda value: value) if params.get("case_sensitive", False) else str.lower
        )
        actual = bool(selected) and all(
            not request_success(r)
            and match(
                normalize(t)
                in normalize(
                    str(
                        r["schedule"]["error"]
                        or r["stream"]["error"]
                        or r["business_error_message"]
                        or ""
                    )
                )
                for t in params["text"]
            )
            for r in selected
        )
    else:
        values = []
        for row in selected:
            end = (
                row["schedule"]["ended_s"]
                if row["schedule"]["status"] == "REJECTED"
                or metric in {"schedule_latency_max", "schedule_latency_min"}
                else row["consumer_exit_s"]
            )
            start = row["schedule"]["started_s"]
            if (
                type(start) not in (int, float)
                or type(end) not in (int, float)
                or not math.isfinite(start)
                or not math.isfinite(end)
                or end < start
            ):
                raise ValueError("latency needs real request timestamps")
            values.append(end - start)
        actual = (
            (
                min(values)
                if metric in {"latency_min", "schedule_latency_min"}
                else max(values)
            )
            if values
            else None
        )
    expected = (
        True
        if metric in {"all_reject_code", "all_schedule_code"}
        else params["expected"]
    )
    good = len(selected) >= params["min_samples"] and actual is not None
    if good:
        good = (
            actual == expected
            if params["op"] == "eq"
            else (
                actual >= expected
                if params["op"] == "ge"
                else (
                    actual <= expected
                    if params["op"] == "le"
                    else (
                        actual < expected if params["op"] == "lt" else actual > expected
                    )
                )
            )
        )
    return StageOutput(
        {"passed": good},
        [
            CheckResult(
                "criterion",
                "PASS" if good else "FAIL",
                actual=actual,
                expected=expected,
                evidence={
                    "metric": metric,
                    "scope": params["scope"],
                    "samples": len(selected),
                },
            )
        ],
    )


HANDLERS = [
    StageHandler("admission_wave", _wave_validate, _wave, {"wave": "admission_wave"}),
    StageHandler("admission_wait", _wait_validate, _wait, {"rows": "admission_rows"}),
    StageHandler(
        "admission_occupy",
        _occupy_validate,
        _occupy,
        {"wave": "admission_wave"},
        checks=frozenset({"all_occupied"}),
    ),
    StageHandler(
        "admission_check",
        _check_validate,
        _check,
        {"passed": "boolean"},
        checks=frozenset({"criterion"}),
    ),
]


def _fire_validate(params, plan):
    p = dict(params)
    spacing = p.pop("spacing_s", 0)
    every = p.pop("spacing_every", 1)
    p.setdefault("consume", "deferred")
    p = _traffic_validate(p, plan)
    if (
        type(spacing) not in (int, float)
        or not math.isfinite(spacing)
        or not 0 <= spacing <= 2
    ):
        raise ValueError("fire spacing must be finite and bounded")
    if type(every) is not int or every < 1:
        raise ValueError("fire spacing cadence must be positive")
    p.update(spacing_s=spacing, spacing_every=every, submission_only=True)
    return p


def _fire(ctx, params, deadline):
    wave = AdmissionWave(ctx, params)
    handle = ctx.register_resource("admission_wave", wave, wave.cleanup)
    for i in range(params["count"]):
        deadline.check()
        wave.submit(deadline)
        item = wave.items[-1]
        item["future"].result(timeout=deadline.remaining())
        if item["error"]:
            raise item["error"]
        if (i + 1) % params["spacing_every"] == 0:
            deadline.sleep(params["spacing_s"])
    wave.persist()
    return StageOutput(
        {
            "wave": handle,
            "rows": ctx.register_resource(
                "admission_rows", wave.rows(), historical=True
            ),
        },
        artifacts=[str(wave.path)],
    )


GAUGES = {
    "active_decode_requests",
    "waiting",
    "running",
    "prefill_waiting_batches",
    "held_blocks",
    "available_blocks",
    "inflight",
}


def _observe_validate(params, plan):
    p = _fields(
        params,
        {"targets", "fields", "duration_s", "until_value", "until_op", "reduce"},
        {"targets", "fields"},
    )
    from .engine_control import ENGINE_NAME

    if (
        not isinstance(p["targets"], list)
        or not p["targets"]
        or any(
            not isinstance(n, str) or not ENGINE_NAME.fullmatch(n) for n in p["targets"]
        )
    ):
        raise ValueError("observation needs explicit valid engine targets")
    if (
        not isinstance(p["fields"], list)
        or not p["fields"]
        or any(f not in GAUGES for f in p["fields"])
    ):
        raise ValueError("unsupported engine gauge")
    p.setdefault("duration_s", 0)
    p.setdefault("reduce", "all")
    if (
        type(p["duration_s"]) not in (int, float)
        or not math.isfinite(p["duration_s"])
        or not 0 <= p["duration_s"] <= 60
    ):
        raise ValueError("observation interval must be bounded")
    if ("until_value" in p) != ("until_op" in p):
        raise ValueError("observation stopping condition requires value and operator")
    if "until_value" in p and (
        type(p["until_value"]) is not int
        or p["until_value"] < 0
        or p["until_op"] not in {"eq", "ge", "le"}
    ):
        raise ValueError("invalid observation stopping condition")
    if p["reduce"] not in {"all", "any"}:
        raise ValueError("observation reduction must be all/any")
    return p


def _observe(ctx, params, deadline):
    samples = []
    until = ctx.clock() + params["duration_s"]
    path = ctx.artifact_dir / f"admission-observation-{len(ctx._resources)}.json"
    try:
        while True:
            deadline.check()
            engines = _engines(_http(ctx.ops, "snapshot", deadline), params["targets"])
            sample = {}
            for name, entry in engines.items():
                sample[name] = {}
                for field in params["fields"]:
                    value = entry.get(field)
                    if type(value) is not int or value < 0:
                        raise ValueError(f"missing/invalid engine gauge {name}.{field}")
                    sample[name][field] = value
            samples.append(dict(time_s=ctx.clock(), engines=sample))
            if "until_value" in params:
                want, op = params["until_value"], params["until_op"]
                matches = [
                    v == want if op == "eq" else v >= want if op == "ge" else v <= want
                    for row in sample.values()
                    for v in row.values()
                ]
                if all(matches) if params["reduce"] == "all" else any(matches):
                    break
            if ctx.clock() >= until:
                break
            deadline.sleep(min(0.2, until - ctx.clock()))
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput(
        {
            "snapshot": ctx.register_resource(
                "admission_observation", samples, historical=True
            )
        },
        artifacts=[str(path)],
    )


def _gauge_validate(params, plan):
    p = _fields(
        params,
        {"snapshot", "fields", "stat", "op", "expected"},
        {"snapshot", "fields", "stat", "op", "expected"},
    )
    plan.reference(p["snapshot"], "admission_observation")
    if (
        not isinstance(p["fields"], list)
        or not p["fields"]
        or any(f not in GAUGES for f in p["fields"])
    ):
        raise ValueError("invalid check fields")
    if (
        p["stat"] not in {"max_seen", "max_latest", "min_latest"}
        or p["op"] not in {"eq", "ge", "le"}
        or type(p["expected"]) is not int
        or p["expected"] < 0
    ):
        raise ValueError("invalid gauge comparison")
    return p


def _gauge(ctx, params, deadline):
    deadline.check()
    samples = ctx.resource(params["snapshot"], "admission_observation")
    if not samples:
        raise ValueError("gauge check has no actual observations")
    used = samples if params["stat"] == "max_seen" else samples[-1:]
    values = [
        row[field]
        for s in used
        for row in s["engines"].values()
        for field in params["fields"]
    ]
    actual = min(values) if params["stat"] == "min_latest" else max(values)
    expected = params["expected"]
    passed = (
        actual == expected
        if params["op"] == "eq"
        else actual >= expected if params["op"] == "ge" else actual <= expected
    )
    return StageOutput(
        {"passed": passed},
        [
            CheckResult(
                "gauge", "PASS" if passed else "FAIL", actual=actual, expected=expected
            )
        ],
    )


def _decode_park_validate(params, plan):
    p = _fields(params, {"snapshot", "gate"}, {"snapshot", "gate"})
    plan.reference(p["snapshot"], "admission_observation")
    if type(p["gate"]) is not int or p["gate"] <= 0:
        raise ValueError("decode gate must be positive")
    return p


def _decode_park(ctx, params, deadline):
    deadline.check()
    samples = ctx.resource(params["snapshot"], "admission_observation")
    if not samples:
        raise ValueError("park inference requires observed decode gauges")
    running = max(
        row["active_decode_requests"] for s in samples for row in s["engines"].values()
    )
    waiting = max(row["waiting"] for s in samples for row in s["engines"].values())
    activated = running >= params["gate"]
    passed = not activated or waiting >= 1
    return StageOutput(
        {"passed": passed},
        [
            CheckResult(
                "conditional_park",
                "PASS" if passed else "FAIL",
                actual=dict(
                    running_max=running, waiting_max=waiting, gate_filled=activated
                ),
                expected=params["gate"],
            )
        ],
    )


HANDLERS += [
    StageHandler(
        "admission_fire",
        _fire_validate,
        _fire,
        {"wave": "admission_wave", "rows": "admission_rows"},
        requires=frozenset({"enqueue_batch"}),
    ),
    StageHandler(
        "admission_observe",
        _observe_validate,
        _observe,
        {"snapshot": "admission_observation"},
    ),
    StageHandler(
        "admission_gauge_check",
        _gauge_validate,
        _gauge,
        {"passed": "boolean"},
        checks=frozenset({"gauge"}),
    ),
    StageHandler(
        "admission_decode_park_check",
        _decode_park_validate,
        _decode_park,
        {"passed": "boolean"},
        checks=frozenset({"conditional_park"}),
    ),
]


def _clean_validate(params, plan):
    p = _fields(params, {"targets"}, {"targets"})
    from .engine_control import ENGINE_NAME

    if (
        not isinstance(p["targets"], list)
        or not p["targets"]
        or any(
            not isinstance(n, str) or not ENGINE_NAME.fullmatch(n) for n in p["targets"]
        )
    ):
        raise ValueError("engine cleanup requires explicit targets")
    return p


def _clean(ctx, params, deadline):
    samples = []
    path = ctx.artifact_dir / f"admission-engine-clean-{len(ctx._resources)}.json"
    try:
        while True:
            snapshot = _engines(_http(ctx.ops, "snapshot", deadline), params["targets"])
            rows = {}
            for name, entry in snapshot.items():
                count, leak = entry.get("inflight"), entry.get("leak_detected")
                if type(count) is not int or count < 0 or type(leak) is not bool:
                    raise ValueError("missing engine-owned inflight/leak evidence")
                rows[name] = dict(inflight=count, leak_detected=leak)
            samples.append(dict(time_s=ctx.clock(), engines=rows))
            if all(
                r["inflight"] == 0 and r["leak_detected"] is False
                for r in rows.values()
            ):
                break
            deadline.sleep(0.2)
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput(
        {"clean": True},
        [CheckResult("engine_clean", "PASS", actual=rows)],
        artifacts=[str(path)],
    )


HANDLERS.append(
    StageHandler(
        "admission_engine_clean",
        _clean_validate,
        _clean,
        {"clean": "boolean"},
        checks=frozenset({"engine_clean"}),
    )
)


def _tracked_validate(params, plan):
    p = dict(params)
    p.setdefault("consume", "immediate")
    p = _fire_validate(p, plan)
    if p["consume"] != "immediate":
        raise ValueError("tracked fire supports immediate consumers only")
    return p


def _park_validate(params, plan):
    p = _clean_validate(params, plan)
    if len(set(p["targets"])) != len(p["targets"]):
        raise ValueError("park counter owners must be distinct")
    return p


def _park_sample(ctx, targets, deadline):
    data = _master_json(ctx, "single", "/rtp_llm/inflight_status", deadline)
    count = data.get("scheduler_inflight")
    if type(count) is not int or count < 0:
        raise ValueError("missing/invalid scheduler_inflight")
    engines = _engines(_http(ctx.ops, "snapshot", deadline), targets)
    counters = {}
    for name, row in engines.items():
        if any(
            type(row.get(k)) is not int or row[k] < 0 for k in ("waiting", "running")
        ):
            raise ValueError("park evidence needs actual engine waiting/running")
        counters[name] = {k: row[k] for k in ("waiting", "running")}
    live = sum(v for row in counters.values() for v in row.values())
    return dict(
        time_s=ctx.clock(),
        scheduler_inflight=count,
        engines=counters,
        engine_live=live,
        parked=count - live,
    )


def _lease_precondition(ctx, params, deadline):
    sample = _park_sample(ctx, params["targets"], deadline)
    passed = sample["scheduler_inflight"] >= 1 and any(
        name.startswith("prefill-") and row["running"] >= 1
        for name, row in sample["engines"].items()
    )
    return StageOutput(
        {"passed": passed},
        [CheckResult("lease_held", "PASS" if passed else "FAIL", actual=sample)],
    )


class AdmissionDrain:
    """Concurrent legacy wait-return observation, distinct from consumer exit time."""

    def __init__(self, ctx, waves):
        self.ctx, self.waves = ctx, waves
        self.pool = ThreadPoolExecutor(max_workers=sum(len(w.items) for w in waves))
        self.jobs = []
        self.path = ctx.artifact_dir / f"admission-drain-{len(ctx._resources)}.json"

    def start(self, deadline):
        for wave in self.waves:
            for item in wave.items:
                if item["error"]:
                    raise item["error"]
                job = dict(
                    done=threading.Event(), future=None, end=None, error=None, item=item
                )
                self.jobs.append(job)

                def run(job=job):
                    try:
                        job["item"]["batch"].wait(deadline)
                        job["end"] = self.ctx.clock()
                    except Exception as exc:
                        job["error"] = exc
                    finally:
                        job["done"].set()

                job["future"] = self.pool.submit(run)

    def await_done(self, deadline):
        for job in self.jobs:
            if not job["done"].wait(deadline.remaining()):
                raise TimeoutError("concurrent drain has no completion signal")
            job["future"].result(timeout=deadline.remaining())
            if job["error"]:
                raise job["error"]
        rows = []
        for wave in self.waves:
            wave_rows = wave.rows()
            by_id = {id(j["item"]): j for j in self.jobs}
            for item, row in zip(wave.items, wave_rows):
                row["await_return_s"] = by_id[id(item)]["end"]
                rows.append(row)
        self.path.write_text(json.dumps(rows, indent=2) + "\n")
        return rows

    def cleanup(self, deadline):
        self.pool.shutdown(wait=False, cancel_futures=True)
        for wave in self.waves:
            for item in wave.items:
                item["batch"].cancel("admission_drain_cleanup")
        futures = [j["future"] for j in self.jobs if j["future"] is not None]
        if futures and wait(futures, timeout=deadline.remaining())[1]:
            raise TimeoutError("concurrent drain workers did not exit")
        if any(
            not j["done"].is_set() and not j["future"].cancelled()
            for j in self.jobs
            if j["future"] is not None
        ):
            raise RuntimeError("concurrent drain missing independent worker exit")
        self.pool.shutdown(wait=True, cancel_futures=True)


def _drain_validate(params, plan):
    p = _fields(params, {"waves"}, {"waves"})
    if not isinstance(p["waves"], list) or not 1 <= len(p["waves"]) <= 8:
        raise ValueError("drain needs bounded explicit waves")
    for ref in p["waves"]:
        plan.reference(ref, "admission_wave")
    return p


def _drain(ctx, params, deadline):
    waves = [ctx.resource(ref, "admission_wave") for ref in params["waves"]]
    if not 1 <= sum(len(w.items) for w in waves) <= 32:
        raise ValueError("concurrent drain limited to 32 requests")
    drain = AdmissionDrain(ctx, waves)
    ctx.register_resource("admission_drain", drain, drain.cleanup)
    drain.start(deadline)
    rows = drain.await_done(deadline)
    return StageOutput(
        {"rows": ctx.register_resource("admission_rows", rows, historical=True)},
        artifacts=[str(drain.path)],
    )


HANDLERS += [
    StageHandler(
        "admission_tracked_fire",
        _tracked_validate,
        _fire,
        {"wave": "admission_wave", "rows": "admission_rows"},
    ),
    StageHandler(
        "admission_lease_precondition",
        _park_validate,
        _lease_precondition,
        {"passed": "boolean"},
        checks=frozenset({"lease_held"}),
    ),
    StageHandler(
        "admission_drain", _drain_validate, _drain, {"rows": "admission_rows"}
    ),
]


def _burst_validate(params, plan):
    p = dict(params)
    await_submissions = p.pop("await_submissions", True)
    if type(await_submissions) is not bool:
        raise ValueError("await_submissions must be boolean")
    p = _fire_validate(p, plan)
    if p["count"] > 32 or p["concurrency"] < p["count"]:
        raise ValueError("burst needs one bounded worker per submitted request")
    if not await_submissions and p["consume"] != "immediate":
        raise ValueError("timed asynchronous burst needs immediate consumers")
    p["await_submissions"] = await_submissions
    p["submission_only"] = await_submissions
    return p


def _burst(ctx, params, deadline):
    wave = AdmissionWave(ctx, params)
    handle = ctx.register_resource("admission_wave", wave, wave.cleanup)
    for i in range(params["count"]):
        deadline.check()
        wave.submit(deadline)
        if (i + 1) % params["spacing_every"] == 0:
            deadline.sleep(params["spacing_s"])
    if params["await_submissions"]:
        for item in wave.items:
            if not item["done"].wait(deadline.remaining()):
                raise TimeoutError("burst submission lacks completion signal")
            item["future"].result(timeout=deadline.remaining())
            if item["error"]:
                raise item["error"]
    return StageOutput({"wave": handle})


def _drain_start(ctx, params, deadline):
    waves = [ctx.resource(ref, "admission_wave") for ref in params["waves"]]
    if not 1 <= sum(len(w.items) for w in waves) <= 32:
        raise ValueError("concurrent drain limited to 32 requests")
    drain = AdmissionDrain(ctx, waves)
    handle = ctx.register_resource("admission_drain", drain, drain.cleanup)
    drain.start(deadline)
    return StageOutput({"drain": handle})


def _drain_collect_validate(params, plan):
    p = _fields(params, {"drain"}, {"drain"})
    plan.reference(p["drain"], "admission_drain")
    return p


def _drain_collect(ctx, params, deadline):
    drain = ctx.resource(params["drain"], "admission_drain")
    rows = drain.await_done(deadline)
    return StageOutput(
        {"rows": ctx.register_resource("admission_rows", rows, historical=True)},
        artifacts=[str(drain.path)],
    )


def _ledger_validate(params, plan):
    p = _fields(params, {"duration_s"}, {"duration_s"})
    if (
        type(p["duration_s"]) not in (int, float)
        or not math.isfinite(p["duration_s"])
        or not 0 < p["duration_s"] <= 60
    ):
        raise ValueError("ledger observation needs finite bounded duration")
    return p


def _ledger(ctx, params, deadline):
    samples = []
    until = ctx.clock() + params["duration_s"]
    path = ctx.artifact_dir / f"admission-ledger-{len(ctx._resources)}.json"
    try:
        while ctx.clock() < until:
            data = _master_json(ctx, "single", "/rtp_llm/inflight_status", deadline)
            sched = data.get("scheduler_inflight")
            endpoints = data.get("prefill_endpoints")
            if (
                type(sched) is not int
                or sched < 0
                or not isinstance(endpoints, list)
                or not endpoints
            ):
                raise ValueError(
                    "ledger requires scheduler and explicit prefill endpoints"
                )
            batches = requests = 0
            for endpoint in endpoints:
                b, r = endpoint.get("inflight_batches"), endpoint.get(
                    "inflight_requests"
                )
                b = len(b) if isinstance(b, list) else b
                if any(type(v) is not int or v < 0 for v in (b, r)):
                    raise ValueError("ledger requires actual owner batch/member counts")
                batches += b
                requests += r
            samples.append(
                dict(
                    time_s=ctx.clock(),
                    scheduler=sched,
                    batches=batches,
                    requests=requests,
                    raw=data,
                )
            )
            deadline.sleep(min(0.2, max(0, until - ctx.clock())))
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput(
        {
            "samples": ctx.register_resource(
                "admission_ledger", samples, historical=True
            )
        },
        artifacts=[str(path)],
    )


def _ledger_check_validate(params, plan):
    p = _fields(
        params,
        {"samples", "n_requests", "expect_intermediate"},
        {"samples", "n_requests", "expect_intermediate"},
    )
    plan.reference(p["samples"], "admission_ledger")
    if (
        type(p["n_requests"]) is not int
        or p["n_requests"] < 1
        or type(p["expect_intermediate"]) is not bool
    ):
        raise ValueError("invalid ledger linkage expectation")
    return p


def _ledger_check(ctx, params, deadline):
    deadline.check()
    samples = ctx.resource(params["samples"], "admission_ledger")
    peak_batches = max((s["batches"] for s in samples), default=-1)
    peak_requests = max((s["requests"] for s in samples), default=-1)
    intermediate = sorted(
        {s["requests"] for s in samples if 0 < s["requests"] < peak_requests}
    )
    monotonic = all(
        b["scheduler"] <= a["scheduler"] for a, b in zip(samples, samples[1:])
    )
    passed = (
        bool(samples)
        and peak_batches == 1
        and peak_requests == params["n_requests"]
        and bool(intermediate) == params["expect_intermediate"]
        and monotonic
    )
    return StageOutput(
        {"passed": passed},
        [
            CheckResult(
                "linkage",
                "PASS" if passed else "FAIL",
                actual=dict(
                    peak_batches=peak_batches,
                    peak_requests=peak_requests,
                    intermediate=intermediate,
                    scheduler_non_increasing=monotonic,
                ),
            )
        ],
    )


def _budget_snapshot_validate(params, plan):
    p = _clean_validate(params, plan)
    if len(p["targets"]) != 1 or not p["targets"][0].startswith("prefill-"):
        raise ValueError("budget snapshot requires one explicit prefill")
    return p


def _budget_snapshot(ctx, params, deadline):
    rows = _engines(_http(ctx.ops, "snapshot", deadline), params["targets"])
    row = rows[params["targets"][0]]
    for name in ("prefill_batches", "prefill_batch_requests", "max_prefill_batch_size"):
        if type(row.get(name)) is not int or row[name] < 0:
            raise ValueError("missing executed-batch counters")
    path = ctx.artifact_dir / f"admission-budget-{len(ctx._resources)}.json"
    path.write_text(json.dumps(row, indent=2) + "\n")
    return StageOutput(
        {
            "snapshot": ctx.register_resource(
                "admission_budget_snapshot", row, historical=True
            )
        },
        artifacts=[str(path)],
    )


def _shape_validate(params, plan):
    p = _fields(
        params, {"before", "after", "expected"}, {"before", "after", "expected"}
    )
    for name in ("before", "after"):
        plan.reference(p[name], "admission_budget_snapshot")
    if (
        not isinstance(p["expected"], list)
        or len(p["expected"]) != 3
        or any(type(x) is not int or x < 0 for x in p["expected"])
    ):
        raise ValueError("shape expectation is three nonnegative counters")
    return p


def _shape(ctx, params, deadline):
    deadline.check()
    before, after = [
        ctx.resource(params[k], "admission_budget_snapshot")
        for k in ("before", "after")
    ]
    actual = [
        after[k] - before[k] for k in ("prefill_batches", "prefill_batch_requests")
    ] + [after["max_prefill_batch_size"]]
    # Old construction gate is diagnostic, explicitly excluded from the verdict.
    path = ctx.artifact_dir / f"admission-shape-{len(ctx.outputs)}.json"
    path.write_text(
        json.dumps(
            dict(
                actual=actual,
                expected=params["expected"],
                matches=actual == params["expected"],
            ),
            indent=2,
        )
        + "\n"
    )
    return StageOutput({"matches": actual == params["expected"]}, artifacts=[str(path)])


def _budget_check_validate(params, plan):
    p = _fields(
        params, {"rows", "snapshot", "baseline_rows", "metric"}, {"rows", "metric"}
    )
    plan.reference(p["rows"], "admission_rows")
    if p["metric"] == "batch_identity":
        plan.reference(p.get("snapshot"), "admission_budget_snapshot")
    elif p["metric"] == "ttft_degradation":
        plan.reference(p.get("baseline_rows"), "admission_rows")
    elif p["metric"] != "client_two_clusters":
        raise ValueError("unknown budget check")
    return p


def _two_clusters(values, sep):
    if len(values) != 4 or any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values
    ):
        return False
    a, b, c, d = sorted(values)
    return b - a <= sep and c - b > sep and d - c <= sep


def _budget_check(ctx, params, deadline):
    deadline.check()
    rows = ctx.resource(params["rows"], "admission_rows")
    if params["metric"] == "batch_identity":
        snap = ctx.resource(params["snapshot"], "admission_budget_snapshot")
        lifecycle = snap.get("request_lifecycle", {})
        values = [
            lifecycle.get(str(r["wire_request_id"]), {}).get("batch_id") for r in rows
        ]
        passed = (
            bool(values) and len(set(values)) == 1 and values[0] not in (None, 0, -1)
        )
        actual = values
    elif params["metric"] == "client_two_clusters":
        actual = [r.get("await_return_s") for r in rows]
        passed = _two_clusters(actual, 1.0)
    else:
        baseline = ctx.resource(params["baseline_rows"], "admission_rows")

        def p50(records):
            values = []
            for r in records:
                start, end = r["schedule"]["started_s"], r["consumer_exit_s"]
                if (
                    not request_success(r)
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v)
                        for v in (start, end)
                    )
                    or end < start
                ):
                    raise ValueError("TTFT requires successful real request timing")
                values.append(end - start)
            return sorted(values)[int(len(values) * 0.5)] if values else None

        base, wave = p50(baseline), p50(rows)
        degradation = (
            ((wave - base) / base * 100 if base > 0 else 0)
            if base is not None and wave is not None
            else None
        )
        passed = degradation is not None and degradation <= 50
        actual = dict(baseline_p50_s=base, wave_p50_s=wave, degradation_pct=degradation)
    return StageOutput(
        {"passed": passed},
        [CheckResult("criterion", "PASS" if passed else "FAIL", actual=actual)],
    )


def _engine_clusters_validate(params, plan):
    p = _fields(params, {"rows", "snapshot"}, {"rows", "snapshot"})
    plan.reference(p["rows"], "admission_rows")
    plan.reference(p["snapshot"], "admission_budget_snapshot")
    return p


def _engine_clusters(ctx, params, deadline):
    deadline.check()
    rows = ctx.resource(params["rows"], "admission_rows")
    snap = ctx.resource(params["snapshot"], "admission_budget_snapshot")
    values = [
        snap.get("request_lifecycle", {})
        .get(str(r["wire_request_id"]), {})
        .get("end_ms", 0)
        for r in rows
    ]
    matches = _two_clusters(values, 1000)
    path = ctx.artifact_dir / f"admission-engine-clusters-{len(ctx.outputs)}.json"
    path.write_text(json.dumps(dict(values=values, matches=matches), indent=2) + "\n")
    return StageOutput({"matches": matches}, artifacts=[str(path)])


HANDLERS += [
    StageHandler(
        "admission_burst",
        _burst_validate,
        _burst,
        {"wave": "admission_wave"},
        requires=frozenset({"enqueue_batch"}),
    ),
    StageHandler(
        "admission_drain_start",
        _drain_validate,
        _drain_start,
        {"drain": "admission_drain"},
    ),
    StageHandler(
        "admission_drain_collect",
        _drain_collect_validate,
        _drain_collect,
        {"rows": "admission_rows"},
    ),
    StageHandler(
        "admission_ledger_observe",
        _ledger_validate,
        _ledger,
        {"samples": "admission_ledger"},
    ),
    StageHandler(
        "admission_ledger_check",
        _ledger_check_validate,
        _ledger_check,
        {"passed": "boolean"},
        checks=frozenset({"linkage"}),
    ),
    StageHandler(
        "admission_budget_snapshot",
        _budget_snapshot_validate,
        _budget_snapshot,
        {"snapshot": "admission_budget_snapshot"},
    ),
    StageHandler(
        "admission_shape_observe", _shape_validate, _shape, {"matches": "boolean"}
    ),
    StageHandler(
        "admission_budget_check",
        _budget_check_validate,
        _budget_check,
        {"passed": "boolean"},
        checks=frozenset({"criterion"}),
    ),
    StageHandler(
        "admission_engine_clusters",
        _engine_clusters_validate,
        _engine_clusters,
        {"matches": "boolean"},
    ),
]
