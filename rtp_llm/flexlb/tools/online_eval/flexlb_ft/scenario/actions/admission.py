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
                schedule_timeout_s=30,
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
                        self.ctx.clock() + 30 + p["request_timeout_s"],
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
    if p["metric"] not in METRICS or p["op"] not in {"eq", "ge", "le", "lt"}:
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
                or metric == "schedule_latency_max"
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
            (min(values) if metric == "latency_min" else max(values))
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
                else actual <= expected if params["op"] == "le" else actual < expected
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
    running = max(row["running"] for s in samples for row in s["engines"].values())
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
