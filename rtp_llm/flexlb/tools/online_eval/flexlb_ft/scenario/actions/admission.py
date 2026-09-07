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
        params, {"count", "concurrency", "input_len", "output_len", "request_timeout_s"}
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
    p.setdefault("request_timeout_s", 20)
    if (
        type(p["request_timeout_s"]) not in (int, float)
        or not math.isfinite(p["request_timeout_s"])
        or not 0 < p["request_timeout_s"] <= 60
    ):
        raise ValueError("invalid admission request deadline")
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
                consume="immediate",
                schedule_timeout_s=30,
                stream_timeout_s=p["request_timeout_s"],
            ),
        )
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
    "success_count",
    "reject_count",
    "serve_error_count",
    "all_error_contains",
    "any_error_contains",
    "all_reject_code",
    "latency_min",
    "latency_max",
}


def _check_validate(params, plan):
    p = _fields(
        params,
        {"rows", "metric", "expected", "op", "text", "scope", "min_samples"},
        {"rows", "metric", "expected", "op"},
    )
    plan.reference(p["rows"], "admission_rows")
    if p["metric"] not in METRICS or p["op"] not in {"eq", "ge", "le", "lt"}:
        raise ValueError("unsupported admission metric")
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
    if metric == "success_count":
        actual = sum(request_success(r) for r in selected)
    elif metric == "reject_count":
        actual = len(rejected)
    elif metric == "serve_error_count":
        actual = sum(
            r["schedule"]["status"] != "REJECTED" and not request_success(r)
            for r in selected
        )
    elif metric == "all_reject_code":
        actual = bool(rejected) and all(
            r["schedule_response"] is not None
            and r["schedule_response"]["code"] == params["expected"]
            for r in rejected
        )
        # This metric compares actual response codes, never text substrings.
        expected = True
    elif metric in {"all_error_contains", "any_error_contains"}:
        match = all if metric == "all_error_contains" else any
        actual = bool(selected) and all(
            not request_success(r)
            and match(
                t.lower()
                in str(
                    r["schedule"]["error"]
                    or r["stream"]["error"]
                    or r["business_error_message"]
                    or ""
                ).lower()
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
                else row["consumer_exit_s"]
            )
            start = row["schedule"]["started_s"]
            if (
                type(start) not in (int, float)
                or type(end) not in (int, float)
                or end < start
            ):
                raise ValueError("latency needs real request timestamps")
            values.append(end - start)
        actual = (
            (min(values) if metric == "latency_min" else max(values))
            if values
            else None
        )
    expected = True if metric == "all_reject_code" else params["expected"]
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
    StageHandler(
        "admission_wave", _traffic_validate, _wave, {"wave": "admission_wave"}
    ),
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
