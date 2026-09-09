"""Balance measurements: finite traffic, explicit snapshots and graded checks.

Python case programs define the steps. This module provides bounded I/O, typed
resources and measurements; YAML supplies configuration data.
"""

import copy
import json
import math
import threading
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ...grade import GradeReport
from ..contracts import CheckResult, StageHandler, StageOutput
from .engine_control import _http


def _params(params, plan, allowed, required=()):
    if (
        not isinstance(params, dict)
        or set(params) - set(allowed)
        or set(required) - set(params)
    ):
        raise ValueError(f"{plan.path}: invalid balance parameters")
    return copy.deepcopy(params)


def _number(value, minimum=0, maximum=1000000, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError("balance parameter outside finite bounds")
    return value


def _artifact(ctx, prefix, value):
    path = ctx.artifact_dir / f"balance-{prefix}-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return str(path)


def _fleet(ctx, deadline, role):
    raw = _http(ctx.ops, "snapshot", deadline)
    rows = raw.get("engines")
    if not isinstance(rows, list):
        raise ValueError("missing mock engine inventory")
    selected = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("role") != role:
            continue
        name, addr = row.get("name"), row.get("grpc_addr")
        if (
            not isinstance(name, str)
            or not isinstance(addr, str)
            or not addr
            or name in selected
        ):
            raise ValueError("missing or duplicate engine identity")
        if row.get("stopped") is not False:
            raise ValueError("balance fleet contains a stopped/unknown engine")
        for field in ("completed", "waiting", "running"):
            _number(row.get(field))
        selected[name] = copy.deepcopy(row)
    if len(selected) < 2:
        raise ValueError(f"balance requires >=2 live {role} workers")
    return dict(
        role=role, engines=selected, time_s=ctx.clock(), env_epoch=ctx.env_epoch
    )


def _identity(before, after):
    a, b = before["engines"], after["engines"]
    if before["env_epoch"] != after["env_epoch"] or set(a) != set(b):
        raise ValueError("balance observation changed environment/fleet")
    for name in a:
        if (a[name]["role"], a[name]["grpc_addr"]) != (
            b[name]["role"],
            b[name]["grpc_addr"],
        ):
            raise ValueError("balance engine endpoint identity changed")


def _snapshot_validate(params, plan):
    p = _params(params, plan, {"role"}, {"role"})
    if p["role"] not in ("prefill", "decode"):
        raise ValueError("balance snapshot role must be prefill or decode")
    return p


def _snapshot(ctx, params, deadline):
    snapshot = _fleet(ctx, deadline, params["role"])
    names = list(snapshot["engines"])
    return StageOutput(
        dict(
            snapshot=ctx.register_resource("snapshot", snapshot),
            first=names[0],
            second=names[1],
        ),
        artifacts=[_artifact(ctx, "snapshot", snapshot)],
    )


def _traffic_validate(params, plan):
    p = _params(
        params,
        plan,
        {
            "count",
            "concurrency",
            "input_len",
            "output_len",
            "interval_s",
            "stream_timeout_s",
            "unique_keys",
            "defer_batch",
            "await_completion",
        },
    )
    defaults = dict(
        count=1,
        concurrency=1,
        input_len=2048,
        output_len=2,
        interval_s=0,
        stream_timeout_s=15,
        unique_keys=True,
        defer_batch=False,
        await_completion=True,
    )
    for k, v in defaults.items():
        p.setdefault(k, v)
    for k, upper in (
        ("count", 50),
        ("concurrency", 20),
        ("input_len", 200000),
        ("output_len", 10000),
    ):
        _number(p[k], 1, upper, integer=True)
    _number(p["interval_s"], 0, 2)
    _number(p["stream_timeout_s"], 1, 60)
    for k in ("unique_keys", "defer_batch", "await_completion"):
        if type(p[k]) is not bool:
            raise ValueError(f"{k} must be boolean")
    if p["defer_batch"] and p["concurrency"] != 1:
        raise ValueError("deferred cohort preserves sequential Schedule ordering")
    return p


class Traffic:
    """Finite requests using the core's terminal-record/consumer-event contract."""

    def __init__(self, ctx, params):
        self.ctx, self.params = ctx, params
        self.batches, self.errors = [], []
        self.lock = threading.RLock()
        self.done, self.cancelled = threading.Event(), threading.Event()
        self.exit_s = None
        self.path = ctx.artifact_dir / f"balance-traffic-{uuid.uuid4().hex}.json"
        self.thread = threading.Thread(
            target=self._pump, daemon=True, name="balance-traffic"
        )

    def snapshot_records(self):
        with self.lock:
            records = []
            for batch in self.batches:
                for record in batch.snapshot_records():
                    record["input_len"] = self.params["input_len"]
                    records.append(record)
            return records

    def persist(self):
        self.path.write_text(
            json.dumps(
                dict(
                    records=self.snapshot_records(),
                    errors=self.errors,
                    pump_done=self.done.is_set(),
                    pump_exit_s=self.exit_s,
                ),
                indent=2,
            )
            + "\n"
        )

    def _one(self):
        from ..backend import RequestBatch
        from ..runtime import Deadline

        if self.cancelled.is_set():
            return
        owner = self

        class ShapedBatch(RequestBatch):
            @property
            def shape(self):
                shape = super().shape
                if owner.params["unique_keys"] and self.entries:
                    rid = self.entries[-1]["record"]["wire_request_id"]
                    shape["block_keys"] = [rid * 100 + j for j in range(3)]
                return shape

            def _consume(self, entry, end):
                super()._consume(
                    entry, min(end, self.ctx.clock() + owner.params["stream_timeout_s"])
                )

        deferred = self.params["defer_batch"] and self.ctx.instance["profile"] in (
            "batch-window",
            "single-batch",
        )
        batch = ShapedBatch(
            self.ctx,
            dict(
                count=1,
                input_len=self.params["input_len"],
                output_len=self.params["output_len"],
                consume="deferred" if deferred else "immediate",
            ),
        )
        batch.artifact = (
            self.ctx.artifact_dir / f"balance-request-{uuid.uuid4().hex}.json"
        )
        with self.lock:
            self.batches.append(batch)
            if self.cancelled.is_set():
                return
        deadline = Deadline(
            self.ctx.instance_deadline_s, self.ctx.clock, self.ctx.sleeper
        )
        try:
            batch.submit(deadline)
            if self.params["await_completion"] and not self.params["defer_batch"]:
                batch.wait(deadline)
        except Exception as exc:
            with self.lock:
                self.errors.append(repr(exc))

    def _pump(self):
        from ..runtime import Deadline

        deadline = Deadline(
            self.ctx.instance_deadline_s, self.ctx.clock, self.ctx.sleeper
        )
        try:
            with ThreadPoolExecutor(max_workers=self.params["concurrency"]) as pool:
                futures = []
                for i in range(self.params["count"]):
                    if self.cancelled.is_set():
                        break
                    deadline.check()
                    future = pool.submit(self._one)
                    futures.append(future)
                    if self.params["concurrency"] == 1:
                        future.result()
                    if i + 1 < self.params["count"] and self.params["interval_s"]:
                        deadline.sleep(self.params["interval_s"])
                for future in futures:
                    future.result()
        except Exception as exc:
            with self.lock:
                self.errors.append(repr(exc))
        finally:
            self.exit_s = self.ctx.clock()
            self.done.set()

    def await_pump(self, deadline):
        while not self.done.is_set():
            self.done.wait(min(0.05, deadline.remaining()))
        self.thread.join(deadline.remaining())
        if self.thread.is_alive() or self.exit_s is None:
            raise RuntimeError("balance pump lacks terminal completion proof")

    def finish(self, deadline):
        self.await_pump(deadline)
        for batch in self.batches:
            try:
                batch.wait(deadline)
            except Exception as exc:
                self.errors.append(repr(exc))
                deadline.check()
        self.persist()

    def cleanup(self, deadline):
        self.cancelled.set()
        with self.lock:
            for batch in self.batches:
                batch.cancel("balance_cleanup")
        try:
            self.await_pump(deadline)
            for batch in self.batches:
                batch.cleanup(deadline)
        finally:
            self.persist()


def _start(ctx, params, deadline):
    traffic = Traffic(ctx, params)
    handle = ctx.register_resource("requests", traffic, traffic.cleanup)
    traffic.thread.start()
    return StageOutput(dict(requests=handle), artifacts=[str(traffic.path)])


def _request_ref(params, plan):
    p = _params(params, plan, {"requests"}, {"requests"})
    plan.reference(p["requests"], "requests")
    return p


def _wait(ctx, params, deadline):
    traffic = ctx.resource(params["requests"], "requests")
    traffic.finish(deadline)
    return StageOutput(
        dict(count=len(traffic.snapshot_records())), artifacts=[str(traffic.path)]
    )


def _pending_validate(params, plan):
    p = _params(params, plan, {"requests", "fleet"}, {"requests", "fleet"})
    plan.reference(p["requests"], "requests")
    plan.reference(p["fleet"], "snapshot")
    return p


def _pending(ctx, params, deadline):
    fleet = ctx.resource(params["fleet"], "snapshot")
    traffic = ctx.resource(params["requests"], "requests")
    addr_names = {row["grpc_addr"]: name for name, row in fleet["engines"].items()}
    observed = []
    while True:
        deadline.check()
        records = traffic.snapshot_records()
        if records:
            row = records[0]
            if row["schedule"]["status"] not in (None, "OK"):
                raise RuntimeError("balance seed Schedule failed")
            name = addr_names.get(row["prefill_addr"])
            if name:
                snap = _fleet(ctx, deadline, "prefill")
                _identity(fleet, snap)
                observed.append(snap)
                engine = snap["engines"][name]
                if engine["waiting"] + engine["running"] >= 1:
                    cool = next(n for n in snap["engines"] if n != name)
                    return StageOutput(
                        dict(hot=name, cool=cool),
                        [
                            CheckResult(
                                "dispatched_pending",
                                "PASS",
                                evidence=dict(
                                    rid=row["wire_request_id"],
                                    engine=name,
                                    owner="mock engine aggregate pending",
                                    per_rid_pending=False,
                                ),
                            )
                        ],
                        [_artifact(ctx, "pending", observed)],
                    )
        deadline.sleep(0.1)


def _pause_validate(params, plan):
    p = _params(params, plan, {"seconds"}, {"seconds"})
    _number(p["seconds"], 0, 120)
    return p


def _pause(ctx, params, deadline):
    deadline.sleep(params["seconds"])
    return StageOutput(dict(elapsed_s=params["seconds"]))


def _empty_validate(params, plan):
    return _params(params, plan, set())


def _decode_load(row):
    present = [key for key in ("inflight_requests", "total_load") if key in row]
    if not present:
        raise ValueError("missing Decode load evidence")
    for key in present:
        _number(row[key])
    # Preserve the legacy compatibility expression: a zero first field does
    # not hide positive total_load when both fields are supplied.
    return row.get("inflight_requests") or row.get("total_load", 0)


def _clean(ctx, params, deadline):
    samples = []
    path = ctx.artifact_dir / f"balance-master-clean-{uuid.uuid4().hex}.json"
    try:
        while True:
            deadline.check()
            request = urllib.request.Request(
                f"http://127.0.0.1:{ctx.ops.master_http_port}/rtp_llm/inflight_status"
            )
            with urllib.request.urlopen(
                request, timeout=min(5, deadline.remaining())
            ) as response:
                data = json.loads(response.read(2000001))
            count = _number(data.get("scheduler_inflight"))
            p_rows, d_rows = data.get("prefill_endpoints"), data.get("decode_endpoints")
            if (
                not isinstance(p_rows, list)
                or not isinstance(d_rows, list)
                or not p_rows
                or not d_rows
            ):
                raise ValueError("missing Master endpoint cleanup evidence")
            p_counts = [_number(row.get("inflight_batches")) for row in p_rows]
            d_counts = [_decode_load(row) for row in d_rows]
            samples.append(dict(time_s=ctx.clock(), raw=data))
            if count == 0 and not any(p_counts + d_counts):
                return StageOutput(
                    {},
                    [
                        CheckResult(
                            "all_owners_zero",
                            "PASS",
                            actual=dict(
                                scheduler=count, prefill=p_counts, decode=d_counts
                            ),
                        )
                    ],
                    [str(path)],
                )
            deadline.sleep(0.5)
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")


def _pressure_validate(params, plan):
    p = _params(params, plan, {"fleet", "target"}, {"fleet", "target"})
    plan.reference(p["fleet"], "snapshot")
    plan.reference(p["target"], "string")
    return p


def _pressure(ctx, params, deadline):
    fleet = ctx.resource(params["fleet"], "snapshot")
    name = ctx.resolve(params["target"])
    row = fleet["engines"][name]
    # These are observed KV capacities, not bounded scenario input knobs.
    # /set_kv_pressure consumes a Java long; preserve the full reported pool.
    max_tokens = (1 << 63) - 1
    total = _number(row.get("available_kv_tokens"), maximum=max_tokens) + _number(
        row.get("active_kv_tokens"), maximum=max_tokens
    )
    _number(total, maximum=max_tokens)
    if total <= 0 or fleet["role"] != "decode":
        raise ValueError("decode KV capacity missing")

    def clear(d):
        _http(ctx.ops, "set_kv_pressure", d, dict(engine=name, active_kv_tokens=0))

    ctx.add_cleanup(f"balance-pressure-{name}", clear)
    response = _http(
        ctx.ops,
        "set_kv_pressure",
        deadline,
        dict(engine=name, active_kv_tokens=int(total)),
    )
    if response.get("status") != "ok" or response.get("engine") != name:
        raise ValueError("KV pressure lacks successful target acknowledgement")
    return StageOutput(
        dict(target=name),
        artifacts=[
            _artifact(
                ctx, "pressure", dict(target=name, tokens=total, response=response)
            )
        ],
    )


def _check_validate(params, plan):
    p = _params(
        params,
        plan,
        {
            "requests",
            "fleet",
            "after",
            "target",
            "baseline",
            "metric",
            "property",
            "grade",
            "bands",
            "relax",
            "min_success",
            "min_workers",
            "allow_admission",
            "diagnostic_reason",
        },
        {"requests", "fleet", "metric", "property"},
    )
    if not isinstance(p["requests"], list) or not p["requests"]:
        raise ValueError("balance checks require explicit request cohorts")
    for ref in p["requests"]:
        plan.reference(ref, "requests")
    plan.reference(p["fleet"], "snapshot")
    for key, kind in (
        ("after", "snapshot"),
        ("target", "string"),
        ("baseline", "requests"),
    ):
        if key in p:
            plan.reference(p[key], kind)
    metrics = {
        "complete",
        "max_share",
        "workers",
        "decode_complete",
        "decode_share",
        "decode_workers",
        "target_delta",
        "takeover",
        "token_share",
        "short_workers",
        "target_share",
        "latency_ratio",
    }
    if "diagnostic_reason" in p and (
        not isinstance(p["diagnostic_reason"], str)
        or not p["diagnostic_reason"].strip()
    ):
        raise ValueError("diagnostic_reason must explain a retired contract")
    if p["metric"] not in metrics or p["property"] not in {
        "P1",
        "P2",
        "P3",
        "P5",
        "P6",
        "P7",
    }:
        raise ValueError("unknown balance metric/property")
    if (
        p["metric"]
        in {
            "decode_complete",
            "decode_share",
            "decode_workers",
            "target_delta",
            "takeover",
        }
        and "after" not in p
    ):
        raise ValueError("decode deltas require before/after snapshots")
    if (
        p["metric"] in {"target_delta", "takeover", "target_share"}
        and "target" not in p
    ):
        raise ValueError("target metric requires target")
    if p["metric"] == "latency_ratio" and "baseline" not in p:
        raise ValueError("latency ratio requires baseline cohort")
    if "grade" in p and p["grade"] not in ("strict", "normal", "loose"):
        raise ValueError("unknown run grade")
    p.setdefault("relax", 0)
    _number(p["relax"], 0, 1, integer=True)
    p.setdefault("min_workers", 2)
    _number(p["min_workers"], 1, 32, integer=True)
    if "min_success" in p:
        _number(p["min_success"], 1, 200, integer=True)
    p.setdefault("allow_admission", False)
    if type(p["allow_admission"]) is not bool:
        raise ValueError("allow_admission must be boolean")
    if "bands" in p:
        if not isinstance(p["bands"], dict) or set(p["bands"]) != {
            "strict",
            "normal",
            "loose",
        }:
            raise ValueError("all three grade bands are required")
        for value in p["bands"].values():
            _number(value)
    return p


def _check(ctx, params, deadline):
    from .elastic import request_success

    cohorts = [ctx.resource(ref, "requests") for ref in params["requests"]]
    records = [r for cohort in cohorts for r in cohort.snapshot_records()]
    if not records:
        raise ValueError("empty balance cohort")
    if any(
        r["consumer_exit_s"] is None or r["transport_terminal_s"] is None
        for r in records
    ):
        raise ValueError("balance check lacks terminal client evidence")
    fleet = ctx.resource(params["fleet"], "snapshot")
    _identity(fleet, _fleet(ctx, deadline, fleet["role"]))
    addr_names = {row["grpc_addr"]: name for name, row in fleet["engines"].items()}
    successes = [r for r in records if request_success(r)]
    failed = [r for r in records if not request_success(r)]
    names = []
    for record in successes:
        name = addr_names.get(record["prefill_addr"])
        if fleet["role"] == "prefill" and name is None:
            raise ValueError("successful request has unknown prefill landing")
        names.append(name)
    dist = Counter(names)
    target = ctx.resolve(params.get("target"))
    metric = params["metric"]
    n = sum(c.params["count"] for c in cohorts)
    if len(records) > n:
        raise ValueError("balance cohort contains more requests than declared")
    deltas = None
    if "after" in params:
        after = ctx.resource(params["after"], "snapshot")
        _identity(fleet, after)
        deltas = {
            name: after["engines"][name]["completed"] - row["completed"]
            for name, row in fleet["engines"].items()
        }
        if any(value < 0 for value in deltas.values()):
            raise ValueError("Decode completed counter decreased")
    value = None
    if metric == "complete":
        allowed = params["allow_admission"]
        expected_count = sum(c.params["count"] for c in cohorts)
        value = (
            len(records) == expected_count
            and len(successes) >= params.get("min_success", expected_count)
            and all(
                allowed and "NO_PREFILL_WORKER" in str(r["schedule"]["error"])
                for r in failed
            )
        )
    elif metric == "max_share":
        value = max(dist.values()) / len(successes) if successes else 1.0
    elif metric == "workers":
        value = len(dist) >= params["min_workers"]
    elif metric == "decode_complete":
        value = len(records) == n and not failed and sum(deltas.values()) >= n
    elif metric == "decode_share":
        value = max(deltas.values()) / n
    elif metric == "decode_workers":
        value = sum(v > 0 for v in deltas.values()) >= params["min_workers"]
    elif metric == "target_delta":
        value = float(deltas[target])
    elif metric == "takeover":
        others = {k: v for k, v in deltas.items() if k != target}
        value = (
            sum(v > 0 for v in others.values()) >= params["min_workers"]
            and sum(others.values()) >= n - deltas[target]
        )
    elif metric == "target_share":
        value = dist[target] / n
    elif metric in ("token_share", "short_workers"):
        tokens, shorts = Counter(), set()
        for record, name in zip(successes, names):
            tokens[name] += record["input_len"]
            if record["input_len"] == 512:
                shorts.add(name)
        value = (
            (max(tokens.values()) / sum(tokens.values()) if tokens else 1.0)
            if metric == "token_share"
            else len(shorts) >= params["min_workers"]
        )
    elif metric == "latency_ratio":
        batch = ctx.instance["profile"] in ("batch-window", "single-batch")
        field = "ended_s" if batch else "first_output_s"

        def duration(r):
            end, start = r["stream"].get(field), r["schedule"].get("started_s")
            return (
                end - start
                if end is not None and start is not None and end > start
                else None
            )

        baseline = ctx.resource(params["baseline"], "requests").snapshot_records()
        denominator = (
            duration(baseline[0])
            if len(baseline) == 1 and request_success(baseline[0])
            else None
        )
        times = [duration(r) for r in successes]
        times = [t for t in times if t is not None]
        if denominator and times:
            value = max(times) / denominator
        else:
            return StageOutput(
                {},
                [
                    CheckResult(
                        "property",
                        "FAIL",
                        "missing baseline/wave timing",
                        evidence=dict(metric=metric, missing_timing=True),
                    )
                ],
            )
    if params.get("diagnostic_reason"):
        evidence = dict(
            metric=metric,
            value=value,
            count=n,
            distribution=dict(dist),
            reason=params["diagnostic_reason"],
        )
        return StageOutput(
            {},
            [
                CheckResult(
                    "property",
                    "SKIP",
                    params["diagnostic_reason"],
                    actual=value,
                    evidence=evidence,
                )
            ],
            [_artifact(ctx, "diagnostic", evidence)],
        )
    report = GradeReport(
        run_grade=params.get("grade", ctx.instance.get("grade", "normal"))
    )
    if params["property"] in ("P2", "P6"):
        if type(value) is not bool:
            raise ValueError("invariant metric must be boolean")
        report.invariant(params["property"], value)
    else:
        if type(value) not in (int, float):
            raise ValueError("graded metric must be numeric")
        report.check(
            params["property"], value, bands=params.get("bands"), relax=params["relax"]
        )
    evidence = dict(
        metric=metric,
        count=n,
        successful=len(successes),
        failed=[r["wire_request_id"] for r in failed],
        distribution=dict(dist),
        decode_deltas=deltas,
        grade_report=report.to_dict(),
    )
    return StageOutput(
        {},
        [
            CheckResult(
                "property",
                "PASS" if report.passed else "FAIL",
                actual=value,
                evidence=evidence,
            )
        ],
        [_artifact(ctx, "check", evidence)],
    )


HANDLERS = [
    StageHandler(
        "balance_snapshot",
        _snapshot_validate,
        _snapshot,
        {"snapshot": "snapshot", "first": "string", "second": "string"},
    ),
    StageHandler("balance_start", _traffic_validate, _start, {"requests": "requests"}),
    StageHandler("balance_wait", _request_ref, _wait, {"count": "integer"}),
    StageHandler(
        "balance_pending",
        _pending_validate,
        _pending,
        {"hot": "string", "cool": "string"},
        checks=frozenset({"dispatched_pending"}),
    ),
    StageHandler("balance_pause", _pause_validate, _pause, {"elapsed_s": "number"}),
    StageHandler(
        "balance_pressure", _pressure_validate, _pressure, {"target": "string"}
    ),
    StageHandler(
        "balance_clean",
        _empty_validate,
        _clean,
        {},
        checks=frozenset({"all_owners_zero"}),
    ),
    StageHandler(
        "balance_check", _check_validate, _check, {}, checks=frozenset({"property"})
    ),
]
