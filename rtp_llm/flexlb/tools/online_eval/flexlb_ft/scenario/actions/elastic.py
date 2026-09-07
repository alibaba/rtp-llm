"""Bounded elastic scenarios with explicit client-side request evidence.

Transport termination and client cancellation are never business completion.
The legacy Python cases remain the compatibility reference during migration.
"""

from __future__ import annotations

import copy
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..contracts import CheckResult, StageHandler, StageOutput


class ClientRecords:
    """Thread-safe observation contract shared with the observation adapter."""

    def __init__(self, env_epoch):
        self.env_epoch = env_epoch
        self._lock = threading.RLock()
        self._records = []

    def snapshot_records(self):
        with self._lock:
            return copy.deepcopy(self._records)

    def snapshot_cohort(self, start_s, end_s, basis="issued"):
        if basis not in {"issued", "submitted", "terminal"}:
            raise ValueError("unknown cohort basis")
        if end_s < start_s:
            raise ValueError("inverted cohort window")
        records = self.snapshot_records()

        def timestamp(record):
            if basis == "submitted":
                return record["schedule"]["started_s"]
            return record["issued_s" if basis == "issued" else "transport_terminal_s"]

        selected = [
            r
            for r in records
            if timestamp(r) is not None and start_s <= timestamp(r) < end_s
        ]
        return dict(
            records=selected,
            record_count=len(selected),
            basis=basis,
            window=[start_s, end_s],
            env_epoch=self.env_epoch,
        )

    def issue(self, rid, clock):
        rpc = dict(
            method=None,
            started_s=None,
            ended_s=None,
            deadline_s=None,
            status=None,
            error=None,
        )
        record = dict(
            schema_version=1,
            wire_request_id=rid,
            attempt=1,
            env_epoch=self.env_epoch,
            endpoint_generation=None,
            issued_s=clock(),
            schedule=dict(rpc, method="Schedule"),
            stream=dict(rpc, first_output_s=None),
            business_finished=False,
            business_error_code=None,
            business_error_message=None,
            transport_terminal_s=None,
            consumer_exit_s=None,
            cancel=dict(
                scope="transport",
                requested_s=None,
                reason=None,
                acknowledged=None,
                ended_s=None,
                error=None,
            ),
            prefill_addr=None,
        )
        with self._lock:
            self._records.append(record)
        return record

    def update(self, record, **fields):
        with self._lock:
            for key, value in fields.items():
                if isinstance(value, dict):
                    record[key].update(value)
                else:
                    record[key] = value


def request_success(record):
    return (
        record["business_finished"] is True
        and record["business_error_code"] in (None, 0)
        and record["schedule"]["status"] == "OK"
        and record["stream"]["status"] == "OK"
        and record["cancel"]["requested_s"] is None
    )


def completeness(records):
    """All issued requests remain accountable, including incomplete/cancelled ones."""
    failures = [r["wire_request_id"] for r in records if not request_success(r)]
    missing = [r["wire_request_id"] for r in records if r["consumer_exit_s"] is None]
    return dict(
        issued=len(records),
        sample_count=len(records),
        min_samples=1,
        complete=bool(records) and not missing,
        completed=sum(request_success(r) for r in records),
        incomplete_request_ids=missing,
        failed_request_ids=failures,
        result_complete=bool(records) and not missing,
        zero_errors=bool(records) and not failures,
    )


class RecordedRequests(ClientRecords):
    def __init__(self, ops, env_epoch, clock=time.monotonic):
        super().__init__(env_epoch)
        self.ops, self.clock = ops, clock
        self._calls = {}
        self._cancelled = False
        self._cancelled_calls = set()

    def _activate(self, record, call):
        with self._lock:
            self._calls[record["wire_request_id"]] = (record, call)
            if self._cancelled:
                self._cancel_call(record, call, "cleanup")

    def _cancel_call(self, record, call, reason):
        identity = (record["wire_request_id"], id(call))
        if identity in self._cancelled_calls:
            return
        self._cancelled_calls.add(identity)
        if record["cancel"]["requested_s"] is None:
            self.update(record, cancel=dict(requested_s=self.clock(), reason=reason))
        try:
            acknowledged = bool(call.cancel())
            self.update(record, cancel=dict(acknowledged=acknowledged))
        except Exception as exc:
            self.update(record, cancel=dict(error=repr(exc)))
        finally:
            self.update(record, cancel=dict(ended_s=self.clock()))

    def cancel_active(self, reason="cleanup"):
        with self._lock:
            self._cancelled = True
            for record, call in list(self._calls.values()):
                self._cancel_call(record, call, reason)

    def run(self, record, shape, timeout_s=90.0):
        ops, clock = self.ops, self.clock
        end = clock() + timeout_s
        phase = "schedule"

        def remaining(cap):
            value = min(cap, end - clock())
            if value <= 0:
                raise TimeoutError("request budget expired")
            return value

        try:
            limit = remaining(30.0)
            self.update(
                record, schedule=dict(started_s=clock(), deadline_s=clock() + limit)
            )
            stub = ops.schedule_pb2_grpc.FlexlbServiceStub(
                ops._channel(ops.master_target())
            )
            call = stub.Schedule.future(
                ops.build_schedule_request(record["wire_request_id"], **shape),
                timeout=limit,
            )
            self._activate(record, call)
            response = call.result()
            self.update(record, schedule=dict(ended_s=clock(), status="OK"))
            if response.code != 200 or not response.success:
                self.update(
                    record,
                    schedule=dict(status="REJECTED", error=response.error_message),
                )
                return
            target = ops.prefill_addr(response)
            self.update(record, prefill_addr=target)
            if not target:
                raise RuntimeError("Schedule returned no prefill address")
            phase = "stream"
            method = (
                "FetchResponse" if response.enqueued_by_master else "GenerateStreamCall"
            )
            limit = remaining(60.0)
            self.update(
                record,
                stream=dict(
                    method=method, started_s=clock(), deadline_s=clock() + limit
                ),
            )
            stub = ops.pb2_grpc.RpcServiceStub(ops._channel(target))
            if response.enqueued_by_master:
                call = stub.FetchResponse(
                    ops.pb2.FetchRequestPB(request_id=record["wire_request_id"]),
                    timeout=limit,
                )
            else:
                inp = ops.build_generate_input(record["wire_request_id"], **shape)
                ops._copy_role_addrs(inp, response)
                call = stub.GenerateStreamCall(inp, timeout=limit)
            self._activate(record, call)
            for output in call:
                if record["stream"]["first_output_s"] is None:
                    self.update(record, stream=dict(first_output_s=clock()))
                if output.HasField("error_info"):
                    self.update(
                        record,
                        business_error_code=int(output.error_info.error_code),
                        business_error_message=output.error_info.error_message,
                    )
                if any(output.flatten_output.finished):
                    self.update(record, business_finished=True)
            self.update(record, stream=dict(status="OK", ended_s=clock()))
        except Exception as exc:
            code = getattr(exc, "code", lambda: None)()
            self.update(
                record,
                **{
                    phase: dict(
                        status=getattr(code, "name", "ERROR"),
                        error=repr(exc),
                        ended_s=clock(),
                    )
                },
            )
        finally:
            self.update(record, transport_terminal_s=clock(), consumer_exit_s=clock())
            with self._lock:
                self._calls.pop(record["wire_request_id"], None)


class BoundedFlow(RecordedRequests):
    """Fixed concurrency and cadence; stop issuance separately from cancellation."""

    def __init__(
        self,
        ops,
        env_epoch,
        families,
        interval_s=0.5,
        max_inflight=2,
        clock=time.monotonic,
    ):
        super().__init__(ops, env_epoch, clock)
        self.families = copy.deepcopy(families)
        self.interval_s, self.max_inflight = interval_s, max_inflight
        self._stop = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(
            target=self._pump, name="elastic-recorded-flow", daemon=True
        )
        self.pump_error = None

    def start(self):
        self.thread.start()

    def _pump(self):
        futures = []
        index = 0
        try:
            with ThreadPoolExecutor(max_workers=self.max_inflight) as pool:
                while not self._stop.is_set():
                    futures = [f for f in futures if not f.done()]
                    if len(futures) < self.max_inflight:
                        record = self.issue(self.ops.next_request_id(), self.clock)
                        keys = self.families[index % len(self.families)]
                        index += 1
                        futures.append(
                            pool.submit(
                                self.run,
                                record,
                                dict(input_len=10240, output_len=2, block_keys=keys),
                            )
                        )
                    self._stop.wait(self.interval_s)
        except Exception as exc:
            self.pump_error = repr(exc)
        finally:
            # Executor shutdown has completed and all worker records are final.
            self.done.set()

    def stop(self, deadline, cancel=False):
        self._stop.set()
        if cancel:
            self.cancel_active()
        if self.thread.ident is not None:
            try:
                if not self.done.wait(max(0, deadline.remaining())):
                    raise TimeoutError("flow completion event did not arrive")
            except TimeoutError:
                self.cancel_active("drain_deadline")
                raise
            records = self.snapshot_records()
            if any(
                r["consumer_exit_s"] is None or r["transport_terminal_s"] is None
                for r in records
            ):
                raise RuntimeError("flow completed without final consumer records")
        return completeness(self.snapshot_records())


def _http(ops, path, deadline, body=None):
    """Use the stage budget in the underlying HTTP call, without fixed 95s waits."""
    deadline.check()
    request = urllib.request.Request(
        f"http://127.0.0.1:{ops.mock_http_port}/{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=min(65.0, deadline.remaining())
    ) as response:
        return json.load(response)


def _snapshot(ctx, deadline):
    return {e["name"]: e for e in _http(ctx.ops, "snapshot", deadline)["engines"]}


def _validate(params, plan, fields, required=()):
    if (
        not isinstance(params, dict)
        or set(params) - set(fields)
        or set(required) - set(params)
    ):
        raise ValueError(f"{plan.path}: invalid elastic parameters")
    return copy.deepcopy(params)


def _seed_validate(params, plan):
    p = _validate(params, plan, {"hot", "cold"})
    p.setdefault("hot", "prefill-0")
    p.setdefault("cold", "prefill-1")
    if (
        any(not isinstance(p[k], str) or not p[k] for k in ("hot", "cold"))
        or p["hot"] == p["cold"]
    ):
        raise ValueError(f"{plan.path}: distinct engine names required")
    return p


def _record_cleanup(ctx, records, name):
    def cleanup(deadline):
        records.cancel_active("cleanup")
        (ctx.artifact_dir / name).write_text(
            json.dumps(records.snapshot_records(), indent=2)
        )

    return cleanup


def _seed(ctx, params, deadline):
    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    handle = ctx.register_resource(
        "requests",
        records,
        cleanup=_record_cleanup(ctx, records, "elastic-seed-records.json"),
    )
    snapshot = _snapshot(ctx, deadline)
    hot, cold = params["hot"], params["cold"]
    if hot not in snapshot or cold not in snapshot:
        raise ValueError("seed requires two live prefill engines")
    addr_names = {e["grpc_addr"]: n for n, e in snapshot.items()}
    selected = {hot: [], cold: []}
    wanted = {hot: 9, cold: 1}
    base = ctx.ops.next_request_id() * 100_000
    for trial in range(40):
        deadline.check()
        keys = [base + trial * 1000 + j for j in range(10)]
        record = records.issue(ctx.ops.next_request_id(), ctx.clock)
        records.run(
            record,
            dict(input_len=10240, output_len=2, block_keys=keys),
            deadline.remaining(),
        )
        deadline.check()
        name = addr_names.get(record["prefill_addr"])
        if not request_success(record) or name not in selected:
            return StageOutput(
                output={"requests": handle},
                checks=[CheckResult("seed_success", "FAIL", evidence=record)],
            )
        if len(selected[name]) < wanted[name]:
            selected[name].append(keys)
        if all(len(selected[n]) == wanted[n] for n in selected):
            break
    if any(len(selected[n]) != wanted[n] for n in selected):
        return StageOutput(
            output={"requests": handle},
            checks=[
                CheckResult("seed_success", "FAIL", "bounded seed placement incomplete")
            ],
        )
    snapshot = _snapshot(ctx, deadline)
    for name in selected:
        keep = {key for family in selected[name] for key in family}
        held = set(snapshot[name]["cache_key_set"])
        if not keep <= held:
            return StageOutput(
                output={"requests": handle},
                checks=[
                    CheckResult(
                        "seed_success", "FAIL", "selected keys missing from engine"
                    )
                ],
            )
        _http(
            ctx.ops,
            "cache_evict",
            deadline,
            {"engine": name, "keys": sorted(held - keep)},
        )
    # Quiet engine cache is a construction check, not proof of Master index convergence.
    deadline.sleep(3.5)
    snapshot = _snapshot(ctx, deadline)
    counts = {n: len(snapshot[n]["cache_key_set"]) for n in selected}
    skew_ok = counts[cold] > 0 and counts[hot] >= 3 * counts[cold]
    families = dict(
        families=selected[hot] + selected[cold],
        hot=hot,
        cold=cold,
        hot_share=0.9,
        counts=counts,
        master_index_converged=None,
    )
    family_handle = ctx.register_resource("snapshot", families, historical=True)
    return StageOutput(
        output={"requests": handle, "families": family_handle},
        checks=[
            CheckResult(
                "seed_success", "PASS", actual=completeness(records.snapshot_records())
            ),
            CheckResult(
                "skew",
                "PASS" if skew_ok else "FAIL",
                actual=counts,
                expected="hot >= 3*cold > 0",
            ),
        ],
    )


def _flow_validate(params, plan):
    p = _validate(
        params, plan, {"families", "interval_s", "max_inflight"}, {"families"}
    )
    plan.reference(p["families"], "snapshot")
    p.setdefault("interval_s", 0.5)
    p.setdefault("max_inflight", 2)
    # This scenario's PC budget derives from this exact uniform workload.
    if (
        p["interval_s"] != 0.5
        or type(p["max_inflight"]) is not int
        or p["max_inflight"] != 2
    ):
        raise ValueError(
            f"{plan.path}: calibrated pilot requires .5s cadence and two inflight requests"
        )
    return p


def _flow_start(ctx, params, deadline):
    families = ctx.resource(params["families"], "snapshot")
    flow = BoundedFlow(ctx.ops, ctx.env_epoch, families["families"], clock=ctx.clock)

    def cleanup(d):
        try:
            flow.stop(d, cancel=True)
        finally:
            (ctx.artifact_dir / "elastic-client-records.json").write_text(
                json.dumps(flow.snapshot_records(), indent=2)
            )

    handle = ctx.register_resource("flow", flow, cleanup=cleanup)
    deadline.check()
    flow.start()
    return StageOutput(output={"flow": handle})


def _flow_stop_validate(params, plan):
    p = _validate(params, plan, {"flow"}, {"flow"})
    plan.reference(p["flow"], "flow")
    return p


def _flow_stop(ctx, params, deadline):
    flow = ctx.resource(params["flow"], "flow")
    result = flow.stop(deadline)
    if flow.pump_error:
        raise RuntimeError(flow.pump_error)
    path = ctx.artifact_dir / "elastic-client-records.json"
    path.write_text(json.dumps(flow.snapshot_records(), indent=2))
    evidence = ctx.register_resource("snapshot", result, historical=True)
    return StageOutput(
        output={
            "complete": result["result_complete"],
            "issued": result["issued"],
            "result": evidence,
        },
        artifacts=[str(path)],
    )


def _scale_validate(params, plan):
    p = _validate(
        params, plan, {"families", "victim", "drain_timeout_ms"}, {"families", "victim"}
    )
    plan.reference(p["families"], "snapshot")
    if p["victim"] not in {"hot", "cold"}:
        raise ValueError(f"{plan.path}: victim must be hot or cold")
    p.setdefault("drain_timeout_ms", 60000)
    if type(p["drain_timeout_ms"]) is not int or p["drain_timeout_ms"] != 60000:
        raise ValueError(
            f"{plan.path}: graceful pilot preserves the 60000ms drain contract"
        )
    return p


def _scale(ctx, params, deadline):
    families = ctx.resource(params["families"], "snapshot")
    before = _snapshot(ctx, deadline)
    counts = {
        n: len(before[n]["cache_key_set"]) for n in (families["hot"], families["cold"])
    }
    if not (
        counts[families["cold"]] > 0
        and counts[families["hot"]] >= 3 * counts[families["cold"]]
    ):
        return StageOutput(
            checks=[CheckResult("pre_scale_skew", "FAIL", actual=counts)]
        )
    victim = families[params["victim"]]
    # Do not issue a 60s server operation with a shorter client stage budget.
    if deadline.remaining() < 65:
        raise TimeoutError("scale requires 65s remaining for bounded server drain")
    started = ctx.clock()
    response = _http(
        ctx.ops,
        "remove_engine",
        deadline,
        dict(engine=victim, mode="graceful", drain_timeout_ms=60000),
    )
    if not isinstance(response, dict) or type(response.get("drained")) is not bool:
        raise ValueError("remove_engine response lacks boolean drained evidence")
    evidence = dict(
        complete=True,
        sample_count=1,
        min_samples=1,
        response=response,
        started_s=started,
        ended_s=ctx.clock(),
        victim=victim,
        before_counts=counts,
    )
    handle = ctx.register_resource("snapshot", evidence, historical=True)
    return StageOutput(
        output={"scale": handle, "drained": response.get("drained") is True},
        checks=[
            CheckResult("pre_scale_skew", "PASS", actual=counts),
        ],
    )


def _recovery_validate(params, plan):
    return _validate(params, plan, set())


def _recovery(ctx, params, deadline):
    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    handle = ctx.register_resource(
        "requests",
        records,
        cleanup=_record_cleanup(ctx, records, "elastic-recovery-records.json"),
    )
    for _ in range(20):
        deadline.check()
        record = records.issue(ctx.ops.next_request_id(), ctx.clock)
        records.run(
            record,
            dict(
                input_len=2048,
                output_len=2,
                block_keys=[record["wire_request_id"] * 100],
            ),
            deadline.remaining(),
        )
    deadline.check()
    summary = completeness(records.snapshot_records())
    rate = summary["completed"] / 20
    return StageOutput(
        output={"requests": handle, "success_rate": rate},
    )


HANDLERS = [
    StageHandler(
        "elastic_seed",
        _seed_validate,
        _seed,
        {"requests": "requests", "families": "snapshot"},
        checks=frozenset({"seed_success", "skew"}),
    ),
    StageHandler("elastic_flow_start", _flow_validate, _flow_start, {"flow": "flow"}),
    StageHandler(
        "elastic_flow_stop",
        _flow_stop_validate,
        _flow_stop,
        {"complete": "boolean", "issued": "integer", "result": "snapshot"},
    ),
    StageHandler(
        "elastic_scale",
        _scale_validate,
        _scale,
        {"scale": "snapshot", "drained": "boolean"},
        checks=frozenset({"pre_scale_skew"}),
    ),
    StageHandler(
        "elastic_recovery",
        _recovery_validate,
        _recovery,
        {"requests": "requests", "success_rate": "number"},
    ),
]


METRICS = {
    "mock_engine_cache_key_hits_total",
    "mock_engine_cache_keys_requested_total",
    "mock_engine_waiting",
    "mock_engine_available_blocks",
    "mock_engine_cache_blocks",
    "rtp_llm_context_tps",
    "rtp_llm_generate_tps",
}


def parse_metrics(text):
    import re

    values = {}
    for line in text.splitlines():
        match = re.fullmatch(
            r"([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+([^\s]+)(?:\s+\S+)?", line.strip()
        )
        if not match or match[1] not in METRICS:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', match[2]))
        if not labels.get("engine_name"):
            raise ValueError("metric has no engine_name")
        value = float(match[3])
        if not __import__("math").isfinite(value):
            raise ValueError("nonfinite metric")
        engine = values.setdefault(labels["engine_name"], {"role": labels.get("role")})
        if match[1] in engine:
            raise ValueError("ambiguous duplicate metric series")
        engine[match[1]] = value
    if not values:
        raise ValueError("empty metric response")
    return values


class ElasticMetrics:
    """Continuous, bounded sampling across a blocking graceful scale call."""

    def __init__(self, ctx, max_duration_s=600):
        self.ctx, self.env_epoch = ctx, ctx.env_epoch
        self.end = ctx.clock() + max_duration_s
        self.samples, self.errors = [], []
        self._lock, self._stop = threading.Lock(), threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="elastic-metrics"
        )

    def _run(self):
        try:
            self._sample_loop()
        except Exception as exc:
            with self._lock:
                self.errors.append(dict(time_s=self.ctx.clock(), error=repr(exc)))
        finally:
            self.done.set()

    def _sample_loop(self):
        while not self._stop.is_set():
            now = self.ctx.clock()
            if now >= self.end:
                with self._lock:
                    self.errors.append(
                        dict(time_s=now, error="metrics duration budget exceeded")
                    )
                return
            try:
                if self.ctx.env_epoch != self.env_epoch:
                    raise ValueError("environment epoch changed during measurement")
                url = f"http://127.0.0.1:{self.ctx.ops.mock_http_port}/metrics?per_engine=true"
                with urllib.request.urlopen(
                    url, timeout=min(2, self.end - now)
                ) as response:
                    body = response.read(2_000_001)
                    if len(body) > 2_000_000:
                        raise ValueError("metrics response exceeds byte budget")
                    values = parse_metrics(body.decode())
                with self._lock:
                    self.samples.append(dict(time_s=self.ctx.clock(), engines=values))
            except Exception as exc:
                with self._lock:
                    self.errors.append(dict(time_s=self.ctx.clock(), error=repr(exc)))
            self._stop.wait(1)

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(
                dict(samples=self.samples, errors=self.errors, env_epoch=self.env_epoch)
            )

    def stop(self, deadline):
        self._stop.set()
        try:
            if self.thread.ident is not None and not self.done.wait(
                max(0, deadline.remaining())
            ):
                raise TimeoutError("metrics completion event did not arrive")
        finally:
            path = self.ctx.artifact_dir / "elastic-metrics.json"
            evidence = self.snapshot()
            evidence["complete"] = self.done.is_set() or self.thread.ident is None
            path.write_text(json.dumps(evidence, indent=2))
        if self.errors:
            raise ValueError(f"metric acquisition errors: {self.errors}")


def metric_window(data, start, end, survivor=None):
    samples = [s for s in data["samples"] if start <= s["time_s"] <= end]
    errors = [e for e in data["errors"] if start <= e["time_s"] <= end]
    if errors:
        raise ValueError(f"metric acquisition failed: {errors}")
    if len(samples) < max(2, int((end - start) / 2)):
        raise ValueError("insufficient metric samples")
    stamps = [start] + [s["time_s"] for s in samples] + [end]
    if any(b - a > 3.5 for a, b in zip(stamps, stamps[1:])):
        raise ValueError("metric window has an uncovered gap")
    by_engine = {}
    for sample in samples:
        for name, values in sample["engines"].items():
            by_engine.setdefault(name, []).append(values)
    hits = requested = 0
    for name, rows in by_engine.items():
        for metric in (
            "mock_engine_cache_key_hits_total",
            "mock_engine_cache_keys_requested_total",
        ):
            if any(metric not in row for row in rows):
                raise ValueError(f"{name}: missing hit counter")
            counters = [r[metric] for r in rows]
            if any(b < a for a, b in zip(counters, counters[1:])):
                raise ValueError(f"{name}: counter epoch reset")
            delta = counters[-1] - counters[0]
            if metric.endswith("hits_total"):
                hits += delta
            else:
                requested += delta
    if requested <= 0 or hits < 0 or hits > requested:
        raise ValueError("nonzero valid cache-request denominator required")
    result = dict(
        start_s=start,
        end_s=end,
        sample_count=len(samples),
        min_samples=max(2, int((end - start) / 2)),
        complete=True,
        hit_rate=hits / requested,
        hits=hits,
        requested=requested,
        membership=sorted(by_engine),
        membership_policy="union of observed engines; removed engines retain their last counter",
    )
    tps = 0
    tps_complete = True
    for rows in by_engine.values():
        metric = (
            "rtp_llm_context_tps"
            if rows[0].get("role", "").lower() == "prefill"
            else "rtp_llm_generate_tps"
        )
        values = [r.get(metric) for r in rows]
        if any(v is None for v in values):
            tps_complete = False
        else:
            tps += sum(values) / len(values)
    result["tps_observation"] = tps if tps_complete else None
    if survivor:
        if any(survivor not in s["engines"] for s in samples):
            raise ValueError("survivor missing from steady samples")
        waiting, occupancy = [], []
        for sample in samples:
            values = sample["engines"][survivor]
            total = values["mock_engine_cache_blocks"]
            available = values["mock_engine_available_blocks"]
            if total <= 0 or not 0 <= available <= total:
                raise ValueError("invalid KV block gauge")
            waiting.append(values["mock_engine_waiting"])
            occupancy.append(1 - available / total)
        result.update(waiting_peak=max(waiting), occupancy_peak=max(occupancy))
    return result


def _metrics_start_validate(params, plan):
    return _validate(params, plan, set())


def _metrics_start(ctx, params, deadline):
    metrics = ElasticMetrics(ctx)
    handle = ctx.register_resource("observation", metrics, cleanup=metrics.stop)
    deadline.check()
    metrics.thread.start()
    return StageOutput(output={"observation": handle})


def _metric_phase_validate(params, plan):
    p = _validate(
        params,
        plan,
        {"observation", "phase", "baseline", "scale", "families", "victim"},
        {"observation", "phase"},
    )
    plan.reference(p["observation"], "observation")
    if p["phase"] not in {"baseline", "transient", "steady"}:
        raise ValueError("unknown elastic metric phase")
    if p["phase"] != "baseline":
        for field in ("baseline", "scale", "families"):
            plan.reference(p[field], "snapshot")
        if p.get("victim") not in {"hot", "cold"}:
            raise ValueError("victim must be hot or cold")
    return p


def _baseline_validate(params, plan):
    p = _metric_phase_validate(params, plan)
    if p["phase"] != "baseline":
        raise ValueError("elastic_baseline requires phase baseline")
    return p


def _window_validate(params, plan):
    p = _metric_phase_validate(params, plan)
    if p["phase"] == "baseline":
        raise ValueError("baseline requires elastic_baseline high-hit guard")
    return p


def _window(ctx, params, deadline):
    metrics = ctx.resource(params["observation"], "observation")
    phase = params["phase"]
    if phase == "baseline":
        start, duration = ctx.clock(), 20
    elif phase == "transient":
        start = ctx.resource(params["scale"], "snapshot")["started_s"]
        duration = 20
    else:
        # Wait for the single survivor topology before starting the 60s settle window.
        from ...harness import http_post_json

        topology_deadline = ctx.clock() + 30
        while True:
            deadline.check()
            if ctx.clock() >= topology_deadline:
                raise TimeoutError(
                    "Master prefill topology did not converge within 30s"
                )
            status, body = http_post_json(
                f"http://127.0.0.1:{ctx.ops.master_http_port}/rtp_llm/master/info",
                {},
                timeout=min(2, deadline.remaining()),
            )
            if status != 200 or not isinstance(body, dict):
                raise ValueError(f"master topology probe failed: {status}")
            summary = body.get("worker_summary", {}).get("PREFILL", {})
            if summary.get("discovered") == 1 and summary.get("alive") == 1:
                break
            deadline.sleep(0.5)
        start, duration = ctx.clock(), 60
    wait = start + duration - ctx.clock()
    if wait > 0:
        deadline.sleep(wait)
    survivor = None
    if phase == "steady":
        families = ctx.resource(params["families"], "snapshot")
        survivor = families["cold" if params["victim"] == "hot" else "hot"]
        start += 40
    evidence = metric_window(
        metrics.snapshot(),
        start,
        start + (20 if phase == "steady" else duration),
        survivor,
    )
    if phase == "steady":
        base = ctx.resource(params["baseline"], "snapshot")["hit_rate"]
        removed_s = ctx.resource(params["scale"], "snapshot")["started_s"]
        raw = metrics.snapshot()
        rolling = []
        anchor = removed_s
        while anchor + 10 <= start + 20:
            try:
                rate = metric_window(raw, anchor, anchor + 10)["hit_rate"]
                rolling.append(dict(start_s=anchor, hit_rate=rate))
            except ValueError as exc:
                rolling.append(
                    dict(start_s=anchor, hit_rate=None, unavailable=str(exc))
                )
            anchor += 2
        available = [r for r in rolling if r["hit_rate"] is not None]
        trough = min((r["hit_rate"] for r in available), default=None)
        recovered = next(
            (
                r["start_s"] - removed_s
                for r in available
                if r["hit_rate"] >= base - 0.15
            ),
            None,
        )
        evidence["observations"] = dict(
            rolling_10s=rolling,
            recovery_duration_s=recovered,
            baseline_hit=base,
            trough=trough,
            gates=False,
            rebound_floor=None if trough is None else trough + 0.5 * (base - trough),
            steady_base_floor=base - 0.15,
        )
    handle = ctx.register_resource("snapshot", evidence, historical=True)
    checks = []
    if phase == "baseline":
        # Nine hot and one cold family are already warm; actual request counters,
        # not holder counts, must establish high hit rate before any retirement.
        checks.append(
            CheckResult(
                "high_hit",
                "PASS" if evidence["hit_rate"] >= 0.9 else "FAIL",
                actual=evidence["hit_rate"],
                expected=0.9,
                evidence=evidence,
            )
        )
    return StageOutput(output={"window": handle}, checks=checks)


HANDLERS += [
    StageHandler(
        "elastic_metrics_start",
        _metrics_start_validate,
        _metrics_start,
        {"observation": "observation"},
    ),
    StageHandler(
        "elastic_baseline",
        _baseline_validate,
        _window,
        {"window": "snapshot"},
        checks=frozenset({"high_hit"}),
    ),
    StageHandler(
        "elastic_window",
        _window_validate,
        _window,
        {"window": "snapshot"},
    ),
]


def _verdict_validate(params, plan):
    fields = {
        "baseline",
        "transient",
        "steady",
        "scale",
        "flow_result",
        "recovery",
        "victim",
    }
    p = _validate(params, plan, fields, fields)
    for field in fields - {"victim", "recovery"}:
        plan.reference(p[field], "snapshot")
    plan.reference(p["recovery"], "requests")
    if p["victim"] not in {"hot", "cold"}:
        raise ValueError("victim must be hot or cold")
    return p


def _verdict(ctx, params, deadline):
    deadline.check()
    evidence = {
        key: ctx.resource(params[key], "snapshot")
        for key in ("baseline", "transient", "steady", "scale", "flow_result")
    }
    recovery = completeness(
        ctx.resource(params["recovery"], "requests").snapshot_records()
    )
    base, transient, steady, scale, flow = (
        evidence[k] for k in ("baseline", "transient", "steady", "scale", "flow_result")
    )
    floor = base["hit_rate"] - (
        0.9 * base["hit_rate"] + 0.1 if params["victim"] == "hot" else 0.1
    )
    rate = recovery["completed"] / 20
    recovery["min_samples"] = 20
    recovery["complete"] = recovery["result_complete"] and recovery["issued"] == 20
    checks = [
        CheckResult(
            "drained",
            "PASS" if scale["response"].get("drained") is True else "FAIL",
            evidence=scale,
        ),
        CheckResult(
            "PC",
            "PASS" if transient["hit_rate"] >= floor else "FAIL",
            actual=transient["hit_rate"],
            expected=floor,
            evidence=transient,
        ),
        CheckResult(
            "PQ",
            "PASS" if steady["waiting_peak"] <= 2 else "FAIL",
            actual=steady["waiting_peak"],
            expected=2,
            evidence=steady,
        ),
        CheckResult(
            "PK",
            "PASS" if steady["occupancy_peak"] <= 0.95 else "FAIL",
            actual=steady["occupancy_peak"],
            expected=0.95,
            evidence=steady,
        ),
        CheckResult(
            "P6",
            "PASS" if flow["zero_errors"] and flow["result_complete"] else "FAIL",
            actual=flow,
            evidence=flow,
        ),
        CheckResult(
            "P2",
            (
                "PASS"
                if recovery["issued"] == 20
                and recovery["result_complete"]
                and rate >= 0.95
                else "FAIL"
            ),
            actual=rate,
            expected=0.95,
            evidence=recovery,
        ),
    ]
    evidence["recovery"] = recovery
    evidence["observations"] = {
        "steady_hit": steady["hit_rate"],
        "baseline_hit": base["hit_rate"],
        "P1": "degenerate with one prefill",
        "PK_spread": "degenerate with one prefill",
    }
    path = ctx.artifact_dir / "elastic-verdict.json"
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(checks=checks, artifacts=[str(path)])


HANDLERS.append(
    StageHandler(
        "elastic_verdict",
        _verdict_validate,
        _verdict,
        {},
        checks=frozenset({"drained", "PC", "PQ", "PK", "P6", "P2"}),
    )
)
