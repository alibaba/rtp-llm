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

    def stop(self, deadline, cancel=False):
        self._stop.set()
        if cancel:
            self.cancel_active()
        if self.thread.ident is not None:
            self.thread.join(max(0, deadline.remaining()))
            if self.thread.is_alive():
                self.cancel_active("drain_deadline")
                raise TimeoutError(
                    "flow drain budget expired; requests cancelled, not completed"
                )
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


def _seed(ctx, params, deadline):
    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    handle = ctx.register_resource(
        "requests", records, cleanup=lambda d: records.cancel_active()
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
    handle = ctx.register_resource(
        "flow", flow, cleanup=lambda d: flow.stop(d, cancel=True)
    )
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
    return StageOutput(
        output={"complete": result["result_complete"], "issued": result["issued"]},
        checks=[
            CheckResult(
                "P6",
                (
                    "PASS"
                    if result["zero_errors"] and result["result_complete"]
                    else "FAIL"
                ),
                actual=result,
            )
        ],
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
    evidence = dict(
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
            CheckResult(
                "drained",
                "PASS" if response.get("drained") is True else "FAIL",
                evidence=evidence,
            ),
        ],
    )


def _recovery_validate(params, plan):
    return _validate(params, plan, set())


def _recovery(ctx, params, deadline):
    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    handle = ctx.register_resource(
        "requests", records, cleanup=lambda d: records.cancel_active()
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
    summary = completeness(records.snapshot_records())
    rate = summary["completed"] / 20
    return StageOutput(
        output={"requests": handle, "success_rate": rate},
        checks=[
            CheckResult(
                "P2",
                "PASS" if rate >= 0.95 and summary["result_complete"] else "FAIL",
                actual=rate,
                expected=0.95,
                evidence=summary,
            )
        ],
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
        {"complete": "boolean", "issued": "integer"},
        checks=frozenset({"P6"}),
    ),
    StageHandler(
        "elastic_scale",
        _scale_validate,
        _scale,
        {"scale": "snapshot", "drained": "boolean"},
        checks=frozenset({"pre_scale_skew", "drained"}),
    ),
    StageHandler(
        "elastic_recovery",
        _recovery_validate,
        _recovery,
        {"requests": "requests", "success_rate": "number"},
        checks=frozenset({"P2"}),
    ),
]
