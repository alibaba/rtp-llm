"""Bounded elastic scenarios with explicit client-side request evidence.

Transport termination and client cancellation are never business completion.
Python case programs compose these actions and provide their configuration.
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

    def run(
        self,
        record,
        shape,
        timeout_s=90.0,
        schedule_timeout_s=30.0,
        stream_timeout_s=60.0,
    ):
        ops, clock = self.ops, self.clock
        end = clock() + timeout_s
        phase = "schedule"

        def remaining(cap):
            value = min(cap, end - clock())
            if value <= 0:
                raise TimeoutError("request budget expired")
            return value

        try:
            limit = remaining(schedule_timeout_s)
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
            limit = remaining(stream_timeout_s)
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


class ColdFlow(BoundedFlow):
    """Preserve legacy serial cold traffic: completion, then a 200ms pause.

    Schedule retains its 30s cap and Fetch its 10s cap. All attempts survive in
    the ledger, including failures. The independent done event comes from the
    same worker that performs and finishes the actual RPC consumption.
    """

    def __init__(self, ops, env_epoch, clock=time.monotonic):
        super().__init__(
            ops, env_epoch, [], interval_s=0.2, max_inflight=1, clock=clock
        )

    def _pump(self):
        end = self.clock() + 600
        try:
            while not self._stop.is_set():
                if self.clock() >= end:
                    raise TimeoutError("cold flow exceeded its 600s lifetime budget")
                rid = self.ops.next_request_id()
                record = self.issue(rid, self.clock)
                self.run(
                    record,
                    dict(output_len=2, block_keys=[rid * 100 + 1]),
                    timeout_s=min(40, end - self.clock()),
                    stream_timeout_s=10,
                )
                self._stop.wait(0.2)
        except Exception as exc:
            self.pump_error = repr(exc)
        finally:
            self.done.set()


def _cold_flow_validate(params, plan):
    return _validate(params, plan, set())


def _cold_flow_start(ctx, params, deadline):
    flow = ColdFlow(ctx.ops, ctx.env_epoch, clock=ctx.clock)
    flow.artifact_path = ctx.artifact_dir / f"elastic-cold-flow-{time.time_ns()}.json"

    def cleanup(d):
        try:
            flow.stop(d, cancel=True)
            if flow.pump_error:
                raise RuntimeError(flow.pump_error)
        finally:
            flow.artifact_path.write_text(json.dumps(flow.snapshot_records(), indent=2))

    handle = ctx.register_resource("flow", flow, cleanup=cleanup)
    deadline.check()
    flow.start()
    return StageOutput(output=dict(flow=handle))


def _flow_assert_validate(params, plan):
    p = _validate(
        params, plan, {"result", "min_success_rate"}, {"result", "min_success_rate"}
    )
    plan.reference(p["result"], "snapshot")
    rate = p["min_success_rate"]
    if type(rate) not in (int, float) or not 0 <= rate <= 1:
        raise ValueError("min_success_rate must be finite and between zero and one")
    return p


def _flow_assert(ctx, params, deadline):
    deadline.check()
    result = ctx.resource(params["result"], "snapshot")
    if (
        any(type(result.get(k)) is not int for k in ("issued", "completed"))
        or type(result.get("result_complete")) is not bool
    ):
        raise ValueError("flow result lacks complete attempt-accounting evidence")
    if not 0 <= result["completed"] <= result["issued"]:
        raise ValueError("flow success count exceeds attempt denominator")
    nonempty = result["issued"] > 0
    rate = result["completed"] / result["issued"] if nonempty else 0
    return StageOutput(
        checks=[
            CheckResult(
                "nonempty",
                "PASS" if nonempty else "FAIL",
                actual=result["issued"],
                expected=">0",
            ),
            CheckResult(
                "complete",
                "PASS" if result["result_complete"] else "FAIL",
                evidence=result,
            ),
            CheckResult(
                "success_rate",
                "PASS" if nonempty and rate >= params["min_success_rate"] else "FAIL",
                actual=rate,
                expected=params["min_success_rate"],
                evidence=result,
            ),
        ]
    )


def _accepted_validate(params, plan):
    p = _validate(
        params,
        plan,
        {"engine", "baseline", "window_s"},
        {"engine", "baseline", "window_s"},
    )
    for key, kind in (("engine", "string"), ("baseline", "integer")):
        if isinstance(p[key], dict):
            plan.reference(p[key], kind)
        elif key == "engine" and (not isinstance(p[key], str) or not p[key]):
            raise ValueError("accepted engine must be a name")
        elif key == "baseline" and (type(p[key]) is not int or p[key] < 0):
            raise ValueError("accepted baseline must be nonnegative")
    if type(p["window_s"]) not in (int, float) or not 0 < p["window_s"] <= 30:
        raise ValueError("accepted window must be in (0,30]")
    return p


def _accepted_wait(ctx, params, deadline):
    name, baseline = ctx.resolve(params["engine"]), ctx.resolve(params["baseline"])
    started = ctx.clock()
    evidence = dict(engine=name, baseline=baseline, started_s=started, samples=[])
    path = ctx.artifact_dir / f"elastic-accepted-{time.time_ns()}.json"
    try:
        while True:
            deadline.check()
            engine = _snapshot(ctx, deadline).get(name)
            if not isinstance(engine, dict) or type(engine.get("accepted")) is not int:
                raise ValueError("engine accepted counter evidence missing")
            accepted = engine["accepted"]
            if accepted < baseline:
                raise ValueError("accepted counter reset during traffic observation")
            elapsed = ctx.clock() - started
            evidence["samples"].append(dict(time_s=ctx.clock(), accepted=accepted))
            received = accepted > baseline and elapsed <= params["window_s"]
            if received or elapsed >= params["window_s"]:
                break
            deadline.sleep(min(0.2, params["window_s"] - elapsed))
        return StageOutput(
            output=dict(accepted=accepted),
            checks=[
                CheckResult(
                    "received",
                    "PASS" if received else "FAIL",
                    actual=accepted - baseline,
                    expected=">0 within window",
                    evidence=evidence,
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


def _accepted_snapshot_validate(params, plan):
    p = _validate(params, plan, {"engine"}, {"engine"})
    _accepted_validate(dict(p, baseline=0, window_s=1), plan)
    return p


def _accepted_snapshot(ctx, params, deadline):
    name = ctx.resolve(params["engine"])
    engine = _snapshot(ctx, deadline).get(name)
    if (
        not isinstance(engine, dict)
        or type(engine.get("accepted")) is not int
        or engine["accepted"] < 0
    ):
        raise ValueError("engine accepted snapshot evidence missing")
    path = ctx.artifact_dir / f"elastic-accepted-baseline-{time.time_ns()}.json"
    path.write_text(
        json.dumps(dict(engine=name, snapshot=engine, time_s=ctx.clock()), indent=2)
    )
    return StageOutput(output=dict(accepted=engine["accepted"]), artifacts=[str(path)])


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


def _add_validate(params, plan):
    p = _validate(params, plan, {"role"}, {"role"})
    if p["role"] not in {"prefill", "decode"}:
        raise ValueError(f"{plan.path}: role must be prefill or decode")
    return p


def _remove_validate(params, plan):
    p = _validate(params, plan, {"engine", "drain_timeout_ms"}, {"engine"})
    if isinstance(p["engine"], dict):
        plan.reference(p["engine"], "string")
    elif not isinstance(p["engine"], str) or not p["engine"]:
        raise ValueError(f"{plan.path}: engine must be a name or string reference")
    p.setdefault("drain_timeout_ms", 60000)
    # Explicit 5s retirement-deadline and 60s planned-drain contracts only.
    if type(p["drain_timeout_ms"]) is not int or p["drain_timeout_ms"] not in {
        5000,
        60000,
    }:
        raise ValueError(f"{plan.path}: drain_timeout_ms must be 5000 or 60000")
    return p


def _engine_identity(engine):
    if not isinstance(engine, dict) or engine.get("role") not in {"prefill", "decode"}:
        raise ValueError("engine snapshot lacks role evidence")
    address = engine.get("grpc_addr")
    if not isinstance(address, str) or ":" not in address:
        raise ValueError("engine snapshot lacks address evidence")
    port = int(address.rsplit(":", 1)[1])
    if not 1 <= port <= 65535:
        raise ValueError("engine snapshot has invalid port")
    return engine["role"], port


def _mutation(ctx, params, deadline, operation, request_http=None):
    """One explicit mutation, retaining its response even if validation fails.

    Snapshot membership is control-plane evidence only. Discovery, master
    convergence, actual traffic and request outcomes are separate stages.
    The instance owns the entire mock cluster and its teardown cleanup.
    """
    evidence = dict(operation=operation, started_s=ctx.clock(), complete=False)
    path = ctx.artifact_dir / f"elastic-{operation}-{time.time_ns()}.json"
    try:
        before = _snapshot(ctx, deadline)
        evidence["before"] = before
        if operation == "add":
            body = dict(role=params["role"])
        else:
            name = ctx.resolve(params["engine"])
            if name not in before:
                raise ValueError(f"cannot remove missing engine {name}")
            role, port = _engine_identity(before[name])
            required = params["drain_timeout_ms"] / 1000 + 5
            if deadline.remaining() < required:
                raise TimeoutError(f"remove requires {required}s remaining for drain")
            body = dict(
                engine=name,
                mode="graceful",
                drain_timeout_ms=params["drain_timeout_ms"],
            )
        evidence["request"] = body
        if request_http is not None:
            evidence["started_s"] = ctx.clock()
        response = (request_http or _http)(
            ctx.ops, f"{operation}_engine", deadline, body
        )
        evidence["response"] = response
        if not isinstance(response, dict) or response.get("status") != "ok":
            raise ValueError("mutation did not acknowledge success")
        acknowledged = response.get("engine")
        if not isinstance(acknowledged, str) or not acknowledged:
            raise ValueError("mutation response lacks engine identity")
        response_port = response.get("port")
        if type(response_port) is not int or not 1 <= response_port <= 65535:
            raise ValueError("mutation response lacks valid port")
        if response.get("action") != ("added" if operation == "add" else "removed"):
            raise ValueError("mutation action acknowledgement mismatch")
        after = _snapshot(ctx, deadline)
        evidence["after"] = after
        if operation == "add":
            name, port = acknowledged, response_port
            if name in before or name not in after:
                raise ValueError("added engine must be new and present in snapshot")
            role, actual_port = _engine_identity(after[name])
            if (
                role != params["role"]
                or actual_port != port
                or response.get("http_port") != port - 1
            ):
                raise ValueError("added engine role or address mismatch")
        else:
            if acknowledged != name or response_port != port or name in after:
                raise ValueError("removed engine identity or membership mismatch")
            if (
                response.get("mode") != "graceful"
                or type(response.get("drained")) is not bool
            ):
                raise ValueError("remove response lacks graceful drain evidence")
        evidence.update(complete=True, engine=name, role=role, port=port)
        # drained=false is retained for a later contract check; an ack alone
        # cannot establish successful draining or zero request failures.
        handle = ctx.register_resource("snapshot", evidence, historical=True)
        return StageOutput(
            output=dict(engine=name, port=port, mutation=handle),
            checks=[
                CheckResult(
                    "membership",
                    "PASS",
                    actual=dict(engine=name, present=operation == "add"),
                )
            ],
            artifacts=[str(path)],
        )
    except BaseException as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence["ended_s"] = ctx.clock()
        path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def _add(ctx, params, deadline):
    return _mutation(ctx, params, deadline, "add")


def _remove(ctx, params, deadline):
    return _mutation(ctx, params, deadline, "remove")


def _topology_validate(params, plan):
    p = _validate(
        params,
        plan,
        {"role", "discovered", "alive", "port", "present"},
        {"role", "discovered", "alive"},
    )
    if p["role"] not in {"PREFILL", "DECODE"}:
        raise ValueError("topology role must be PREFILL or DECODE")
    for name in ("discovered", "alive"):
        if type(p[name]) is not int or not 0 <= p[name] <= 32:
            raise ValueError(f"topology {name} must be an integer in [0,32]")
    if p["alive"] > p["discovered"]:
        raise ValueError("alive cannot exceed discovered")
    if ("port" in p) != ("present" in p):
        raise ValueError("topology port and present must be specified together")
    if "port" in p:
        if isinstance(p["port"], dict):
            plan.reference(p["port"], "integer")
        elif type(p["port"]) is not int or not 1 < p["port"] <= 65535:
            raise ValueError("topology port must be a valid grpc port")
        if type(p["present"]) is not bool:
            raise ValueError("topology present must be boolean")
    return p


def _topology(ctx, params, deadline):
    """File convergence (10s) and Master convergence (30s) stay distinct.

    A stopped engine can remain discovered while becoming non-alive. File
    entries and Master discovered counts therefore are never substituted for
    each other. Every failed/missing probe remains in the artifact.
    """
    from ...harness import http_post_json

    started = ctx.clock()
    path = ctx.artifact_dir / f"elastic-topology-{time.time_ns()}.json"
    evidence = dict(started_s=started, params=params, samples=[], complete=False)
    domain = {
        "PREFILL": "mock.prefill.hosts.address",
        "DECODE": "mock.decode.hosts.address",
    }[params["role"]]
    port = ctx.resolve(params["port"]) if "port" in params else None
    file_ok = master_ok = False
    file_finished = False
    try:
        while True:
            deadline.check()
            now = ctx.clock()
            sample = dict(time_s=now)
            payload = json.loads(ctx.env.discovery_file.read_text(encoding="utf-8"))
            hosts = payload.get(domain) if isinstance(payload, dict) else None
            if not isinstance(hosts, list) or any(
                not isinstance(h, str) for h in hosts
            ):
                raise ValueError("discovery file lacks role host-list evidence")
            sample["discovery_hosts"] = hosts
            file_matches = len(hosts) == params["discovered"]
            if port is not None:
                found = any(h.endswith(f":{port - 1}") for h in hosts)
                file_matches = file_matches and found == params["present"]
            if not file_finished:
                file_ok = file_matches and now - started <= 10
                file_finished = file_ok or now - started >= 10
                if file_finished:
                    evidence["file_convergence_s"] = now - started
            status, body = http_post_json(
                f"http://127.0.0.1:{ctx.ops.master_http_port}/rtp_llm/master/info",
                {},
                timeout=min(2, deadline.remaining()),
            )
            sample["master_status"] = status
            sample["master_response"] = body
            evidence["samples"].append(sample)
            if status != 200 or not isinstance(body, dict):
                raise ValueError(f"master topology probe failed: {status}")
            summary = body.get("worker_summary", {}).get(params["role"], {})
            if any(type(summary.get(k)) is not int for k in ("discovered", "alive")):
                raise ValueError("master topology lacks discovered/alive evidence")
            master_ok = (
                all(summary[k] == params[k] for k in ("discovered", "alive"))
                and ctx.clock() - started <= 30
            )
            if (file_finished and master_ok) or ctx.clock() - started >= 30:
                break
            deadline.sleep(0.2)
        file_ok = file_ok and file_matches
        evidence.update(
            complete=True, file_converged=file_ok, master_converged=master_ok
        )
        handle = ctx.register_resource("snapshot", evidence, historical=True)
        return StageOutput(
            output=dict(snapshot=handle),
            checks=[
                CheckResult(
                    "discovery",
                    "PASS" if file_ok else "FAIL",
                    actual=evidence.get("file_convergence_s"),
                    expected="file topology within 10s",
                    evidence=evidence,
                ),
                CheckResult(
                    "master",
                    "PASS" if master_ok else "FAIL",
                    actual=summary,
                    expected={k: params[k] for k in ("discovered", "alive")},
                    evidence=evidence,
                ),
            ],
            artifacts=[str(path)],
        )
    except BaseException as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence["ended_s"] = ctx.clock()
        path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")


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
        if hasattr(records, "cleanup"):
            records.cleanup(deadline)
        else:
            records.cancel_active("cleanup")
        (ctx.artifact_dir / name).write_text(
            json.dumps(records.snapshot_records(), indent=2)
        )

    return cleanup


def _skew_cache_quiet(ctx, names, deadline):
    end = ctx.clock() + 8
    previous, changed, samples = {}, {}, []
    while ctx.clock() < end:
        deadline.check()
        snap = _snapshot(ctx, deadline)
        now = ctx.clock()
        keys = {}
        for name in names:
            raw = snap.get(name, {}).get("cache_key_set")
            if not isinstance(raw, list):
                raise ValueError("skew seed lacks explicit cache key-set evidence")
            keys[name] = frozenset(int(k) for k in raw)
            if keys[name] != previous.get(name):
                previous[name], changed[name] = keys[name], now
        samples.append(dict(time_s=now, keys={n: sorted(v) for n, v in keys.items()}))
        if all(now - changed[n] >= 3.5 for n in names):
            return dict(quiet=True, samples=samples, last_change_s=changed)
        deadline.sleep(0.5)
    return dict(quiet=False, samples=samples, last_change_s=changed)


def _seed(ctx, params, deadline):
    from .elastic_skew_requests import SkewRecordedRequests, legacy_success, summary

    records = SkewRecordedRequests(ctx, "seed", 30, deadline.expires_at)
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
        if not legacy_success(record) or name not in selected:
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
    quiet = _skew_cache_quiet(ctx, selected, deadline)
    if not quiet["quiet"]:
        return StageOutput(
            output={"requests": handle},
            checks=[
                CheckResult(
                    "seed_success",
                    "FAIL",
                    actual=quiet,
                    expected="each cache key set unchanged for 3.5s within 8s",
                )
            ],
        )
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
        cache_quiet=quiet,
    )
    family_handle = ctx.register_resource("snapshot", families, historical=True)
    return StageOutput(
        output={"requests": handle, "families": family_handle},
        checks=[
            CheckResult(
                "seed_success", "PASS", actual=summary(records.snapshot_records())
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
    from .elastic_skew_requests import SkewFlow

    flow = SkewFlow(ctx, families["families"])

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
    path = getattr(
        flow, "artifact_path", ctx.artifact_dir / "elastic-client-records.json"
    )
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
    from .elastic_skew_requests import SkewRecordedRequests, summary

    records = SkewRecordedRequests(ctx, "recovery", 15, deadline.expires_at)
    handle = ctx.register_resource(
        "requests",
        records,
        cleanup=_record_cleanup(ctx, records, "elastic-recovery-records.json"),
    )
    issued = [records.issue(ctx.ops.next_request_id(), ctx.clock) for _ in range(20)]

    def run(record):
        records.run(
            record,
            dict(
                input_len=2048,
                output_len=2,
                block_keys=[record["wire_request_id"] * 100 + j for j in range(3)],
            ),
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(run, issued))
    deadline.check()
    result = summary(records.snapshot_records())
    return StageOutput(
        output={"requests": handle, "success_rate": result["completed"] / 20}
    )


HANDLERS = [
    StageHandler(
        "elastic_accepted_snapshot",
        _accepted_snapshot_validate,
        _accepted_snapshot,
        {"accepted": "integer"},
    ),
    StageHandler(
        "elastic_accepted",
        _accepted_validate,
        _accepted_wait,
        {"accepted": "integer"},
        checks=frozenset({"received"}),
    ),
    StageHandler(
        "elastic_cold_flow", _cold_flow_validate, _cold_flow_start, {"flow": "flow"}
    ),
    StageHandler(
        "elastic_flow_assert",
        _flow_assert_validate,
        _flow_assert,
        {},
        checks=frozenset({"nonempty", "complete", "success_rate"}),
    ),
    StageHandler(
        "elastic_topology",
        _topology_validate,
        _topology,
        {"snapshot": "snapshot"},
        checks=frozenset({"discovery", "master"}),
    ),
    StageHandler(
        "elastic_add",
        _add_validate,
        _add,
        {"engine": "string", "port": "integer", "mutation": "snapshot"},
        checks=frozenset({"membership"}),
        max_dynamic_additions=1,
    ),
    StageHandler(
        "elastic_remove",
        _remove_validate,
        _remove,
        {"engine": "string", "port": "integer", "mutation": "snapshot"},
        checks=frozenset({"membership"}),
    ),
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
    "mock_engine_completed_total",
    "mock_engine_accepted_total",
    "mock_engine_decode_ms_avg",
    "mock_engine_lack_mem_rejects_total",
    "mock_engine_kv_admission_fails_total",
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
    metrics.skew_started_s = ctx.clock()
    metrics.thread.start()
    return StageOutput(output={"observation": handle})


def _metric_phase_validate(params, plan):
    p = _validate(
        params,
        plan,
        {
            "observation",
            "phase",
            "baseline",
            "scale",
            "families",
            "victim",
            "transient",
        },
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
    if p["phase"] == "steady":
        plan.reference(p["transient"], "snapshot")
    return p


def _baseline_validate(params, plan):
    p = _metric_phase_validate(params, plan)
    if p["phase"] != "baseline":
        raise ValueError("elastic_baseline requires phase baseline")
    return p


def _window_validate(params, plan):
    p = _metric_phase_validate(params, plan)
    if p["phase"] == "baseline":
        raise ValueError("baseline requires elastic_baseline traffic evidence")
    return p


def _window(ctx, params, deadline):
    metrics = ctx.resource(params["observation"], "observation")
    phase = params["phase"]
    extra = {}
    if phase == "baseline":
        start = metrics.skew_started_s
        deadline.sleep(20)
        end = ctx.clock()
    elif phase == "transient":
        scale = ctx.resource(params["scale"], "snapshot")
        start = scale["started_s"]
        extra["remove_returned_s"] = scale["ended_s"]
        extra["post_remove_wait_started_s"] = ctx.clock()
        deadline.sleep(20)
        end = ctx.clock()
    else:
        from ...harness import http_post_json

        alive_end = ctx.clock() + 30
        alive_ok, probes = False, []
        while ctx.clock() < alive_end:
            deadline.check()
            status, body = http_post_json(
                f"http://127.0.0.1:{ctx.ops.master_http_port}/rtp_llm/master/info",
                {},
                timeout=min(5, deadline.remaining()),
            )
            probes.append(dict(time_s=ctx.clock(), status=status, raw=body))
            alive = (
                body.get("worker_summary", {}).get("PREFILL", {}).get("alive")
                if status == 200 and isinstance(body, dict)
                else None
            )
            if type(alive) is int and alive >= 1:
                alive_ok = True
                break
            deadline.sleep(0.5)
        settled = ctx.clock()
        deadline.sleep(60)
        start, end = settled + 40, ctx.clock()
        extra.update(
            alive_ok=alive_ok,
            alive_probes=probes,
            settled_s=settled,
            alive_policy="legacy diagnostic alive>=1, not a hard topology gate",
        )
    survivor = None
    if phase == "steady":
        families = ctx.resource(params["families"], "snapshot")
        survivor = families["cold" if params["victim"] == "hot" else "hot"]
    evidence = metric_window(metrics.snapshot(), start, end, survivor)
    evidence.update(extra)
    if phase == "steady":
        base = ctx.resource(params["baseline"], "snapshot")["hit_rate"]
        transient = ctx.resource(params["transient"], "snapshot")["hit_rate"]
        removed_s = ctx.resource(params["scale"], "snapshot")["started_s"]
        raw, windows = metrics.snapshot(), []
        anchor = removed_s
        while anchor + 10 <= end:
            try:
                rate = metric_window(raw, anchor, anchor + 10)["hit_rate"]
                windows.append(
                    dict(start_s=anchor, midpoint_s=anchor + 5, hit_rate=rate)
                )
            except ValueError as exc:
                windows.append(
                    dict(
                        start_s=anchor,
                        midpoint_s=anchor + 5,
                        hit_rate=None,
                        unavailable=str(exc),
                    )
                )
            anchor += 10
        recovered = next(
            (
                r["midpoint_s"] - removed_s
                for r in windows
                if r["hit_rate"] is not None and r["hit_rate"] >= base - 0.15
            ),
            None,
        )
        evidence["observations"] = dict(
            windows_10s=windows,
            recovery_duration_s=recovered,
            baseline_hit=base,
            transient_hit=transient,
            gates=False,
            rebound_floor=(
                transient + 0.5 * (base - transient) if base > transient else None
            ),
            steady_base_floor=base - 0.15,
        )
    handle = ctx.register_resource("snapshot", evidence, historical=True)
    checks = []
    if phase == "baseline":
        checks.append(
            CheckResult(
                "traffic",
                "PASS",
                actual=evidence["requested"],
                expected=">0 cache-key observations",
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
        checks=frozenset({"traffic"}),
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
    from .elastic_skew_requests import summary

    recovery = summary(ctx.resource(params["recovery"], "requests").snapshot_records())
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
        checks=frozenset({"PC", "PQ", "PK", "P6", "P2"}),
    )
)

# Kept in a separate module so lifecycle measurement does not expand the
# cache-shrink implementation. Registration still belongs to the core catalog.
from .elastic_lifecycle import HANDLERS as LIFECYCLE_HANDLERS

HANDLERS += LIFECYCLE_HANDLERS

from .elastic_concurrent import HANDLERS as CONCURRENT_HANDLERS

HANDLERS += CONCURRENT_HANDLERS

from .elastic_pending import HANDLERS as PENDING_HANDLERS

HANDLERS += PENDING_HANDLERS

from .elastic_balance import HANDLERS as BALANCE_HANDLERS

HANDLERS += BALANCE_HANDLERS

from .elastic_full import HANDLERS as FULL_HANDLERS

HANDLERS += FULL_HANDLERS

from .elastic_transient import HANDLERS as TRANSIENT_HANDLERS

HANDLERS += TRANSIENT_HANDLERS
