"""Java mock backend: bounded owned processes and finite, recorded RPC batches.

Imports of the legacy process harness happen only on setup, after lease checks.
"""

import json
import os
import signal
import subprocess
import threading
from pathlib import Path

from .actions.elastic import ClientRecords, request_success
from .runtime import StageTimeout


class BoundedOps:
    """A lane-scoped facade; ordinary operations preserve EngineOps semantics."""

    def __init__(self, ops, budget, lease):
        self._ops, self._budget, self._lease = ops, budget, lease
        self._additions = 0
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._ops, name)

    def add_engine(self, role, port=None):
        if port is not None:
            raise ValueError("scenario adapters cannot allocate explicit worker ports")
        if role not in ("prefill", "decode"):
            raise ValueError("unknown engine role")
        with self._lock:
            if self._additions >= self._budget["max_dynamic_additions"]:
                raise ValueError("declared cumulative engine-add budget exhausted")
            # Count attempts conservatively: an HTTP failure can follow a server
            # side allocation, so its port must not silently become free budget.
            self._additions += 1
            status, body = self._ops.add_engine(role)
            if status == 200:
                allocated = body.get("port") if isinstance(body, dict) else None
                first = self._lease["mock_base"]
                last = (
                    first
                    + self._budget["initial_workers"]
                    + self._budget["max_dynamic_additions"]
                    - 1
                )
                if type(allocated) is not int or not first <= allocated <= last:
                    raise RuntimeError(
                        "mock allocated a worker outside its declared lane budget"
                    )
            return status, body


class RequestBatch(ClientRecords):
    """Finite issuance; deferred mode really performs zero Fetch until wait."""

    def __init__(self, ctx, params):
        super().__init__(ctx.env_epoch)
        self.ctx, self.ops, self.params = ctx, ctx.ops, params
        self.entries = []
        self.cancelled = False
        self.artifact = ctx.artifact_dir / f"requests-{len(ctx._resources)+1}.json"

    def persist(self):
        self.artifact.write_text(json.dumps(self.snapshot_records(), indent=2) + "\n")

    def _error(self, record, phase, exc):
        code = getattr(exc, "code", lambda: None)()
        self.update(
            record,
            **{
                phase: dict(
                    status=getattr(code, "name", "ERROR"),
                    error=repr(exc),
                    ended_s=self.ctx.clock(),
                )
            },
        )

    def submit(self, deadline):
        for _ in range(self.params["count"]):
            deadline.check()
            record = self.issue(self.ops.next_request_id(), self.ctx.clock)
            entry = dict(record=record, response=None, call=None, thread=None)
            self.entries.append(entry)
            rid = record["wire_request_id"]
            try:
                limit = min(30, deadline.remaining())
                self.update(
                    record,
                    schedule=dict(
                        started_s=self.ctx.clock(), deadline_s=self.ctx.clock() + limit
                    ),
                )
                stub = self.ops.schedule_pb2_grpc.FlexlbServiceStub(
                    self.ops._channel(self.ops.master_target())
                )
                call = stub.Schedule.future(
                    self.ops.build_schedule_request(rid, **self.shape), timeout=limit
                )
                entry["call"] = call
                response = call.result(timeout=limit)
                entry["response"], entry["call"] = response, None
                self.update(
                    record,
                    schedule=dict(status="OK", ended_s=self.ctx.clock()),
                    prefill_addr=self.ops.prefill_addr(response),
                    enqueued_by_master=bool(response.enqueued_by_master),
                    fetch_invocations=0,
                )
                if response.code != 200 or not response.success:
                    self.update(
                        record,
                        schedule=dict(status="REJECTED", error=response.error_message),
                        transport_terminal_s=self.ctx.clock(),
                        consumer_exit_s=self.ctx.clock(),
                    )
                    continue
                if (
                    self.params["consume"] == "deferred"
                    and not response.enqueued_by_master
                ):
                    raise RuntimeError("deferred request was not enqueued by master")
                if self.params["consume"] == "immediate":
                    self._start_consumer(entry, self.ctx.instance_deadline_s)
            except Exception as exc:
                self._error(record, "schedule", exc)
                if entry["call"] is not None:
                    entry["call"].cancel()
                self.update(
                    record,
                    transport_terminal_s=self.ctx.clock(),
                    consumer_exit_s=self.ctx.clock(),
                )
                raise
            finally:
                self.persist()

    @property
    def shape(self):
        return {key: self.params[key] for key in ("input_len", "output_len")}

    def _start_consumer(self, entry, end):
        if entry["thread"] is not None or entry["record"]["schedule"]["status"] != "OK":
            return
        if self.cancelled:
            return
        thread = threading.Thread(
            target=self._consume,
            args=(entry, end),
            name="scenario-request-consumer",
            daemon=True,
        )
        entry["thread"] = thread
        thread.start()

    def _consume(self, entry, end):
        record, response = entry["record"], entry["response"]
        phase = "stream"
        try:
            limit = min(60, end - self.ctx.clock())
            if limit <= 0:
                raise StageTimeout("stream deadline expired before opening")
            method = (
                "FetchResponse" if response.enqueued_by_master else "GenerateStreamCall"
            )
            self.update(
                record,
                stream=dict(
                    method=method,
                    started_s=self.ctx.clock(),
                    deadline_s=self.ctx.clock() + limit,
                ),
            )
            stub = self.ops.pb2_grpc.RpcServiceStub(
                self.ops._channel(self.ops.prefill_addr(response))
            )
            if response.enqueued_by_master:
                self.update(record, fetch_invocations=1)
                call = stub.FetchResponse(
                    self.ops.pb2.FetchRequestPB(request_id=record["wire_request_id"]),
                    timeout=limit,
                )
            else:
                inp = self.ops.build_generate_input(
                    record["wire_request_id"], **self.shape
                )
                self.ops._copy_role_addrs(inp, response)
                call = stub.GenerateStreamCall(inp, timeout=limit)
            with self._lock:
                entry["call"] = call
                if self.cancelled:
                    call.cancel()
            for output in call:
                if record["stream"]["first_output_s"] is None:
                    self.update(record, stream=dict(first_output_s=self.ctx.clock()))
                if output.HasField("error_info"):
                    self.update(
                        record,
                        business_error_code=int(output.error_info.error_code),
                        business_error_message=output.error_info.error_message,
                    )
                if any(output.flatten_output.finished):
                    self.update(record, business_finished=True)
            self.update(record, stream=dict(status="OK", ended_s=self.ctx.clock()))
        except Exception as exc:
            self._error(record, phase, exc)
        finally:
            self.update(
                record,
                transport_terminal_s=self.ctx.clock(),
                consumer_exit_s=self.ctx.clock(),
            )
            with self._lock:
                entry["call"] = None

    def wait(self, deadline):
        for entry in self.entries:
            self._start_consumer(
                entry, min(deadline.expires_at, self.ctx.instance_deadline_s)
            )
        for entry in self.entries:
            thread = entry["thread"]
            while thread is not None and thread.is_alive():
                thread.join(min(0.1, deadline.remaining()))
        self.persist()
        records = self.snapshot_records()
        for record in records:
            for phase in ("schedule", "stream"):
                rpc = record[phase]
                if rpc["status"] == "DEADLINE_EXCEEDED":
                    raise StageTimeout(
                        f"{phase} RPC deadline exceeded for {record['wire_request_id']}"
                    )
                if rpc["status"] not in (None, "OK", "REJECTED"):
                    raise RuntimeError(
                        f"{phase} RPC failed for {record['wire_request_id']}: {rpc['error']}"
                    )
        return dict(
            completed=bool(records) and all(request_success(r) for r in records),
            error_count=sum(not request_success(r) for r in records),
        )

    def cancel(self, reason="explicit_cancel"):
        count = 0
        with self._lock:
            self.cancelled = True
            for entry in self.entries:
                record, call = entry["record"], entry["call"]
                if (
                    request_success(record)
                    or record["cancel"]["requested_s"] is not None
                ):
                    continue
                count += 1
                self.update(
                    record,
                    cancel=dict(
                        requested_s=self.ctx.clock(),
                        reason=reason,
                        scope="client_transport",
                    ),
                )
                try:
                    ack = bool(call.cancel()) if call is not None else None
                    self.update(record, cancel=dict(acknowledged=ack))
                except Exception as exc:
                    self.update(record, cancel=dict(error=repr(exc)))
                finally:
                    self.update(record, cancel=dict(ended_s=self.ctx.clock()))
        return count

    def cleanup(self, deadline):
        self.cancel("cleanup")
        try:
            for entry in self.entries:
                thread = entry["thread"]
                if thread is not None and thread.is_alive():
                    thread.join(max(0, deadline.expires_at - self.ctx.clock()))
                    if thread.is_alive():
                        raise StageTimeout(
                            "request consumer remains alive after cancel and join"
                        )
            if any(r["cancel"]["error"] for r in self.snapshot_records()):
                raise RuntimeError("one or more transport cancellations failed")
        finally:
            self.persist()

    def cancel_server(self, deadline):
        """Cancel the owned client calls and issue bounded owner-directed Cancel RPCs."""
        issued = self.cancel()
        for entry in self.entries:
            record, response = entry["record"], entry["response"]
            if record["cancel"]["requested_s"] is None:
                continue
            ops = self.ops
            request = ops.schedule_pb2.FlexlbCancelRequestPB(
                request_id=record["wire_request_id"],
                reason=ops.schedule_pb2.CANCEL_REASON_CLIENT_CANCELLED,
            )
            if (
                response is not None
                and response.HasField("lifecycle")
                and response.lifecycle.batch_id
            ):
                request.batch_id = response.lifecycle.batch_id
            calls = [
                (
                    "master",
                    ops.schedule_pb2_grpc.FlexlbServiceStub(
                        ops._channel(ops.master_target())
                    ).Cancel,
                    request,
                )
            ]
            if response is not None and not response.enqueued_by_master:
                stub = ops.pb2_grpc.RpcServiceStub(
                    ops._channel(ops.prefill_addr(response))
                )
                calls.append(
                    (
                        "worker",
                        stub.Cancel,
                        ops.pb2.CancelRequestPB(request_id=record["wire_request_id"]),
                    )
                )
            outcomes = []
            try:
                for owner, rpc, request in calls:
                    limit = min(10, deadline.remaining())
                    started = self.ctx.clock()
                    # The stage wall-clock guard and gRPC deadline both bound
                    # this synchronous unary RPC. Completion is not business FINISHED.
                    response_pb = rpc(request, timeout=limit)
                    outcomes.append(
                        dict(
                            owner=owner,
                            transport_status="OK",
                            started_s=started,
                            ended_s=self.ctx.clock(),
                            response=str(response_pb),
                        )
                    )
            finally:
                self.update(record, server_cancel=outcomes)
                self.persist()
        return issued


class JavaMockBackend:
    def __init__(self, lease):
        self.lease = lease
        self.manager = self.raw_ops = None
        self.environments = []

    def setup(self, ctx, plan, deadline):
        from flexlb_cfg import OMIT, ConfigOverride
        from flexlb_ft.context import CaseContext
        from flexlb_ft.engine_ops import EngineOps
        from flexlb_ft.harness import EnvManager, EnvSpec, default_perf, fault_env_perf

        owner = self

        class OwnedManager(EnvManager):
            def _start_mock(self, env):
                owner.environments.append(env)
                return super()._start_mock(env)

            def _stop_env_processes(self, env):
                # The executor registered cleanup before setup. Retain partial
                # references for its separately budgeted, observable teardown.
                if env not in owner.environments:
                    owner.environments.append(env)

        kwargs = {
            k: OMIT if isinstance(v, dict) else v
            for k, v in plan["config_overrides"].items()
        }
        spec = EnvSpec(
            label="scenario",
            n_prefill=plan["n_prefill"],
            n_decode=plan["n_decode"],
            master_profile=ctx.instance["profile"],
            config_overrides=ConfigOverride(**kwargs),
            discovery=plan["discovery"],
            perf=(
                fault_env_perf()
                if plan["perf_preset"] == "fault_env"
                else default_perf()
            ),
            master_env=(
                {"FLEXLB_DEBUG_ENABLED": "true"} if plan["debug_enabled"] else {}
            ),
        )
        for key in ("prefill_cache_blocks", "decode_cache_blocks"):
            if key in plan:
                setattr(spec, key, plan[key])
        self.manager = OwnedManager(ctx.artifact_dir / "environment")
        env = self.manager.ensure(spec)
        deadline.check()
        if (
            env.base_grpc_port != self.lease["mock_base"]
            or env.master_http_port != self.lease["master_base"]
            or env.master_management_port != self.lease["master_base"] + 1
        ):
            raise RuntimeError("harness changed a leased port")
        self.raw_ops = EngineOps("127.0.0.1", env.master_http_port, env.mock_http_port)
        ops = BoundedOps(self.raw_ops, ctx.instance["resource_budget"], self.lease)
        ctx.case_context = CaseContext(
            self.manager, ctx.instance["profile"], ctx.artifact_dir
        )
        (ctx.artifact_dir / "environment.json").write_text(
            json.dumps(
                dict(
                    fingerprint=spec.fingerprint(),
                    lease=self.lease,
                    resolved_config=plan["resolved_config"],
                ),
                indent=2,
            )
            + "\n"
        )
        return env, ops

    def start_requests(self, ctx, params, deadline):
        batch = RequestBatch(ctx, params)
        handle = ctx.register_resource(
            "requests", batch, batch.cleanup, historical=True
        )
        batch.submit(deadline)
        return handle

    def wait_requests(self, ctx, requests, deadline):
        return requests.wait(deadline)

    def cancel_requests(self, ctx, requests, deadline):
        deadline.check()
        return requests.cancel_server(deadline)

    def teardown(self, ctx, deadline):
        if self.raw_ops is not None:
            self.raw_ops.close()
        processes = []
        for env in self.environments:
            processes.extend(env.load_clients)
            processes.extend(env.victims.values())
            processes.extend(env.masters.values())
            processes.extend([env.master, env.zk_helper, env.mock])
        unique = {mp.pid: mp for mp in processes if mp is not None}
        # Signal every owned PID before any wait. No process-name scans or kills.
        for mp in unique.values():
            if mp.alive():
                try:
                    os.kill(mp.pid, signal.SIGCONT)
                    mp.proc.terminate()
                except ProcessLookupError:
                    pass
        for mp in unique.values():
            if not mp.alive():
                continue
            remaining = max(0, deadline.expires_at - ctx.clock())
            try:
                mp.proc.wait(timeout=min(2, remaining / max(1, len(unique))))
            except subprocess.TimeoutExpired:
                mp.proc.kill()
        remaining_pids = []
        for mp in unique.values():
            try:
                mp.proc.wait(timeout=max(0, deadline.expires_at - ctx.clock()))
            except subprocess.TimeoutExpired:
                remaining_pids.append(mp.pid)
        (ctx.artifact_dir / "process-cleanup.json").write_text(
            json.dumps(
                dict(owned_pids=list(unique), remaining_pids=remaining_pids), indent=2
            )
            + "\n"
        )
        if remaining_pids:
            raise StageTimeout(f"owned processes not reaped: {remaining_pids}")
        if self.manager is not None:
            self.manager.current = None
