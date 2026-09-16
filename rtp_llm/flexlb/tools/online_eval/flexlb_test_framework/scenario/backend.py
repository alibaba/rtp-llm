"""Java mock backend: bounded owned processes and finite, recorded RPC batches.

Imports of the legacy process harness happen only on setup, after lease checks.
"""

import base64
import hashlib
import json
import os
import signal
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

from online_eval.requests import ClientRecords, request_success
from .runtime import StageTimeout


def error_trailer_evidence(exc, pb2):
    """Record typed engine errors without interpreting transport cancellation.

    Missing, ambiguous or malformed metadata never becomes a zero error code.
    Keep at most 64 KiB of each of the first two matching values; retain length
    and SHA256 for oversized values. gRPC itself bounds received metadata.
    """
    result = dict(trailer_error_code=None, trailer_error_message=None)
    evidence = dict(status="absent", values=[])
    result["error_trailer"] = evidence
    try:
        read = getattr(exc, "trailing_metadata", None)
        metadata = read() if callable(read) else None
        values = [v for k, v in (metadata or ()) if k == "grpc-status-details-bin"]
        for value in values[:2]:
            if not isinstance(value, bytes):
                evidence["values"].append(dict(value_type=type(value).__name__))
                continue
            evidence["values"].append(
                dict(
                    base64=base64.b64encode(value[:65536]).decode("ascii"),
                    size_bytes=len(value),
                    sha256=hashlib.sha256(value).hexdigest(),
                    truncated=len(value) > 65536,
                )
            )
        evidence["value_count"] = len(values)
        if not values:
            return result
        if len(values) != 1:
            evidence["status"] = "ambiguous"
            return result
        raw = values[0]
        if not isinstance(raw, bytes) or not raw or len(raw) > 65536:
            evidence["status"] = "invalid_value"
            return result
        details = pb2.ErrorDetailsPB.FromString(raw)
        # Proto3 defaults alone are not evidence that a typed code was sent.
        # ListFields distinguishes an unknown-only/wrong-message payload.
        if not any(field.name == "error_code" for field, _ in details.ListFields()):
            evidence["status"] = "missing_error_code"
            return result
        result["trailer_error_code"] = int(details.error_code)
        result["trailer_error_message"] = details.error_message
        evidence["status"] = "parsed"
    except Exception as error:
        evidence.update(status="parse_error", error=repr(error))
    return result


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
                    + self._budget.get(
                        "max_environment_workers", self._budget["initial_workers"]
                    )
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
        details = error_trailer_evidence(exc, self.ops.pb2) if phase == "stream" else {}
        self.update(
            record,
            **{
                phase: dict(
                    status=getattr(code, "name", "ERROR"),
                    error=repr(exc),
                    ended_s=self.ctx.clock(),
                    **details,
                )
            },
        )

    def submit(self, deadline):
        for _ in range(self.params["count"]):
            deadline.check()
            if self.cancelled:
                raise RuntimeError("request cancelled before Schedule")
            record = self.issue(self.ops.next_request_id(), self.ctx.clock)
            entry = dict(
                record=record,
                response=None,
                call=None,
                thread=None,
                done=threading.Event(),
            )
            self.entries.append(entry)
            with self._lock:
                record["request_shape"] = dict(self.shape)
                record["qos_level"] = self.params.get("qos_level")
            rid = record["wire_request_id"]
            try:
                limit = min(
                    self.params.get("schedule_timeout_s", 30), deadline.remaining()
                )
                self.update(
                    record,
                    schedule=dict(
                        started_s=self.ctx.clock(), deadline_s=self.ctx.clock() + limit
                    ),
                )
                stub = self.ops.schedule_pb2_grpc.FlexlbServiceStub(
                    self.ops._channel(self.ops.master_target())
                )
                rpc_options = {"timeout": limit}
                if "qos_level" in self.params:
                    from ..engine_ops import QOS_LEVEL_HEADER

                    rpc_options["metadata"] = (
                        (QOS_LEVEL_HEADER, str(self.params["qos_level"])),
                    )
                call = stub.Schedule.future(
                    self.ops.build_schedule_request(rid, **self.shape), **rpc_options
                )
                self._register_schedule_call(entry, call)
                # The RPC already owns its transport deadline. Reusing that
                # duration as the local future wait races gRPC's completion
                # callback and can replace DEADLINE_EXCEEDED with an untyped
                # FutureTimeoutError. Await the actual outcome within the stage
                # budget; the RPC timeout above remains unchanged.
                response = call.result(timeout=deadline.remaining())
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
                if self.params.get("post_issue_delay_s", 0):
                    deadline.sleep(self.params["post_issue_delay_s"])

    def _register_schedule_call(self, entry, call):
        # Future creation may overlap aggregate cancellation. Publish and
        # recheck under the cancellation lock before waiting on its result.
        with self._lock:
            entry["call"] = call
            if self.cancelled:
                call.cancel()
                raise RuntimeError("request cancelled during Schedule startup")

    @property
    def shape(self):
        return {
            key: self.params[key]
            for key in ("input_len", "output_len", "block_keys", "priority")
            if key in self.params
        }

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

    def _open_stream(self, entry, end):
        record, response = entry["record"], entry["response"]
        limit = min(self.params.get("stream_timeout_s", 60), end - self.ctx.clock())
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
            inp = self.ops.build_generate_input(record["wire_request_id"], **self.shape)
            self.ops._copy_role_addrs(inp, response)
            call = stub.GenerateStreamCall(inp, timeout=limit)
        return call

    def _consume(self, entry, end):
        record = entry["record"]
        phase = "stream"
        try:
            call = entry.pop("opened_call", None)
            if call is None:
                call = self._open_stream(entry, end)
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
                consumer_done=True,
            )
            with self._lock:
                entry["call"] = None
            # Published only after all terminal record fields are committed.
            entry["done"].set()

    def _await_consumer(self, entry, deadline):
        thread = entry["thread"]
        if thread is None:
            return
        # is_alive()/join alone are not a completion witness when a main-thread
        # interruption can race with waiting. Wait for the consumer's own signal.
        while not entry["done"].is_set():
            entry["done"].wait(min(0.1, deadline.remaining()))
        thread.join(max(0, deadline.expires_at - self.ctx.clock()))
        record = entry["record"]
        if (
            thread.is_alive()
            or record.get("consumer_done") is not True
            or record["consumer_exit_s"] is None
            or record["transport_terminal_s"] is None
            or record["stream"]["ended_s"] is None
            or record["stream"]["status"] is None
        ):
            raise RuntimeError(
                "consumer completion signal lacks terminal exit evidence"
            )
        self.update(record, consumer_completion_verified=True)

    def wait(self, deadline):
        for entry in self.entries:
            self._start_consumer(
                entry, min(deadline.expires_at, self.ctx.instance_deadline_s)
            )
        for entry in self.entries:
            self._await_consumer(entry, deadline)
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
                self._await_consumer(entry, deadline)
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


def make_env_spec(plan, profile, lease):
    """Controlled environment rendering; no JVM is started by this function."""
    from flexlb_cfg import OMIT, ConfigOverride
    from flexlb_test_framework.harness import (
        EnvSpec,
        MasterSpec,
        default_perf,
        fault_env_perf,
    )

    kwargs = {
        k: OMIT if v == {"omit": True} else v
        for k, v in plan["config_overrides"].items()
    }
    spec = EnvSpec(
        label="scenario",
        n_prefill=plan["n_prefill"],
        n_decode=plan["n_decode"],
        master_profile=profile,
        config_overrides=ConfigOverride(**kwargs),
        discovery=plan["discovery"],
        master_stable_window_s=plan.get("master_stable_window_s", 3),
        masters=(
            [
                MasterSpec(name="A", http_port=lease["master_base"]),
                MasterSpec(name="B", http_port=lease["master_base"] + 3),
            ]
            if plan.get("master_layout", "single") == "dual_standalone"
            else []
        ),
        perf=(
            fault_env_perf() if plan["perf_preset"] == "fault_env" else default_perf()
        ),
        master_env=({"FLEXLB_DEBUG_ENABLED": "true"} if plan["debug_enabled"] else {}),
        master_debug_log=plan.get("master_debug_log", False),
    )
    for key in (
        "prefill_cache_blocks",
        "decode_cache_blocks",
        "mock_auto_fetch",
        "mock_fetch_attach_timeout_ms",
    ):
        if key in plan:
            setattr(spec, key, plan[key])
    if plan["debug_enabled"]:
        spec.master_extra_args.append("--flexlb.debug.enabled=true")
    if "metric_whitelist" in plan:
        spec.master_extra_args.append(
            "--flexlb.monitor.metric-whitelist=" + plan["metric_whitelist"]
        )
    if "prefill_perf" in plan:
        spec.perf["prefill"] = dict(plan["prefill_perf"])
    if "prefill_max_waiting_batches" in plan:
        spec.perf.setdefault("prefill", {})["max_waiting_batches"] = plan[
            "prefill_max_waiting_batches"
        ]
    return spec


def configure_master_sync_log(spec, artifact_dir, env_epoch):
    """Choose a private per-setup log directory, with no arbitrary path input."""
    from pathlib import Path

    if spec.masters or type(env_epoch) is not int or env_epoch < 1:
        raise ValueError(
            "private sync log requires one master and a positive environment epoch"
        )
    log_dir = Path(artifact_dir).resolve() / f"master-sync-{env_epoch}"
    log_dir.mkdir(parents=True, exist_ok=False)
    spec.master_extra_args = [*spec.master_extra_args, f"--flexlb.log.path={log_dir}"]
    return log_dir / "sync.log"


class JavaMockBackend:
    def __init__(self, lease):
        self.lease = lease
        self.manager = self.raw_ops = None
        self.environments = []
        self.owned_processes = {}

    def remember_processes(self, env):
        for mp in [
            env.mock,
            env.master,
            env.zk_helper,
            *env.masters.values(),
            *env.victims.values(),
            *env.load_clients,
        ]:
            if mp is not None:
                self.owned_processes[mp.pid] = mp

    def setup(self, ctx, plan, deadline):
        return self._setup(ctx, plan, deadline)

    def _setup(self, ctx, plan, deadline, raw_config=None):
        if ":" in str(ctx.artifact_dir.resolve()):
            raise ValueError(
                "Java mock artifact path contains the JVM -Xlog colon delimiter"
            )
        workers = plan["n_prefill"] + plan["n_decode"]
        budget = ctx.instance["resource_budget"]
        bound = budget.get("max_environment_workers", budget["initial_workers"])
        if (
            workers > bound
            or workers + budget["max_dynamic_additions"] > self.lease["worker_capacity"]
        ):
            raise ValueError(
                "environment topology exceeds compiled or leased worker capacity"
            )
        from flexlb_test_framework.engine_ops import EngineOps
        from flexlb_test_framework.harness import EnvManager

        owner = self

        class OwnedManager(EnvManager):
            def _start_mock(self, env):
                owner.environments.append(env)
                master_spec = env.spec
                try:
                    if raw_config is not None:
                        # A negative startup probe targets Master parsing. The
                        # mock also parses its config to construct performance
                        # models, so feeding it the invalid document prevents
                        # the target Master from ever being launched.
                        env.spec = replace(master_spec, raw_config=None)
                    return super()._start_mock(env)
                finally:
                    env.spec = master_spec
                    owner.remember_processes(env)

            def start_master(self, env, *args, **kwargs):
                owner.remember_processes(env)
                try:
                    return super().start_master(env, *args, **kwargs)
                finally:
                    owner.remember_processes(env)

            def start_master_instance(self, env, *args, **kwargs):
                owner.remember_processes(env)
                try:
                    return super().start_master_instance(env, *args, **kwargs)
                finally:
                    owner.remember_processes(env)

            def _stop_env_processes(self, env):
                # The executor registered cleanup before setup. Retain partial
                # references for its separately budgeted, observable teardown.
                if env not in owner.environments:
                    owner.environments.append(env)

        spec = make_env_spec(plan, ctx.instance["profile"], self.lease)
        # Keep the first-epoch artifact layout compatible with existing runs.
        # Later environments never overwrite its config, logs or cleanup proof.
        artifact_dir = (
            ctx.artifact_dir
            if ctx.env_epoch == 1
            else ctx.artifact_dir / f"environment-epoch-{ctx.env_epoch}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.current_artifact_dir = artifact_dir
        if raw_config is not None:
            spec.raw_config = raw_config
        sync_log_path = (
            configure_master_sync_log(spec, artifact_dir, ctx.env_epoch)
            if plan.get("master_sync_log", False)
            else None
        )
        private_log = sync_log_path.parent if sync_log_path else None
        if not spec.masters and private_log is None:
            private_log = (artifact_dir / "master-logs").resolve()
            private_log.mkdir(exist_ok=False)
            spec.master_extra_args.append(f"--flexlb.log.path={private_log}")
        # Dual Master startup already assigns one private directory per owner.
        # A single Master always gets one too, not only negative-test probes.
        ctx.master_log_dir = private_log
        self.manager = OwnedManager(artifact_dir / "environment")
        (artifact_dir / "environment.json").write_text(
            json.dumps(
                dict(
                    fingerprint=spec.fingerprint(),
                    lease=self.lease,
                    resolved_config=plan["resolved_config"],
                    raw_config=raw_config,
                    raw_config_target="master" if raw_config is not None else None,
                    env_epoch=ctx.env_epoch,
                    master_log_dir=str(private_log) if private_log else None,
                    master_sync_log_path=str(sync_log_path) if sync_log_path else None,
                ),
                indent=2,
            )
            + "\n"
        )
        env = self.manager.ensure(spec)
        env.master_log_dir = private_log
        env.master_sync_log_path = sync_log_path
        ctx.master_sync_log_path = sync_log_path
        deadline.check()
        if (
            env.base_grpc_port != self.lease["mock_base"]
            or env.master_http_port != self.lease["master_base"]
            or env.master_management_port != self.lease["master_base"] + 1
        ):
            raise RuntimeError("harness changed a leased port")
        self.raw_ops = EngineOps("127.0.0.1", env.master_http_port, env.mock_http_port)
        ops = BoundedOps(self.raw_ops, ctx.instance["resource_budget"], self.lease)
        return env, ops

    def probe_startup(self, ctx, plan, raw_config, deadline):
        """Actually launch the raw config; parsing a Python mirror is no probe."""
        first = len(self.environments)
        error = None
        try:
            ctx.env, ctx.ops = self._setup(ctx, plan, deadline, raw_config=raw_config)
        except RuntimeError as exc:
            error = repr(exc)
        # Timeout/interrupt/IO failure is not an expected parser rejection.
        deadline.check()
        masters = [
            env.master for env in self.environments[first:] if env.master is not None
        ]
        if not masters:
            raise RuntimeError("startup probe has no owned Master process evidence")
        logs = []
        app_log = self.current_artifact_dir / "master-logs" / "application.log"
        for path in [app_log, *[mp.log_file for mp in masters]]:
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 262144))
                    logs.append(
                        dict(
                            path=str(path),
                            tail=stream.read(262144).decode("utf-8", errors="replace"),
                        )
                    )
        return dict(
            startup_error=error,
            started=error is None,
            master_pids=[mp.pid for mp in masters],
            master_returncodes=[mp.proc.poll() for mp in masters],
            logs=logs,
            current_absent_before_cleanup=self.manager.current is None,
        )

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
        processes = list(self.owned_processes.values())
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
        current = getattr(self, "current_artifact_dir", ctx.artifact_dir)
        if current == ctx.artifact_dir:
            current = ctx.artifact_dir / f"environment-epoch-{ctx.env_epoch}"
        current.mkdir(parents=True, exist_ok=True)
        (current / "process-cleanup.json").write_text(
            (ctx.artifact_dir / "process-cleanup.json").read_text()
        )
        if remaining_pids:
            raise StageTimeout(f"owned processes not reaped: {remaining_pids}")
        if self.manager is not None:
            self.manager.current = None
