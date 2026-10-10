"""Finite request issuance, stream ownership and typed RPC error evidence."""

import base64
import hashlib
import json
import threading

from runtime.requests import ClientRecords, request_success
from scenario.runtime import StageTimeout


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


class RequestBatch(ClientRecords):
    """Finite issuance; deferred mode really performs zero Fetch until wait."""

    def __init__(self, ctx, params):
        super().__init__(ctx.env_epoch)
        self.ctx, self.ops, self.params = ctx, ctx.ops, params
        self.entries = []
        self.cancelled = False
        self.artifact = ctx.artifact_dir / f"requests-{ctx.resource_count+1}.json"

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
                    from runtime.engine_ops import QOS_LEVEL_HEADER

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
