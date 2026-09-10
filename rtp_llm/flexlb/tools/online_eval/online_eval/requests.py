"""Shared client evidence and bounded request execution; no scenario dependencies."""

import copy
import threading
import time


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
