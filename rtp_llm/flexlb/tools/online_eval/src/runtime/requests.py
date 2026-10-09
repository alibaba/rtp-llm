"""Shared client records and completion evidence; no scenario dependencies."""

import copy
import threading


class ClientRecords:
    """Thread-safe observation contract shared with the observation adapter."""

    def __init__(self, env_epoch):
        self.env_epoch = env_epoch
        self._lock = threading.RLock()
        self._records = []

    def snapshot_records(self):
        with self._lock:
            return copy.deepcopy(self._records)

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
