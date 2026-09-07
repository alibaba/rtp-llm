"""Elastic adapter contracts without JVMs, sockets or production changes."""

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexlb_ft.scenario.actions import elastic as e


class Deadline:
    def __init__(self, remaining=100):
        self.end = time.monotonic() + remaining

    def remaining(self):
        return self.end - time.monotonic()

    def check(self):
        if self.remaining() <= 0:
            raise TimeoutError()

    def sleep(self, seconds):
        self.check()


class Call:
    def __init__(self, value=None, error=None, frames=()):
        self.value, self.error, self.frames = value, error, frames
        self.cancelled = False

    def result(self):
        if self.error:
            raise self.error
        return self.value

    def __iter__(self):
        yield from self.frames
        if self.error:
            raise self.error

    def cancel(self):
        self.cancelled = True
        return True


def frame(finished=False, code=0):
    return NS(
        HasField=lambda name: bool(code),
        error_info=NS(error_code=code, error_message="retired"),
        flatten_output=NS(finished=[finished]),
    )


def ops(schedule_error=None, stream_error=None, frames=(), enqueued=True):
    response = NS(code=200, success=True, enqueued_by_master=enqueued)
    scheduled = Call(response, schedule_error)
    streamed = Call(error=stream_error, frames=frames)
    timeouts = []

    def future(req, timeout):
        timeouts.append(("Schedule", timeout))
        return scheduled

    def fetch(req, timeout):
        timeouts.append(("FetchResponse", timeout))
        return streamed

    result = NS(
        _channel=lambda target: None,
        master_target=lambda: "master",
        build_schedule_request=lambda rid, **shape: rid,
        prefill_addr=lambda resp: "p:1",
        next_request_id=lambda: 1,
        schedule_pb2_grpc=NS(
            FlexlbServiceStub=lambda channel: NS(Schedule=NS(future=future))
        ),
        pb2_grpc=NS(RpcServiceStub=lambda channel: NS(FetchResponse=fetch)),
        pb2=NS(FetchRequestPB=lambda **kw: kw),
    )
    return result, timeouts, scheduled, streamed


class ElasticEvidenceTests(unittest.TestCase):
    def run_record(self, **kwargs):
        operation, timeouts, scheduled, streamed = ops(**kwargs)
        records = e.RecordedRequests(operation, 7)
        record = records.issue(123, time.monotonic)
        records.run(record, {}, timeout_s=5)
        return records, records.snapshot_records()[0], timeouts

    def test_finished_and_transport_success_required(self):
        records, record, timeouts = self.run_record(frames=[frame(True)])
        self.assertTrue(e.request_success(record))
        self.assertTrue(e.completeness(records.snapshot_records())["zero_errors"])
        self.assertTrue(all(0 < timeout <= 5 for _, timeout in timeouts))
        self.assertEqual(record["stream"]["method"], "FetchResponse")
        self.assertIsNone(record["endpoint_generation"])

    def test_empty_transport_is_not_business_success(self):
        records, record, _ = self.run_record()
        self.assertIsNotNone(record["transport_terminal_s"])
        self.assertFalse(e.request_success(record))
        self.assertEqual(
            e.completeness(records.snapshot_records())["failed_request_ids"], [123]
        )

    def test_typed_retirement_preserved_even_with_finished(self):
        _, record, _ = self.run_record(frames=[frame(True, 8510)])
        self.assertEqual(record["business_error_code"], 8510)
        self.assertFalse(e.request_success(record))

    def test_schedule_failure_not_misattributed_to_fetch(self):
        _, record, timeouts = self.run_record(
            schedule_error=TimeoutError("Schedule expired")
        )
        self.assertEqual(record["schedule"]["status"], "ERROR")
        self.assertIsNone(record["stream"]["method"])
        self.assertEqual(len(timeouts), 1)

    def test_fetch_failure_after_finished_remains_failure(self):
        _, record, _ = self.run_record(
            frames=[frame(True)], stream_error=TimeoutError("Fetch expired")
        )
        self.assertTrue(record["business_finished"])
        self.assertFalse(e.request_success(record))
        self.assertEqual(record["stream"]["status"], "ERROR")

    def test_records_are_deep_copies_and_nonterminal_kept(self):
        records = e.ClientRecords(1)
        record = records.issue(4, lambda: 10)
        snapshot = records.snapshot_records()
        snapshot[0]["schedule"]["status"] = "corrupt"
        self.assertIsNone(records.snapshot_records()[0]["schedule"]["status"])
        self.assertEqual(records.snapshot_cohort(10, 11)["record_count"], 1)
        self.assertEqual(records.snapshot_cohort(10, 11, "terminal")["record_count"], 0)
        self.assertFalse(e.completeness(records.snapshot_records())["result_complete"])
        self.assertFalse(e.completeness([])["zero_errors"])

    def test_cancellation_is_explicit_and_idempotent(self):
        operation, *_ = ops()
        records = e.RecordedRequests(operation, 2)
        record = records.issue(9, time.monotonic)
        call = Call()
        records._activate(record, call)
        records.cancel_active("test")
        stamp = record["cancel"]["requested_s"]
        records.cancel_active("second")
        self.assertTrue(call.cancelled)
        self.assertEqual(record["cancel"]["requested_s"], stamp)
        self.assertEqual(record["cancel"]["reason"], "test")
        self.assertFalse(e.request_success(record))

    def test_cancel_before_call_activation_is_not_lost(self):
        operation, *_ = ops()
        records = e.RecordedRequests(operation, 1)
        record = records.issue(1, time.monotonic)
        records.cancel_active()
        call = Call()
        records._activate(record, call)
        self.assertTrue(call.cancelled)

    def test_http_success_does_not_imply_drained(self):
        families = dict(hot="p0", cold="p1")
        ctx = NS(
            ops=object(),
            clock=time.monotonic,
            resource=lambda *args: families,
            register_resource=lambda *args, **kwargs: {
                "kind": "snapshot",
                "id": "1",
                "env_epoch": 1,
            },
        )
        snap = {
            "p0": {"cache_key_set": list(range(90))},
            "p1": {"cache_key_set": list(range(10))},
        }
        with patch.object(e, "_snapshot", return_value=snap), patch.object(
            e, "_http", return_value={"drained": False}
        ):
            output = e._scale(ctx, {"families": {}, "victim": "hot"}, Deadline())
        self.assertEqual(output.checks[-1].status, "FAIL")
        self.assertFalse(output.output["drained"])

    def test_scale_refuses_short_client_budget_before_side_effect(self):
        ctx = NS(resource=lambda *args: dict(hot="p0", cold="p1"))
        snap = {
            "p0": {"cache_key_set": list(range(90))},
            "p1": {"cache_key_set": list(range(10))},
        }
        with patch.object(e, "_snapshot", return_value=snap), patch.object(
            e, "_http"
        ) as http:
            with self.assertRaises(TimeoutError):
                e._scale(ctx, {"families": {}, "victim": "hot"}, Deadline(5))
            http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
