"""Protocol checks use independent owner evidence and real bounded fake-RPC drivers."""

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_ft.scenario.actions import status_protocol as status
from flexlb_ft.scenario.contracts import PlanContext
from flexlb_ft.scenario.runtime import Deadline, RuntimeContext
from test_scenario_backend import Ops


def owner_frame(decode_load=0):
    return {
        "inflight": {
            "scheduler_inflight": 0,
            "prefill_endpoints": [
                {"ip_port": "p", "inflight_batches": 0, "inflight_requests": 0}
            ],
            "decode_endpoints": [
                {
                    "ip_port": "d",
                    "reserved_total": decode_load,
                    "master_queued": 0,
                    "confirmed_accepted": 0,
                    "confirmed_running": 0,
                    "total_load": decode_load,
                    "engine_load": 0,
                    "active_dispatch_permits": 0,
                    "engine_capacity_used": 0,
                }
            ],
        }
    }


class StatusProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ctx = RuntimeContext({}, None, self.tmp.name, time.monotonic, time.sleep)
        self.ctx.env_epoch = 1
        self.ctx.env = NS(
            mock_http_port=10000, master_http_port=10001, master_management_port=10002
        )
        self.ctx.ops = Ops()
        self.ctx.instance_deadline_s = time.monotonic() + 5
        self.plan = PlanContext("test", {}, profiles=("batch-window",))
        self.addCleanup(lambda: self.ctx.cleanup(2))

    def deadline(self):
        return Deadline(time.monotonic() + 2, time.monotonic, time.sleep)

    def test_decode_total_load_and_fingerprint_never_use_missing_legacy_field(self):
        frame = owner_frame(3)
        self.assertEqual(3, status.metric(frame, "decode_total_load"))
        self.assertNotEqual(
            status.metric(frame, "fingerprint"),
            status.metric(owner_frame(), "fingerprint"),
        )
        del frame["inflight"]["decode_endpoints"][0]["total_load"]
        with self.assertRaises(KeyError):
            status.metric(frame, "decode_total_load")

    def test_unknown_fault_fields_and_channels_rejected_before_io(self):
        for params in [
            {"fault": "status_fake_task", "config": {"batchId": 1}},
            {"fault": "missing"},
            {"fault": "enqueue_ack_drop", "expected_http": [200, 400]},
            {"fault": "status_fake_task", "selection": "landing"},
        ]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                status.validate_control(params, self.plan)

    def test_injection_cleanup_registered_before_partial_failure_and_epoch_safe(self):
        p = status.validate_control({"fault": "status_suppress_finished"}, self.plan)
        calls = []

        def http(ctx, server, path, deadline, body=None, allowed=(200,), text=False):
            calls.append(body)
            if body["engine"] == "p1" and body["enabled"]:
                raise OSError("injection reply missing")
            return 200, {}

        with patch.object(status, "_targets", return_value=["p0", "p1"]), patch.object(
            status, "_http", side_effect=http
        ):
            with self.assertRaises(OSError):
                status.execute_control(self.ctx, p, self.deadline())
            self.assertEqual(1, len(self.ctx._cleanup))
            results = self.ctx.cleanup(1)
        self.assertEqual(["PASS"], [r["status"] for r in results])
        self.assertEqual([True, True, False, False], [x["enabled"] for x in calls])
        with patch.object(status, "_targets", return_value=["p0"]), patch.object(
            status, "_http", return_value=(200, {})
        ):
            status.execute_control(self.ctx, p, self.deadline())
        self.ctx.env_epoch = 2
        with patch.object(status, "_http") as http:
            results = self.ctx.cleanup(1)
        http.assert_not_called()
        self.assertEqual("ERROR", results[0]["status"])

    def test_prepared_cohort_is_bound_before_dispatch_and_deferred_wait_fetches(self):
        p = status.validate_prepare(
            {"count": 3, "concurrency": 2, "consume": "deferred"}, self.plan
        )
        result = status.execute_prepare(self.ctx, p, self.deadline())
        requests = self.ctx.resource(result.output["requests"], "requests")
        before = requests.snapshot_records()
        ids = [r["wire_request_id"] for r in before]
        self.assertEqual(3, len(set(ids)))
        self.assertTrue(all(r["issued_s"] is None for r in before))
        self.assertEqual(0, self.ctx.ops.fetch_count)
        requests.dispatch(self.deadline())
        self.assertEqual(
            ids, [r["wire_request_id"] for r in requests.snapshot_records()]
        )
        self.assertEqual(0, self.ctx.ops.fetch_count)
        self.assertEqual(
            {"completed": True, "error_count": 0}, requests.wait(self.deadline())
        )
        self.assertEqual(3, self.ctx.ops.fetch_count)
        self.assertTrue(all(event.is_set() for event in requests.done))
        self.assertEqual(2, len(requests.exit_records))

    def test_undispatched_records_cannot_pass_outcome_checks(self):
        p = status.validate_prepare({"count": 1}, self.plan)
        result = status.execute_prepare(self.ctx, p, self.deadline())
        with self.assertRaises(RuntimeError):
            status.execute_outcomes(
                self.ctx,
                {"requests": result.output["requests"], "success_min": 0},
                self.deadline(),
            )

    def test_ack_and_execution_errors_are_separate_normal_checks(self):
        def record(phase):
            return dict(
                issued_s=1,
                consumer_exit_s=2,
                business_finished=False,
                business_error_code=8500 if phase == "execution" else None,
                business_error_message="injected",
                cancel={"requested_s": None},
                schedule={
                    "status": "OK" if phase == "execution" else "REJECTED",
                    "error": "8500",
                },
                stream={
                    "status": "OK" if phase == "execution" else None,
                    "started_s": 1 if phase == "execution" else None,
                },
            )

        for phase in ("schedule", "execution"):
            provider = NS(snapshot_records=lambda: [record(phase)])
            handle = self.ctx.register_resource("requests", provider)
            for expected in ("schedule", "execution"):
                result = status.execute_outcomes(
                    self.ctx,
                    {"requests": handle, "error_code": 8500, "failure_phase": expected},
                    self.deadline(),
                )
                self.assertEqual(
                    "PASS" if phase == expected else "FAIL", result.checks[0].status
                )
        provider = NS(
            snapshot_records=lambda: [
                dict(
                    record("execution"),
                    stream={"status": "DEADLINE_EXCEEDED", "started_s": 1},
                )
            ]
        )
        handle = self.ctx.register_resource("requests", provider)
        with self.assertRaises(TimeoutError):
            status.execute_outcomes(self.ctx, {"requests": handle}, self.deadline())

    def test_stability_false_is_not_ignored_and_missing_owner_source_is_error(self):
        frames = [owner_frame(), owner_frame()]
        result = status._frozen(self.ctx, "test", {"frames": frames})
        args = {
            "snapshot": result.output["snapshot"],
            "metric": "scheduler",
            "aggregate": "stable",
            "op": "eq",
            "expected": False,
        }
        self.assertEqual(
            "FAIL",
            status.execute_check(self.ctx, args, self.deadline()).checks[0].status,
        )
        frames = [{}]
        result = status._frozen(self.ctx, "missing", {"frames": frames})
        args["snapshot"] = result.output["snapshot"]
        with self.assertRaises(KeyError):
            status.execute_check(self.ctx, args, self.deadline())

    def test_sampling_source_failure_is_not_an_empty_success(self):
        p = status.validate_sample({}, self.plan)
        with patch.object(status, "_http", side_effect=OSError("source missing")):
            with self.assertRaises(OSError):
                status.execute_sample(self.ctx, p, self.deadline())


if __name__ == "__main__":
    unittest.main()
