"""Cancellation primitives reuse real consumer threads with bounded fake RPCs."""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_ft.scenario.actions import cancel
from flexlb_ft.scenario.contracts import PlanContext
from flexlb_ft.scenario.runtime import Deadline, RuntimeContext
from test_scenario_backend import Ops


class CancelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ctx = RuntimeContext({}, None, self.tmp.name, time.monotonic, time.sleep)
        self.ctx.env_epoch = 1
        self.ctx.env = NS()
        self.ctx.ops = Ops()
        self.ctx.instance_deadline_s = time.monotonic() + 5
        self.plan = PlanContext("test", {}, profiles=("batch-window",))
        self.addCleanup(lambda: self.ctx.cleanup(2))

    def deadline(self):
        return Deadline(time.monotonic() + 2, time.monotonic, time.sleep)

    def prepare(self, **params):
        result = cancel.execute_prepare(
            self.ctx, cancel.validate_prepare(params, self.plan), self.deadline()
        )
        return result.output["requests"], self.ctx.resource(
            result.output["requests"], "requests"
        )

    def test_manual_open_performs_no_fetch_before_explicit_stage(self):
        handle, cohort = self.prepare(consume="manual")
        cohort.dispatch(self.deadline())
        self.assertEqual(0, self.ctx.ops.fetch_count)
        self.assertIsNone(cohort.snapshot_records()[0]["stream"]["started_s"])
        cohort.open_streams(self.deadline())
        for child in cohort.children:
            child.wait(self.deadline())
        cohort.prove_ended(self.deadline())
        self.assertEqual(1, self.ctx.ops.fetch_count)
        frame = {"records": cohort.snapshot_records()}
        self.assertEqual(1, cancel.metric(frame, "stream_ended"))
        self.assertEqual(1, cancel.metric(frame, "success"))

    def test_unknown_master_cancel_has_typed_false_and_no_worker_call(self):
        handle, _ = self.prepare()
        calls = []
        self.ctx.ops.schedule_pb2 = NS(
            FlexlbCancelRequestPB=lambda **kwargs: NS(**kwargs),
            CANCEL_REASON_CLIENT_CANCELLED=1,
        )
        self.ctx.ops.schedule_pb2_grpc.FlexlbServiceStub = lambda channel: NS(
            Cancel=lambda req, timeout: (
                calls.append((req.request_id, timeout)) or NS(found=False)
            )
        )
        with patch.object(self.ctx.ops.pb2_grpc, "RpcServiceStub") as worker:
            result = cancel.execute_rpc(
                self.ctx,
                {
                    "requests": handle,
                    "destination": "master",
                    "expected_rpc_statuses": ["NOT_FOUND"],
                },
                self.deadline(),
            )
        worker.assert_not_called()
        source = self.ctx.resource(result.output["snapshot"], "snapshot").to_dict()
        self.assertEqual(1, cancel.metric(source, "rpc_not_found"))
        self.assertEqual(1, len(calls))
        self.assertGreater(
            source["receipts"][0]["ended_s"], source["receipts"][0]["started_s"]
        )

    def test_route_is_required_for_engine_cancel(self):
        handle, _ = self.prepare()
        with self.assertRaisesRegex(RuntimeError, "actual scheduled route"):
            cancel.execute_rpc(
                self.ctx,
                {
                    "requests": handle,
                    "destination": "prefill",
                    "expected_rpc_statuses": [],
                },
                self.deadline(),
            )

    def test_missing_receipt_is_error_and_late_receipt_is_normal_failure(self):
        with self.assertRaises(RuntimeError):
            cancel.metric({"receipts": [{"rpc_status": "OK"}]}, "rpc_ok")
        snapshot = cancel.status._frozen(
            self.ctx,
            "late",
            {
                "anchor_s": 1,
                "frames": [
                    {
                        "capture_finished_s": 7,
                        "records": [{"wire_request_id": 1}],
                        "mock": {"p": {"cancelled_rids": [1]}},
                    }
                ],
            },
        )
        result = cancel.execute_check(
            self.ctx,
            {
                "snapshot": snapshot.output["snapshot"],
                "metric": "engine_cancelled",
                "op": "eq",
                "expected": 1,
                "within_s": 5,
            },
            self.deadline(),
        )
        self.assertEqual("FAIL", result.checks[0].status)

    def test_unverified_exit_cannot_count_as_ended(self):
        record = {
            "stream": {"ended_s": 2},
            "consumer_done": True,
            "consumer_completion_verified": False,
            "consumer_exit_s": 2,
            "transport_terminal_s": 2,
        }
        self.assertEqual(0, cancel.metric({"records": [record]}, "stream_ended"))
        del record["stream"]["ended_s"]
        with self.assertRaises(KeyError):
            cancel.metric({"records": [record]}, "stream_ended")

    def test_async_schedule_drop_proves_the_owned_future_exit(self):
        stopped = threading.Event()
        entered = threading.Event()
        cancelled_error = type(
            "FutureCancelledError", (Exception,), {"__module__": "grpc"}
        )

        def result(timeout):
            entered.set()
            if not stopped.wait(timeout):
                raise TimeoutError("fake Schedule was never cancelled")
            raise cancelled_error()

        self.ctx.ops.future = lambda *args, **kwargs: NS(
            result=result,
            cancel=lambda: (stopped.set() or True),
            cancelled=stopped.is_set,
        )
        handle, cohort = self.prepare(consume="manual")
        cohort.begin(self.deadline())
        self.assertTrue(entered.wait(1))
        cancel.execute_transport(
            self.ctx, {"requests": handle, "phase": "schedule"}, self.deadline()
        )
        cohort.join_submission(self.deadline())
        self.assertFalse(cohort.errors)
        self.assertEqual(0, self.ctx.ops.fetch_count)
        self.assertEqual(
            1,
            cancel.metric({"records": cohort.snapshot_records()}, "schedule_cancelled"),
        )
        self.assertTrue(all(not t.is_alive() for t in cohort.threads))

    def test_group_rejects_duplicate_before_any_submission(self):
        handle, cohort = self.prepare()
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            cancel.execute_group(
                self.ctx, {"cohorts": [handle, handle]}, self.deadline()
            )
        self.assertFalse(cohort.dispatched)

    def test_client_only_window_does_not_query_http_or_mock(self):
        handle, cohort = self.prepare(consume="manual")
        cohort.dispatch(self.deadline())
        with patch.object(cancel.status, "_http") as http, patch.object(
            cancel.status, "_mock"
        ) as mock:
            result = cancel.execute_observe(
                self.ctx,
                {
                    "requests": handle,
                    "include": ["client_records"],
                    "duration_s": 0,
                    "interval_s": 0.01,
                },
                self.deadline(),
            )
        http.assert_not_called()
        mock.assert_not_called()
        frame = self.ctx.resource(result.output["snapshot"], "snapshot").to_dict()[
            "frames"
        ][-1]
        self.assertEqual(0, cancel.metric(frame, "first_output"))
        cohort.open_streams(self.deadline())
        cohort.prove_ended(self.deadline())

    def test_delta_until_waits_for_new_cancel_not_existing_total(self):
        baseline = cancel.status._frozen(
            self.ctx,
            "baseline",
            {"frames": [{"mock": {"p": {"rpc_counts": {"cancel": 7}}}}]},
        )
        frames = iter([{"mock": {"p": {"rpc_counts": {"cancel": n}}}} for n in [7, 8]])
        with patch.object(
            cancel.status, "_frame", side_effect=lambda *args: next(frames)
        ):
            result = cancel.execute_observe(
                self.ctx,
                {
                    "baseline": baseline.output["snapshot"],
                    "include": ["mock"],
                    "duration_s": 0.2,
                    "interval_s": 0.01,
                    "until": {"metric": "cancel_rpc_count", "op": "ge", "value": 1},
                },
                self.deadline(),
            )
        source = self.ctx.resource(result.output["snapshot"], "snapshot").to_dict()
        self.assertEqual(2, len(source["frames"]))

    def test_initial_lifecycle_programs_keep_batch_and_nonbatch_contracts_explicit(
        self,
    ):
        from flexlb_ft.scenario.catalog import handlers
        from flexlb_ft.scenario.compiler import compile_scenarios, plan_counts
        from flexlb_ft.scenario.loader import load_scenarios

        root = Path(__file__).resolve().parents[1] / "scenarios/cancel"
        registry = handlers()
        registry.update({h.name: h for h in cancel.HANDLERS})
        plans = compile_scenarios(load_scenarios(root), handlers=registry)
        self.assertEqual(38, plan_counts(plans)["instances"])
        for plan in plans:
            ids = {s["id"] for s in plan["stages"]}
            if plan["variant_id"] == "basic_batch":
                self.assertIn("engine_receives_cancel_within_five", ids)
            if plan["variant_id"] == "basic_nonbatch":
                self.assertIn("first_worker_cancel", ids)
                self.assertNotIn("engine_receives_cancel_within_five", ids)
            if plan["variant_id"] == "idempotent_batch":
                self.assertIn("second_cancel_not_forwarded", ids)
                self.assertNotIn("closing_drain_scheduler", ids)


if __name__ == "__main__":
    unittest.main()
