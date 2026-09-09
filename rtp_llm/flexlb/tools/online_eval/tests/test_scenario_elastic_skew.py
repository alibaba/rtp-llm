"""Skew-specific observation contracts, with actual request consumers."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_test_framework.scenario.actions import elastic as e
from flexlb_test_framework.scenario.actions.elastic_skew_requests import (
    SkewFlow,
    SkewRecordedRequests,
    summary,
)
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext
from test_scenario_backend import Ops
from test_scenario_elastic_runtime import Clock


class SkewTests(unittest.TestCase):
    def test_remove_call_duration_and_diagnostic_alive_windows(self):
        clock = Clock()
        clock.now = 25
        objects = dict(
            observation=NS(snapshot=lambda: {}),
            scale=dict(started_s=0, ended_s=25),
            baseline=dict(hit_rate=0.8),
            transient=dict(hit_rate=0.4),
            families=dict(hot="p0", cold="p1"),
        )
        saved, calls = [], []

        def window(raw, start, end, survivor=None):
            calls.append((start, end, survivor))
            return dict(hit_rate=0.8, requested=100)

        ctx = NS(
            clock=clock,
            ops=NS(master_http_port=1),
            resource=lambda value, kind: objects[value],
            register_resource=lambda kind, value, **kw: saved.append(value),
        )
        params = dict(
            observation="observation",
            scale="scale",
            baseline="baseline",
            transient="transient",
            families="families",
            victim="hot",
            phase="transient",
        )
        with patch.object(e, "metric_window", window):
            e._window(ctx, params, Deadline(200, clock, clock.sleep))
            self.assertEqual(calls[0][:2], (0, 45))
            params["phase"] = "steady"
            with patch(
                "flexlb_test_framework.harness.http_post_json",
                return_value=(200, {"worker_summary": {"PREFILL": {"alive": 0}}}),
            ):
                e._window(ctx, params, Deadline(200, clock, clock.sleep))
        self.assertEqual(calls[1], (115, 135, "p1"))
        self.assertFalse(saved[-1]["alive_ok"])
        obs = saved[-1]["observations"]
        self.assertEqual(
            [w["start_s"] for w in obs["windows_10s"]], list(range(0, 130, 10))
        )
        self.assertEqual(obs["recovery_duration_s"], 5)
        self.assertAlmostEqual(obs["rebound_floor"], 0.6)

    def test_cache_change_resets_quiet_period(self):
        clock = Clock()

        def snapshot(*args):
            return {
                "p0": {"cache_key_set": [1 if clock() < 2 else 2]},
                "p1": {"cache_key_set": []},
            }

        with patch.object(e, "_snapshot", snapshot):
            result = e._skew_cache_quiet(
                NS(clock=clock), ["p0", "p1"], Deadline(20, clock, clock.sleep)
            )
        self.assertTrue(result["quiet"])
        self.assertEqual(clock(), 5.5)
        self.assertEqual(result["last_change_s"], {"p0": 2, "p1": 0})

    def context(self, root):
        ctx = RuntimeContext({}, None, root, time.monotonic, time.sleep)
        ctx.instance_deadline_s = time.monotonic() + 30
        ctx.ops = Ops(True)
        return ctx

    def test_recovery_twenty_requests_ten_concurrent_three_keys_and_private_records(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            ctx = self.context(root)
            original = ctx.ops.future
            shapes, waits, lock = [], [], threading.Lock()
            barrier = threading.Barrier(10)

            def future(req, timeout, metadata=None):
                with lock:
                    shapes.append((req, timeout))
                    number = len(shapes)
                response = original(req, timeout, metadata)

                def result(timeout):
                    if number <= 10:
                        barrier.wait(5)
                    return response.result(timeout)

                return NS(result=result, cancel=response.cancel)

            ctx.ops.schedule_pb2_grpc = NS(
                FlexlbServiceStub=lambda channel: NS(Schedule=NS(future=future))
            )
            from flexlb_test_framework.scenario.observed import ObservedRequestBatch

            join = ObservedRequestBatch._join_window

            def observe(batch, entry, seconds):
                waits.append(seconds)
                return join(batch, entry, seconds)

            with patch.object(ObservedRequestBatch, "_join_window", observe):
                output = e._recovery(ctx, {}, Deadline(ctx.instance_deadline_s))
            self.assertEqual(output.output["success_rate"], 1)
            self.assertEqual(len(shapes), 20)
            self.assertEqual(waits, [15] * 20)
            for (rid, shape), timeout in shapes:
                self.assertEqual(shape["input_len"], 2048)
                self.assertEqual(shape["output_len"], 2)
                self.assertEqual(shape["block_keys"], [rid * 100 + j for j in range(3)])
                self.assertLessEqual(timeout, 30)
            self.assertEqual((ctx.ops.fetch_count, ctx.ops.generate_count), (20, 0))
            files = list(Path(root).glob("skew/recovery/*/aggregate-record.json"))
            self.assertEqual(len(files), 20)
            for f in files:
                record = json.loads(f.read_text())
                self.assertEqual(record["wire_request_id"], int(f.parent.name))
                self.assertTrue(
                    record["consumer_done"] and record["consumer_completion_verified"]
                )
            self.assertTrue(all(r["status"] == "PASS" for r in ctx.cleanup(5)))

    def test_cancel_before_registration_preserves_failure_without_rpc(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = self.context(root)
            records = SkewRecordedRequests(ctx, "seed", 30)
            record = records.issue(77, ctx.clock)
            records.cancel_active()
            records.run(record, dict(input_len=10, output_len=2, block_keys=[1]))
            with self.assertRaisesRegex(RuntimeError, "before child registration"):
                summary(records.snapshot_records())
            self.assertEqual(ctx.ops.responses, [])
            self.assertFalse(record.get("consumer_done", False))
            records.cleanup(Deadline(time.monotonic() + 2))

    def test_cancel_during_schedule_open_cancels_call_before_result(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = self.context(root)
            records = SkewRecordedRequests(ctx, "seed", 30)
            cancelled, waited = [], []

            def future(req, timeout):
                records.cancel_active("racing Schedule creation")
                return NS(
                    cancel=lambda: cancelled.append(True),
                    result=lambda **kw: waited.append(True),
                )

            ctx.ops.schedule_pb2_grpc = NS(
                FlexlbServiceStub=lambda channel: NS(Schedule=NS(future=future))
            )
            record = records.issue(79, ctx.clock)
            records.run(record, dict(input_len=10, output_len=2, block_keys=[1]))
            self.assertTrue(cancelled)
            self.assertEqual(waited, [])
            self.assertTrue(records.worker_done[79].is_set())
            self.assertEqual(ctx.ops.fetch_count, 0)
            with self.assertRaisesRegex(
                RuntimeError, "cancelled during Schedule startup"
            ):
                summary(records.snapshot_records())
            records.cleanup(Deadline(time.monotonic() + 2))

    def test_cancel_during_rpc_open_cancels_returned_call_and_never_claims_exit(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = self.context(root)
            records = SkewRecordedRequests(ctx, "pump", 30)
            original = ctx.ops.fetch

            def fetch(req, timeout):
                call = original(req, timeout)
                records.cancel_active("racing open")
                return call

            ctx.ops.pb2_grpc = NS(
                RpcServiceStub=lambda channel: NS(FetchResponse=fetch)
            )
            record = records.issue(78, ctx.clock)
            records.run(record, dict(input_len=10, output_len=2, block_keys=[1]))
            self.assertTrue(ctx.ops.streams[0].cancelled.is_set())
            self.assertFalse(record.get("consumer_done", False))
            with self.assertRaisesRegex(
                RuntimeError, "cancelled during stream startup"
            ):
                summary(records.snapshot_records())
            records.cleanup(Deadline(time.monotonic() + 2))

    def test_schedule_timeout_is_failed_request_not_missing_consumer_error(self):
        import grpc

        for phase, end_wait in (("seed", 30), ("pump", 30), ("recovery", 15)):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as root:
                ctx = self.context(root)
                records = SkewRecordedRequests(ctx, phase, end_wait)

                def result(timeout):
                    raise grpc.FutureTimeoutError()

                ctx.ops.schedule_pb2_grpc = NS(
                    FlexlbServiceStub=lambda channel: NS(
                        Schedule=NS(
                            future=lambda *a, **kw: NS(
                                result=result, cancel=lambda: True
                            )
                        )
                    )
                )
                record = records.issue(20060, ctx.clock)
                records.run(record, dict(input_len=10240, output_len=2, block_keys=[1]))
                outcome = summary(records.snapshot_records())
                self.assertTrue(outcome["result_complete"])
                self.assertFalse(outcome["zero_errors"])
                self.assertEqual(outcome["failed_request_ids"], [20060])
                self.assertEqual(outcome["completed"], 0)
                self.assertNotIn("execution_error", record)
                self.assertEqual(record["request_error"]["kind"], "FutureTimeoutError")
                self.assertEqual(
                    record["request_error"]["detail"], "FutureTimeoutError()"
                )
                self.assertEqual(record["transport_record"]["rpc_observation"], {})
                self.assertEqual(record["schedule"]["error"], "FutureTimeoutError()")
                self.assertIsNone(record["stream"]["method"])
                self.assertFalse(record.get("consumer_done", False))
                self.assertTrue(record["rpc_observation"]["schedule_failed"])
                records.cleanup(Deadline(time.monotonic() + 2))

    def test_recovery_continues_after_schedule_failure_and_verdict_preserves_p6(self):
        import grpc

        with tempfile.TemporaryDirectory() as root:
            ctx = self.context(root)
            original = ctx.ops.future

            def future(req, timeout, metadata=None):
                if req[0] == 1:

                    def failed(timeout):
                        raise grpc.FutureTimeoutError()

                    return NS(result=failed, cancel=lambda: True)
                return original(req, timeout, metadata)

            ctx.ops.schedule_pb2_grpc = NS(
                FlexlbServiceStub=lambda channel: NS(Schedule=NS(future=future))
            )
            output = e._recovery(ctx, {}, Deadline(ctx.instance_deadline_s))
            self.assertEqual(output.output["success_rate"], 0.95)
            records = ctx.resource(
                output.output["requests"], "requests"
            ).snapshot_records()
            failed = next(r for r in records if r["wire_request_id"] == 1)
            snapshots = dict(
                baseline=dict(hit_rate=1),
                transient=dict(hit_rate=1),
                steady=dict(hit_rate=1, waiting_peak=0, occupancy_peak=0.5),
                scale=dict(response=dict(drained=False)),
                flow_result=summary([failed]),
            )
            params = {
                k: ctx.register_resource("snapshot", v) for k, v in snapshots.items()
            }
            params.update(recovery=output.output["requests"], victim="hot")
            checks = {
                c.id: c.status
                for c in e._verdict(
                    ctx, params, Deadline(ctx.instance_deadline_s)
                ).checks
            }
            self.assertEqual(checks["P6"], "FAIL")
            self.assertEqual(checks["P2"], "PASS")
            self.assertEqual(ctx.ops.fetch_count, 19)
            self.assertTrue(all(r["status"] == "PASS" for r in ctx.cleanup(5)))

    def test_schedule_rpc_error_is_request_failure_but_parent_timeout_is_not(self):
        import grpc

        class RpcFailure(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.DEADLINE_EXCEEDED

        for expired in (False, True):
            with self.subTest(expired=expired), tempfile.TemporaryDirectory() as root:
                clock = Clock()
                ctx = RuntimeContext({}, None, root, clock, clock.sleep)
                ctx.ops = Ops(True)
                ctx.instance_deadline_s = 20
                records = SkewRecordedRequests(ctx, "pump", 30)

                def result(timeout):
                    if expired:
                        clock.sleep(20)
                    raise RpcFailure()

                ctx.ops.schedule_pb2_grpc = NS(
                    FlexlbServiceStub=lambda channel: NS(
                        Schedule=NS(
                            future=lambda *a, **kw: NS(
                                result=result, cancel=lambda: True
                            )
                        )
                    )
                )
                record = records.issue(1, clock)
                records.run(record, dict(input_len=10, output_len=2, block_keys=[1]))
                if expired:
                    from flexlb_test_framework.scenario.runtime import StageTimeout

                    with self.assertRaises(StageTimeout):
                        summary(records.snapshot_records())
                else:
                    self.assertTrue(
                        summary(records.snapshot_records())["result_complete"]
                    )
                    self.assertFalse(summary(records.snapshot_records())["zero_errors"])
                    self.assertEqual(record["schedule"]["status"], "DEADLINE_EXCEEDED")

    def test_schedule_programming_error_remains_execution_error(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = self.context(root)
            records = SkewRecordedRequests(ctx, "pump", 30)

            def result(timeout):
                raise ValueError("bad request fixture")

            ctx.ops.schedule_pb2_grpc = NS(
                FlexlbServiceStub=lambda channel: NS(
                    Schedule=NS(
                        future=lambda *a, **kw: NS(result=result, cancel=lambda: True)
                    )
                )
            )
            record = records.issue(1, ctx.clock)
            records.run(record, dict(input_len=10, output_len=2, block_keys=[1]))
            with self.assertRaisesRegex(RuntimeError, "bad request fixture"):
                summary(records.snapshot_records())
            self.assertNotIn("request_error", record)

    def test_flow_worker_error_is_not_lost_when_future_leaves_active_set(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = self.context(root)
            flow = SkewFlow(ctx, [[1]])
            flow.pump_error = "fixture pump failed"
            with self.assertRaisesRegex(RuntimeError, "fixture pump failed"):
                flow.stop(Deadline(time.monotonic() + 2))


if __name__ == "__main__":
    unittest.main()
