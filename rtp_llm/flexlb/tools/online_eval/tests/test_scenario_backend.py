"""Actual consumer threads with fake RPC transports: no Java or network required."""

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexlb_test_framework.scenario.backend import BoundedOps, RequestBatch
from flexlb_test_framework.scenario.lease import validate_lease
from flexlb_test_framework.scenario.loader import ScenarioError
from flexlb_test_framework.scenario.runtime import (
    Deadline,
    RuntimeContext,
    StageTimeout,
    interruptible,
)


class Stream:
    def __init__(self, hanging=False, error=None):
        self.hanging, self.error = hanging, error
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()
        return True

    def __iter__(self):
        if self.hanging:
            self.cancelled.wait(2)
            if not self.cancelled.is_set():
                raise RuntimeError("test stream cancellation was not issued")
            return
        if self.error:
            raise self.error
        yield NS(HasField=lambda key: False, flatten_output=NS(finished=[True]))


class Ops:
    def __init__(self, batch=True, hanging=False):
        self.batch, self.hanging = batch, hanging
        self.streams, self.fetch_count, self.generate_count, self.counter = [], 0, 0, 0
        self.responses = []
        self.schedule_pb2_grpc = NS(
            FlexlbServiceStub=lambda channel: NS(Schedule=NS(future=self.future))
        )
        self.pb2_grpc = NS(
            RpcServiceStub=lambda channel: NS(
                FetchResponse=self.fetch, GenerateStreamCall=self.generate
            )
        )
        self.pb2 = NS(FetchRequestPB=lambda **kw: kw)

    def future(self, req, timeout, metadata=None):
        self.last_schedule = (req, timeout, metadata)
        response = NS(
            code=200, success=True, error_message="", enqueued_by_master=self.batch
        )
        self.responses.append(response)
        return NS(result=lambda timeout: response, cancel=lambda: True)

    def next_request_id(self):
        self.counter += 1
        return self.counter

    def _channel(self, target):
        return target

    def master_target(self):
        return "master"

    def build_schedule_request(self, rid, **shape):
        return (rid, shape)

    def prefill_addr(self, response):
        return "prefill"

    def build_generate_input(self, rid, **shape):
        return (rid, shape)

    def _copy_role_addrs(self, inp, response):
        pass

    def fetch(self, req, timeout):
        self.fetch_count += 1
        stream = Stream(self.hanging)
        self.streams.append(stream)
        return stream

    def generate(self, req, timeout):
        self.generate_count += 1
        stream = Stream(self.hanging)
        self.streams.append(stream)
        return stream


def lease_manifest():
    m, b = 28000, 55000
    env = dict(
        FLEXLB_FT_MASTER_HTTP_PORT=str(m),
        FLEXLB_FT_MASTER_MANAGEMENT_PORT=str(m + 1),
        FLEXLB_FT_HA_MASTER_A_HTTP_PORT=str(m),
        FLEXLB_FT_HA_MASTER_B_HTTP_PORT=str(m + 3),
        FLEXLB_FT_MOCK_BASE_GRPC_PORT=str(b),
    )
    return dict(
        schema_version=1,
        lane=0,
        backend="java_mock",
        master_base=m,
        mock_base=b,
        worker_capacity=6,
        intervals=[
            dict(side="master", first=m, last=m + 5),
            dict(side="mock", first=b - 1, last=b + 151),
        ],
        lock_names=[f"m{m}_{m+5}.lock", f"g{b-1}_{b+151}.lock"],
        child_env=env,
    )


class BackendTest(unittest.TestCase):
    def test_schedule_deadline_preserves_transport_status_after_callback_delay(self):
        import grpc

        class RpcDeadline(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.DEADLINE_EXCEEDED

        with tempfile.TemporaryDirectory() as root:
            ctx = RuntimeContext({}, None, root, time.monotonic, time.sleep)
            ctx.ops = Ops()
            ctx.instance_deadline_s = time.monotonic() + 2
            observed = {}

            def future(req, timeout):
                observed["rpc_timeout"] = timeout

                def result(timeout):
                    observed["wait_timeout"] = timeout
                    # A completed transport can publish its callback after its
                    # deadline. A second identical timer loses the typed error.
                    if timeout <= observed["rpc_timeout"]:
                        raise grpc.FutureTimeoutError()
                    raise RpcDeadline()

                return NS(result=result, cancel=lambda: True)

            ctx.ops.future = future
            requests = RequestBatch(
                ctx,
                dict(
                    count=1,
                    input_len=10,
                    output_len=2,
                    consume="immediate",
                    schedule_timeout_s=0.02,
                ),
            )
            with self.assertRaises(RpcDeadline):
                requests.submit(Deadline(time.monotonic() + 1))
            record = requests.snapshot_records()[0]
            self.assertEqual(observed["rpc_timeout"], 0.02)
            self.assertGreater(observed["wait_timeout"], 0.02)
            self.assertLessEqual(observed["wait_timeout"], 1)
            self.assertEqual(record["schedule"]["status"], "DEADLINE_EXCEEDED")
            self.assertIsNotNone(record["schedule"]["ended_s"])
            self.assertIsNone(record["stream"]["started_s"])

    def run_batch(self, consume, batch=True, hanging=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ctx = RuntimeContext({}, None, tmp.name, time.monotonic, time.sleep)
        ctx.ops = Ops(batch, hanging)
        ctx.instance_deadline_s = time.monotonic() + 2
        requests = RequestBatch(
            ctx, dict(count=2, input_len=10, output_len=2, consume=consume)
        )
        ctx.register_resource("requests", requests, requests.cleanup)
        requests.submit(Deadline(time.monotonic() + 1))
        return requests, ctx

    def test_schedule_open_cancellation_reaches_new_call_before_result(self):
        for batch_mode in (True, False):
            with self.subTest(batch=batch_mode), tempfile.TemporaryDirectory() as root:
                ctx = RuntimeContext({}, None, root, time.monotonic, time.sleep)
                ctx.ops = Ops(batch_mode)
                ctx.instance_deadline_s = time.monotonic() + 2
                requests = RequestBatch(
                    ctx, dict(count=2, consume="immediate", input_len=10, output_len=2)
                )
                cancelled, waited, created = [], [], []

                def future(req, timeout):
                    created.append(True)
                    requests.cancel("race during Schedule.future")
                    return NS(
                        cancel=lambda: cancelled.append(True),
                        result=lambda **kw: waited.append(True),
                    )

                ctx.ops.schedule_pb2_grpc = NS(
                    FlexlbServiceStub=lambda channel: NS(Schedule=NS(future=future))
                )
                with self.assertRaisesRegex(
                    RuntimeError, "cancelled during Schedule startup"
                ):
                    requests.submit(Deadline(ctx.instance_deadline_s))
                self.assertTrue(cancelled)
                self.assertEqual(created, [True])
                self.assertEqual(waited, [])
                self.assertEqual((ctx.ops.fetch_count, ctx.ops.generate_count), (0, 0))
                requests.cleanup(Deadline(time.monotonic() + 2))

    def test_precancelled_batch_never_opens_schedule(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = RuntimeContext({}, None, root, time.monotonic, time.sleep)
            ctx.ops = Ops(True)
            requests = RequestBatch(
                ctx, dict(count=2, consume="immediate", input_len=10, output_len=2)
            )
            requests.cancel("already stopped")
            with self.assertRaisesRegex(RuntimeError, "cancelled before Schedule"):
                requests.submit(Deadline(time.monotonic() + 2))
            self.assertEqual(ctx.ops.responses, [])
            self.assertEqual(requests.snapshot_records(), [])

    def test_deferred_has_zero_fetch_until_wait_and_business_completion(self):
        requests, ctx = self.run_batch("deferred")
        self.assertEqual(ctx.ops.fetch_count, 0)
        self.assertTrue(
            all(r["consumer_exit_s"] is None for r in requests.snapshot_records())
        )
        self.assertEqual(
            requests.wait(Deadline(time.monotonic() + 1)),
            dict(completed=True, error_count=0),
        )
        self.assertEqual(ctx.ops.fetch_count, 2)
        self.assertTrue(
            all(
                r["fetch_invocations"] == 1 and r["business_finished"]
                for r in requests.snapshot_records()
            )
        )
        self.assertEqual([r["status"] for r in ctx.cleanup(1)], ["PASS"])

    def test_nonbatch_opens_generate_once(self):
        requests, ctx = self.run_batch("immediate", batch=False)
        self.assertTrue(requests.wait(Deadline(time.monotonic() + 1))["completed"])
        self.assertEqual((ctx.ops.fetch_count, ctx.ops.generate_count), (0, 2))
        ctx.cleanup(1)

    def test_explicit_prefix_and_priority_reach_both_protocol_builders(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = RuntimeContext({}, None, root, time.monotonic, time.sleep)
            ctx.ops = Ops(batch=False)
            ctx.ops.build_generate_input = Mock(wraps=ctx.ops.build_generate_input)
            ctx.instance_deadline_s = time.monotonic() + 2
            params = dict(
                count=1,
                input_len=1024,
                output_len=2,
                consume="immediate",
                block_keys=[42, 43],
                priority=70,
                qos_level=30,
                schedule_timeout_s=0.4,
                stream_timeout_s=0.5,
            )
            requests = RequestBatch(ctx, params)
            ctx.register_resource("requests", requests, requests.cleanup)
            requests.submit(Deadline(time.monotonic() + 1))
            self.assertTrue(requests.wait(Deadline(time.monotonic() + 1))["completed"])
            request, timeout, metadata = ctx.ops.last_schedule
            self.assertEqual(request[1]["block_keys"], [42, 43])
            self.assertEqual(request[1]["priority"], 70)
            self.assertNotIn("qos_level", request[1])
            self.assertEqual(metadata, (("x-dashscope-inner-qos-level", "30"),))
            self.assertLessEqual(timeout, 0.4)
            self.assertEqual(ctx.ops.build_generate_input.call_args.kwargs, request[1])
            record = requests.snapshot_records()[0]
            self.assertEqual(record["request_shape"], request[1])
            self.assertLessEqual(
                record["stream"]["deadline_s"] - record["stream"]["started_s"], 0.501
            )
            self.assertEqual(ctx.cleanup(1)[0]["status"], "PASS")

    def test_omitted_priority_stays_unset_without_qos_header(self):
        requests, ctx = self.run_batch("immediate")
        requests.wait(Deadline(time.monotonic() + 1))
        request, timeout, metadata = ctx.ops.last_schedule
        self.assertNotIn("priority", request[1])
        self.assertIsNone(metadata)
        self.assertNotIn("priority", requests.snapshot_records()[0]["request_shape"])
        ctx.cleanup(1)

    def test_generated_shape_is_recorded_after_request_identity_exists(self):
        class Generated(RequestBatch):
            @property
            def shape(self):
                result = super().shape
                if self.entries:
                    result["block_keys"] = [
                        self.entries[-1]["record"]["wire_request_id"] * 100
                    ]
                return result

        with tempfile.TemporaryDirectory() as root:
            ctx = RuntimeContext({}, None, root, time.monotonic, time.sleep)
            ctx.ops = Ops()
            ctx.instance_deadline_s = time.monotonic() + 2
            batch = Generated(
                ctx, dict(count=1, input_len=10, output_len=2, consume="immediate")
            )
            ctx.register_resource("requests", batch, batch.cleanup)
            batch.submit(Deadline(time.monotonic() + 1))
            batch.wait(Deadline(time.monotonic() + 1))
            self.assertEqual(
                batch.snapshot_records()[0]["request_shape"],
                ctx.ops.last_schedule[0][1],
            )
            self.assertEqual(ctx.cleanup(1)[0]["status"], "PASS")

    def test_paced_deferred_wave_preserves_each_post_issue_delay_without_fetch(self):
        now = [0.0]
        clock = lambda: now[0]

        def sleep(value):
            now[0] += value

        with tempfile.TemporaryDirectory() as root:
            ctx = RuntimeContext({}, None, root, clock, sleep)
            ctx.ops = Ops()
            ctx.instance_deadline_s = 10
            batch = RequestBatch(
                ctx,
                dict(
                    count=3,
                    input_len=10,
                    output_len=2,
                    consume="deferred",
                    post_issue_delay_s=0.12,
                ),
            )
            ctx.register_resource("requests", batch, batch.cleanup)
            batch.submit(Deadline(5, clock, sleep))
            self.assertEqual(
                [row["issued_s"] for row in batch.snapshot_records()], [0, 0.12, 0.24]
            )
            self.assertAlmostEqual(now[0], 0.36)
            self.assertEqual(ctx.ops.fetch_count, 0)
            self.assertTrue(batch.wait(Deadline(5, clock, sleep))["completed"])
            self.assertEqual(ctx.cleanup(1)[0]["status"], "PASS")

    def test_wait_timeout_cleanup_cancels_actual_call_and_joins(self):
        requests, ctx = self.run_batch("immediate", hanging=True)
        with self.assertRaises(StageTimeout):
            requests.wait(Deadline(time.monotonic() + 0.02))
        self.assertEqual([r["status"] for r in ctx.cleanup(1)], ["PASS"])
        self.assertTrue(all(s.cancelled.is_set() for s in ctx.ops.streams))
        self.assertFalse(any(e["thread"].is_alive() for e in requests.entries))
        self.assertTrue(
            all(not r["business_finished"] for r in requests.snapshot_records())
        )

    def test_partial_submission_resources_registered_before_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RuntimeContext({}, None, tmp, time.monotonic, time.sleep)
            ctx.ops = Ops(batch=False)
            ctx.instance_deadline_s = time.monotonic() + 2
            requests = RequestBatch(
                ctx, dict(count=1, input_len=10, output_len=2, consume="deferred")
            )
            ctx.register_resource("requests", requests, requests.cleanup)
            with self.assertRaisesRegex(RuntimeError, "not enqueued"):
                requests.submit(Deadline(time.monotonic() + 1))
            self.assertEqual(len(requests.snapshot_records()), 1)
            self.assertEqual(ctx.cleanup(1)[0]["status"], "PASS")

    def test_lease_is_exact_and_worker_budget_cannot_exceed_capacity(self):
        lease = lease_manifest()
        budget = dict(
            backend="java_mock",
            bounded=True,
            initial_workers=4,
            max_dynamic_additions=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lease.json"
            path.write_text(json.dumps(lease))
            self.assertEqual(validate_lease(path, budget, lease["child_env"]), lease)
            with self.assertRaises(ScenarioError):
                validate_lease(path, budget, {})
            budget["max_dynamic_additions"] = 3
            with self.assertRaises(ScenarioError):
                validate_lease(path, budget, lease["child_env"])

    def test_dynamic_allocation_counts_failed_attempts_and_rejects_explicit(self):
        raw = NS(add_engine=Mock(return_value=(503, {})))
        ops = BoundedOps(
            raw, dict(initial_workers=4, max_dynamic_additions=1), lease_manifest()
        )
        with self.assertRaises(ValueError):
            ops.add_engine("prefill", port=55010)
        self.assertEqual(ops.add_engine("prefill")[0], 503)
        with self.assertRaises(ValueError):
            ops.add_engine("decode")
        self.assertEqual(raw.add_engine.call_count, 1)

    def test_wallclock_guard_interrupts_blocking_call_without_abandoned_thread(self):
        with self.assertRaises(StageTimeout):
            with interruptible(Deadline(time.monotonic() + 0.02), True):
                time.sleep(1)

    def test_cleanup_needs_consumer_witness_even_if_thread_reports_dead(self):
        # A controlled false-alive fixture tests the proof gap. It does not
        # assert that a particular CPython/signal implementation causes it.
        requests, ctx = self.run_batch("immediate", hanging=True)
        ready = Deadline(time.monotonic() + 1)
        while len(ctx.ops.streams) != 2:
            ready.sleep(0.001)
        for stream in ctx.ops.streams:
            stream.cancel = Mock(return_value=True)  # acknowledged, still blocked
        patches = [
            patch.object(e["thread"], "is_alive", return_value=False)
            for e in requests.entries
        ]
        for p in patches:
            p.start()
        try:
            rows = ctx.cleanup(0.02)
            self.assertEqual(rows[0]["status"], "TIMEOUT")
            persisted = json.loads(requests.artifact.read_text())
            self.assertTrue(all(r["consumer_exit_s"] is None for r in persisted))
        finally:
            for p in patches:
                p.stop()
            for stream in ctx.ops.streams:
                stream.cancelled.set()
            for entry in requests.entries:
                self.assertTrue(entry["done"].wait(1))
                entry["thread"].join(1)

    def test_cleanup_persists_consumer_completion_before_reporting_pass(self):
        requests, ctx = self.run_batch("immediate", hanging=True)
        ready = Deadline(time.monotonic() + 1)
        while len(ctx.ops.streams) != 2:
            ready.sleep(0.001)
        self.assertEqual(ctx.cleanup(1)[0]["status"], "PASS")
        persisted = json.loads(requests.artifact.read_text())
        for row in persisted:
            self.assertIs(row["consumer_done"], True)
            self.assertIs(row["consumer_completion_verified"], True)
            self.assertIsNotNone(row["consumer_exit_s"])
            self.assertIsNotNone(row["transport_terminal_s"])
            self.assertIsNotNone(row["stream"]["ended_s"])


if __name__ == "__main__":
    unittest.main()
