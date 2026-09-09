"""Real consumer threads driven by a virtual clock and virtual timed joins."""

import tempfile
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_test_framework.scenario.observed import ObservedRequestBatch
from flexlb_test_framework.scenario.runtime import (
    Deadline,
    RuntimeContext,
    StageTimeout,
)
from test_scenario_backend import Ops
from test_scenario_rpc_fault_probe import RpcError


class Clock:
    def __init__(self):
        self.now = 0.0
        self.stream = self.entry = None

    def __call__(self):
        return self.now

    def advance(self, delta):
        self.now += delta
        stream = self.stream
        if stream is None:
            return
        stream.tick.set()
        if not stream.cancelled and self.now >= stream.first:
            if not stream.first_consumed.wait(1):
                raise AssertionError("real consumer did not publish first output")
        if self.now >= stream.end or (stream.cancelled and stream.cancel_delay == 0):
            if not self.entry["done"].wait(1):
                raise AssertionError("real consumer did not publish exit")

    sleep = advance


class ScheduledStream:
    def __init__(self, clock, first, end, finished, cancel_delay):
        self.clock = clock
        self.first, self.end = clock() + first, clock() + end
        self.finished, self.cancel_delay = finished, cancel_delay
        self.tick, self.ready, self.first_consumed = (
            threading.Event() for _ in range(3)
        )
        self.cancelled = False
        clock.stream = self

    def cancel(self):
        self.cancelled = True
        self.end = min(self.end, self.clock() + self.cancel_delay)
        self.tick.set()
        return True

    def __iter__(self):
        self.ready.set()
        while self.clock() < self.first and not self.cancelled:
            self.tick.wait(1)
            self.tick.clear()
        if not self.cancelled:
            yield NS(
                HasField=lambda key: False, flatten_output=NS(finished=[self.finished])
            )
            self.first_consumed.set()
        while self.clock() < self.end:
            self.tick.wait(1)
            self.tick.clear()
        if self.cancelled:
            raise RpcError("CANCELLED")


class ObservedTests(unittest.TestCase):
    def run_observation(
        self,
        *,
        mode="stream_ttft",
        first=1,
        end=2,
        open_cost=0,
        finished=True,
        cancel_delay=0,
        parent=125,
        batch=True,
        end_wait_s=15
    ):
        clock = Clock()
        real_thread = threading.Thread
        joins = []
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RuntimeContext({}, None, tmp, clock, clock.sleep)
            ctx.instance_deadline_s = 500
            ops = Ops(batch)
            ctx.ops = ops

            def stream(*args, **kwargs):
                clock.advance(open_cost)
                ops.last_rpc_timeout = kwargs["timeout"]
                return ScheduledStream(clock, first, end, finished, cancel_delay)

            ops.pb2_grpc = NS(
                RpcServiceStub=lambda channel: NS(
                    FetchResponse=stream, GenerateStreamCall=stream
                )
            )

            def thread_factory(*args, **kwargs):
                thread = real_thread(*args, **kwargs)
                entry = kwargs["args"][0]
                clock.entry = entry
                original_start, original_join = thread.start, thread.join

                def start():
                    original_start()
                    self.assertTrue(clock.stream.ready.wait(1))
                    if first == 0:
                        self.assertTrue(clock.stream.first_consumed.wait(1))
                        if end == 0:
                            self.assertTrue(entry["done"].wait(1))

                def join(timeout=None):
                    if not thread.is_alive():
                        return original_join(0)
                    joins.append((clock(), timeout))
                    clock.advance(min(timeout, max(0, clock.stream.end - clock())))
                    if entry["done"].is_set():
                        original_join(1)

                thread.start, thread.join = start, join
                return thread

            deadline = Deadline(parent, clock, clock.sleep)
            requests = ObservedRequestBatch(
                ctx,
                dict(
                    count=1,
                    consume="immediate",
                    input_len=2048,
                    output_len=10,
                    schedule_timeout_s=30,
                    stream_timeout_s=60,
                ),
                mode,
                deadline,
                end_wait_s=end_wait_s,
            )
            caught = None
            with patch(
                "flexlb_test_framework.scenario.backend.threading.Thread",
                thread_factory,
            ):
                try:
                    requests.submit(deadline)
                except Exception as exc:
                    caught = exc
                finally:
                    requests.cancel("test cleanup")
                    if clock.stream is not None:
                        clock.advance(max(0, clock.stream.end - clock()))
                    requests.cleanup(Deadline(clock() + 10, clock, clock.sleep))
            record = requests.snapshot_records()[0]
            return record, joins, caught, ops.last_rpc_timeout

    def test_synchronous_open_cost_excluded_and_early_first_never_negative(self):
        for batch in (False, True):
            record, _, error, timeout = self.run_observation(
                open_cost=2, first=0.1, end=0.2, batch=batch
            )
            self.assertIsNone(error)
            observation = record["rpc_observation"]
            self.assertEqual(observation["observer_started_s"], 2)
            latency = (
                observation["first_observed_s"] - observation["observer_started_s"]
            )
            self.assertGreaterEqual(latency, 0.1)
            self.assertLess(latency, 0.13)
            self.assertEqual(timeout, 60)
            self.assertTrue(
                record["consumer_done"] and record["consumer_completion_verified"]
            )
            self.assertEqual(
                record["stream"]["method"],
                "FetchResponse" if batch else "GenerateStreamCall",
            )
        record, _, error, _ = self.run_observation(first=0, end=0)
        self.assertIsNone(error)
        observation = record["rpc_observation"]
        self.assertLessEqual(
            record["stream"]["first_output_s"], observation["observer_started_s"]
        )
        self.assertEqual(
            observation["first_observed_s"] - observation["observer_started_s"], 0
        )

    def test_separate_first_and_end_windows_are_not_one_total_budget(self):
        for first, end, expected in (
            (11, 22, True),
            (12.01, 12.02, True),
            (13, 14, False),
            (1, 14, False),
        ):
            with self.subTest(first=first, end=end):
                record, joins, error, _ = self.run_observation(first=first, end=end)
                self.assertIsNone(error)
                observation = record["rpc_observation"]
                self.assertIs(observation["legacy_success"], expected)
                if expected:
                    self.assertGreater(joins[0][0], 10.9)
                    self.assertEqual(joins[0][1], 12)
                if first == 1:
                    self.assertFalse(observation["ended_in_window"])
                    self.assertEqual(joins[-1][1], 5)

    def test_observer_uses_poll_return_time_instead_of_raw_first_arrival(self):
        record, _, error, _ = self.run_observation(first=0.031, end=0.2)
        self.assertIsNone(error)
        self.assertAlmostEqual(record["rpc_observation"]["first_observed_s"], 0.04)

    def test_end_grace_cannot_repair_ttft_but_enqueue_keeps_old_completed_predicate(
        self,
    ):
        for mode, expected in (("stream_ttft", False), ("request_total", True)):
            record, joins, error, _ = self.run_observation(
                mode=mode, first=1, end=40, finished=True, cancel_delay=2
            )
            self.assertIsNone(error)
            observation = record["rpc_observation"]
            self.assertFalse(observation["ended_in_window"])
            self.assertIs(observation["legacy_success"], expected)
            self.assertEqual(record["stream"]["status"], "CANCELLED")
            self.assertEqual(joins[-1][1], 5)

    def test_ttft_normal_eof_does_not_invent_old_finished_requirement(self):
        record, _, error, _ = self.run_observation(finished=False)
        self.assertIsNone(error)
        self.assertFalse(record["business_finished"])
        self.assertTrue(record["rpc_observation"]["legacy_success"])
        record, _, error, _ = self.run_observation(mode="request_total", finished=False)
        self.assertIsNone(error)
        self.assertFalse(record["rpc_observation"]["legacy_success"])

    def test_skew_end_window_thirty_and_recovery_fifteen_keep_raw_cancel(self):
        for wait in (15, 30):
            record, joins, error, _ = self.run_observation(
                mode="request_total", first=1, end=40, cancel_delay=2, end_wait_s=wait
            )
            self.assertIsNone(error)
            self.assertEqual(joins[0][1], wait)
            self.assertEqual(record["stream"]["status"], "CANCELLED")
            self.assertTrue(record["rpc_observation"]["legacy_success"])
            self.assertFalse(record["rpc_observation"]["ended_in_window"])

    def test_parent_deadline_remains_timeout_not_expected_success(self):
        record, _, error, _ = self.run_observation(first=11, end=22, parent=5)
        self.assertIsInstance(error, StageTimeout)
        self.assertFalse(record["rpc_observation"]["complete"])
        self.assertFalse(record["rpc_observation"]["legacy_success"])


if __name__ == "__main__":
    unittest.main()
