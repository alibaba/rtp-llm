"""Asynchronous traffic-client tests with controllable RPC and stream lifecycles."""

import concurrent.futures
import threading
import time
import unittest
from types import SimpleNamespace

from prefill_traffic_client import AsyncQueryClient, _Query


def item(request_id):
    return dict(
        request_id=request_id,
        stage="steady",
        input_tokens=4096,
        keys=[10, 11, 12, request_id + 100],
        session=f"session-{request_id}",
        cold=False,
        offset_s=0,
    )


def response(success=True, batch=False):
    return SimpleNamespace(
        success=success,
        code=200 if success else 8431,
        error_message="" if success else "Prefill queue overload",
        enqueued_by_master=batch,
    )


class FakeStream:
    def __init__(
        self, complete=True, error=None, code=None, held=False, stubborn=False
    ):
        self.snap = SimpleNamespace(
            terminated=False,
            completed=False,
            error=None,
            stream_error_code=None,
            stream_error_message=None,
            first_received_s=None,
            terminated_s=None,
        )
        self.cancelled = threading.Event()
        self.release = threading.Event()
        self.stubborn = stubborn
        if not held:
            self.release.set()

        def consume():
            self.release.wait(5)
            self.snap.first_received_s = time.monotonic()
            self.snap.completed = complete
            self.snap.error = error
            self.snap.stream_error_code = code
            self.snap.terminated = True
            self.snap.terminated_s = time.monotonic()

        self.thread = threading.Thread(target=consume, daemon=True)
        self.thread.start()

    def cancel(self):
        self.cancelled.set()
        if not self.stubborn:
            self.release.set()


class FakeOps:
    def __init__(self):
        self.calls = {}
        self.requests = {}
        self.streams = {}
        self.stream_factory = lambda: FakeStream()
        self.start_threads = []
        self.generated = {}
        self.stub = SimpleNamespace(Schedule=SimpleNamespace(future=self.schedule))
        self.schedule_pb2_grpc = SimpleNamespace(
            FlexlbServiceStub=lambda channel: self.stub
        )

    def _channel(self, target):
        return "channel"

    def master_target(self):
        return "master"

    def build_schedule_request(self, request_id, **kwargs):
        request = SimpleNamespace(request_id=request_id, **kwargs)
        self.requests[request_id] = request
        return request

    def schedule(self, request, timeout):
        future = concurrent.futures.Future()
        self.calls[request.request_id] = future
        return future

    def prefill_addr(self, reply):
        return "prefill:1234" if reply.success else ""

    def build_generate_input(self, request_id, **kwargs):
        self.generated[request_id] = kwargs
        return SimpleNamespace(request_id=request_id, **kwargs)

    def start_stream(self, reply, request_id, generated):
        self.start_threads.append(threading.current_thread().name)
        stream = self.stream_factory()
        self.streams[request_id] = stream
        return stream


def until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.001)


class AsyncQueryClientTest(unittest.TestCase):
    def submit(self, client, request_id=1):
        return client.submit(item(request_id), time.monotonic(), time.time() * 1000)

    def test_1500_waiting_schedule_rpcs_use_one_monitor_not_worker_threads(self):
        ops = FakeOps()
        before = {thread.ident for thread in threading.enumerate()}
        with AsyncQueryClient(ops) as client:
            futures = [self.submit(client, request_id) for request_id in range(1500)]
            extra = [
                thread for thread in threading.enumerate() if thread.ident not in before
            ]
            self.assertEqual(
                ["traffic-completion-monitor"], [thread.name for thread in extra]
            )
            self.assertEqual(1500, len(ops.calls))
            self.assertFalse(any(future.done() for future in futures))
            for rpc in ops.calls.values():
                rpc.set_result(response(success=False))
            rows = [future.result(timeout=2) for future in futures]
            self.assertTrue(
                all(row["code"] == 8431 and not row["success"] for row in rows)
            )
        self.assertFalse(client._thread.is_alive())

    def test_success_both_dispatch_paths_preserves_shape_and_client_times(self):
        for batch in (False, True):
            ops = FakeOps()
            with AsyncQueryClient(ops) as client:
                future = self.submit(client)
                ops.calls[1].set_result(response(batch=batch))
                row = future.result(timeout=2)
                self.assertTrue(row["success"])
                self.assertEqual(batch, row["enqueued_by_master"])
                self.assertEqual("session-1", ops.requests[1].session_id)
                self.assertEqual(item(1)["keys"], ops.generated[1]["block_keys"])
                self.assertEqual(4096, ops.generated[1]["input_len"])
                self.assertEqual(2, ops.generated[1]["output_len"])
                self.assertEqual(
                    row["first_received_s"] - row["started_s"], row["ttft_s"]
                )
                self.assertLessEqual(
                    row["schedule_completed_s"], row["stream_started_s"]
                )
                self.assertLessEqual(row["stream_started_s"], row["ended_s"])
                self.assertEqual(["traffic-completion-monitor"], ops.start_threads)

    def test_rpc_callback_does_not_construct_stream_or_block_rpc_poll_thread(self):
        ops = FakeOps()
        start_entered, release = threading.Event(), threading.Event()

        def slow_start():
            start_entered.set()
            release.wait(2)
            return FakeStream()

        ops.stream_factory = slow_start
        with AsyncQueryClient(ops) as client:
            future = self.submit(client)
            callback_returned = threading.Event()

            def rpc_complete():
                ops.calls[1].set_result(response())
                callback_returned.set()

            caller = threading.Thread(target=rpc_complete, name="grpc-poll-test")
            caller.start()
            self.assertTrue(start_entered.wait(1))
            self.assertTrue(callback_returned.wait(0.2))
            release.set()
            caller.join(1)
            self.assertTrue(future.result(timeout=2)["success"])

    def test_rpc_failure_and_stream_setup_failure_are_returned_as_failures(self):
        ops = FakeOps()
        with AsyncQueryClient(ops) as client:
            first = self.submit(client, 1)
            ops.calls[1].set_exception(RuntimeError("schedule transport failed"))
            self.assertIn("schedule transport failed", first.result(timeout=2)["error"])
            second = self.submit(client, 2)
            ops.stream_factory = lambda: (_ for _ in ()).throw(
                RuntimeError("stream setup failed")
            )
            ops.calls[2].set_result(response())
            row = second.result(timeout=2)
            self.assertEqual(200, row["code"])
            self.assertFalse(row["success"])
            self.assertIn("stream setup failed", row["error"])

    def test_inband_error_transport_error_and_unfinished_stream_never_succeed(self):
        for kwargs in (
            dict(code=8500),
            dict(error="transport failed"),
            dict(complete=False),
        ):
            ops = FakeOps()
            ops.stream_factory = lambda: FakeStream(**kwargs)
            with AsyncQueryClient(ops) as client:
                future = self.submit(client)
                ops.calls[1].set_result(response())
                row = future.result(timeout=2)
                self.assertFalse(row["success"])
                self.assertEqual(200, row["code"])

    def test_stream_timeout_cancels_reader_and_cannot_turn_into_success(self):
        ops = FakeOps()
        ops.stream_factory = lambda: FakeStream(held=True)
        with AsyncQueryClient(ops, stream_timeout_s=0.02) as client:
            future = self.submit(client)
            ops.calls[1].set_result(response())
            row = future.result(timeout=2)
            self.assertFalse(row["success"])
            self.assertIn("timeout", row["error"])
            self.assertTrue(ops.streams[1].cancelled.is_set())
            self.assertFalse(ops.streams[1].thread.is_alive())

    def test_close_cancels_pending_rpc_and_stream_then_joins_monitor(self):
        ops = FakeOps()
        ops.stream_factory = lambda: FakeStream(held=True)
        client = AsyncQueryClient(ops).__enter__()
        pending = self.submit(client, 1)
        streaming = self.submit(client, 2)
        ops.calls[2].set_result(response())
        until(lambda: 2 in ops.streams)
        client.close()
        self.assertTrue(ops.calls[1].cancelled())
        self.assertTrue(ops.streams[2].cancelled.is_set())
        self.assertFalse(ops.streams[2].thread.is_alive())
        self.assertFalse(client._thread.is_alive())
        self.assertFalse(pending.result()["success"])
        self.assertFalse(streaming.result()["success"])

    def test_exceptional_context_exit_cleans_up(self):
        ops = FakeOps()
        with self.assertRaisesRegex(ValueError, "case failed"):
            with AsyncQueryClient(ops) as client:
                future = self.submit(client)
                raise ValueError("case failed")
        self.assertFalse(client._thread.is_alive())
        self.assertTrue(ops.calls[1].cancelled())
        self.assertFalse(future.result()["success"])

    def test_cancelling_result_future_cancels_schedule_rpc(self):
        ops = FakeOps()
        with AsyncQueryClient(ops) as client:
            future = self.submit(client)
            self.assertTrue(future.cancel())
            until(lambda: ops.calls[1].cancelled())
        self.assertTrue(future.cancelled())

    def test_result_completion_and_cancellation_share_one_atomic_owner(self):
        ops = FakeOps()
        client = AsyncQueryClient(ops)

        class CancellationAtClaim(concurrent.futures.Future):
            def set_running_or_notify_cancel(self):
                self.cancel()
                return super().set_running_or_notify_cancel()

        result = CancellationAtClaim()
        entry = _Query(dict(success=False), result)
        client._entries[1] = entry
        client._finish(1, entry)
        self.assertTrue(result.cancelled())
        self.assertFalse(client._entries)

    def test_monitor_failure_rejects_further_submissions_and_cancels_calls(self):
        ops = FakeOps()
        client = AsyncQueryClient(ops)

        def broken_poll(*args):
            raise RuntimeError("monitor failure")

        client._poll = broken_poll
        client.__enter__()
        future = self.submit(client)
        row = future.result(timeout=2)
        self.assertFalse(row["success"])
        self.assertIn("monitor failure", row["error"])
        self.assertTrue(ops.calls[1].cancelled())
        with self.assertRaisesRegex(RuntimeError, "not open"):
            self.submit(client, 2)
        with self.assertRaisesRegex(RuntimeError, "monitor failure"):
            client.close()

    def test_stream_start_monitor_backlog_is_measured_separately(self):
        ops = FakeOps()
        release = threading.Event()
        entered = threading.Event()

        def held_setup():
            entered.set()
            release.wait(2)
            return FakeStream()

        ops.stream_factory = held_setup
        with AsyncQueryClient(ops) as client:
            first, second = self.submit(client, 1), self.submit(client, 2)
            ops.calls[1].set_result(response())
            self.assertTrue(entered.wait(1))
            ops.calls[2].set_result(response())
            time.sleep(0.02)
            release.set()
            first_row, second_row = first.result(timeout=2), second.result(timeout=2)
            self.assertGreaterEqual(first_row["stream_launch_lag_s"], 0.02)
            self.assertGreaterEqual(second_row["stream_launch_lag_s"], 0.02)
            self.assertLess(second_row["pacing_lag_s"], 0.02)

    def test_stuck_reader_cleanup_is_bounded_and_explicit_failure(self):
        ops = FakeOps()
        ops.stream_factory = lambda: FakeStream(held=True, stubborn=True)
        client = AsyncQueryClient(ops, close_timeout_s=0.02).__enter__()
        future = self.submit(client)
        ops.calls[1].set_result(response())
        until(lambda: 1 in ops.streams)
        try:
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                client.close()
            self.assertFalse(client._thread.is_alive())
            self.assertFalse(future.result()["success"])
        finally:
            ops.streams[1].release.set()
            ops.streams[1].thread.join(1)

    def test_duplicate_ids_closed_client_and_invalid_timeouts_fail_fast(self):
        ops = FakeOps()
        with AsyncQueryClient(ops) as client:
            self.submit(client)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.submit(client)
        with self.assertRaisesRegex(RuntimeError, "not open"):
            self.submit(client, 2)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                AsyncQueryClient(ops, stream_timeout_s=value)


if __name__ == "__main__":
    unittest.main()
