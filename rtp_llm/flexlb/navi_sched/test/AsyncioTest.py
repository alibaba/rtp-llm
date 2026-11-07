import asyncio
import os
import unittest
from collections import deque
from types import SimpleNamespace

from navi_sched.asyncio import AsyncClient


class _NativeClient:
    def __init__(self):
        self.fd = os.eventfd(1, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self.responses = deque()
        self.decisions = {}
        self.aborted = []
        self.submitted = []
        self.reads = 0

    def notification_fd(self):
        return self.fd

    def submit_schedule(self, request, target_biz, timeout_ms):
        self.submitted.append(request.request_id)

    def take_schedule_result(self, request_id):
        return self.decisions.pop(request_id, None)

    def take_next_forward_response(self, timeout_ms):
        if timeout_ms != 0:
            raise AssertionError("the event loop must not wait in native code")
        self.reads += 1
        return self.responses.popleft() if self.responses else None

    def abort_forward_request(self, request_id):
        self.aborted.append(request_id)

    def notify(self):
        os.eventfd_write(self.fd, 1)


class AsyncioTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.native = _NativeClient()
        self.outputs = []
        self.errors = []
        self.client = AsyncClient(self.native, self.outputs.append, self.errors.append)

    async def asyncTearDown(self):
        self.client.close()
        os.close(self.native.fd)

    async def test_coalesced_wakeup_and_bounded_drain_preserve_all_outputs(self):
        first_drain = asyncio.get_running_loop().create_future()
        finished = asyncio.Event()

        def output(value):
            self.outputs.append(value)
            if value == 31:
                asyncio.get_running_loop().call_soon(
                    lambda: first_drain.set_result(list(self.outputs))
                )
            if value == 99:
                finished.set()

        self.client.response = output
        self.native.responses.extend(range(100))
        self.native.notify()
        self.assertEqual(await asyncio.wait_for(first_drain, 1), list(range(32)))
        await asyncio.wait_for(finished.wait(), 1)
        self.assertEqual(self.outputs, list(range(100)))
        self.assertEqual(self.errors, [])
        reads = self.native.reads
        await asyncio.sleep(0.02)
        self.assertEqual(self.native.reads, reads)

    async def test_late_decision_wakes_future_without_polling(self):
        task = asyncio.create_task(
            self.client.schedule(SimpleNamespace(request_id=7), "worker", 1000)
        )
        await asyncio.sleep(0)
        self.assertEqual(self.native.submitted, [7])
        self.assertFalse(task.done())
        decision = SimpleNamespace(target="worker")
        self.native.decisions[7] = decision
        self.native.notify()
        self.assertIs(await asyncio.wait_for(task, 1), decision)
        self.assertEqual(self.client.pending, {})

    async def test_close_removes_reader_and_cancels_burst_continuation(self):
        self.native.responses.extend(range(64))
        self.client.notification._ready()
        self.assertEqual(self.outputs, list(range(32)))
        self.assertIsNotNone(self.client.notification._continuation)
        self.client.close()
        with self.assertRaises(KeyError):
            asyncio.get_running_loop()._selector.get_key(self.native.fd)
        await asyncio.sleep(0)
        self.assertEqual(self.outputs, list(range(32)))
        self.assertIsNone(self.client.notification._continuation)

    async def test_cancellation_consumes_late_native_completion(self):
        task = asyncio.create_task(
            self.client.schedule(SimpleNamespace(request_id=8), "worker", 1000)
        )
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.native.aborted, [8])
        self.native.decisions[8] = SimpleNamespace(target=None)
        self.native.notify()
        for _ in range(4):
            await asyncio.sleep(0)
        self.assertEqual(self.client.pending, {})

    async def test_reader_failure_closes_and_fails_waiters(self):
        task = asyncio.create_task(
            self.client.schedule(SimpleNamespace(request_id=9), "worker", 1000)
        )
        await asyncio.sleep(0)

        def fail(response):
            raise ValueError("invalid response")

        self.client.response = fail
        self.native.responses.append("bad")
        self.native.notify()
        with self.assertRaisesRegex(ValueError, "invalid response"):
            await task
        self.assertTrue(self.client.notification.closed)
        self.assertEqual(self.client.pending, {})
        self.assertEqual([str(error) for error in self.errors], ["invalid response"])


if __name__ == "__main__":
    unittest.main()
