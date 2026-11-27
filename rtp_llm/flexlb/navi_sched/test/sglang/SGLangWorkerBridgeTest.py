import asyncio
import unittest

from navi_sched.adapters.sglang.SGLangWorkerBridge import SGLangWorkerBridge


class SGLangWorkerBridgeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sent = []
        self.bridge = SGLangWorkerBridge(
            lambda worker, message: self.sent.append((worker, message)), 2, 2, 1000
        )

    async def start(self, worker="worker-a"):
        stream = self.bridge.generate(worker, b"request")
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        return stream, task, self.sent[-1][1]["call_id"]

    async def test_reordered_workers_keep_frames_separate(self):
        a, ta, ida = await self.start()
        b, tb, idb = await self.start("worker-b")
        self.assertFalse(self.bridge.receive("worker-b", ida, b"wrong"))
        self.bridge.receive("worker-b", idb, b"b", terminal=True)
        self.bridge.receive("worker-a", ida, b"a", terminal=True)
        self.assertEqual(await ta, b"a")
        self.assertEqual(await tb, b"b")
        for stream in (a, b):
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)
        self.assertFalse(self.bridge.active)
        self.assertEqual(len(self.sent), 2)

    async def test_cancel_then_late_frame_cannot_reach_new_request(self):
        old, task, old_id = await self.start()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await old.aclose()
        self.assertEqual(
            self.sent[-1], ("worker-a", {"kind": "abort", "call_id": old_id})
        )
        new, task, new_id = await self.start()
        self.assertNotEqual(old_id, new_id)
        self.assertFalse(
            self.bridge.receive("worker-a", old_id, b"late", terminal=True)
        )
        self.bridge.receive("worker-a", new_id, b"new", terminal=True)
        self.assertEqual(await task, b"new")
        await new.aclose()
        self.assertFalse(self.bridge.active)

    async def test_worker_error_releases_request(self):
        _, task, call_id = await self.start()
        self.bridge.receive(
            "worker-a", call_id, error=ValueError("temperature is invalid")
        )
        with self.assertRaisesRegex(ValueError, "temperature"):
            await task
        self.assertFalse(self.bridge.active)

    async def test_backpressure_aborts_once_and_bounds_storage(self):
        generator, task, call_id = await self.start()
        for frame in (b"one", b"two"):
            self.assertTrue(self.bridge.receive("worker-a", call_id, frame))
        self.assertFalse(self.bridge.receive("worker-a", call_id, b"three"))
        self.assertEqual(await task, b"one")
        stream = self.bridge.active[call_id]
        self.assertLessEqual(stream.frames.qsize(), 2)
        # Closing the consumer must not send a second cancellation.
        await generator.aclose()
        self.assertFalse(self.bridge.active)
        self.assertEqual([m["kind"] for _, m in self.sent], ["execute", "abort"])

    async def test_timeout_cancels_worker_and_releases_request(self):
        bridge = SGLangWorkerBridge(lambda w, m: self.sent.append((w, m)), 1, 1, 1)
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            await anext(bridge.generate("worker-a", b"request"))
        self.assertFalse(bridge.active)
        self.assertEqual([m["kind"] for _, m in self.sent], ["execute", "abort"])

    async def test_close_wakes_blocked_reader(self):
        _, task, _ = await self.start()
        self.bridge.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await task
        self.assertFalse(self.bridge.active)
        self.bridge.close()
        self.assertEqual(len(self.sent), 2)


if __name__ == "__main__":
    unittest.main()
