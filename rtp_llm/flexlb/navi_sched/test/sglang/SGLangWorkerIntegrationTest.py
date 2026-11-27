import asyncio
import json
import unittest
from types import SimpleNamespace

from navi_sched.adapters.sglang.SGLangWorkerBridge import SGLangWorkerBridge
from navi_sched.adapters.sglang.SGLangWorkerExecutor import SGLangWorkerExecutor


class Codec:
    def decode_request(self, payload):
        return SimpleNamespace(**json.loads(payload))

    def encode_response(self, value):
        return json.dumps(value).encode()

    def encode_error(self, error):
        return str(error)


class SGLangWorkerIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.aborted = []
        self.closed = []
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.codec = Codec()
        self.workers = {}
        self.bridge = SGLangWorkerBridge(
            lambda worker, message: self.workers[worker].receive(message), 4, 4, 1000
        )

        async def generate(manager, obj, request):
            try:
                if obj.rid == "invalid":
                    raise ValueError("temperature must be nonnegative")
                yield {"text": manager.name, "meta_info": {"finish_reason": None}}
                if obj.rid == "wait":
                    self.started.set()
                    await self.gate.wait()
                yield {
                    "text": manager.name + " done",
                    "meta_info": {"finish_reason": {"type": "stop"}},
                }
            finally:
                self.closed.append((manager.name, obj.rid))

        for worker in ("a", "b"):

            def send(message, worker=worker):
                error = ValueError(message["error"]) if "error" in message else None
                self.bridge.receive(
                    worker,
                    message["call_id"],
                    message.get("payload"),
                    error=error,
                    terminal=message.get("terminal", False),
                )

            manager = SimpleNamespace(name=worker)
            self.workers[worker] = SGLangWorkerExecutor(
                manager,
                generate,
                lambda manager, rid: self.aborted.append((manager.name, rid)),
                self.codec,
                send,
                2,
            )

    async def asyncTearDown(self):
        self.bridge.close()
        await asyncio.gather(
            *(worker.wait_closed() for worker in self.workers.values())
        )

    def stream(self, worker, rid):
        return self.bridge.generate(worker, json.dumps({"rid": rid}).encode())

    async def collect(self, worker, rid):
        return [json.loads(frame) async for frame in self.stream(worker, rid)]

    async def test_two_workers_return_complete_separate_streams(self):
        a, b = await asyncio.gather(self.collect("a", "one"), self.collect("b", "two"))
        self.assertEqual([x["text"] for x in a], ["a", "a done"])
        self.assertEqual([x["text"] for x in b], ["b", "b done"])
        self.assertCountEqual(self.aborted, [("a", "one"), ("b", "two")])
        self.assertFalse(self.bridge.active)

    async def test_disconnect_cancels_exact_worker_and_releases_stream(self):
        stream = self.stream("b", "wait")
        await anext(stream)
        await self.started.wait()
        await stream.aclose()
        await asyncio.sleep(0)
        self.assertIn(("b", "wait"), self.closed)
        self.assertEqual(self.aborted, [("b", "wait")])
        self.assertFalse(self.bridge.active)

    async def test_worker_error_propagates_then_same_worker_recovers(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            await self.collect("a", "invalid")
        output = await self.collect("a", "valid")
        self.assertEqual(output[-1]["text"], "a done")
        self.assertFalse(self.bridge.active)

    async def test_duplicate_call_does_not_replace_active_execution(self):
        stream = self.stream("a", "wait")
        await anext(stream)
        call_id = next(iter(self.bridge.active))
        original = self.workers["a"].tasks[call_id]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.workers["a"].receive(
                {"kind": "execute", "call_id": call_id, "payload": b"{}"}
            )
        self.assertIs(self.workers["a"].tasks[call_id], original)
        await stream.aclose()

    async def test_closed_worker_rejects_new_requests_without_task(self):
        self.workers["a"].close()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            await self.collect("a", "request")
        self.assertFalse(self.workers["a"].tasks)


if __name__ == "__main__":
    unittest.main()
