import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from navi_sched.adapters.sglang.SGLangConfig import SGLangConfig
from navi_sched.adapters.sglang.SGLangRuntime import SGLangRuntime, _ForwardBatch
from sglang.srt.managers.io_struct import GenerateReqInput


class _Bridge:
    def __init__(self):
        self.events = []

    def finish_batch(self, handle, stats):
        self.events.append(("prefill", stats))
        return True

    def release_forward_batch(self, batch):
        self.events.append(("release", batch.batch_id))

    def publish_forward_response(self, request_id, frames, terminal):
        if not frames:
            raise AssertionError("native NaviSched rejects empty output frames")
        self.events.append(("output", terminal))
        return True

    def fail_forward_response(self, request_id, error):
        self.events.append(("error", error))

    def disable_forwarding(self):
        pass

    def abort_forward_batch_requests(self, batch, request_ids):
        self.events.append(("abort", request_ids))
        batch.request_ids = [rid for rid in batch.request_ids if rid not in request_ids]
        return True


class SGLangRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_first_token_finishes_prefill_and_last_frame_finishes_response(self):
        aborted = []

        async def original(manager, obj, request):
            self.assertTrue(obj.stream)
            self.assertEqual(obj.rid, "navi_sched_7")
            for finish in (None, {"type": "length"}):
                yield {
                    "text": "answer",
                    "meta_info": {
                        "id": obj.rid,
                        "prompt_tokens": 8,
                        "cached_tokens": 4,
                        "finish_reason": finish,
                    },
                }

        with patch.object(
            SGLangConfig, "from_env", return_value=SGLangConfig("worker", "master")
        ):
            runtime = SGLangRuntime(
                SimpleNamespace(abort_request=aborted.append), original
            )
        bridge = _Bridge()
        runtime.bridge = bridge
        state = _ForwardBatch(
            SimpleNamespace(batch_id=1, request_ids=[7]), object(), remaining={7}
        )
        runtime._batches[1] = state
        try:
            await runtime._execute(
                7,
                runtime.codec.encode_request(
                    GenerateReqInput(text="test", rid="external")
                ),
                state,
            )
            self.assertEqual(
                bridge.events,
                [
                    ("prefill", [(7, 8, 4)]),
                    ("release", 1),
                    ("output", False),
                    ("output", True),
                ],
            )
            self.assertEqual(aborted, ["navi_sched_7"])
            self.assertEqual(runtime._batches, {})
            self.assertEqual(runtime.snapshot["prefill_completed_request_count"], 1)
            self.assertEqual(runtime.snapshot["response_frame_count"], 2)
        finally:
            runtime.close()

    async def test_failed_item_releases_its_share_and_finishes_ready_prefill(self):
        async def original(manager, obj, request):
            raise ValueError("bad request")
            yield

        with patch.object(
            SGLangConfig, "from_env", return_value=SGLangConfig("worker", "master")
        ):
            runtime = SGLangRuntime(
                SimpleNamespace(abort_request=lambda rid: None), original
            )
        runtime.bridge = _Bridge()
        state = _ForwardBatch(
            SimpleNamespace(batch_id=1, request_ids=[7, 8]),
            object(),
            stats={8: (8, 8, 4)},
            remaining={7, 8},
        )
        runtime._batches[1] = state
        try:
            await runtime._execute(
                7, runtime.codec.encode_request(GenerateReqInput(text="test")), state
            )
            self.assertEqual(
                runtime.bridge.events,
                [
                    ("error", "bad request"),
                    ("abort", [7]),
                    ("prefill", [(8, 8, 4)]),
                    ("release", 1),
                ],
            )
            self.assertEqual(runtime._batches, {})
            self.assertIsNone(state.handle)
            self.assertEqual(state.remaining, {8})
        finally:
            runtime.close()

    async def test_close_releases_reporter_before_event_loop_stops(self):
        aborted = []
        with patch.object(
            SGLangConfig, "from_env", return_value=SGLangConfig("worker", "master")
        ):
            runtime = SGLangRuntime(SimpleNamespace(abort_request=aborted.append), None)
        runtime.bridge = _Bridge()
        state = _ForwardBatch(
            SimpleNamespace(batch_id=1, request_ids=[7]), object(), remaining={7}
        )
        runtime._batches[1] = state
        task = asyncio.create_task(asyncio.Event().wait())
        runtime._execution_tasks[7] = task
        runtime.close()
        self.assertEqual(runtime._batches, {})
        self.assertIsNone(state.handle)
        self.assertEqual(aborted, ["navi_sched_7"])
        self.assertEqual(runtime.bridge.events, [("release", 1)])
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
