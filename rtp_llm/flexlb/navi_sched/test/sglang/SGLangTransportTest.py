import asyncio
import os
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import anyio
from navi_sched.adapters.sglang.SGLangConfig import SGLangConfig
from navi_sched.adapters.sglang.SGLangForwardProtocol import SGLangForwardCodec
from navi_sched.adapters.sglang.SGLangTransport import SGLangTransport
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import ImageData

KINDS = SimpleNamespace(ACK=0, OUTPUT=1, ERROR=2)


class _Client:
    def __init__(self):
        self.codec = SGLangForwardCodec()
        self.responses = deque()
        self.decisions = {}
        self.fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self.scheduled = []
        self.aborted = []
        self.master = True
        self.respond = True
        self.terminal = True
        self.reject = False
        self.fallback = 0
        self.allocated = 0

    def has_master_biz(self):
        return self.master

    def allocate_request_id(self):
        self.allocated += 1
        return self.allocated

    def report_local_fallback(self):
        self.fallback += 1

    def abort_forward_request(self, request_id):
        self.aborted.append(request_id)
        return True

    def notification_fd(self):
        return self.fd

    def submit_schedule(self, request, target_biz, **kwargs):
        self.scheduled.append(request)
        if self.respond and not self.reject:
            obj = self.codec.decode_request(request.forward_payload)
            self.responses.append(
                SimpleNamespace(
                    request_id=request.request_id,
                    kind=KINDS.OUTPUT,
                    frames=[
                        self.codec.encode_response(
                            {
                                "text": "result",
                                "meta_info": {"id": obj.rid, "prompt_tokens": 4},
                            }
                        )
                    ],
                    terminal=self.terminal,
                )
            )
        self.decisions[request.request_id] = SimpleNamespace(
            target=None if self.reject else "target", reason="capacity"
        )
        os.eventfd_write(self.fd, 1)

    def take_schedule_result(self, request_id):
        return self.decisions.pop(request_id, None)

    def take_next_forward_response(self, timeout_ms):
        if timeout_ms != 0:
            raise AssertionError("async integration must never block")
        return self.responses.popleft() if self.responses else None

    def stop(self):
        pass


class SGLangTransportTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = _Client()

        async def tokenize(text, is_cross_encoder):
            return [1, 2, 3, 4], None

        self.manager = SimpleNamespace(
            server_args=SimpleNamespace(page_size=4),
            max_req_input_len=4096,
            rid_to_state={},
            auto_create_handle_loop=lambda: None,
            _set_default_priority=lambda obj: None,
            _tokenize_texts=tokenize,
        )

        def init_states(obj, request):
            items = [obj] if obj.is_single else [obj[i] for i in range(len(obj.rid))]
            self.manager.rid_to_state.update((item.rid, object()) for item in items)

        self.manager._init_req_state = init_states
        self.transport = SGLangTransport(
            self.manager,
            SGLangConfig("target", "master", request_timeout_ms=1000),
            self.client,
            KINDS,
        )
        self.local_calls = 0

    async def asyncTearDown(self):
        self.transport.close()
        os.close(self.client.fd)

    async def original(self, manager, obj, request):
        self.local_calls += 1
        yield {"text": "local"}

    async def test_remote_media_is_not_processed_at_source(self):
        obj = GenerateReqInput(
            text="describe",
            image_data=ImageData("https://invalid.example/image.png"),
            rid="media",
        )
        result = [
            value async for value in self.transport.generate(obj, None, self.original)
        ]
        self.assertEqual(result[0]["meta_info"]["id"], "media")
        self.assertEqual(self.local_calls, 0)
        self.assertEqual(self.client.scheduled[0].prompt_token_count, 4096)
        self.assertEqual(self.client.scheduled[0].block_hashes, [])
        restored = self.client.codec.decode_request(
            self.client.scheduled[0].forward_payload
        )
        self.assertEqual(restored.image_data.url, "https://invalid.example/image.png")
        self.assertEqual(self.transport._streams, {})

    async def test_no_master_executes_locally_once(self):
        self.client.master = False
        result = [
            value
            async for value in self.transport.generate(
                GenerateReqInput(text="test"), None, self.original
            )
        ]
        self.assertEqual(result, [{"text": "local"}])
        self.assertEqual(self.client.fallback, 1)
        self.assertEqual(self.local_calls, 1)
        self.assertEqual(self.client.allocated, 0)

    async def test_rejection_does_not_execute_a_second_copy(self):
        self.client.reject = True
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            async for _ in self.transport.generate(
                GenerateReqInput(input_ids=[1]), None, self.original
            ):
                pass
        self.assertEqual(self.local_calls, 0)
        self.assertEqual(self.client.aborted, [1])

    async def test_session_without_target_affinity_is_rejected_before_delivery(self):
        with self.assertRaisesRegex(ValueError, "target affinity"):
            async for _ in self.transport.generate(
                GenerateReqInput(input_ids=[1], session_params={"id": "session"}),
                None,
                self.original,
            ):
                pass
        self.assertEqual(self.client.scheduled, [])
        self.assertEqual(self.client.allocated, 0)
        self.assertEqual(self.manager.rid_to_state, {})

    async def test_batch_order_and_parallel_sampling(self):
        obj = GenerateReqInput(input_ids=[[1], [2]], rid=["first", "second"])
        result = [
            value async for value in self.transport.generate(obj, None, self.original)
        ]
        self.assertEqual(
            [value["meta_info"]["id"] for value in result[0]], ["first", "second"]
        )
        self.assertEqual(len(self.client.scheduled), 2)

    async def test_remote_generation_never_uses_an_executor(self):
        with patch.object(
            asyncio.get_running_loop(),
            "run_in_executor",
            side_effect=AssertionError("adapter must not create waiting threads"),
        ):
            result = [
                value
                async for value in self.transport.generate(
                    GenerateReqInput(input_ids=[1]), None, self.original
                )
            ]
        self.assertEqual(result[0]["text"], "result")

    async def test_batch_streams_share_one_bounded_output_buffer(self):
        self.client.terminal = False
        stream = self.transport.generate(
            GenerateReqInput(input_ids=[[1], [2]], stream=True), None, self.original
        )
        await anext(stream)
        streams = list(self.transport._streams.values())
        self.assertEqual(len(streams), 2)
        self.assertIs(streams[0].queue, streams[1].queue)
        self.assertEqual(streams[0].queue.maxsize, 64)
        await stream.aclose()
        self.assertEqual(self.transport._streams, {})
        self.assertEqual(self.transport._request_tasks, {})

    async def test_closing_a_stream_aborts_native_request(self):
        self.client.terminal = False
        stream = self.transport.generate(
            GenerateReqInput(input_ids=[1], stream=True), None, self.original
        )
        result = await anext(stream)
        self.assertEqual(result["text"], "result")
        await stream.aclose()
        self.assertEqual(self.client.aborted, [1])
        self.assertEqual(self.transport._streams, {})
        self.assertEqual(self.transport._active_rids, set())

    async def test_timeout_releases_stream_and_cancels_delivery(self):
        self.transport.config = SGLangConfig("target", "master", request_timeout_ms=30)
        self.client.respond = False
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            async for _ in self.transport.generate(
                GenerateReqInput(input_ids=[1]), None, self.original
            ):
                pass
        self.assertEqual(self.client.aborted, [1])
        self.assertEqual(self.transport._streams, {})

    async def test_cancelled_http_scope_releases_request_id(self):
        self.client.terminal = False
        stream = self.transport.generate(
            GenerateReqInput(input_ids=[1], stream=True, rid="disconnect"),
            None,
            self.original,
        )
        self.assertEqual((await anext(stream))["text"], "result")
        try:
            with anyio.CancelScope() as scope:
                scope.cancel()
                await stream.aclose()
        except asyncio.CancelledError:
            pass
        self.assertEqual(self.transport._active_rids, set())
        self.assertEqual(self.transport._request_tasks, {})
        self.assertEqual(self.manager.rid_to_state, {})
        self.assertEqual(self.transport._streams, {})
        self.assertEqual(self.client.aborted, [1])
        self.client.terminal = True
        result = [
            value
            async for value in self.transport.generate(
                GenerateReqInput(input_ids=[1], rid="disconnect"),
                None,
                self.original,
            )
        ]
        self.assertEqual(result[0]["text"], "result")

    async def test_abort_during_tokenization_releases_source_state(self):
        started = asyncio.Event()

        async def tokenize(text, is_cross_encoder):
            started.set()
            await asyncio.Event().wait()

        self.manager._tokenize_texts = tokenize

        async def consume():
            async for _ in self.transport.generate(
                GenerateReqInput(text="test", rid="preparing"), None, self.original
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), 1)
        self.assertEqual(set(self.manager.rid_to_state), {"preparing"})
        self.assertEqual(self.transport._streams, {})
        self.assertTrue(self.transport.abort("preparing"))
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, 1)
        self.assertEqual(self.manager.rid_to_state, {})
        self.assertEqual(self.transport._request_tasks, {})
        self.assertEqual(self.transport._active_rids, set())
        self.assertEqual(self.client.allocated, 0)

    async def test_deadline_includes_source_tokenization(self):
        async def tokenize(text, is_cross_encoder):
            await asyncio.Event().wait()

        self.manager._tokenize_texts = tokenize
        self.transport.config = SGLangConfig("target", "master", request_timeout_ms=30)
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            async for _ in self.transport.generate(
                GenerateReqInput(text="test", rid="preparing"), None, self.original
            ):
                pass
        self.assertEqual(self.manager.rid_to_state, {})
        self.assertEqual(self.transport._request_tasks, {})
        self.assertEqual(self.transport._active_rids, set())
        self.assertEqual(self.client.allocated, 0)

    async def test_failed_source_state_initialization_does_not_leak_capacity(self):
        def fail(obj, request):
            raise ValueError("bad state")

        self.manager._init_req_state = fail
        with self.assertRaisesRegex(ValueError, "bad state"):
            async for _ in self.transport.generate(
                GenerateReqInput(input_ids=[1], rid="bad"), None, self.original
            ):
                pass
        self.assertEqual(self.transport._active_rids, set())
        self.assertEqual(self.transport._request_tasks, {})
        self.assertEqual(self.client.allocated, 0)

    async def test_parallel_samples_are_not_expanded_twice(self):
        obj = GenerateReqInput(input_ids=[1, 2], sampling_params={"n": 3})
        outputs = [
            value async for value in self.transport.generate(obj, None, self.original)
        ]
        self.assertEqual(len(outputs[0]), 3)
        self.assertEqual(len(self.client.scheduled), 3)
        self.assertEqual(len({value["meta_info"]["id"] for value in outputs[0]}), 3)
        for request in self.client.scheduled:
            restored = self.client.codec.decode_request(request.forward_payload)
            self.assertEqual(restored.sampling_params["n"], 1)
        self.assertEqual(self.manager.rid_to_state, {})


if __name__ == "__main__":
    unittest.main()
