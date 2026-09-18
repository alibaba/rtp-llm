import asyncio
import json
import time
from typing import Any
from unittest import TestCase, main

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.structure.request_constants import request_id_field_name
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import init_controller, set_global_controller


class FakePipelinResponse(BaseModel):
    res: str


class FakeFrontendWorker(object):
    class FakeBackendRpcServerVisitor:
        def __init__(self):
            self.refresh_calls = []

        def is_backend_service_ready(self, refresh: bool = False):
            self.refresh_calls.append(refresh)
            return True

    def __init__(self):
        self.backend_rpc_server_visitor = self.FakeBackendRpcServerVisitor()

    def inference(self, prompt: str, *args: Any, **kwargs: Any):
        response_generator = self._inference(prompt, *args, **kwargs)
        return CompleteResponseAsyncGenerator(
            response_generator, CompleteResponseAsyncGenerator.get_last_value
        )

    def tokenizer_encode(self, prompt: str):
        return [1, 2, 3, 4], ["b", "c", "d", "e"]

    async def _inference(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)

    def is_streaming(self, *args: Any, **kwargs: Any):
        return False


class FakeRawRequest(object):
    headers = {}

    async def is_disconnected(self):
        return False


class StallingFrontendWorker(FakeFrontendWorker):
    """A worker whose generation never yields -- the abandoned-request scenario.

    A non-streaming backend buffers every output and yields only once generation completes, so this
    is what a stalled prefill/decode looks like from the frontend: no item ever arrives.
    """

    def __init__(self):
        super().__init__()
        self.entered = False
        self.finalized = False

    async def _inference(self, prompt: str, *args: Any, **kwargs: Any):
        self.entered = True
        try:
            await asyncio.sleep(3600)
        finally:
            # Reached only if the cancellation is propagated down the nested generator chain, which
            # is what makes the model-RPC client cancel the gRPC call and free the backend stream.
            self.finalized = True
        yield FakePipelinResponse(res=prompt)


class DisconnectingRawRequest(FakeRawRequest):
    """Reports connected for the first ``connected_polls`` polls, then disconnected."""

    def __init__(self, connected_polls: int = 1):
        self.connected_polls = connected_polls
        self.polls = 0

    async def is_disconnected(self):
        self.polls += 1
        return self.polls > self.connected_polls


class FrontendServerTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Create PyEnvConfigs with default values for testing
        py_env_configs = PyEnvConfigs()
        set_global_controller(init_controller(py_env_configs.concurrency_config))
        py_env_configs.server_config.start_port = 0
        py_env_configs.server_config.rank_id = 0
        self.frontend_server = FrontendServer(
            rank_id=0,
            server_id=0,
            py_env_configs=py_env_configs,
        )
        self.frontend_server._frontend_worker = FakeFrontendWorker()

    async def _async_run(self, *args: Any, **kwargs: Any):
        res = await self.frontend_server.inference(*args, **kwargs)
        return res

    def test_simple(self):
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(
            self._async_run(req={"prompt": "hello"}, raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )
        res = loop.run_until_complete(
            self._async_run(req='{"prompt": "hello"}', raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )

    def test_encode(self):
        res = self.frontend_server.tokenizer_encode('{"prompt": "b c d e"}')
        self.assertEqual(
            res.body.decode("utf-8"),
            '{"token_ids":[1,2,3,4],"tokens":["b","c","d","e"],"error":""}',
        )
        # test error input
        res = self.frontend_server.tokenizer_encode('{"text": "b c d e"}')
        self.assertEqual(json.loads(res.body.decode("utf-8"))["error_code"], 514)

    def test_check_health_uses_cached_service_discovery(self):
        self.assertTrue(self.frontend_server.check_health())
        visitor = self.frontend_server._frontend_worker.backend_rpc_server_visitor
        self.assertEqual(visitor.refresh_calls, [False])

    def test_disconnect_during_a_stalled_non_streaming_generation_is_bounded(self):
        # Regression: the disconnect check used to sit inside `async for x in res`. A non-streaming
        # response yields nothing until generation completes, so with a STALLED generation the check
        # never ran and the request stayed resident on the backend (engine stream, KV blocks and the
        # per-rank admission slot) indefinitely after the client was gone. The poller must bound it,
        # and the cancellation must reach the generator so the backend RPC is cancelled.
        worker = StallingFrontendWorker()
        self.frontend_server._frontend_worker = worker

        def generate_call():
            return worker.inference(prompt="hello")

        loop = asyncio.new_event_loop()
        started = time.monotonic()
        try:
            with self.assertRaises(asyncio.CancelledError):
                loop.run_until_complete(
                    self.frontend_server._infer_impl(
                        # inference() normally injects the request id, and _infer_impl's access log
                        # reads it, so a direct call has to supply it.
                        req={"prompt": "hello", request_id_field_name: 12345},
                        raw_request=DisconnectingRawRequest(connected_polls=1),
                        generate_call=generate_call,
                    )
                )
        finally:
            elapsed = time.monotonic() - started
            loop.close()

        self.assertTrue(worker.entered, "generation never started")
        self.assertTrue(worker.finalized, "the stalled generator was not finalized on disconnect")
        self.assertLess(
            elapsed, 5.0, f"disconnect was not acted on promptly (took {elapsed:.1f}s)"
        )

    def test_disconnect_during_a_stalled_generation_returns_a_handled_response(self):
        # Same scenario through the public entry point: _infer_wrap turns the cancellation into an
        # error response, so the caller gets a prompt reply instead of a hung connection.
        worker = StallingFrontendWorker()
        self.frontend_server._frontend_worker = worker

        loop = asyncio.new_event_loop()
        started = time.monotonic()
        try:
            rep = loop.run_until_complete(
                self.frontend_server.inference(
                    req={"prompt": "hello"},
                    raw_request=DisconnectingRawRequest(connected_polls=1),
                )
            )
        finally:
            elapsed = time.monotonic() - started
            loop.close()

        self.assertIsNotNone(rep)
        self.assertNotIsInstance(rep, StreamingResponse)
        self.assertTrue(worker.finalized, "the stalled generator was not finalized on disconnect")
        self.assertLess(elapsed, 5.0, f"took {elapsed:.1f}s to act on the disconnect")

    def test_a_connected_client_still_gets_its_response(self):
        # Guard the other direction: the poller must not fire for a client that stays connected.
        loop = asyncio.new_event_loop()
        try:
            res = loop.run_until_complete(
                self.frontend_server.inference(
                    req={"prompt": "hello"}, raw_request=FakeRawRequest()
                )
            )
        finally:
            loop.close()
        self.assertEqual(res.body.decode("utf-8"), '{"res":"hello"}')


main()
