import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch

from fastapi import Request

from rtp_llm.frontend.frontend_server import (
    _AdmissionOwner,
    AdmissionStreamingResponse,
    FrontendServer,
)
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import ConcurrencyController


class _RawRequest:
    def __init__(self):
        self._receive_blocker = asyncio.Event()

    async def is_disconnected(self):
        return False

    async def receive(self):
        await self._receive_blocker.wait()
        raise AssertionError("unreachable")


class _ControlledReceive:
    def __init__(self):
        self.disconnect = asyncio.Event()
        self.calls = 0
        self.active_receives = 0
        self.max_active_receives = 0

    async def __call__(self):
        self.calls += 1
        self.active_receives += 1
        self.max_active_receives = max(
            self.max_active_receives, self.active_receives
        )
        try:
            await self.disconnect.wait()
            return {"type": "http.disconnect"}
        finally:
            self.active_receives -= 1


class _Chunk:
    def model_dump_json(self, exclude_none=True):
        return '{"text":"ok"}'


class _ModelResult:
    def model_dump(self, exclude_none=True):
        return {"result": "ok"}


class _ControlledStreamingSource:
    def __init__(self):
        self.next_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.close_calls = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.next_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def aclose(self):
        self.close_calls += 1
        self.close_started.set()
        await self.allow_close.wait()
        self.closed = True


class _FailingCloseStreamingSource:
    def __init__(self):
        self.next_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.next_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def aclose(self):
        self.close_calls += 1
        self.close_started.set()
        raise RuntimeError("backend close failed")


class _TerminalSource:
    def __init__(self, error, close_error=None, block_close=False):
        self.error = error
        self.close_error = close_error
        self.block_close = block_close
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.close_calls = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self.error

    async def aclose(self):
        self.close_calls += 1
        self.close_started.set()
        if self.block_close:
            await self.allow_close.wait()
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class FrontendAdmissionOverflowTest(IsolatedAsyncioTestCase):
    def make_server(self, max_concurrency=1) -> FrontendServer:
        server = object.__new__(FrontendServer)
        server._global_controller = ConcurrencyController(
            max_concurrency=max_concurrency
        )
        server._access_logger = MagicMock()
        server.rank_id = "0"
        server.server_id = "0"
        server.py_env_configs = SimpleNamespace(
            server_config=SimpleNamespace(ip="127.0.0.1", server_port=30000)
        )
        return server

    @staticmethod
    def make_finite_response() -> CompleteResponseAsyncGenerator:
        async def source():
            yield _Chunk()

        async def collect_response(responses):
            async for _ in responses:
                pass
            return {"result": "ok"}

        return CompleteResponseAsyncGenerator(source(), collect_response)

    def configure_language_backends(self, server: FrontendServer) -> None:
        async def batch_chat_completion(*args, **kwargs):
            return [_ModelResult()]

        async def batch_infer(*args, **kwargs):
            return _ModelResult()

        server._frontend_worker = SimpleNamespace(
            inference=lambda **request: self.make_finite_response(),
            batch_infer=batch_infer,
            is_streaming=lambda request: request.get("stream", False),
        )
        server._openai_endpoint = SimpleNamespace(
            chat_completion=lambda *args, **kwargs: self.make_finite_response(),
            batch_chat_completion=batch_chat_completion,
        )
        server._embedding_endpoint = SimpleNamespace(
            embedding=lambda request: asyncio.sleep(
                0, result=({"data": [], "usage": {}}, None)
            )
        )

    @staticmethod
    def make_response(source) -> CompleteResponseAsyncGenerator:
        async def collect_response(responses):
            async for _ in responses:
                pass
            return {"result": "ok"}

        return CompleteResponseAsyncGenerator(source, collect_response)

    def configure_terminal_backend(self, server, route, source, streaming=False):
        response_factory = lambda *args, **kwargs: self.make_response(source)
        if route == "native":
            server._frontend_worker = SimpleNamespace(
                inference=response_factory,
                is_streaming=lambda request: streaming,
            )
        else:
            server._frontend_worker = SimpleNamespace(
                is_streaming=lambda request: streaming,
            )
            server._openai_endpoint = SimpleNamespace(
                chat_completion=response_factory,
            )

    @staticmethod
    def make_raw_request(receive) -> Request:
        return Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "method": "POST",
                "path": "/",
                "headers": [],
                "query_string": b"",
            },
            receive=receive,
        )

    async def invoke_language_route(
        self, server, route, streaming=False, raw_request=None
    ):
        raw_request = raw_request or _RawRequest()
        if route == "native":
            return await server.inference(
                {"prompt": "test", "stream": streaming}, raw_request
            )
        request = MagicMock()
        request.model_dump.return_value = {"model": "test", "stream": streaming}
        return await server.chat_completion(request, raw_request)

    def make_streaming_server(self, max_concurrency=1) -> FrontendServer:
        server = self.make_server(max_concurrency=max_concurrency)

        async def collect_last(responses):
            response = None
            async for response in responses:
                pass
            return response

        def inference(**request):
            async def source():
                await asyncio.Event().wait()
                yield _Chunk()

            return CompleteResponseAsyncGenerator(source(), collect_last)

        server._frontend_worker = SimpleNamespace(
            inference=inference,
            is_streaming=lambda request: request.get("stream", False),
        )
        return server

    async def disconnect_response(self, response):
        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        await response(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "method": "POST",
                "path": "/",
                "headers": [],
                "query_string": b"",
            },
            receive,
            send,
        )

    def assert_concurrency_response(
        self, response, expected_status=409, expected_limit=1
    ):
        self.assertEqual(response.status_code, expected_status)
        self.assertEqual(response.media_type, "application/json")
        body = json.loads(response.body)
        self.assertEqual(body["error_code"], 409)
        self.assertEqual(body["error_code_str"], "409_CONCURRENCY_LIMIT_ERROR")
        self.assertIn(
            f"Concurrency limit {expected_limit} reached", body["message"]
        )

    async def test_openai_admission_overflow_returns_structured_http_conflict(self):
        server = self.make_server()
        server._global_controller.increment()
        request = MagicMock()
        request.model_dump.return_value = {"model": "test"}

        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response = await FrontendServer.chat_completion(
                server, request, MagicMock()
            )

        self.assert_concurrency_response(response)
        self.assertEqual(server._global_controller.get_available_concurrency(), 0)
        server._global_controller.decrement()
        self.assertEqual(server._global_controller.get_available_concurrency(), 1)

    async def test_all_language_routes_share_the_http_conflict_contract(self):
        cases = (
            ("native", lambda server: server.inference({}, MagicMock())),
            ("embedding", lambda server: server.embedding({}, MagicMock())),
            (
                "batch_chat",
                lambda server: server.batch_chat_completion(MagicMock(), MagicMock()),
            ),
            ("batch_infer", lambda server: server.batch_infer({}, MagicMock())),
        )

        for name, invoke in cases:
            with self.subTest(route=name):
                server = self.make_server()
                server._global_controller.increment()
                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response = await invoke(server)
                self.assert_concurrency_response(response)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 0
                )

    async def test_request_id_failure_releases_acquired_slot(self):
        request = MagicMock()
        request.model_dump.return_value = {"model": "test"}
        cases = (
            ("native", lambda server: server.inference({}, MagicMock())),
            ("openai", lambda server: server.chat_completion(request, MagicMock())),
            ("embedding", lambda server: server.embedding({}, MagicMock())),
            (
                "batch_chat",
                lambda server: server.batch_chat_completion(request, MagicMock()),
            ),
            ("batch_infer", lambda server: server.batch_infer({}, MagicMock())),
        )

        for name, invoke in cases:
            with self.subTest(route=name):
                server = self.make_server()
                with patch(
                    "rtp_llm.frontend.frontend_server.generate_request_id",
                    side_effect=RuntimeError("request id failed"),
                ), patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response = await invoke(server)

                self.assertEqual(response.status_code, 500)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_request_dump_failure_releases_acquired_slot(self):
        server = self.make_server()
        request = MagicMock()
        request.model_dump.side_effect = RuntimeError("request dump failed")

        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response = await FrontendServer.chat_completion(
                server, request, MagicMock()
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(server._global_controller.get_available_concurrency(), 1)

    async def test_all_language_routes_release_slot_after_success(self):
        request = MagicMock()
        request.model_dump.return_value = {"model": "test"}
        cases = (
            ("native", lambda server: server.inference({}, _RawRequest())),
            ("openai", lambda server: server.chat_completion(request, _RawRequest())),
            ("embedding", lambda server: server.embedding({}, _RawRequest())),
            (
                "batch_chat",
                lambda server: server.batch_chat_completion(request, _RawRequest()),
            ),
            ("batch_infer", lambda server: server.batch_infer({}, _RawRequest())),
        )

        for name, invoke in cases:
            with self.subTest(route=name):
                server = self.make_server()
                self.configure_language_backends(server)
                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response = await invoke(server)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_all_language_routes_release_slot_after_backend_error(self):
        request = MagicMock()
        request.model_dump.return_value = {"model": "test"}

        def raise_error(*args, **kwargs):
            raise RuntimeError("backend failed")

        async def raise_async_error(*args, **kwargs):
            raise RuntimeError("backend failed")

        cases = ("native", "openai", "embedding", "batch_chat", "batch_infer")
        for name in cases:
            with self.subTest(route=name):
                server = self.make_server()
                self.configure_language_backends(server)
                if name == "native":
                    server._frontend_worker.inference = raise_error
                    invoke = lambda: server.inference({}, _RawRequest())
                elif name == "openai":
                    server._openai_endpoint.chat_completion = raise_error
                    invoke = lambda: server.chat_completion(request, _RawRequest())
                elif name == "embedding":
                    server._embedding_endpoint.embedding = raise_async_error
                    invoke = lambda: server.embedding({}, _RawRequest())
                elif name == "batch_chat":
                    server._openai_endpoint.batch_chat_completion = raise_async_error
                    invoke = lambda: server.batch_chat_completion(
                        request, _RawRequest()
                    )
                else:
                    server._frontend_worker.batch_infer = raise_async_error
                    invoke = lambda: server.batch_infer({}, _RawRequest())

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    if name in ("native", "openai", "embedding"):
                        response = await invoke()
                        self.assertEqual(response.status_code, 500)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "backend failed"):
                            await invoke()
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_all_language_routes_release_slot_after_cancellation(self):
        request = MagicMock()
        request.model_dump.return_value = {"model": "test"}

        def cancel(*args, **kwargs):
            raise asyncio.CancelledError("cancelled")

        async def cancel_async(*args, **kwargs):
            raise asyncio.CancelledError("cancelled")

        cases = ("native", "openai", "embedding", "batch_chat", "batch_infer")
        for name in cases:
            with self.subTest(route=name):
                server = self.make_server()
                self.configure_language_backends(server)
                if name == "native":
                    server._frontend_worker.inference = cancel
                    invoke = lambda: server.inference({}, _RawRequest())
                elif name == "openai":
                    server._openai_endpoint.chat_completion = cancel
                    invoke = lambda: server.chat_completion(request, _RawRequest())
                elif name == "embedding":
                    server._embedding_endpoint.embedding = cancel_async
                    invoke = lambda: server.embedding({}, _RawRequest())
                elif name == "batch_chat":
                    server._openai_endpoint.batch_chat_completion = cancel_async
                    invoke = lambda: server.batch_chat_completion(
                        request, _RawRequest()
                    )
                else:
                    server._frontend_worker.batch_infer = cancel_async
                    invoke = lambda: server.batch_infer({}, _RawRequest())

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    if name in ("native", "openai", "embedding"):
                        response = await invoke()
                        self.assertEqual(response.status_code, 500)
                    else:
                        with self.assertRaises(asyncio.CancelledError):
                            await invoke()
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_nonstream_iteration_error_releases_slot_after_cleanup(self):
        for route in ("native", "openai"):
            with self.subTest(route=route):
                server = self.make_server()
                source = _TerminalSource(RuntimeError("backend iteration failed"))
                self.configure_terminal_backend(server, route, source)

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response = await self.invoke_language_route(server, route)

                self.assertEqual(response.status_code, 500)
                self.assertTrue(source.closed)
                self.assertEqual(source.close_calls, 1)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_nonstream_iteration_cancellation_releases_slot_after_cleanup(self):
        for route in ("native", "openai"):
            with self.subTest(route=route):
                server = self.make_server()
                source = _TerminalSource(asyncio.CancelledError("backend cancelled"))
                self.configure_terminal_backend(server, route, source)

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response = await self.invoke_language_route(server, route)

                self.assertEqual(response.status_code, 500)
                self.assertTrue(source.closed)
                self.assertEqual(source.close_calls, 1)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_nonstream_disconnect_closes_backend_before_releasing_slot(self):
        for route in ("native", "openai"):
            with self.subTest(route=route):
                server = self.make_server()
                source = _ControlledStreamingSource()
                self.configure_terminal_backend(server, route, source)
                receive = _ControlledReceive()
                raw_request = self.make_raw_request(receive)

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response_task = asyncio.create_task(
                        self.invoke_language_route(
                            server, route, raw_request=raw_request
                        )
                    )
                    try:
                        await asyncio.wait_for(source.next_started.wait(), timeout=1)
                        receive.disconnect.set()
                        await asyncio.wait_for(
                            source.close_started.wait(), timeout=0.2
                        )
                        self.assertFalse(response_task.done())
                        self.assertEqual(
                            server._global_controller.get_available_concurrency(), 0
                        )
                        rejected = await server.batch_infer({}, MagicMock())
                        self.assert_concurrency_response(rejected)
                        source.allow_close.set()
                        response = await asyncio.wait_for(response_task, timeout=1)
                    except BaseException:
                        source.allow_close.set()
                        if not response_task.done():
                            response_task.cancel()
                        await asyncio.gather(response_task, return_exceptions=True)
                        raise

                self.assertEqual(response.status_code, 500)
                self.assertTrue(response_task.done())
                self.assertTrue(source.closed)
                self.assertEqual(source.close_calls, 1)
                self.assertGreaterEqual(receive.calls, 1)
                self.assertEqual(receive.max_active_receives, 1)
                self.assertEqual(receive.active_receives, 0)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_nonstream_completion_cancels_disconnect_receiver(self):
        for route in ("native", "openai"):
            with self.subTest(route=route):
                server = self.make_server()
                self.configure_language_backends(server)
                receive = _ControlledReceive()
                raw_request = self.make_raw_request(receive)

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response = await self.invoke_language_route(
                        server, route, raw_request=raw_request
                    )

                self.assertEqual(response.status_code, 200)
                self.assertLessEqual(receive.max_active_receives, 1)
                self.assertEqual(receive.active_receives, 0)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_nonstream_observes_disconnect_already_consumed_by_request(self):
        server = self.make_server()
        source = _ControlledStreamingSource()
        self.configure_terminal_backend(server, "openai", source)
        receive = _ControlledReceive()
        raw_request = self.make_raw_request(receive)
        receive.disconnect.set()
        self.assertTrue(await raw_request.is_disconnected())

        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response_task = asyncio.create_task(
                self.invoke_language_route(
                    server, "openai", raw_request=raw_request
                )
            )
            await asyncio.wait_for(source.close_started.wait(), timeout=1)
            try:
                self.assertFalse(response_task.done())
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 0
                )
            finally:
                source.allow_close.set()
            response = await asyncio.wait_for(response_task, timeout=1)

        self.assertEqual(response.status_code, 500)
        self.assertTrue(source.closed)
        self.assertEqual(source.close_calls, 1)
        self.assertEqual(receive.active_receives, 0)
        self.assertEqual(server._global_controller.get_available_concurrency(), 1)

    async def test_nonstream_success_wins_when_disconnect_arrives_at_completion(self):
        for route in ("native", "openai"):
            with self.subTest(route=route):
                server = self.make_server()
                receive = _ControlledReceive()

                async def source():
                    yield _Chunk()

                async def collect_response(responses):
                    async for _ in responses:
                        pass
                    receive.disconnect.set()
                    return {"result": "ok"}

                response_factory = lambda *args, **kwargs: (
                    CompleteResponseAsyncGenerator(source(), collect_response)
                )
                if route == "native":
                    server._frontend_worker = SimpleNamespace(
                        inference=response_factory,
                        is_streaming=lambda request: False,
                    )
                else:
                    server._frontend_worker = SimpleNamespace(
                        is_streaming=lambda request: False,
                    )
                    server._openai_endpoint = SimpleNamespace(
                        chat_completion=response_factory
                    )

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response = await self.invoke_language_route(
                        server,
                        route,
                        raw_request=self.make_raw_request(receive),
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(json.loads(response.body), {"result": "ok"})
                self.assertEqual(receive.active_receives, 0)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_nonstream_cleanup_block_holds_slot_until_cleanup_finishes(self):
        for route in ("native", "openai"):
            with self.subTest(route=route):
                server = self.make_server()
                source = _TerminalSource(
                    RuntimeError("backend iteration failed"), block_close=True
                )
                self.configure_terminal_backend(server, route, source)

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response_task = asyncio.create_task(
                        self.invoke_language_route(server, route)
                    )
                    await asyncio.wait_for(source.close_started.wait(), timeout=1)
                    try:
                        self.assertFalse(response_task.done())
                        self.assertEqual(
                            server._global_controller.get_available_concurrency(), 0
                        )
                        rejected = await server.batch_infer({}, MagicMock())
                        self.assert_concurrency_response(rejected)
                    finally:
                        source.allow_close.set()
                    response = await asyncio.wait_for(response_task, timeout=1)

                self.assertEqual(response.status_code, 500)
                self.assertTrue(source.closed)
                self.assertEqual(source.close_calls, 1)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_nonstream_cleanup_failure_keeps_slot_admitted(self):
        for route in ("native", "openai"):
            with self.subTest(route=route):
                server = self.make_server()
                source = _TerminalSource(
                    RuntimeError("backend iteration failed"),
                    close_error=RuntimeError("backend close failed"),
                )
                self.configure_terminal_backend(server, route, source)

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response = await self.invoke_language_route(server, route)

                self.assertEqual(response.status_code, 500)
                self.assertFalse(source.closed)
                self.assertGreaterEqual(source.close_calls, 1)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 0
                )
                rejected = await server.batch_infer({}, MagicMock())
                self.assert_concurrency_response(rejected)

    async def test_nonstream_cancelled_waiter_holds_slot_until_cleanup_finishes(self):
        for route in ("native", "openai"):
            with self.subTest(route=route):
                server = self.make_server()
                source = _TerminalSource(
                    RuntimeError("backend iteration failed"), block_close=True
                )
                self.configure_terminal_backend(server, route, source)

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response_task = asyncio.create_task(
                        self.invoke_language_route(server, route)
                    )
                    await asyncio.wait_for(source.close_started.wait(), timeout=1)
                    response_task.cancel()
                    try:
                        await asyncio.sleep(0)
                        self.assertFalse(response_task.done())
                        self.assertEqual(
                            server._global_controller.get_available_concurrency(), 0
                        )
                        rejected = await server.batch_infer({}, MagicMock())
                        self.assert_concurrency_response(rejected)
                    finally:
                        source.allow_close.set()
                    response = await asyncio.wait_for(response_task, timeout=1)

                self.assertEqual(response.status_code, 500)
                self.assertTrue(source.closed)
                self.assertEqual(source.close_calls, 1)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    async def test_nonstream_cleanup_completion_race_releases_before_cancellation(self):
        server = self.make_server()
        lease = server._global_controller.acquire()
        admission_owner = _AdmissionOwner(lease)
        admission_owner.transfer_to_cleanup()
        response = MagicMock()
        closed = False

        async def close_response():
            nonlocal closed
            closed = True

        async def cancel_after_completion(close_task):
            await close_task
            raise asyncio.CancelledError("cancelled after cleanup completed")

        response.aclose = close_response
        with patch(
            "rtp_llm.frontend.frontend_server.asyncio.shield",
            side_effect=cancel_after_completion,
        ):
            with self.assertRaisesRegex(
                asyncio.CancelledError, "cancelled after cleanup completed"
            ):
                await FrontendServer._close_nonstream_response(
                    response, admission_owner
                )

        self.assertTrue(closed)
        self.assertEqual(server._global_controller.get_available_concurrency(), 1)

    async def test_streaming_iteration_error_releases_slot_after_cleanup(self):
        for route in ("native", "openai"):
            with self.subTest(route=route):
                server = self.make_server()
                source = _TerminalSource(RuntimeError("backend iteration failed"))
                self.configure_terminal_backend(
                    server, route, source, streaming=True
                )
                sent = []

                async def receive():
                    await asyncio.Event().wait()
                    raise AssertionError("unreachable")

                async def send(message):
                    sent.append(message)

                with patch(
                    "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
                ):
                    response = await self.invoke_language_route(
                        server, route, streaming=True
                    )
                    await response(
                        {
                            "type": "http",
                            "asgi": {"version": "3.0"},
                            "method": "POST",
                            "path": "/",
                            "headers": [],
                            "query_string": b"",
                        },
                        receive,
                        send,
                    )

                body = b"".join(
                    message.get("body", b"")
                    for message in sent
                    if message["type"] == "http.response.body"
                )
                self.assertIn(b"backend iteration failed", body)
                self.assertTrue(source.closed)
                self.assertEqual(source.close_calls, 1)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 1
                )

    def test_lease_release_is_thread_safe_and_idempotent(self):
        controller = ConcurrencyController(max_concurrency=32)
        leases = [controller.acquire() for _ in range(32)]
        self.assertEqual(controller.get_available_concurrency(), 0)

        with ThreadPoolExecutor(max_workers=64) as executor:
            releases = list(
                executor.map(
                    lambda lease: lease.release(),
                    [lease for lease in leases for _ in range(2)],
                )
            )

        self.assertEqual(releases.count(True), 32)
        self.assertEqual(releases.count(False), 32)
        self.assertEqual(controller.get_available_concurrency(), 32)
        with self.assertRaisesRegex(RuntimeError, "without an active request"):
            controller.decrement()
        self.assertEqual(controller.get_available_concurrency(), 32)

    async def test_immediate_asgi_disconnect_releases_streaming_slot_once(self):
        for _ in range(100):
            server = self.make_streaming_server()
            with patch(
                "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
            ):
                response = await server.inference(
                    {"prompt": "test", "stream": True}, _RawRequest()
                )
                self.assertIsInstance(response, AdmissionStreamingResponse)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 0
                )
                await self.disconnect_response(response)
                await response.body_iterator.aclose()

            self.assertEqual(
                server._global_controller.get_available_concurrency(), 1
            )

    async def test_streaming_response_holds_slot_until_asgi_termination(self):
        server = self.make_streaming_server()
        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response = await server.inference(
                {"prompt": "test", "stream": True}, _RawRequest()
            )
            self.assertEqual(server._global_controller.get_available_concurrency(), 0)

            rejected = await server.batch_infer({}, MagicMock())
            self.assert_concurrency_response(rejected)
            self.assertEqual(server._global_controller.get_available_concurrency(), 0)

            await self.disconnect_response(response)

        self.assertEqual(server._global_controller.get_available_concurrency(), 1)

    async def test_streaming_natural_completion_releases_slot(self):
        server = self.make_server()
        self.configure_language_backends(server)
        sent = []

        async def receive():
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send(message):
            sent.append(message)

        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response = await server.inference(
                {"prompt": "test", "stream": True}, _RawRequest()
            )
            await response(
                {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "method": "POST",
                    "path": "/",
                    "headers": [],
                    "query_string": b"",
                },
                receive,
                send,
            )

        body = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        self.assertIn(b"data: [DONE]", body)
        self.assertEqual(server._global_controller.get_available_concurrency(), 1)

    async def test_c32_holders_reject_c64_mixed_routes_and_fully_recover(self):
        server = self.make_streaming_server(max_concurrency=32)
        request = MagicMock()
        holders = []
        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            for _ in range(32):
                holders.append(
                    await server.inference(
                        {"prompt": "test", "stream": True}, _RawRequest()
                    )
                )
            self.assertEqual(server._global_controller.get_available_concurrency(), 0)

            invocations = (
                lambda: server.inference({}, _RawRequest()),
                lambda: server.chat_completion(request, _RawRequest()),
                lambda: server.batch_chat_completion(request, _RawRequest()),
                lambda: server.batch_infer({}, _RawRequest()),
            )
            rejected = await asyncio.gather(
                *(invocations[index % len(invocations)]() for index in range(64))
            )
            for response in rejected:
                self.assert_concurrency_response(response, expected_limit=32)
            self.assertEqual(server._global_controller.get_available_concurrency(), 0)

            await asyncio.gather(
                *(self.disconnect_response(response) for response in holders)
            )

        self.assertEqual(server._global_controller.get_available_concurrency(), 32)

    async def test_disconnect_holds_slot_until_backend_close_completes(self):
        server = self.make_server()
        source = _ControlledStreamingSource()

        async def collect_last(responses):
            response = None
            async for response in responses:
                pass
            return response

        server._frontend_worker = SimpleNamespace(
            inference=lambda **request: CompleteResponseAsyncGenerator(
                source, collect_last
            ),
            is_streaming=lambda request: True,
        )
        disconnect = asyncio.Event()

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response = await server.inference(
                {"prompt": "test", "stream": True}, _RawRequest()
            )
            response_task = asyncio.create_task(
                response(
                    {
                        "type": "http",
                        "asgi": {"version": "3.0"},
                        "method": "POST",
                        "path": "/",
                        "headers": [],
                        "query_string": b"",
                    },
                    receive,
                    send,
                )
            )
            await asyncio.wait_for(source.next_started.wait(), timeout=1)
            disconnect.set()
            await asyncio.wait_for(source.close_started.wait(), timeout=1)
            try:
                await asyncio.sleep(0.02)

                self.assertFalse(response_task.done())
                self.assertFalse(response._cleanup_task.done())
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 0
                )
                rejected = await server.batch_infer({}, MagicMock())
                self.assert_concurrency_response(rejected)
            finally:
                source.allow_close.set()
            await asyncio.wait_for(response_task, timeout=1)

        self.assertTrue(source.closed)
        self.assertEqual(source.close_calls, 1)
        self.assertEqual(server._global_controller.get_available_concurrency(), 1)

    async def test_cancelled_asgi_waiter_does_not_release_before_backend_close(self):
        server = self.make_server()
        source = _ControlledStreamingSource()

        async def collect_last(responses):
            async for _ in responses:
                pass
            return None

        server._frontend_worker = SimpleNamespace(
            inference=lambda **request: CompleteResponseAsyncGenerator(
                source, collect_last
            ),
            is_streaming=lambda request: True,
        )
        disconnect = asyncio.Event()

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response = await server.inference(
                {"prompt": "test", "stream": True}, _RawRequest()
            )
            response_task = asyncio.create_task(
                response(
                    {
                        "type": "http",
                        "asgi": {"version": "3.0"},
                        "method": "POST",
                        "path": "/",
                        "headers": [],
                        "query_string": b"",
                    },
                    receive,
                    send,
                )
            )
            await asyncio.wait_for(source.next_started.wait(), timeout=1)
            disconnect.set()
            await asyncio.wait_for(source.close_started.wait(), timeout=1)
            try:
                response_task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(response_task.done())
                self.assertFalse(response._cleanup_task.done())
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 0
                )
            finally:
                source.allow_close.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(response_task, timeout=1)
            await asyncio.wait_for(asyncio.shield(response._cleanup_task), timeout=1)

        self.assertTrue(source.closed)
        self.assertEqual(source.close_calls, 1)
        self.assertEqual(server._global_controller.get_available_concurrency(), 1)

    async def test_backend_close_failure_keeps_streaming_slot_admitted(self):
        server = self.make_server()
        source = _FailingCloseStreamingSource()

        async def collect_last(responses):
            async for _ in responses:
                pass
            return None

        server._frontend_worker = SimpleNamespace(
            inference=lambda **request: CompleteResponseAsyncGenerator(
                source, collect_last
            ),
            is_streaming=lambda request: True,
        )
        disconnect = asyncio.Event()

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response = await server.inference(
                {"prompt": "test", "stream": True}, _RawRequest()
            )
            response_task = asyncio.create_task(
                response(
                    {
                        "type": "http",
                        "asgi": {"version": "3.0"},
                        "method": "POST",
                        "path": "/",
                        "headers": [],
                        "query_string": b"",
                    },
                    receive,
                    send,
                )
            )
            await asyncio.wait_for(source.next_started.wait(), timeout=1)
            disconnect.set()
            await asyncio.wait_for(source.close_started.wait(), timeout=1)

            with self.assertRaisesRegex(RuntimeError, "backend close failed"):
                await asyncio.wait_for(response_task, timeout=1)

        self.assertGreaterEqual(source.close_calls, 1)
        self.assertEqual(server._global_controller.get_available_concurrency(), 0)


if __name__ == "__main__":
    main()
