import asyncio
import os
import signal
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from uvicorn import Config

from rtp_llm.frontend.frontend_app import (
    FrontendApp,
    GracefulShutdownServer,
    _pre_stop_drain_seconds,
)
from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager


class FakeController:
    max_concurrency = 4


class FakeFrontendServer:
    def __init__(self, is_embedding=False):
        self._global_controller = FakeController()
        self.is_embedding = is_embedding
        self.close_called = False

    def check_health(self):
        return True

    async def inference(self, req, raw_request):
        async def gen():
            yield b"data:first\r\n\r\n"
            await asyncio.sleep(0)
            yield b"data:second\r\n\r\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def chat_completion(self, request, raw_request):
        return ORJSONResponse({"ok": True})

    async def chat_render(self, request, raw_request):
        return ORJSONResponse({"ok": True})

    async def embedding(self, request, raw_request):
        return ORJSONResponse({"ok": True})

    async def close(self):
        self.close_called = True


class FrontendShutdownManagerTest(unittest.TestCase):
    def wait_until(self, predicate, timeout=1.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def test_draining_rejects_new_requests_but_keeps_liveness(self):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = FakeFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = SimpleNamespace(http_port=0)
        app_owner.grpc_client = None

        app = app_owner.create_app()
        client = TestClient(app)
        self.assertEqual(client.get("/health").status_code, 200)

        app_owner.shutdown_manager.start_draining("unit test")

        self.assertEqual(client.get("/liveness").status_code, 200)
        self.assertEqual(client.get("/health").status_code, 503)
        response = client.post("/", json={"prompt": "hello"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers.get("retry-after"), "1")
        chat_response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(chat_response.status_code, 503)

        embedding_app_owner = FrontendApp.__new__(FrontendApp)
        embedding_app_owner.frontend_server = FakeFrontendServer(is_embedding=True)
        embedding_app_owner.shutdown_manager = FrontendShutdownManager()
        embedding_app_owner.separated_frontend = True
        embedding_app_owner.server_config = SimpleNamespace(http_port=0)
        embedding_app_owner.grpc_client = None
        embedding_app = embedding_app_owner.create_app()
        embedding_client = TestClient(embedding_app)
        embedding_app_owner.shutdown_manager.start_draining("unit test")
        embedding_response = embedding_client.post("/v1/embeddings", json={})
        self.assertEqual(embedding_response.status_code, 503)

    def test_streaming_request_is_counted_until_body_iterator_finishes(self):
        manager = FrontendShutdownManager()
        self.assertTrue(manager.try_begin_request())
        self.assertEqual(manager.active_request_count(), 1)

        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.shutdown_manager = manager

        async def gen():
            yield b"one"
            self.assertEqual(manager.active_request_count(), 1)
            yield b"two"

        async def consume():
            chunks = []
            async for chunk in app_owner._track_streaming_response(gen()):
                chunks.append(chunk)
            return chunks

        self.assertEqual(asyncio.run(consume()), [b"one", b"two"])
        self.assertEqual(manager.active_request_count(), 0)

    def test_uvicorn_signal_marks_frontend_draining(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(FakeFrontendServer(), manager)

        with patch.dict(os.environ, {"FRONTEND_PRE_STOP_DRAIN_SECONDS": "0"}):
            server.handle_exit(signal.SIGTERM, None)

        self.assertTrue(manager.is_draining())
        self.assertTrue(server.should_exit)

    def test_sigterm_waits_for_pre_stop_drain_before_uvicorn_shutdown(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(FakeFrontendServer(), manager)

        with patch.dict(os.environ, {"FRONTEND_PRE_STOP_DRAIN_SECONDS": "0.01"}):
            server.handle_exit(signal.SIGTERM, None)
            self.assertFalse(manager.is_draining())
            self.assertFalse(server.should_exit)
            self.assertTrue(
                self.wait_until(lambda: manager.is_draining() and server.should_exit)
            )

        self.assertTrue(manager.is_draining())
        self.assertTrue(server.should_exit)

    def test_second_sigterm_skips_pre_stop_drain(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(FakeFrontendServer(), manager)

        with patch.dict(os.environ, {"FRONTEND_PRE_STOP_DRAIN_SECONDS": "100"}):
            server.handle_exit(signal.SIGTERM, None)
            self.assertFalse(manager.is_draining())
            self.assertFalse(server.should_exit)
            server.handle_exit(signal.SIGTERM, None)

        self.assertTrue(manager.is_draining())
        self.assertTrue(server.should_exit)

    def test_sigterm_after_timer_fires_does_not_rearm_pre_stop_drain(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(FakeFrontendServer(), manager)

        with patch.dict(os.environ, {"FRONTEND_PRE_STOP_DRAIN_SECONDS": "0.01"}):
            server.handle_exit(signal.SIGTERM, None)
            self.assertTrue(
                self.wait_until(lambda: manager.is_draining() and server.should_exit)
            )
            server.handle_exit(signal.SIGTERM, None)

        self.assertTrue(manager.is_draining())
        self.assertTrue(server.should_exit)
        self.assertIsNone(server._pre_stop_timer)

    def test_frontend_pre_stop_uses_frontend_env_before_shared_env(self):
        with patch.dict(
            os.environ,
            {
                "FRONTEND_PRE_STOP_DRAIN_SECONDS": "2.5",
                "DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "9",
            },
        ):
            self.assertEqual(_pre_stop_drain_seconds(), 2.5)

    def test_frontend_pre_stop_falls_back_to_shared_env(self):
        with patch.dict(
            os.environ,
            {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "9"},
            clear=True,
        ):
            self.assertEqual(_pre_stop_drain_seconds(), 9.0)

    def test_frontend_pre_stop_clamps_to_shutdown_timeout(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(
            Config(lambda scope: None, timeout_graceful_shutdown=10)
        )
        server.set_server(FakeFrontendServer(), manager)

        with patch.dict(os.environ, {"FRONTEND_PRE_STOP_DRAIN_SECONDS": "30"}):
            self.assertEqual(server._effective_pre_stop_drain_seconds(), 10.0)


if __name__ == "__main__":
    unittest.main()
