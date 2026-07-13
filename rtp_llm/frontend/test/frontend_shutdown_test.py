import asyncio
import signal
import unittest
from types import SimpleNamespace

from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from uvicorn import Config

from rtp_llm.frontend.frontend_app import FrontendApp, GracefulShutdownServer
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

        server.handle_exit(signal.SIGTERM, None)

        self.assertTrue(manager.is_draining())
        self.assertTrue(server.should_exit)


if __name__ == "__main__":
    unittest.main()
