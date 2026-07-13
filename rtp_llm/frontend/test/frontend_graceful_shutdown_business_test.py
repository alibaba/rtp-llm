import asyncio
import json
import signal
import socket
import threading
import time
import unittest
import urllib.request
from types import SimpleNamespace

from fastapi.responses import StreamingResponse
from uvicorn import Config

from rtp_llm.frontend.frontend_app import FrontendApp, GracefulShutdownServer
from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager


class FakeController:
    max_concurrency = 4


class StreamingFrontendServer:
    def __init__(self):
        self._global_controller = FakeController()
        self.is_embedding = False
        self.first_chunk_sent = threading.Event()
        self.close_called = threading.Event()

    def check_health(self):
        return True

    async def inference(self, req, raw_request):
        async def gen():
            yield b"data:first\r\n\r\n"
            self.first_chunk_sent.set()
            await asyncio.sleep(0.5)
            yield b"data:second\r\n\r\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def close(self):
        self.close_called.set()


def get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class FrontendGracefulShutdownBusinessTest(unittest.TestCase):
    def build_server(self, port):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = StreamingFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = SimpleNamespace(http_port=0)
        app_owner.grpc_client = None
        app = app_owner.create_app()

        config = Config(
            app,
            host="127.0.0.1",
            port=port,
            log_config=None,
            timeout_graceful_shutdown=5,
        )
        server = GracefulShutdownServer(config)
        server.set_server(app_owner.frontend_server, app_owner.shutdown_manager)
        return server, app_owner.frontend_server

    def wait_until_ready(self, port):
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1
                ) as response:
                    if response.status == 200:
                        return
            except Exception:
                time.sleep(0.1)
        raise TimeoutError("frontend server did not become ready")

    def test_streaming_request_finishes_after_shutdown_signal(self):
        port = get_free_port()
        server, frontend_server = self.build_server(port)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        self.wait_until_ready(port)

        stream_lines = []
        stream_error = []

        def read_stream():
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/",
                    data=json.dumps({"prompt": "hello", "stream": True}).encode(
                        "utf-8"
                    ),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8").strip()
                        if line:
                            stream_lines.append(line)
            except Exception as e:
                stream_error.append(e)

        stream_thread = threading.Thread(target=read_stream)
        stream_thread.start()
        self.assertTrue(frontend_server.first_chunk_sent.wait(timeout=5))

        server.handle_exit(signal.SIGTERM, None)

        stream_thread.join(timeout=10)
        server_thread.join(timeout=10)

        self.assertFalse(stream_thread.is_alive())
        self.assertFalse(server_thread.is_alive())
        self.assertEqual(stream_error, [])
        self.assertIn("data:first", stream_lines)
        self.assertIn("data:second", stream_lines)
        self.assertTrue(frontend_server.close_called.is_set())


if __name__ == "__main__":
    unittest.main()
