import asyncio
import unittest
from contextlib import nullcontext

from uvicorn import Config, Server

from rtp_llm.frontend.frontend_socket import create_frontend_socket


class StartupApp:
    def __init__(self, fail=False):
        self.entered = asyncio.Event()
        self.ready = asyncio.Event()
        self.fail = fail

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await receive()
            self.entered.set()
            await self.ready.wait()
            if self.fail:
                await send({"type": "lifespan.startup.failed", "message": "not ready"})
                return
            await send({"type": "lifespan.startup.complete"})
            await receive()
            await send({"type": "lifespan.shutdown.complete"})
        else:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ready"})


class FrontendSocketTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.workers = []

    async def asyncTearDown(self):
        for sock, app, server, task in self.workers:
            app.ready.set()
            if not task.done():
                await self.make_ready(app, server)
            server.should_exit = True
            try:
                await asyncio.wait_for(task, 5)
            finally:
                sock.close()

    async def start_worker(self, port=0, fail=False):
        sock = create_frontend_socket("127.0.0.1", port)
        app = StartupApp(fail=fail)
        server = Server(
            Config(
                app,
                fd=sock.fileno(),
                loop="asyncio",
                lifespan="on",
                log_config=None,
                access_log=False,
            )
        )
        # This test has no process-level signal ownership.
        server.capture_signals = nullcontext
        task = asyncio.create_task(server.serve())
        self.workers.append((sock, app, server, task))
        await asyncio.wait_for(app.entered.wait(), 5)
        return sock, app, server, task

    async def make_ready(self, app, server):
        app.ready.set()

        async def wait_started():
            while not server.started:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_started(), 5)

    async def request(self, port):
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port), 1
        )
        try:
            writer.write(
                b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), 1)
            self.assertIn(b"200 OK", response)
            self.assertIn(b"ready", response)
        finally:
            writer.close()
            await writer.wait_closed()

    async def assert_not_listening(self, port):
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port), 1
            )
        except ConnectionRefusedError:
            return
        writer.close()
        await writer.wait_closed()
        self.fail("frontend accepted a connection before startup completed")

    async def test_no_listener_until_startup_completes(self):
        sock, app, server, _ = await self.start_worker()
        port = sock.getsockname()[1]
        await self.assert_not_listening(port)
        await self.make_ready(app, server)
        await self.request(port)

    async def test_unready_worker_does_not_capture_ready_workers_requests(self):
        ready_sock, ready_app, ready_server, _ = await self.start_worker()
        await self.make_ready(ready_app, ready_server)
        port = ready_sock.getsockname()[1]
        _, pending_app, pending_server, _ = await self.start_worker(port)
        await asyncio.gather(*(self.request(port) for _ in range(32)))
        await self.make_ready(pending_app, pending_server)
        await asyncio.gather(*(self.request(port) for _ in range(32)))

    async def test_failed_startup_never_listens(self):
        sock, app, _, task = await self.start_worker(fail=True)
        app.ready.set()
        await asyncio.wait_for(task, 5)
        await self.assert_not_listening(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
