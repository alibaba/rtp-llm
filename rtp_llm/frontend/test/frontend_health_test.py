import asyncio
import os
import socket
import struct
import subprocess
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import uvicorn

from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.server.backend_app import BackendApp
from rtp_llm.server.backend_server import BackendServer
from rtp_llm.start_server import check_server_health


class FrontendHealthTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"CONSTRAINT_TREE_REQUIRED": "false"})
        env.start()
        self.addCleanup(env.stop)

    def test_backend_starts_bootstrap_after_native_engine_before_returning(self):
        from rtp_llm.config.task_type import TaskType
        from rtp_llm.ops import RoleType
        from rtp_llm.server.constraint_tree_bootstrap import ConstraintTreeBootstrap

        backend = object.__new__(BackendServer)
        backend._gang_server = Mock()
        engine = Mock(task_type=TaskType.LANGUAGE_MODEL)
        engine.config.role_type = RoleType.PDFUSION
        config = SimpleNamespace(
            profiling_debug_config=SimpleNamespace(debug_start_fake_process=0)
        )
        events = []

        def create_engine(*args):
            events.append("native_started")
            return engine

        registration = Mock()
        registration.start.side_effect = lambda: events.append("bootstrap_started")
        with patch(
            "rtp_llm.server.backend_server.ModelFactory.create_from_env",
            side_effect=create_engine,
        ), patch("rtp_llm.server.backend_server.BackendRPCServerVisitor"), patch(
            "rtp_llm.server.backend_server.LoraManager"
        ), patch(
            "rtp_llm.server.backend_server.WeightManager"
        ), patch.object(
            ConstraintTreeBootstrap, "from_env", return_value=registration
        ) as factory:
            backend.start(config)
            events.append("backend_start_returned")
            self.assertEqual(
                ["native_started", "bootstrap_started", "backend_start_returned"],
                events,
            )
            self.assertEqual(RoleType.PDFUSION, factory.call_args.args[2])
            backend.stop()
            registration.stop.assert_called_once()

        with patch.dict(
            os.environ,
            {
                "CONSTRAINT_TREE_REQUIRED": "on",
                "MODEL_SERVICE_CONFIG": '{"service_id":"aigc.text-generation.generation.engine_service", "master_endpoint":{"address":"master.vip"}}',
            },
        ):
            bootstrap = ConstraintTreeBootstrap.from_env(
                Mock(), 23495, RoleType.PDFUSION
            )
            self.assertEqual("PDFUSION", bootstrap.body["role"])

    @staticmethod
    def frontend_app(required=False):
        frontend = object.__new__(FrontendApp)
        frontend.separated_frontend = False
        frontend.frontend_server = Mock()
        frontend.frontend_server._global_controller = SimpleNamespace(max_concurrency=2)
        with patch.dict(os.environ, {"CONSTRAINT_TREE_REQUIRED": str(required)}):
            return frontend.create_app()

    def test_launcher_can_check_liveness_without_claiming_readiness(self):
        with patch("rtp_llm.start_server.requests.get") as get:
            get.return_value = SimpleNamespace(status_code=503, text='"ok"')
            self.assertFalse(check_server_health(18005))
            get.return_value = SimpleNamespace(status_code=200, text='"ok"')
            self.assertTrue(check_server_health(18005, "live"))
            get.assert_called_with("http://localhost:18005/live", timeout=60)
            get.return_value = SimpleNamespace(
                status_code=200, text='{"error":"unavailable"}'
            )
            self.assertFalse(check_server_health(18005))

    async def test_unready_backend_is_not_wrapped_in_http_200(self):
        app = self.frontend_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            with patch(
                "rtp_llm.frontend.frontend_app.async_request_server",
                new_callable=AsyncMock,
            ) as backend:
                for result in (
                    {"error": "HTTP Error 503"},
                    {"error": "Connection failed"},
                    None,
                ):
                    backend.return_value = result
                    for path in (
                        "/health",
                        "/health_check",
                        "/GraphService/cm2_status",
                        "/",
                    ):
                        response = await (
                            client.post(path)
                            if path == "/health_check"
                            else client.get(path)
                        )
                        self.assertEqual(503, response.status_code)
                backend.side_effect = asyncio.TimeoutError
                self.assertEqual(503, (await client.get("/health")).status_code)
                backend.side_effect = None
                backend.return_value = "ok"
                self.assertEqual(200, (await client.get("/health")).status_code)
                self.assertEqual(200, (await client.get("/live")).status_code)
                backend.assert_awaited_with("get", unittest.mock.ANY, "live", {})
                backend.return_value = {"status": "home"}
                self.assertEqual(200, (await client.get("/")).status_code)

    async def test_required_tree_checks_correct_rank_and_keeps_liveness(self):
        # A nonzero rank's FE port is already adjusted by FrontendApp.__init__.
        frontend_port = WorkerInfo.server_port_offset(2, 18000)
        engine_port = WorkerInfo.http_port_offset(0, frontend_port)
        backend_port = WorkerInfo.backend_server_port_offset(0, frontend_port)
        ports = SimpleNamespace(
            server_port=frontend_port, backend_server_port=backend_port
        )
        engine_result = "ok"
        python_result = "ok"

        async def response(method, port, path, body):
            if port == engine_port:
                self.assertEqual((method, path), ("get", "health"))
                if isinstance(engine_result, Exception):
                    raise engine_result
                return engine_result
            self.assertEqual(port, backend_port)
            return {"status": "home"} if path == "" else python_result

        with patch("rtp_llm.frontend.frontend_app.g_worker_info", ports), patch(
            "rtp_llm.frontend.frontend_app.async_request_server", side_effect=response
        ) as requests:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.frontend_app(True)),
                base_url="http://test",
            ) as client:
                for engine_result in (
                    {"error": "HTTP Error 503"},
                    {"error": "Connection failed"},
                    asyncio.TimeoutError(),
                    None,
                    "ok",
                ):
                    expected = 200 if engine_result == "ok" else 503
                    for path in (
                        "/health",
                        "/status",
                        "/GraphService/cm2_status",
                        "/SearchService/cm2_status",
                        "/",
                    ):
                        self.assertEqual(expected, (await client.get(path)).status_code)
                    self.assertEqual(
                        expected, (await client.post("/health_check")).status_code
                    )
                    requests.reset_mock()
                    self.assertEqual(200, (await client.get("/live")).status_code)
                    requests.assert_awaited_once_with("get", backend_port, "live", {})
                python_result = {"error": "Connection failed"}
                self.assertEqual(503, (await client.get("/health")).status_code)

    @unittest.skipUnless(
        os.environ.get("CONSTRAINT_TREE_CPP_WORKER_BINARY"),
        "requires native CSR HTTP fixture",
    )
    async def test_real_http_frontend_backend_and_native_tree_lifecycle(self):
        """Real TCP HTTP, production Python apps + native CSR service, no mocked health replies."""
        binary = os.environ["CONSTRAINT_TREE_CPP_WORKER_BINARY"]
        self.assertTrue(os.access(binary, os.X_OK), binary)
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            engine_port = reserve.getsockname()[1]

        @asynccontextmanager
        async def serve(app):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sock.listen(128)
            port = sock.getsockname()[1]
            server = uvicorn.Server(
                uvicorn.Config(app, log_config=None, access_log=False, lifespan="off")
            )
            task = asyncio.create_task(server.serve(sockets=[sock]))
            try:
                for _ in range(200):
                    if server.started:
                        break
                    if task.done():
                        await task
                        self.fail("HTTP server exited before startup")
                    await asyncio.sleep(0.01)
                self.assertTrue(server.started)
                yield port
            finally:
                server.should_exit = True
                await asyncio.wait_for(task, 10)
                sock.close()

        @asynccontextmanager
        async def native():
            process = subprocess.Popen(
                [binary, str(engine_port)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            try:
                async with httpx.AsyncClient() as client:
                    for _ in range(200):
                        self.assertIsNone(process.poll(), "native fixture exited")
                        try:
                            if (
                                await client.get(
                                    f"http://127.0.0.1:{engine_port}/live", timeout=1
                                )
                            ).status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(0.05)
                    else:
                        self.fail("native fixture startup timed out")
                yield
            finally:
                if process.stdin:
                    process.stdin.close()
                try:
                    await asyncio.to_thread(process.wait, timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait, timeout=10)

        def artifact(version, token=17):
            # One SID: start=1 -> token -> EOS=99, RTPCSR01 format.
            return struct.pack(
                "<8sIIQiiiiQ7i",
                b"RTPCSR01",
                1,
                48,
                version,
                1,
                99,
                2,
                2,
                1,
                0,
                1,
                2,
                token,
                99,
                1,
                -1,
            )

        backend = object.__new__(BackendApp)
        backend.backend_server = Mock()
        backend.backend_server._global_controller = SimpleNamespace(max_concurrency=2)
        async with serve(backend.create_app(SimpleNamespace())) as backend_port:
            ports = SimpleNamespace(
                server_port=engine_port - 5, backend_server_port=backend_port
            )
            with patch("rtp_llm.frontend.frontend_app.g_worker_info", ports):
                async with serve(self.frontend_app(True)) as frontend_port:
                    async with httpx.AsyncClient(
                        base_url=f"http://127.0.0.1:{frontend_port}"
                    ) as front, httpx.AsyncClient(
                        base_url=f"http://127.0.0.1:{engine_port}"
                    ) as engine:
                        self.assertEqual(503, (await front.get("/health")).status_code)
                        self.assertEqual(200, (await front.get("/live")).status_code)
                        for restart in range(2):
                            async with native():
                                self.assertEqual(
                                    200,
                                    (
                                        await front.get(
                                            f"http://127.0.0.1:{backend_port}/health"
                                        )
                                    ).status_code,
                                )
                                self.assertEqual(
                                    503, (await front.get("/health")).status_code
                                )
                                self.assertEqual(
                                    200, (await front.get("/live")).status_code
                                )
                                for version, token, state in (
                                    (1, 17, "ready"),
                                    (2, -5, "failed"),
                                    (3, 19, "ready"),
                                ):
                                    self.assertEqual(
                                        200,
                                        (
                                            await engine.post(
                                                "/update_constraint_tree",
                                                content=artifact(version, token),
                                            )
                                        ).status_code,
                                    )
                                    for _ in range(200):
                                        current = (
                                            await engine.get("/constraint_tree_status")
                                        ).json()
                                        if version > 1:
                                            self.assertEqual(
                                                200,
                                                (
                                                    await front.get("/health")
                                                ).status_code,
                                            )
                                        if current["status"] == state:
                                            break
                                        await asyncio.sleep(0.01)
                                    self.assertEqual(state, current["status"])
                                    self.assertEqual(
                                        1 if state == "failed" else version,
                                        current["version"],
                                    )
                                    self.assertEqual(
                                        200, (await front.get("/health")).status_code
                                    )
                                    self.assertEqual(
                                        200, (await front.get("/")).status_code
                                    )
                            self.assertEqual(
                                503, (await front.get("/health")).status_code
                            )
                            self.assertEqual(
                                200, (await front.get("/live")).status_code
                            )


if __name__ == "__main__":
    unittest.main()
