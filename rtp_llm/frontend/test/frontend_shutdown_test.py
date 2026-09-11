import asyncio
import signal
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec, patch

from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from uvicorn import Config, Server

from rtp_llm.frontend.frontend_app import (
    FrontendApp,
    GracefulShutdownServer,
    _AllocatorDumpRequestGuard,
    _pre_stop_drain_seconds,
)
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager
from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper


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


class HangingFrontendServer(FakeFrontendServer):
    async def close(self):
        await asyncio.Event().wait()


class FakeGrpcClient:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or {"status": "ok"}

    async def post_request(self, endpoint, payload):
        self.calls.append((endpoint, payload))
        return self.response


class FrontendShutdownManagerTest(unittest.TestCase):
    _DUMP_TOKEN = "unit-test-dump-secret"
    _DUMP_HEADER = "X-Test-Allocator-Dump-Token"

    def setUp(self):
        self._log_path = tempfile.TemporaryDirectory()
        self._log_path_patch = patch.dict(
            "os.environ", {"LOG_PATH": self._log_path.name}
        )
        self._log_path_patch.start()

    def tearDown(self):
        self._log_path_patch.stop()
        self._log_path.cleanup()

    def wait_until(self, predicate, timeout=1.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def allocator_dump_config(self, cooldown: float = 0) -> SimpleNamespace:
        return SimpleNamespace(
            http_port=0,
            enable_torch_allocator_dump=True,
            torch_allocator_dump_auth_token=self._DUMP_TOKEN,
            torch_allocator_dump_auth_header=self._DUMP_HEADER,
            torch_allocator_dump_cooldown_seconds=cooldown,
        )

    def dump_headers(self):
        return {self._DUMP_HEADER: self._DUMP_TOKEN}

    def test_draining_rejects_new_business_and_marks_health_unavailable(self):
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
        self.assertEqual(chat_response.headers.get("retry-after"), "1")

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
        self.assertEqual(embedding_response.headers.get("retry-after"), "1")

    def test_pre_stop_unavailable_rejects_new_business(self):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = FakeFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = SimpleNamespace(http_port=0)
        app_owner.grpc_client = None

        app = app_owner.create_app()
        client = TestClient(app)
        app_owner.shutdown_manager.start_unavailable("unit test")

        self.assertEqual(client.get("/liveness").status_code, 200)
        self.assertEqual(client.get("/health").status_code, 503)
        chat_response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(chat_response.status_code, 503)
        self.assertEqual(chat_response.headers.get("retry-after"), "1")

    def test_draining_rejects_admin_backend_requests(self):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = FakeFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = SimpleNamespace(http_port=0)
        app_owner.grpc_client = FakeGrpcClient()

        app = app_owner.create_app()
        client = TestClient(app)
        app_owner.shutdown_manager.start_draining("unit test")

        requests = [
            ("get", "/v1/models", None),
            ("get", "/cache_status", None),
            ("get", "/worker_status", None),
            ("post", "/set_log_level", {"log_level": "DEBUG"}),
            ("post", "/start_profile", {}),
            ("post", "/dump_torch_allocator", {}),
            ("post", "/update_eplb_config", {"mode": "NONE"}),
            ("post", "/update_scheduler_info", {}),
            ("post", "/tokenizer/encode", {"prompt": "hello"}),
            ("post", "/tokenize", {"prompt": "hello"}),
        ]
        for method, path, payload in requests:
            if method == "get":
                response = client.get(path)
            else:
                response = client.post(path, json=payload)
            self.assertEqual(response.status_code, 503, path)
            self.assertEqual(response.headers.get("retry-after"), "1", path)

        self.assertEqual(app_owner.grpc_client.calls, [])

    def test_pre_stop_unavailable_rejects_admin_backend_requests(self):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = FakeFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = SimpleNamespace(http_port=0)
        app_owner.grpc_client = FakeGrpcClient()

        app = app_owner.create_app()
        client = TestClient(app)
        app_owner.shutdown_manager.start_unavailable("unit test")

        requests = [
            ("get", "/v1/models", None),
            ("get", "/cache_status", None),
            ("get", "/worker_status", None),
            ("post", "/set_log_level", {"log_level": "DEBUG"}),
            ("post", "/start_profile", {}),
            ("post", "/dump_torch_allocator", {}),
            ("post", "/update_eplb_config", {"mode": "NONE"}),
            ("post", "/update_scheduler_info", {}),
            ("post", "/tokenizer/encode", {"prompt": "hello"}),
            ("post", "/tokenize", {"prompt": "hello"}),
        ]
        for method, path, payload in requests:
            if method == "get":
                response = client.get(path)
            else:
                response = client.post(path, json=payload)
            self.assertEqual(response.status_code, 503, path)
            self.assertEqual(response.headers.get("retry-after"), "1", path)

        self.assertEqual(app_owner.grpc_client.calls, [])

    def test_dump_torch_allocator_is_disabled_by_default(self):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = FakeFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = SimpleNamespace(http_port=0)
        app_owner.grpc_client = FakeGrpcClient()

        response = TestClient(app_owner.create_app()).post(
            "/dump_torch_allocator", json={}
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(app_owner.grpc_client.calls, [])

    def test_dump_torch_allocator_rejects_missing_or_wrong_auth_without_rpc(self):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = FakeFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = self.allocator_dump_config()
        app_owner.grpc_client = FakeGrpcClient()
        client = TestClient(app_owner.create_app())

        missing = client.post("/dump_torch_allocator")
        wrong = client.post(
            "/dump_torch_allocator",
            headers={self._DUMP_HEADER: "wrong-secret"},
        )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(app_owner.grpc_client.calls, [])
        self.assertNotIn(self._DUMP_TOKEN, missing.text + wrong.text)

    def test_dump_torch_allocator_rejects_enabled_config_without_secret(self):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = FakeFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = SimpleNamespace(
            http_port=0,
            enable_torch_allocator_dump=True,
            torch_allocator_dump_auth_token="",
            torch_allocator_dump_cooldown_seconds=0,
        )
        app_owner.grpc_client = FakeGrpcClient()

        with self.assertRaisesRegex(ValueError, "authentication token"):
            app_owner.create_app()

    def test_dump_torch_allocator_forwards_when_enabled_without_leaking_details(self):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = FakeFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = self.allocator_dump_config()
        app_owner.grpc_client = FakeGrpcClient(
            {
                "status": "ok",
                "backends": [
                    {
                        "pid": 123,
                        "dp_address": "10.0.0.1:8089",
                        "file_path": "/private/allocator.log",
                    }
                ],
            }
        )

        response = TestClient(app_owner.create_app()).post(
            "/dump_torch_allocator", json={}, headers=self.dump_headers()
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"status", "dump_id"})
        self.assertEqual(response.json()["status"], "ok")
        dump_id = response.json()["dump_id"]
        self.assertEqual(len(dump_id), 24)
        self.assertEqual(
            app_owner.grpc_client.calls,
            [
                (
                    "dump_torch_allocator",
                    {"auth_token": self._DUMP_TOKEN, "dump_id": dump_id},
                )
            ],
        )

    def test_dump_torch_allocator_returns_sanitized_backend_failure(self):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = FakeFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = self.allocator_dump_config()
        app_owner.grpc_client = FakeGrpcClient(
            {
                "status": "error",
                "backends": [{"pid": 123, "file_path": "/private/allocator.log"}],
                "errors": ["10.0.0.1:8089: raw backend failure"],
            }
        )

        response = TestClient(app_owner.create_app()).post(
            "/dump_torch_allocator", json={}, headers=self.dump_headers()
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(set(response.json()), {"status", "dump_id", "error"})
        self.assertEqual(
            response.json()["error"], "allocator dump failed; see server logs"
        )
        self.assertNotIn("10.0.0.1", response.text)
        self.assertNotIn("/private", response.text)
        self.assertNotIn("raw backend failure", response.text)

    def test_dump_torch_allocator_endpoint_enforces_cooldown(self):
        app_owner = FrontendApp.__new__(FrontendApp)
        app_owner.frontend_server = FakeFrontendServer()
        app_owner.shutdown_manager = FrontendShutdownManager()
        app_owner.separated_frontend = True
        app_owner.server_config = self.allocator_dump_config(cooldown=60)
        app_owner.grpc_client = FakeGrpcClient()
        client = TestClient(app_owner.create_app())

        self.assertEqual(
            client.post(
                "/dump_torch_allocator", headers=self.dump_headers()
            ).status_code,
            200,
        )
        response = client.post(
            "/rtp_llm/dump_torch_allocator", headers=self.dump_headers()
        )

        self.assertEqual(response.status_code, 429)
        self.assertGreaterEqual(int(response.headers["retry-after"]), 1)
        self.assertEqual(len(app_owner.grpc_client.calls), 1)

    def test_allocator_dump_guard_shares_single_flight_and_cooldown(self):
        now = [100.0]
        runtime_dir = Path(self._log_path.name)
        first_worker = _AllocatorDumpRequestGuard(
            10.0, runtime_dir=runtime_dir, clock=lambda: now[0]
        )
        second_worker = _AllocatorDumpRequestGuard(
            10.0, runtime_dir=runtime_dir, clock=lambda: now[0]
        )

        self.assertEqual(first_worker.try_begin(), ("ok", 0.0))
        self.assertEqual(second_worker.try_begin(), ("in_flight", 0.0))
        first_worker.finish()
        now[0] = 105.0
        self.assertEqual(second_worker.try_begin(), ("cooldown", 5.0))
        now[0] = 110.0
        self.assertEqual(second_worker.try_begin(), ("ok", 0.0))
        second_worker.finish()

    def test_allocator_dump_guard_rejects_invalid_cooldowns(self):
        runtime_dir = Path(self._log_path.name)
        for cooldown in (-1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(cooldown=cooldown), self.assertRaises(ValueError):
                _AllocatorDumpRequestGuard(cooldown, runtime_dir=runtime_dir)

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
        server.set_server(
            FakeFrontendServer(), manager, pre_stop_drain_seconds=0
        )

        server.handle_exit(signal.SIGTERM, None)
        self.assertTrue(server.wait_for_signal_dispatch())

        self.assertTrue(manager.is_draining())
        self.assertTrue(server.should_exit)

    def test_uvicorn_shutdown_closes_production_frontend_and_grpc_contracts(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        frontend_server = create_autospec(
            FrontendServer, instance=True, spec_set=True
        )
        grpc_client = create_autospec(
            GrpcClientWrapper, instance=True, spec_set=True
        )
        server.set_server(frontend_server, manager, grpc_client)

        with patch.object(Server, "shutdown", new_callable=AsyncMock):
            asyncio.run(server.shutdown())

        frontend_server.close.assert_awaited_once_with()
        grpc_client.close.assert_awaited_once_with()

    def test_frontend_close_failure_does_not_block_grpc_cleanup(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        frontend_server = create_autospec(
            FrontendServer, instance=True, spec_set=True
        )
        frontend_server.close.side_effect = RuntimeError("frontend close failed")
        grpc_client = create_autospec(
            GrpcClientWrapper, instance=True, spec_set=True
        )
        server.set_server(frontend_server, manager, grpc_client)

        with patch.object(Server, "shutdown", new_callable=AsyncMock):
            with self.assertLogs(level="WARNING") as logs:
                asyncio.run(server.shutdown())

        frontend_server.close.assert_awaited_once_with()
        grpc_client.close.assert_awaited_once_with()
        self.assertIn("Failed to close frontend server", "\n".join(logs.output))

    def test_pre_stop_signal_marks_unavailable_without_uvicorn_shutdown(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(FakeFrontendServer(), manager)

        server.handle_pre_stop_drain_signal(signal.SIGUSR1, None)
        self.assertTrue(server.wait_for_signal_dispatch())

        self.assertTrue(manager.is_unavailable())
        self.assertFalse(manager.is_draining())
        self.assertFalse(server.should_exit)
        self.assertIsNotNone(server._pre_stop_timer)
        server._pre_stop_timer.cancel()
        server._pre_stop_timer = None

    def test_pre_stop_signal_watchdog_starts_uvicorn_shutdown(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(FakeFrontendServer(), manager, pre_stop_drain_seconds=0.01)

        server.handle_pre_stop_drain_signal(signal.SIGUSR1, None)
        self.assertTrue(server.wait_for_signal_dispatch())
        self.assertTrue(manager.is_unavailable())
        self.assertFalse(manager.is_draining())
        self.assertFalse(server.should_exit)
        self.assertTrue(self.wait_until(lambda: server.should_exit))

        self.assertTrue(manager.is_unavailable())
        self.assertTrue(manager.is_draining())
        self.assertTrue(server.should_exit)
        self.assertIsNone(server._pre_stop_timer)

    def test_pre_stop_timer_uses_remaining_drain_seconds(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(FakeFrontendServer(), manager, pre_stop_drain_seconds=10)

        with patch.object(manager, "drain_elapsed_seconds", return_value=7.0):
            server.handle_pre_stop_drain_signal(signal.SIGUSR1, None)
            server.handle_exit(signal.SIGTERM, None)
            self.assertTrue(server.wait_for_signal_dispatch())

        self.assertTrue(manager.is_unavailable())
        self.assertFalse(manager.is_draining())
        self.assertFalse(server.should_exit)
        self.assertIsNotNone(server._pre_stop_timer)
        self.assertAlmostEqual(server._pre_stop_timer.interval, 3.0)
        self.assertFalse(manager.try_begin_request())
        server._pre_stop_timer.cancel()
        server._pre_stop_timer = None

    def test_cancelled_watchdog_callback_does_not_force_exit(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(FakeFrontendServer(), manager, pre_stop_drain_seconds=100)

        server.handle_pre_stop_drain_signal(signal.SIGUSR1, None)
        server.handle_exit(signal.SIGINT, None)
        self.assertTrue(server.wait_for_signal_dispatch())
        server._begin_shutdown(signal.SIGUSR1, None, "SIGUSR1", False)

        self.assertTrue(server.should_exit)
        self.assertFalse(server.force_exit)
        self.assertIsNone(server._pre_stop_timer)

    def test_sigterm_after_pre_stop_signal_keeps_existing_drain_timer(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(
            FakeFrontendServer(), manager, pre_stop_drain_seconds=10
        )
        server.handle_pre_stop_drain_signal(signal.SIGUSR1, None)
        self.assertTrue(server.wait_for_signal_dispatch())
        pre_stop_timer = server._pre_stop_timer
        self.assertIsNotNone(pre_stop_timer)

        with patch.object(manager, "drain_elapsed_seconds", return_value=7.0):
            server.handle_exit(signal.SIGTERM, None)
            self.assertTrue(server.wait_for_signal_dispatch())

        self.assertTrue(manager.is_unavailable())
        self.assertFalse(manager.is_draining())
        self.assertFalse(server.should_exit)
        self.assertIs(server._pre_stop_timer, pre_stop_timer)
        self.assertFalse(manager.try_begin_request())
        server._pre_stop_timer.cancel()
        server._pre_stop_timer = None

    def test_sigterm_after_elapsed_pre_stop_signal_starts_shutdown(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(
            FakeFrontendServer(), manager, pre_stop_drain_seconds=10
        )
        server.handle_pre_stop_drain_signal(signal.SIGUSR1, None)
        self.assertTrue(server.wait_for_signal_dispatch())

        with patch.object(manager, "drain_elapsed_seconds", return_value=10.0):
            server.handle_exit(signal.SIGTERM, None)
            self.assertTrue(server.wait_for_signal_dispatch())

        self.assertTrue(manager.is_unavailable())
        self.assertTrue(manager.is_draining())
        self.assertTrue(server.should_exit)
        self.assertIsNone(server._pre_stop_timer)

    def test_sigterm_waits_for_pre_stop_drain_before_uvicorn_shutdown(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(
            FakeFrontendServer(), manager, pre_stop_drain_seconds=0.01
        )

        server.handle_exit(signal.SIGTERM, None)
        self.assertTrue(server.wait_for_signal_dispatch())
        self.assertTrue(manager.is_unavailable())
        self.assertFalse(manager.is_draining())
        self.assertFalse(manager.try_begin_request())
        self.assertFalse(server.should_exit)
        self.assertTrue(self.wait_until(lambda: server.should_exit))

        self.assertTrue(manager.is_draining())
        self.assertTrue(server.should_exit)

    def test_duplicate_sigterm_keeps_pre_stop_drain(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(
            FakeFrontendServer(), manager, pre_stop_drain_seconds=100
        )

        server.handle_exit(signal.SIGTERM, None)
        self.assertTrue(server.wait_for_signal_dispatch())
        self.assertTrue(manager.is_unavailable())
        self.assertFalse(manager.is_draining())
        self.assertFalse(server.should_exit)
        server.handle_exit(signal.SIGTERM, None)
        self.assertTrue(server.wait_for_signal_dispatch())

        self.assertFalse(manager.is_draining())
        self.assertFalse(server.should_exit)
        self.assertIsNotNone(server._pre_stop_timer)
        self.assertFalse(manager.try_begin_request())
        server._pre_stop_timer.cancel()
        server._pre_stop_timer = None

    def test_sigterm_after_timer_fires_does_not_rearm_pre_stop_drain(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(Config(lambda scope: None))
        server.set_server(
            FakeFrontendServer(), manager, pre_stop_drain_seconds=0.01
        )

        server.handle_exit(signal.SIGTERM, None)
        self.assertTrue(server.wait_for_signal_dispatch())
        self.assertTrue(
            self.wait_until(lambda: manager.is_draining() and server.should_exit)
        )
        server.handle_exit(signal.SIGTERM, None)
        self.assertTrue(server.wait_for_signal_dispatch())

        self.assertTrue(manager.is_draining())
        self.assertTrue(server.should_exit)
        self.assertIsNone(server._pre_stop_timer)

    def test_frontend_pre_stop_uses_configured_value(self):
        self.assertEqual(_pre_stop_drain_seconds(2.5), 2.5)

    def test_frontend_pre_stop_clamps_negative_config_to_zero(self):
        self.assertEqual(_pre_stop_drain_seconds(-1), 0.0)

    def test_frontend_pre_stop_clamps_to_shutdown_timeout(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(
            Config(lambda scope: None, timeout_graceful_shutdown=10)
        )
        server.set_server(
            FakeFrontendServer(), manager, pre_stop_drain_seconds=30
        )

        self.assertEqual(server._effective_pre_stop_drain_seconds(), 9.0)

    def test_frontend_pre_stop_reserves_shutdown_headroom(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(
            Config(lambda scope: None, timeout_graceful_shutdown=600)
        )
        server.set_server(
            FakeFrontendServer(), manager, pre_stop_drain_seconds=600
        )

        self.assertEqual(server._effective_pre_stop_drain_seconds(), 540.0)

    def test_frontend_graceful_timeout_uses_remaining_pre_stop_budget(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(
            Config(lambda scope: None, timeout_graceful_shutdown=10)
        )
        server.set_server(FakeFrontendServer(), manager)
        manager.start_draining("unit test")

        with patch.object(manager, "drain_elapsed_seconds", return_value=7.0):
            server._limit_graceful_shutdown_to_remaining_budget()
            self.assertEqual(server._remaining_shutdown_timeout_after_pre_stop(), 3.0)

        self.assertEqual(server.config.timeout_graceful_shutdown, 3.0)

    def test_frontend_close_is_bounded_by_remaining_shutdown_budget(self):
        manager = FrontendShutdownManager()
        server = GracefulShutdownServer(
            Config(lambda scope: None, timeout_graceful_shutdown=0.01)
        )
        frontend_server = HangingFrontendServer()
        server.set_server(frontend_server, manager)
        manager.start_draining("unit test")

        start = time.monotonic()
        asyncio.run(
            server._close_with_remaining_shutdown_budget(
                "frontend server", frontend_server.close
            )
        )

        self.assertLess(time.monotonic() - start, 0.5)


if __name__ == "__main__":
    unittest.main()
