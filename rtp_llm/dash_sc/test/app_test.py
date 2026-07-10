"""Unit tests for ``rtp_llm.dash_sc.app`` helpers (echo_prefix startup derivation)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import signal
import threading
import time
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import grpc

from rtp_llm.dash_sc import app as bg_app
from rtp_llm.dash_sc.app import (
    DashScShutdownManager,
    _create_proxy_servicer_on_loop,
    _derive_echo_prefix_ids,
    _is_proxy_mode_enabled,
    _pre_stop_drain_seconds,
)
from rtp_llm.dash_sc.server import DashScGrpcDrainAioInterceptor, DashScGrpcServer


class _EnvCfg:
    def __init__(self, think_mode: int = 1, think_start_tag: str = "<think>\n"):
        self.think_mode = think_mode
        self.think_start_tag = think_start_tag


class _FakeTokenizer:
    def __init__(self, *, ids=None, id_map=None, raise_exc: bool = False):
        self._ids = ids or []
        self._id_map = id_map or {}
        self._raise = raise_exc
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text, add_special_tokens=True):
        if self._raise:
            raise RuntimeError("tokenizer.encode failed")
        self.encode_calls.append((text, add_special_tokens))
        if text in self._id_map:
            return list(self._id_map[text])
        return list(self._ids)


class _BaseTok:
    def __init__(self, tok):
        self.tokenizer = tok


class DeriveEchoPrefixIdsTest(TestCase):
    def test_encodes_think_start_tag(self) -> None:
        tok = _FakeTokenizer(ids=[154841])
        ids = _derive_echo_prefix_ids(_EnvCfg(), _BaseTok(tok))
        self.assertEqual(ids, [154841])
        # Must encode without special tokens so only the tag bytes become ids.
        self.assertEqual(tok.encode_calls, [("<think>\n", False)])

    def test_disabled_when_think_mode_off(self) -> None:
        tok = _FakeTokenizer(ids=[154841])
        ids = _derive_echo_prefix_ids(_EnvCfg(think_mode=0), _BaseTok(tok))
        self.assertEqual(ids, [])
        self.assertEqual(tok.encode_calls, [])

    def test_disabled_when_tag_empty(self) -> None:
        tok = _FakeTokenizer(ids=[154841])
        ids = _derive_echo_prefix_ids(_EnvCfg(think_start_tag=""), _BaseTok(tok))
        self.assertEqual(ids, [])
        self.assertEqual(tok.encode_calls, [])

    def test_fail_open_on_tokenizer_error(self) -> None:
        ids = _derive_echo_prefix_ids(
            _EnvCfg(), _BaseTok(_FakeTokenizer(raise_exc=True))
        )
        self.assertEqual(ids, [])


class CreateProxyServicerOnLoopTest(TestCase):
    def test_constructs_inside_running_loop(self) -> None:
        created_loops = []
        sentinel = object()

        def fake_servicer(**kwargs):
            created_loops.append(asyncio.get_running_loop())
            self.assertEqual(kwargs, {"rank_id": 7, "server_id": "42"})
            return sentinel

        async def run():
            with patch.object(bg_app, "DashScProxyServicer", side_effect=fake_servicer):
                loop = asyncio.get_running_loop()
                servicer = await _create_proxy_servicer_on_loop(
                    rank_id=7,
                    server_id="42",
                )
            return loop, servicer

        loop, servicer = asyncio.run(run())
        self.assertIs(servicer, sentinel)
        self.assertEqual(created_loops, [loop])


class ProxyModeEnvTest(TestCase):
    def test_service_route_alone_does_not_enable_proxy_mode(self) -> None:
        with patch.dict(
            os.environ,
            {"SERVICE_ROUTE": ('{"type": "ip_port_list", "address": "127.0.0.1:1"}')},
            clear=True,
        ):
            self.assertFalse(_is_proxy_mode_enabled())

    def test_proxy_mode_enabled_only_by_one(self) -> None:
        with patch.dict(os.environ, {"DASH_SC_GRPC_PROXY_MODE": "1"}, clear=True):
            self.assertTrue(_is_proxy_mode_enabled())
        with patch.dict(os.environ, {"DASH_SC_GRPC_PROXY_MODE": "true"}, clear=True):
            self.assertFalse(_is_proxy_mode_enabled())

    def test_legacy_forward_addr_still_enables_proxy_mode(self) -> None:
        with patch.dict(
            os.environ,
            {"DASH_SC_GRPC_FORWARD_ADDR": "127.0.0.1:1"},
            clear=True,
        ):
            self.assertTrue(_is_proxy_mode_enabled())


class PreStopDrainSecondsTest(TestCase):
    def test_default_pre_stop_drain(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_pre_stop_drain_seconds(), 120.0)

    def test_env_pre_stop_drain(self) -> None:
        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "2.5"}, clear=True
        ):
            self.assertEqual(_pre_stop_drain_seconds(), 2.5)

    def test_bad_pre_stop_drain_uses_default(self) -> None:
        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "bad"}, clear=True
        ):
            self.assertEqual(_pre_stop_drain_seconds(), 120.0)

    def test_effective_pre_stop_drain_clamps_to_shutdown_timeout(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)

        class _ServerConfig:
            shutdown_timeout = 10

        app.server_config = _ServerConfig()
        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "30"}, clear=True
        ):
            self.assertEqual(app._effective_pre_stop_drain_seconds(), 9.0)

    def test_effective_pre_stop_drain_reserves_shutdown_headroom(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)

        class _ServerConfig:
            shutdown_timeout = 600

        app.server_config = _ServerConfig()
        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "600"}, clear=True
        ):
            self.assertEqual(app._effective_pre_stop_drain_seconds(), 540.0)

    def test_grpc_stop_grace_uses_remaining_pre_stop_budget(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)

        class _ServerConfig:
            shutdown_timeout = 10

        app.server_config = _ServerConfig()
        app._shutdown_manager = DashScShutdownManager()
        app._shutdown_started_at = 100.0

        with patch("rtp_llm.dash_sc.app.time.monotonic", return_value=107.0):
            self.assertEqual(app._remaining_grpc_stop_grace_seconds(), 3.0)

    def test_grpc_stop_grace_counts_prior_drain_signal_time(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)

        class _ServerConfig:
            shutdown_timeout = 10

        app.server_config = _ServerConfig()
        app._shutdown_manager = DashScShutdownManager()
        app._shutdown_started_at = 100.0

        with patch.object(
            app._shutdown_manager, "drain_elapsed_seconds", return_value=7.0
        ), patch("rtp_llm.dash_sc.app.time.monotonic", return_value=101.0):
            self.assertEqual(app._remaining_grpc_stop_grace_seconds(), 3.0)

    def test_pre_stop_active_wait_counts_prior_drain_signal_time(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)
        app._shutdown_started_at = 100.0
        app._shutdown_manager = DashScShutdownManager()
        self.assertTrue(app._shutdown_manager.try_begin_request())
        app._shutdown_manager.start_unavailable("unit test")

        class _ServerConfig:
            shutdown_timeout = 30

        app.server_config = _ServerConfig()

        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "10"}, clear=True
        ), patch.object(
            app._shutdown_manager, "drain_elapsed_seconds", return_value=9.0
        ), patch.object(
            app._shutdown_manager,
            "wait_for_no_active_requests",
            return_value=False,
        ) as wait_for_no_active:
            app._sleep_before_stop_for_drain()

        wait_for_no_active.assert_called_once_with(1.0)

    def test_pre_stop_route_linger_runs_without_active_rpc(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)
        app._shutdown_started_at = 100.0
        app._shutdown_manager = DashScShutdownManager()
        app._shutdown_manager.start_unavailable("unit test")

        class _ServerConfig:
            shutdown_timeout = 30

        app.server_config = _ServerConfig()

        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "10"}, clear=True
        ), patch.object(
            app._shutdown_manager,
            "drain_elapsed_seconds",
            return_value=9.0,
        ), patch.object(
            app._shutdown_manager,
            "wait_for_no_active_requests",
            return_value=False,
        ) as wait_for_no_active, patch(
            "rtp_llm.dash_sc.app.time.sleep"
        ) as sleep:
            app._sleep_before_stop_for_drain()

        wait_for_no_active.assert_not_called()
        sleep.assert_called_once_with(1.0)

    def test_sleep_before_stop_skips_when_already_draining(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)
        app._shutdown_started_at = 100.0
        app._shutdown_manager = DashScShutdownManager()
        app._shutdown_manager.start_draining("signal 2")

        class _ServerConfig:
            shutdown_timeout = 30

        app.server_config = _ServerConfig()

        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "10"}, clear=True
        ), patch("rtp_llm.dash_sc.app.time.sleep") as sleep:
            app._sleep_before_stop_for_drain()

        sleep.assert_not_called()

    def test_pre_stop_signal_watchdog_releases_service_loop(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)
        app._shutdown_started_at = None
        app._shutdown_requested = False
        app._pre_stop_lock = threading.RLock()
        app._pre_stop_timer = None
        app._shutdown_manager = DashScShutdownManager()
        app._shutdown_event = threading.Event()
        self.assertTrue(app._shutdown_manager.try_begin_request())
        handlers = {}

        class _ServerConfig:
            shutdown_timeout = 30

        app.server_config = _ServerConfig()

        def capture_signal(sig, handler):
            handlers[sig] = handler

        with patch("rtp_llm.dash_sc.app.signal.signal", side_effect=capture_signal):
            app._install_signal_handlers()

        service_loop = threading.Thread(target=app._shutdown_event.wait, daemon=True)
        service_loop.start()
        with patch.dict(
            os.environ,
            {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "0.1"},
            clear=True,
        ):
            handlers[signal.SIGUSR1](signal.SIGUSR1, None)
            self.assertTrue(app._shutdown_manager.is_unavailable())
            self.assertFalse(app._shutdown_manager.is_draining())
            self.assertEqual(app._shutdown_manager.drain_reason(), "signal 10")
            self.assertEqual(app._shutdown_manager.active_request_count(), 1)
            self.assertFalse(app._shutdown_manager.try_begin_request())
            self.assertEqual(app._shutdown_manager.finish_request(), 0)
            service_loop.join(timeout=1.0)

        self.assertFalse(service_loop.is_alive())
        self.assertTrue(app._shutdown_manager.is_draining())
        self.assertTrue(app._shutdown_manager.is_unavailable())
        self.assertIsNotNone(app._shutdown_started_at)
        self.assertFalse(app._shutdown_manager.try_begin_request())
        self.assertEqual(app._shutdown_manager.active_request_count(), 0)

    def test_sigterm_marks_unavailable_until_grpc_stop(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)
        app._shutdown_started_at = None
        app._shutdown_requested = False
        app._pre_stop_lock = threading.RLock()
        app._pre_stop_timer = None
        app._shutdown_manager = DashScShutdownManager()
        app._shutdown_event = Mock()
        handlers = {}

        class _ServerConfig:
            shutdown_timeout = 30

        app.server_config = _ServerConfig()

        def capture_signal(sig, handler):
            handlers[sig] = handler

        with patch("rtp_llm.dash_sc.app.signal.signal", side_effect=capture_signal):
            app._install_signal_handlers()

        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "10"}, clear=True
        ):
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        self.assertFalse(app._shutdown_manager.try_begin_request())
        self.assertFalse(app._shutdown_manager.is_draining())
        self.assertTrue(app._shutdown_manager.is_unavailable())
        self.assertIsNotNone(app._shutdown_started_at)
        self.assertTrue(app._shutdown_requested)
        self.assertIsNone(app._pre_stop_timer)
        app._shutdown_event.set.assert_called_once()

    def test_sigterm_cancels_pre_stop_watchdog(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)
        app._shutdown_started_at = None
        app._shutdown_requested = False
        app._pre_stop_lock = threading.RLock()
        app._pre_stop_timer = None
        app._shutdown_manager = DashScShutdownManager()
        app._shutdown_event = Mock()
        handlers = {}

        class _ServerConfig:
            shutdown_timeout = 30

        app.server_config = _ServerConfig()

        def capture_signal(sig, handler):
            handlers[sig] = handler

        with patch("rtp_llm.dash_sc.app.signal.signal", side_effect=capture_signal):
            app._install_signal_handlers()

        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "10"}, clear=True
        ):
            handlers[signal.SIGUSR1](signal.SIGUSR1, None)
            watchdog = app._pre_stop_timer
            self.assertIsNotNone(watchdog)
            with patch.object(watchdog, "cancel", wraps=watchdog.cancel) as cancel:
                handlers[signal.SIGTERM](signal.SIGTERM, None)

        cancel.assert_called_once()
        self.assertTrue(app._shutdown_requested)
        self.assertIsNone(app._pre_stop_timer)
        app._shutdown_event.set.assert_called_once()

    def test_grpc_stop_marks_draining_after_pre_stop_window(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)
        app._shutdown_started_at = 100.0
        app._shutdown_manager = DashScShutdownManager()
        app._shutdown_manager.start_unavailable("signal 15")
        app._grpc_server = Mock()
        app._stop_enqueue_loop = Mock()

        class _ServerConfig:
            shutdown_timeout = 30

        app.server_config = _ServerConfig()

        with patch.object(app, "_sleep_before_stop_for_drain"), patch.object(
            app, "_remaining_grpc_stop_grace_seconds", return_value=5.0
        ):
            app.stop()

        self.assertTrue(app._shutdown_manager.is_draining())
        app._grpc_server.stop.assert_called_once_with(5.0)
        app._stop_enqueue_loop.assert_called_once()

    def test_grpc_stop_waits_for_active_rpc_to_drain(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)
        app._shutdown_started_at = 100.0
        app._shutdown_manager = DashScShutdownManager()
        app._grpc_server = Mock()
        app._stop_enqueue_loop = Mock()

        class _ServerConfig:
            shutdown_timeout = 30

        app.server_config = _ServerConfig()
        self.assertTrue(app._shutdown_manager.try_begin_request())

        def finish_request():
            time.sleep(0.05)
            app._shutdown_manager.finish_request()

        t = threading.Thread(target=finish_request)
        t.start()
        start = time.monotonic()
        with patch.object(app, "_sleep_before_stop_for_drain"), patch.object(
            app, "_remaining_grpc_stop_grace_seconds", return_value=1.0
        ):
            app.stop()
        t.join(timeout=1.0)

        self.assertGreaterEqual(time.monotonic() - start, 0.04)
        self.assertTrue(app._shutdown_manager.is_draining())
        app._grpc_server.stop.assert_called_once_with(1.0)
        app._stop_enqueue_loop.assert_called_once()


class DashScGrpcServerStopTest(TestCase):
    def test_stop_wait_is_bounded_by_remaining_grace(self) -> None:
        grpc_server = DashScGrpcServer()
        grpc_server._server = Mock()
        grpc_server._servicer = None
        grpc_server._loop = Mock()
        stop_future = Mock()
        stop_future.result.side_effect = concurrent.futures.TimeoutError

        def submit(coroutine, loop):
            coroutine.close()
            return stop_future

        with patch(
            "rtp_llm.dash_sc.server.asyncio.run_coroutine_threadsafe",
            side_effect=submit,
        ):
            grpc_server.stop(grace=0.25)

        stop_future.result.assert_called_once_with(timeout=0.25)
        stop_future.cancel.assert_called_once()
        self.assertFalse(grpc_server.is_running)


class DashScShutdownManagerTest(TestCase):
    class _AbortContext:
        def __init__(self) -> None:
            self.abort_args = None

        async def abort(self, *args):
            self.abort_args = args
            raise RuntimeError("aborted")

    def test_unary_unary_drain_interceptor_rejects_new_rpc(self) -> None:
        manager = DashScShutdownManager()
        interceptor = DashScGrpcDrainAioInterceptor(manager)
        called = False

        async def unary_handler(_request, _context):
            nonlocal called
            called = True
            return "ok"

        async def continuation(_details):
            return grpc.unary_unary_rpc_method_handler(unary_handler)

        manager.start_draining("unit test")

        async def run():
            handler = await interceptor.intercept_service(
                continuation, SimpleNamespace(method="/test.Service/Unary")
            )
            context = self._AbortContext()
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                await handler.unary_unary(object(), context)
            return context.abort_args

        abort_args = asyncio.run(run())
        self.assertEqual(abort_args[0], grpc.StatusCode.UNAVAILABLE)
        self.assertIn("dash_sc is draining", abort_args[1])
        self.assertFalse(called)
        self.assertEqual(manager.active_request_count(), 0)

    def test_pre_stop_unavailable_interceptor_rejects_new_rpc(self) -> None:
        manager = DashScShutdownManager()
        interceptor = DashScGrpcDrainAioInterceptor(manager)
        called = False

        async def unary_handler(_request, _context):
            nonlocal called
            called = True
            return "ok"

        async def continuation(_details):
            return grpc.unary_unary_rpc_method_handler(unary_handler)

        manager.start_unavailable("signal 10")

        async def run():
            handler = await interceptor.intercept_service(
                continuation, SimpleNamespace(method="/test.Service/Unary")
            )
            context = self._AbortContext()
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                await handler.unary_unary(object(), context)
            return context.abort_args

        abort_args = asyncio.run(run())
        self.assertEqual(abort_args[0], grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(
            abort_args[1],
            "dash_sc is unavailable: reason=signal 10 active_requests=0",
        )
        self.assertFalse(called)
        self.assertEqual(manager.active_request_count(), 0)

    def test_pre_stop_unavailable_allows_server_live_liveness(self) -> None:
        manager = DashScShutdownManager()
        interceptor = DashScGrpcDrainAioInterceptor(manager)
        called = False

        async def unary_handler(_request, _context):
            nonlocal called
            called = True
            return "live"

        async def continuation(_details):
            return grpc.unary_unary_rpc_method_handler(unary_handler)

        manager.start_unavailable("unit test")

        async def run():
            handler = await interceptor.intercept_service(
                continuation,
                SimpleNamespace(method="/inference.GRPCInferenceService/ServerLive"),
            )
            return await handler.unary_unary(object(), object())

        self.assertEqual(asyncio.run(run()), "live")
        self.assertTrue(called)
        self.assertEqual(manager.active_request_count(), 0)

    def test_pre_stop_unavailable_keeps_existing_rpc_counted(self) -> None:
        manager = DashScShutdownManager()
        self.assertTrue(manager.try_begin_request())
        manager.start_unavailable("unit test")

        self.assertFalse(manager.try_begin_request())
        self.assertEqual(manager.active_request_count(), 1)
        self.assertEqual(manager.finish_request(), 0)

    def test_wait_for_no_active_requests_notifies_on_finish(self) -> None:
        manager = DashScShutdownManager()
        self.assertTrue(manager.try_begin_request())

        def finish_request():
            time.sleep(0.05)
            manager.finish_request()

        t = threading.Thread(target=finish_request)
        t.start()
        start = time.monotonic()
        self.assertTrue(manager.wait_for_no_active_requests(1.0))
        t.join(timeout=1.0)
        self.assertGreaterEqual(time.monotonic() - start, 0.04)
        self.assertEqual(manager.active_request_count(), 0)

    def test_wait_for_no_active_requests_times_out(self) -> None:
        manager = DashScShutdownManager()
        self.assertTrue(manager.try_begin_request())

        start = time.monotonic()
        self.assertFalse(manager.wait_for_no_active_requests(0.02))
        self.assertGreaterEqual(time.monotonic() - start, 0.015)
        self.assertEqual(manager.active_request_count(), 1)
        manager.finish_request()

    def test_draining_rejects_new_stream_rpc(self) -> None:
        manager = DashScShutdownManager()
        interceptor = DashScGrpcDrainAioInterceptor(manager)

        async def request_iter():
            yield object()

        async def stream_stream_handler(requests, _context):
            async for _ in requests:
                self.assertEqual(manager.active_request_count(), 1)
                yield "response"

        async def continuation(_details):
            return grpc.stream_stream_rpc_method_handler(stream_stream_handler)

        async def run_ok():
            handler = await interceptor.intercept_service(
                continuation, SimpleNamespace(method="/test.Service/Stream")
            )
            responses = []
            async for resp in handler.stream_stream(request_iter(), object()):
                responses.append(resp)
            return responses

        self.assertEqual(asyncio.run(run_ok()), ["response"])
        self.assertEqual(manager.active_request_count(), 0)

        manager.start_draining("unit test")

        async def run_draining():
            handler = await interceptor.intercept_service(
                continuation, SimpleNamespace(method="/test.Service/Stream")
            )
            context = self._AbortContext()
            responses = []
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                async for resp in handler.stream_stream(request_iter(), context):
                    responses.append(resp)
            return responses, context.abort_args

        responses, abort_args = asyncio.run(run_draining())
        self.assertEqual(responses, [])
        self.assertEqual(abort_args[0], grpc.StatusCode.UNAVAILABLE)
        self.assertIn("dash_sc is draining", abort_args[1])
        self.assertEqual(manager.active_request_count(), 0)


class CloseServicerOnLoopTest(TestCase):
    def test_closes_servicer_on_enqueue_loop(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)
        loop = app._start_enqueue_loop()
        closed_loops = []

        class _Closable:
            async def close(self):
                closed_loops.append(asyncio.get_running_loop())

        try:
            app._close_servicer_on_loop(_Closable())
        finally:
            app._stop_enqueue_loop()

        self.assertEqual(closed_loops, [loop])


if __name__ == "__main__":
    main()
