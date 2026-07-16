"""DashSc gRPC process lifecycle (grpc.aio).

Owner: :class:`DashScGrpcServer` instances hold the ``grpc.aio.Server`` +
servicer; :class:`DashScApp` constructs one per process and drives it on the
same asyncio loop that hosts the backend ``enqueue`` coroutine. No
module-level mutable state.

Standalone fake: ``python -m rtp_llm.dash_sc.server [--port PORT]``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Optional

import grpc

from rtp_llm.dash_sc.access_log import (
    DASH_SC_GRPC_ACCESS_LOG_FILENAME,
    DASH_SC_GRPC_QUERY_LOG_FILENAME,
    init_dash_sc_grpc_access_logger,
    init_dash_sc_grpc_query_logger,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc
from rtp_llm.dash_sc.proxy.servicer import DashScProxyServicer


def _resolve_dash_sc_grpc_config(dash_sc_grpc_config):
    if dash_sc_grpc_config is not None:
        return dash_sc_grpc_config
    from rtp_llm.ops import DashScGrpcConfig

    return DashScGrpcConfig()


def dash_sc_grpc_server_channel_options(dash_sc_grpc_config) -> list[tuple[str, int]]:
    """``grpc.aio.server(..., options=...)`` from ``DashScGrpcConfig.get_server_config()``."""
    cfg = _resolve_dash_sc_grpc_config(dash_sc_grpc_config)
    return sorted((str(k), int(v)) for k, v in cfg.get_server_config().items())


# Max time for grpc.aio.Server.start() + bind before start_on_loop returns.
_DEFAULT_DASH_SC_GRPC_STARTUP_TIMEOUT_S = 30.0

# HTTP/2 keepalive permissions for the dash_sc gRPC server side. The upstream
# (whoever calls us: dash_sc forwarder, client SDK, …) needs to be able to
# send keepalive PINGs every ~30s to defeat the 100s LBS idle timeout — but
# grpcio's server default is to GOAWAY any client that exceeds 2 PINGs in
# 5 minutes without data (``min_ping_interval_without_data_ms=300000`` +
# ``max_pings_without_data=2``). Raising permit here so the client-side
# keepalive we ship in ``forward_service._FORWARD_CHANNEL_OPTS`` actually
# works end-to-end instead of racing a GOAWAY.
#
# We also enable server-originated PINGs (``keepalive_time_ms=30000``) so
# the server probes the client just as actively; this is symmetric and
# catches the case where the downstream is healthy but the client went dark.
#
# Merged via ``setdefault`` — anything explicitly set by
# ``DashScGrpcConfig.get_server_config()`` wins, so operators can still
# override per-deployment.
_SERVER_KEEPALIVE_OPTS: list[tuple[str, int]] = [
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 0),
    ("grpc.http2.min_ping_interval_without_data_ms", 10000),
    ("grpc.http2.max_pings_without_data", 0),
]


def _merge_server_keepalive(
    opts: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Add keepalive defaults to ``opts`` without overriding explicit config."""
    merged = dict(opts)
    for k, v in _SERVER_KEEPALIVE_OPTS:
        merged.setdefault(k, v)
    return sorted(merged.items())


class DashScGrpcServer:
    """Per-process owner of the DashSc ``grpc.aio.Server``.

    Each instance holds the server + servicer + owning event loop as instance
    state; multiple instances (e.g. in tests) are independent.
    :class:`DashScApp` constructs exactly one per process and pins it to the
    same loop that hosts the backend ``enqueue`` coroutine so there is no
    cross-thread hop on the request path.

    Typical production flow (from ``DashScApp.start``):

    1. ``grpc_server = DashScGrpcServer(dash_sc_grpc_config=...)``
    2. ``grpc_server.start_on_loop(loop, port=port, backend_visitor=...)``
       schedules ``start()`` on ``loop`` and blocks the calling thread until
       the bind succeeds or raises.
    3. On SIGTERM/SIGINT the app calls ``grpc_server.stop(grace=...)``.

    Not reusable across ``start`` → ``stop`` cycles: once stopped, build a
    fresh instance. This keeps lifecycle state strictly bounded and rules
    out stale-server reuse bugs.
    """

    def __init__(self, dash_sc_grpc_config=None):
        self._config = dash_sc_grpc_config
        self._server: Optional[grpc.aio.Server] = None
        self._servicer: Optional[predict_v2_pb2_grpc.GRPCInferenceServiceServicer] = (
            None
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_running(self) -> bool:
        return self._server is not None

    async def start(
        self,
        port: int,
        *,
        servicer: predict_v2_pb2_grpc.GRPCInferenceServiceServicer,
        shutdown_manager: Optional[Any] = None,
        server_id: str = "",
        log_path: str = "",
        backup_count: int = 0,
        rank_id: Optional[int] = None,
    ) -> grpc.aio.Server:
        """Bind + start the aio gRPC server. Must be awaited on the owning loop.

        ``servicer`` is fully constructed by the caller — the server no longer
        inspects env or decides between proxy/inference mode. This keeps mode
        selection at the process-boundary (``DashScApp`` / ``__main__``) where
        it belongs, not inside a shared infrastructure class.

        ``log_path`` / ``backup_count`` / ``rank_id`` / ``server_id`` configure
        the access log file (``<log_path>/dash_sc_grpc_access_r{rank}_s{server}.log``).
        Empty ``log_path`` -> no file handler (the servicers still call the
        emit functions, so tests see logger calls).
        """
        if self._server is not None:
            logging.warning("[DashScGrpc] server already started")
            return self._server
        cfg = _resolve_dash_sc_grpc_config(self._config)
        opts = _merge_server_keepalive(dash_sc_grpc_server_channel_options(cfg))

        server_id_int: Optional[int]
        try:
            server_id_int = int(server_id) if server_id not in (None, "") else None
        except (TypeError, ValueError):
            server_id_int = None
        init_dash_sc_grpc_access_logger(
            log_path=log_path,
            backup_count=backup_count,
            rank_id=rank_id,
            server_id=server_id_int,
        )
        init_dash_sc_grpc_query_logger(
            log_path=log_path,
            backup_count=backup_count,
            rank_id=rank_id,
            server_id=server_id_int,
        )
        if log_path:
            from rtp_llm.access_logger.log_utils import get_process_log_filename

            logging.info(
                "[DashScGrpc] access log enabled: path=%s/%s",
                log_path,
                get_process_log_filename(
                    DASH_SC_GRPC_ACCESS_LOG_FILENAME, rank_id, server_id_int
                ),
            )
            logging.info(
                "[DashScGrpc] query log enabled: path=%s/%s",
                log_path,
                get_process_log_filename(
                    DASH_SC_GRPC_QUERY_LOG_FILENAME, rank_id, server_id_int
                ),
            )
        # The shared access-log interceptor is gone: each servicer owns its
        # access-log lifecycle inline in ``ModelStreamInfer``, with its
        # rank/server identity injected at construction by ``DashScApp``. The
        # server only opens the proxy's lazy outbound channel cache.
        if isinstance(servicer, DashScProxyServicer):
            await servicer.open()
        # The only interceptor left is the graceful pre-stop drain (the shared
        # access-log interceptor was dissolved into the servicers), and it is
        # installed only when the caller wired a ``shutdown_manager``.
        interceptors: list[grpc.aio.ServerInterceptor] = []
        if shutdown_manager is not None:
            interceptors.append(DashScGrpcDrainAioInterceptor(shutdown_manager))
        # Deliberately no ``maximum_concurrent_rpcs`` — under grpc.aio concurrent
        # RPCs are coroutines on one loop, not threads, so any positive value
        # becomes a hard admission cap (RESOURCE_EXHAUSTED once N long streams
        # are in flight). Backpressure comes from the backend visitor's own
        # concurrency instead. ``DashScGrpcConfig.max_server_workers`` is
        # retained on the C++ struct for wire compatibility but ignored here.
        server = grpc.aio.server(options=opts, interceptors=interceptors)

        predict_v2_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f"0.0.0.0:{port}")
        await server.start()
        self._server = server
        self._servicer = servicer
        logging.info(
            "[DashScGrpc] Listening on 0.0.0.0:%s (predict_v2.proto, servicer=%s)",
            port,
            type(servicer).__name__,
        )
        return server

    def start_on_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        port: int,
        servicer: predict_v2_pb2_grpc.GRPCInferenceServiceServicer,
        shutdown_manager: Optional[Any] = None,
        server_id: str = "",
        log_path: str = "",
        backup_count: int = 0,
        rank_id: Optional[int] = None,
        startup_timeout_s: float = _DEFAULT_DASH_SC_GRPC_STARTUP_TIMEOUT_S,
    ) -> None:
        """Schedule ``start()`` on ``loop`` and block until it returns or raises.

        The calling thread is *not* the loop thread — ``loop`` runs in a
        background thread owned by :class:`DashScApp`. We submit the coroutine
        with ``run_coroutine_threadsafe`` and ``.result(timeout=...)`` to
        surface bind/start failures synchronously so callers don't proceed
        with a dead server.
        """
        fut = asyncio.run_coroutine_threadsafe(
            self.start(
                port=port,
                servicer=servicer,
                shutdown_manager=shutdown_manager,
                server_id=server_id,
                log_path=log_path,
                backup_count=backup_count,
                rank_id=rank_id,
            ),
            loop,
        )
        fut.result(timeout=startup_timeout_s)
        self._loop = loop

    def stop(self, grace: Optional[float] = None) -> None:
        """Stop the ``grpc.aio.Server`` and clear instance state.

        ``grace`` is passed to ``grpc.aio.Server.stop`` (seconds to let
        in-flight RPCs finish); ``None`` uses grpcio default. Safe to call
        multiple times or if the server was never started.
        """
        server = self._server
        loop = self._loop
        if server is None or loop is None:
            return
        servicer = self._servicer

        async def _do_stop():
            try:
                await server.stop(grace)
                logging.info(
                    "[DashScGrpc] server stopped (grace=%r)",
                    grace,
                )
            except Exception as e:
                logging.warning(
                    "[DashScGrpc] server.stop failed: %s",
                    e,
                    exc_info=True,
                )
            if servicer is not None:
                close = getattr(servicer, "close", None)
                if close is not None:
                    try:
                        maybe = close()
                        if asyncio.iscoroutine(maybe):
                            await maybe
                    except Exception as e:
                        logging.warning(
                            "[DashScGrpc] servicer.close failed: %s",
                            e,
                            exc_info=True,
                        )

        fut = asyncio.run_coroutine_threadsafe(_do_stop(), loop)
        stop_timeout = None if grace is None else max(0.0, float(grace))
        try:
            fut.result(timeout=stop_timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            logging.warning(
                "[DashScGrpc] stop timed out after %.3fs",
                stop_timeout,
            )
        except Exception as e:
            logging.warning(
                "[DashScGrpc] stop scheduling failed: %s",
                e,
                exc_info=True,
            )
        finally:
            self._server = None
            self._servicer = None
            self._loop = None

    def wait_for_termination(self) -> None:
        """Block on ``grpc.aio.Server.wait_for_termination``. No-op if not started."""
        server = self._server
        loop = self._loop
        if server is None or loop is None:
            return
        asyncio.run_coroutine_threadsafe(server.wait_for_termination(), loop).result()


class DashScGrpcDrainAioInterceptor(grpc.aio.ServerInterceptor):
    """Track DashSc RPCs during pre-stop drain.

    Health/service-discovery should remove the node once pre-stop starts, but
    upstream clients can still have stale endpoints cached for a short window.
    Reject new RPCs as soon as the process enters the pre-stop unavailable
    state so no new request starts on a node that is leaving service discovery.
    RPCs that already entered the handler keep their active counter until the
    handler completes, and grpc stop later uses its grace window for those
    in-flight RPCs.
    """

    def __init__(self, shutdown_manager: Any):
        self._shutdown_manager = shutdown_manager

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None:
            return handler

        method = handler_call_details.method
        if self._is_liveness_method(method):
            return handler
        if handler.request_streaming and handler.response_streaming:
            return grpc.stream_stream_rpc_method_handler(
                self._wrap_stream_stream(handler.stream_stream, method),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.request_streaming and not handler.response_streaming:
            return grpc.stream_unary_rpc_method_handler(
                self._wrap_stream_unary(handler.stream_unary, method),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if not handler.request_streaming and handler.response_streaming:
            return grpc.unary_stream_rpc_method_handler(
                self._wrap_unary_stream(handler.unary_stream, method),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return grpc.unary_unary_rpc_method_handler(
            self._wrap_unary_unary(handler.unary_unary, method),
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )

    @staticmethod
    def _is_liveness_method(method: str) -> bool:
        return method.endswith("/ServerLive")

    async def _begin_or_abort(self, context, method: str) -> bool:
        if self._shutdown_manager.try_begin_request():
            return True
        state = "draining" if self._shutdown_manager.is_draining() else "unavailable"
        detail = "dash_sc is %s: reason=%s active_requests=%s" % (
            state,
            self._shutdown_manager.drain_reason(),
            self._shutdown_manager.active_request_count(),
        )
        logging.info(
            "[DashScGrpc] rejecting new RPC during %s: %s %s", state, method, detail
        )
        await context.abort(grpc.StatusCode.UNAVAILABLE, detail)
        return False

    def _finish(self, method: str) -> None:
        active = self._shutdown_manager.finish_request()
        if self._shutdown_manager.is_draining():
            logging.info(
                "[DashScGrpc] RPC finished during drain: method=%s active_requests=%s",
                method,
                active,
            )

    def _wrap_unary_unary(self, inner, method: str):
        async def behavior(request, context):
            began = await self._begin_or_abort(context, method)
            if not began:
                return None
            try:
                return await inner(request, context)
            finally:
                if began:
                    self._finish(method)

        return behavior

    def _wrap_unary_stream(self, inner, method: str):
        async def behavior(request, context):
            began = await self._begin_or_abort(context, method)
            if not began:
                return
            try:
                async for resp in inner(request, context):
                    yield resp
            finally:
                if began:
                    self._finish(method)

        return behavior

    def _wrap_stream_unary(self, inner, method: str):
        async def behavior(request_iterator, context):
            began = await self._begin_or_abort(context, method)
            if not began:
                return None
            try:
                return await inner(request_iterator, context)
            finally:
                if began:
                    self._finish(method)

        return behavior

    def _wrap_stream_stream(self, inner, method: str):
        async def behavior(request_iterator, context):
            began = await self._begin_or_abort(context, method)
            if not began:
                return
            try:
                async for resp in inner(request_iterator, context):
                    yield resp
            finally:
                if began:
                    self._finish(method)

        return behavior


def main():
    """Standalone fake-inference entry. Usage: python -m rtp_llm.dash_sc.server [--port PORT].

    Preserved for backward compatibility with existing docs / tooling. New
    callers should prefer the per-mode entry points:
    ``python -m rtp_llm.dash_sc.inference`` (fake inference) and
    ``python -m rtp_llm.dash_sc.proxy`` (reverse-proxy).
    """
    from rtp_llm.dash_sc.inference.__main__ import main as inference_main

    inference_main()


if __name__ == "__main__":
    main()
