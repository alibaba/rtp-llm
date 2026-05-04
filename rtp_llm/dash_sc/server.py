"""DashSc gRPC process lifecycle (grpc.aio).

Owner: :class:`DashScGrpcServer` instances hold the ``grpc.aio.Server`` +
servicer; :class:`DashScApp` constructs one per process and drives it on the
same asyncio loop that hosts the backend ``enqueue`` coroutine. No
module-level mutable state.

Standalone fake: ``python -m rtp_llm.dash_sc.server [--port PORT]``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import grpc

from rtp_llm.dash_sc.access_log import (
    DASH_SC_GRPC_ACCESS_LOG_FILENAME,
    DASH_SC_GRPC_QUERY_LOG_FILENAME,
    DashScGrpcAccessLogAioInterceptor,
    init_dash_sc_grpc_access_logger,
    init_dash_sc_grpc_query_logger,
)
from rtp_llm.dash_sc.inference.servicer import DashScInferenceServicer
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
        backend_visitor=None,
        ip: str = "",
        server_id: str = "",
        log_path: str = "",
        backup_count: int = 0,
        rank_id: Optional[int] = None,
        echo_prefix_ids: Optional[list[int]] = None,
    ) -> grpc.aio.Server:
        """Bind + start the aio gRPC server. Must be awaited on the owning loop.

        ``backend_visitor=None`` -> fake mode.

        ``ip`` / ``server_id`` + the dash_sc gRPC ``port`` feed ``generate_request_id``
        inside the servicer so the backend ``GenerateInput.request_id`` follows the same
        snowflake scheme as the HTTP path in ``FrontendServer``.

        ``log_path`` / ``backup_count`` / ``rank_id`` configure the access log file
        (``<log_path>/dash_sc_grpc_access_r{rank}_s{server}.log``). Empty ``log_path``
        -> no file handler (the interceptor still runs so tests see logger calls).
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
        is_forward = DashScProxyServicer.has_forward_config()
        interceptor = DashScGrpcAccessLogAioInterceptor(
            rank_id=rank_id,
            server_id=server_id_int,
            raw_mode=is_forward,
        )
        # Deliberately no ``maximum_concurrent_rpcs`` — under grpc.aio concurrent
        # RPCs are coroutines on one loop, not threads, so any positive value
        # becomes a hard admission cap (RESOURCE_EXHAUSTED once N long streams
        # are in flight). Backpressure comes from the backend visitor's own
        # concurrency instead. ``DashScGrpcConfig.max_server_workers`` is
        # retained on the C++ struct for wire compatibility but ignored here.
        server = grpc.aio.server(
            options=opts,
            interceptors=[interceptor],
        )
        if is_forward:
            servicer = DashScProxyServicer()
            mode = "pure forward"
        else:
            servicer = DashScInferenceServicer(
                backend_visitor=backend_visitor,
                ip=ip,
                port=port,
                server_id=server_id,
                echo_prefix_ids=echo_prefix_ids,
            )
            mode = "real" if backend_visitor else "fake"

        predict_v2_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f"0.0.0.0:{port}")
        await server.start()
        self._server = server
        self._servicer = servicer
        logging.info(
            "[DashScGrpc] Listening on 0.0.0.0:%s (predict_v2.proto, %s)",
            port,
            mode,
        )
        return server

    def start_on_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        port: int,
        backend_visitor=None,
        ip: str = "",
        server_id: str = "",
        log_path: str = "",
        backup_count: int = 0,
        rank_id: Optional[int] = None,
        startup_timeout_s: float = _DEFAULT_DASH_SC_GRPC_STARTUP_TIMEOUT_S,
        echo_prefix_ids: Optional[list[int]] = None,
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
                backend_visitor=backend_visitor,
                ip=ip,
                server_id=server_id,
                log_path=log_path,
                backup_count=backup_count,
                rank_id=rank_id,
                echo_prefix_ids=echo_prefix_ids,
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

        try:
            asyncio.run_coroutine_threadsafe(_do_stop(), loop).result()
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


def main():
    """Standalone entry (fake mode only). Usage: python -m rtp_llm.dash_sc.server [--port PORT]"""
    import argparse

    parser = argparse.ArgumentParser(
        description="DashSc gRPC server (predict_v2.proto wire, fake mode)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="gRPC port (default: 8000)"
    )
    parser.add_argument(
        "--dash_sc_grpc_config_json",
        type=str,
        default="",
        help="Optional JSON for DashScGrpcConfig (same shape as --dash_sc_grpc_config_json on main server).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    dash_sc_cfg = None
    if args.dash_sc_grpc_config_json.strip():
        from rtp_llm.ops import DashScGrpcConfig

        dash_sc_cfg = DashScGrpcConfig()
        dash_sc_cfg.from_json(args.dash_sc_grpc_config_json.strip())

    async def _run():
        grpc_server = DashScGrpcServer(dash_sc_grpc_config=dash_sc_cfg)
        server = await grpc_server.start(args.port, backend_visitor=None)
        await server.wait_for_termination()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
