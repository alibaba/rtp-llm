"""DashSc gRPC process lifecycle.

Owner: :class:`DashScGrpcServer` instances hold the ``grpc.Server`` + servicer + worker
thread + lock; :class:`DashScApp` constructs one per process. No module-level mutable state.

Standalone fake: ``python -m rtp_llm.dash_sc.server [--port PORT]``.
"""

from __future__ import annotations

import logging
import threading
from concurrent import futures
from typing import Optional

import grpc

from rtp_llm.dash_sc.access_log import (
    DASH_SC_GRPC_ACCESS_LOG_FILENAME,
    DashScGrpcAccessLogInterceptor,
    init_dash_sc_grpc_access_logger,
)
from rtp_llm.dash_sc.forward_service import PureForwardServicer
from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc
from rtp_llm.dash_sc.service import DashScGrpcInferenceServicer


def _resolve_dash_sc_grpc_config(dash_sc_grpc_config):
    if dash_sc_grpc_config is not None:
        return dash_sc_grpc_config
    from rtp_llm.ops import DashScGrpcConfig

    return DashScGrpcConfig()


def dash_sc_grpc_server_channel_options(dash_sc_grpc_config) -> list[tuple[str, int]]:
    """``grpc.server(..., options=...)`` from ``DashScGrpcConfig.get_server_config()``."""
    cfg = _resolve_dash_sc_grpc_config(dash_sc_grpc_config)
    return sorted((str(k), int(v)) for k, v in cfg.get_server_config().items())


def dash_sc_grpc_server_max_server_workers(dash_sc_grpc_config) -> int:
    cfg = _resolve_dash_sc_grpc_config(dash_sc_grpc_config)
    n = int(cfg.max_server_workers)
    assert n > 0, "dash_sc_grpc_config.max_server_workers must be > 0"
    return n


# Max time for grpc.Server.start() + bind before start_in_thread returns.
_DEFAULT_DASH_SC_GRPC_STARTUP_TIMEOUT_S = 30.0


class DashScGrpcServer:
    """Per-process owner of the DashSc gRPC ``grpc.Server``.

    Each instance holds the server + servicer + worker thread + lock as instance state,
    so multiple instances (e.g. in tests) are independent. :class:`DashScApp` constructs
    exactly one per process; the standalone CLI builds a transient instance and blocks
    on :meth:`wait_for_termination`.

    Typical production flow (from ``DashScApp.start``):

    1. ``grpc_server = DashScGrpcServer(dash_sc_grpc_config=...)``
    2. ``grpc_server.start_in_thread(port, backend_visitor=..., ip=..., server_id=..., ...)``
       blocks until ``grpc.Server.start()`` succeeds or raises on bind error.
    3. On SIGTERM/SIGINT the app calls ``grpc_server.stop(grace=...)``.

    Not reusable across ``start`` → ``stop`` cycles: once stopped, build a fresh instance.
    This keeps lifecycle state strictly bounded and rules out stale-server reuse bugs.
    """

    def __init__(self, dash_sc_grpc_config=None):
        self._config = dash_sc_grpc_config
        self._server: Optional[grpc.Server] = None
        self._servicer: Optional[predict_v2_pb2_grpc.GRPCInferenceServiceServicer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._server is not None

    def start(
        self,
        port: int,
        *,
        backend_visitor=None,
        ip: str = "",
        server_id: str = "",
        log_path: str = "",
        backup_count: int = 0,
        rank_id: Optional[int] = None,
    ) -> grpc.Server:
        """Bind + start the gRPC server synchronously. ``backend_visitor=None`` -> fake mode.

        ``ip`` / ``server_id`` + the dash_sc gRPC ``port`` feed ``generate_request_id``
        inside the servicer so the backend ``GenerateInput.request_id`` follows the same
        snowflake scheme as the HTTP path in ``FrontendServer``.

        ``log_path`` / ``backup_count`` / ``rank_id`` configure the access log file
        (``<log_path>/dash_sc_grpc_access_r{rank}_s{server}.log``). Empty ``log_path``
        -> no file handler (the interceptor still runs so tests see logger calls).
        """
        with self._lock:
            if self._server is not None:
                logging.warning("[DashScGrpc] server already started")
                return self._server
            cfg = _resolve_dash_sc_grpc_config(self._config)
            opts = dash_sc_grpc_server_channel_options(cfg)
            pool_size = dash_sc_grpc_server_max_server_workers(cfg)

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
            if log_path:
                from rtp_llm.access_logger.log_utils import get_process_log_filename

                logging.info(
                    "[DashScGrpc] access log enabled: path=%s/%s",
                    log_path,
                    get_process_log_filename(
                        DASH_SC_GRPC_ACCESS_LOG_FILENAME, rank_id, server_id_int
                    ),
                )
            interceptor = DashScGrpcAccessLogInterceptor(
                rank_id=rank_id, server_id=server_id_int
            )
            server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=pool_size),
                options=opts,
                interceptors=[interceptor],
            )
            if PureForwardServicer.has_forward_config():
                servicer = PureForwardServicer()
                mode = "pure forward"
            else:
                servicer = DashScGrpcInferenceServicer(
                    backend_visitor=backend_visitor,
                    ip=ip,
                    port=port,
                    server_id=server_id,
                )
                mode = "real" if backend_visitor else "fake"

            predict_v2_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(servicer, server)
            server.add_insecure_port(f"0.0.0.0:{port}")
            server.start()
            self._server = server
            self._servicer = servicer
            logging.info(
                "[DashScGrpc] Listening on 0.0.0.0:%s (predict_v2.proto, %s, max_server_workers=%s)",
                port,
                mode,
                pool_size,
            )
            return server

    def start_in_thread(
        self,
        port: int,
        *,
        backend_visitor=None,
        ip: str = "",
        server_id: str = "",
        log_path: str = "",
        backup_count: int = 0,
        rank_id: Optional[int] = None,
        startup_timeout_s: float = _DEFAULT_DASH_SC_GRPC_STARTUP_TIMEOUT_S,
    ) -> None:
        """Start gRPC in a daemon thread and block until ``server.start()`` succeeds.

        The daemon thread runs ``wait_for_termination()`` so the process keeps serving.
        Bind/start failures or startup beyond ``startup_timeout_s`` raise synchronously
        (either the underlying exception or ``TimeoutError``) so callers do not proceed
        while the port is not listening.
        """
        started_ok = threading.Event()
        start_error: list[BaseException] = []

        def _run():
            try:
                server = self.start(
                    port,
                    backend_visitor=backend_visitor,
                    ip=ip,
                    server_id=server_id,
                    log_path=log_path,
                    backup_count=backup_count,
                    rank_id=rank_id,
                )
            except BaseException as e:
                start_error.append(e)
                started_ok.set()
                return
            started_ok.set()
            if server is not None:
                server.wait_for_termination()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if not started_ok.wait(timeout=startup_timeout_s):
            raise TimeoutError(
                f"DashSc gRPC server did not finish starting within {startup_timeout_s}s "
                f"(port {port})"
            )
        if start_error:
            raise start_error[0]

    def stop(self, grace: Optional[float] = None) -> None:
        """Stop the ``grpc.Server`` and clear instance state.

        ``grace`` is passed to ``grpc.Server.stop`` (seconds to let in-flight RPCs
        finish); ``None`` uses grpcio default. Safe to call multiple times or if the
        server was never started. Holds ``_lock`` for the full ``stop()`` so a
        concurrent ``start`` cannot race on the same listen port during shutdown.
        """
        with self._lock:
            server = self._server
            if server is None:
                return
            try:
                server.stop(grace)
                logging.info(
                    "[DashScGrpc] server stopped (grace=%r); worker thread wait_for_termination returns",
                    grace,
                )
            except Exception as e:
                logging.warning("[DashScGrpc] server.stop failed: %s", e, exc_info=True)
            try:
                if self._servicer is not None:
                    self._servicer.close()
            except Exception as e:
                logging.warning("[DashScGrpc] servicer.close failed: %s", e, exc_info=True)
            finally:
                self._server = None
                self._servicer = None
                self._thread = None

    def wait_for_termination(self) -> None:
        """Block on the underlying ``grpc.Server.wait_for_termination``. No-op if not started."""
        server = self._server
        if server is not None:
            server.wait_for_termination()


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
    grpc_server = DashScGrpcServer(dash_sc_grpc_config=dash_sc_cfg)
    grpc_server.start(args.port, backend_visitor=None)
    grpc_server.wait_for_termination()


if __name__ == "__main__":
    main()
