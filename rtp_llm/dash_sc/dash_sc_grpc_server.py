"""
DashSc gRPC process lifecycle: bind ``grpc.Server``, start/stop, daemon thread, CLI ``main``.

Servicer implementation: ``dash_sc_grpc_service.DashScGrpcInferenceServicer``.
Started from FrontendApp via ``start_dash_sc_grpc_server_in_thread`` (daemon thread holds
``wait_for_termination``; caller blocks until ``server.start()`` succeeds).
Standalone fake: ``python -m rtp_llm.dash_sc.dash_sc_grpc_server [--port PORT]``.
"""

from __future__ import annotations

import logging
import threading
from concurrent import futures
from typing import Optional

import grpc

from rtp_llm.dash_sc.dash_sc_grpc_access_log import (
    DASH_SC_GRPC_ACCESS_LOG_FILENAME,
    DashScGrpcAccessLogInterceptor,
    init_dash_sc_grpc_access_logger,
)
from rtp_llm.dash_sc.dash_sc_grpc_enqueue_loop import (
    get_dash_sc_grpc_enqueue_event_loop,
    set_dash_sc_grpc_enqueue_event_loop,
)
from rtp_llm.dash_sc.dash_sc_grpc_pure_forward_service import PureForwardServicer
from rtp_llm.dash_sc.dash_sc_grpc_real_infer import _iter_enqueue_sync
from rtp_llm.dash_sc.dash_sc_grpc_service import DashScGrpcInferenceServicer
from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc


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


_dash_sc_grpc_server: Optional[grpc.Server] = None
_dash_sc_grpc_servicer: Optional[predict_v2_pb2_grpc.GRPCInferenceServiceServicer] = None
_dash_sc_grpc_server_thread: Optional[threading.Thread] = None
_dash_sc_grpc_server_lock = threading.Lock()

# Max time for grpc.Server.start() + bind before start_dash_sc_grpc_server_in_thread returns.
_DEFAULT_DASH_SC_GRPC_STARTUP_TIMEOUT_S = 30.0


def start_dash_sc_grpc_server(
    port: int,
    backend_visitor=None,
    dash_sc_grpc_config=None,
    *,
    ip: str = "",
    server_id: str = "",
    log_path: str = "",
    backup_count: int = 0,
    rank_id: Optional[int] = None,
) -> grpc.Server:
    """Create and start DashSc gRPC server. backend_visitor=None -> fake (mock); else use enqueue.

    ``ip`` / ``server_id`` plus the dash_sc gRPC ``port`` feed ``generate_request_id`` inside
    the servicer so the backend ``GenerateInput.request_id`` follows the same snowflake scheme
    as the HTTP path in ``FrontendServer``.

    ``log_path`` / ``backup_count`` / ``rank_id`` configure the access log file
    (``<log_path>/dash_sc_grpc_access_r{rank}_s{server}.log``). If ``log_path`` is
    empty no file handler is attached — the interceptor still runs so tests see
    logger calls, and production without log config silently drops.
    """
    global _dash_sc_grpc_server, _dash_sc_grpc_servicer
    with _dash_sc_grpc_server_lock:
        if _dash_sc_grpc_server is not None:
            logging.warning("[DashScGrpc] server already started")
            return _dash_sc_grpc_server
        cfg = _resolve_dash_sc_grpc_config(dash_sc_grpc_config)
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
        _dash_sc_grpc_server = server
        _dash_sc_grpc_servicer = servicer
        logging.info(
            "[DashScGrpc] Listening on 0.0.0.0:%s (predict_v2.proto, %s, max_server_workers=%s)",
            port,
            mode,
            pool_size,
        )
        return server


def stop_dash_sc_grpc_server(grace: Optional[float] = None) -> None:
    """Stop the global DashSc ``grpc.Server`` and clear module state (allows a later start).

    ``grace`` is passed to ``grpc.Server.stop`` (seconds to let in-flight RPCs finish); ``None``
    uses grpcio default. Safe to call multiple times or if the server was never started.

    Holds ``_dash_sc_grpc_server_lock`` for the full ``stop()`` so ``start_*`` cannot race on
    the same listen port during shutdown.
    """
    global _dash_sc_grpc_server, _dash_sc_grpc_servicer, _dash_sc_grpc_server_thread
    with _dash_sc_grpc_server_lock:
        server = _dash_sc_grpc_server
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
            if _dash_sc_grpc_servicer is not None:
                _dash_sc_grpc_servicer.close()
        except Exception as e:
            logging.warning("[DashScGrpc] servicer.close failed: %s", e, exc_info=True)
        finally:
            _dash_sc_grpc_server = None
            _dash_sc_grpc_servicer = None
            _dash_sc_grpc_server_thread = None


def start_dash_sc_grpc_server_in_thread(
    port: int,
    backend_visitor=None,
    dash_sc_grpc_config=None,
    *,
    ip: str = "",
    server_id: str = "",
    log_path: str = "",
    backup_count: int = 0,
    rank_id: Optional[int] = None,
    startup_timeout_s: float = _DEFAULT_DASH_SC_GRPC_STARTUP_TIMEOUT_S,
) -> None:
    """Start DashSc gRPC in a daemon thread and block until ``server.start()`` succeeds.

    The daemon thread then runs ``wait_for_termination()`` so the process keeps serving.
    If bind/start fails or does not finish within ``startup_timeout_s``, raises the
    underlying exception or ``TimeoutError`` so callers (e.g. FastAPI startup) do not
    proceed while the port is not actually listening.

    ``ip`` / ``server_id`` are forwarded to ``start_dash_sc_grpc_server`` for request_id
    generation (see that function's docstring).
    """
    global _dash_sc_grpc_server, _dash_sc_grpc_server_thread

    started_ok = threading.Event()
    start_error: list[BaseException] = []

    def _run():
        try:
            server = start_dash_sc_grpc_server(
                port,
                backend_visitor=backend_visitor,
                dash_sc_grpc_config=dash_sc_grpc_config,
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

    _dash_sc_grpc_server_thread = threading.Thread(target=_run, daemon=True)
    _dash_sc_grpc_server_thread.start()
    if not started_ok.wait(timeout=startup_timeout_s):
        raise TimeoutError(
            f"DashSc gRPC server did not finish starting within {startup_timeout_s}s "
            f"(port {port})"
        )
    if start_error:
        raise start_error[0]


def main():
    """Standalone entry (fake mode only). Usage: python -m rtp_llm.dash_sc.dash_sc_grpc_server [--port PORT]"""
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
    server = start_dash_sc_grpc_server(
        args.port, backend_visitor=None, dash_sc_grpc_config=dash_sc_cfg
    )
    server.wait_for_termination()


if __name__ == "__main__":
    main()
