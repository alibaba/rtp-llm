"""Access/query log emission for the dash_sc gRPC path.

This module owns log *writing* only — the loggers and the two emit functions.
Lifecycle (when to capture / report / emit) lives inline in each servicer's
``ModelStreamInfer``; the data path lives in
:mod:`rtp_llm.dash_sc.access_record`; status classification lives in
:mod:`rtp_llm.dash_sc.status`; kmonitor fan-out lives in
:mod:`rtp_llm.dash_sc.grpc_metrics`.

Two log channels, both flat one-JSON-line-per-event:

- **access log** — one ``rpc_completed`` record per RPC, written at completion
  (``emit_access_log``). On a long streaming RPC this can be minutes after the
  request arrived.
- **query log** — one arrival breadcrumb per RPC, written at handler entry
  (``emit_query_log``), so ``tail -f`` shows arrivals immediately and link
  latency (forwarder arrival → frontend arrival) is debuggable. Mirrors the HTTP
  frontend convention (``access.log`` vs ``query_access.log`` in
  ``rtp_llm/access_logger/access_logger.py``).

File: ``<log_path>/dash_sc_grpc_access_r{rank}_s{server}.log`` (per-process via
``get_process_log_filename``).

Performance: file I/O is async — ``AsyncRotatingFileHandler.emit`` just
``put_nowait`` on a bounded queue; a background worker thread does the write.
Queue full → drop, never block. So disk latency never affects RPC latency.
"""

from __future__ import annotations

import logging
from typing import Optional

import orjson

from rtp_llm.access_logger.log_utils import get_handler
from rtp_llm.dash_sc.access_record import GrpcAccessRecord, format_access_log_ts

DASH_SC_GRPC_ACCESS_LOGGER_NAME = "dash_sc_grpc_access_logger"
DASH_SC_GRPC_ACCESS_LOG_FILENAME = "dash_sc_grpc_access.log"

DASH_SC_GRPC_QUERY_LOGGER_NAME = "dash_sc_grpc_query_logger"
DASH_SC_GRPC_QUERY_LOG_FILENAME = "dash_sc_grpc_query.log"


def init_dash_sc_grpc_access_logger(
    log_path: str,
    backup_count: int,
    rank_id: Optional[int] = None,
    server_id: Optional[int] = None,
    async_mode: bool = True,
) -> None:
    """Configure the dedicated gRPC access logger. Safe to call multiple times."""
    logger = logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME)
    handler = get_handler(
        DASH_SC_GRPC_ACCESS_LOG_FILENAME,
        log_path,
        backup_count,
        rank_id,
        server_id,
        async_mode,
    )
    logger.handlers.clear()
    logger.parent = None
    logger.setLevel(logging.INFO)
    if handler is not None:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def init_dash_sc_grpc_query_logger(
    log_path: str,
    backup_count: int,
    rank_id: Optional[int] = None,
    server_id: Optional[int] = None,
    async_mode: bool = True,
) -> None:
    """Configure the dedicated gRPC query (arrival) logger. Safe to call multiple times."""
    logger = logging.getLogger(DASH_SC_GRPC_QUERY_LOGGER_NAME)
    handler = get_handler(
        DASH_SC_GRPC_QUERY_LOG_FILENAME,
        log_path,
        backup_count,
        rank_id,
        server_id,
        async_mode,
    )
    logger.handlers.clear()
    logger.parent = None
    logger.setLevel(logging.INFO)
    if handler is not None:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def emit_query_log(
    record: GrpcAccessRecord,
    *,
    rank_id: Optional[int],
    server_id: Optional[int],
) -> None:
    """Write one arrival breadcrumb to the query log — called at handler entry.

    Fires exactly once per RPC, before any inbound body read or backend work, so
    the line hits disk at arrival. No proto-derived fields (``request_id`` /
    ``model_name`` / ``input_len`` / payload details) — those stay in the
    completion access log; the query line shape never depends on first-frame
    content.
    """
    payload = {
        "ts": format_access_log_ts(record.start_ts),
        "arrival_ts_epoch_ms": int(record.start_ts * 1000),
        "server_id": server_id,
        "rank_id": rank_id,
        "method": record.method,
        "stream_type": record.stream_type,
        "peer": record.peer,
        "upstream_request_id": record.upstream_request_id,
        "upstream_request_id_key": record.upstream_request_id_key,
    }
    logger = logging.getLogger(DASH_SC_GRPC_QUERY_LOGGER_NAME)
    try:
        logger.info(orjson.dumps(payload).decode("utf-8"))
    except Exception as e:
        logging.warning("[DashScGrpc] query log emit failed: %s", e)


def emit_access_log(
    record: GrpcAccessRecord,
    *,
    rank_id: Optional[int],
    server_id: Optional[int],
    end_ts: Optional[float] = None,
) -> None:
    """Write one ``rpc_completed`` record to the access log — called at RPC end.

    ``record.resolve_status`` must have run first (status classification). This builds
    the flat schema via ``record.build_record`` and writes one JSON line; a
    ``repetition_alert`` also raises a WARNING so tool-call loops surface in the
    main log without a dashboard.
    """
    payload = record.build_record(server_id, rank_id, end_ts=end_ts)
    if payload.get("repetition_alert"):
        logging.warning(
            "[DashScGrpc] repetition detected request_id=%s status=%s "
            "output_len=%s reason=%s repetition_token_len=%s "
            "tool_call_loop_token_len=%s primary_source=%s "
            "repetition_monitor_impl=%s repetition_monitor_available=%s "
            "repetition_monitor_unavailable_reason=%s tool_call_loop_error=%s",
            payload.get("request_id"),
            payload.get("status"),
            payload.get("output_token_len"),
            payload.get("repetition_reason"),
            payload.get("repetition_token_len"),
            payload.get("tool_call_loop_token_len"),
            payload.get("repetition_primary_source"),
            payload.get("repetition_monitor_impl"),
            payload.get("repetition_monitor_available"),
            payload.get("repetition_monitor_unavailable_reason"),
            payload.get("tool_call_loop_error"),
        )
    logger = logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME)
    try:
        # orjson emits bytes — decode once for the logging framework.
        logger.info(orjson.dumps(payload).decode("utf-8"))
    except Exception as e:
        logging.warning("[DashScGrpc] access log emit failed: %s", e)
