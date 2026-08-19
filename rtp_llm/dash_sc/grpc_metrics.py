"""kmonitor fan-out for the dash_sc gRPC path.

Stateless leaf functions called from each servicer's ``ModelStreamInfer`` —
``report_arrival_priority`` at RPC entry, ``report_chunk`` per response frame,
and ``report_frontend_rpc_done`` / ``report_forwarder_rpc_done`` at the finally
tail (after the access log line is emitted). They mirror the metric family the
HTTP frontend reports in
:class:`rtp_llm.frontend.frontend_server.FrontendServer` (``py_rtp_framework_qps``
/ ``py_rtp_framework_rt`` / ``py_rtp_response_first_token_rt`` / …) so the two
protocols share metric names and dashboards can slice by the ``protocol`` tag.

This module owns no lifecycle and no record construction: it reads a
fully-populated :class:`~rtp_llm.dash_sc.access_record.GrpcAccessRecord` plus the
servicer's ``rank_id`` / ``server_id``, and emits metrics. The record is the
single source of truth; these functions are a thin projection of it onto
kmonitor. The kmonitor tag schema (``protocol`` / ``method`` dimensions
alongside rank/server) is a metrics concern and lives here — see
:func:`_metric_tags`.
"""

from __future__ import annotations

import functools
import time
from typing import Optional

from rtp_llm.dash_sc.access_record import DASH_SC_GRPC_PROTOCOL, GrpcAccessRecord
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor, qos_priority_tag

# The single RPC method both servicers implement — a constant kmonitor dimension.
_METRIC_METHOD = "ModelStreamInfer"


@functools.lru_cache(maxsize=None)
def _metric_tags(rank_id: Optional[int], server_id: Optional[int]) -> dict[str, str]:
    """Project a servicer's rank/server ids onto the kmonitor tag set, memoized.

    The tag set is fixed for a servicer's whole lifetime, so ``lru_cache`` (keyed
    by the two ids) returns the same dict to every report call — the per-token
    hot path (:func:`report_chunk`) never rebuilds it or re-runs ``str()``. Every
    caller treats the dict as read-only (``kmonitor.report`` copies it internally;
    the error / tool-loop branches spread it into a fresh ``{**tags, ...}``), so
    sharing one cached instance is safe.
    """
    return {
        "protocol": DASH_SC_GRPC_PROTOCOL,
        "rank_id": str(rank_id) if rank_id is not None else "",
        "server_id": str(server_id) if server_id is not None else "",
        "method": _METRIC_METHOD,
    }


def _priority_tag(record: GrpcAccessRecord) -> str:
    """The kmonitor ``priority`` tag value for this RPC.

    Reads the qos priority the record captured from the
    ``x-dashscope-inner-qos-level`` chain on the first frame parse
    (``capture_structured_request``); ``"0"`` when the RPC never reached that
    parse (frame-less, parse error) or carried no qos value. Per-request, so it
    is merged into the memoized base tags at report time (``{**tags, ...}``)
    instead of widening the ``_metric_tags`` cache key.
    """
    cfg = record.generate_config
    return qos_priority_tag(cfg.get("qos_priority") if cfg else None)


def report_arrival_priority(
    record: GrpcAccessRecord,
    *,
    rank_id: Optional[int],
    server_id: Optional[int],
) -> None:
    """Emit the priority-tagged arrival QPS, once per RPC.

    Deferred to the first frame parse so it reports the true qos priority.
    RPCs that never reach that parse (frame-less, parse error) are covered by
    the fallback call in :func:`_report_rpc_done_common`, which reports the
    "0" bucket. The ``priority_arrival_reported`` flag ensures exactly one
    tagged arrival per RPC.
    """
    if record.priority_arrival_reported:
        return
    record.priority_arrival_reported = True
    tags = _metric_tags(rank_id, server_id)
    kmonitor.report(
        AccMetrics.QPS_METRIC, 1, {**tags, "priority": _priority_tag(record)}
    )


def report_chunk(
    record: GrpcAccessRecord,
    *,
    rank_id: Optional[int],
    server_id: Optional[int],
    is_first: bool,
    now: Optional[float] = None,
) -> None:
    """Mirror ``_call_generate_with_report`` per-chunk metrics.

    ``is_first`` picks FIRST_TOKEN_RT vs ITER_RT/ITER_QPS — same split as the
    HTTP path. :func:`_metric_tags` returns the memoized tag dict, read-only
    here, so the per-token path allocates nothing.
    """
    tags = _metric_tags(rank_id, server_id)
    if now is None:
        now = time.time()
    if is_first:
        ttfb_ms = (now - record.start_ts) * 1000.0
        kmonitor.report(GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC, ttfb_ms, tags)
    else:
        last = record.last_chunk_ts or record.first_resp_ts or record.start_ts
        iter_rt_ms = (now - last) * 1000.0
        kmonitor.report(GaugeMetrics.RESPONSE_ITER_RT_METRIC, iter_rt_ms, tags)
    kmonitor.report(AccMetrics.ITER_QPS_METRIC, 1, tags)
    record.last_chunk_ts = now


def _report_rpc_done_common(
    record: GrpcAccessRecord,
    *,
    rank_id: Optional[int],
    server_id: Optional[int],
    status: str,
) -> None:
    """Common RPC tail metrics for both dash-sc frontend and forwarder.

    The success / cancel / error QPS counters carry the per-request
    ``priority`` tag (``"0"`` when the RPC had no qos context).
    """
    tags = _metric_tags(rank_id, server_id)
    # Fallback for RPCs that never reached the first frame parse (frame-less /
    # parse error) — no-op when the servicer already reported the true value.
    report_arrival_priority(record, rank_id=rank_id, server_id=server_id)
    priority_tags = {**tags, "priority": _priority_tag(record)}
    total_ms = (time.time() - record.start_ts) * 1000.0
    kmonitor.report(GaugeMetrics.LANTENCY_METRIC, total_ms, tags)
    kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, record.resp_count, tags)
    if status == "OK":
        kmonitor.report(AccMetrics.SUCCESS_QPS_METRIC, 1, priority_tags)
    elif status == "CANCELLED":
        kmonitor.report(AccMetrics.CANCEL_QPS_METRIC, 1, priority_tags)
    else:
        error_code = record.backend_error_code or status
        kmonitor.report(
            AccMetrics.ERROR_QPS_METRIC,
            1,
            {**priority_tags, "error_code": error_code},
        )


def _report_frontend_structured_metrics(
    record: GrpcAccessRecord,
    *,
    rank_id: Optional[int],
    server_id: Optional[int],
) -> None:
    tags = _metric_tags(rank_id, server_id)
    if record.input_len is not None:
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, record.input_len, tags)
    kmonitor.report(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC, record.output_len, tags)
    monitor = record.repetition_monitor
    if monitor.tool_call_loop_check_ms is not None:
        kmonitor.report(
            GaugeMetrics.TOOL_CALL_LOOP_CHECK_RT_METRIC,
            monitor.tool_call_loop_check_ms,
            tags,
        )
    loop = monitor.tool_call_loop_result
    if loop is not None and loop.hit:
        loop_tags = {**tags, "action": "metric"}
        kmonitor.report(AccMetrics.TOOL_CALL_LOOP_QPS_METRIC, 1, loop_tags)
        kmonitor.report(
            GaugeMetrics.TOOL_CALL_LOOP_REPEAT_COUNT_METRIC,
            loop.repeat_count,
            loop_tags,
        )
        kmonitor.report(
            GaugeMetrics.TOOL_CALL_LOOP_CURRENT_SPAN_TOKENS_METRIC,
            loop.current_span_tokens,
            loop_tags,
        )


def report_frontend_rpc_done(
    record: GrpcAccessRecord,
    *,
    rank_id: Optional[int],
    server_id: Optional[int],
    status: str,
) -> None:
    """Mirror ``_call_generate_with_report`` frontend tail metrics."""
    _report_rpc_done_common(record, rank_id=rank_id, server_id=server_id, status=status)
    _report_frontend_structured_metrics(record, rank_id=rank_id, server_id=server_id)


def report_forwarder_rpc_done(
    record: GrpcAccessRecord,
    *,
    rank_id: Optional[int],
    server_id: Optional[int],
    status: str,
) -> None:
    """Forwarder tail metrics: common RPC counters only, no token payload metrics."""
    _report_rpc_done_common(record, rank_id=rank_id, server_id=server_id, status=status)
