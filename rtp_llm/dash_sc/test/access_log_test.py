"""Unit tests for ``rtp_llm.dash_sc.access_log``.

Covers the four stream-type code paths, content capture, status handling, and
JSON line shape. Does not spin up a real gRPC server — builds ``RpcMethodHandler``
fakes and drives ``DashScGrpcAccessLogInterceptor.intercept_service`` directly.
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Any
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import grpc

from rtp_llm.dash_sc import access_log as access_log_module
from rtp_llm.dash_sc.access_log import (
    DASH_SC_GRPC_ACCESS_LOGGER_NAME,
    DASH_SC_GRPC_QUERY_LOGGER_NAME,
    DashScGrpcAccessLogInterceptor,
    init_dash_sc_grpc_access_logger,
    init_dash_sc_grpc_query_logger,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.metrics import AccMetrics, GaugeMetrics


def _make_infer_request(
    *,
    request_id: str = "trace-123",
    model_name: str = "qwen3-8b",
    input_ids: list[int] | None = None,
    sampling: dict[str, Any] | None = None,
) -> predict_v2_pb2.ModelInferRequest:
    req = predict_v2_pb2.ModelInferRequest()
    req.id = request_id
    req.model_name = model_name
    if input_ids is not None:
        inp = req.inputs.add()
        inp.name = "input_ids"
        inp.datatype = "INT32"
        inp.shape.append(len(input_ids))
        req.raw_input_contents.append(struct.pack(f"<{len(input_ids)}i", *input_ids))
    if sampling:
        for name, val in sampling.items():
            inp = req.inputs.add()
            inp.name = name
            if isinstance(val, int):
                inp.datatype = "INT32"
                inp.shape.append(1)
                req.raw_input_contents.append(struct.pack("<i", val))
            elif isinstance(val, float):
                inp.datatype = "FP32"
                inp.shape.append(1)
                req.raw_input_contents.append(struct.pack("<f", val))
    return req


def _make_stream_response(
    *,
    generated_ids: list[int] | None = None,
    finish_reason: int | None = None,
    prompt_cached_token_num: int | None = None,
) -> predict_v2_pb2.ModelStreamInferResponse:
    resp = predict_v2_pb2.ModelStreamInferResponse()
    infer = resp.infer_response
    if generated_ids is not None:
        out = infer.outputs.add()
        out.name = "generated_ids"
        out.datatype = "INT32"
        out.shape.extend([1, len(generated_ids)])
        raw = (
            struct.pack(f"<{len(generated_ids)}i", *generated_ids)
            if generated_ids
            else b""
        )
        infer.raw_output_contents.append(raw)
    if finish_reason is not None:
        out = infer.outputs.add()
        out.name = "finish_reason"
        out.datatype = "INT64"
        out.shape.append(1)
        infer.raw_output_contents.append(struct.pack("<q", finish_reason))
    if prompt_cached_token_num is not None:
        out = infer.outputs.add()
        out.name = "prompt_cached_token_num"
        out.datatype = "INT32"
        out.shape.append(1)
        infer.raw_output_contents.append(struct.pack("<i", prompt_cached_token_num))
    return resp


class FakeContext:
    def __init__(self, peer: str = "ipv4:1.2.3.4:5678") -> None:
        self._peer = peer
        self._code = grpc.StatusCode.OK
        self._details = ""
        self._active = True
        self._metadata: tuple[tuple[str, str], ...] = ()

    def peer(self) -> str:
        return self._peer

    def code(self):
        return self._code

    def details(self) -> str:
        return self._details

    def is_active(self) -> bool:
        return self._active

    def invocation_metadata(self):
        return self._metadata

    def set_code(self, code) -> None:
        self._code = code

    def set_details(self, details: str) -> None:
        self._details = details

    def set_active(self, active: bool) -> None:
        self._active = active

    def set_metadata(self, metadata) -> None:
        self._metadata = tuple((str(k), str(v)) for k, v in metadata)


def _make_handler(*, request_streaming: bool, response_streaming: bool, inner):
    handler = MagicMock()
    handler.request_streaming = request_streaming
    handler.response_streaming = response_streaming
    handler.request_deserializer = None
    handler.response_serializer = None
    handler.unary_unary = (
        inner if (not request_streaming and not response_streaming) else None
    )
    handler.unary_stream = (
        inner if (not request_streaming and response_streaming) else None
    )
    handler.stream_unary = (
        inner if (request_streaming and not response_streaming) else None
    )
    handler.stream_stream = (
        inner if (request_streaming and response_streaming) else None
    )
    return handler


def _wrapped_behavior(interceptor, handler):
    """Drive intercept_service once; return the behavior callable extracted from the wrapped handler."""
    details = MagicMock()
    details.method = "/test.Service/TestMethod"
    continuation = MagicMock(return_value=handler)

    # Patch the rpc_method_handler factory so we capture the behavior passed in
    captured = {}

    def capture_ss(behavior, **_kw):
        captured["behavior"] = behavior
        return ("ss", behavior)

    def capture_su(behavior, **_kw):
        captured["behavior"] = behavior
        return ("su", behavior)

    def capture_us(behavior, **_kw):
        captured["behavior"] = behavior
        return ("us", behavior)

    def capture_uu(behavior, **_kw):
        captured["behavior"] = behavior
        return ("uu", behavior)

    with patch.object(
        grpc, "stream_stream_rpc_method_handler", side_effect=capture_ss
    ), patch.object(
        grpc, "stream_unary_rpc_method_handler", side_effect=capture_su
    ), patch.object(
        grpc, "unary_stream_rpc_method_handler", side_effect=capture_us
    ), patch.object(
        grpc, "unary_unary_rpc_method_handler", side_effect=capture_uu
    ):
        interceptor.intercept_service(continuation, details)
    return captured["behavior"]


class InitLoggerTest(TestCase):
    def test_init_logger_empty_log_path_is_noop(self) -> None:
        init_dash_sc_grpc_access_logger(log_path="", backup_count=0)
        logger = logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME)
        # No handlers attached when log_path="" (get_handler returns None)
        self.assertEqual(len(logger.handlers), 0)

    def test_init_logger_sets_log_level(self) -> None:
        init_dash_sc_grpc_access_logger(log_path="", backup_count=0)
        logger = logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME)
        self.assertEqual(logger.level, logging.INFO)


class InterceptorTestBase(TestCase):
    def setUp(self) -> None:
        # Patch the module-level ``kmonitor`` for ALL interceptor tests — not just the
        # dedicated KMonitorReportTest below. Otherwise MetricReporter.report would log
        # "no metric named ..." warnings during every access-log test run (the reporter
        # is only initialized inside DashScApp.start, not in the servicer tests).
        # KMonitorReportTest extends this and uses ``self.kmon_calls`` for its assertions.
        self.kmon_patch = patch.object(access_log_module, "kmonitor")
        self.mock_kmon = self.kmon_patch.start()
        self.kmon_calls: list[tuple[Any, float, dict]] = []

        def _record(metric, value=1, tags=None):
            self.kmon_calls.append((metric, value, dict(tags or {})))

        self.mock_kmon.report.side_effect = _record

        self.interceptor = DashScGrpcAccessLogInterceptor(rank_id=0, server_id=1)
        self.records: list[dict[str, Any]] = []
        self.query_records: list[dict[str, Any]] = []

        def capture(msg, *args, **_kw):
            # Logger.info is called with a pre-formatted JSON string; parse it
            rendered = msg % args if args else msg
            self.records.append(json.loads(rendered))

        def capture_query(msg, *args, **_kw):
            rendered = msg % args if args else msg
            self.query_records.append(json.loads(rendered))

        self.logger_patch = patch.object(
            self.interceptor._logger, "info", side_effect=capture
        )
        self.logger_patch.start()
        self.query_logger_patch = patch.object(
            self.interceptor._query_logger, "info", side_effect=capture_query
        )
        self.query_logger_patch.start()

    def tearDown(self) -> None:
        self.query_logger_patch.stop()
        self.logger_patch.stop()
        self.kmon_patch.stop()


class UnaryUnaryTest(InterceptorTestBase):
    def test_unary_happy_path(self) -> None:
        def inner(request, context):
            return _make_stream_response(
                generated_ids=[1, 2, 3], finish_reason=0, prompt_cached_token_num=4
            )

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)

        ctx = FakeContext()
        request = _make_infer_request(
            input_ids=[10, 20, 30], sampling={"max_new_tokens": 32, "top_p": 0.8}
        )
        resp = behavior(request, ctx)

        self.assertIsNotNone(resp)
        self.assertEqual(len(self.records), 1)
        rec = self.records[0]
        self.assertEqual(rec["method"], "/test.Service/TestMethod")
        self.assertEqual(rec["stream_type"], "unary")
        self.assertEqual(rec["capture_mode"], "struct")
        self.assertEqual(rec["peer"], "ipv4:1.2.3.4:5678")
        self.assertEqual(rec["req_count"], 1)
        self.assertEqual(rec["resp_count"], 1)
        self.assertEqual(rec["status"], "OK")
        self.assertIsNone(rec["status_detail"])
        self.assertEqual(rec["request_id"], "trace-123")
        self.assertEqual(rec["model_name"], "qwen3-8b")
        self.assertEqual(rec["input_ids"], [10, 20, 30])
        self.assertEqual(rec["input_len"], 3)
        self.assertEqual(rec["generate_config"]["max_new_tokens"], 32)
        self.assertAlmostEqual(rec["generate_config"]["top_p"], 0.8, places=5)
        self.assertEqual(rec["generated_ids"], [1, 2, 3])
        self.assertEqual(rec["output_len"], 3)
        self.assertEqual(rec["finish_reason"], 0)
        self.assertEqual(rec["prompt_cached_token_num"], 4)
        self.assertEqual(rec["server_id"], 1)
        self.assertEqual(rec["rank_id"], 0)
        self.assertLessEqual(rec["latency_ttfb_ms"], rec["latency_total_ms"])
        self.assertLess(rec["latency_total_ms"] - rec["latency_ttfb_ms"], 50.0)

    def test_unary_without_input_ids_tensor(self) -> None:
        """Health-check style RPC with no input_ids tensor: content fields null but trafic fine."""

        def inner(request, context):
            return _make_stream_response()  # empty response

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        request = _make_infer_request(input_ids=None)

        behavior(request, ctx)
        rec = self.records[0]
        self.assertIsNone(rec["input_ids"])
        self.assertIsNone(rec["input_len"])
        self.assertEqual(rec["output_len"], 0)
        self.assertIsNone(rec["generated_ids"])
        self.assertIsNone(rec["finish_reason"])

    def test_unary_error_propagates_and_logs(self) -> None:
        def inner(request, context):
            raise RuntimeError("boom")

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        request = _make_infer_request(input_ids=[1])

        with self.assertRaises(RuntimeError):
            behavior(request, ctx)
        self.assertEqual(len(self.records), 1)
        rec = self.records[0]
        self.assertEqual(rec["status"], "UNKNOWN_RuntimeError")
        self.assertIn("boom", rec["status_detail"])


class BidiStreamTest(InterceptorTestBase):
    def test_bidi_happy_path_accumulates_generated_ids(self) -> None:
        def inner(request_iterator, context):
            list(request_iterator)  # consume
            yield _make_stream_response(generated_ids=[10], prompt_cached_token_num=8)
            yield _make_stream_response(generated_ids=[20, 30])
            yield _make_stream_response(generated_ids=[40], finish_reason=0)

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        request = _make_infer_request(input_ids=[1, 2, 3])

        out = list(behavior(iter([request]), ctx))
        self.assertEqual(len(out), 3)
        self.assertEqual(len(self.records), 1)
        rec = self.records[0]
        self.assertEqual(rec["stream_type"], "bidi_stream")
        self.assertEqual(rec["req_count"], 1)
        self.assertEqual(rec["resp_count"], 3)
        self.assertEqual(rec["generated_ids"], [10, 20, 30, 40])
        self.assertEqual(rec["output_len"], 4)
        self.assertEqual(rec["finish_reason"], 0)
        self.assertEqual(rec["prompt_cached_token_num"], 8)
        self.assertIsNotNone(rec["latency_ttfb_ms"])
        self.assertLessEqual(rec["latency_ttfb_ms"], rec["latency_total_ms"])

    def test_bidi_cancelled(self) -> None:
        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[10])
            context.set_code(grpc.StatusCode.CANCELLED)
            context.set_details("client cancelled")

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        request = _make_infer_request(input_ids=[1])

        list(behavior(iter([request]), ctx))
        rec = self.records[0]
        self.assertEqual(rec["status"], "CANCELLED")
        self.assertEqual(rec["status_detail"], "client cancelled")
        self.assertEqual(rec["generated_ids"], [10])

    def test_bidi_empty_generated_ids_chunk_not_polluting(self) -> None:
        """Empty generated_ids chunk (shape=[1,0] with 4-byte filler) must not pollute the accumulator.

        ``codec._append_generated_ids_output`` writes
        ``struct.pack("<i", 0)`` when the delta is empty (e.g., a finish-only
        chunk). Without shape-aware decoding this would become a bogus ``[0]``.
        """

        def empty_generated_chunk() -> predict_v2_pb2.ModelStreamInferResponse:
            resp = predict_v2_pb2.ModelStreamInferResponse()
            infer = resp.infer_response
            out = infer.outputs.add()
            out.name = "generated_ids"
            out.datatype = "INT32"
            out.shape.extend([1, 0])
            infer.raw_output_contents.append(struct.pack("<i", 0))  # filler
            out = infer.outputs.add()
            out.name = "finish_reason"
            out.datatype = "INT64"
            out.shape.append(1)
            infer.raw_output_contents.append(struct.pack("<q", 0))
            return resp

        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[42, 43])
            yield empty_generated_chunk()  # heartbeat-ish: no new tokens
            yield _make_stream_response(generated_ids=[44], finish_reason=0)

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        request = _make_infer_request(input_ids=[1])

        list(behavior(iter([request]), ctx))
        rec = self.records[0]
        self.assertEqual(rec["generated_ids"], [42, 43, 44])  # no spurious 0
        self.assertEqual(rec["output_len"], 3)
        self.assertEqual(rec["resp_count"], 3)
        self.assertEqual(rec["finish_reason"], 0)

    def test_bidi_error_mid_stream(self) -> None:
        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[10])
            raise RuntimeError("downstream died")

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        request = _make_infer_request(input_ids=[1])

        got = []
        with self.assertRaises(RuntimeError):
            for r in behavior(iter([request]), ctx):
                got.append(r)
        self.assertEqual(len(got), 1)
        rec = self.records[0]
        self.assertEqual(rec["status"], "UNKNOWN_RuntimeError")
        self.assertIn("downstream died", rec["status_detail"])
        self.assertEqual(rec["exc_type"], "RuntimeError")
        self.assertEqual(rec["generated_ids"], [10])

    def test_bidi_grpc_rpc_error_populates_diagnostics(self) -> None:
        """Empty-iterator ``grpc.RpcError`` path — reproduces the production log
        entries where ``status_detail == 'RpcError()'`` lost both the concrete
        subclass and the gRPC status code. The three diagnostic fields must
        preserve enough signal to tell peer cancel apart from real errors."""

        class _RendezvousLike(grpc.RpcError):
            pass

        def inner(request_iterator, context):
            for _ in request_iterator:
                pass
            raise _RendezvousLike()
            yield  # pragma: no cover (coerce to generator for stream_stream handler)

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        ctx.set_code(grpc.StatusCode.CANCELLED)
        ctx.set_active(False)

        with self.assertRaises(_RendezvousLike):
            list(behavior(iter([]), ctx))

        rec = self.records[0]
        # Empty-iterator + bare ``grpc.RpcError`` subclass (no ``details()``,
        # no ``code()``) → ``_is_bare_peer_closed`` claims it and routes to
        # CANCELLED. The three diagnostic fields below still carry the signal
        # needed to tell peer-cancel apart from a real backend error.
        self.assertEqual(rec["status"], "CANCELLED")
        self.assertEqual(rec["exc_type"], "_RendezvousLike")
        self.assertEqual(rec["context_code"], "CANCELLED")
        self.assertFalse(rec["context_active"])
        self.assertEqual(rec["req_count"], 0)
        self.assertEqual(rec["resp_count"], 0)


class ServerStreamTest(InterceptorTestBase):
    def test_unary_stream(self) -> None:
        def inner(request, context):
            yield _make_stream_response(generated_ids=[7])
            yield _make_stream_response(generated_ids=[8], finish_reason=0)

        handler = _make_handler(
            request_streaming=False, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        request = _make_infer_request(input_ids=[1])

        list(behavior(request, ctx))
        rec = self.records[0]
        self.assertEqual(rec["stream_type"], "server_stream")
        self.assertEqual(rec["req_count"], 1)
        self.assertEqual(rec["resp_count"], 2)
        self.assertEqual(rec["generated_ids"], [7, 8])


class JsonShapeTest(InterceptorTestBase):
    def test_json_line_is_single_line_compact(self) -> None:
        raw_emissions: list[str] = []

        def capture(msg, *args, **_kw):
            raw_emissions.append(msg % args if args else msg)

        self.interceptor._logger.info.side_effect = capture  # override the test base

        def inner(request, context):
            return _make_stream_response(generated_ids=[1, 2])

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        behavior(_make_infer_request(input_ids=[1]), ctx)

        self.assertEqual(len(raw_emissions), 1)
        line = raw_emissions[0]
        self.assertNotIn("\n", line)
        # Compact separator: no ", " / ": " with spaces
        self.assertNotIn(", ", line)
        self.assertNotIn(": ", line)
        parsed = json.loads(line)
        self.assertIn("ts", parsed)
        self.assertIn("method", parsed)


class KMonitorReportTest(InterceptorTestBase):
    """Verify the interceptor fans lifecycle events out to kmonitor with ``protocol=grpc``.

    The base class already patches ``access_log.kmonitor`` and records calls into
    ``self.kmon_calls`` — we only assert on that here. The intent is to prove parity with
    the HTTP path in ``FrontendServer``: same metric names, same rank_id/server_id tag
    shape, plus ``protocol=grpc`` / ``method`` so dashboards can split the two transports.
    """

    def _metrics_called(self) -> list[Any]:
        return [c[0] for c in self.kmon_calls]

    def _calls_for(self, metric) -> list[tuple[Any, float, dict]]:
        return [c for c in self.kmon_calls if c[0] == metric]

    def _assert_base_tags(self, tags: dict, *, method: str) -> None:
        self.assertEqual(tags["protocol"], "grpc")
        self.assertEqual(tags["rank_id"], "0")
        self.assertEqual(tags["server_id"], "1")
        self.assertEqual(tags["method"], method)

    def test_unary_happy_path_fires_arrival_chunk_success_and_gauges(self) -> None:
        def inner(request, context):
            return _make_stream_response(
                generated_ids=[1, 2, 3], finish_reason=0, prompt_cached_token_num=4
            )

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        behavior(_make_infer_request(input_ids=[10, 20, 30]), ctx)

        method = "/test.Service/TestMethod"
        metrics = self._metrics_called()
        # Arrival QPS reported exactly once.
        self.assertEqual(metrics.count(AccMetrics.QPS_METRIC), 1)
        # Single chunk -> first-token RT + one iter QPS.
        self.assertEqual(metrics.count(GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC), 1)
        self.assertNotIn(GaugeMetrics.RESPONSE_ITER_RT_METRIC, metrics)
        self.assertEqual(metrics.count(AccMetrics.ITER_QPS_METRIC), 1)
        # Tail metrics: success + latency + iterate_count + token sizes.
        self.assertEqual(metrics.count(AccMetrics.SUCCESS_QPS_METRIC), 1)
        self.assertEqual(metrics.count(AccMetrics.ERROR_QPS_METRIC), 0)
        self.assertEqual(metrics.count(AccMetrics.CANCEL_QPS_METRIC), 0)
        self.assertEqual(metrics.count(GaugeMetrics.LANTENCY_METRIC), 1)
        self.assertEqual(metrics.count(GaugeMetrics.RESPONSE_ITERATE_COUNT), 1)
        self.assertEqual(metrics.count(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC), 1)
        self.assertEqual(metrics.count(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC), 1)

        # Every call carries protocol=grpc + rank/server/method.
        for _, _, tags in self.kmon_calls:
            self._assert_base_tags(tags, method=method)

        # Token-size gauges carry the right values.
        _, input_len, _ = self._calls_for(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC)[0]
        self.assertEqual(input_len, 3)
        _, output_len, _ = self._calls_for(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC)[0]
        self.assertEqual(output_len, 3)
        _, iterate_count, _ = self._calls_for(GaugeMetrics.RESPONSE_ITERATE_COUNT)[0]
        self.assertEqual(iterate_count, 1)

    def test_server_stream_fires_first_then_iter_rt_per_chunk(self) -> None:
        def inner(request, context):
            yield _make_stream_response(generated_ids=[7])
            yield _make_stream_response(generated_ids=[8])
            yield _make_stream_response(generated_ids=[9], finish_reason=0)

        handler = _make_handler(
            request_streaming=False, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        list(behavior(_make_infer_request(input_ids=[1]), ctx))

        # First chunk -> FIRST_TOKEN_RT; remaining 2 -> ITER_RT. Three total iter QPS.
        self.assertEqual(
            len(self._calls_for(GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC)), 1
        )
        self.assertEqual(len(self._calls_for(GaugeMetrics.RESPONSE_ITER_RT_METRIC)), 2)
        self.assertEqual(len(self._calls_for(AccMetrics.ITER_QPS_METRIC)), 3)
        # Success on normal exit, no error/cancel.
        self.assertEqual(len(self._calls_for(AccMetrics.SUCCESS_QPS_METRIC)), 1)
        self.assertEqual(len(self._calls_for(AccMetrics.ERROR_QPS_METRIC)), 0)
        # Iterate count = chunks seen.
        _, iter_count, _ = self._calls_for(GaugeMetrics.RESPONSE_ITERATE_COUNT)[0]
        self.assertEqual(iter_count, 3)

    def test_cancelled_rpc_fires_cancel_qps_not_error(self) -> None:
        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[10])
            context.set_code(grpc.StatusCode.CANCELLED)
            context.set_details("client cancelled")

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        list(behavior(iter([_make_infer_request(input_ids=[1])]), ctx))

        self.assertEqual(len(self._calls_for(AccMetrics.CANCEL_QPS_METRIC)), 1)
        self.assertEqual(len(self._calls_for(AccMetrics.ERROR_QPS_METRIC)), 0)
        self.assertEqual(len(self._calls_for(AccMetrics.SUCCESS_QPS_METRIC)), 0)
        # Cancel tag set doesn't include error_code (CANCELLED is its own metric).
        _, _, cancel_tags = self._calls_for(AccMetrics.CANCEL_QPS_METRIC)[0]
        self.assertNotIn("error_code", cancel_tags)

    def test_exception_mid_rpc_fires_error_qps_with_error_code_tag(self) -> None:
        def inner(request, context):
            raise RuntimeError("boom")

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        with self.assertRaises(RuntimeError):
            behavior(_make_infer_request(input_ids=[1]), ctx)

        self.assertEqual(len(self._calls_for(AccMetrics.ERROR_QPS_METRIC)), 1)
        self.assertEqual(len(self._calls_for(AccMetrics.SUCCESS_QPS_METRIC)), 0)
        _, _, error_tags = self._calls_for(AccMetrics.ERROR_QPS_METRIC)[0]
        # Exception with no grpc code -> status "UNKNOWN_<ExcClass>" lands in
        # error_code tag so alert rules can split crash-class categories
        # (RuntimeError / ValueError / …) without re-reading log lines.
        self.assertEqual(error_tags["error_code"], "UNKNOWN_RuntimeError")
        self.assertEqual(error_tags["protocol"], "grpc")

    def test_generator_exit_routes_to_cancel_qps_not_error(self) -> None:
        """Client disconnect mid-stream -> framework ``gen.close()`` injects
        ``GeneratorExit`` at the wrapper's ``yield``. This is benign (not a
        real RPC failure), so it must hit CANCEL_QPS, not ERROR_QPS. The log
        line's ``status`` must be distinct from the generic ``UNKNOWN`` bucket
        so operators can tell client-gone apart from actual errors at a glance.
        """

        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[10])
            yield _make_stream_response(generated_ids=[20])

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()

        gen = behavior(iter([_make_infer_request(input_ids=[1])]), ctx)
        # Drain one chunk, then simulate gRPC closing our generator because
        # the client handler terminated. ``gen.close()`` injects GeneratorExit
        # at the suspended ``yield resp`` and swallows it on the way out; the
        # finally block still runs _finalize with the exception.
        next(gen)
        gen.close()

        self.assertEqual(len(self._calls_for(AccMetrics.CANCEL_QPS_METRIC)), 1)
        self.assertEqual(len(self._calls_for(AccMetrics.ERROR_QPS_METRIC)), 0)
        rec = self.records[0]
        self.assertEqual(rec["status"], "CANCELLED")
        self.assertEqual(rec["status_detail"], "client closed generator")
        self.assertEqual(rec["exc_type"], "GeneratorExit")

    def test_client_request_iter_failure_routes_to_cancel_qps(self) -> None:
        """grpcio ``_MultiThreadedRendezvous`` with
        ``details='Exception iterating requests!'`` — client disappeared
        before sending a frame — must route to CANCEL_QPS, not ERROR_QPS.
        Signature on the log line is ``latency_total_ms<1 / req_count=0 /
        resp_count=0``, which is not a timeout despite status bucket
        previously being the generic UNKNOWN.
        """

        class _FakeRendezvous(grpc.RpcError):
            def details(self) -> str:
                return "Exception iterating requests!"

            def code(self):
                return grpc.StatusCode.UNKNOWN

        def inner(request_iterator, context):
            # The forwarder's upstream stub call surfaces the error on the
            # response-iteration side — mimic that shape here.
            for _ in request_iterator:
                pass
            raise _FakeRendezvous()
            yield  # pragma: no cover

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()

        with self.assertRaises(_FakeRendezvous):
            list(behavior(iter([]), ctx))

        self.assertEqual(len(self._calls_for(AccMetrics.CANCEL_QPS_METRIC)), 1)
        self.assertEqual(len(self._calls_for(AccMetrics.ERROR_QPS_METRIC)), 0)
        rec = self.records[0]
        self.assertEqual(rec["status"], "CANCELLED")
        self.assertEqual(rec["status_detail"], "client request iterator failed")
        self.assertEqual(rec["exc_type"], "_FakeRendezvous")

    def test_rpc_without_input_ids_omits_input_token_size_gauge(self) -> None:
        """Health-check style RPC (no input_ids) must not emit a bogus 0-length gauge."""

        def inner(request, context):
            return _make_stream_response()

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        behavior(_make_infer_request(input_ids=None), FakeContext())

        self.assertEqual(len(self._calls_for(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC)), 0)
        # OUTPUT size is always reported (len of empty list = 0).
        _, out_len, _ = self._calls_for(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC)[0]
        self.assertEqual(out_len, 0)


class DownstreamDiagnosticsTest(InterceptorTestBase):
    """Forward-path diagnostics (``downstream_addr`` / ``downstream_resp_count`` /
    ``buffered_stage``) must round-trip onto the access-log line.

    These fields are populated by :class:`DashScProxyServicer` via the
    ``context._dash_sc_access_agg`` hook the interceptor installs; here we
    simulate that write-back and assert the record ends up with the right
    shape regardless of capture mode.
    """

    def test_context_gets_aggregate_attached(self) -> None:
        """Interceptor must attach the aggregate to the context for servicer readback."""

        captured: dict[str, Any] = {}

        def inner(request, context):
            captured["agg"] = getattr(context, "_dash_sc_access_agg", None)
            return _make_stream_response(generated_ids=[1])

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        behavior(_make_infer_request(input_ids=[1]), FakeContext())
        self.assertIsNotNone(captured["agg"])
        # The aggregate exposes the writable diagnostics attributes.
        self.assertTrue(hasattr(captured["agg"], "downstream_addr"))
        self.assertTrue(hasattr(captured["agg"], "downstream_resp_count"))
        self.assertTrue(hasattr(captured["agg"], "buffered_stage"))

    def test_defaults_emitted_when_no_forwarder_writes(self) -> None:
        """Struct-mode path doesn't touch diagnostics — record still has the fields."""

        def inner(request, context):
            return _make_stream_response(generated_ids=[1])

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        behavior(_make_infer_request(input_ids=[1]), FakeContext())
        rec = self.records[0]
        self.assertIsNone(rec["downstream_addr"])
        self.assertEqual(rec["downstream_resp_count"], 0)
        self.assertIsNone(rec["buffered_stage"])

    def test_forwarder_writes_round_trip_to_record(self) -> None:
        """Simulate the forwarder writing diag fields — record mirrors them verbatim."""

        def inner(request, context):
            agg = context._dash_sc_access_agg
            agg.downstream_addr = "10.0.0.7:9000"
            agg.downstream_resp_count = 42
            agg.buffered_stage = "dropped_buffered_on_exception"
            return _make_stream_response(generated_ids=[1])

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        behavior(_make_infer_request(input_ids=[1]), FakeContext())
        rec = self.records[0]
        self.assertEqual(rec["downstream_addr"], "10.0.0.7:9000")
        self.assertEqual(rec["downstream_resp_count"], 42)
        self.assertEqual(rec["buffered_stage"], "dropped_buffered_on_exception")


class RawModeInterceptorTestBase(InterceptorTestBase):
    """Mirror of InterceptorTestBase with ``raw_mode=True``.

    Shares the kmonitor patching, logger capture, and record list, but flips the
    servicer into raw-mode: every request message and every response message is
    dumped into the record as a decoded proto dict instead of struct fields.
    """

    def setUp(self) -> None:
        super().setUp()
        # Rebuild the interceptor in raw mode and re-patch its logger.
        self.logger_patch.stop()
        self.interceptor = DashScGrpcAccessLogInterceptor(
            rank_id=0, server_id=1, raw_mode=True
        )

        def capture(msg, *args, **_kw):
            rendered = msg % args if args else msg
            self.records.append(json.loads(rendered))

        self.logger_patch = patch.object(
            self.interceptor._logger, "info", side_effect=capture
        )
        self.logger_patch.start()


class RawModeBidiTest(RawModeInterceptorTestBase):
    """Forward-servicer path: dump every request/response proto verbatim."""

    def test_bidi_raw_mode_captures_every_request_and_response(self) -> None:
        def inner(request_iterator, context):
            for _ in request_iterator:
                pass
            yield _make_stream_response(generated_ids=[10])
            yield _make_stream_response(generated_ids=[20, 30], finish_reason=0)

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        req1 = _make_infer_request(request_id="r1", input_ids=[1, 2, 3])
        req2 = _make_infer_request(request_id="r2", input_ids=[4, 5])

        list(behavior(iter([req1, req2]), ctx))
        self.assertEqual(len(self.records), 1)
        rec = self.records[0]

        # Record shape flags raw capture.
        self.assertEqual(rec["capture_mode"], "raw")
        self.assertEqual(rec["req_count"], 2)
        self.assertEqual(rec["resp_count"], 2)
        # Every request captured into raw_requests.
        self.assertEqual(len(rec["raw_requests"]), 2)
        self.assertEqual(rec["raw_requests"][0]["id"], "r1")
        self.assertEqual(rec["raw_requests"][1]["id"], "r2")
        # Tensor base64 decoded to integer list.
        self.assertEqual(rec["raw_requests"][0]["inputs"][0]["decoded"], [1, 2, 3])
        # Every response captured.
        self.assertEqual(len(rec["raw_responses"]), 2)
        gen_ids_0 = rec["raw_responses"][0]["infer_response"]["outputs"][0]
        self.assertEqual(gen_ids_0["decoded"], [10])
        # Truncation flags default False.
        self.assertFalse(rec["raw_requests_truncated"])
        self.assertFalse(rec["raw_responses_truncated"])
        # Struct fields are absent in raw mode.
        self.assertNotIn("input_ids", rec)
        self.assertNotIn("generated_ids", rec)
        self.assertNotIn("generate_config", rec)

    def test_raw_mode_caps_chunks_and_sets_truncated_flag(self) -> None:
        """Exceeding ``_RAW_MAX_CHUNKS`` drops extras and flips the truncated flag."""
        original_cap = access_log_module._RAW_MAX_CHUNKS
        access_log_module._RAW_MAX_CHUNKS = 3
        try:

            def inner(request_iterator, context):
                for _ in request_iterator:
                    pass
                for i in range(5):
                    yield _make_stream_response(generated_ids=[i])

            handler = _make_handler(
                request_streaming=True, response_streaming=True, inner=inner
            )
            behavior = _wrapped_behavior(self.interceptor, handler)
            ctx = FakeContext()
            list(behavior(iter([_make_infer_request(input_ids=[1])]), ctx))
            rec = self.records[0]
            self.assertEqual(len(rec["raw_responses"]), 3)
            self.assertTrue(rec["raw_responses_truncated"])
            # req_count still counts every message; only capture is capped.
            self.assertEqual(rec["resp_count"], 5)
        finally:
            access_log_module._RAW_MAX_CHUNKS = original_cap

    def test_raw_mode_caps_per_tensor_decoded_elements(self) -> None:
        """Large tensors keep only the head and flag ``decoded_truncated``."""
        original_cap = access_log_module._RAW_MAX_DECODED_ELEMENTS
        access_log_module._RAW_MAX_DECODED_ELEMENTS = 4
        try:

            def inner(request_iterator, context):
                for _ in request_iterator:
                    pass
                yield _make_stream_response(generated_ids=[1, 2, 3, 4, 5, 6, 7, 8])

            handler = _make_handler(
                request_streaming=True, response_streaming=True, inner=inner
            )
            behavior = _wrapped_behavior(self.interceptor, handler)
            ctx = FakeContext()
            big = list(range(10))
            list(behavior(iter([_make_infer_request(input_ids=big)]), ctx))
            rec = self.records[0]
            # Input tensor truncated.
            t_in = rec["raw_requests"][0]["inputs"][0]
            self.assertEqual(t_in["decoded"], [0, 1, 2, 3])
            self.assertTrue(t_in["decoded_truncated"])
            self.assertEqual(t_in["decoded_total"], 10)
            # Output tensor truncated.
            t_out = rec["raw_responses"][0]["infer_response"]["outputs"][0]
            self.assertEqual(t_out["decoded"], [1, 2, 3, 4])
            self.assertTrue(t_out["decoded_truncated"])
            self.assertEqual(t_out["decoded_total"], 8)
        finally:
            access_log_module._RAW_MAX_DECODED_ELEMENTS = original_cap


class CorrelationHeaderTest(InterceptorTestBase):
    """Upstream correlation ID (``x-dashscope-request-id`` etc.) must round-trip
    onto the access-log line, populated from ``context.invocation_metadata()``
    at RPC arrival so it's present even when ``req_count=0``.
    """

    def test_dashscope_request_id_lands_on_record(self) -> None:
        def inner(request, context):
            return _make_stream_response(generated_ids=[1])

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        ctx.set_metadata([("x-dashscope-request-id", "corr-abc-42")])
        behavior(_make_infer_request(input_ids=[1]), ctx)

        rec = self.records[0]
        self.assertEqual(rec["upstream_request_id"], "corr-abc-42")
        self.assertEqual(rec["upstream_request_id_key"], "x-dashscope-request-id")

    def test_no_metadata_leaves_upstream_fields_null(self) -> None:
        def inner(request, context):
            return _make_stream_response(generated_ids=[1])

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        behavior(_make_infer_request(input_ids=[1]), FakeContext())

        rec = self.records[0]
        self.assertIsNone(rec["upstream_request_id"])
        self.assertIsNone(rec["upstream_request_id_key"])

    def test_header_match_is_case_insensitive(self) -> None:
        def inner(request, context):
            return _make_stream_response(generated_ids=[1])

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        ctx.set_metadata([("X-DashScope-Request-Id", "CaseMixed-7")])
        behavior(_make_infer_request(input_ids=[1]), ctx)

        rec = self.records[0]
        self.assertEqual(rec["upstream_request_id"], "CaseMixed-7")
        self.assertEqual(rec["upstream_request_id_key"], "x-dashscope-request-id")

    def test_priority_dashscope_over_generic_request_id(self) -> None:
        """When both headers present, dashscope-specific wins over generic
        ``x-request-id`` — this is the operational ask: keep the dashscope
        ID attached to the RPC when available so three-layer correlation
        reaches its actual origin."""

        def inner(request, context):
            return _make_stream_response(generated_ids=[1])

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        ctx.set_metadata(
            [("x-request-id", "generic-1"), ("x-dashscope-request-id", "ds-2")]
        )
        behavior(_make_infer_request(input_ids=[1]), ctx)

        rec = self.records[0]
        self.assertEqual(rec["upstream_request_id"], "ds-2")
        self.assertEqual(rec["upstream_request_id_key"], "x-dashscope-request-id")

    def test_traceparent_used_as_fallback(self) -> None:
        """When no dashscope/x-request-id, fall back to W3C ``traceparent`` so
        gRPC shares a correlation convention with the HTTP path (flexlb's
        ``HttpHeaderNames.TRACE_PARENT``)."""
        tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

        def inner(request, context):
            return _make_stream_response(generated_ids=[1])

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        ctx.set_metadata([("traceparent", tp)])
        behavior(_make_infer_request(input_ids=[1]), ctx)

        rec = self.records[0]
        self.assertEqual(rec["upstream_request_id"], tp)
        self.assertEqual(rec["upstream_request_id_key"], "traceparent")

    def test_correlation_id_captured_when_no_request_frame_arrives(self) -> None:
        """The key production scenario: peer closes before any body frame,
        so the proto ``id``-derived ``request_id`` stays null — but the
        metadata-sourced ``upstream_request_id`` must still make it to the
        log line. Without this the ``req_count=0`` access-log entry has no
        correlation handle to the upstream dashscope-serving request."""

        class _BarePeerGone(grpc.RpcError):
            pass

        def inner(request_iterator, context):
            for _ in request_iterator:
                pass
            raise _BarePeerGone()
            yield  # pragma: no cover

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        ctx.set_metadata([("x-dashscope-request-id", "prod-xyz")])

        with self.assertRaises(_BarePeerGone):
            list(behavior(iter([]), ctx))

        rec = self.records[0]
        self.assertEqual(rec["req_count"], 0)
        self.assertIsNone(rec["request_id"])
        self.assertEqual(rec["upstream_request_id"], "prod-xyz")
        self.assertEqual(rec["upstream_request_id_key"], "x-dashscope-request-id")


class BarePeerClosedTest(InterceptorTestBase):
    """Content-less ``grpc.RpcError()`` on a frame-less RPC → CANCELLED, not
    UNKNOWN. Backend-frontend view of the same peer-closed shape that the
    forwarder sees as ``"Exception iterating requests!"``.
    """

    def test_bare_rpc_error_req_count_zero_routes_to_cancel(self) -> None:
        def inner(request_iterator, context):
            for _ in request_iterator:
                pass
            raise grpc.RpcError()
            yield  # pragma: no cover

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()

        with self.assertRaises(grpc.RpcError):
            list(behavior(iter([]), ctx))

        rec = self.records[0]
        self.assertEqual(rec["status"], "CANCELLED")
        self.assertEqual(rec["status_detail"], "peer closed before request arrived")
        self.assertEqual(rec["req_count"], 0)
        self.assertEqual(rec["resp_count"], 0)

        cancel_metric = [
            c for c in self.kmon_calls if c[0] == AccMetrics.CANCEL_QPS_METRIC
        ]
        error_metric = [
            c for c in self.kmon_calls if c[0] == AccMetrics.ERROR_QPS_METRIC
        ]
        self.assertEqual(len(cancel_metric), 1)
        self.assertEqual(len(error_metric), 0)

    def test_bare_rpc_error_after_request_received_still_routes_to_error(self) -> None:
        """Bare ``RpcError`` with ``req_count>=1`` is real server-side
        failure — must stay in ERROR bucket. Keeps the CANCELLED branch
        narrowly scoped to the frame-less peer-gone shape."""

        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[1])
            raise grpc.RpcError()

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()

        with self.assertRaises(grpc.RpcError):
            list(behavior(iter([_make_infer_request(input_ids=[1])]), ctx))

        rec = self.records[0]
        # Bare ``RpcError`` (no code, no details) with a frame already received
        # carries the subclass name through so Grafana can tell RpcError from
        # ValueError from RuntimeError without reading the log line.
        self.assertEqual(rec["status"], "UNKNOWN_RpcError")
        self.assertEqual(rec["req_count"], 1)

        cancel_metric = [
            c for c in self.kmon_calls if c[0] == AccMetrics.CANCEL_QPS_METRIC
        ]
        error_metric = [
            c for c in self.kmon_calls if c[0] == AccMetrics.ERROR_QPS_METRIC
        ]
        self.assertEqual(len(cancel_metric), 0)
        self.assertEqual(len(error_metric), 1)


class RpcErrorWithCodeTest(InterceptorTestBase):
    """Before this fix every ``_MultiThreadedRendezvous`` on the forwarder
    path dumped into ``error_code=UNKNOWN``. Now the real gRPC status code
    surfaces on the log line and as the kmonitor ``error_code`` tag.
    """

    def test_rpc_error_code_surfaces_on_log_and_kmonitor(self) -> None:
        class _Rendezvous(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self) -> str:
                return "recvmsg:Connection reset by peer"

        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[10])
            raise _Rendezvous()

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        with self.assertRaises(_Rendezvous):
            list(behavior(iter([_make_infer_request(input_ids=[1])]), FakeContext()))

        rec = self.records[0]
        self.assertEqual(rec["status"], "UNAVAILABLE")
        self.assertEqual(rec["status_detail"], "recvmsg:Connection reset by peer")

        error_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_calls), 1)
        self.assertEqual(error_calls[0][2]["error_code"], "UNAVAILABLE")


class ProtocolErrorMessageTest(InterceptorTestBase):
    """``ModelStreamInferResponse(error_message=...)`` is the ``predict_v2.proto``
    wire-level error channel: gRPC status stays OK but the frame signals backend
    failure. Before this branch existed, access_log never looked at
    ``error_message`` and routed the RPC to ``SUCCESS_QPS`` — the "success_qps
    混失败" half of the Grafana mismatch on ``whale_prod_GLM-5_0_H20_141_cp_reuse``.
    """

    @staticmethod
    def _make_error_response(msg: str) -> predict_v2_pb2.ModelStreamInferResponse:
        return predict_v2_pb2.ModelStreamInferResponse(error_message=msg)

    def test_unary_error_message_frame_routes_to_error_qps(self) -> None:
        def inner(request, context):
            return self._make_error_response("empty outputs_list from backend")

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        behavior(_make_infer_request(input_ids=[1]), FakeContext())

        rec = self.records[0]
        # Pattern-matched short form for the zero-output case so Grafana
        # distinguishes it from the generic ``BACKEND_INTERNAL`` bucket.
        self.assertEqual(rec["status"], "BACKEND_EMPTY_OUTPUTS")
        self.assertEqual(rec["status_detail"], "empty outputs_list from backend")
        self.assertEqual(rec["error_message"], "empty outputs_list from backend")
        error_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.ERROR_QPS_METRIC
        ]
        success_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.SUCCESS_QPS_METRIC
        ]
        self.assertEqual(len(error_calls), 1)
        self.assertEqual(len(success_calls), 0)
        self.assertEqual(error_calls[0][2]["error_code"], "BACKEND_EMPTY_OUTPUTS")

    def test_error_message_in_streaming_frame_routes_to_error_qps(self) -> None:
        """Mid-stream ``error_message`` frame (e.g. backend hit exception after
        a few tokens) still routes to ERROR_QPS; the first non-empty error
        message wins so a later clean frame can't overwrite it."""

        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[1])
            yield self._make_error_response("backend enqueue failed")

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        list(behavior(iter([_make_infer_request(input_ids=[1])]), FakeContext()))

        rec = self.records[0]
        # No typed ``ClassName:`` prefix and no short-form pattern match —
        # falls through to the bounded ``BACKEND_INTERNAL`` catch-all.
        self.assertEqual(rec["status"], "BACKEND_INTERNAL")
        self.assertEqual(rec["status_detail"], "backend enqueue failed")
        self.assertEqual(rec["error_message"], "backend enqueue failed")
        error_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_calls), 1)
        self.assertEqual(error_calls[0][2]["error_code"], "BACKEND_INTERNAL")

    def test_error_message_overrides_late_finish_reason(self) -> None:
        """If the backend yields ``error_message`` AND a follow-up frame carries
        ``finish_reason=0``, the error channel still wins — the inference did
        not produce a valid result. Prevents success from masking an error."""

        def inner(request_iterator, context):
            list(request_iterator)
            yield self._make_error_response("kv cache exhausted")
            yield _make_stream_response(finish_reason=0)

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        list(behavior(iter([_make_infer_request(input_ids=[1])]), FakeContext()))

        rec = self.records[0]
        self.assertEqual(rec["status"], "BACKEND_INTERNAL")
        self.assertEqual(rec["error_message"], "kv cache exhausted")

    def test_typed_class_prefix_routes_to_named_bucket(self) -> None:
        """Primary path: ``service.py`` formats backend exceptions as
        ``f"{type(e).__name__}: {e}"``. The leading class name before the
        first ``":"`` becomes the Grafana ``error_code`` tag, bounded by the
        finite Python exception class space."""

        def inner(request, context):
            return self._make_error_response("RuntimeError: kv cache oom")

        handler = _make_handler(
            request_streaming=False, response_streaming=False, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        behavior(_make_infer_request(input_ids=[1]), FakeContext())

        rec = self.records[0]
        self.assertEqual(rec["status"], "BACKEND_RuntimeError")
        error_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.ERROR_QPS_METRIC
        ]
        self.assertEqual(error_calls[0][2]["error_code"], "BACKEND_RuntimeError")


class CompletedInferenceTeardownTest(InterceptorTestBase):
    """Inference completed (``finish_reason`` observed) and a teardown signal
    showed up afterwards — client cancel / LBS drop / late ``grpc.RpcError`` /
    server writing non-OK status at close. Before this branch, these RPCs were
    classified by the tail signal and landed in ERROR_QPS / CANCEL_QPS — the
    "error_qps 混成功" half of the Grafana mismatch. A completed inference
    must stay in SUCCESS_QPS regardless of what the disconnect looks like.
    """

    def test_finish_reason_then_teardown_exception_stays_ok(self) -> None:
        """Completed inference then ``grpc.RpcError`` at close (e.g. client
        already gone) must still classify as OK — the work was delivered."""

        class _LateRendezvous(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self) -> str:
                return "recvmsg:Connection reset by peer"

        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[1, 2], finish_reason=0)
            raise _LateRendezvous()

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        with self.assertRaises(_LateRendezvous):
            list(behavior(iter([_make_infer_request(input_ids=[1])]), FakeContext()))

        rec = self.records[0]
        self.assertEqual(rec["status"], "OK")
        self.assertIsNone(rec["status_detail"])
        # The teardown exception is still recorded for triage, just not used
        # for classification.
        self.assertEqual(rec["exc_type"], "_LateRendezvous")
        self.assertEqual(rec["finish_reason"], 0)
        success_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.SUCCESS_QPS_METRIC
        ]
        error_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.ERROR_QPS_METRIC
        ]
        cancel_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.CANCEL_QPS_METRIC
        ]
        self.assertEqual(len(success_calls), 1)
        self.assertEqual(len(error_calls), 0)
        self.assertEqual(len(cancel_calls), 0)

    def test_finish_reason_then_non_ok_context_code_stays_ok(self) -> None:
        """Server wrote a late non-OK context code (e.g. CANCELLED at close
        because the client was gone) after inference already completed —
        still SUCCESS because the work was delivered."""

        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[1, 2], finish_reason=0)
            context.set_code(grpc.StatusCode.CANCELLED)
            context.set_details("client cancelled at close")

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        list(behavior(iter([_make_infer_request(input_ids=[1])]), FakeContext()))

        rec = self.records[0]
        self.assertEqual(rec["status"], "OK")
        self.assertIsNone(rec["status_detail"])
        # context_code still recorded for triage, but not used for classification.
        self.assertEqual(rec["context_code"], "CANCELLED")
        success_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.SUCCESS_QPS_METRIC
        ]
        cancel_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.CANCEL_QPS_METRIC
        ]
        self.assertEqual(len(success_calls), 1)
        self.assertEqual(len(cancel_calls), 0)

    def test_no_finish_reason_and_cancel_still_routes_to_cancel(self) -> None:
        """Narrow-scope guard: without ``finish_reason`` observed, a late
        CANCELLED code must still go to CANCEL_QPS — the new override must
        not swallow the pre-existing cancel path."""

        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[1])  # no finish_reason
            context.set_code(grpc.StatusCode.CANCELLED)
            context.set_details("client cancelled")

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        list(behavior(iter([_make_infer_request(input_ids=[1])]), FakeContext()))

        rec = self.records[0]
        self.assertEqual(rec["status"], "CANCELLED")
        self.assertEqual(rec["status_detail"], "client cancelled")
        cancel_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.CANCEL_QPS_METRIC
        ]
        success_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.SUCCESS_QPS_METRIC
        ]
        self.assertEqual(len(cancel_calls), 1)
        self.assertEqual(len(success_calls), 0)


class RawModeErrorMessageTest(RawModeInterceptorTestBase):
    """Raw mode (forward servicer) must honor the same protocol error channel
    so forwarder-logged RPCs classify consistently with real-servicer RPCs.
    Before the capture-side refactor this only worked in struct mode.
    """

    def test_raw_mode_error_message_frame_routes_to_error_qps(self) -> None:
        def inner(request_iterator, context):
            list(request_iterator)
            yield predict_v2_pb2.ModelStreamInferResponse(
                error_message="downstream backend returned error"
            )

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        list(behavior(iter([_make_infer_request(input_ids=[1])]), FakeContext()))

        rec = self.records[0]
        self.assertEqual(rec["capture_mode"], "raw")
        self.assertEqual(rec["status"], "BACKEND_INTERNAL")
        self.assertEqual(rec["error_message"], "downstream backend returned error")
        error_calls = [
            c for c in self.kmon_calls if c[0] == AccMetrics.ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_calls), 1)


class QueryLogTest(InterceptorTestBase):
    """Query log fires at handler entry — one arrival breadcrumb per RPC,
    symmetric across the forwarder's two tiers. See ``_emit_query_log`` for
    why arrival-time (not inbound-drain) is the correct trigger.
    """

    def test_query_log_records_arrival_timestamp(self) -> None:
        def inner(request_iterator, context):
            list(request_iterator)
            yield _make_stream_response(generated_ids=[1], finish_reason=0)

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        ctx = FakeContext()
        ctx.set_metadata([("x-dashscope-request-id", "corr-xyz")])
        list(behavior(iter([_make_infer_request(input_ids=[10, 20, 30])]), ctx))

        # One line per RPC carrying the arrival timestamp + correlate keys.
        self.assertEqual(len(self.query_records), 1)
        q = self.query_records[0]
        self.assertIsInstance(q["arrival_ts_epoch_ms"], int)
        self.assertEqual(q["upstream_request_id"], "corr-xyz")
        # Proto-body fields belong in the completion log, not here.
        for removed in (
            "request_id",
            "model_name",
            "input_len",
            "capture_mode",
            "input_ids",
            "generated_ids",
            "raw_requests",
            "req_read_done_ts_epoch_ms",
        ):
            self.assertNotIn(removed, q)

    def test_query_log_emitted_on_empty_inbound_stream(self) -> None:
        """Even a frame-less RPC writes one query log line — handler entry
        always fires, and the line's presence lets operators tell "forward
        never saw the RPC" (no line) apart from "forward saw it, handler
        returned error" (line present, completion log carries the error)."""

        def inner(request_iterator, context):
            for _ in request_iterator:
                pass
            raise grpc.RpcError()
            yield  # pragma: no cover

        handler = _make_handler(
            request_streaming=True, response_streaming=True, inner=inner
        )
        behavior = _wrapped_behavior(self.interceptor, handler)
        with self.assertRaises(grpc.RpcError):
            list(behavior(iter([]), FakeContext()))

        self.assertEqual(len(self.query_records), 1)
        # Completion log still fires.
        self.assertEqual(len(self.records), 1)


if __name__ == "__main__":
    main()
