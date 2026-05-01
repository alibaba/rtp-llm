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
    DashScGrpcAccessLogInterceptor,
    init_dash_sc_grpc_access_logger,
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

    def peer(self) -> str:
        return self._peer

    def code(self):
        return self._code

    def details(self) -> str:
        return self._details

    def is_active(self) -> bool:
        return self._active

    def set_code(self, code) -> None:
        self._code = code

    def set_details(self, details: str) -> None:
        self._details = details

    def set_active(self, active: bool) -> None:
        self._active = active


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

        def capture(msg, *args, **_kw):
            # Logger.info is called with a pre-formatted JSON string; parse it
            rendered = msg % args if args else msg
            self.records.append(json.loads(rendered))

        self.logger_patch = patch.object(
            self.interceptor._logger, "info", side_effect=capture
        )
        self.logger_patch.start()

    def tearDown(self) -> None:
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
        self.assertEqual(rec["status"], "UNKNOWN")
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
        self.assertEqual(rec["status"], "UNKNOWN")
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
        self.assertEqual(rec["status"], "UNKNOWN")
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
        # Exception with no grpc code -> status "UNKNOWN" lands in error_code tag so
        # alert rules can split crash vs handled gRPC codes.
        self.assertEqual(error_tags["error_code"], "UNKNOWN")
        self.assertEqual(error_tags["protocol"], "grpc")

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


if __name__ == "__main__":
    main()
