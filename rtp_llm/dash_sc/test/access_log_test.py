"""Unit tests for the dash_sc gRPC access-log data + leaf functions.

The shared access-log interceptor is gone — each servicer owns its
``ModelStreamInfer`` lifecycle inline (see ``inference/servicer.py`` /
``proxy/servicer.py`` and their tests). What's left to cover here is the
transport-agnostic machinery those servicers compose:

- status classification (:func:`rtp_llm.dash_sc.status.classify_rpc_exception` /
  :func:`~rtp_llm.dash_sc.status.classify_error_message` /
  :meth:`GrpcAccessRecord.resolve_status`);
- upstream correlation capture (:meth:`GrpcAccessRecord.create`);
- response-frame accounting + flat schema (:meth:`GrpcAccessRecord.build_record`);
- log emission (:func:`emit_query_log` / :func:`emit_access_log`);
- kmonitor fan-out (:mod:`rtp_llm.dash_sc.grpc_metrics`).
"""

from __future__ import annotations

import json
import logging
import struct
import time
from types import SimpleNamespace
from typing import Any
from unittest import TestCase, main
from unittest.mock import patch

import grpc

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.dash_sc import grpc_metrics
from rtp_llm.dash_sc.access_log import (
    DASH_SC_GRPC_ACCESS_LOGGER_NAME,
    DASH_SC_GRPC_QUERY_LOGGER_NAME,
    emit_access_log,
    emit_query_log,
    init_dash_sc_grpc_access_logger,
)
from rtp_llm.dash_sc.access_record import GrpcAccessRecord
from rtp_llm.dash_sc.codec import LLMFinishReason
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.repetition_monitor import ToolCallLoopResult
from rtp_llm.dash_sc.status import classify_error_message, classify_rpc_exception
from rtp_llm.metrics import AccMetrics, GaugeMetrics
from rtp_llm.ops import RoleType

# ---------------------------------------------------------------------------
# Proto + context fixtures
# ---------------------------------------------------------------------------


def _make_infer_request(
    *,
    request_id: str = "trace-123",
    model_name: str = "qwen3-8b",
    input_ids: list[int] | None = None,
    sampling: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
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
    for name, val in (sampling or {}).items():
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
    for name, val in (parameters or {}).items():
        param = req.parameters[name]
        if isinstance(val, bool):
            param.bool_param = val
        elif isinstance(val, int):
            param.int64_param = val
        else:
            param.string_param = val if isinstance(val, str) else json.dumps(val)
    return req


def _make_stream_response(
    *,
    generated_ids: list[int] | None = None,
    finish_reason: int | None = None,
    finished: bool | None = None,
    prompt_token_num: int | None = None,
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
    if finished is not None:
        out = infer.outputs.add()
        out.name = "finished"
        out.datatype = "BOOL"
        out.shape.append(1)
        infer.raw_output_contents.append(b"\x01" if finished else b"\x00")
    if prompt_token_num is not None:
        out = infer.outputs.add()
        out.name = "prompt_token_num"
        out.datatype = "INT32"
        out.shape.append(1)
        infer.raw_output_contents.append(struct.pack("<i", prompt_token_num))
    if prompt_cached_token_num is not None:
        out = infer.outputs.add()
        out.name = "prompt_cached_token_num"
        out.datatype = "INT32"
        out.shape.append(1)
        infer.raw_output_contents.append(struct.pack("<i", prompt_cached_token_num))
    return resp


def _make_dash_error_response(
    *,
    error_no: int = 8,
) -> predict_v2_pb2.ModelStreamInferResponse:
    resp = predict_v2_pb2.ModelStreamInferResponse()
    resp.infer_response.parameters["error_no"].int64_param = error_no
    return resp


class _FakeContext:
    """Minimal grpc.aio context surface used by ``create`` / ``resolve_status``."""

    def __init__(
        self,
        *,
        peer: str = "ipv4:1.2.3.4:5678",
        code: Any = None,
        details: str = "",
        active: bool = True,
        metadata: Any = (),
    ) -> None:
        self._peer = peer
        self._code = code
        self._details = details
        self._active = active
        self._metadata = tuple((str(k), str(v)) for k, v in metadata)

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


def _make_record(*, raw_mode: bool = False, **overrides) -> GrpcAccessRecord:
    values = dict(
        method="ModelStreamInfer",
        stream_type="bidi_stream",
        peer="ipv4:1.2.3.4:5678",
        start_ts=time.time(),
        raw_mode=raw_mode,
    )
    values.update(overrides)
    return GrpcAccessRecord(**values)


# ---------------------------------------------------------------------------
# Logger init
# ---------------------------------------------------------------------------


class InitLoggerTest(TestCase):
    def test_empty_log_path_is_noop(self) -> None:
        init_dash_sc_grpc_access_logger(log_path="", backup_count=0)
        logger = logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME)
        self.assertEqual(len(logger.handlers), 0)

    def test_sets_info_level(self) -> None:
        init_dash_sc_grpc_access_logger(log_path="", backup_count=0)
        logger = logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME)
        self.assertEqual(logger.level, logging.INFO)


# ---------------------------------------------------------------------------
# Status classification (module functions)
# ---------------------------------------------------------------------------


class ClassifyErrorMessageTest(TestCase):
    def test_typed_class_prefix(self) -> None:
        # service.py formats backend exceptions as ``f"{type(e).__name__}: {e}"``.
        self.assertEqual(
            classify_error_message("RuntimeError: kv cache oom"),
            "BACKEND_RuntimeError",
        )

    def test_short_form_pattern(self) -> None:
        self.assertEqual(
            classify_error_message("empty outputs_list from backend"),
            "BACKEND_EMPTY_OUTPUTS",
        )

    def test_unrecognized_collapses_to_internal(self) -> None:
        self.assertEqual(
            classify_error_message("backend enqueue failed"), "BACKEND_INTERNAL"
        )

    def test_empty_is_internal(self) -> None:
        self.assertEqual(classify_error_message(None), "BACKEND_INTERNAL")


class ClassifyRpcExceptionTest(TestCase):
    def test_generator_exit_is_cancelled(self) -> None:
        status, detail = classify_rpc_exception(GeneratorExit(), req_count=0)
        self.assertEqual(status, "CANCELLED")
        self.assertEqual(detail, "client closed generator")

    def test_client_request_iterator_failure_is_cancelled(self) -> None:
        class _Rendezvous(grpc.RpcError):
            def details(self) -> str:
                return "Exception iterating requests!"

            def code(self):
                return grpc.StatusCode.UNKNOWN

        status, detail = classify_rpc_exception(_Rendezvous(), req_count=0)
        self.assertEqual(status, "CANCELLED")
        self.assertEqual(detail, "client request iterator failed")

    def test_bare_rpc_error_frameless_is_cancelled(self) -> None:
        status, detail = classify_rpc_exception(grpc.RpcError(), req_count=0)
        self.assertEqual(status, "CANCELLED")
        self.assertEqual(detail, "peer closed before request arrived")

    def test_bare_rpc_error_after_frame_keeps_subclass(self) -> None:
        status, _ = classify_rpc_exception(grpc.RpcError(), req_count=1)
        self.assertEqual(status, "UNKNOWN_RpcError")

    def test_rpc_error_with_code_uses_code_name(self) -> None:
        class _Rendezvous(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self) -> str:
                return "recvmsg:Connection reset by peer"

        status, detail = classify_rpc_exception(_Rendezvous(), req_count=1)
        self.assertEqual(status, "UNAVAILABLE")
        self.assertEqual(detail, "recvmsg:Connection reset by peer")

    def test_plain_exception_keeps_class_name(self) -> None:
        status, _ = classify_rpc_exception(RuntimeError("boom"), req_count=0)
        self.assertEqual(status, "UNKNOWN_RuntimeError")


class ResolveStatusTest(TestCase):
    """``resolve_status`` precedence: error_message > completed-teardown > exc/code."""

    def test_error_message_frame_routes_to_backend_bucket(self) -> None:
        rec = _make_record(error_message="empty outputs_list from backend")
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), None)
        self.assertEqual(rec.status, "BACKEND_EMPTY_OUTPUTS")
        self.assertEqual(rec.status_detail, "empty outputs_list from backend")

    def test_inner_dash_error_routes_to_dash_error_bucket(self) -> None:
        rec = _make_record()
        rec.record_response_chunk(_make_dash_error_response(error_no=5))
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), None)

        self.assertEqual(rec.status, "DASH_ERROR_5")
        self.assertIsNone(rec.status_detail)
        self.assertTrue(rec.terminal_seen)

    def test_missing_inner_dash_error_does_not_mutate_success_frame(self) -> None:
        rec = _make_record()
        resp = _make_stream_response(generated_ids=[1])
        self.assertNotIn("error_no", resp.infer_response.parameters)

        rec.record_response_chunk(resp)
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), None)

        self.assertNotIn("error_no", resp.infer_response.parameters)
        self.assertEqual(rec.status, "OK")

    def test_zero_inner_dash_error_is_ignored(self) -> None:
        rec = _make_record()
        resp = predict_v2_pb2.ModelStreamInferResponse()
        resp.infer_response.parameters["error_no"].int64_param = 0

        rec.record_response_chunk(resp)
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), None)

        self.assertEqual(rec.status, "OK")

    def test_outer_error_message_short_circuits_inner_dash_error(self) -> None:
        rec = _make_record()
        resp = _make_dash_error_response(error_no=5)
        resp.error_message = "empty outputs_list from backend"

        rec.record_response_chunk(resp)
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), None)

        self.assertEqual(rec.status, "BACKEND_EMPTY_OUTPUTS")
        self.assertEqual(rec.status_detail, "empty outputs_list from backend")

    def test_inner_dash_error_does_not_parse_error_msg_for_access_detail(self) -> None:
        rec = _make_record()
        resp = predict_v2_pb2.ModelStreamInferResponse()
        resp.infer_response.parameters["error_no"].int64_param = 19
        payload = {
            "status_code": 500,
            "status_name": "InternalError",
            "status_message": "x" * 5000,
        }
        resp.infer_response.parameters["error_msg"].string_param = json.dumps(payload)

        rec.record_response_chunk(resp)
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), None)

        self.assertEqual(rec.status, "DASH_ERROR_19")
        self.assertIsNone(rec.status_detail)

    def test_backend_error_code_overrides_error_message(self) -> None:
        rec = _make_record(
            error_message="FtRuntimeException: no worker",
            backend_error_code="8400_MASTER_NO_AVAILABLE_WORKER",
        )
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), None)
        self.assertEqual(rec.status, "8400_MASTER_NO_AVAILABLE_WORKER")

    def test_completed_then_teardown_exception_stays_ok(self) -> None:
        # Inference delivered (terminal_seen) then a late RpcError at close.
        rec = _make_record(terminal_seen=True)
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), grpc.RpcError())
        self.assertEqual(rec.status, "OK")
        self.assertIsNone(rec.status_detail)
        self.assertEqual(rec.exc_type, "RpcError")

    def test_uncompleted_exception_is_classified(self) -> None:
        rec = _make_record()
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), RuntimeError("boom"))
        self.assertEqual(rec.status, "UNKNOWN_RuntimeError")

    def test_clean_ok(self) -> None:
        rec = _make_record()
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), None)
        self.assertEqual(rec.status, "OK")

    def test_non_ok_context_code_without_terminal(self) -> None:
        rec = _make_record()
        rec.resolve_status(
            _FakeContext(code=grpc.StatusCode.CANCELLED, details="client cancelled"),
            None,
        )
        self.assertEqual(rec.status, "CANCELLED")
        self.assertEqual(rec.status_detail, "client cancelled")

    def test_non_ok_context_code_after_terminal_stays_ok(self) -> None:
        rec = _make_record(terminal_seen=True)
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.CANCELLED), None)
        self.assertEqual(rec.status, "OK")
        self.assertEqual(rec.context_code, "CANCELLED")


# ---------------------------------------------------------------------------
# Correlation capture via create()
# ---------------------------------------------------------------------------


class CorrelationHeaderTest(TestCase):
    def _create(self, metadata):
        ctx = _FakeContext(metadata=metadata)
        return GrpcAccessRecord.create(
            ctx, "ModelStreamInfer", "bidi_stream", raw_mode=False
        )

    def test_dashscope_request_id(self) -> None:
        rec = self._create([("x-dashscope-request-id", "corr-abc-42")])
        self.assertEqual(rec.upstream_request_id, "corr-abc-42")
        self.assertEqual(rec.upstream_request_id_key, "x-dashscope-request-id")

    def test_case_insensitive(self) -> None:
        rec = self._create([("X-DashScope-Request-Id", "CaseMixed-7")])
        self.assertEqual(rec.upstream_request_id, "CaseMixed-7")
        self.assertEqual(rec.upstream_request_id_key, "x-dashscope-request-id")

    def test_dashscope_wins_over_generic(self) -> None:
        rec = self._create(
            [("x-request-id", "generic-1"), ("x-dashscope-request-id", "ds-2")]
        )
        self.assertEqual(rec.upstream_request_id, "ds-2")

    def test_traceparent_fallback(self) -> None:
        tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        rec = self._create([("traceparent", tp)])
        self.assertEqual(rec.upstream_request_id, tp)
        self.assertEqual(rec.upstream_request_id_key, "traceparent")

    def test_no_metadata_leaves_null(self) -> None:
        rec = self._create([])
        self.assertIsNone(rec.upstream_request_id)
        self.assertIsNone(rec.upstream_request_id_key)

    def test_create_attaches_record_to_context(self) -> None:
        ctx = _FakeContext()
        rec = GrpcAccessRecord.create(
            ctx, "ModelStreamInfer", "bidi_stream", raw_mode=False
        )
        self.assertIs(GrpcAccessRecord.from_context(ctx), rec)


# ---------------------------------------------------------------------------
# Frame accounting + flat schema
# ---------------------------------------------------------------------------


class BuildRecordTest(TestCase):
    def test_struct_mode_token_accounting(self) -> None:
        rec = _make_record()
        rec.capture_structured_request(
            _make_infer_request(input_ids=[10, 20, 30], sampling={"max_new_tokens": 32})
        )
        for resp, ids, finished, finish_reason, cached in (
            (_make_stream_response(generated_ids=[10]), [10], False, 2, 8),
            (_make_stream_response(generated_ids=[20, 30]), [20, 30], False, 2, None),
            (_make_stream_response(generated_ids=[40]), [40], True, 0, None),
        ):
            rec.record_generated_ids(ids)
            _, now = rec.record_response_chunk(resp)
            if rec.prompt_cached_token_num is None and cached is not None:
                rec.prompt_cached_token_num = cached
            rec.finish_reason = finish_reason
            rec.finished = finished
            delta_len = len(ids)
            rec.output_len += delta_len
            rec.token_frame_count += 1
            rec.max_tokens_per_frame = max(rec.max_tokens_per_frame, delta_len)
            if delta_len > 1:
                rec.multi_token_frame_count += 1
            if rec.first_token_ts is None:
                rec.first_token_ts = now
                rec.first_token_frame_len = delta_len
            rec.last_token_ts = now
            if finished and not rec.terminal_seen:
                rec.terminal_seen = True
                rec.terminal_ts = now
        payload = rec.build_record(server_id=1, rank_id=0)

        self.assertEqual(payload["capture_mode"], "struct")
        self.assertEqual(payload["component_role"], "frontend")
        self.assertEqual(payload["input_token_len"], 3)
        self.assertEqual(payload["output_token_len"], 4)
        self.assertEqual(payload["token_frame_count"], 3)
        self.assertEqual(payload["multi_token_frame_count"], 1)
        self.assertEqual(payload["max_tokens_per_frame"], 2)
        self.assertEqual(payload["finish_reason"], LLMFinishReason.STOP)
        self.assertEqual(payload["prompt_cached_token_num"], 8)
        self.assertEqual(payload["generate_config"]["max_new_tokens"], 32)
        self.assertEqual(payload["server_id"], 1)
        self.assertEqual(payload["rank_id"], 0)
        # Frontend struct path records the actual token ids the servicer fed.
        self.assertEqual(payload["input_ids"], [10, 20, 30])
        self.assertEqual(payload["generated_ids"], [10, 20, 30, 40])

    def test_frontend_records_generate_config_role_addrs_by_phase(self) -> None:
        rec = _make_record()
        rec.record_generate_config_role_addrs(
            GenerateConfig(
                role_addrs=[
                    RoleAddr(
                        role=RoleType.PREFILL,
                        ip="10.0.0.1",
                        http_port=8080,
                        grpc_port=8081,
                    ),
                    RoleAddr(
                        role=RoleType.DECODE,
                        ip="10.0.0.2",
                        http_port=9080,
                        grpc_port=9081,
                    ),
                ]
            ),
            phase="phase1",
        )
        payload = rec.build_record(server_id=1, rank_id=0)

        phase1 = payload["generate_config_role_addrs"]["phase1"]
        self.assertEqual(phase1[0]["role"], "PREFILL")
        self.assertEqual(phase1[1]["role"], "DECODE")
        self.assertEqual(phase1[0]["grpc_port"], 8081)
        self.assertEqual(phase1[1]["ip"], "10.0.0.2")

    def test_request_controls_are_generic_and_omit_inputs(self) -> None:
        rec = _make_record()
        rec.record_request_frame(
            _make_infer_request(
                input_ids=[1, 2, 3],
                parameters={
                    "traceparent": "00-aaa-bbb-01",
                    "custom_bool": True,
                    "accessToken": "access-token-secret",
                    "clientSecret": "client-secret-value",
                    "session_token_id": "session-token-id-secret",
                    "x-dashscope-apikey": "sk-parameter-secret",
                    "eos_token_id": "[[151643]]",
                    "ds_header_attributes": {
                        "Authorization": "Bearer secret",
                        "securityToken": "security-token-secret",
                        "X-DashScope-New-Dynamic-Field": "CaseKept",
                        "x-dashscope-apikey": "sk-header-secret",
                        "x-ds-new-dynamic-field": "kept",
                        "x-ds-llm-input-tokens": "123",
                        "nested": {"api_key": "sk-nested", "new_control": 7},
                    },
                },
            )
        )
        payload = rec.build_record(server_id=1, rank_id=0)
        controls = payload["request_controls"]

        self.assertEqual(controls["parameters"]["traceparent"], "00-aaa-bbb-01")
        self.assertEqual(controls["parameters"]["custom_bool"], True)
        self.assertEqual(controls["parameters"]["accessToken"], "<redacted>")
        self.assertEqual(controls["parameters"]["clientSecret"], "<redacted>")
        self.assertEqual(controls["parameters"]["session_token_id"], "<redacted>")
        self.assertEqual(controls["parameters"]["x-dashscope-apikey"], "<redacted>")
        self.assertEqual(controls["parameters"]["eos_token_id"], [[151643]])
        self.assertEqual(
            controls["ds_header_attributes"]["Authorization"], "<redacted>"
        )
        self.assertEqual(
            controls["ds_header_attributes"]["securityToken"], "<redacted>"
        )
        self.assertEqual(
            controls["ds_header_attributes"]["x-dashscope-apikey"], "<redacted>"
        )
        self.assertEqual(
            controls["ds_header_attributes"]["X-DashScope-New-Dynamic-Field"],
            "CaseKept",
        )
        self.assertEqual(
            controls["ds_header_attributes"]["x-ds-new-dynamic-field"], "kept"
        )
        self.assertEqual(
            controls["ds_header_attributes"]["x-ds-llm-input-tokens"], "123"
        )
        self.assertEqual(
            controls["ds_header_attributes"]["nested"]["api_key"], "<redacted>"
        )
        self.assertEqual(controls["ds_header_attributes"]["nested"]["new_control"], 7)
        self.assertNotIn("inputs", controls)
        self.assertNotIn("raw_input_contents", controls)
        self.assertIsNone(payload["input_ids"])
        serialized = json.dumps(payload)
        self.assertNotIn("access-token-secret", serialized)
        self.assertNotIn("client-secret-value", serialized)
        self.assertNotIn("session-token-id-secret", serialized)
        self.assertNotIn("security-token-secret", serialized)

    def test_request_controls_include_grpc_metadata(self) -> None:
        rec = GrpcAccessRecord.create(
            _FakeContext(
                metadata=[
                    ("traceparent", "00-aaa-bbb-01"),
                    ("x-custom-dynamic-header", "kept"),
                    ("authorization", "Bearer secret"),
                    ("x-auth", "custom-auth-secret"),
                    ("x-jwt", "jwt-secret"),
                    ("x-dashscope-apikey", "sk-metadata-secret"),
                    ("x-dashscope-apikeyid", "ak-id-secret"),
                    ("x-acs-accesskey-id", "access-key-secret"),
                    ("x-acs-security-token", "sts-secret"),
                    ("cookie", "sid=secret"),
                ]
            ),
            "ModelStreamInfer",
            "bidi_stream",
            raw_mode=True,
        )
        rec.record_request_frame(_make_infer_request())
        payload = rec.build_record(server_id=1, rank_id=0)

        metadata = payload["request_controls"]["metadata"]
        self.assertIn({"key": "traceparent", "value": "00-aaa-bbb-01"}, metadata)
        self.assertIn({"key": "x-custom-dynamic-header", "value": "kept"}, metadata)
        self.assertIn({"key": "authorization", "value": "<redacted>"}, metadata)
        self.assertIn({"key": "x-auth", "value": "<redacted>"}, metadata)
        self.assertIn({"key": "x-jwt", "value": "<redacted>"}, metadata)
        self.assertIn({"key": "x-dashscope-apikey", "value": "<redacted>"}, metadata)
        self.assertIn({"key": "x-dashscope-apikeyid", "value": "<redacted>"}, metadata)
        self.assertIn({"key": "x-acs-accesskey-id", "value": "<redacted>"}, metadata)
        self.assertIn({"key": "x-acs-security-token", "value": "<redacted>"}, metadata)
        self.assertIn({"key": "cookie", "value": "<redacted>"}, metadata)
        self.assertNotIn("custom-auth-secret", json.dumps(payload))
        self.assertNotIn("jwt-secret", json.dumps(payload))
        self.assertNotIn("sk-metadata-secret", json.dumps(payload))

    def test_empty_generated_chunk_not_counted(self) -> None:
        rec = _make_record()
        resp = predict_v2_pb2.ModelStreamInferResponse()
        rec.record_response_chunk(resp)
        rec.empty_frame_count += 1
        payload = rec.build_record(server_id=0, rank_id=0)
        self.assertEqual(payload["output_token_len"], 0)
        self.assertEqual(payload["empty_frame_count"], 1)

    def test_forward_summary_mode_and_backend_diagnostics(self) -> None:
        rec = _make_record(raw_mode=True)
        rec.record_request_frame(_make_infer_request(input_ids=[1, 2, 3]))
        rec.mark_backend_call_start("10.0.0.7:9000")
        for resp in (
            _make_stream_response(generated_ids=[10], prompt_token_num=3),
            _make_stream_response(
                generated_ids=[20, 30],
                finish_reason=LLMFinishReason.STOP,
                finished=True,
            ),
        ):
            rec.capture_backend_response_chunk(resp)
            rec.record_response_chunk(resp)
        rec.buffered_stage = "flushed_both"
        payload = rec.build_record(server_id=1, rank_id=0)

        self.assertEqual(payload["capture_mode"], "forward_summary")
        self.assertEqual(payload["component_role"], "forwarder")
        self.assertEqual(payload["forward_log_version"], 1)
        self.assertEqual(payload["backend_addr"], "10.0.0.7:9000")
        self.assertEqual(payload["backend_resp_count"], 2)
        self.assertEqual(payload["buffered_stage"], "flushed_both")
        self.assertIsNotNone(payload["request_controls"])
        self.assertNotIn("raw_responses", payload)
        # Forwarder keeps its no-structured-payload contract: token ids and
        # frontend-only statistics are not observed, so they are not logged.
        for field in (
            "latency_ttft_ms",
            "latency_tpot_ms",
            "first_token_ts_epoch_ms",
            "input_token_len",
            "backend_input_token_len",
            "output_token_len",
            "finish_reason",
            "finished",
            "terminal_seen",
            "prompt_cached_token_num",
            "token_frame_count",
            "empty_frame_count",
            "finished_only_frame_count",
            "multi_token_frame_count",
            "max_tokens_per_frame",
            "generate_config",
            "generate_config_role_addrs",
            "input_ids",
            "generated_ids",
            "repetition_monitor_impl",
            "repetition_monitor_available",
            "repetition_monitor_unavailable_reason",
            "tool_call_loop_impl",
            "tool_call_loop_error",
        ):
            self.assertNotIn(field, payload)

    def test_backend_error_frame_marks_terminal_after_backend_capture(self) -> None:
        rec = _make_record(raw_mode=True)
        resp = predict_v2_pb2.ModelStreamInferResponse(error_message="backend failed")

        rec.capture_backend_response_chunk(resp)
        rec.record_response_chunk(resp)
        payload = rec.build_record(server_id=1, rank_id=0)

        self.assertEqual(payload["error_message"], "backend failed")
        self.assertTrue(rec.terminal_seen)
        self.assertIsNotNone(payload["finished_ts_epoch_ms"])


# ---------------------------------------------------------------------------
# Log emission
# ---------------------------------------------------------------------------


class EmitLogTest(TestCase):
    def test_query_log_arrival_breadcrumb(self) -> None:
        rec = _make_record(
            upstream_request_id="corr-xyz",
            upstream_request_id_key="x-dashscope-request-id",
        )
        with patch.object(
            logging.getLogger(DASH_SC_GRPC_QUERY_LOGGER_NAME), "info"
        ) as info:
            emit_query_log(rec, rank_id=0, server_id=1)
        self.assertEqual(info.call_count, 1)
        line = info.call_args.args[0]
        self.assertNotIn("\n", line)
        q = json.loads(line)
        self.assertIsInstance(q["arrival_ts_epoch_ms"], int)
        self.assertEqual(q["upstream_request_id"], "corr-xyz")
        # Proto-body fields belong in the completion log, not the breadcrumb.
        for field in ("request_id", "model_name", "input_token_len", "capture_mode"):
            self.assertNotIn(field, q)

    def test_access_log_single_compact_line(self) -> None:
        rec = _make_record()
        rec.record_response_chunk(_make_stream_response(generated_ids=[1, 2]))
        rec.resolve_status(_FakeContext(code=grpc.StatusCode.OK), None)
        with patch.object(
            logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME), "info"
        ) as info:
            emit_access_log(rec, rank_id=0, server_id=1)
        self.assertEqual(info.call_count, 1)
        line = info.call_args.args[0]
        self.assertNotIn("\n", line)
        parsed = json.loads(line)
        self.assertEqual(parsed["status"], "OK")
        self.assertIn("ts", parsed)
        self.assertIn("method", parsed)

    def test_access_log_repetition_alert_warns(self) -> None:
        fake = SimpleNamespace(
            build_record=lambda server_id, rank_id, *, end_ts=None: {
                "repetition_alert": True,
                "request_id": "r1",
                "status": "OK",
            }
        )
        with patch.object(
            logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME), "info"
        ), patch("rtp_llm.dash_sc.access_log.logging.warning") as warn:
            emit_access_log(fake, rank_id=0, server_id=0)
        self.assertTrue(warn.called)


# ---------------------------------------------------------------------------
# kmonitor fan-out
# ---------------------------------------------------------------------------


class GrpcMetricsTest(TestCase):
    def setUp(self) -> None:
        self.calls: list[tuple[Any, float, dict]] = []
        patcher = patch.object(grpc_metrics, "kmonitor")
        self.mock_kmon = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_kmon.report.side_effect = lambda m, v=1, tags=None: self.calls.append(
            (m, v, dict(tags or {}))
        )
        # The per-process rank/server ids the servicers hold once on construction
        # and pass to every report call; ``grpc_metrics`` derives the memoized tag
        # dict from them.
        self.rank_id = 0
        self.server_id = 1
        self.tags = grpc_metrics._metric_tags(self.rank_id, self.server_id)

    def _metrics(self):
        return [c[0] for c in self.calls]

    def _for(self, metric):
        return [c for c in self.calls if c[0] == metric]

    def test_base_tags_are_pre_stringified(self) -> None:
        self.assertEqual(self.tags["protocol"], "grpc")
        self.assertEqual(self.tags["rank_id"], "0")
        self.assertEqual(self.tags["server_id"], "1")
        self.assertEqual(self.tags["method"], "ModelStreamInfer")
        self.assertEqual(grpc_metrics._metric_tags(None, None)["rank_id"], "")

    def test_arrival_fires_qps_with_base_tags(self) -> None:
        grpc_metrics.report_arrival(rank_id=self.rank_id, server_id=self.server_id)
        self.assertEqual(self._metrics().count(AccMetrics.QPS_METRIC), 1)
        self.assertEqual(self._for(AccMetrics.QPS_METRIC)[0][2], self.tags)

    def test_chunk_first_vs_iter(self) -> None:
        rec = _make_record()
        grpc_metrics.report_chunk(
            rec, rank_id=self.rank_id, server_id=self.server_id, is_first=True
        )
        grpc_metrics.report_chunk(
            rec, rank_id=self.rank_id, server_id=self.server_id, is_first=False
        )
        metrics = self._metrics()
        self.assertEqual(metrics.count(GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC), 1)
        self.assertEqual(metrics.count(GaugeMetrics.RESPONSE_ITER_RT_METRIC), 1)
        self.assertEqual(metrics.count(AccMetrics.ITER_QPS_METRIC), 2)

    def test_done_success(self) -> None:
        rec = _make_record(input_len=3, output_len=5, resp_count=2)
        grpc_metrics.report_frontend_rpc_done(
            rec, rank_id=self.rank_id, server_id=self.server_id, status="OK"
        )
        metrics = self._metrics()
        self.assertEqual(metrics.count(AccMetrics.SUCCESS_QPS_METRIC), 1)
        self.assertEqual(metrics.count(AccMetrics.ERROR_QPS_METRIC), 0)
        self.assertEqual(self._for(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC)[0][1], 3)
        self.assertEqual(self._for(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC)[0][1], 5)
        self.assertEqual(self._for(GaugeMetrics.RESPONSE_ITERATE_COUNT)[0][1], 2)

    def test_done_cancel(self) -> None:
        grpc_metrics.report_frontend_rpc_done(
            _make_record(),
            rank_id=self.rank_id,
            server_id=self.server_id,
            status="CANCELLED",
        )
        self.assertEqual(len(self._for(AccMetrics.CANCEL_QPS_METRIC)), 1)
        self.assertEqual(len(self._for(AccMetrics.ERROR_QPS_METRIC)), 0)
        self.assertNotIn("error_code", self._for(AccMetrics.CANCEL_QPS_METRIC)[0][2])

    def test_done_error_tags_error_code(self) -> None:
        grpc_metrics.report_frontend_rpc_done(
            _make_record(),
            rank_id=self.rank_id,
            server_id=self.server_id,
            status="UNAVAILABLE",
        )
        error = self._for(AccMetrics.ERROR_QPS_METRIC)
        self.assertEqual(len(error), 1)
        self.assertEqual(error[0][2]["error_code"], "UNAVAILABLE")
        # The shared dict must not be mutated by the cold error branch.
        self.assertNotIn("error_code", self.tags)

    def test_done_error_code_prefers_backend_error_code(self) -> None:
        rec = _make_record(backend_error_code="8400_MASTER_NO_AVAILABLE_WORKER")
        grpc_metrics.report_frontend_rpc_done(
            rec,
            rank_id=self.rank_id,
            server_id=self.server_id,
            status="BACKEND_INTERNAL",
        )
        self.assertEqual(
            self._for(AccMetrics.ERROR_QPS_METRIC)[0][2]["error_code"],
            "8400_MASTER_NO_AVAILABLE_WORKER",
        )

    def test_done_error_uses_status_for_inner_dash_error(self) -> None:
        rec = _make_record(status="DASH_ERROR_5")
        grpc_metrics.report_frontend_rpc_done(
            rec,
            rank_id=self.rank_id,
            server_id=self.server_id,
            status=rec.status,
        )
        self.assertEqual(len(self._for(AccMetrics.SUCCESS_QPS_METRIC)), 0)
        self.assertEqual(
            self._for(AccMetrics.ERROR_QPS_METRIC)[0][2]["error_code"],
            "DASH_ERROR_5",
        )

    def test_outer_error_status_is_used_for_metrics(self) -> None:
        rec = _make_record(
            error_message="empty outputs_list from backend",
            status="BACKEND_EMPTY_OUTPUTS",
            backend_error_code="BACKEND_EMPTY_OUTPUTS",
        )
        grpc_metrics.report_frontend_rpc_done(
            rec,
            rank_id=self.rank_id,
            server_id=self.server_id,
            status="BACKEND_EMPTY_OUTPUTS",
        )
        self.assertEqual(
            self._for(AccMetrics.ERROR_QPS_METRIC)[0][2]["error_code"],
            "BACKEND_EMPTY_OUTPUTS",
        )

    def test_done_without_input_len_omits_input_gauge(self) -> None:
        grpc_metrics.report_frontend_rpc_done(
            _make_record(input_len=None),
            rank_id=self.rank_id,
            server_id=self.server_id,
            status="OK",
        )
        self.assertEqual(len(self._for(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC)), 0)
        self.assertEqual(self._for(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC)[0][1], 0)

    def test_done_forwarder_omits_token_and_loop_metrics(self) -> None:
        rec = _make_record(raw_mode=True, input_len=3, output_len=5)
        grpc_metrics.report_forwarder_rpc_done(
            rec, rank_id=self.rank_id, server_id=self.server_id, status="OK"
        )
        metrics = self._metrics()
        self.assertNotIn(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, metrics)
        self.assertNotIn(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC, metrics)
        self.assertNotIn(GaugeMetrics.TOOL_CALL_LOOP_CHECK_RT_METRIC, metrics)

    def test_done_tool_call_loop_family(self) -> None:
        rec = _make_record()
        # Monitor verdict is computed before report_frontend_rpc_done; here we stub it to
        # the "loop hit" shape and assert the metric projection.
        rec._repetition_monitor = SimpleNamespace(
            tool_call_loop_check_ms=1.5,
            tool_call_loop_result=ToolCallLoopResult(
                hit=True, repeat_count=5, current_span_tokens=6, marker_index=0
            ),
        )
        grpc_metrics.report_frontend_rpc_done(
            rec, rank_id=self.rank_id, server_id=self.server_id, status="OK"
        )
        metrics = self._metrics()
        self.assertEqual(metrics.count(AccMetrics.TOOL_CALL_LOOP_QPS_METRIC), 1)
        self.assertEqual(metrics.count(GaugeMetrics.TOOL_CALL_LOOP_CHECK_RT_METRIC), 1)
        loop = self._for(AccMetrics.TOOL_CALL_LOOP_QPS_METRIC)[0]
        self.assertEqual(loop[2]["action"], "metric")


if __name__ == "__main__":
    main()
