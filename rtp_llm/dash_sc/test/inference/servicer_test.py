"""Unit tests for ``rtp_llm.dash_sc.inference.servicer`` (grpc.aio).

Covers:
- ``iter_real_model_stream_infer``: success, empty-stream fallback, exception propagation.
- ``DashScInferenceServicer.ModelStreamInfer``: real mode, missing input_ids,
  request_id snowflake scheme alignment with HTTP
  ``generate_request_id``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, ThinkingMode
from rtp_llm.dash_sc import grpc_metrics
from rtp_llm.dash_sc.access_log import DASH_SC_GRPC_ACCESS_LOGGER_NAME
from rtp_llm.dash_sc.access_record import (
    GrpcAccessRecord,
    extract_span_external_request_id,
)
from rtp_llm.dash_sc.codec import (
    DASH_ERROR_ABORT,
    DASH_ERROR_ADMISSION_OVERLOADED,
    DASH_ERROR_AUTO_TPM_PREEMPTED,
    DASH_ERROR_BAD_REQUEST,
    DASH_ERROR_CAPACITY,
    DASH_ERROR_INTERNAL,
    DASH_ERROR_INVALID_OUTPUT,
    DASH_ERROR_RESOURCE_EXHAUSTED,
    DASH_ERROR_TIMEOUT,
    DASH_ERROR_TOO_LONG,
    DASH_ERROR_UNSUPPORTED,
    DashScParameterError,
    DashScRequestControls,
    LLMFinishReason,
    SamplingParams,
)
from rtp_llm.dash_sc.inference.servicer import (
    DashScInferenceServicer,
    _build_mm_inputs_from_request,
    _dash_error_mapping_for_ft_exception,
    _dash_error_spec_for_ft_exception,
    _derive_max_token_id,
    _finish_server_trace,
    _request_qos_level,
    build_think_runtime,
    iter_real_model_stream_infer,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.metrics import AccMetrics
from rtp_llm.ops import RoleType
from rtp_llm.server.master_client import MasterClient
from rtp_llm.telemetry import tracing
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
)


def _add_input_tensor(
    req: predict_v2_pb2.ModelInferRequest,
    name: str,
    datatype: str,
    shape: list[int],
    raw: bytes,
) -> None:
    inp = req.inputs.add()
    inp.name = name
    inp.datatype = datatype
    inp.shape[:] = shape
    req.raw_input_contents.append(raw)


def _unpack_int32_le(raw: bytes) -> list[int]:
    return list(struct.unpack("<%di" % (len(raw) // 4), raw))


class _FakeAsyncStream:
    """Simple async iterator over a fixed chunk list, with optional error injection."""

    def __init__(self, chunks, raise_after: int | None = None):
        self._chunks = list(chunks)
        self._raise_after = raise_after
        self._emitted = 0
        self.aclose_called = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._raise_after is not None and self._emitted >= self._raise_after:
            raise RuntimeError("backend down")
        if self._emitted >= len(self._chunks):
            raise StopAsyncIteration
        item = self._chunks[self._emitted]
        self._emitted += 1
        return item

    async def aclose(self):
        self.aclose_called = True


class _FakeVisitor:
    """Async ``enqueue`` that returns a prebuilt ``_FakeAsyncStream``."""

    def __init__(self, stream: _FakeAsyncStream):
        self._stream = stream
        self.enqueue_called = 0
        self.last_generate_input = None
        self.generate_inputs = []

    async def enqueue(self, _generate_input):
        self.enqueue_called += 1
        self.last_generate_input = _generate_input
        self.generate_inputs.append(_generate_input)
        return self._stream


class _MultiStreamVisitor:
    """Async ``enqueue`` that returns one stream per call."""

    def __init__(self, streams):
        self._streams = list(streams)
        self.enqueue_called = 0
        self.last_generate_input = None
        self.generate_inputs = []

    async def enqueue(self, generate_input):
        self.enqueue_called += 1
        self.last_generate_input = generate_input
        self.generate_inputs.append(generate_input)
        return self._streams[self.enqueue_called - 1]


class DashErrorSpecForFtExceptionTest(unittest.TestCase):
    def test_non_default_exception_groups(self) -> None:
        cases = (
            (ExceptionType.INVALID_PARAMS, DASH_ERROR_BAD_REQUEST),
            (ExceptionType.LONG_PROMPT_ERROR, DASH_ERROR_TOO_LONG),
            (ExceptionType.UNSUPPORTED_OPERATION, DASH_ERROR_UNSUPPORTED),
            (ExceptionType.MASTER_NO_AVAILABLE_WORKER, DASH_ERROR_CAPACITY),
            (ExceptionType.GENERATE_TIMEOUT, DASH_ERROR_TIMEOUT),
            (ExceptionType.OUT_OF_VOCAB_RANGE, DASH_ERROR_INVALID_OUTPUT),
            (ExceptionType.CANCELLED_ERROR, DASH_ERROR_ABORT),
        )
        for exception_type, expected in cases:
            with self.subTest(exception_type=exception_type):
                self.assertEqual(
                    _dash_error_spec_for_ft_exception(
                        FtRuntimeException(exception_type, "boom")
                    ),
                    expected,
                )

    def test_internal_exception_maps_to_internal(self) -> None:
        self.assertEqual(
            _dash_error_spec_for_ft_exception(
                FtRuntimeException(ExceptionType.CONNECT_FAILED, "boom")
            ),
            DASH_ERROR_INTERNAL,
        )

    def test_priority_preempted_has_dedicated_non_capacity_spec(self) -> None:
        self.assertEqual(
            _dash_error_mapping_for_ft_exception(
                FtRuntimeException(ExceptionType.PRIORITY_PREEMPTED, "boom"),
                qos_level=50,
            ).error_spec,
            DASH_ERROR_AUTO_TPM_PREEMPTED,
        )
        self.assertEqual(
            _dash_error_spec_for_ft_exception(
                FtRuntimeException(ExceptionType.PRIORITY_PREEMPTED, "boom")
            ),
            DASH_ERROR_CAPACITY,
        )
        self.assertEqual(
            _dash_error_spec_for_ft_exception(
                FtRuntimeException(ExceptionType.MASTER_NO_AVAILABLE_WORKER, "boom")
            ),
            DASH_ERROR_CAPACITY,
        )

    def test_typed_admission_reason_matrix(self) -> None:
        cases = (
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
                DASH_ERROR_CAPACITY,
                "Service unavailable.",
                False,
            ),
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                DASH_ERROR_CAPACITY,
                "Service unavailable.",
                False,
            ),
            (
                ExceptionType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
                DASH_ERROR_CAPACITY,
                "Service unavailable.",
                False,
            ),
            (
                ExceptionType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED,
                DASH_ERROR_CAPACITY,
                "Service unavailable.",
                False,
            ),
        )
        for exception_type, reason, expected_spec, expected_message, invalid in cases:
            with self.subTest(exception_type=exception_type, reason=reason):
                mapping = _dash_error_mapping_for_ft_exception(
                    FtRuntimeException(
                        exception_type,
                        "private scheduler diagnostic",
                        admission_reject_reason=reason,
                    )
                )
                self.assertEqual(expected_spec, mapping.error_spec)
                self.assertEqual(expected_message, mapping.public_message)
                self.assertEqual(invalid, mapping.protocol_error)

    def test_illegal_admission_reason_pair_has_safe_fallback(self) -> None:
        mapping = _dash_error_mapping_for_ft_exception(
            FtRuntimeException(
                ExceptionType.RESOURCE_EXHAUSTED,
                "do not parse this text",
                admission_reject_reason=AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
            )
        )

        self.assertEqual(DASH_ERROR_CAPACITY, mapping.error_spec)
        self.assertEqual("Service unavailable.", mapping.public_message)
        self.assertTrue(mapping.protocol_error)


class _FakeTokenizer:
    eos_token_id = 2
    vocab_size = 200000

    def __init__(self, mapping: dict[str, list[int]]):
        self._mapping = mapping
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append((text, add_special_tokens))
        return list(self._mapping[text])

    def get_real_tokenizer(self):
        return self

    def __len__(self) -> int:
        return int(self.vocab_size)


class _GenerateEnvCfg:
    think_mode = "adaptive"
    think_end_token_id = -1
    think_start_tag = "<think>\n"
    think_end_tag = "</think>\n\n"


def _dsv4_tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer(
        {
            "<think>\n": [128821, 198],
            "</think>\n\n": [128822, 271],
            "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
            "</think>": [128822],
        }
    )


async def _drain(aiter):
    return [x async for x in aiter]


def _gen_ids(chunk) -> list[int]:
    infer = chunk.infer_response
    for i, out in enumerate(infer.outputs):
        if out.name == "generated_ids":
            shape = list(out.shape)
            declared_len = shape[-1] if shape else 0
            if declared_len <= 0:
                return []
            return _unpack_int32_le(infer.raw_output_contents[i])
    return []


def _finish_reason(chunk) -> int | None:
    infer = chunk.infer_response
    for i, out in enumerate(infer.outputs):
        if out.name == "finish_reason":
            return int(struct.unpack("<q", infer.raw_output_contents[i])[0])
    return None


def _dash_error_payload(chunk) -> tuple[int, dict]:
    infer = chunk.infer_response
    return (
        infer.parameters["error_no"].int64_param,
        json.loads(infer.parameters["error_msg"].string_param),
    )


def _assert_parameter_error_response(
    testcase, resp, expected_message_part: str
) -> None:
    testcase.assertFalse(resp.error_message)
    infer = resp.infer_response
    testcase.assertEqual(infer.parameters["error_no"].int64_param, 8)
    payload = json.loads(infer.parameters["error_msg"].string_param)
    testcase.assertEqual(payload["status_code"], 400)
    testcase.assertEqual(payload["status_name"], "InvalidParameter")
    testcase.assertIn(
        expected_message_part,
        payload["status_message"],
    )
    # 4xx contract: finish_reason=USE_PARAMETER_STATUS tells DashScope api-server to
    # read the standalone status_* parameters instead of mapping STOP_ENGINE_PARAM
    # onto a generic 500 EngineAbort.
    testcase.assertEqual(_finish_reason(resp), LLMFinishReason.USE_PARAMETER_STATUS)
    testcase.assertEqual(
        infer.parameters["status_code"].int64_param, payload["status_code"]
    )
    testcase.assertEqual(
        infer.parameters["status_name"].string_param, payload["status_name"]
    )
    testcase.assertEqual(
        infer.parameters["status_message"].string_param, payload["status_message"]
    )
    testcase.assertEqual(_gen_ids(resp), [])


class DeriveMaxTokenIdTest(unittest.TestCase):
    def test_uses_len_when_available(self) -> None:
        tok = _FakeTokenizer({})

        self.assertEqual(_derive_max_token_id(tok), 199999)

    def test_falls_back_to_vocab_size_when_len_is_unavailable(self) -> None:
        class _Tokenizer:
            vocab_size = "42"

            def __len__(self):
                raise TypeError("len unavailable")

        self.assertEqual(_derive_max_token_id(_Tokenizer()), 41)

    def test_returns_none_without_positive_size(self) -> None:
        class _Tokenizer:
            vocab_size = "bad"

            def __len__(self):
                raise TypeError("len unavailable")

        self.assertIsNone(_derive_max_token_id(None))
        self.assertIsNone(_derive_max_token_id(_Tokenizer()))


class BuildThinkRuntimeTest(unittest.TestCase):
    def test_derives_eos_token_id_from_tokenizer(self) -> None:
        runtime = build_think_runtime(_FakeTokenizer({}), None, None)

        self.assertEqual(runtime.eos_token_id, 2)

    def test_preserves_zero_eos_token_id(self) -> None:
        tokenizer = _FakeTokenizer({})
        tokenizer.eos_token_id = 0

        runtime = build_think_runtime(tokenizer, None, None)

        self.assertEqual(runtime.eos_token_id, 0)


class IterRealModelStreamInferTest(unittest.IsolatedAsyncioTestCase):
    def _minimal_request(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "trace-real"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [2], struct.pack("<2i", 1, 2))
        return req

    async def test_yields_one_chunk_from_mock_enqueue(self) -> None:
        req = self._minimal_request()
        out = GenerateOutput(
            output_ids=torch.tensor([3, 4], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=2, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                visitor,
                rtp_llm_request_id=1,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].error_message)
        infer = chunks[0].infer_response
        self.assertEqual(infer.id, "trace-real")
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [3, 4])
        self.assertEqual(infer.parameters["prompt_token_num"].int64_param, 2)
        self.assertEqual(infer.parameters["prompt_cached_token_num"].int64_param, 0)

    async def test_multimodal_inputs_reach_backend_generate_input(self) -> None:
        req = self._minimal_request()
        out = GenerateOutput(
            output_ids=torch.tensor([3], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=2, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        mm_input = object()

        await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                visitor,
                rtp_llm_request_id=1,
                mm_inputs=[mm_input],
            )
        )

        self.assertEqual(visitor.last_generate_input.mm_inputs, [mm_input])

    def test_builds_generic_multimodal_inputs_from_payload(self) -> None:
        req = self._minimal_request()
        req.parameters["payload"].string_param = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "http://x.png"},
                                "min_pixels": 128,
                                "max_pixels": 4096,
                                "fps": 3,
                                "min_frames": 5,
                                "max_frames": 17,
                            }
                        ],
                    }
                ]
            }
        )

        mm_inputs = _build_mm_inputs_from_request(req)

        self.assertEqual(len(mm_inputs), 1)
        self.assertEqual(mm_inputs[0].url, "http://x.png")
        self.assertEqual(mm_inputs[0].mm_preprocess_config.min_pixels, 128)
        self.assertEqual(mm_inputs[0].mm_preprocess_config.max_pixels, 4096)
        self.assertEqual(mm_inputs[0].mm_preprocess_config.fps, 3)
        self.assertEqual(mm_inputs[0].mm_preprocess_config.min_frames, 5)
        self.assertEqual(mm_inputs[0].mm_preprocess_config.max_frames, 17)
        self.assertEqual(mm_inputs[0].mm_preprocess_config.mm_timeout_ms, -1)

    def test_builds_multimodal_inputs_from_nested_dashscope_payload(self) -> None:
        req = self._minimal_request()
        req.parameters["payload"].string_param = json.dumps(
            {
                "payload": {
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"image": "http://nested.png"}],
                            }
                        ]
                    }
                }
            }
        )

        mm_inputs = _build_mm_inputs_from_request(req)

        self.assertEqual(len(mm_inputs), 1)
        self.assertEqual(mm_inputs[0].url, "http://nested.png")

    async def test_reuses_parsed_input_tensor_without_copy(self) -> None:
        """The codec-built INT32 tensor must reach the engine without a re-copy."""
        req = self._minimal_request()
        input_ids_list = [1, 2]
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.int32)
        out = GenerateOutput(
            output_ids=torch.tensor([3], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=2, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )

        await _drain(
            iter_real_model_stream_infer(
                req,
                input_ids_list,
                SamplingParams(),
                DashScRequestControls(),
                visitor,
                rtp_llm_request_id=1,
                input_ids_tensor=input_ids_tensor,
            )
        )

        self.assertIs(visitor.last_generate_input.token_ids, input_ids_tensor)

    async def test_reasoning_effort_override_reaches_generate_config(self) -> None:
        req = self._minimal_request()
        out = GenerateOutput(
            output_ids=torch.tensor([3], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=2, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(reasoning_effort="xhigh"),
                visitor,
                rtp_llm_request_id=1,
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            visitor.last_generate_input.generate_config.chat_template_kwargs,
            {"reasoning_effort": "xhigh"},
        )

    async def test_finished_at_max_new_tokens_reports_length_repro_p1(self) -> None:
        req = self._minimal_request()
        out = GenerateOutput(
            output_ids=torch.tensor([7, 8, 9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=2, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(max_new_tokens=3),
                DashScRequestControls(),
                visitor,
                rtp_llm_request_id=1,
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(_gen_ids(chunks[0]), [7, 8, 9])
        self.assertEqual(_finish_reason(chunks[0]), LLMFinishReason.LENGTH)

    async def test_empty_list_yields_error_response(self) -> None:
        req = self._minimal_request()
        visitor = _FakeVisitor(_FakeAsyncStream([]))

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                visitor,
                rtp_llm_request_id=1,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].error_message)
        error_no, payload = _dash_error_payload(chunks[0])
        self.assertEqual(error_no, 19)
        self.assertEqual(payload["status_code"], 500)
        self.assertIn("empty outputs_list", payload["status_message"])
        self.assertEqual(_finish_reason(chunks[0]), LLMFinishReason.INNER_ENGINE_ERROR)

    async def test_enqueue_exception_yields_error_message(self) -> None:
        req = self._minimal_request()

        class _BoomVisitor:
            async def enqueue(self, _gi):
                raise RuntimeError("backend down")

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                _BoomVisitor(),
                rtp_llm_request_id=1,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].error_message)
        error_no, payload = _dash_error_payload(chunks[0])
        self.assertEqual(error_no, 19)
        self.assertEqual(payload["status_code"], 500)
        self.assertIn("backend down", payload["status_message"])
        self.assertEqual(_finish_reason(chunks[0]), LLMFinishReason.INNER_ENGINE_ERROR)

    async def test_ft_exception_sets_access_backend_error_code(self) -> None:
        req = self._minimal_request()
        exception_aux_info = {
            "input_len": 2,
            "reuse_len": 1,
            "remote_reuse_len": 1,
            "aux_string": "route-diagnostic",
        }

        class _BoomVisitor:
            async def enqueue(self, _gi):
                error = FtRuntimeException(
                    ExceptionType.ROUTE_ERROR,
                    "route failed",
                )
                error.aux_info = exception_aux_info
                raise error

        access_agg = GrpcAccessRecord(
            method="ModelStreamInfer",
            stream_type="bidi_stream",
            peer="",
            start_ts=0.0,
            raw_mode=False,
        )
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                _BoomVisitor(),
                rtp_llm_request_id=1,
                access_agg=access_agg,
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].error_message)
        error_no, payload = _dash_error_payload(chunks[0])
        self.assertEqual(error_no, 5)
        self.assertEqual(payload["status_code"], 503)
        self.assertEqual(payload["status_name"], "ServiceUnavailable")
        self.assertIn("route failed", payload["status_message"])
        self.assertEqual(
            _finish_reason(chunks[0]), LLMFinishReason.USE_PARAMETER_STATUS
        )
        self.assertEqual(access_agg.backend_error_code, "8500_ROUTE_ERROR")
        self.assertEqual(access_agg.aux_info, exception_aux_info)

    async def test_generic_exception_captures_access_aux_info(self) -> None:
        """A non-Ft engine failure still carries the enqueue-side aux_info."""
        req = self._minimal_request()
        exception_aux_info = {
            "input_len": 2,
            "output_len": 0,
            "step_output_len": 0,
            "reuse_len": 0,
            "pd_sep": True,
        }

        class _BoomVisitor:
            async def enqueue(self, _gi):
                error = RuntimeError("backend down")
                error.aux_info = exception_aux_info
                raise error

        access_agg = GrpcAccessRecord(
            method="ModelStreamInfer",
            stream_type="bidi_stream",
            peer="",
            start_ts=0.0,
            raw_mode=False,
        )
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                _BoomVisitor(),
                rtp_llm_request_id=1,
                access_agg=access_agg,
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(_finish_reason(chunks[0]), LLMFinishReason.INNER_ENGINE_ERROR)
        self.assertEqual(access_agg.aux_info, exception_aux_info)
        # backend_error_code stays unset: only FtRuntimeException carries one.
        self.assertIsNone(access_agg.backend_error_code)

    async def test_auto_tpm_preempted_maps_to_throttling_aborted(self) -> None:
        """8429 uses the QoS 429 contract only with an explicit valid header."""
        req = self._minimal_request()

        class _PreemptedVisitor:
            async def enqueue(self, _gi):
                raise FtRuntimeException(
                    ExceptionType.PRIORITY_PREEMPTED,
                    "AUTO_TPM_PREEMPTED victim request 703 by request 77",
                )

        for (
            request_headers,
            expected_error_no,
            expected_status,
            expected_name,
            expected_message,
        ) in (
            (
                {},
                LLMFinishReason.TASK_LIST_FULL,
                503,
                "ServiceUnavailable",
                "Service unavailable.",
            ),
            (
                {"x-dashscope-inner-qos-level": "49"},
                LLMFinishReason.ABORT,
                429,
                "Throttling.Aborted",
                "Too many requests.",
            ),
            (
                {"x-dashscope-inner-qos-level": "50"},
                LLMFinishReason.ABORT,
                429,
                "Throttling.Aborted",
                "Too many requests.",
            ),
        ):
            with self.subTest(request_headers=request_headers):
                access_agg = GrpcAccessRecord(
                    method="ModelStreamInfer",
                    stream_type="bidi_stream",
                    peer="",
                    start_ts=0.0,
                    raw_mode=False,
                )
                chunks = await _drain(
                    iter_real_model_stream_infer(
                        req,
                        [1, 2],
                        SamplingParams(),
                        DashScRequestControls(request_headers=request_headers),
                        _PreemptedVisitor(),
                        rtp_llm_request_id=1,
                        access_agg=access_agg,
                    )
                )

                self.assertEqual(len(chunks), 1)
                self.assertFalse(chunks[0].error_message)
                infer = chunks[0].infer_response
                error_no, payload = _dash_error_payload(chunks[0])
                self.assertEqual(error_no, expected_error_no)
                self.assertEqual(payload["status_code"], expected_status)
                self.assertEqual(payload["status_name"], expected_name)
                self.assertEqual(payload["status_message"], expected_message)
                # Legacy error_msg JSON and standalone status_* parameters must
                # stay byte-for-byte equivalent for old and new consumers.
                self.assertEqual(
                    infer.parameters["status_code"].int64_param, expected_status
                )
                self.assertEqual(
                    infer.parameters["status_name"].string_param,
                    expected_name,
                )
                self.assertEqual(
                    infer.parameters["status_message"].string_param,
                    expected_message,
                )
                self.assertEqual(
                    _finish_reason(chunks[0]), LLMFinishReason.USE_PARAMETER_STATUS
                )
                public_payload = (
                    infer.parameters["error_msg"].string_param
                    + infer.parameters["status_message"].string_param
                )
                self.assertNotIn("AUTO_TPM_PREEMPTED", public_payload)
                self.assertNotIn("703", public_payload)
                self.assertNotIn("77", public_payload)
                self.assertEqual(
                    access_agg.backend_error_code,
                    "8429_PRIORITY_PREEMPTED",
                )

    async def test_qos_header_maps_admission_rejections_by_priority_tier(
        self,
    ) -> None:
        req = self._minimal_request()
        cases = (
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
            ),
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD,
            ),
            (
                ExceptionType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
            ),
        )

        for exception_type, reason in cases:

            class _RejectedVisitor:
                async def enqueue(self, _gi):
                    raise FtRuntimeException(
                        exception_type,
                        "private scheduler diagnostic",
                        admission_reject_reason=reason,
                    )

            for qos, expected_name in (
                ("49", "Throttling.ServiceOverloaded"),
                ("50", "Throttling.ResourceExhausted"),
            ):
                with self.subTest(
                    exception_type=exception_type, reason=reason, qos=qos
                ):
                    chunks = await _drain(
                        iter_real_model_stream_infer(
                            req,
                            [1, 2],
                            SamplingParams(),
                            DashScRequestControls(
                                request_headers={"x-dashscope-inner-qos-level": qos}
                            ),
                            _RejectedVisitor(),
                            rtp_llm_request_id=1,
                        )
                    )

                    self.assertEqual(1, len(chunks))
                    _, payload = _dash_error_payload(chunks[0])
                    self.assertEqual(429, payload["status_code"])
                    self.assertEqual(expected_name, payload["status_name"])
                    self.assertEqual("Too many requests.", payload["status_message"])
                    self.assertNotIn(
                        "private scheduler diagnostic",
                        chunks[0].infer_response.parameters["error_msg"].string_param,
                    )

    async def test_admission_rejections_without_qos_header_use_legacy_503(
        self,
    ) -> None:
        req = self._minimal_request()
        cases = (
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                "ServiceUnavailable",
                "Service unavailable.",
            ),
            (
                ExceptionType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
                "ServiceUnavailable",
                "Service unavailable.",
            ),
            (
                ExceptionType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED,
                "ServiceUnavailable",
                "Service unavailable.",
            ),
            (
                ExceptionType.BATCH_SLO_EXPIRED,
                AdmissionRejectReason.UNSPECIFIED,
                "ServiceUnavailable",
                "private scheduler diagnostic",
            ),
        )

        for exception_type, reason, expected_name, expected_message in cases:

            class _RejectedVisitor:
                async def enqueue(self, _gi):
                    raise FtRuntimeException(
                        exception_type,
                        "private scheduler diagnostic",
                        admission_reject_reason=reason,
                    )

            with self.subTest(exception_type=exception_type, reason=reason):
                chunks = await _drain(
                    iter_real_model_stream_infer(
                        req,
                        [1, 2],
                        SamplingParams(),
                        DashScRequestControls(),
                        _RejectedVisitor(),
                        rtp_llm_request_id=1,
                    )
                )

                self.assertEqual(1, len(chunks))
                _, payload = _dash_error_payload(chunks[0])
                self.assertEqual(503, payload["status_code"])
                self.assertEqual(expected_name, payload["status_name"])
                self.assertEqual(expected_message, payload["status_message"])

    async def test_admission_qos_mapping_reads_invocation_metadata(self) -> None:
        req = self._minimal_request()

        class _RejectedVisitor:
            async def enqueue(self, _gi):
                raise FtRuntimeException(
                    ExceptionType.RESOURCE_EXHAUSTED,
                    "private scheduler diagnostic",
                    admission_reject_reason=AdmissionRejectReason.RESOURCE_EXHAUSTED,
                )

        for qos, expected_name in (
            ("49", "Throttling.ServiceOverloaded"),
            ("50", "Throttling.ResourceExhausted"),
        ):
            with self.subTest(qos=qos):
                chunks = await _drain(
                    iter_real_model_stream_infer(
                        req,
                        [1, 2],
                        SamplingParams(),
                        DashScRequestControls(),
                        _RejectedVisitor(),
                        rtp_llm_request_id=1,
                        invocation_metadata=(("x-dashscope-inner-qos-level", qos),),
                    )
                )

                _, payload = _dash_error_payload(chunks[0])
                self.assertEqual(expected_name, payload["status_name"])
                self.assertEqual("Too many requests.", payload["status_message"])

    async def test_priority_attribution_unavailable_is_503_even_with_qos(
        self,
    ) -> None:
        req = self._minimal_request()

        class _UnavailableVisitor:
            async def enqueue(self, _gi):
                raise FtRuntimeException(
                    ExceptionType.ADMISSION_UNAVAILABLE,
                    "occupant priority provenance is unavailable",
                    admission_reject_reason=AdmissionRejectReason.UNSPECIFIED,
                )

        for request_headers in (
            {},
            {"x-dashscope-inner-qos-level": "49"},
            {"x-dashscope-inner-qos-level": "50"},
        ):
            with self.subTest(request_headers=request_headers):
                chunks = await _drain(
                    iter_real_model_stream_infer(
                        req,
                        [1, 2],
                        SamplingParams(),
                        DashScRequestControls(request_headers=request_headers),
                        _UnavailableVisitor(),
                        rtp_llm_request_id=1,
                    )
                )

                _, payload = _dash_error_payload(chunks[0])
                self.assertEqual(503, payload["status_code"])
                self.assertEqual("ServiceUnavailable", payload["status_name"])
                self.assertEqual("Service unavailable.", payload["status_message"])

    async def test_invalid_qos_header_keeps_existing_admission_mapping(self) -> None:
        req = self._minimal_request()

        class _RejectedVisitor:
            async def enqueue(self, _gi):
                raise FtRuntimeException(
                    ExceptionType.PRIORITY_ADMISSION_REJECTED,
                    "private scheduler diagnostic",
                    admission_reject_reason=AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                )

        for qos in ("invalid", "0", "-1", "101"):
            with self.subTest(qos=qos):
                chunks = await _drain(
                    iter_real_model_stream_infer(
                        req,
                        [1, 2],
                        SamplingParams(),
                        DashScRequestControls(
                            request_headers={"x-dashscope-inner-qos-level": qos}
                        ),
                        _RejectedVisitor(),
                        rtp_llm_request_id=1,
                    )
                )

                _, payload = _dash_error_payload(chunks[0])
                self.assertEqual("ServiceUnavailable", payload["status_name"])
                self.assertEqual("Service unavailable.", payload["status_message"])

    def test_invalid_metadata_qos_falls_back_to_valid_request_header(self) -> None:
        for metadata_qos in ("invalid", "0", "-1", "101"):
            with self.subTest(metadata_qos=metadata_qos):
                qos_level = _request_qos_level(
                    DashScRequestControls(
                        request_headers={"x-dashscope-inner-qos-level": "50"}
                    ),
                    (("x-dashscope-inner-qos-level", metadata_qos),),
                )
                self.assertEqual(50, qos_level)

    def test_invalid_metadata_and_request_qos_are_not_explicit_qos(
        self,
    ) -> None:
        for qos in ("invalid", "0", "-1", "101"):
            with self.subTest(qos=qos):
                qos_level = _request_qos_level(
                    DashScRequestControls(
                        request_headers={"x-dashscope-inner-qos-level": qos}
                    ),
                    (("x-dashscope-inner-qos-level", qos),),
                )
                self.assertIsNone(qos_level)

    async def test_valid_metadata_qos_overrides_request_header(self) -> None:
        req = self._minimal_request()

        class _RejectedVisitor:
            async def enqueue(self, _gi):
                raise FtRuntimeException(
                    ExceptionType.RESOURCE_EXHAUSTED,
                    "private scheduler diagnostic",
                    admission_reject_reason=AdmissionRejectReason.RESOURCE_EXHAUSTED,
                )

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(
                    request_headers={"x-dashscope-inner-qos-level": "49"}
                ),
                _RejectedVisitor(),
                rtp_llm_request_id=1,
                invocation_metadata=(("x-dashscope-inner-qos-level", "50"),),
            )
        )

        _, payload = _dash_error_payload(chunks[0])
        self.assertEqual("Throttling.ResourceExhausted", payload["status_name"])
        self.assertEqual("Too many requests.", payload["status_message"])

    def test_explicit_qos_does_not_hide_invalid_admission_reason_pair(self) -> None:
        invalid_pairs = (
            (
                ExceptionType.PRIORITY_PREEMPTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
            ),
            (
                ExceptionType.PRIORITY_PREEMPTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD,
            ),
            (
                ExceptionType.PRIORITY_PREEMPTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
            ),
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.UNSPECIFIED,
            ),
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
            ),
            (
                ExceptionType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.UNSPECIFIED,
            ),
            (
                ExceptionType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
            ),
            (
                ExceptionType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD,
            ),
        )

        for exception_type, reason in invalid_pairs:
            for qos_level in (49, 50):
                with self.subTest(
                    exception_type=exception_type,
                    reason=reason,
                    qos_level=qos_level,
                ):
                    mapping = _dash_error_mapping_for_ft_exception(
                        FtRuntimeException(
                            exception_type,
                            "private scheduler diagnostic",
                            admission_reject_reason=reason,
                        ),
                        qos_level=qos_level,
                    )

                    self.assertEqual(
                        DASH_ERROR_CAPACITY,
                        mapping.error_spec,
                    )
                    self.assertEqual("Service unavailable.", mapping.public_message)
                    self.assertTrue(mapping.protocol_error)

        invalid_raw_reason = FtRuntimeException(
            ExceptionType.PRIORITY_PREEMPTED,
            "private scheduler diagnostic",
        )
        invalid_raw_reason.admission_reject_reason = 999
        for qos_level in (None, 49, 50):
            with self.subTest(raw_reason=999, qos_level=qos_level):
                mapping = _dash_error_mapping_for_ft_exception(
                    invalid_raw_reason,
                    qos_level=qos_level,
                )
                self.assertEqual(DASH_ERROR_CAPACITY, mapping.error_spec)
                self.assertEqual("Service unavailable.", mapping.public_message)
                self.assertTrue(mapping.protocol_error)

    async def test_generic_capacity_mapping_does_not_depend_on_qos_priority(
        self,
    ) -> None:
        """A QoS value is not a rejection reason.

        Generic capacity failures keep the historical 503 contract for every
        priority; only typed Auto-TPM outcomes opt into the QoS 429 mapping.
        """
        req = self._minimal_request()

        class _CapacityVisitor:
            async def enqueue(self, _gi):
                raise FtRuntimeException(
                    ExceptionType.TRAFFIC_LIMIT_ERROR,
                    "traffic limit exceeded",
                )

        for request_headers in (
            {},
            {"x-dashscope-inner-qos-level": "40"},
            {"x-dashscope-inner-qos-level": "50"},
        ):
            with self.subTest(request_headers=request_headers):
                chunks = await _drain(
                    iter_real_model_stream_infer(
                        req,
                        [1, 2],
                        SamplingParams(),
                        DashScRequestControls(request_headers=request_headers),
                        _CapacityVisitor(),
                        rtp_llm_request_id=1,
                    )
                )
                self.assertEqual(len(chunks), 1)
                self.assertFalse(chunks[0].error_message)
                error_no, payload = _dash_error_payload(chunks[0])
                self.assertEqual(error_no, 5)
                self.assertEqual(payload["status_code"], 503)
                self.assertEqual(payload["status_name"], "ServiceUnavailable")
                self.assertIn("traffic limit exceeded", payload["status_message"])
                self.assertEqual(
                    _finish_reason(chunks[0]),
                    LLMFinishReason.USE_PARAMETER_STATUS,
                )

    async def test_error_text_does_not_override_typed_internal_code(self) -> None:
        """Diagnostic text is never used as a capacity classification."""
        req = self._minimal_request()

        class _TaskListFullVisitor:
            async def enqueue(self, _gi):
                raise FtRuntimeException(
                    ExceptionType.UNKNOWN_ERROR,
                    "Inference engine abort. Finish reason: [TASK_LIST_FULL].",
                )

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                _TaskListFullVisitor(),
                rtp_llm_request_id=1,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].error_message)
        error_no, payload = _dash_error_payload(chunks[0])
        self.assertEqual(error_no, 19)
        self.assertEqual(payload["status_code"], 500)
        self.assertEqual(payload["status_name"], "InternalError")
        self.assertIn("TASK_LIST_FULL", payload["status_message"])
        self.assertEqual(_finish_reason(chunks[0]), LLMFinishReason.INNER_ENGINE_ERROR)

    async def test_generic_error_text_does_not_create_capacity_reason(self) -> None:
        req = self._minimal_request()

        class _TaskListFullGenericVisitor:
            async def enqueue(self, _gi):
                raise RuntimeError(
                    "Inference engine abort. Finish reason: [TASK_LIST_FULL]."
                )

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                _TaskListFullGenericVisitor(),
                rtp_llm_request_id=1,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].error_message)
        error_no, payload = _dash_error_payload(chunks[0])
        self.assertEqual(error_no, 19)
        self.assertEqual(payload["status_code"], 500)
        self.assertEqual(payload["status_name"], "InternalError")
        self.assertIn("TASK_LIST_FULL", payload["status_message"])
        self.assertEqual(_finish_reason(chunks[0]), LLMFinishReason.INNER_ENGINE_ERROR)

    async def test_stream_exception_yields_error_message(self) -> None:
        req = self._minimal_request()
        visitor = _FakeVisitor(_FakeAsyncStream([], raise_after=0))

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                visitor,
                rtp_llm_request_id=1,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].error_message)
        error_no, payload = _dash_error_payload(chunks[0])
        self.assertEqual(error_no, 19)
        self.assertEqual(payload["status_code"], 500)
        self.assertIn("backend down", payload["status_message"])

    async def test_no_thinking_budget_zero_sets_sampler_mask_config_without_filtering(
        self,
    ) -> None:
        req = self._minimal_request()
        _add_input_tensor(
            req,
            "max_new_think_tokens",
            "INT32",
            [1],
            struct.pack("<i", 0),
        )
        out = GenerateOutput(
            output_ids=torch.tensor([10, 128822, 271], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=2, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )

        chunks = await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 1)
        gc = visitor.last_generate_input.generate_config
        self.assertFalse(gc.in_think_mode)
        self.assertEqual(gc.max_thinking_tokens, 0)
        self.assertEqual(gc.begin_think_token_ids, [128821, 198])
        self.assertEqual(gc.end_think_token_ids, [128822, 271])
        self.assertEqual(_gen_ids(chunks[0]), [10, 128822, 271])

    async def test_max_think_length_wins_final_config_over_max_new_think_tokens(
        self,
    ) -> None:
        req = self._minimal_request()
        _add_input_tensor(
            req,
            "max_new_think_tokens",
            "INT32",
            [1],
            struct.pack("<i", 0),
        )
        _add_input_tensor(
            req,
            "max_think_length",
            "INT32",
            [1],
            struct.pack("<i", -1),
        )
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        context = MagicMock()
        context.invocation_metadata.return_value = ()

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), context))

        self.assertEqual(visitor.enqueue_called, 1)
        gc = visitor.last_generate_input.generate_config
        self.assertTrue(gc.in_think_mode)
        self.assertEqual(gc.thinking_mode, ThinkingMode.ENABLED)
        self.assertEqual(gc.max_thinking_tokens, 2_147_483_647)
        self.assertEqual(gc.end_think_token_ids, [128822, 271])
        self.assertEqual(gc.structural_tag["format"]["type"], "sequence")

    async def test_budget_zero_disables_thinking(self) -> None:
        """Request-level zero budget must still produce a full think mask config."""
        req = self._minimal_request()
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()

        await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(max_new_think_tokens=0),
                DashScRequestControls(),
                visitor,
                rtp_llm_request_id=1,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
            )
        )

        gc = visitor.last_generate_input.generate_config
        self.assertFalse(gc.in_think_mode)
        self.assertEqual(gc.max_thinking_tokens, 0)
        self.assertEqual(gc.begin_think_token_ids, [128821, 198])
        self.assertEqual(gc.end_think_token_ids, [128822, 271])

    async def test_deepseek_v4_multi_think_uses_first_close_only(self) -> None:
        req = self._minimal_request()
        chunks_proto = []
        for ids, finished in (
            ([10, 128822, 11], False),
            ([12, 128822, 13], True),
        ):
            out = GenerateOutput(
                output_ids=torch.tensor(ids, dtype=torch.int32),
                finished=finished,
                aux_info=AuxInfo(input_len=2, reuse_len=0),
            )
            chunks_proto.append(GenerateOutputs(generate_outputs=[out]))
        visitor = _FakeVisitor(_FakeAsyncStream(chunks_proto))
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                visitor,
                rtp_llm_request_id=1,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "qwen2"),
            )
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(_gen_ids(chunks[0]), [10, 128822, 11])
        self.assertEqual(_gen_ids(chunks[1]), [12, 128822, 13])
        self.assertEqual(
            chunks[0].infer_response.parameters["generate_think_token_num"].int64_param,
            2,
        )
        self.assertEqual(
            chunks[1].infer_response.parameters["generate_think_token_num"].int64_param,
            2,
        )

    async def test_deepseek_v4_token1_forces_empty_think_phase2_prompt(self) -> None:
        req = self._minimal_request()
        req.parameters["payload"].string_param = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"image": "http://example/phase2.png"}],
                    }
                ]
            }
        )
        mm_inputs = _build_mm_inputs_from_request(req)
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 1, 99], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20, 21, 22], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )

        class _RoutingVisitor(_MultiStreamVisitor):
            async def enqueue(self, generate_input):
                if self.enqueue_called == 0:
                    generate_input.generate_config.role_addrs = [
                        RoleAddr(
                            role=RoleType.PREFILL,
                            ip="127.0.0.1",
                            http_port=8080,
                            grpc_port=8081,
                        )
                    ]
                return await super().enqueue(generate_input)

        visitor = _RoutingVisitor(
            [_FakeAsyncStream([phase1]), _FakeAsyncStream([phase2])]
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(
                    response_format=json.dumps({"type": "json_object"}),
                ),
                DashScRequestControls(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
                mm_inputs=mm_inputs,
            )
        )

        self.assertEqual(visitor.enqueue_called, 2)
        self.assertEqual(visitor.generate_inputs[0].request_id, 100)
        self.assertEqual(visitor.generate_inputs[1].request_id, 200)
        self.assertEqual(visitor.generate_inputs[0].mm_inputs, mm_inputs)
        self.assertEqual(visitor.generate_inputs[1].mm_inputs, mm_inputs)
        self.assertTrue(visitor.generate_inputs[0].generate_config.in_think_mode)
        self.assertEqual(
            visitor.generate_inputs[0].generate_config.begin_think_token_ids,
            [128821, 198],
        )
        self.assertEqual(
            visitor.generate_inputs[0].generate_config.end_think_token_ids,
            [128822, 271],
        )
        self.assertEqual(_gen_ids(chunks[0]), [128821, 10, 11])
        self.assertEqual(_gen_ids(chunks[1]), [128822, 271])
        self.assertEqual(_gen_ids(chunks[2]), [20, 21, 22])
        self.assertEqual(chunks[2].infer_response.id, "trace-real-2")
        # phase-2 trace_id mirrors phase-1 (no -2 suffix) so dashscope log search
        # finds both halves under a single trace.
        self.assertEqual(
            visitor.generate_inputs[1].generate_config.trace_id, "trace-real"
        )
        phase2_input_ids = visitor.generate_inputs[1].token_ids.cpu().int().tolist()
        self.assertEqual(phase2_input_ids, [7, 8, 128821, 271, 128822, 271])
        phase1_config = visitor.generate_inputs[0].generate_config
        phase2_config = visitor.generate_inputs[1].generate_config
        self.assertIsNone(phase1_config.response_format)
        self.assertIsNotNone(phase1_config.structural_tag)
        self.assertIsNone(phase2_config.response_format)
        self.assertIsNone(phase2_config.structural_tag)
        self.assertEqual(
            phase2_config.json_schema,
            {"anyOf": [{"type": "object"}, {"type": "array"}]},
        )
        self.assertFalse(visitor.generate_inputs[1].generate_config.in_think_mode)
        self.assertEqual(len(visitor.generate_inputs[0].generate_config.role_addrs), 1)
        self.assertEqual(visitor.generate_inputs[1].generate_config.role_addrs, [])
        self.assertNotIn(10, phase2_input_ids)
        self.assertNotIn(11, phase2_input_ids)
        self.assertEqual(
            chunks[1].infer_response.parameters["generate_think_token_num"].int64_param,
            3,
        )

    async def test_phase2_finished_at_max_new_tokens_reports_length(self) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 1], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20, 21], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [_FakeAsyncStream([phase1]), _FakeAsyncStream([phase2])]
        )
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(
                    max_new_tokens=2,
                    max_new_tokens_from_completion_alias=True,
                ),
                DashScRequestControls(enable_thinking=True, max_new_think_tokens=10),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        phase2_chunks = [c for c in chunks if c.infer_response.id.endswith("-2")]
        self.assertEqual(visitor.generate_inputs[0].generate_config.max_new_tokens, 2)
        self.assertEqual(visitor.generate_inputs[1].generate_config.max_new_tokens, 2)
        self.assertEqual(len(phase2_chunks), 1)
        self.assertEqual(_gen_ids(phase2_chunks[0]), [20, 21])
        self.assertEqual(_finish_reason(phase2_chunks[0]), LLMFinishReason.LENGTH)

    async def test_phase2_completion_alias_respects_max_tokens_total_cap(
        self,
    ) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor(
                        list(range(10, 20)) + [1], dtype=torch.int32
                    ),
                    finished=False,
                    aux_info=AuxInfo(input_len=2, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [_FakeAsyncStream([phase1]), _FakeAsyncStream([phase2])]
        )
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(
                    max_new_tokens=100,
                    max_new_tokens_from_completion_alias=True,
                    max_total_tokens=105,
                ),
                DashScRequestControls(enable_thinking=True, max_new_think_tokens=10),
                visitor,
                rtp_llm_request_id=100,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.generate_inputs[0].generate_config.max_new_tokens, 100)
        self.assertEqual(visitor.generate_inputs[1].generate_config.max_new_tokens, 95)
        self.assertEqual(
            chunks[1].infer_response.parameters["generate_think_token_num"].int64_param,
            10,
        )

    async def test_phase2_completion_alias_budget_zero_does_not_enqueue_phase2(
        self,
    ) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 12, 1], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=2, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor([_FakeAsyncStream([phase1])])
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(
                    max_new_tokens=3,
                    max_new_tokens_from_completion_alias=True,
                    max_total_tokens=3,
                ),
                DashScRequestControls(enable_thinking=True, max_new_think_tokens=10),
                visitor,
                rtp_llm_request_id=100,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertFalse(any(c.error_message for c in chunks))
        self.assertEqual(_gen_ids(chunks[-1]), [128822, 271])
        self.assertEqual(_finish_reason(chunks[-1]), LLMFinishReason.LENGTH)

    async def test_token1_phase2_closes_phase1_stream_before_phase2_enqueue(
        self,
    ) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 1], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase1_stream = _FakeAsyncStream([phase1])
        visitor = _MultiStreamVisitor([phase1_stream, _FakeAsyncStream([phase2])])
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()

        await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                DashScRequestControls(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 2)
        self.assertTrue(phase1_stream.aclose_called)

    async def test_consumer_close_closes_phase2_stream_immediately(self) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 1], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([128822, 271, 20], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase1_stream = _FakeAsyncStream([phase1])
        phase2_stream = _FakeAsyncStream([phase2])
        visitor = _MultiStreamVisitor([phase1_stream, phase2_stream])
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        response_iter = iter_real_model_stream_infer(
            req,
            [7, 8, 128821],
            SamplingParams(),
            DashScRequestControls(enable_thinking=True),
            visitor,
            rtp_llm_request_id=100,
            echo_prefix_ids=[128821, 198],
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
            phase2_request_id_factory=lambda: 200,
        )

        while visitor.enqueue_called < 2:
            await response_iter.__anext__()
        self.assertFalse(phase2_stream.aclose_called)
        await response_iter.aclose()

        self.assertTrue(phase1_stream.aclose_called)
        self.assertTrue(phase2_stream.aclose_called)

    async def test_request_disable_thinking_prevents_token1_phase2(self) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 1, 99], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _FakeVisitor(_FakeAsyncStream([phase1]))
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                DashScRequestControls(enable_thinking=False, max_new_think_tokens=0),
                visitor,
                rtp_llm_request_id=100,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(_gen_ids(chunks[0]), [10, 11, 1, 99])
        self.assertNotEqual(chunks[0].infer_response.id, "trace-real-2")
        cfg = visitor.generate_inputs[0].generate_config
        self.assertFalse(cfg.in_think_mode)
        self.assertEqual(cfg.max_thinking_tokens, 0)
        self.assertEqual(cfg.begin_think_token_ids, [128821, 198])
        self.assertEqual(cfg.end_think_token_ids, [128822, 271])
        self.assertNotIn("max_new_think_tokens", chunks[0].infer_response.parameters)
        self.assertNotIn(
            "generate_think_token_num", chunks[0].infer_response.parameters
        )

    async def test_deepseek_v4_token1_before_close_wins_within_chunk(self) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 1, 128822, 99], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [_FakeAsyncStream([phase1]), _FakeAsyncStream([phase2])]
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                DashScRequestControls(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 2)
        self.assertEqual(_gen_ids(chunks[0]), [128821, 10])
        self.assertEqual(_gen_ids(chunks[1]), [128822, 271])
        self.assertEqual(_gen_ids(chunks[2]), [20])
        self.assertEqual(
            chunks[1].infer_response.parameters["generate_think_token_num"].int64_param,
            2,
        )

    async def test_terminate_token_id_disabled_keeps_token_in_stream(self) -> None:
        """``terminate_token_id=None`` disables the in-stream "stop thinking"
        branch: token id 1 is emitted as a regular content token, with no
        truncation and no phase-2 enqueue. (No ``</think>`` in the stream so the
        regular close-driven phase-2 path also stays dormant.)"""
        req = self._minimal_request()
        out = GenerateOutput(
            output_ids=torch.tensor([10, 1, 11], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=2, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                DashScRequestControls(),
                visitor,
                rtp_llm_request_id=1,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(
                    tok, env_cfg, "deepseek_v4", terminate_token_id=None
                ),
            )
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(_gen_ids(chunks[0]), [10, 1, 11])

    async def test_terminate_token_id_configurable_value(self) -> None:
        """A non-default ``terminate_token_id`` (here 42) drives the same
        truncation + phase-2 prompt rewrite that token id 1 does by default."""
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 42, 99], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [_FakeAsyncStream([phase1]), _FakeAsyncStream([phase2])]
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                DashScRequestControls(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(
                    tok, env_cfg, "deepseek_v4", terminate_token_id=42
                ),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 2)
        self.assertEqual(_gen_ids(chunks[0]), [128821, 10, 11])
        self.assertEqual(_gen_ids(chunks[1]), [128822, 271])
        self.assertEqual(_gen_ids(chunks[2]), [20])
        self.assertNotIn(42, visitor.generate_inputs[1].token_ids.cpu().int().tolist())

    async def test_natural_finish_without_close_does_not_trigger_phase2(self) -> None:
        """Phase-1 finishes naturally without ``</think>`` or terminate_token —
        the model dumped the answer entirely into reasoning. After the
        DashLLM-alignment change, phase-2 is NO LONGER fired in this case;
        only the terminate-token-id (DSV4 token 1) abort path enters phase-2.
        The whole phase-1 output is streamed as reasoning content.
        """
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=3, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor([_FakeAsyncStream([phase1])])
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                DashScRequestControls(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        # Single phase-1 enqueue; no second request.
        self.assertEqual(visitor.enqueue_called, 1)
        # Whole phase-1 output streamed as reasoning, no phase-2 chunk follows.
        self.assertEqual(_gen_ids(chunks[0]), [128821, 10, 11, 12])
        for chunk in chunks:
            self.assertNotEqual(chunk.infer_response.id, "trace-real-2")

    async def test_dsv4_natural_close_does_not_trigger_phase2(self) -> None:
        """DSV4 phase-1 with a normal ``</think>`` close emits content in the
        same stream — phase-2 MUST NOT fire. Mirrors DashLLM ``_think.py``
        line 622-628: natural close only updates ``generate_think_token_num``.
        """
        req = self._minimal_request()
        # Stream: think tokens, ``</think>\n\n`` (128822, 271), then answer.
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor(
                        [10, 11, 128822, 271, 200, 201],
                        dtype=torch.int32,
                    ),
                    finished=True,
                    aux_info=AuxInfo(input_len=3, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor([_FakeAsyncStream([phase1])])
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                DashScRequestControls(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        # Exactly one enqueue: phase-1 only, no phase-2 follow-up.
        self.assertEqual(visitor.enqueue_called, 1)
        for chunk in chunks:
            # Phase-2 chunks would carry the ``-2`` suffix on infer_response.id.
            self.assertNotEqual(chunk.infer_response.id, "trace-real-2")

    async def test_dsv4_token1_phase2_reports_metric_once(self) -> None:
        """Phase-2 entry MUST fan out exactly one increment of the DSV4 phase-2
        metric — guarded by ``phase2_triggered`` so the rate matches
        "requests with a think-abort", not "abort tokens seen"."""
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 1, 99], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=3, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20, 21], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=8, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [_FakeAsyncStream([phase1]), _FakeAsyncStream([phase2])]
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )
        env_cfg = _GenerateEnvCfg()

        with patch("rtp_llm.dash_sc.inference.servicer.kmonitor.report") as mock_report:
            await _drain(
                iter_real_model_stream_infer(
                    req,
                    [7, 8, 128821],
                    SamplingParams(),
                    DashScRequestControls(enable_thinking=True),
                    visitor,
                    rtp_llm_request_id=100,
                    echo_prefix_ids=[128821, 198],
                    tokenizer=tok,
                    generate_env_config=env_cfg,
                    think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                    phase2_request_id_factory=lambda: 200,
                )
            )

        from rtp_llm.metrics import AccMetrics

        phase2_calls = [
            call_args
            for call_args in mock_report.call_args_list
            if call_args.args
            and call_args.args[0] is AccMetrics.DASH_SC_DSV4_PHASE2_QPS_METRIC
        ]
        self.assertEqual(len(phase2_calls), 1)
        _metric, value, tags = phase2_calls[0].args
        self.assertEqual(value, 1)
        self.assertEqual(tags["protocol"], "dash_sc_grpc")

    async def test_phase2_grammar_streams_each_backend_chunk_immediately(self) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 1], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2_first = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20, 21], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2_final = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([22], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2_stream = _FakeAsyncStream([phase2_first, phase2_final])
        visitor = _MultiStreamVisitor([_FakeAsyncStream([phase1]), phase2_stream])
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        responses = iter_real_model_stream_infer(
            req,
            [7, 8, 128821],
            SamplingParams(
                response_format=json.dumps({"type": "json_object"}),
            ),
            DashScRequestControls(enable_thinking=True),
            visitor,
            rtp_llm_request_id=100,
            echo_prefix_ids=[128821, 198],
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
            phase2_request_id_factory=lambda: 200,
        )

        first_phase2 = None
        async for response in responses:
            if response.infer_response.id.endswith("-2"):
                first_phase2 = response
                break

        self.assertIsNotNone(first_phase2)
        self.assertEqual(_gen_ids(first_phase2), [20, 21])
        self.assertEqual(phase2_stream._emitted, 1)

        remaining = await _drain(responses)
        self.assertEqual([_gen_ids(chunk) for chunk in remaining], [[22]])

    async def test_phase2_preserves_leading_thinking_then_close(self) -> None:
        """Phase-2 output is streamed unchanged, matching DashLLM ownership."""
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 1, 99], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2_a = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([55, 56], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2_b = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([128822, 271, 20, 21], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [
                _FakeAsyncStream([phase1]),
                _FakeAsyncStream([phase2_a, phase2_b]),
            ]
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                DashScRequestControls(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        # Phase-2 servicer output is a transparent chunk-for-chunk pass-through.
        phase2_chunks = [c for c in chunks if c.infer_response.id.endswith("-2")]
        self.assertEqual(len(phase2_chunks), 2)
        self.assertEqual(
            [_gen_ids(chunk) for chunk in phase2_chunks],
            [[55, 56], [128822, 271, 20, 21]],
        )

    async def test_phase2_preserves_trailing_think_close(self) -> None:
        """Phase-2 trailing close tokens remain backend-owned output."""
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 1], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor(
                        [30, 31, 32, 128822, 271], dtype=torch.int32
                    ),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [_FakeAsyncStream([phase1]), _FakeAsyncStream([phase2])]
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                DashScRequestControls(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        phase2_chunks = [c for c in chunks if c.infer_response.id.endswith("-2")]
        self.assertEqual(len(phase2_chunks), 1)
        self.assertEqual(
            _gen_ids(phase2_chunks[0]),
            [30, 31, 32, 128822, 271],
        )


class IterRealModelStreamInferEchoTest(unittest.IsolatedAsyncioTestCase):
    """Echo-prefill integration for ``iter_real_model_stream_infer``."""

    def _req(self, req_id: str = "echo-trace") -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = req_id
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [2], struct.pack("<2i", 99, 100))
        return req

    async def _run(self, *, input_ids, echo_prefix_ids, upstream_ids):
        chunks_proto = []
        for ids in upstream_ids:
            out = GenerateOutput(
                output_ids=torch.tensor(ids, dtype=torch.int32) if ids else None,
                finished=False,
                aux_info=AuxInfo(input_len=len(input_ids), reuse_len=0),
            )
            chunks_proto.append(GenerateOutputs(generate_outputs=[out]))
        visitor = _FakeVisitor(_FakeAsyncStream(chunks_proto))
        return await _drain(
            iter_real_model_stream_infer(
                self._req(),
                input_ids,
                SamplingParams(),
                DashScRequestControls(),
                visitor,
                rtp_llm_request_id=1,
                echo_prefix_ids=echo_prefix_ids,
            )
        )

    def _gen_ids(self, chunk) -> list[int]:
        infer = chunk.infer_response
        for i, out in enumerate(infer.outputs):
            if out.name == "generated_ids":
                raw = infer.raw_output_contents[i]
                shape = list(out.shape)
                declared_len = shape[-1] if shape else 0
                if declared_len <= 0:
                    return []
                return _unpack_int32_le(raw)
        return []

    async def test_echoes_prefix_when_input_tail_matches(self) -> None:
        chunks = await self._run(
            input_ids=[1, 2, 99, 100],
            echo_prefix_ids=[99, 100],
            upstream_ids=[[3, 4], [5, 6]],
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(self._gen_ids(chunks[0]), [99, 100, 3, 4])
        self.assertEqual(self._gen_ids(chunks[1]), [5, 6])

    async def test_no_echo_when_tail_mismatch(self) -> None:
        chunks = await self._run(
            input_ids=[1, 2, 3],
            echo_prefix_ids=[99, 100],
            upstream_ids=[[3, 4]],
        )
        self.assertEqual(self._gen_ids(chunks[0]), [3, 4])

    async def test_no_echo_when_prefix_empty(self) -> None:
        chunks = await self._run(
            input_ids=[1, 2, 99, 100],
            echo_prefix_ids=[],
            upstream_ids=[[3, 4]],
        )
        self.assertEqual(self._gen_ids(chunks[0]), [3, 4])

    async def test_echo_skips_empty_chunks_and_applies_to_first_non_empty(self) -> None:
        chunks = await self._run(
            input_ids=[99, 100],
            echo_prefix_ids=[99, 100],
            upstream_ids=[[], [3, 4], [5]],
        )
        self.assertEqual(self._gen_ids(chunks[0]), [])
        self.assertEqual(self._gen_ids(chunks[1]), [99, 100, 3, 4])
        self.assertEqual(self._gen_ids(chunks[2]), [5])


class IterRealModelStreamInferStopWordsTest(unittest.IsolatedAsyncioTestCase):
    """``extra_stop_word_ids`` injection (renderer + env extras the dash-sc path
    misses because pre-tokenized input bypasses the OpenAI endpoint)."""

    def _req(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "stop-trace"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [1], struct.pack("<i", 42))
        return req

    async def _captured_stop_words(self, *, extra_stop_word_ids):
        captured: list = []

        class _CaptureVisitor:
            async def enqueue(self, gi):
                captured.append(gi)
                return _FakeAsyncStream([])

        await _drain(
            iter_real_model_stream_infer(
                self._req(),
                [42],
                SamplingParams(),
                DashScRequestControls(),
                _CaptureVisitor(),
                rtp_llm_request_id=1,
                extra_stop_word_ids=extra_stop_word_ids,
            )
        )
        self.assertEqual(len(captured), 1)
        return list(captured[0].generate_config.stop_words_list or [])

    async def test_extra_stop_word_ids_appended(self) -> None:
        sw = await self._captured_stop_words(extra_stop_word_ids=[[154827], [154829]])
        self.assertIn([154827], sw)
        self.assertIn([154829], sw)

    async def test_none_leaves_stop_words_unchanged(self) -> None:
        sw = await self._captured_stop_words(extra_stop_word_ids=None)
        self.assertNotIn([154827], sw)
        self.assertNotIn([154829], sw)

    async def test_dedup_against_request_stop_words(self) -> None:
        """When the request carries a stop_word that's also in extras, the
        merged list keeps a single entry. (Extras themselves are pre-deduped
        at startup by ``_derive_stop_word_ids_list``, so the hot path only
        dedups extras-vs-request, not extras-vs-extras.)"""
        captured: list = []

        class _CaptureVisitor:
            async def enqueue(self, gi):
                captured.append(gi)
                return _FakeAsyncStream([])

        await _drain(
            iter_real_model_stream_infer(
                self._req(),
                [42],
                SamplingParams(stop_words_list=((154827,),)),
                DashScRequestControls(),
                _CaptureVisitor(),
                rtp_llm_request_id=1,
                extra_stop_word_ids=[[154827], [154829]],
            )
        )
        sw = list(captured[0].generate_config.stop_words_list or [])
        self.assertEqual(sw.count([154827]), 1)
        self.assertIn([154829], sw)


async def _areq_iter(requests):
    for r in requests:
        yield r


class _FakeGrpcContext:
    def __init__(self, metadata=()):
        self._metadata = tuple(metadata)
        self.initial_metadata = []

    def invocation_metadata(self):
        return self._metadata

    async def send_initial_metadata(self, metadata):
        self.initial_metadata.append(tuple(metadata))

    def peer(self):
        return "ipv4:1.2.3.4:5678"

    def code(self):
        return None

    def is_active(self):
        return True

    def details(self):
        return ""


class DashScInferenceServicerTest(unittest.IsolatedAsyncioTestCase):
    def _valid_infer_request(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "srv-1"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [1], struct.pack("<i", 42))
        return req

    def _terminal_visitor(self, token_ids: list[int] | None = None) -> _FakeVisitor:
        out = GenerateOutput(
            output_ids=torch.tensor(
                token_ids if token_ids is not None else [9], dtype=torch.int32
            ),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        return _FakeVisitor(_FakeAsyncStream([GenerateOutputs(generate_outputs=[out])]))

    async def test_model_stream_infer_passes_multimodal_payload_to_backend(
        self,
    ) -> None:
        request = self._valid_infer_request()
        request.parameters["payload"].string_param = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "http://example/image.png"},
                                "min_pixels": 128,
                                "max_pixels": 4096,
                            }
                        ],
                    }
                ]
            }
        )
        visitor = self._terminal_visitor()
        servicer = DashScInferenceServicer(backend_visitor=visitor)

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([request]), _FakeGrpcContext())
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(visitor.enqueue_called, 1)
        mm_inputs = visitor.last_generate_input.mm_inputs
        self.assertEqual(len(mm_inputs), 1)
        self.assertEqual(mm_inputs[0].url, "http://example/image.png")
        self.assertEqual(mm_inputs[0].mm_preprocess_config.min_pixels, 128)
        self.assertEqual(mm_inputs[0].mm_preprocess_config.max_pixels, 4096)

    async def test_access_log_records_input_and_generated_ids(self) -> None:
        # Frontend struct path: the emitted access line carries the real token
        # ids, proving they travel servicer -> capture -> emit end to end.
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(
                cost_time=12.5,
                first_token_cost_time=3.5,
                wait_time=1.25,
                iter_count=4,
                input_len=1,
                output_len=1,
                reuse_len=2,
                local_reuse_len=1,
                remote_reuse_len=1,
                aux_string="backend-diagnostic",
            ),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        with patch.object(
            logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME), "info"
        ) as info:
            await _drain(
                servicer.ModelStreamInfer(
                    _areq_iter([self._valid_infer_request()]), _FakeGrpcContext()
                )
            )
        payload = json.loads(info.call_args.args[0])
        self.assertEqual(payload["component_role"], "frontend")
        self.assertEqual(payload["input_ids"], [42])
        self.assertEqual(payload["generated_ids"], [9])
        self.assertEqual(payload["backend_input_token_len"], 1)
        self.assertEqual(payload["output_token_len"], 1)
        self.assertEqual(payload["prompt_cached_token_num"], 2)
        self.assertEqual(payload["aux_info"]["cost_time"], 12.5)
        self.assertEqual(payload["aux_info"]["first_token_cost_time"], 3.5)
        self.assertEqual(payload["aux_info"]["wait_time"], 1.25)
        self.assertEqual(payload["aux_info"]["iter_count"], 4)
        self.assertEqual(payload["aux_info"]["local_reuse_len"], 1)
        self.assertEqual(payload["aux_info"]["remote_reuse_len"], 1)
        self.assertEqual(payload["aux_info"]["aux_string"], "backend-diagnostic")

    async def test_access_log_records_generate_config_role_addrs(self) -> None:
        role_addrs = [
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
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0, role_addrs=role_addrs),
        )

        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        with patch.object(
            logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME), "info"
        ) as info:
            await _drain(
                servicer.ModelStreamInfer(
                    _areq_iter([self._valid_infer_request()]), _FakeGrpcContext()
                )
            )

        payload = json.loads(info.call_args.args[0])
        phase1 = payload["generate_config_role_addrs"]["phase1"]
        self.assertEqual(phase1[0]["role"], "PREFILL")
        self.assertEqual(phase1[1]["role"], "DECODE")
        self.assertEqual(phase1[0]["grpc_port"], 8081)
        self.assertEqual(payload["aux_info"]["role_addrs"], phase1)

    async def test_empty_request_stream_marks_request_done(self) -> None:
        servicer = DashScInferenceServicer(
            backend_visitor=_FakeVisitor(_FakeAsyncStream([]))
        )
        with patch.object(
            logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME), "info"
        ) as info:
            responses = await _drain(
                servicer.ModelStreamInfer(_areq_iter([]), _FakeGrpcContext())
            )

        self.assertEqual(responses, [])
        payload = json.loads(info.call_args.args[0])
        self.assertEqual(payload["req_count"], 0)
        self.assertEqual(payload["request_read_status"], "eof")
        self.assertIsNotNone(payload["request_end_ts_epoch_ms"])

    async def test_debug_score_param_is_control_only(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        req.parameters["dash_sc_debug_score_token_ids"].string_param = "1,2"

        with patch.object(
            logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME), "info"
        ) as info:
            responses = await _drain(
                servicer.ModelStreamInfer(_areq_iter([req]), _FakeGrpcContext())
            )

        self.assertEqual(len(responses), 1)
        self.assertFalse(visitor.last_generate_input.generate_config.return_logits)
        payload = json.loads(info.call_args.args[0])
        self.assertEqual(
            payload["request_controls"]["parameters"]["dash_sc_debug_score_token_ids"],
            "1,2",
        )

    async def test_access_log_emitted_before_rpc_done_metric(self) -> None:
        # 铁律: log first, metrics second — a kmonitor hiccup in report_frontend_rpc_done
        # must never delay or drop the access line, so the finally block must
        # call emit_access_log strictly before report_frontend_rpc_done.
        servicer = DashScInferenceServicer(backend_visitor=self._terminal_visitor())
        order = MagicMock()
        with patch(
            "rtp_llm.dash_sc.inference.servicer.emit_access_log",
            order.emit_access_log,
        ), patch(
            "rtp_llm.dash_sc.inference.servicer.report_frontend_rpc_done",
            order.report_frontend_rpc_done,
        ):
            await _drain(
                servicer.ModelStreamInfer(
                    _areq_iter([self._valid_infer_request()]), _FakeGrpcContext()
                )
            )
        self.assertEqual(
            [c[0] for c in order.mock_calls],
            ["emit_access_log", "report_frontend_rpc_done"],
        )

    def _capture_kmonitor_calls(self):
        # Patch the kmonitor the grpc_metrics leaf functions report through;
        # the servicer calls them by imported name, so the module-global
        # kmonitor reference is the single choke point.
        calls: list[tuple] = []
        patcher = patch.object(grpc_metrics, "kmonitor")
        mock_kmon = patcher.start()
        self.addCleanup(patcher.stop)
        mock_kmon.report.side_effect = lambda m, v=1, tags=None: calls.append(
            (m, v, dict(tags or {}))
        )
        return calls

    @staticmethod
    def _tagged_arrivals(calls):
        return [
            c for c in calls if c[0] == AccMetrics.QPS_METRIC and "priority" in c[2]
        ]

    async def test_priority_arrival_reports_true_qos_exactly_once(self) -> None:
        # Normal RPC with a qos level: the tagged arrival fires once with
        # the true value (after the first frame parse), and the done-tail
        # fallback stays a no-op.
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"x-dashscope-inner-qos-level": 7}
        )
        calls = self._capture_kmonitor_calls()

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), _FakeGrpcContext()))

        tagged = self._tagged_arrivals(calls)
        self.assertEqual(len(tagged), 1)
        self.assertEqual(tagged[0][2]["priority"], "7")
        # No untagged arrival series (single-report).
        untagged = [
            c for c in calls if c[0] == AccMetrics.QPS_METRIC and "priority" not in c[2]
        ]
        self.assertEqual(len(untagged), 0)
        # Success carries the same priority.
        success = [c for c in calls if c[0] == AccMetrics.SUCCESS_QPS_METRIC]
        self.assertEqual(len(success), 1)
        self.assertEqual(success[0][2]["priority"], "7")

    async def test_priority_arrival_falls_back_to_zero_on_parse_error(self) -> None:
        # Request that fails before the first frame parse completes (missing
        # input_ids): the done-tail fallback back-fills the "0" bucket, still
        # exactly one tagged arrival.
        servicer = DashScInferenceServicer(
            backend_visitor=_FakeVisitor(_FakeAsyncStream([]))
        )
        bad = predict_v2_pb2.ModelInferRequest()
        bad.id = "x"
        bad.model_name = "m"
        calls = self._capture_kmonitor_calls()

        await _drain(servicer.ModelStreamInfer(_areq_iter([bad]), _FakeGrpcContext()))

        tagged = self._tagged_arrivals(calls)
        self.assertEqual(len(tagged), 1)
        self.assertEqual(tagged[0][2]["priority"], "0")

    async def test_priority_arrival_falls_back_to_zero_on_frameless_rpc(
        self,
    ) -> None:
        # RPC with no request frame at all: no parse ever runs, the done-tail
        # fallback reports the "0" bucket exactly once.
        servicer = DashScInferenceServicer(
            backend_visitor=_FakeVisitor(_FakeAsyncStream([]))
        )
        calls = self._capture_kmonitor_calls()

        await _drain(servicer.ModelStreamInfer(_areq_iter([]), _FakeGrpcContext()))

        tagged = self._tagged_arrivals(calls)
        self.assertEqual(len(tagged), 1)
        self.assertEqual(tagged[0][2]["priority"], "0")

    async def test_missing_input_ids_error(self) -> None:
        servicer = DashScInferenceServicer(
            backend_visitor=_FakeVisitor(_FakeAsyncStream([]))
        )
        bad = predict_v2_pb2.ModelInferRequest()
        bad.id = "x"
        bad.model_name = "m"
        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([bad]), MagicMock())
        )
        self.assertEqual(len(responses), 1)
        _assert_parameter_error_response(self, responses[0], "input_ids")

    async def test_real_mode_uses_enqueue(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )

        servicer = DashScInferenceServicer(backend_visitor=visitor)
        responses = await _drain(
            servicer.ModelStreamInfer(
                _areq_iter([self._valid_infer_request()]), MagicMock()
            )
        )
        self.assertEqual(len(responses), 1)
        self.assertEqual(visitor.enqueue_called, 1)
        infer = responses[0].infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [9])

    async def test_timeout_request_sets_dashscope_partial_response_metadata(
        self,
    ) -> None:
        servicer = DashScInferenceServicer(backend_visitor=self._terminal_visitor())
        req = self._valid_infer_request()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"x-dashscope-inner-timeout": 1}
        )
        context = _FakeGrpcContext()

        responses = await _drain(servicer.ModelStreamInfer(_areq_iter([req]), context))

        self.assertEqual(len(responses), 1)
        self.assertIn(
            (("x-dashscope-partialresponse", "true"),),
            context.initial_metadata,
        )

    async def test_max_new_tokens_negative_rejected_before_enqueue_repro_p3(
        self,
    ) -> None:
        """max_new_tokens=-1 returns Dash-compatible 400 before enqueue."""
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        _add_input_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", -1))

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 0)
        self.assertEqual(len(responses), 1)
        _assert_parameter_error_response(self, responses[0], "max_new_tokens")

    async def test_bad_structural_tag_shape_returns_parameter_error(
        self,
    ) -> None:
        tag = {}
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            [tag], ensure_ascii=False
        )

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 0)
        self.assertEqual(len(responses), 1)
        _assert_parameter_error_response(self, responses[0], "tool_call_structural_tag")

    async def test_bad_response_format_json_returns_parameter_error(self) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        req.parameters["response_format"].string_param = "not-json"

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 0)
        self.assertEqual(len(responses), 1)
        _assert_parameter_error_response(self, responses[0], "response_format")

    async def test_parser_type_error_is_not_masked_as_parameter_error(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()

        with patch(
            "rtp_llm.dash_sc.inference.servicer.parse_dash_sc_grpc_request",
            side_effect=TypeError("parser bug"),
        ):
            with self.assertRaisesRegex(TypeError, "parser bug"):
                await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 0)

    async def test_explicit_parameter_error_is_returned_before_enqueue(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()

        with patch(
            "rtp_llm.dash_sc.inference.servicer.parse_dash_sc_grpc_request",
            side_effect=DashScParameterError("bad parameter"),
        ):
            responses = await _drain(
                servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
            )

        self.assertEqual(visitor.enqueue_called, 0)
        self.assertEqual(len(responses), 1)
        _assert_parameter_error_response(self, responses[0], "bad parameter")

    async def test_video_frame_list_is_rejected_before_enqueue(self) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        req.parameters["payload"].string_param = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"video": ["http://f1.jpg", "http://f2.jpg"]}],
                    }
                ]
            }
        )

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 0)
        self.assertEqual(len(responses), 1)
        _assert_parameter_error_response(
            self, responses[0], "video frame lists are not supported"
        )

    async def test_openai_compat_max_new_tokens_negative_uses_default(
        self,
    ) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"x-envoy-original-path": "/compatible-mode/v1/chat/completions"}
        )
        _add_input_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", -1))

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(
            visitor.last_generate_input.generate_config.max_new_tokens,
            32000,
        )

    async def test_max_completion_tokens_non_positive_rejected_before_enqueue(
        self,
    ) -> None:
        """max_completion_tokens<=0 is a request error, not an engine abort."""
        for value in (-1, 0):
            with self.subTest(value=value):
                visitor = _FakeVisitor(_FakeAsyncStream([]))
                servicer = DashScInferenceServicer(backend_visitor=visitor)
                req = self._valid_infer_request()
                req.parameters["max_completion_tokens"].int64_param = value

                responses = await _drain(
                    servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
                )

                self.assertEqual(visitor.enqueue_called, 0)
                self.assertEqual(len(responses), 1)
                _assert_parameter_error_response(
                    self, responses[0], "max_completion_tokens"
                )

    async def test_max_completion_tokens_non_positive_rejected_before_legacy_aliases(
        self,
    ) -> None:
        for value in (-1, 0):
            with self.subTest(value=value):
                visitor = _FakeVisitor(_FakeAsyncStream([]))
                servicer = DashScInferenceServicer(backend_visitor=visitor)
                req = self._valid_infer_request()
                _add_input_tensor(
                    req,
                    "max_completion_tokens",
                    "INT32",
                    [1],
                    struct.pack("<i", value),
                )
                _add_input_tensor(
                    req,
                    "max_new_tokens",
                    "INT32",
                    [1],
                    struct.pack("<i", -1),
                )

                responses = await _drain(
                    servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
                )

                self.assertEqual(visitor.enqueue_called, 0)
                self.assertEqual(len(responses), 1)
                _assert_parameter_error_response(
                    self, responses[0], "max_completion_tokens"
                )

    async def test_dash_generation_without_enable_thinking_inherits_adaptive_env(
        self,
    ) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.parameters["max_new_tokens"].int64_param = 3
        req.parameters["result_format"].string_param = "message"
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "x-dashscope-inner-timeout": 1800,
                "user_id": "u1",
            }
        )

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(_gen_ids(responses[0]), [9])
        generate_config = visitor.last_generate_input.generate_config
        self.assertEqual(generate_config.max_new_tokens, 3)
        self.assertFalse(generate_config.in_think_mode)
        self.assertEqual(generate_config.thinking_mode, ThinkingMode.ADAPTIVE)
        self.assertEqual(generate_config.max_thinking_tokens, 32000)

    async def test_dash_generation_without_explicit_mode_inherits_env(
        self,
    ) -> None:
        cases = {
            "disabled": (ThinkingMode.DISABLED, False, None),
            "adaptive": (ThinkingMode.ADAPTIVE, False, "or"),
            "enabled": (ThinkingMode.ENABLED, True, "sequence"),
            "0": (ThinkingMode.DISABLED, False, None),
            "1": (ThinkingMode.ENABLED, True, "sequence"),
        }
        for env_mode, (expected_mode, expected_in_think, grammar_type) in cases.items():
            with self.subTest(env_mode=env_mode):
                visitor = _FakeVisitor(_FakeAsyncStream([]))
                tok = _dsv4_tokenizer()
                env_cfg = _GenerateEnvCfg()
                env_cfg.think_mode = env_mode
                servicer = DashScInferenceServicer(
                    backend_visitor=visitor,
                    tokenizer=tok,
                    generate_env_config=env_cfg,
                    think_runtime=build_think_runtime(tok, env_cfg, "qwen"),
                )
                request = self._valid_infer_request()

                await _drain(
                    servicer.ModelStreamInfer(_areq_iter([request]), MagicMock())
                )

                generate_config = visitor.last_generate_input.generate_config
                self.assertEqual(generate_config.thinking_mode, expected_mode)
                self.assertEqual(generate_config.in_think_mode, expected_in_think)
                if grammar_type is None:
                    self.assertEqual(generate_config.max_thinking_tokens, 0)
                    self.assertIsNone(generate_config.structural_tag)
                else:
                    self.assertEqual(generate_config.max_thinking_tokens, 32000)
                    self.assertEqual(
                        generate_config.structural_tag["format"]["type"],
                        grammar_type,
                    )

    async def test_implicit_adaptive_multi_sequence_falls_back_to_disabled(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        _add_input_tensor(
            req,
            "num_return_sequences",
            "INT32",
            [1],
            struct.pack("<i", 2),
        )

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 1)
        generate_config = visitor.last_generate_input.generate_config
        self.assertEqual(generate_config.thinking_mode, ThinkingMode.DISABLED)
        self.assertFalse(generate_config.in_think_mode)
        self.assertEqual(generate_config.max_thinking_tokens, 0)
        self.assertEqual(generate_config.begin_think_token_ids, [128821, 198])
        self.assertEqual(generate_config.end_think_token_ids, [128822, 271])
        self.assertIsNone(generate_config.structural_tag)

    async def test_dash_generation_enable_thinking_true_without_budget_keeps_thinking(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.parameters["enable_thinking"].bool_param = True

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 1)
        generate_config = visitor.last_generate_input.generate_config
        self.assertTrue(generate_config.in_think_mode)
        self.assertEqual(generate_config.max_thinking_tokens, 32000)
        self.assertEqual(generate_config.structural_tag["format"]["type"], "sequence")

    async def test_dash_generation_response_format_is_finalized_before_enqueue(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.parameters["enable_thinking"].bool_param = True
        req.parameters["response_format"].string_param = json.dumps(
            {"type": "json_object"}
        )

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(len(responses), 1)
        config = visitor.last_generate_input.generate_config
        self.assertIsNone(config.response_format)
        self.assertIsNone(config.json_schema)
        self.assertIsNotNone(config.structural_tag)
        structural_tag = config.structural_tag
        elements = structural_tag["format"]["elements"]
        self.assertEqual(elements[0]["begin"], "<think>\n")
        self.assertEqual(elements[0]["end"], "</think>\n\n")
        self.assertEqual(elements[1]["type"], "json_schema")

    async def test_dash_generation_omits_think_begin_when_input_already_has_it(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            echo_prefix_ids=[128821, 198],
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.inputs[0].shape[:] = [3]
        req.raw_input_contents[0] = struct.pack("<3i", 7, 128821, 198)
        req.parameters["enable_thinking"].bool_param = True
        req.parameters["response_format"].string_param = json.dumps(
            {"type": "json_object"}
        )

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 1)
        config = visitor.last_generate_input.generate_config
        structural_tag = config.structural_tag
        reasoning_tag = structural_tag["format"]["elements"][0]
        self.assertEqual(reasoning_tag["begin"], "")
        self.assertEqual(reasoning_tag["end"], "</think>\n\n")

    async def test_dash_grammar_request_rejects_its_own_multi_sequence(self) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        _add_input_tensor(
            req,
            "num_return_sequences",
            "INT32",
            [1],
            struct.pack("<i", 2),
        )
        req.parameters["response_format"].string_param = json.dumps(
            {"type": "json_object"}
        )

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 0)
        self.assertEqual(len(responses), 1)
        _assert_parameter_error_response(self, responses[0], "num_return_sequences > 1")

    async def test_dash_generation_guided_json_is_finalized_before_enqueue(
        self,
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"其他实体": {"type": "array", "items": {"type": "string"}}},
            "required": ["其他实体"],
        }
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.parameters["enable_thinking"].bool_param = True
        req.parameters["guided_json"].string_param = json.dumps(
            [schema], ensure_ascii=False
        )

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(len(responses), 1)
        config = visitor.last_generate_input.generate_config
        self.assertIsNone(config.response_format)
        self.assertIsNone(config.json_schema)
        structural_tag = config.structural_tag
        final_format = structural_tag["format"]["elements"][-1]
        self.assertEqual(final_format["type"], "json_schema")
        self.assertEqual(final_format["json_schema"], schema)

    async def test_dash_generation_tool_call_structural_tag_is_finalized_before_enqueue(
        self,
    ) -> None:
        tag = {
            "format": {
                "type": "triggered_tags",
                "triggers": ["<｜DSML｜invoke"],
                "tags": [
                    {
                        "type": "tag",
                        "begin": '<｜DSML｜invoke name="get_weather">',
                        "content": {
                            "type": "json_schema",
                            "json_schema": {"type": "object"},
                        },
                        "end": "</｜DSML｜invoke>",
                    }
                ],
            }
        }
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            tag, ensure_ascii=False
        )

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(len(responses), 1)
        config = visitor.last_generate_input.generate_config
        self.assertIsNone(config.response_format)
        canonical_tag = config.structural_tag
        self.assertEqual(canonical_tag["type"], "structural_tag")
        self.assertEqual(canonical_tag["format"], tag["format"])

    async def test_dash_generation_locally_wraps_tool_call_answer_format(
        self,
    ) -> None:
        tag = {
            "format": {
                "type": "triggered_tags",
                "triggers": ["<｜DSML｜invoke"],
                "tags": [
                    {
                        "type": "tag",
                        "begin": '<｜DSML｜invoke name="get_weather">',
                        "content": {
                            "type": "json_schema",
                            "json_schema": {"type": "object"},
                        },
                        "end": "</｜DSML｜invoke>",
                    }
                ],
            }
        }
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.parameters["enable_thinking"].bool_param = True
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            tag, ensure_ascii=False
        )

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 1)
        config = visitor.last_generate_input.generate_config
        canonical_tag = config.structural_tag
        elements = canonical_tag["format"]["elements"]
        self.assertEqual(elements[0]["begin"], "<think>\n")
        self.assertEqual(elements[0]["end"], "</think>\n\n")
        self.assertEqual(elements[1], tag["format"])

    async def test_dash_generation_budget_aliases_without_enable_thinking_are_enabled(
        self,
    ) -> None:
        for param_name in ("thinking_budget", "max_new_think_tokens"):
            with self.subTest(param_name=param_name):
                visitor = _FakeVisitor(_FakeAsyncStream([]))
                tok = _dsv4_tokenizer()
                env_cfg = _GenerateEnvCfg()
                servicer = DashScInferenceServicer(
                    backend_visitor=visitor,
                    tokenizer=tok,
                    generate_env_config=env_cfg,
                    think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                )
                req = self._valid_infer_request()
                req.parameters[param_name].int64_param = 10
                req.parameters["response_format"].string_param = json.dumps(
                    {"type": "json_object"}
                )

                await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

                self.assertEqual(visitor.enqueue_called, 1)
                generate_config = visitor.last_generate_input.generate_config
                self.assertTrue(generate_config.in_think_mode)
                self.assertEqual(generate_config.thinking_mode, ThinkingMode.ENABLED)
                self.assertEqual(generate_config.max_thinking_tokens, 10)
                fixed = generate_config.structural_tag["format"]
                self.assertEqual(fixed["type"], "sequence")
                think_branch = fixed
                self.assertEqual(
                    think_branch["elements"][0]["content"]["max_tokens"], 10
                )
                self.assertEqual(think_branch["elements"][1]["type"], "json_schema")

    async def test_max_completion_tokens_thinking_budget_keeps_backend_limit_repro(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.parameters["max_new_tokens"].int64_param = 200
        req.parameters["max_completion_tokens"].int64_param = 100
        req.parameters["enable_thinking"].bool_param = True
        req.parameters["thinking_budget"].int64_param = 10

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 1)
        generate_config = visitor.last_generate_input.generate_config
        self.assertEqual(generate_config.max_new_tokens, 100)
        self.assertTrue(generate_config.in_think_mode)
        self.assertEqual(generate_config.max_thinking_tokens, 10)

    async def test_real_mode_request_id_matches_generate_request_id(self) -> None:
        """Backend ``GenerateInput.request_id`` follows the same snowflake scheme as HTTP path."""
        from rtp_llm.frontend import request_id_generator as rig

        captured: list[int] = []

        class _CaptureVisitor:
            request_id_factory = None

            def set_request_id_factory(self, factory):
                self.request_id_factory = factory

            async def enqueue(self, gi):
                captured.append(gi.request_id)
                return _FakeAsyncStream([])

        visitor = _CaptureVisitor()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            ip="10.0.0.1",
            port=12345,
            server_id="srv-xyz",
        )
        with patch.object(rig.time, "time", return_value=1_700_000_000.0):
            await _drain(
                servicer.ModelStreamInfer(
                    _areq_iter([self._valid_infer_request()]), MagicMock()
                )
            )
            expected = rig.generate_request_id("10.0.0.1", 12345, "srv-xyz", 1)
            retry_id = visitor.request_id_factory()
            expected_retry_id = rig.generate_request_id("10.0.0.1", 12345, "srv-xyz", 2)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], expected)
        self.assertEqual(retry_id, expected_retry_id)

    async def test_real_mode_passes_invocation_metadata_to_generate_input(self) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        context = MagicMock()
        context.invocation_metadata.return_value = (
            ("User_ID", "u2"),
            ("x-dashscope-apikeyid", "ak2"),
            ("authorization", "secret"),
        )

        await _drain(
            servicer.ModelStreamInfer(
                _areq_iter([self._valid_infer_request()]), context
            )
        )

        self.assertIsNotNone(visitor.last_generate_input)
        self.assertEqual(
            visitor.last_generate_input.headers,
            {"user_id": "u2", "x-dashscope-apikeyid": "ak2"},
        )

    async def test_real_mode_uses_ds_header_attributes_for_backend_controls(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        context = MagicMock()
        context.invocation_metadata.return_value = ()
        request = self._valid_infer_request()
        request.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "x-dashscope-inner-timeout": 1800,
                "x-ds-request-priority": "10",
                "user_id": "u1",
                "x-dashscope-apikeyid": "ak1",
            }
        )
        request.parameters["enable_thinking"].bool_param = False
        request.parameters["thinking_budget"].int64_param = 100

        await _drain(servicer.ModelStreamInfer(_areq_iter([request]), context))

        self.assertIsNotNone(visitor.last_generate_input)
        generate_config = visitor.last_generate_input.generate_config
        self.assertFalse(generate_config.in_think_mode)
        self.assertEqual(generate_config.max_thinking_tokens, 0)
        self.assertEqual(generate_config.end_think_token_ids, [])
        self.assertIsNone(generate_config.structural_tag)
        # 1_800_000 request timeout minus the 5s gateway margin capped by
        # ``_apply_dash_sc_controls_to_generate_config``, so the engine times out
        # before the upstream gateway sends RST_STREAM.
        self.assertEqual(generate_config.timeout_ms, 1_795_000)
        self.assertEqual(generate_config.ttft_timeout_ms, 1_795_000)
        self.assertEqual(generate_config.traffic_reject_priority, 10)
        self.assertEqual(
            visitor.last_generate_input.headers,
            {"user_id": "u1", "x-dashscope-apikeyid": "ak1"},
        )
        # qos_priority must NOT be set when x-dashscope-inner-qos-level
        # is absent from the request.
        self.assertIsNone(generate_config.qos_priority)

    async def test_qos_priority_set_from_ds_header_attributes(self) -> None:
        """dash_sc path must set generate_config.qos_priority from
        x-dashscope-inner-qos-level, mirroring openai_endpoint.py."""
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        context = MagicMock()
        context.invocation_metadata.return_value = ()
        request = self._valid_infer_request()
        request.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "x-dashscope-inner-qos-level": "77",
                "x-ds-request-priority": "10",
                "user_id": "u1",
                "x-dashscope-apikeyid": "ak1",
            }
        )

        await _drain(servicer.ModelStreamInfer(_areq_iter([request]), context))

        self.assertIsNotNone(visitor.last_generate_input)
        generate_config = visitor.last_generate_input.generate_config
        # Channel 2: qos_priority set by _apply_request_overrides
        self.assertEqual(generate_config.qos_priority, 77)
        # traffic_reject_priority still comes from x-ds-request-priority
        self.assertEqual(generate_config.traffic_reject_priority, 10)
        # Channel 1: headers also carry the qos level
        self.assertEqual(
            visitor.last_generate_input.headers.get("x-dashscope-inner-qos-level"),
            "77",
        )
        # _extract_priority returns 77 via either channel
        self.assertEqual(
            MasterClient._extract_priority(visitor.last_generate_input), 77
        )

    async def test_extract_priority_fallback_to_qos_priority(self) -> None:
        """When GenerateInput.headers is empty (e.g. after IPC),
        _extract_priority must fall back to generate_config.qos_priority."""
        gc = GenerateConfig()
        gc.qos_priority = 77
        input_no_headers = GenerateInput(
            request_id=1,
            token_ids=torch.tensor([1, 2], dtype=torch.int),
            mm_inputs=[],
            generate_config=gc,
            headers={},
        )
        self.assertEqual(MasterClient._extract_priority(input_no_headers), 77)

    async def test_extract_priority_returns_50_when_no_priority(self) -> None:
        """When neither headers nor qos_priority carry a value,
        _extract_priority returns the default 50."""
        gc = GenerateConfig()
        self.assertIsNone(gc.qos_priority)
        input_no_priority = GenerateInput(
            request_id=1,
            token_ids=torch.tensor([1, 2], dtype=torch.int),
            mm_inputs=[],
            generate_config=gc,
            headers={},
        )
        self.assertEqual(MasterClient._extract_priority(input_no_priority), 50)

    async def test_int64_input_overflow_is_rejected_at_parse_boundary(
        self,
    ) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "overflow"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT64", [1], struct.pack("<q", 2**40))
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), _FakeGrpcContext())
        )

        self.assertEqual(visitor.enqueue_called, 0)
        self.assertEqual(len(responses), 1)
        self.assertEqual(
            _finish_reason(responses[0]), DASH_ERROR_INTERNAL.finish_reason
        )

    async def test_late_cancel_does_not_overwrite_backend_aux_info(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(
                input_len=1,
                output_len=1,
                reuse_len=7,
                local_reuse_len=3,
                memory_reuse_len=4,
            ),
        )
        cancel = asyncio.CancelledError()
        cancel.aux_info = {
            "input_len": 1,
            "output_len": 0,
            "step_output_len": 0,
            "reuse_len": 0,
        }

        class _LateCancelStream(_FakeAsyncStream):
            async def __anext__(self):
                if self._emitted >= len(self._chunks):
                    raise cancel
                return await super().__anext__()

        visitor = _FakeVisitor(
            _LateCancelStream([GenerateOutputs(generate_outputs=[out])])
        )
        servicer = DashScInferenceServicer(backend_visitor=visitor)

        with patch.object(
            logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME), "info"
        ) as info, self.assertRaises(asyncio.CancelledError):
            await _drain(
                servicer.ModelStreamInfer(
                    _areq_iter([self._valid_infer_request()]), _FakeGrpcContext()
                )
            )

        payload = json.loads(info.call_args.args[0])
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["exc_type"], "CancelledError")
        self.assertEqual(payload["aux_info"]["output_len"], 1)
        self.assertEqual(payload["aux_info"]["reuse_len"], 7)
        self.assertEqual(payload["aux_info"]["local_reuse_len"], 3)
        self.assertEqual(payload["aux_info"]["memory_reuse_len"], 4)


@unittest.skipUnless(tracing.OTEL_AVAILABLE, "opentelemetry not installed")
class DashScInferenceTracingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        tracing.shutdown_telemetry()
        with tracing._state_lock:
            tracing._state = tracing.TelemetryState.UNINITIALIZED
            tracing._provider = None
        tracing.CURRENT_TRACE_STATE.set(None)
        self.exporter = InMemorySpanExporter()
        self.assertTrue(
            tracing.init_telemetry_for_test(self.exporter, role="dash_sc", tp_rank=0)
        )

    async def asyncTearDown(self) -> None:
        tracing.shutdown_telemetry()
        with tracing._state_lock:
            tracing._state = tracing.TelemetryState.UNINITIALIZED
            tracing._provider = None
        tracing.CURRENT_TRACE_STATE.set(None)

    def _finished_spans(self):
        self.assertTrue(tracing._provider.force_flush())
        return self.exporter.get_finished_spans()

    @staticmethod
    def _request(request_id: str) -> predict_v2_pb2.ModelInferRequest:
        request = predict_v2_pb2.ModelInferRequest()
        request.id = request_id
        request.model_name = "default"
        _add_input_tensor(request, "input_ids", "INT32", [1], struct.pack("<i", 42))
        return request

    @staticmethod
    def _terminal_stream():
        output = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(
                input_len=1,
                output_len=4,
                reuse_len=0,
                cost_time=42.5,
                first_token_cost_time=12.5,
            ),
        )
        return _FakeAsyncStream([GenerateOutputs(generate_outputs=[output])])

    class _ClientSpanVisitor:
        def __init__(self, stream_factory):
            self._stream_factory = stream_factory
            self.metadata = []

        async def enqueue(self, _generate_input):
            handle, metadata = tracing.start_client_span(
                "rtp_llm.generate_stream_call", "127.0.0.1:1234"
            )
            self.metadata.append(metadata)
            await asyncio.sleep(0)
            if handle is not None:
                handle.finish()
            return self._stream_factory()

    async def test_upstream_parent_server_client_and_attributes(self) -> None:
        metadata_trace_id_hex = "11111111111111111111111111111111"
        metadata_parent_span_hex = "2222222222222222"
        body_trace_id_hex = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        body_parent_span_hex = "bbbbbbbbbbbbbbbb"
        metadata = (
            (
                "traceparent",
                f"00-{metadata_trace_id_hex}-{metadata_parent_span_hex}-01",
            ),
            ("tracestate", "dash=test"),
            ("x-request-id", "metadata-request"),
        )
        visitor = self._ClientSpanVisitor(self._terminal_stream)
        servicer = DashScInferenceServicer(
            backend_visitor=visitor, ip="127.0.0.1", port=18096, server_id="7"
        )

        request = self._request("upstream")
        request.parameters["traceparent"].string_param = (
            f"00-{body_trace_id_hex}-{body_parent_span_hex}-01"
        )
        request.parameters["tracestate"].string_param = "bailian=e2e"
        request.parameters["baggage"].string_param = (
            "traffic.llm_sdk.scene=chat,test.test=1"
        )
        request.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "traceparent": "00-cccccccccccccccccccccccccccccccc-dddddddddddddddd-01",
                "traceparent_new": f"00-{body_trace_id_hex}-{body_parent_span_hex}-01",
                "x-dashscope-requestid": "body-dashscope-request",
            }
        )

        responses = await _drain(
            servicer.ModelStreamInfer(
                _areq_iter([request]),
                _FakeGrpcContext(metadata),
            )
        )

        self.assertEqual(len(responses), 1)
        spans = {span.name: span for span in self._finished_spans()}
        server = spans["dash_sc.ModelStreamInfer"]
        client = spans["rtp_llm.generate_stream_call"]
        self.assertEqual(server.context.trace_id, int(body_trace_id_hex, 16))
        self.assertEqual(server.parent.span_id, int(body_parent_span_hex, 16))
        self.assertEqual(client.parent.span_id, server.context.span_id)
        self.assertEqual(server.attributes["rtp_llm.trace_context_source"], "body")
        self.assertEqual(server.attributes["scene"], "chat")
        self.assertNotIn("test.test", server.attributes)
        self.assertEqual(
            server.attributes["rtp_llm.external_request_id"], "metadata-request"
        )
        self.assertEqual(server.attributes["gen_ai.span.kind"], "LLM")
        self.assertEqual(server.attributes["gen_ai.operation.name"], "chat")
        self.assertEqual(server.attributes["gen_ai.system"], "rtp_llm")
        self.assertGreaterEqual(
            server.attributes["gen_ai.response.time_to_first_token"], 0.0
        )
        self.assertNotIn("rtp_llm.frontend.time_per_output_token_ms", server.attributes)
        self.assertNotIn("rtp_llm.engine.time_to_first_token_ms", server.attributes)
        self.assertNotIn("rtp_llm.engine.time_per_output_token_ms", server.attributes)
        self.assertEqual(server.attributes["rpc.system"], "grpc")
        self.assertEqual(
            server.attributes["rpc.method"],
            "GRPCInferenceService/ModelStreamInfer",
        )
        self.assertEqual(
            server.attributes["request_id"],
            str(server.attributes["rtp_llm.request_id"]),
        )
        self.assertIn("traceparent", dict(visitor.metadata[0]))
        self.assertEqual(server.status.status_code.name, "OK")
        self.assertEqual(client.status.status_code.name, "OK")

    async def test_parser_error_still_uses_valid_body_parent(self) -> None:
        metadata_trace_id = "11111111111111111111111111111111"
        body_trace_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        body_parent_id = "bbbbbbbbbbbbbbbb"
        metadata = (("traceparent", f"00-{metadata_trace_id}-2222222222222222-01"),)
        request = self._request("parser-error")
        request.parameters["traceparent"].string_param = (
            f"00-{body_trace_id}-{body_parent_id}-01"
        )
        servicer = DashScInferenceServicer(
            backend_visitor=self._ClientSpanVisitor(self._terminal_stream),
            ip="127.0.0.1",
            port=18096,
            server_id="7",
        )

        with patch(
            "rtp_llm.dash_sc.inference.servicer.parse_dash_sc_grpc_request",
            side_effect=DashScParameterError("bad parameter"),
        ):
            await _drain(
                servicer.ModelStreamInfer(
                    _areq_iter([request]), _FakeGrpcContext(metadata)
                )
            )

        server = self._finished_spans()[-1]
        self.assertEqual(server.name, "dash_sc.ModelStreamInfer")
        self.assertEqual(server.context.trace_id, int(body_trace_id, 16))
        self.assertEqual(server.parent.span_id, int(body_parent_id, 16))
        self.assertEqual(server.attributes["rtp_llm.trace_context_source"], "body")

    async def test_invalid_body_falls_back_without_mixing_metadata_tracestate(
        self,
    ) -> None:
        metadata_trace_id = "11111111111111111111111111111111"
        metadata_parent_id = "2222222222222222"
        metadata = (
            (
                "traceparent",
                f"00-{metadata_trace_id}-{metadata_parent_id}-01",
            ),
            ("tracestate", "vendor=metadata"),
        )
        servicer = DashScInferenceServicer(
            backend_visitor=self._ClientSpanVisitor(self._terminal_stream),
            ip="127.0.0.1",
            port=18096,
            server_id="7",
        )
        invalid = self._request("invalid-body")
        invalid.parameters["traceparent"].string_param = "garbage"
        await _drain(
            servicer.ModelStreamInfer(_areq_iter([invalid]), _FakeGrpcContext(metadata))
        )
        fallback = next(
            span
            for span in self._finished_spans()
            if span.name == "dash_sc.ModelStreamInfer"
        )
        self.assertEqual(fallback.context.trace_id, int(metadata_trace_id, 16))
        self.assertEqual(fallback.parent.span_id, int(metadata_parent_id, 16))
        self.assertEqual(
            fallback.attributes["rtp_llm.trace_context_source"], "metadata"
        )

        body_trace_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        body_parent_id = "bbbbbbbbbbbbbbbb"
        body_only = self._request("body-no-state")
        body_only.parameters["traceparent"].string_param = (
            f"00-{body_trace_id}-{body_parent_id}-01"
        )
        await _drain(
            servicer.ModelStreamInfer(
                _areq_iter([body_only]), _FakeGrpcContext(metadata)
            )
        )
        body_span = [
            span
            for span in self._finished_spans()
            if span.name == "dash_sc.ModelStreamInfer"
        ][-1]
        self.assertEqual(body_span.context.trace_id, int(body_trace_id, 16))
        self.assertEqual(list(body_span.context.trace_state), [])

    async def test_no_visible_tokens_omit_frontend_token_latencies(self) -> None:
        state = tracing.start_server_span("dash_sc.ModelStreamInfer", {})
        self.assertIsNotNone(state)
        record = GrpcAccessRecord(
            method="ModelStreamInfer",
            stream_type="bidi_stream",
            peer="test-peer",
            start_ts=10.0,
        )

        _finish_server_trace(state, record, None)

        span = self._finished_spans()[-1]
        self.assertNotIn("gen_ai.response.time_to_first_token", span.attributes)
        self.assertNotIn("rtp_llm.frontend.time_per_output_token_ms", span.attributes)
        self.assertNotIn("rtp_llm.engine.time_to_first_token_ms", span.attributes)
        self.assertNotIn("rtp_llm.engine.time_per_output_token_ms", span.attributes)

    async def test_no_parent_bad_request_and_no_frame_statuses(self) -> None:
        servicer = DashScInferenceServicer(
            backend_visitor=self._ClientSpanVisitor(self._terminal_stream)
        )
        await _drain(
            servicer.ModelStreamInfer(
                _areq_iter([self._request("local-root")]), _FakeGrpcContext()
            )
        )
        local_root = self._finished_spans()[-1]
        self.assertEqual(local_root.name, "dash_sc.ModelStreamInfer")
        self.assertIsNone(local_root.parent)
        self.assertEqual(local_root.status.status_code.name, "OK")
        self.assertEqual(local_root.attributes["rtp_llm.trace_context_source"], "none")

        bad = predict_v2_pb2.ModelInferRequest(id="bad", model_name="default")
        await _drain(servicer.ModelStreamInfer(_areq_iter([bad]), _FakeGrpcContext()))
        bad_span = self._finished_spans()[-1]
        self.assertEqual(bad_span.status.status_code.name, "ERROR")
        self.assertEqual(bad_span.attributes["error.type"], "DASH_ERROR_8")

        await _drain(servicer.ModelStreamInfer(_areq_iter([]), _FakeGrpcContext()))
        no_frame = self._finished_spans()[-1]
        self.assertEqual(no_frame.status.status_code.name, "OK")
        self.assertEqual(no_frame.attributes["rtp_llm.trace_context_source"], "none")
        self.assertNotIn("request_id", no_frame.attributes)

    async def test_metadata_parent_and_external_request_id_fallbacks(self) -> None:
        trace_id_hex = "33333333333333333333333333333333"
        parent_span_hex = "4444444444444444"
        servicer = DashScInferenceServicer(
            backend_visitor=self._ClientSpanVisitor(self._terminal_stream)
        )

        request = self._request("body-request-id")
        request.parameters["ds_header_attributes"].string_param = json.dumps(
            {"x-dashscope-requestid": "body-dashscope-request"}
        )
        await _drain(
            servicer.ModelStreamInfer(
                _areq_iter([request]),
                _FakeGrpcContext(
                    (("traceparent", f"00-{trace_id_hex}-{parent_span_hex}-01"),)
                ),
            )
        )
        span = self._finished_spans()[-1]
        self.assertEqual(span.context.trace_id, int(trace_id_hex, 16))
        self.assertEqual(span.parent.span_id, int(parent_span_hex, 16))
        self.assertEqual(span.attributes["rtp_llm.trace_context_source"], "metadata")
        self.assertEqual(
            span.attributes["rtp_llm.external_request_id"],
            "body-dashscope-request",
        )

        body_id_request = self._request("body-request-id")
        await _drain(
            servicer.ModelStreamInfer(_areq_iter([body_id_request]), _FakeGrpcContext())
        )
        body_id_span = self._finished_spans()[-1]
        self.assertEqual(
            body_id_span.attributes["rtp_llm.external_request_id"],
            "body-request-id",
        )

        trace_only_request = self._request("")
        await _drain(
            servicer.ModelStreamInfer(
                _areq_iter([trace_only_request]),
                _FakeGrpcContext(
                    (("traceparent", f"00-{trace_id_hex}-{parent_span_hex}-01"),)
                ),
            )
        )
        trace_only_span = self._finished_spans()[-1]
        self.assertNotIn("rtp_llm.external_request_id", trace_only_span.attributes)

    def test_external_request_id_all_sources_priority_and_length_cap(self) -> None:
        request = self._request("request-body")
        request.parameters["ds_header_attributes"].string_param = json.dumps(
            {"x-dashscope-requestid": "ds-body"}
        )
        self.assertEqual(
            extract_span_external_request_id(
                (
                    ("dashscope-request-id", "dashscope"),
                    ("x-request-id", "generic"),
                    ("x-dashscope-request-id", "dashscope-specific"),
                ),
                request,
            ),
            "dashscope-specific",
        )
        self.assertEqual(
            extract_span_external_request_id((("x-request-id", "generic"),), request),
            "generic",
        )
        self.assertEqual(
            extract_span_external_request_id(
                (("dashscope-request-id", "dashscope"),), request
            ),
            "dashscope",
        )
        self.assertEqual(extract_span_external_request_id((), request), "ds-body")
        del request.parameters["ds_header_attributes"]
        self.assertEqual(extract_span_external_request_id((), request), "request-body")

        oversized = "x" * 256
        for metadata, use_ds_body in (
            ((("x-dashscope-request-id", oversized),), False),
            ((("x-request-id", oversized),), False),
            ((("dashscope-request-id", oversized),), False),
            ((), True),
            ((), False),
        ):
            candidate = self._request(oversized)
            if use_ds_body:
                candidate.parameters["ds_header_attributes"].string_param = json.dumps(
                    {"x-dashscope-requestid": oversized}
                )
            self.assertEqual(
                extract_span_external_request_id(metadata, candidate), "x" * 128
            )

    async def test_cancelled_stream_sets_cancelled_status(self) -> None:
        class _CancelVisitor:
            async def enqueue(self, _generate_input):
                async def stream():
                    raise asyncio.CancelledError()
                    yield  # pragma: no cover

                return stream()

        servicer = DashScInferenceServicer(backend_visitor=_CancelVisitor())
        with self.assertRaises(asyncio.CancelledError):
            await _drain(
                servicer.ModelStreamInfer(
                    _areq_iter([self._request("cancel")]), _FakeGrpcContext()
                )
            )

        span = self._finished_spans()[-1]
        self.assertEqual(span.status.status_code.name, "ERROR")
        self.assertEqual(span.attributes["error.type"], "Cancelled")

    async def test_consumer_close_closes_backend_stream_immediately(self) -> None:
        chunk = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([42], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=2, reuse_len=0),
                )
            ]
        )
        backend_stream = _FakeAsyncStream([chunk])
        servicer = DashScInferenceServicer(backend_visitor=_FakeVisitor(backend_stream))
        response_iter = servicer.ModelStreamInfer(
            _areq_iter([self._request("consumer-close")]), _FakeGrpcContext()
        )

        await response_iter.__anext__()
        self.assertFalse(backend_stream.aclose_called)
        await response_iter.aclose()

        self.assertTrue(backend_stream.aclose_called)

    async def test_concurrent_stream_contexts_do_not_cross(self) -> None:
        visitor = self._ClientSpanVisitor(self._terminal_stream)
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        trace_ids = (
            "33333333333333333333333333333333",
            "44444444444444444444444444444444",
        )

        async def run(index: int) -> None:
            metadata = (
                (
                    "traceparent",
                    f"00-{trace_ids[index]}-{index + 1:016x}-01",
                ),
            )
            await _drain(
                servicer.ModelStreamInfer(
                    _areq_iter([self._request(f"concurrent-{index}")]),
                    _FakeGrpcContext(metadata),
                )
            )

        await asyncio.gather(run(0), run(1))

        spans_by_trace = {}
        for span in self._finished_spans():
            spans_by_trace.setdefault(span.context.trace_id, []).append(span)
        for trace_id_hex in trace_ids:
            spans = spans_by_trace[int(trace_id_hex, 16)]
            self.assertEqual(len(spans), 2)
            server = next(s for s in spans if s.name == "dash_sc.ModelStreamInfer")
            client = next(s for s in spans if s.name == "rtp_llm.generate_stream_call")
            self.assertEqual(client.parent.span_id, server.context.span_id)

    async def test_backend_error_frame_sets_span_error_type(self) -> None:
        class _BoomVisitor:
            async def enqueue(self, _generate_input):
                raise RuntimeError("backend down")

        servicer = DashScInferenceServicer(backend_visitor=_BoomVisitor())
        await _drain(
            servicer.ModelStreamInfer(
                _areq_iter([self._request("backend-error")]), _FakeGrpcContext()
            )
        )

        span = self._finished_spans()[-1]
        self.assertEqual(span.name, "dash_sc.ModelStreamInfer")
        self.assertEqual(span.status.status_code.name, "ERROR")
        self.assertEqual(span.attributes["error.type"], "DASH_ERROR_19")

    async def test_prologue_reporting_failure_still_ends_server_span(self) -> None:
        """A throwing prologue reporting call must not leak the SERVER span.

        ``emit_query_log`` runs inside the handler ``try`` with no exception
        guard of its own: the ``finally`` still ends the span and clears
        CURRENT_TRACE_STATE.
        """
        servicer = DashScInferenceServicer(
            backend_visitor=self._ClientSpanVisitor(self._terminal_stream)
        )
        with patch(
            "rtp_llm.dash_sc.inference.servicer.emit_query_log",
            side_effect=RuntimeError("kmonitor down"),
        ):
            with self.assertRaises(RuntimeError):
                await _drain(
                    servicer.ModelStreamInfer(
                        _areq_iter([self._request("prologue-boom")]),
                        _FakeGrpcContext(),
                    )
                )

        span = self._finished_spans()[-1]
        self.assertEqual(span.name, "dash_sc.ModelStreamInfer")
        self.assertTrue(span.end_time)
        self.assertEqual(span.status.status_code.name, "ERROR")
        self.assertIsNone(tracing.CURRENT_TRACE_STATE.get())


if __name__ == "__main__":
    unittest.main()
