import asyncio
import json
import struct
import sys
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

# Mock the ops module to avoid CUDA dependency in this unit test
# This MUST be at the very top before any other imports, even before unittest
mock_ops = MagicMock()
mock_comm = MagicMock()
mock_nccl_op = MagicMock()
mock_compute_ops = MagicMock()


class _FakeRoleType(Enum):
    PDFUSION = 0
    PREFILL = 1
    DECODE = 2
    VIT = 3
    FRONTEND = 4


mock_comm.nccl_op = mock_nccl_op
mock_ops.comm = mock_comm
mock_ops.compute_ops = mock_compute_ops
mock_ops.RoleType = _FakeRoleType
sys.modules["rtp_llm.ops"] = mock_ops
sys.modules["rtp_llm.ops.comm"] = mock_comm
sys.modules["rtp_llm.ops.compute_ops"] = mock_compute_ops
sys.modules["rtp_llm.ops.comm.nccl_op"] = mock_nccl_op

import unittest
from typing import AsyncGenerator
from unittest import TestCase, main

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import (
    GenerateConfig,
    RoleAddr,
    RoleType,
    ThinkingMode,
)
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.response_format_compiler import ReasoningFormat
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    ModelRpcClient,
    StreamState,
    _trans_rpc_error_code,
    trans_input,
    trans_output,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    BatchGenerateOutputsPB,
    ErrorCodePB,
    GenerateConfigPB,
    GenerateInputPB,
    GenerateOutputsPB,
    RoleAddrPB,
    TensorPB,
)
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutputs,
    RequestInfo,
)
from rtp_llm.utils.grpc_util import trans_tensor


class FakeStub:
    async def GenerateStreamCall(self, input: GenerateInputPB, timeout=None):
        # 1. 第一个响应：包含第一个生成的 token
        outputs_pb1 = GenerateOutputsPB()
        output_pb1 = outputs_pb1.flatten_output
        output_pb1.output_ids.data_type = TensorPB.DataType.INT32
        output_pb1.output_ids.shape.extend([1, 1])
        output_pb1.output_ids.int32_data = struct.pack("<i", 0)
        aux_info = output_pb1.aux_info.add()
        aux_info.iter_count = 1
        aux_info.output_len = 1
        output_pb1.logits.data_type = TensorPB.DataType.FP32
        output_pb1.logits.shape.extend([1, 1, 2])
        output_pb1.logits.fp32_data = struct.pack("<ff", 0.0, 0.0)
        output_pb1.finished.extend([False])
        yield outputs_pb1

        # 2. 第二个响应：包含累积的两个 token
        outputs_pb2 = GenerateOutputsPB()
        output_pb2 = outputs_pb2.flatten_output
        output_pb2.output_ids.data_type = TensorPB.DataType.INT32
        output_pb2.output_ids.shape.extend([1, 2])
        output_pb2.output_ids.int32_data = struct.pack("<ii", 0, 1)
        aux_info2 = output_pb2.aux_info.add()
        aux_info2.iter_count = 2
        aux_info2.output_len = 2
        aux_info2.speculative_draft_rounds = 4
        aux_info2.speculative_accepted_tokens_per_pos.extend([3, 2, 1])
        output_pb2.logits.data_type = TensorPB.DataType.FP32
        output_pb2.logits.shape.extend([1, 1, 2])
        output_pb2.logits.fp32_data = struct.pack("<ff", 0.1, 0.2)
        output_pb2.finished.extend([False])
        yield outputs_pb2

        # 3. 最终响应：标记结束，并携带最后一个状态
        outputs_pb3 = GenerateOutputsPB()
        output_pb3_item = outputs_pb3.flatten_output
        output_pb3_item.CopyFrom(output_pb2)
        output_pb3_item.finished[0] = True
        yield outputs_pb3


class FakeModelRpcClient(ModelRpcClient):
    def __init__(self):
        # Call parent __init__ with minimal required parameters
        super().__init__(
            [],  # addresses: empty list for fake client
            {},  # client_config: empty dict for fake client
            0,  # max_rpc_timeout_ms
            False,  # decode_entrance
        )
        self.stub = FakeStub()

    async def enqueue(
        self, input_py: GenerateInput
    ) -> AsyncGenerator[GenerateOutputs, None]:
        input_pb = trans_input(input_py)
        stream_state = StreamState()

        async for response_pb in self.stub.GenerateStreamCall(input_pb):
            yield trans_output(input_py, response_pb, stream_state)


class _FakeResponseIterator:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.cancelled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._responses)
        except StopIteration:
            raise StopAsyncIteration

    def cancel(self):
        self.cancelled = True


class _FakeChannelPool:
    def __init__(self):
        self.targets = []

    async def get(self, target_address):
        self.targets.append(target_address)
        return object()


class _RoutingStub:
    def __init__(self, fetch_responses=None, generate_responses=None):
        self.fetch_iterator = _FakeResponseIterator(fetch_responses or [])
        self.generate_iterator = _FakeResponseIterator(generate_responses or [])
        self.fetch_calls = []
        self.generate_calls = []

    def FetchResponse(self, request, **kwargs):
        self.fetch_calls.append((request, kwargs))
        return self.fetch_iterator

    def GenerateStreamCall(self, request, **kwargs):
        self.generate_calls.append((request, kwargs))
        return self.generate_iterator


def _make_response(finished=True, output_len=None):
    outputs_pb = GenerateOutputsPB()
    outputs_pb.flatten_output.finished.extend([finished])
    if output_len is not None:
        outputs_pb.flatten_output.aux_info.add().output_len = output_len
    return outputs_pb


def _prefill_role_addr(ip="prefill", grpc_port=9000):
    return RoleAddr(role=RoleType.PREFILL, ip=ip, http_port=8000, grpc_port=grpc_port)


def _decode_role_addr(ip="decode", grpc_port=9001):
    return RoleAddr(role=RoleType.DECODE, ip=ip, http_port=8001, grpc_port=grpc_port)


class ModelRpcClientTest(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # self.client = FakeModelRpcClient()

    @staticmethod
    async def _run(client, input):
        responses = []
        async for res in client.enqueue(input):
            responses.extend(res.generate_outputs)
        return responses

    def test_trans_input_serializes_typed_request_info(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([1, 2]),
            mm_inputs=[],
            generate_config=GenerateConfig(),
            headers={"x-trace-id": "header-trace"},
            request_info=RequestInfo(
                frontend_ip="frontend-ip",
                dash_ip="dash-ip",
                trace_id="request-trace",
                request_id="request-id",
                source_role="frontend",
            ),
        )

        request_info_pb = trans_input(input_py).request_info

        self.assertEqual(request_info_pb.frontend_ip, "frontend-ip")
        self.assertEqual(request_info_pb.dash_ip, "dash-ip")
        self.assertEqual(request_info_pb.trace_id, "request-trace")
        self.assertEqual(request_info_pb.request_id, "request-id")
        self.assertEqual(request_info_pb.source_role, "frontend")

    def test_trans_input_fills_request_info_from_typed_headers(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([1, 2]),
            mm_inputs=[],
            generate_config=GenerateConfig(),
            headers={
                "x-trace-id": "header-trace",
                "x-request-id": "header-request-id",
            },
        )

        request_info_pb = trans_input(input_py).request_info

        self.assertEqual(request_info_pb.trace_id, "header-trace")
        self.assertEqual(request_info_pb.request_id, "header-request-id")

    @staticmethod
    def _make_generate_input(generate_config: GenerateConfig) -> GenerateInput:
        return GenerateInput(
            request_id=1,
            token_ids=torch.tensor([1], dtype=torch.int32),
            mm_inputs=[],
            generate_config=generate_config,
        )

    @staticmethod
    def _make_batch_client() -> ModelRpcClient:
        channel_pool = MagicMock()
        channel_pool.get = AsyncMock(return_value=MagicMock())
        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.GrpcHostChannelPool",
            return_value=channel_pool,
        ):
            return ModelRpcClient(
                addresses=["localhost:1234"],
                client_config={},
                max_rpc_timeout_ms=0,
                decode_entrance=False,
            )

    def test_thinking_mode_values_match_proto_contract(self):
        cases = (
            (ThinkingMode.UNSPECIFIED, GenerateConfigPB.THINKING_MODE_UNSPECIFIED),
            (ThinkingMode.DISABLED, GenerateConfigPB.THINKING_MODE_DISABLED),
            (ThinkingMode.ADAPTIVE, GenerateConfigPB.THINKING_MODE_ADAPTIVE),
            (ThinkingMode.ENABLED, GenerateConfigPB.THINKING_MODE_ENABLED),
        )

        for python_mode, proto_mode in cases:
            with self.subTest(mode=python_mode):
                self.assertEqual(int(python_mode), proto_mode)

    def test_trans_input_writes_thinking_mode(self):
        cases = (
            (ThinkingMode.UNSPECIFIED, GenerateConfigPB.THINKING_MODE_UNSPECIFIED),
            (ThinkingMode.DISABLED, GenerateConfigPB.THINKING_MODE_DISABLED),
            (ThinkingMode.ADAPTIVE, GenerateConfigPB.THINKING_MODE_ADAPTIVE),
            (ThinkingMode.ENABLED, GenerateConfigPB.THINKING_MODE_ENABLED),
        )

        for python_mode, proto_mode in cases:
            with self.subTest(mode=python_mode):
                config = GenerateConfig(thinking_mode=python_mode)
                config_before_rpc = config.model_dump()

                input_pb = trans_input(self._make_generate_input(config))

                self.assertEqual(config.model_dump(), config_before_rpc)
                self.assertEqual(input_pb.generate_config.thinking_mode, proto_mode)

    def _round_trip_generate_config(self, config: GenerateConfig):
        input_pb = trans_input(self._make_generate_input(config))
        round_tripped_input_pb = GenerateInputPB()
        round_tripped_input_pb.ParseFromString(input_pb.SerializeToString())
        return round_tripped_input_pb.generate_config

    def test_trans_input_serializes_resolved_positive_default_max_new_tokens(self):
        config = GenerateConfig()

        self.assertGreater(config.max_new_tokens, 0)
        config_pb = self._round_trip_generate_config(config)

        self.assertEqual(config_pb.max_new_tokens, config.max_new_tokens)
        self.assertFalse(config_pb.prefill_only)
        self.assertIn(
            "max_new_tokens", {field.name for field, _ in config_pb.ListFields()}
        )

    def test_trans_input_round_trips_zero_max_new_tokens_with_prefill_flag(self):
        config_pb = self._round_trip_generate_config(GenerateConfig(max_new_tokens=0))

        self.assertEqual(config_pb.max_new_tokens, 0)
        self.assertTrue(config_pb.prefill_only)
        fields = {field.name for field, _ in config_pb.ListFields()}
        self.assertNotIn("max_new_tokens", fields)
        self.assertIn("prefill_only", fields)

    def test_trans_input_round_trips_prompt_logits_with_zero_max_new_tokens(self):
        config_pb = self._round_trip_generate_config(
            GenerateConfig(max_new_tokens=0, return_prompt_logits=True)
        )

        self.assertTrue(config_pb.return_prompt_logits)
        self.assertEqual(config_pb.max_new_tokens, 1)
        self.assertFalse(config_pb.prefill_only)

    def test_trans_input_round_trips_prompt_logits_after_zero_token_update(self):
        config = GenerateConfig(max_new_tokens=1, return_prompt_logits=True)

        config.update({"max_new_tokens": 0})
        config_pb = self._round_trip_generate_config(config)

        self.assertTrue(config_pb.return_prompt_logits)
        self.assertEqual(config_pb.max_new_tokens, 1)
        self.assertFalse(config_pb.prefill_only)

    def test_trans_input_round_trips_explicit_positive_max_new_tokens(self):
        config_pb = self._round_trip_generate_config(GenerateConfig(max_new_tokens=7))

        self.assertEqual(config_pb.max_new_tokens, 7)
        self.assertFalse(config_pb.prefill_only)
        self.assertIn(
            "max_new_tokens", {field.name for field, _ in config_pb.ListFields()}
        )

    def test_trans_input_round_trips_return_prompt_logits(self):
        config_pb = self._round_trip_generate_config(
            GenerateConfig(
                return_prompt_logits=True,
                prompt_logits_top_k=17,
                prompt_logits_start=2,
                prompt_logits_end=5,
                return_target_logprob=False,
            )
        )

        self.assertTrue(config_pb.return_prompt_logits)
        self.assertEqual(config_pb.prompt_logits_top_k, 17)
        self.assertEqual(config_pb.prompt_logits_start, 2)
        self.assertEqual(config_pb.prompt_logits_end, 5)
        self.assertFalse(config_pb.return_target_logprob)

    def test_rpc_error_code_mapping_covers_every_declared_enum(self):
        for error_code in ErrorCodePB.values():
            with self.subTest(error_code=ErrorCodePB.Name(error_code)):
                exception_type = _trans_rpc_error_code(error_code)
                self.assertIsInstance(exception_type, ExceptionType)
                if error_code not in (
                    ErrorCodePB.NONE_ERROR,
                    ErrorCodePB.UNKNOWN_ERROR,
                ):
                    self.assertNotEqual(exception_type, ExceptionType.UNKNOWN_ERROR)

    def test_rpc_error_code_overrides_preserve_client_semantics(self):
        self.assertEqual(
            _trans_rpc_error_code(ErrorCodePB.CANCELLED),
            ExceptionType.CANCELLED_ERROR,
        )
        self.assertEqual(
            _trans_rpc_error_code(ErrorCodePB.P2P_CONNECTOR_WORKER_READ_CANCELED),
            ExceptionType.P2P_CONNECTOR_WORKER_READ_CANCELLED,
        )

    def test_batch_enqueue_maps_error_codes_and_handles_missing_messages(self):
        cases = (
            (
                ErrorCodePB.INVALID_PARAMS,
                "batch error",
                ExceptionType.INVALID_PARAMS,
                "batch error",
            ),
            (
                ErrorCodePB.CANCELLED,
                "",
                ExceptionType.CANCELLED_ERROR,
                "CANCELLED",
            ),
            (
                ErrorCodePB.EXECUTION_EXCEPTION,
                "handler threw",
                ExceptionType.EXECUTION_EXCEPTION,
                "handler threw",
            ),
            (
                ErrorCodePB.NONE_ERROR,
                "batch error",
                ExceptionType.UNKNOWN_ERROR,
                "batch error",
            ),
        )

        for (
            error_code,
            error_message,
            expected_exception_type,
            expected_message,
        ) in cases:
            with self.subTest(error_code=error_code, error_message=error_message):
                response = BatchGenerateOutputsPB()
                result = response.results.add()
                result.error_info.error_code = error_code
                result.error_info.error_message = error_message

                client = self._make_batch_client()

                stub = MagicMock()
                stub.BatchGenerateCall = AsyncMock(return_value=response)
                with (
                    patch(
                        "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                        return_value=stub,
                    ),
                    self.assertRaises(FtRuntimeException) as context,
                ):
                    asyncio.run(
                        client.batch_enqueue(
                            [self._make_generate_input(GenerateConfig())]
                        )
                    )

                self.assertEqual(
                    context.exception.exception_type, expected_exception_type
                )
                self.assertEqual(
                    context.exception.message,
                    f"batch item 0 failed: {expected_message}",
                )

    def test_batch_enqueue_stops_before_later_items_after_first_error(self):
        response = BatchGenerateOutputsPB()
        failed = response.results.add()
        failed.error_info.error_code = ErrorCodePB.INVALID_PARAMS
        failed.error_info.error_message = "first item failed"
        response.results.add()
        client = self._make_batch_client()
        stub = MagicMock()
        stub.BatchGenerateCall = AsyncMock(return_value=response)

        with (
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                return_value=stub,
            ),
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.trans_output"
            ) as trans_output_mock,
            self.assertRaises(FtRuntimeException) as context,
        ):
            asyncio.run(
                client.batch_enqueue(
                    [
                        self._make_generate_input(GenerateConfig()),
                        self._make_generate_input(GenerateConfig()),
                    ]
                )
            )

        self.assertEqual(context.exception.exception_type, ExceptionType.INVALID_PARAMS)
        trans_output_mock.assert_not_called()

    def test_batch_prefill_only_rejects_generated_tokens(self):
        response = BatchGenerateOutputsPB()
        output = response.results.add().final_output.flatten_output
        output.finished.append(True)
        output.aux_info.add().output_len = 1
        client = self._make_batch_client()
        stub = MagicMock()
        stub.BatchGenerateCall = AsyncMock(return_value=response)

        with (
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                return_value=stub,
            ),
            self.assertLogs(
                "rtp_llm.cpp.model_rpc.model_rpc_client", level="ERROR"
            ) as logs,
            self.assertRaises(FtRuntimeException) as context,
        ):
            asyncio.run(
                client.batch_enqueue(
                    [self._make_generate_input(GenerateConfig(max_new_tokens=0))]
                )
            )

        self.assertEqual(
            context.exception.exception_type, ExceptionType.EXECUTION_EXCEPTION
        )
        self.assertIn("output_len=1", context.exception.message)
        log_text = "\n".join(logs.output)
        self.assertIn("request_id=1", log_text)
        self.assertIn("aux_info[0]", log_text)

    def test_batch_prefill_only_rejects_missing_aux_info(self):
        response = BatchGenerateOutputsPB()
        response.results.add()
        client = self._make_batch_client()
        stub = MagicMock()
        stub.BatchGenerateCall = AsyncMock(return_value=response)

        with (
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                return_value=stub,
            ),
            self.assertLogs(
                "rtp_llm.cpp.model_rpc.model_rpc_client", level="ERROR"
            ) as logs,
            self.assertRaises(FtRuntimeException) as context,
        ):
            asyncio.run(
                client.batch_enqueue(
                    [self._make_generate_input(GenerateConfig(max_new_tokens=0))]
                )
            )

        self.assertEqual(
            context.exception.exception_type, ExceptionType.EXECUTION_EXCEPTION
        )
        self.assertIn("completed without aux_info", context.exception.message)
        self.assertIn("request_id=1", "\n".join(logs.output))

    def test_batch_enqueue_maps_unknown_numeric_error_code_without_message(self):
        unknown_error_code = 2_000_000_000
        response = BatchGenerateOutputsPB()
        result = response.results.add()
        result.error_info.error_code = unknown_error_code

        client = self._make_batch_client()

        stub = MagicMock()
        stub.BatchGenerateCall = AsyncMock(return_value=response)
        with (
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                return_value=stub,
            ),
            self.assertRaises(FtRuntimeException) as context,
        ):
            asyncio.run(
                client.batch_enqueue([self._make_generate_input(GenerateConfig())])
            )

        self.assertEqual(context.exception.exception_type, ExceptionType.UNKNOWN_ERROR)
        self.assertIn(
            f"UNRECOGNIZED_ERROR_CODE_{unknown_error_code}",
            context.exception.message,
        )

    def test_trans_tensor_preserves_scalar_and_zero_numel_shapes(self):
        cases = (
            ("scalar", [], struct.pack("<i", 7), (), 7),
            ("empty_vector", [0], b"", (0,), []),
            ("empty_matrix", [0, 4], b"", (0, 4), []),
            ("prefill_only_output", [1, 1, 0], b"", (1, 1, 0), [[[]]]),
        )

        for name, shape, payload, expected_shape, expected_value in cases:
            with self.subTest(name=name):
                tensor_pb = TensorPB(
                    data_type=TensorPB.DataType.INT32, int32_data=payload
                )
                tensor_pb.shape.extend(shape)

                tensor = trans_tensor(tensor_pb)

                self.assertEqual(tensor.dtype, torch.int32)
                self.assertEqual(tuple(tensor.shape), expected_shape)
                self.assertEqual(tensor.tolist(), expected_value)

    def test_trans_tensor_preserves_default_empty_sentinel(self):
        tensor = trans_tensor(TensorPB())

        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(tensor.shape), (0,))

    def test_trans_tensor_rejects_invalid_shape_and_payload(self):
        cases = (
            ("negative_dimension", [1, -1], b"", "non-negative"),
            ("short_payload", [2], struct.pack("<i", 1), "expected 8 bytes, got 4"),
            (
                "long_scalar_payload",
                [],
                struct.pack("<ii", 1, 2),
                "expected 4 bytes, got 8",
            ),
        )

        for name, shape, payload, expected_message in cases:
            with self.subTest(name=name):
                tensor_pb = TensorPB(
                    data_type=TensorPB.DataType.INT32, int32_data=payload
                )
                tensor_pb.shape.extend(shape)

                with self.assertRaisesRegex(ValueError, expected_message):
                    trans_tensor(tensor_pb)

    def test_trans_output_accepts_prefill_only_empty_output_ids(self):
        input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))
        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.append(True)
        output_pb.aux_info.add().output_len = 0
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([1, 1, 0])

        outputs = trans_output(input_py, outputs_pb, StreamState())

        self.assertEqual(len(outputs.generate_outputs), 1)
        output_ids = outputs.generate_outputs[0].output_ids
        self.assertEqual(output_ids.dtype, torch.int32)
        self.assertEqual(tuple(output_ids.shape), (1, 0))
        self.assertEqual(output_ids.numel(), 0)

    def test_trans_output_round_trips_prompt_logits_payload(self):
        input_py = self._make_generate_input(
            GenerateConfig(return_prompt_logits=True, prompt_logits_top_k=3)
        )
        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.append(True)
        prompt_logits = output_pb.prompt_logits
        prompt_logits.topk_logprobs.data_type = TensorPB.DataType.FP32
        prompt_logits.topk_logprobs.shape.extend([2, 3])
        prompt_logits.topk_logprobs.fp32_data = struct.pack(
            "<6f", -0.1, -0.2, -0.3, -0.4, -0.5, -0.6
        )
        prompt_logits.topk_token_ids.data_type = TensorPB.DataType.INT32
        prompt_logits.topk_token_ids.shape.extend([2, 3])
        prompt_logits.topk_token_ids.int32_data = struct.pack("<6i", 1, 2, 3, 4, 5, 6)
        prompt_logits.target_logprobs.data_type = TensorPB.DataType.FP32
        prompt_logits.target_logprobs.shape.extend([2])
        prompt_logits.target_logprobs.fp32_data = struct.pack("<2f", -1.5, -2.5)
        prompt_logits.start_pos = 2
        prompt_logits.end_pos = 4

        outputs = trans_output(input_py, outputs_pb, StreamState())

        self.assertEqual(len(outputs.generate_outputs), 1)
        actual = outputs.generate_outputs[0].prompt_logits
        self.assertIsNotNone(actual)
        torch.testing.assert_close(
            actual["topk_logprobs"],
            torch.tensor([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]]),
        )
        torch.testing.assert_close(
            actual["topk_token_ids"],
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            actual["target_logprobs"], torch.tensor([-1.5, -2.5])
        )
        self.assertEqual(actual["start_pos"], 2)
        self.assertEqual(actual["end_pos"], 4)

    def test_trans_input_writes_typed_grammar_fields_consistently(self):
        grammar_fields = ("json_schema", "regex", "ebnf", "structural_tag")
        cases = [
            (
                "json_schema",
                {"type": "object"},
                '{"type":"object"}',
                lambda pb: pb.json_schema,
            ),
            ("regex", r"[a-z]+", r"[a-z]+", lambda pb: pb.regex),
            ("ebnf", 'root ::= "a"', 'root ::= "a"', lambda pb: pb.ebnf),
            (
                "structural_tag",
                {
                    "type": "structural_tag",
                    "format": {"type": "regex", "pattern": "a"},
                },
                '{"type":"structural_tag","format":{"type":"regex","pattern":"a"}}',
                lambda pb: pb.structural_tag,
            ),
        ]

        for field, value, expected, field_value in cases:
            with self.subTest(field=field):
                config = GenerateConfig(**{field: value})
                config_before_rpc = config.model_dump()
                input_pb = trans_input(self._make_generate_input(config))

                self.assertEqual(config.model_dump(), config_before_rpc)
                self.assertTrue(input_pb.generate_config.HasField(field))
                self.assertEqual(field_value(input_pb.generate_config).value, expected)
                for removed_field in (
                    "response_format",
                    "grammar_terminate_without_stop_token",
                ):
                    self.assertNotIn(
                        removed_field,
                        input_pb.generate_config.DESCRIPTOR.fields_by_name,
                    )
                for other_field in grammar_fields:
                    if other_field != field:
                        self.assertFalse(input_pb.generate_config.HasField(other_field))

    def test_trans_input_does_not_reapply_reasoning_envelope(self):
        config = GenerateConfig(
            response_format={"type": "json_object"},
            in_think_mode=True,
            end_think_token_ids=[7],
            max_thinking_tokens=16,
        )
        config.finalize_response_format(
            reasoning_format=ReasoningFormat(tag_begin="", tag_end="</think>")
        )
        config_before_rpc = config.model_dump()

        input_pb = trans_input(self._make_generate_input(config))

        self.assertEqual(config.model_dump(), config_before_rpc)
        structural_tag = json.loads(input_pb.generate_config.structural_tag.value)
        elements = structural_tag["format"]["elements"]
        self.assertEqual(len(elements), 2)
        self.assertEqual(elements[0]["type"], "tag")
        self.assertEqual(elements[1]["type"], "json_schema")

    def test_trans_output_preserves_speculative_acceptance_counters(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2], dtype=torch.int32),
            generate_config=GenerateConfig(aux_info=True),
            request_id=1,
            mm_inputs=[],
        )
        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.append(False)
        aux_info = output_pb.aux_info.add()
        aux_info.speculative_draft_rounds = 7
        aux_info.speculative_accepted_tokens_per_pos.extend([6, 4, 2])

        outputs = trans_output(input_py, outputs_pb, StreamState())

        self.assertEqual(len(outputs.generate_outputs), 1)
        actual = outputs.generate_outputs[0].aux_info
        self.assertEqual(actual.speculative_draft_rounds, 7)
        self.assertEqual(actual.speculative_accepted_tokens_per_pos, [6, 4, 2])

    @unittest.skip("need fix")
    def test_generate_stream(self):
        client = FakeModelRpcClient()
        generate_config: GenerateConfig = GenerateConfig(using_hf_sampling=False)
        input = GenerateInput(
            token_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            generate_config=generate_config,
        )
        res = asyncio.run(self._run(client, input))
        self.assertEqual(len(res), 3)
        self.assertEqual(list(res[0].output_ids.shape), [1, 1])
        self.assertEqual(res[0].output_ids.tolist(), [[0]])
        self.assertEqual(res[0].finished, False)
        self.assertEqual(res[0].aux_info.iter_count, 2)
        self.assertEqual(res[0].aux_info.output_len, 1)

        self.assertEqual(list(res[1].output_ids.shape), [1, 2])
        self.assertEqual(res[1].output_ids.tolist(), [[0, 1]])
        self.assertEqual(res[1].finished, False)
        self.assertEqual(res[1].aux_info.iter_count, 3)
        self.assertEqual(res[1].aux_info.output_len, 2)

        self.assertEqual(res[2].finished, True)

    def test_generate_stream_with_logits_index(self):
        client = FakeModelRpcClient()
        generate_config: GenerateConfig = GenerateConfig(
            return_logits=True,
            logits_index=1,
            return_incremental=True,
            is_streaming=True,
        )
        input = GenerateInput(
            token_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            generate_config=generate_config,
            request_id=123,
            mm_inputs=[],
        )
        res = asyncio.run(self._run(client, input))

        self.assertEqual(len(res), 3)

        # res[0] 是第一个token
        self.assertTrue(hasattr(res[0], "logits"))
        self.assertIsNotNone(res[0].logits)
        logits_0 = res[0].logits.tolist()
        self.assertAlmostEqual(logits_0[0][0], 0.0, places=6)
        self.assertAlmostEqual(logits_0[0][1], 0.0, places=6)

        # res[1] 是第二个token
        self.assertTrue(hasattr(res[1], "logits"))
        self.assertIsNotNone(res[1].logits)
        logits_1 = res[1].logits.tolist()
        self.assertAlmostEqual(logits_1[0][0], 0.1, places=6)
        self.assertAlmostEqual(logits_1[0][1], 0.2, places=6)
        self.assertEqual(res[1].aux_info.speculative_draft_rounds, 4)
        self.assertEqual(res[1].aux_info.speculative_accepted_tokens_per_pos, [3, 2, 1])

        # res[2] 是完成标记，包含指定位置token的logits
        self.assertTrue(res[2].finished)
        self.assertTrue(hasattr(res[2], "logits"))
        self.assertIsNotNone(res[2].logits)
        logits_2 = res[2].logits.tolist()
        self.assertAlmostEqual(logits_2[0][0], 0.0, places=6)
        self.assertAlmostEqual(logits_2[0][1], 0.0, places=6)

    def test_trans_input_request_info(self):
        input_pb = trans_input(
            GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(trace_id="trace-from-config"),
                request_id=123,
                mm_inputs=[],
                headers={"x-request-id": "header-request-id"},
                request_info=RequestInfo(
                    frontend_ip="10.0.0.1",
                    dash_ip="10.0.0.2",
                    trace_id="trace-from-info",
                    request_id="source-request-id",
                    source_role="frontend",
                ),
            )
        )

        self.assertEqual(input_pb.request_info.frontend_ip, "10.0.0.1")
        self.assertEqual(input_pb.request_info.dash_ip, "10.0.0.2")
        self.assertEqual(input_pb.request_info.trace_id, "trace-from-info")
        self.assertEqual(input_pb.request_info.request_id, "source-request-id")
        self.assertEqual(input_pb.request_info.source_role, "frontend")

    def test_trans_input_dual_writes_role_addrs(self):
        input_pb = trans_input(
            GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(
                    role_addrs=[
                        _prefill_role_addr(),
                        _decode_role_addr(),
                    ]
                ),
                request_id=123,
                mm_inputs=[],
            )
        )

        self.assertEqual(len(input_pb.generate_config.role_addrs), 2)
        self.assertEqual(
            [role_addr.role for role_addr in input_pb.generate_config.role_addrs],
            [RoleAddrPB.PREFILL, RoleAddrPB.DECODE],
        )
        self.assertEqual(
            [role_addr.role_str for role_addr in input_pb.generate_config.role_addrs],
            ["PREFILL", "DECODE"],
        )

    def test_trans_input_request_info_fallback(self):
        input_pb = trans_input(
            GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(trace_id="trace-from-config"),
                request_id=123,
                mm_inputs=[],
                headers={"x-request-id": "header-request-id"},
            )
        )

        self.assertEqual(input_pb.request_info.trace_id, "trace-from-config")
        self.assertEqual(input_pb.request_info.request_id, "header-request-id")

    def test_trans_input_request_info_trace_header_fallback(self):
        traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
        input_pb = trans_input(
            GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(),
                request_id=123,
                mm_inputs=[],
                headers={"traceparent": traceparent},
            )
        )

        self.assertEqual(
            input_pb.request_info.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736"
        )
        self.assertEqual(
            input_pb.request_info.request_id, "4bf92f3577b34da6a3ce929d0e0e4736"
        )

    def test_enqueue_fetches_response_when_master_already_enqueued(self):
        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=False,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(fetch_responses=[_make_response(finished=True)])
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(
                timeout_ms=1000,
                role_addrs=[_prefill_role_addr("prefill-worker", 9000)],
            ),
            request_id=321,
            mm_inputs=[],
            enqueued_by_master=True,
        )

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            responses = asyncio.run(self._run(client, input_py))

        self.assertEqual(len(responses), 1)
        self.assertEqual(client._channel_pool.targets, ["prefill-worker:9000"])
        self.assertEqual(len(stub.fetch_calls), 1)
        self.assertEqual(stub.fetch_calls[0][0].request_id, 321)
        self.assertEqual(stub.fetch_calls[0][1]["timeout"], 1.0)
        self.assertEqual(stub.generate_calls, [])

    def test_enqueue_uses_generate_stream_without_master_enqueue(self):
        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=False,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(generate_responses=[_make_response(finished=True)])
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(timeout_ms=1000),
            request_id=322,
            mm_inputs=[],
        )

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            responses = asyncio.run(self._run(client, input_py))

        self.assertEqual(len(responses), 1)
        self.assertEqual(len(stub.generate_calls), 1)
        self.assertEqual(stub.generate_calls[0][0].request_id, 322)
        self.assertEqual(stub.fetch_calls, [])

    def test_prefill_only_fails_fast_when_backend_reports_generated_tokens(self):
        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=False,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(generate_responses=[_make_response(output_len=1)])
        input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))

        with (
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                return_value=stub,
            ),
            self.assertLogs(
                "rtp_llm.cpp.model_rpc.model_rpc_client", level="ERROR"
            ) as logs,
            self.assertRaises(FtRuntimeException) as context,
        ):
            asyncio.run(self._run(client, input_py))

        self.assertEqual(
            context.exception.exception_type, ExceptionType.EXECUTION_EXCEPTION
        )
        self.assertIn("output_len=1", context.exception.message)
        self.assertIn("may not support", context.exception.message)
        self.assertIn(
            "prefill-only response contract violation",
            "\n".join(logs.output),
        )
        log_text = "\n".join(logs.output)
        self.assertIn("output_len=1", log_text)
        self.assertIn("request_id=1", log_text)
        self.assertIn("aux_info[0]", log_text)
        self.assertTrue(stub.generate_iterator.cancelled)

    def test_prefill_only_does_not_reject_missing_first_packet_aux_info(self):
        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=False,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(
            generate_responses=[
                _make_response(finished=False),
                _make_response(finished=True, output_len=0),
            ]
        )
        input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            responses = asyncio.run(self._run(client, input_py))

        self.assertEqual(len(responses), 2)
        self.assertFalse(stub.generate_iterator.cancelled)

    def test_prefill_only_fails_when_stream_never_exposes_aux_info(self):
        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=False,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(generate_responses=[_make_response(finished=True)])
        input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))

        with (
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                return_value=stub,
            ),
            self.assertLogs(
                "rtp_llm.cpp.model_rpc.model_rpc_client", level="ERROR"
            ) as logs,
            self.assertRaises(FtRuntimeException) as context,
        ):
            asyncio.run(self._run(client, input_py))

        self.assertEqual(
            context.exception.exception_type, ExceptionType.EXECUTION_EXCEPTION
        )
        self.assertIn("completed without aux_info", context.exception.message)
        self.assertIn("capability cannot be verified", context.exception.message)
        self.assertIn(
            "prefill-only capability verification failed",
            "\n".join(logs.output),
        )
        log_text = "\n".join(logs.output)
        self.assertIn("completed without aux_info", log_text)
        self.assertIn("request_id=1", log_text)
        self.assertFalse(stub.generate_iterator.cancelled)

    def test_enqueue_cancels_fetch_stream_on_early_close(self):
        async def run_and_close():
            gen = client.enqueue(input_py)
            first = await gen.__anext__()
            await gen.aclose()
            return first

        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=False,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(
            fetch_responses=[
                _make_response(finished=False),
                _make_response(finished=True),
            ]
        )
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(
                timeout_ms=1000,
                role_addrs=[_prefill_role_addr("prefill-worker", 9000)],
            ),
            request_id=323,
            mm_inputs=[],
            enqueued_by_master=True,
        )

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            asyncio.run(run_and_close())

        self.assertTrue(stub.fetch_iterator.cancelled)

    def test_enqueue_fetch_uses_prefill_when_decode_entrance(self):
        async def run_and_close():
            gen = client.enqueue(input_py)
            await gen.__anext__()
            await gen.aclose()

        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=True,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(fetch_responses=[_make_response(finished=False)])
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(
                timeout_ms=1000,
                role_addrs=[
                    _prefill_role_addr("prefill-worker", 9000),
                    _decode_role_addr("decode-worker", 9001),
                ],
            ),
            request_id=325,
            mm_inputs=[],
            enqueued_by_master=True,
        )

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            asyncio.run(run_and_close())

        self.assertEqual(client._channel_pool.targets, ["prefill-worker:9000"])
        self.assertEqual(len(stub.fetch_calls), 1)

    def test_enqueue_does_not_cancel_after_finished_response_is_seen(self):
        async def run_and_close_after_finished():
            gen = client.enqueue(input_py)
            first = await gen.__anext__()
            self.assertTrue(first.generate_outputs[0].finished)
            await gen.aclose()

        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=False,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(fetch_responses=[_make_response(finished=True)])
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(
                timeout_ms=1000,
                role_addrs=[_prefill_role_addr("prefill-worker", 9000)],
            ),
            request_id=324,
            mm_inputs=[],
            enqueued_by_master=True,
        )

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            asyncio.run(run_and_close_after_finished())

        self.assertFalse(stub.fetch_iterator.cancelled)


if __name__ == "__main__":
    setup_logging()
    main()
