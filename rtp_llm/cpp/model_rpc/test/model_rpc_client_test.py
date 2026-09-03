import asyncio
import json
import struct
import sys
from enum import Enum
from unittest.mock import MagicMock, patch

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

import logging
import os
import unittest
from typing import AsyncGenerator
from unittest import TestCase, main

import grpc
import torch
from grpc import StatusCode

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
    _engine_reported_finished,
    _record_client_span_latency,
    _record_client_span_usage,
    _request_completed_normally,
    _settle_client_span_after_rpc,
    trans_input,
    trans_output,
)
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    ErrorDetailsPB,
    GenerateConfigPB,
    GenerateInputPB,
    GenerateOutputsPB,
    RoleAddrPB,
    TensorPB,
)
from rtp_llm.telemetry import CURRENT_TRACE_STATE, tracing
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutputs,
    RequestInfo,
)


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


def _make_response(finished=True):
    outputs_pb = GenerateOutputsPB()
    outputs_pb.flatten_output.finished.extend([finished])
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

    def test_trans_output_prefill_cuda_graph_status_compatibility(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2], dtype=torch.int32),
            generate_config=GenerateConfig(aux_info=True),
            request_id=1,
            mm_inputs=[],
        )
        for wire_status, expected_status in (
            ("replayed", "replayed"),
            ("", "not_requested"),
        ):
            with self.subTest(wire_status=wire_status):
                outputs_pb = GenerateOutputsPB()
                output_pb = outputs_pb.flatten_output
                output_pb.finished.append(False)
                output_pb.aux_info.add().prefill_cuda_graph_status = wire_status

                outputs = trans_output(input_py, outputs_pb, StreamState())

                self.assertEqual(
                    outputs.generate_outputs[0].aux_info.prefill_cuda_graph_status,
                    expected_status,
                )

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


class _MetadataCaptureServicer(model_rpc_service_pb2_grpc.RpcServiceServicer):
    def __init__(self):
        self.metadata = None
        self.metadata_ready = asyncio.Event()

    async def GenerateStreamCall(self, request, context):
        self.metadata = {item.key: item.value for item in context.invocation_metadata()}
        self.metadata_ready.set()
        outputs = GenerateOutputsPB()
        output = outputs.flatten_output
        output.output_ids.data_type = TensorPB.DataType.INT32
        output.output_ids.shape.extend([1, 1])
        output.output_ids.int32_data = struct.pack("<i", 7)
        output.finished.extend([True])
        aux_info = output.aux_info.add()
        aux_info.input_len = 3
        aux_info.output_len = 1
        yield outputs


class _DelayedTerminalServicer(model_rpc_service_pb2_grpc.RpcServiceServicer):
    def __init__(self, terminal_delay):
        self.terminal_delay = terminal_delay
        self.terminal_released = asyncio.Event()

    async def GenerateStreamCall(self, request, context):
        outputs = GenerateOutputsPB()
        output = outputs.flatten_output
        output.output_ids.data_type = TensorPB.DataType.INT32
        output.output_ids.shape.extend([1, 1])
        output.output_ids.int32_data = struct.pack("<i", 7)
        output.finished.extend([True])
        aux_info = output.aux_info.add()
        aux_info.input_len = 3
        aux_info.output_len = 1
        yield outputs
        await asyncio.sleep(self.terminal_delay)
        self.terminal_released.set()


class _RealChannelPool:
    def __init__(self, channel):
        self.channel = channel

    async def get(self, _target_address):
        return self.channel


class ModelRpcClientGrpcMetadataTest(TestCase):
    def test_trace_disabled_full_consumer_waits_for_real_grpc_terminal(self):
        self.addCleanup(tracing.reset_telemetry_for_test)

        async def run():
            self.assertTrue(tracing.reset_telemetry_for_test())
            server = grpc.aio.server()
            servicer = _DelayedTerminalServicer(terminal_delay=0.05)
            model_rpc_service_pb2_grpc.add_RpcServiceServicer_to_server(
                servicer, server
            )
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()
            channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
            client = ModelRpcClient([f"127.0.0.1:{port}"], {}, max_rpc_timeout_ms=1000)
            client._channel_pool = _RealChannelPool(channel)
            input_py = GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(timeout_ms=1000),
                request_id=902,
                mm_inputs=[],
            )
            try:
                responses = []
                with patch(
                    "rtp_llm.cpp.model_rpc.model_rpc_client.RPC_SETTLE_TIMEOUT_SECONDS",
                    0.01,
                ):
                    async for response in client.enqueue(input_py):
                        responses.extend(response.generate_outputs)
                self.assertEqual(len(responses), 1)
                self.assertTrue(responses[0].finished)
                self.assertTrue(servicer.terminal_released.is_set())
            finally:
                await channel.close()
                await server.stop(None)

        asyncio.run(run())

    @unittest.skipUnless(tracing.OTEL_AVAILABLE, "opentelemetry SDK not available")
    def test_traceparent_crosses_real_grpc_boundary(self):
        self.addCleanup(tracing.reset_telemetry_for_test)

        async def run():
            server = grpc.aio.server()
            servicer = _MetadataCaptureServicer()
            model_rpc_service_pb2_grpc.add_RpcServiceServicer_to_server(
                servicer, server
            )
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()
            channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")

            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )

            exporter = InMemorySpanExporter()
            self.assertTrue(tracing.reset_telemetry_for_test())
            self.assertTrue(
                tracing.init_telemetry_for_test(exporter, role="frontend", tp_rank=0)
            )
            root = tracing.start_server_span("root", {})
            client = ModelRpcClient([f"127.0.0.1:{port}"], {}, max_rpc_timeout_ms=1000)
            client._channel_pool = _RealChannelPool(channel)
            input_py = GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(timeout_ms=1000),
                request_id=901,
                mm_inputs=[],
            )
            try:
                responses = []
                async for response in client.enqueue(input_py):
                    responses.extend(response.generate_outputs)
                await asyncio.wait_for(servicer.metadata_ready.wait(), timeout=5)
                self.assertEqual(len(responses), 1)
                self.assertIsNotNone(servicer.metadata)
                self.assertIn("traceparent", servicer.metadata)
                self.assertTrue(servicer.metadata["traceparent"].startswith("00-"))
                root.finish()
                self.assertTrue(tracing.shutdown_telemetry())
                spans = {span.name: span for span in exporter.get_finished_spans()}
                self.assertIn("rtp_llm.generate_stream_call", spans)
                self.assertEqual(
                    spans["rtp_llm.generate_stream_call"].parent.span_id,
                    spans["root"].context.span_id,
                )
            finally:
                await channel.close()
                await server.stop(None)
                self.assertTrue(tracing.reset_telemetry_for_test())

        asyncio.run(run())

    def test_trace_enabled_full_consumer_does_not_cancel_slow_terminal(self):
        async def run():
            server = grpc.aio.server()
            servicer = _DelayedTerminalServicer(terminal_delay=0.05)
            model_rpc_service_pb2_grpc.add_RpcServiceServicer_to_server(
                servicer, server
            )
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()
            channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
            span = _FakeClientSpan()
            client = ModelRpcClient([f"127.0.0.1:{port}"], {}, max_rpc_timeout_ms=1000)
            client._channel_pool = _RealChannelPool(channel)
            input_py = GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(timeout_ms=1000),
                request_id=903,
                mm_inputs=[],
            )
            try:
                with patch(
                    "rtp_llm.cpp.model_rpc.model_rpc_client.start_client_span",
                    return_value=(span, []),
                ), patch(
                    "rtp_llm.cpp.model_rpc.model_rpc_client.RPC_SETTLE_TIMEOUT_SECONDS",
                    0.01,
                ):
                    responses = [
                        response async for response in client.enqueue(input_py)
                    ]
                self.assertEqual(len(responses), 1)
                self.assertTrue(servicer.terminal_released.is_set())
                await asyncio.wait_for(span.finished_event.wait(), timeout=5)
                self.assertEqual(span.status, "OK")
                self.assertEqual(span.attributes["rpc.response.status_code"], "OK")
            finally:
                await channel.close()
                await server.stop(None)

        asyncio.run(run())

    def test_trace_enabled_aclose_preserves_real_late_terminal(self):
        async def run():
            server = grpc.aio.server()
            servicer = _DelayedTerminalServicer(terminal_delay=0.05)
            model_rpc_service_pb2_grpc.add_RpcServiceServicer_to_server(
                servicer, server
            )
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()
            channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
            span = _FakeClientSpan()
            client = ModelRpcClient([f"127.0.0.1:{port}"], {}, max_rpc_timeout_ms=1000)
            client._channel_pool = _RealChannelPool(channel)
            input_py = GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(timeout_ms=1000),
                request_id=904,
                mm_inputs=[],
            )
            try:
                with patch(
                    "rtp_llm.cpp.model_rpc.model_rpc_client.start_client_span",
                    return_value=(span, []),
                ), patch(
                    "rtp_llm.cpp.model_rpc.model_rpc_client.RPC_SETTLE_TIMEOUT_SECONDS",
                    0.2,
                ):
                    gen = client.enqueue(input_py)
                    response = await gen.__anext__()
                    self.assertTrue(response.generate_outputs[0].finished)
                    await gen.aclose()
                    self.assertFalse(servicer.terminal_released.is_set())
                    await asyncio.wait_for(span.finished_event.wait(), timeout=5)
                self.assertTrue(servicer.terminal_released.is_set())
                self.assertEqual(span.status, "OK")
                self.assertEqual(span.attributes["rpc.response.status_code"], "OK")
            finally:
                await channel.close()
                await server.stop(None)

        asyncio.run(run())


class _FakeAux:
    def __init__(
        self, input_len, output_len, first_token_cost_time=8.5, cost_time=20.0
    ):
        self.input_len = input_len
        self.output_len = output_len
        self.first_token_cost_time = first_token_cost_time
        self.cost_time = cost_time


class _FakeOut:
    def __init__(
        self,
        finished,
        input_len=8,
        output_len=3,
        first_token_cost_time=8.5,
        cost_time=20.0,
    ):
        self.finished = finished
        self.aux_info = _FakeAux(
            input_len, output_len, first_token_cost_time, cost_time
        )


class _AsyncReturn:
    def __init__(self, value):
        self._value = value

    async def __call__(self, *args, **kwargs):
        return self._value


class _FakeClientSpan:
    """Mirrors tracing.ClientSpanHandle: idempotent finish, writes dropped after."""

    def __init__(self):
        self.attributes = {}
        self.status = None
        self.error_type = None
        self.finished = False
        self.finish_calls = 0
        self.finished_event = asyncio.Event()

    def set_attribute(self, key, value):
        if not self.finished:
            self.attributes[key] = value

    def finish(self, error=None, error_type=""):
        self.finish_calls += 1
        if self.finished:
            return
        self.finished = True
        if error is not None or error_type:
            self.status = "ERROR"
            self.error_type = error_type or type(error).__name__
        else:
            self.status = "OK"
        self.finished_event.set()


class _FakeTraceState:
    """Mirrors the request completion contract used during stream teardown."""

    def __init__(self, settled_ok=None, renderer_completed=False):
        self.settled_ok = settled_ok
        self.renderer_completed = renderer_completed

    def set_attribute(self, key, value):
        pass


class _FakeRpcError(grpc.RpcError):
    def __init__(self, status, trailing_metadata=None):
        self._status = status
        self._trailing_metadata = trailing_metadata or {}

    def code(self):
        return self._status

    def details(self):
        return "injected terminal RPC error"

    def trailing_metadata(self):
        return self._trailing_metadata


class _SpanAwareStub:
    """Yields `total` responses; the last one carries the engine finished flag."""

    def __init__(
        self,
        total,
        finish_last=True,
        terminal_error=None,
        terminal_delay=0.0,
        terminal_never=False,
    ):
        self._total = total
        self._finish_last = finish_last
        self._terminal_error = terminal_error
        self._terminal_delay = terminal_delay
        self._terminal_never = terminal_never
        self.iterator = None

    def GenerateStreamCall(self, input_pb, timeout=None, metadata=None):
        total, finish_last, terminal_error, terminal_delay, terminal_never = (
            self._total,
            self._finish_last,
            self._terminal_error,
            self._terminal_delay,
            self._terminal_never,
        )

        class _Iterator:
            def __init__(self):
                self.cancelled = False
                self.code_waited = False
                self.code_resolved = False
                self.events = []
                self.cancelled_event = asyncio.Event()
                self.code_started = asyncio.Event()
                self._terminal_status = None
                self._terminal_ready = asyncio.Event()

            def __aiter__(self):
                return self._gen()

            def cancel(self):
                self.events.append("cancel")
                if self._terminal_ready.is_set():
                    return False
                self.cancelled = True
                self.cancelled_event.set()
                self._terminal_status = StatusCode.CANCELLED
                self._terminal_ready.set()
                return True

            async def code(self):
                self.code_waited = True
                self.events.append("code")
                self.code_started.set()
                await self._terminal_ready.wait()
                self.code_resolved = True
                return self._terminal_status

            async def _gen(self):
                for i in range(total):
                    outputs_pb = GenerateOutputsPB()
                    output_pb = outputs_pb.flatten_output
                    output_pb.output_ids.data_type = TensorPB.DataType.INT32
                    output_pb.output_ids.shape.extend([1, i + 1])
                    output_pb.output_ids.int32_data = struct.pack(
                        "<" + "i" * (i + 1), *range(i + 1)
                    )
                    aux_info = output_pb.aux_info.add()
                    aux_info.iter_count = i + 1
                    aux_info.input_len = 8
                    aux_info.output_len = i + 1
                    aux_info.first_token_cost_time_us = 8500
                    aux_info.cost_time_us = 20000
                    output_pb.finished.extend([finish_last and i == total - 1])
                    if finish_last and i == total - 1:
                        # The real server can settle independently while the
                        # Python message iterator remains suspended at yield.
                        if not terminal_never:
                            self._terminal_status = (
                                terminal_error.code()
                                if terminal_error is not None
                                else StatusCode.OK
                            )
                            if terminal_delay:
                                asyncio.get_running_loop().call_later(
                                    terminal_delay, self._terminal_ready.set
                                )
                            else:
                                asyncio.get_running_loop().call_soon(
                                    self._terminal_ready.set
                                )
                    yield outputs_pb
                if terminal_error is not None:
                    self._terminal_status = terminal_error.code()
                    self._terminal_ready.set()
                    raise terminal_error
                if not terminal_never:
                    self._terminal_status = StatusCode.OK
                    self._terminal_ready.set()

        self.iterator = _Iterator()
        return self.iterator


class ClientSpanSettlementTest(TestCase):
    """Regression guard for the CLIENT span settlement timing and status.

    render_response_stream breaks out of its `async for` as soon as every
    sequence has a finish_reason, so enqueue() stays suspended on its last
    yield until aclose()/GC injects GeneratorExit. Settling the span there
    marked successful requests Cancelled, dropped the usage attributes and
    pushed the span end past the root span.

    The application frame must be published before waiting for grpc.aio's
    terminal status; settlement runs in a bounded independent task so a slow or
    absent RPC deadline cannot delay the data plane.
    """

    USAGE_KEYS = (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.prompt_tokens",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.total_tokens",
    )

    def _build_client(
        self,
        span,
        total,
        finish_last=True,
        trace_state=None,
        terminal_error=None,
        terminal_delay=0.0,
        terminal_never=False,
    ):
        client = ModelRpcClient(["127.0.0.1:1234"], {}, 0, False)
        stub = _SpanAwareStub(
            total,
            finish_last,
            terminal_error,
            terminal_delay,
            terminal_never,
        )
        client._channel_pool = MagicMock()
        client._channel_pool.get = _AsyncReturn(MagicMock())
        token = CURRENT_TRACE_STATE.set(trace_state)
        self.addCleanup(CURRENT_TRACE_STATE.reset, token)
        patcher_stub = patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        )
        patcher_span = patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.start_client_span",
            return_value=(span, []),
        )
        self.addCleanup(patcher_stub.stop)
        self.addCleanup(patcher_span.stop)
        patcher_stub.start()
        patcher_span.start()
        client._test_stub = stub
        return client

    @staticmethod
    def _make_input():
        return GenerateInput(
            token_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            generate_config=GenerateConfig(is_streaming=True),
            request_id=7,
            mm_inputs=[],
        )

    def test_engine_reported_finished_predicate(self):
        self.assertFalse(_engine_reported_finished(None))
        self.assertFalse(_engine_reported_finished(GenerateOutputs()))
        outputs = GenerateOutputs()
        outputs.generate_outputs = [_FakeOut(True), _FakeOut(False)]
        self.assertFalse(_engine_reported_finished(outputs))
        outputs.generate_outputs = [_FakeOut(True), _FakeOut(True)]
        self.assertTrue(_engine_reported_finished(outputs))

    def test_usage_attributes_skip_non_positive(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [_FakeOut(True, input_len=0, output_len=5)]
        _record_client_span_usage(span, outputs)
        self.assertEqual(span.attributes, {})
        outputs.generate_outputs = [_FakeOut(True, input_len=8, output_len=3)]
        _record_client_span_usage(span, outputs)
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)
        self.assertEqual(span.attributes["gen_ai.usage.total_tokens"], 11)

    def test_usage_attributes_sum_all_choices(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(True, input_len=8, output_len=3),
            _FakeOut(True, input_len=8, output_len=5),
        ]

        _record_client_span_usage(span, outputs)

        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 8)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 8)
        self.assertEqual(span.attributes["gen_ai.usage.prompt_tokens"], 8)
        self.assertEqual(span.attributes["gen_ai.usage.completion_tokens"], 8)
        self.assertEqual(span.attributes["gen_ai.usage.total_tokens"], 16)

    def test_beam_usage_attributes_count_primary_sequence_only(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(True, input_len=8, output_len=3),
            _FakeOut(True, input_len=8, output_len=5),
        ]

        _record_client_span_usage(span, outputs, include_all_sequences=False)

        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 8)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 3)
        self.assertEqual(span.attributes["gen_ai.usage.total_tokens"], 11)

    def test_usage_attributes_skip_inconsistent_choices(self):
        for choices in (
            [
                _FakeOut(True, input_len=8, output_len=3),
                _FakeOut(True, input_len=9, output_len=5),
            ],
            [
                _FakeOut(True, input_len=8, output_len=3),
                _FakeOut(True, input_len=8, output_len=0),
            ],
        ):
            with self.subTest(choices=choices):
                span = _FakeClientSpan()
                outputs = GenerateOutputs()
                outputs.generate_outputs = choices
                _record_client_span_usage(span, outputs)
                self.assertEqual(span.attributes, {})

    def test_engine_latency_attributes_from_single_sequence_stream(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(
                True,
                output_len=5,
                first_token_cost_time=8.5,
                cost_time=20.0,
            )
        ]

        _record_client_span_latency(span, outputs)

        self.assertEqual(span.attributes["rtp_llm.engine.time_to_first_token_ms"], 8.5)
        self.assertEqual(
            span.attributes["rtp_llm.engine.time_per_output_token_ms"], 2.875
        )

    def test_engine_latency_attributes_require_coherent_aux_info(self):
        cases = (
            _FakeOut(True, output_len=0),
            _FakeOut(True, output_len=1),
            _FakeOut(True, output_len=5, first_token_cost_time=0),
            _FakeOut(True, output_len=5, first_token_cost_time=8.5, cost_time=8.0),
        )
        for output in cases:
            with self.subTest(output=output):
                span = _FakeClientSpan()
                outputs = GenerateOutputs(generate_outputs=[output])
                _record_client_span_latency(span, outputs)
                if output.aux_info.output_len == 1:
                    self.assertEqual(
                        span.attributes["rtp_llm.engine.time_to_first_token_ms"],
                        8.5,
                    )
                elif output.aux_info.cost_time < output.aux_info.first_token_cost_time:
                    self.assertEqual(
                        span.attributes["rtp_llm.engine.time_to_first_token_ms"],
                        8.5,
                    )
                else:
                    self.assertEqual(span.attributes, {})
                self.assertNotIn(
                    "rtp_llm.engine.time_per_output_token_ms", span.attributes
                )

    def test_multi_return_shares_ttft_but_omits_ambiguous_tpot(self):
        """n>1 rides one physical stream: TPOT per sequence is not a span value."""
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(True, output_len=5, first_token_cost_time=8.5, cost_time=20.0),
            _FakeOut(True, output_len=7, first_token_cost_time=8.5, cost_time=26.0),
        ]

        _record_client_span_latency(span, outputs)

        self.assertEqual(span.attributes["rtp_llm.engine.time_to_first_token_ms"], 8.5)
        self.assertNotIn("rtp_llm.engine.time_per_output_token_ms", span.attributes)

    def test_multi_return_disagreeing_on_first_token_omits_latency(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(True, output_len=5, first_token_cost_time=8.5, cost_time=20.0),
            _FakeOut(True, output_len=5, first_token_cost_time=9.5, cost_time=20.0),
        ]

        _record_client_span_latency(span, outputs)

        self.assertEqual(span.attributes, {})

    def test_multi_return_with_one_empty_sequence_omits_latency(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(True, output_len=5, first_token_cost_time=8.5, cost_time=20.0),
            _FakeOut(True, output_len=0, first_token_cost_time=8.5, cost_time=20.0),
        ]

        _record_client_span_latency(span, outputs)

        self.assertEqual(span.attributes, {})

    def test_finished_frame_is_not_blocked_by_rpc_termination(self):
        span = _FakeClientSpan()
        client = self._build_client(span, total=3, terminal_delay=0.05)

        async def run():
            gen = client.enqueue(self._make_input())
            while True:
                outputs = await gen.__anext__()
                if outputs.generate_outputs and outputs.generate_outputs[0].finished:
                    self.assertFalse(client._test_stub.iterator.code_waited)
                    break
            await gen.aclose()
            await asyncio.wait_for(span.finished_event.wait(), timeout=5)
            self.assertTrue(span.finished, "settlement runs after the finished frame")
            self.assertEqual(client._test_stub.iterator.events, ["code"])

        asyncio.run(run())
        self.assertEqual(span.status, "OK")
        self.assertIsNone(span.error_type)
        self.assertEqual(span.attributes["rpc.response.status_code"], "OK")
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_finished_frame_with_late_rpc_error_marks_span_error(self):
        span = _FakeClientSpan()
        client = self._build_client(
            span,
            total=1,
            terminal_error=_FakeRpcError(StatusCode.UNAVAILABLE),
        )

        async def run():
            gen = client.enqueue(self._make_input())
            outputs = await gen.__anext__()
            self.assertTrue(outputs.generate_outputs[0].finished)
            await asyncio.wait_for(span.finished_event.wait(), timeout=5)
            self.assertEqual(span.status, "ERROR")
            with self.assertRaises(Exception):
                await gen.__anext__()

        asyncio.run(run())
        self.assertEqual(span.attributes["rpc.response.status_code"], "UNAVAILABLE")
        self.assertEqual(span.error_type, "RpcError")

    def test_finished_frame_eventually_cancels_unsettled_rpc_with_or_without_trace(
        self,
    ):
        async def run(client, span):
            gen = client.enqueue(self._make_input())
            outputs = await gen.__anext__()
            self.assertTrue(outputs.generate_outputs[0].finished)
            await gen.aclose()
            iterator = client._test_stub.iterator
            await asyncio.wait_for(iterator.cancelled_event.wait(), timeout=5)
            self.assertTrue(iterator.cancelled)
            if span is not None:
                await asyncio.wait_for(span.finished_event.wait(), timeout=5)
                self.assertEqual(span.status, "ERROR")
                self.assertEqual(
                    span.attributes["rpc.response.status_code"], "CANCELLED"
                )
            else:
                self.assertFalse(iterator.code_waited)
                self.assertEqual(iterator.events, ["cancel"])

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RPC_SETTLE_TIMEOUT_SECONDS",
            0.01,
        ):
            span = _FakeClientSpan()
            client = self._build_client(span, total=1, terminal_never=True)
            asyncio.run(run(client, span))
            client = self._build_client(None, total=1, terminal_never=True)
            asyncio.run(run(client, None))

    def test_settlement_task_cancellation_respects_transport_ownership(self):
        async def run(abandoned):
            stub = _SpanAwareStub(total=0, terminal_never=True)
            iterator = stub.GenerateStreamCall(None)
            span = _FakeClientSpan()
            abandoned_event = asyncio.Event()
            if abandoned:
                abandoned_event.set()
            task = asyncio.create_task(
                _settle_client_span_after_rpc(iterator, span, None, abandoned_event)
            )
            await asyncio.wait_for(iterator.code_started.wait(), timeout=5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(span.status, "ERROR")
            self.assertEqual(span.error_type, "RpcSettlementCancelled")
            self.assertEqual(span.finish_calls, 1)
            self.assertEqual(iterator.cancelled, abandoned)

        asyncio.run(run(False))
        asyncio.run(run(True))

    def test_active_observer_waits_for_grpc_terminal_without_local_cancel(self):
        async def run(has_deadline):
            stub = _SpanAwareStub(total=1, terminal_delay=0.05)
            iterator = stub.GenerateStreamCall(None)
            stream = iterator.__aiter__()
            await stream.__anext__()
            span = _FakeClientSpan()
            abandoned_event = asyncio.Event()
            active_deadline = (
                asyncio.get_running_loop().time() + 0.01 if has_deadline else None
            )
            with patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RPC_SETTLE_TIMEOUT_SECONDS",
                0.01,
            ):
                status = await _settle_client_span_after_rpc(
                    iterator,
                    span,
                    None,
                    abandoned_event,
                    active_deadline=active_deadline,
                )
            self.assertEqual(status, StatusCode.OK)
            self.assertFalse(iterator.cancelled)
            self.assertEqual(span.status, "OK")
            await stream.aclose()

        asyncio.run(run(True))
        asyncio.run(run(False))

    def test_abandoned_cleanup_timeout_starts_at_ownership_transfer(self):
        async def run():
            stub = _SpanAwareStub(total=0, terminal_never=True)
            iterator = stub.GenerateStreamCall(None)
            span = _FakeClientSpan()
            abandoned_event = asyncio.Event()
            active_deadline = asyncio.get_running_loop().time() + 0.001
            with patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RPC_SETTLE_TIMEOUT_SECONDS",
                0.05,
            ):
                task = asyncio.create_task(
                    _settle_client_span_after_rpc(
                        iterator,
                        span,
                        None,
                        abandoned_event,
                        active_deadline=active_deadline,
                    )
                )
                await asyncio.wait_for(iterator.code_started.wait(), timeout=5)
                await asyncio.sleep(0.01)
                abandoned_event.set()
                await asyncio.sleep(0.01)
                self.assertFalse(iterator.cancelled)
                await asyncio.wait_for(iterator.cancelled_event.wait(), timeout=1)
                await task

            self.assertEqual(span.error_type, "RpcError")
            self.assertEqual(span.attributes["rpc.response.status_code"], "CANCELLED")

        asyncio.run(run())

    def test_outer_cancellation_during_final_rpc_wait_propagates(self):
        span = _FakeClientSpan()
        client = self._build_client(
            span, total=1, finish_last=False, terminal_never=True
        )

        async def run():
            async def consume():
                async for _ in client.enqueue(self._make_input()):
                    pass

            task = asyncio.create_task(consume())
            while client._test_stub.iterator is None:
                await asyncio.sleep(0)
            iterator = client._test_stub.iterator
            await asyncio.wait_for(iterator.code_started.wait(), timeout=5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(iterator.cancelled)
            self.assertEqual(span.status, "ERROR")
            self.assertEqual(span.error_type, "Cancelled")

        asyncio.run(run())

    def test_trace_disabled_does_not_wait_for_rpc_termination(self):
        client = self._build_client(None, total=1, terminal_delay=0.05)

        async def run():
            async for outputs in client.enqueue(self._make_input()):
                self.assertTrue(outputs.generate_outputs[0].finished)
            self.assertFalse(client._test_stub.iterator.code_waited)

        asyncio.run(run())

    def test_stream_iterated_to_completion_keeps_span_ok(self):
        span = _FakeClientSpan()
        client = self._build_client(span, total=2, finish_last=False)

        async def run():
            async for _ in client.enqueue(self._make_input()):
                pass

        asyncio.run(run())
        self.assertEqual(span.status, "OK")
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_break_before_finished_still_reports_cancelled(self):
        """Genuine interruption (client disconnect) must stay Cancelled.

        The root span is unsettled here: the cancellation is still propagating
        outward, so the request has not succeeded.
        """
        span = _FakeClientSpan()
        client = self._build_client(span, total=3, finish_last=False)

        async def run():
            gen = client.enqueue(self._make_input())
            async for _ in gen:
                break
            self.assertFalse(span.finished, "no finished flag seen yet")
            await gen.aclose()
            self.assertTrue(client._test_stub.iterator.cancelled)
            self.assertTrue(client._test_stub.iterator.code_waited)
            self.assertEqual(client._test_stub.iterator.events[:2], ["cancel", "code"])

        asyncio.run(run())
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.error_type, "Cancelled")
        self.assertEqual(span.attributes["rpc.response.status_code"], "CANCELLED")
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_stop_word_break_with_renderer_milestone_keeps_span_ok(self):
        """Stop-word truncation is normal before the root span is settled.

        Context-dependent tokenization can make a string stop word miss the
        engine's token-level list, so the renderer can break while the engine is
        still generating. Its explicit milestone keeps that cleanup path OK.
        """
        span = _FakeClientSpan()
        client = self._build_client(
            span,
            total=3,
            finish_last=False,
            trace_state=_FakeTraceState(renderer_completed=True),
        )

        async def run():
            gen = client.enqueue(self._make_input())
            seen = 0
            async for _ in gen:
                seen += 1
                if seen == 3:
                    break  # renderer hit a stop word
            await gen.aclose()

        asyncio.run(run())
        self.assertEqual(span.status, "OK")
        self.assertIsNone(span.error_type)
        self.assertEqual(span.attributes["rtp_llm.engine.time_to_first_token_ms"], 8.5)
        self.assertEqual(
            span.attributes["rtp_llm.engine.time_per_output_token_ms"], 5.75
        )
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_break_with_failed_root_reports_cancelled(self):
        """Root span settled with an error keeps the child Cancelled."""
        span = _FakeClientSpan()
        client = self._build_client(
            span, total=3, finish_last=False, trace_state=_FakeTraceState(False)
        )

        async def run():
            gen = client.enqueue(self._make_input())
            async for _ in gen:
                break
            await gen.aclose()

        asyncio.run(run())
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.error_type, "Cancelled")

    def test_response_conversion_error_cancels_before_waiting_for_code(self):
        span = _FakeClientSpan()
        client = self._build_client(span, total=2, finish_last=False)
        call_count = 0

        def fail_second_response(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("conversion failed")
            return trans_output(*args)

        async def run():
            with patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.trans_output",
                side_effect=fail_second_response,
            ):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    async for _ in client.enqueue(self._make_input()):
                        pass

        asyncio.run(run())
        iterator = client._test_stub.iterator
        self.assertEqual(iterator.events[:2], ["cancel", "code"])
        self.assertEqual(span.attributes["rpc.response.status_code"], "CANCELLED")
        self.assertEqual(span.status, "ERROR")
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_rpc_error_after_output_keeps_last_confirmed_usage(self):
        span = _FakeClientSpan()
        client = self._build_client(
            span,
            total=1,
            finish_last=False,
            terminal_error=_FakeRpcError(StatusCode.UNAVAILABLE),
        )

        async def run():
            with self.assertRaisesRegex(Exception, "injected terminal RPC error"):
                async for _ in client.enqueue(self._make_input()):
                    pass

        asyncio.run(run())
        self.assertEqual(span.attributes["rpc.response.status_code"], "UNAVAILABLE")
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.error_type, "RpcError")
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_request_completed_normally_predicate(self):
        self.assertFalse(_request_completed_normally(None))
        self.assertFalse(_request_completed_normally(_FakeTraceState(None)))
        self.assertFalse(_request_completed_normally(_FakeTraceState(False)))
        self.assertTrue(_request_completed_normally(_FakeTraceState(True)))
        self.assertTrue(
            _request_completed_normally(_FakeTraceState(None, renderer_completed=True))
        )

    def test_unknown_detailed_error_code_falls_back_without_value_error(self):
        details = ErrorDetailsPB(error_code=999999, error_message="future error")
        error = _FakeRpcError(
            StatusCode.INTERNAL,
            {"grpc-status-details-bin": details.SerializeToString()},
        )
        client = ModelRpcClient.__new__(ModelRpcClient)

        with self.assertRaises(FtRuntimeException) as raised:
            client._handle_grpc_error(error, "request: [7]", "worker:9000")

        self.assertEqual(raised.exception.exception_type, ExceptionType.UNKNOWN_ERROR)
        self.assertEqual(raised.exception.message, "future error")


if __name__ == "__main__":
    setup_logging()
    main()
