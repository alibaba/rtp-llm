import asyncio
import json
import struct
import sys
from unittest.mock import MagicMock

# Mock the ops module to avoid CUDA dependency in this unit test
# This MUST be at the very top before any other imports, even before unittest
mock_ops = MagicMock()
mock_comm = MagicMock()
mock_nccl_op = MagicMock()
mock_compute_ops = MagicMock()
mock_comm.nccl_op = mock_nccl_op
mock_ops.comm = mock_comm
mock_ops.compute_ops = mock_compute_ops
sys.modules["rtp_llm.ops"] = mock_ops
sys.modules["rtp_llm.ops.comm"] = mock_comm
sys.modules["rtp_llm.ops.compute_ops"] = mock_compute_ops
sys.modules["rtp_llm.ops.comm.nccl_op"] = mock_nccl_op

import logging
import os
import unittest
from typing import AsyncGenerator
from unittest import TestCase, main

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.response_format_compiler import ReasoningFormat
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    CustomOutputSelector,
    ModelRpcClient,
    StreamState,
    trans_input,
    trans_output,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    GenerateInputPB,
    GenerateOutputsPB,
    TensorPB,
)
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

    def test_trans_input_resolves_relative_position_without_mutating_request(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([7, 42, 8, 42, 9]),
            mm_inputs=[],
            generate_config=GenerateConfig(),
        )

        with unittest.mock.patch.dict(
            os.environ,
            {"CUSTOM_OUTPUT_TOKEN_POSITION": "-2"},
            clear=True,
        ):
            input_pb = trans_input(input_py)

        self.assertTrue(input_pb.HasField("custom_output_token_position"))
        self.assertEqual(input_pb.custom_output_token_position.value, 3)
        self.assertEqual(input_py.custom_output_token_position, -1)

    def test_trans_input_respects_explicit_position(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([42, 8, 9]),
            mm_inputs=[],
            generate_config=GenerateConfig(),
            custom_output_token_position=1,
        )

        with unittest.mock.patch.dict(
            os.environ,
            {"CUSTOM_OUTPUT_TOKEN_POSITION": "-2"},
            clear=True,
        ):
            input_pb = trans_input(input_py)

        self.assertEqual(input_pb.custom_output_token_position.value, 1)

    def test_trans_input_rejects_relative_position_outside_prompt(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([42]),
            mm_inputs=[],
            generate_config=GenerateConfig(),
        )

        with unittest.mock.patch.dict(
            os.environ,
            {"CUSTOM_OUTPUT_TOKEN_POSITION": "-2"},
            clear=True,
        ):
            with self.assertRaisesRegex(FtRuntimeException, "outside prompt length 1"):
                trans_input(input_py)

    def test_trans_input_rejects_invalid_request_and_configured_positions(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([42]),
            mm_inputs=[],
            generate_config=GenerateConfig(),
            custom_output_token_position=-2,
        )
        with self.assertRaisesRegex(ValueError, "must be -1 or non-negative"):
            trans_input(input_py)

        input_py.custom_output_token_position = -1
        with unittest.mock.patch.dict(
            os.environ,
            {"CUSTOM_OUTPUT_TOKEN_POSITION": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be between"):
                trans_input(input_py)

    def test_trans_input_validates_expected_token_at_minus_two(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([7, 42, 8, 42, 9]),
            mm_inputs=[],
            generate_config=GenerateConfig(),
        )

        with unittest.mock.patch.dict(
            os.environ,
            {
                "CUSTOM_OUTPUT_TOKEN_POSITION": "-2",
                "CUSTOM_OUTPUT_EXPECTED_TOKEN_ID": "42",
            },
            clear=True,
        ):
            input_pb = trans_input(input_py)

        self.assertEqual(input_pb.custom_output_token_position.value, 3)

    def test_trans_input_rejects_unexpected_token_at_minus_two(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([7, 8, 9]),
            mm_inputs=[],
            generate_config=GenerateConfig(),
        )

        with unittest.mock.patch.dict(
            os.environ,
            {
                "CUSTOM_OUTPUT_TOKEN_POSITION": "-2",
                "CUSTOM_OUTPUT_EXPECTED_TOKEN_ID": "42",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(FtRuntimeException, "is 8, expected 42"):
                trans_input(input_py)

    def test_trans_input_rejects_legacy_tracked_token_selector(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([7, 42, 9]),
            mm_inputs=[],
            generate_config=GenerateConfig(),
        )

        with unittest.mock.patch.dict(
            os.environ,
            {
                "CUSTOM_OUTPUT_TRACKED_TOKEN_ID": "42",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "is not supported"):
                trans_input(input_py)

    def test_expected_token_requires_minus_two_selector(self):
        with unittest.mock.patch.dict(
            os.environ,
            {"CUSTOM_OUTPUT_EXPECTED_TOKEN_ID": "42"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "requires"):
                CustomOutputSelector.from_env()

    def test_custom_output_selector_validates_environment_at_construction(self):
        invalid_cases = [
            ({"CUSTOM_OUTPUT_TOKEN_POSITION": "not-an-int"}, "must be an integer"),
            ({"CUSTOM_OUTPUT_TOKEN_POSITION": "-1"}, "must be -2"),
            ({"CUSTOM_OUTPUT_TOKEN_POSITION": "-3"}, "must be -2"),
            (
                {
                    "CUSTOM_OUTPUT_TOKEN_POSITION": "-2",
                    "CUSTOM_OUTPUT_EXPECTED_TOKEN_ID": "not-an-int",
                },
                "must be an integer",
            ),
            (
                {
                    "CUSTOM_OUTPUT_TOKEN_POSITION": "-2",
                    "CUSTOM_OUTPUT_EXPECTED_TOKEN_ID": "-1",
                },
                "must be between",
            ),
        ]
        for environment, message in invalid_cases:
            with self.subTest(environment=environment):
                with unittest.mock.patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(ValueError, message):
                        CustomOutputSelector.from_env()

    @staticmethod
    def _make_generate_input(generate_config: GenerateConfig) -> GenerateInput:
        return GenerateInput(
            request_id=1,
            token_ids=torch.tensor([1], dtype=torch.int32),
            mm_inputs=[],
            generate_config=generate_config,
        )

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


if __name__ == "__main__":
    setup_logging()
    main()
