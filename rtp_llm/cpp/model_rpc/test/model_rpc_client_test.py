import asyncio
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
from types import SimpleNamespace
from typing import AsyncGenerator
from unittest import TestCase, main
from unittest.mock import AsyncMock, patch

import grpc
import torch
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    ModelRpcClient,
    StreamState,
    batch_error_exception_type,
    trans_input,
    trans_output,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    ErrorCodePB,
    ErrorDetailsPB,
    GenerateInputPB,
    GenerateOutputsPB,
    TensorPB,
)
from rtp_llm.utils.base_model_datatypes import GenerateInput, GenerateOutputs

LONG_DURATION_US = (1 << 31) + 12345


class FakeStub:

    async def GenerateStreamCall(self, input: GenerateInputPB, timeout=None):
        # 1. 第一个响应：包含第一个生成的 token
        outputs_pb1 = GenerateOutputsPB()
        output_pb1 = outputs_pb1.flatten_output
        output_pb1.output_ids.data_type = TensorPB.DataType.INT32
        output_pb1.output_ids.shape.extend([1, 1])
        output_pb1.output_ids.int32_data = struct.pack("<i", 0)
        aux_info = output_pb1.aux_info.add()
        aux_info.cost_time_us = LONG_DURATION_US
        aux_info.first_token_cost_time_us = LONG_DURATION_US - 1
        aux_info.wait_time_us = LONG_DURATION_US - 2
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
            [],     # addresses: empty list for fake client
            {},     # client_config: empty dict for fake client
            0,      # max_rpc_timeout_ms
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
        self.assertEqual(res[0].aux_info.cost_time, LONG_DURATION_US / 1000.0)
        self.assertEqual(
            res[0].aux_info.first_token_cost_time, (LONG_DURATION_US - 1) / 1000.0
        )
        self.assertEqual(res[0].aux_info.wait_time, (LONG_DURATION_US - 2) / 1000.0)

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

    def test_trans_input_carries_unique_key(self):
        generate_config = GenerateConfig(unique_key="reuse-session-a")
        input = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=generate_config,
            request_id=123,
            mm_inputs=[],
        )

        input_pb = trans_input(input)

        self.assertEqual(input_pb.generate_config.unique_key, "reuse-session-a")

    def test_trans_input_preserves_random_seed_presence(self):
        for random_seed, expected_presence in (
            (None, False),
            ([], False),
            (0, True),
            (17, True),
        ):
            with self.subTest(random_seed=random_seed):
                generate_config = GenerateConfig(random_seed=random_seed)
                input = GenerateInput(
                    token_ids=torch.tensor([1, 2, 3]),
                    generate_config=generate_config,
                    request_id=123,
                    mm_inputs=[],
                )

                input_pb = trans_input(input)

                self.assertEqual(
                    input_pb.generate_config.HasField("random_seed"), expected_presence
                )
                if expected_presence:
                    self.assertEqual(
                        input_pb.generate_config.random_seed.value, random_seed
                    )

    def test_trans_input_keeps_empty_optional_lists_unset(self):
        generate_config = GenerateConfig(adapter_name=[])
        input = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=generate_config,
            request_id=123,
            mm_inputs=[],
        )

        input_pb = trans_input(input)

        self.assertFalse(input_pb.generate_config.HasField("adapter_name"))

    def test_compute_grpc_timeout_uses_remaining_budget_and_server_cap(self):
        client = ModelRpcClient([], {}, 100, False)

        self.assertAlmostEqual(client._compute_grpc_timeout(1000), 0.1, places=6)

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.current_monotonic_time_s",
            return_value=10.8,
        ):
            timeout = client._compute_grpc_timeout(1000, 11.0)

        self.assertAlmostEqual(timeout, 0.1, places=6)

    def test_compute_grpc_timeout_rejects_expired_budget(self):
        client = ModelRpcClient([], {}, 0, False)

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.current_monotonic_time_s",
            return_value=11.0,
        ):
            with self.assertRaises(FtRuntimeException) as context:
                client._compute_grpc_timeout(1000, 11.0)

        self.assertEqual(
            context.exception.exception_type, ExceptionType.GENERATE_TIMEOUT
        )

    def test_canonical_deadline_wins_over_conflicting_cancel_details(self):
        class DeadlineRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.DEADLINE_EXCEEDED

            def details(self):
                return "transport deadline"

            def trailing_metadata(self):
                details = ErrorDetailsPB(
                    error_code=int(ExceptionType.CANCELLED),
                    error_message="server observed cancellation after deadline",
                )
                return {"grpc-status-details-bin": details.SerializeToString()}

        client = ModelRpcClient([], {}, 0, False)

        with self.assertRaises(FtRuntimeException) as context:
            client._handle_grpc_error(DeadlineRpcError(), "request: [7]")

        self.assertEqual(
            context.exception.exception_type, ExceptionType.GENERATE_TIMEOUT
        )
        self.assertEqual(
            context.exception.message, "server observed cancellation after deadline"
        )

    def test_trans_input_carries_absolute_deadline_and_timeout_override(self):
        input = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(timeout_ms=1000),
            request_id=123,
            mm_inputs=[],
        )
        input.request_deadline_monotonic_s = 11.0
        input.request_deadline_unix_ms = 21_000

        input_pb = trans_input(input, timeout_ms=200)

        self.assertEqual(input_pb.generate_config.timeout_ms, 200)
        self.assertEqual(input_pb.request_deadline_unix_ms, 21_000)
        self.assertEqual(input.generate_config.timeout_ms, 1000)

    def test_enqueue_recomputes_remaining_budget_after_channel_and_serialization(self):
        class EmptyStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            def cancel(self):
                return None

        class CapturingStub:
            def __init__(self):
                self.input_pb = None
                self.timeout = None

            def GenerateStreamCall(self, input_pb, timeout=None):
                self.input_pb = input_pb
                self.timeout = timeout
                return EmptyStream()

        client = ModelRpcClient(["backend:1234"], {}, 0, False)
        client._channel_pool = SimpleNamespace(get=AsyncMock(return_value=object()))
        stub = CapturingStub()
        request = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(timeout_ms=1000),
            request_id=123,
            mm_inputs=[],
        )
        request.request_deadline_monotonic_s = 11.0
        request.request_deadline_unix_ms = 21_000

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.current_monotonic_time_s",
            side_effect=[10.0, 10.0, 10.7, 10.8],
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.current_unix_time_ms",
            return_value=20_000,
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            asyncio.run(self._run(client, request))

        self.assertEqual(stub.input_pb.generate_config.timeout_ms, 200)
        self.assertEqual(stub.input_pb.request_deadline_unix_ms, 21_000)
        self.assertAlmostEqual(stub.timeout, 0.2, places=6)
        self.assertEqual(request.generate_config.timeout_ms, 1000)

    def test_batch_preserves_each_item_budget_without_mutating_config(self):
        class CapturingStub:
            def __init__(self):
                self.input_pb = None
                self.timeout = None

            async def BatchGenerateCall(self, input_pb, timeout=None):
                self.input_pb = input_pb
                self.timeout = timeout
                return SimpleNamespace(results=[])

        client = ModelRpcClient(["backend:1234"], {}, 0, False)
        client._channel_pool = SimpleNamespace(get=AsyncMock(return_value=object()))
        stub = CapturingStub()
        requests = []
        for index, (deadline, unix_deadline) in enumerate(
            ((11.0, 21_000), (12.0, 22_000))
        ):
            request = GenerateInput(
                token_ids=torch.tensor([1]),
                generate_config=GenerateConfig(timeout_ms=1000),
                request_id=index,
                mm_inputs=[],
            )
            request.request_deadline_monotonic_s = deadline
            request.request_deadline_unix_ms = unix_deadline
            requests.append(request)

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.current_monotonic_time_s",
            side_effect=[10.0, 10.5],
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.current_unix_time_ms",
            return_value=20_000,
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            result = asyncio.run(client.batch_enqueue(requests))

        self.assertEqual(result, [])
        self.assertEqual(
            [item.generate_config.timeout_ms for item in stub.input_pb.inputs],
            [500, 1000],
        )
        self.assertAlmostEqual(stub.timeout, 1.0, places=6)
        self.assertEqual(
            [request.generate_config.timeout_ms for request in requests],
            [1000, 1000],
        )

    def test_batch_error_codes_preserve_timeout_and_cancel(self):
        self.assertEqual(
            batch_error_exception_type(ErrorCodePB.GENERATE_TIMEOUT),
            ExceptionType.GENERATE_TIMEOUT,
        )
        self.assertEqual(
            batch_error_exception_type(ErrorCodePB.CANCELLED),
            ExceptionType.CANCELLED_ERROR,
        )

    def test_generate_input_deadline_field_preserves_unknown_wire_field(self):
        file_proto = descriptor_pb2.FileDescriptorProto(
            name="legacy_generate_input.proto", syntax="proto3"
        )
        message = file_proto.message_type.add(name="LegacyGenerateInputPB")
        request_id = message.field.add(
            name="request_id",
            number=1,
            type=descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
        )
        request_id.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        generate_config = message.field.add(
            name="generate_config",
            number=4,
            type=descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
        )
        generate_config.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        pool = descriptor_pool.DescriptorPool()
        descriptor = pool.Add(file_proto).message_types_by_name[
            "LegacyGenerateInputPB"
        ]
        if hasattr(message_factory, "GetMessageClass"):
            legacy_type = message_factory.GetMessageClass(descriptor)
        else:
            legacy_type = message_factory.MessageFactory(pool).GetPrototype(descriptor)

        current = GenerateInputPB(request_id=7, request_deadline_unix_ms=21_000)
        current.generate_config.timeout_ms = 500
        legacy = legacy_type()
        legacy.ParseFromString(current.SerializeToString())
        self.assertEqual(legacy.request_id, 7)
        self.assertTrue(legacy.generate_config)

        current_roundtrip = GenerateInputPB()
        current_roundtrip.ParseFromString(legacy.SerializeToString())
        self.assertEqual(current_roundtrip.request_deadline_unix_ms, 21_000)

        legacy_only = legacy_type(request_id=9, generate_config=legacy.generate_config)
        parsed_by_current = GenerateInputPB()
        parsed_by_current.ParseFromString(legacy_only.SerializeToString())
        self.assertEqual(parsed_by_current.request_id, 9)
        self.assertEqual(parsed_by_current.generate_config.timeout_ms, 500)
        self.assertEqual(parsed_by_current.request_deadline_unix_ms, 0)

if __name__ == "__main__":
    setup_logging()
    main()
