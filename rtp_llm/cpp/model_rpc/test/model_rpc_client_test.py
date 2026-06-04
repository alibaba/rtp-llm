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
from unittest.mock import patch

import grpc
import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.config.log_config import setup_logging
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    ModelRpcClient,
    StreamState,
    trans_input,
    trans_output,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    BatchGenerateOutputsPB,
    GenerateInputPB,
    GenerateOutputsPB,
    TensorPB,
)
from rtp_llm.utils.base_model_datatypes import GenerateInput, GenerateOutputs


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


class FakeBatchStub:

    def __init__(self):
        self.last_timeout = None
        self.last_batch_size = 0
        self.last_channel = None

    async def BatchGenerateCall(self, input, timeout=None):
        self.last_timeout = timeout
        self.last_batch_size = len(input.inputs)
        response = BatchGenerateOutputsPB()
        for _ in input.inputs:
            result = response.results.add()
            output_pb = result.final_output.flatten_output
            output_pb.output_ids.data_type = TensorPB.DataType.INT32
            output_pb.output_ids.shape.extend([1, 1])
            output_pb.output_ids.int32_data = struct.pack("<i", 0)
            aux_info = output_pb.aux_info.add()
            aux_info.iter_count = 1
            aux_info.output_len = 1
            output_pb.finished.extend([True])
        return response


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

    def test_batch_enqueue_returns_decode_role_addr_only_in_decode_entrance(self):
        client = ModelRpcClient(
            ["127.0.0.1:10101"],
            {},
            0,
            True,
        )
        fake_stub = FakeBatchStub()

        async def fake_get(_):
            return object()

        client._channel_pool.get = fake_get

        config_1 = GenerateConfig(aux_info=True)
        config_1.role_addrs = [
            SimpleNamespace(
                role=RoleType.PREFILL,
                ip="10.0.0.2",
                http_port=3000,
                grpc_port=3001,
            ),
            SimpleNamespace(
                role=RoleType.DECODE,
                ip="10.0.0.1",
                http_port=2000,
                grpc_port=2001,
            ),
        ]
        input_1 = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=config_1,
            request_id=0,
            mm_inputs=[],
        )
        config_2 = GenerateConfig(aux_info=True)
        config_2.role_addrs = [
            SimpleNamespace(
                role=RoleType.PREFILL,
                ip="10.0.0.2",
                http_port=3000,
                grpc_port=3001,
            ),
            SimpleNamespace(
                role=RoleType.DECODE,
                ip="10.0.0.1",
                http_port=2000,
                grpc_port=2001,
            ),
        ]
        input_2 = GenerateInput(
            token_ids=torch.tensor([4, 5, 6]),
            generate_config=config_2,
            request_id=1,
            mm_inputs=[],
        )

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=fake_stub,
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RoleAddr",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            results = asyncio.run(client.batch_enqueue([input_1, input_2]))

        self.assertEqual(fake_stub.last_batch_size, 2)
        result_role_addrs = results[0].generate_outputs[0].aux_info.role_addrs
        self.assertEqual(len(result_role_addrs), 1)
        self.assertEqual(
            {role_addr.role for role_addr in result_role_addrs},
            {RoleType.DECODE},
        )
        self.assertEqual(
            next(
                role_addr.ip
                for role_addr in result_role_addrs
                if role_addr.role == RoleType.DECODE
            ),
            "10.0.0.1",
        )

    def test_batch_enqueue_preserves_per_request_role_addrs_in_decode_entrance(self):
        client = ModelRpcClient(
            ["127.0.0.1:10101"],
            {},
            0,
            True,
        )
        fake_stub = FakeBatchStub()

        async def fake_get(_):
            return object()

        client._channel_pool.get = fake_get

        config_1 = GenerateConfig(aux_info=True)
        config_1.role_addrs = [
            SimpleNamespace(
                role=RoleType.PREFILL,
                ip="10.0.0.2",
                http_port=3000,
                grpc_port=3001,
            ),
            SimpleNamespace(
                role=RoleType.DECODE,
                ip="10.0.0.1",
                http_port=2000,
                grpc_port=2001,
            ),
        ]
        input_1 = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=config_1,
            request_id=0,
            mm_inputs=[],
        )
        config_2 = GenerateConfig(aux_info=True)
        config_2.role_addrs = [
            SimpleNamespace(
                role=RoleType.PREFILL,
                ip="10.0.0.3",
                http_port=4000,
                grpc_port=4001,
            ),
            SimpleNamespace(
                role=RoleType.DECODE,
                ip="10.0.0.1",
                http_port=2000,
                grpc_port=2001,
            ),
        ]
        input_2 = GenerateInput(
            token_ids=torch.tensor([4, 5, 6]),
            generate_config=config_2,
            request_id=1,
            mm_inputs=[],
        )

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=fake_stub,
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RoleAddr",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            results = asyncio.run(client.batch_enqueue([input_1, input_2]))

        self.assertEqual(fake_stub.last_batch_size, 2)
        self.assertEqual(
            {
                role_addr.role
                for role_addr in results[0].generate_outputs[0].aux_info.role_addrs
            },
            {RoleType.DECODE},
        )
        self.assertEqual(
            next(
                role_addr.ip
                for role_addr in results[0].generate_outputs[0].aux_info.role_addrs
                if role_addr.role == RoleType.DECODE
            ),
            "10.0.0.1",
        )
        self.assertEqual(
            next(
                role_addr.ip
                for role_addr in results[1].generate_outputs[0].aux_info.role_addrs
                if role_addr.role == RoleType.DECODE
            ),
            "10.0.0.1",
        )

    def test_batch_enqueue_uses_first_selected_backend_for_multi_address_batch(self):
        client = ModelRpcClient(
            ["10.0.0.10:10101", "10.0.0.11:10111"],
            {},
            0,
            True,
        )
        fake_stub = FakeBatchStub()
        requested_channels = []

        async def fake_get(address):
            requested_channels.append(address)
            return object()

        client._channel_pool.get = fake_get

        input_1 = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(aux_info=True),
            request_id=0,
            mm_inputs=[],
        )
        input_2 = GenerateInput(
            token_ids=torch.tensor([4, 5, 6]),
            generate_config=GenerateConfig(aux_info=True),
            request_id=1,
            mm_inputs=[],
        )

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=fake_stub,
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RoleAddr",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            results = asyncio.run(client.batch_enqueue([input_1, input_2]))

        self.assertEqual(fake_stub.last_batch_size, 2)
        self.assertEqual(requested_channels, ["10.0.0.10:10101"])
        for result in results:
            decode_role_addr = next(
                role_addr
                for role_addr in result.generate_outputs[0].aux_info.role_addrs
                if role_addr.role == RoleType.DECODE
            )
            self.assertEqual(decode_role_addr.ip, "10.0.0.10")
            self.assertEqual(decode_role_addr.grpc_port, 10101)
            self.assertEqual(decode_role_addr.http_port, 10100)

    def test_batch_enqueue_rejects_conflicting_explicit_backend_role_addrs(self):
        client = ModelRpcClient(
            ["10.0.0.10:10101", "10.0.0.11:10111"],
            {},
            0,
            True,
        )

        config_1 = GenerateConfig(aux_info=True)
        config_1.role_addrs = [
            SimpleNamespace(
                role=RoleType.DECODE,
                ip="10.0.0.10",
                http_port=10100,
                grpc_port=10101,
            ),
        ]
        config_2 = GenerateConfig(aux_info=True)
        config_2.role_addrs = [
            SimpleNamespace(
                role=RoleType.DECODE,
                ip="10.0.0.11",
                http_port=10110,
                grpc_port=10111,
            ),
        ]
        input_1 = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=config_1,
            request_id=0,
            mm_inputs=[],
        )
        input_2 = GenerateInput(
            token_ids=torch.tensor([4, 5, 6]),
            generate_config=config_2,
            request_id=1,
            mm_inputs=[],
        )

        with self.assertRaisesRegex(
            Exception,
            "conflicting explicit backends",
        ):
            asyncio.run(client.batch_enqueue([input_1, input_2]))

    def test_enqueue_fallback_decode_role_addr_uses_selected_target_address(self):
        client = ModelRpcClient(
            ["10.0.0.10:10101", "10.0.0.11:10111"],
            {},
            0,
            True,
        )

        async def fake_get(_):
            return object()

        client._channel_pool.get = fake_get

        response = GenerateOutputsPB()
        output_pb = response.flatten_output
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([1, 1])
        output_pb.output_ids.int32_data = struct.pack("<i", 0)
        aux_info = output_pb.aux_info.add()
        aux_info.iter_count = 1
        aux_info.output_len = 1
        output_pb.finished.extend([True])

        class _SingleResponseIterator:
            def __init__(self, response_pb):
                self._response_pb = response_pb
                self._done = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._done:
                    raise StopAsyncIteration
                self._done = True
                return self._response_pb

            def cancel(self):
                return None

        fake_stub = SimpleNamespace(
            GenerateStreamCall=lambda *_args, **_kwargs: _SingleResponseIterator(
                response
            )
        )

        config = GenerateConfig(aux_info=True)
        config.role_addrs = [
            SimpleNamespace(
                role=RoleType.PREFILL,
                ip="10.0.0.2",
                http_port=3000,
                grpc_port=3001,
            )
        ]
        input_obj = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=config,
            request_id=1,
            mm_inputs=[],
        )

        async def _collect_output():
            async for output in client.enqueue(input_obj):
                return output
            return None

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=fake_stub,
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RoleAddr",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            result = asyncio.run(_collect_output())

        self.assertIsNotNone(result)
        decode_role_addr = next(
            role_addr
            for role_addr in result.generate_outputs[0].aux_info.role_addrs
            if role_addr.role == RoleType.DECODE
        )
        self.assertEqual(decode_role_addr.ip, "10.0.0.11")
        self.assertEqual(decode_role_addr.grpc_port, 10111)
        self.assertEqual(decode_role_addr.http_port, 10110)

    def test_handle_grpc_error_maps_resource_exhausted_to_malloc_error(self):
        # Regression: prefill's LACK MEM surfaces to decode as a grpc
        # RESOURCE_EXHAUSTED status. Without an explicit case in the fallback
        # branch, _handle_grpc_error collapses it to UNKNOWN_ERROR (514), and
        # frontend_server reports "514_UNKNOWN_ERROR" instead of
        # "602_MALLOC_ERROR" on the error_qps metric.
        class _FakeRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.RESOURCE_EXHAUSTED

            def details(self):
                return "LACK MEM"

            def trailing_metadata(self):
                return ()

        client = ModelRpcClient(["127.0.0.1:10101"], {}, 0, False)
        with self.assertRaises(FtRuntimeException) as cm:
            client._handle_grpc_error(_FakeRpcError(), "test-request")
        self.assertEqual(cm.exception.exception_type, ExceptionType.MALLOC_ERROR)
        self.assertEqual(cm.exception.message, "LACK MEM")

    def test_trans_input_serializes_unique_key(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(unique_key="decode-batch-unique-key"),
            request_id=7,
            mm_inputs=[],
        )

        input_pb = trans_input(input_py)

        self.assertEqual(input_pb.generate_config.unique_key, "decode-batch-unique-key")


if __name__ == "__main__":
    setup_logging()
    main()
