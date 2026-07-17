import asyncio
import struct
import sys
from types import SimpleNamespace
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

from rtp_llm.config.generate_config import GenerateConfig, RoleType
from rtp_llm.config.log_config import setup_logging
from rtp_llm.cpp.model_rpc.model_rpc_client import (
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


class BatchAddressSelectionTest(TestCase):
    """The dispatcher stamps generate_config.role_addrs on the prompt_batch endpoints, and
    route_ips then skips FE's own master round-trip because they are present. batch_enqueue must
    actually send the chunk to that pre-assigned backend — otherwise the scheduling decision is
    silently discarded and the request lands on whatever the static address list says.

    rtp_llm.ops is mocked at the top of this file (no CUDA here), so RoleType is a MagicMock and a
    real pydantic RoleAddr cannot be built. _select_batch_address only reads a few attributes, and
    the role comparison is against that same mocked RoleType, so lightweight stubs exercise the
    real logic faithfully.
    """

    @staticmethod
    def _client(addresses):
        client = ModelRpcClient.__new__(ModelRpcClient)
        client._addresses = addresses
        client._decode_entrance = False
        return client

    @staticmethod
    def _addr(ip, role, grpc_port=8089):
        return SimpleNamespace(role=role, ip=ip, http_port=8088, grpc_port=grpc_port)

    @staticmethod
    def _input(request_id, role_addrs=()):
        return SimpleNamespace(
            request_id=request_id,
            generate_config=SimpleNamespace(role_addrs=list(role_addrs)),
        )

    def test_pre_assigned_role_addr_wins_over_static_addresses(self):
        client = self._client(["10.0.0.99:100"])
        addr = self._addr("10.0.0.7", RoleType.PDFUSION)
        inputs = [self._input(1, [addr]), self._input(2, [addr])]

        self.assertEqual("10.0.0.7:8089", client._select_batch_address(inputs))

    def test_falls_back_to_static_addresses_when_nothing_pre_assigned(self):
        client = self._client(["10.0.0.1:100", "10.0.0.2:100"])
        inputs = [self._input(3), self._input(4)]

        self.assertEqual("10.0.0.2:100", client._select_batch_address(inputs))

    def test_inconsistent_pre_assignment_is_rejected_not_silently_mis_routed(self):
        client = self._client(["10.0.0.1:100"])
        inputs = [
            self._input(5, [self._addr("10.0.0.7", RoleType.PDFUSION)]),
            self._input(6, [self._addr("10.0.0.8", RoleType.PDFUSION)]),
        ]

        with self.assertRaises(ValueError):
            client._select_batch_address(inputs)

    def test_empty_static_addresses_raises_instead_of_dividing_by_zero(self):
        client = self._client([])
        with self.assertRaises(ValueError):
            client._select_batch_address([self._input(7)])


if __name__ == "__main__":
    setup_logging()
    main()
