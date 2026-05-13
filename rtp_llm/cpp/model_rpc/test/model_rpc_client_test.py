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


class FakeGrpcStream:
    """Mimics a gRPC async response stream with __aiter__ and cancel()."""

    def __init__(self, outputs):
        self._outputs = outputs

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for out in self._outputs:
            yield out

    def cancel(self):
        pass


class DpControllerFetchOnlyTest(TestCase):
    """Verify that DP_CONTROLLER_MANAGED mode only does FetchResponse (no Enqueue)."""

    def _make_client(self, addresses):
        ModelRpcClient, _, _, _ = _load_model_rpc_symbols()
        with patch.dict(os.environ, {"DP_CONTROLLER_MANAGED": "true"}):
            client = ModelRpcClient(addresses, {}, 0, False)
        return client

    def _make_input(self, request_id=0):
        return GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(using_hf_sampling=False),
            request_id=request_id,
            mm_inputs=[],
        )

    def _make_final_output(self):
        out = GenerateOutputsPB()
        out.flatten_output.finished.extend([True])
        out.flatten_output.output_ids.data_type = TensorPB.DataType.INT32
        out.flatten_output.output_ids.shape.extend([1, 1])
        out.flatten_output.output_ids.int32_data = struct.pack("<i", 0)
        aux = out.flatten_output.aux_info.add()
        aux.iter_count = 1
        aux.output_len = 1
        return out

    def test_fetch_response_only_no_enqueue(self):
        """In DP_CONTROLLER_MANAGED mode, enqueue() should call FetchResponse directly, never Enqueue."""
        addresses = ["dp0:20001"]
        client = self._make_client(addresses)
        final_output = self._make_final_output()
        calls = {"fetch": 0, "enqueue": 0}

        async def mock_pool_get(addr):
            return MagicMock(name=f"channel_{addr}")
        client._channel_pool = MagicMock()
        client._channel_pool.get = mock_pool_get

        class StubTracker:
            def __init__(self, channel):
                pass
            async def Enqueue(self, req, timeout=None):
                calls["enqueue"] += 1
            def FetchResponse(self, req, timeout=None):
                calls["fetch"] += 1
                return FakeGrpcStream([final_output])

        with patch("rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub", StubTracker):
            async def run():
                return [out async for out in client.enqueue(self._make_input(request_id=0))]
            asyncio.run(run())

        self.assertEqual(calls["enqueue"], 0, "Should NOT call Enqueue")
        self.assertEqual(calls["fetch"], 1, "Should call FetchResponse exactly once")

    def test_cancel_on_early_exit(self):
        """On early exit, Cancel should be sent to the DPx."""
        addresses = ["dp0:20001"]
        client = self._make_client(addresses)
        non_final = self._make_final_output()
        non_final.flatten_output.finished[0] = False
        cancel_calls = []

        async def mock_pool_get(addr):
            return MagicMock(name=f"channel_{addr}")
        client._channel_pool = MagicMock()
        client._channel_pool.get = mock_pool_get

        class StubTracker:
            def __init__(self, channel):
                pass
            def FetchResponse(self, req, timeout=None):
                return FakeGrpcStream([non_final, non_final, non_final])
            async def Cancel(self, req, timeout=None):
                cancel_calls.append(req.request_id)

        with patch("rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub", StubTracker):
            async def run():
                async for out in client.enqueue(self._make_input(request_id=42)):
                    break  # early exit → triggers Cancel
            asyncio.run(run())

        self.assertEqual(len(cancel_calls), 1)
        self.assertEqual(cancel_calls[0], 42)

    def test_role_addrs_determines_fetch_target(self):
        """role_addrs from FlexLB should determine which address to FetchResponse from."""
        from rtp_llm.config.generate_config import RoleAddr

        addresses = ["dp0:20001", "dp1:20001"]
        client = self._make_client(addresses)
        final_output = self._make_final_output()
        fetch_targets = []

        async def mock_pool_get(addr):
            return MagicMock(name=f"channel_{addr}")
        client._channel_pool = MagicMock()
        client._channel_pool.get = mock_pool_get

        class StubTracker:
            def __init__(self, channel):
                self._channel = channel
            def FetchResponse(self, req, timeout=None):
                fetch_targets.append(self._channel._mock_name)
                return FakeGrpcStream([final_output])

        input_py = self._make_input(request_id=0)
        input_py.generate_config.role_addrs = [
            RoleAddr(role=RoleType.PREFILL, ip="10.0.0.99", grpc_port=30001, http_port=8080)
        ]

        with patch("rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub", StubTracker):
            async def run():
                return [out async for out in client.enqueue(input_py)]
            asyncio.run(run())

        self.assertEqual(len(fetch_targets), 1)
        self.assertEqual(fetch_targets[0], "channel_10.0.0.99:30001")


if __name__ == "__main__":
    setup_logging()
    main()
