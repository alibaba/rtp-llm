import asyncio
from enum import Enum
import struct
import sys
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

import torch

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
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
from rtp_llm.utils.base_model_datatypes import GenerateInput, GenerateOutputs, RequestInfo


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
        self.cancel_calls = []

    def FetchResponse(self, request, **kwargs):
        self.fetch_calls.append((request, kwargs))
        return self.fetch_iterator

    def GenerateStreamCall(self, request, **kwargs):
        self.generate_calls.append((request, kwargs))
        return self.generate_iterator

    async def Cancel(self, request, **kwargs):
        self.cancel_calls.append((request, kwargs))


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
        self.assertEqual(stub.cancel_calls, [])

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
        self.assertEqual(stub.cancel_calls, [])

    def test_enqueue_cancels_master_enqueued_fetch_on_early_close(self):
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
        self.assertEqual(len(stub.cancel_calls), 1)
        self.assertEqual(stub.cancel_calls[0][0].request_id, 323)
        self.assertEqual(stub.cancel_calls[0][1]["timeout"], 5.0)

    def test_enqueue_fetch_cancel_uses_prefill_when_decode_entrance(self):
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
        self.assertEqual(len(stub.cancel_calls), 1)
        self.assertEqual(stub.cancel_calls[0][0].request_id, 325)

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
        self.assertEqual(stub.cancel_calls, [])


if __name__ == "__main__":
    setup_logging()
    main()
