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

from rtp_llm.config.generate_config import GenerateConfig
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

    def test_compact_logprobs_config_and_output_roundtrip(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=2),
            request_id=123,
            mm_inputs=[],
        )
        input_pb = trans_input(input_py)
        self.assertTrue(input_pb.generate_config.return_logprobs)
        self.assertEqual(input_pb.generate_config.top_logprobs, 2)

        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.append(False)
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([1, 2])
        output_pb.output_ids.int32_data = struct.pack("<ii", 10, 11)

        output_pb.token_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.token_logprobs.shape.extend([1, 2])
        output_pb.token_logprobs.fp32_data = struct.pack("<ff", -0.1, -0.2)

        output_pb.top_logprob_token_ids.data_type = TensorPB.DataType.INT32
        output_pb.top_logprob_token_ids.shape.extend([1, 2, 2])
        output_pb.top_logprob_token_ids.int32_data = struct.pack(
            "<iiii", 10, 12, 11, 13
        )

        output_pb.top_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.top_logprobs.shape.extend([1, 2, 2])
        output_pb.top_logprobs.fp32_data = struct.pack("<ffff", -0.1, -1.1, -0.2, -1.2)
        output_pb.logprobs_offsets.append(0)
        output_pb.logprobs_counts.append(2)

        result = trans_output(input_py, outputs_pb, StreamState())
        self.assertEqual(len(result.generate_outputs), 1)
        output = result.generate_outputs[0]
        self.assertEqual(output.token_logprobs.shape, torch.Size([2]))
        self.assertEqual(output.top_logprob_token_ids.shape, torch.Size([2, 2]))
        self.assertEqual(output.top_logprobs.shape, torch.Size([2, 2]))
        self.assertEqual(output.logprobs_offset, 0)
        self.assertEqual(output.logprobs_count, 2)
        self.assertTrue(
            torch.equal(
                output.top_logprob_token_ids,
                torch.tensor([[10, 12], [11, 13]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.allclose(
                output.token_logprobs,
                torch.tensor([-0.1, -0.2], dtype=torch.float32),
            )
        )

    def test_compact_logprobs_zero_top_k_output_roundtrip(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=0),
            request_id=123,
            mm_inputs=[],
        )

        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.append(False)
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([1, 2])
        output_pb.output_ids.int32_data = struct.pack("<ii", 10, 11)

        output_pb.token_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.token_logprobs.shape.extend([1, 2])
        output_pb.token_logprobs.fp32_data = struct.pack("<ff", -0.1, -0.2)

        output_pb.top_logprob_token_ids.data_type = TensorPB.DataType.INT32
        output_pb.top_logprob_token_ids.shape.extend([1, 2, 0])
        output_pb.top_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.top_logprobs.shape.extend([1, 2, 0])
        output_pb.logprobs_offsets.append(0)
        output_pb.logprobs_counts.append(2)

        result = trans_output(input_py, outputs_pb, StreamState())

        self.assertEqual(len(result.generate_outputs), 1)
        output = result.generate_outputs[0]
        self.assertEqual(output.token_logprobs.shape, torch.Size([2]))
        self.assertEqual(output.top_logprob_token_ids.shape, torch.Size([2, 0]))
        self.assertEqual(output.top_logprob_token_ids.dtype, torch.int32)
        self.assertEqual(output.top_logprobs.shape, torch.Size([2, 0]))
        self.assertEqual(output.top_logprobs.dtype, torch.float32)

    def test_compact_logprobs_boundary_uses_count_to_remove_rpc_padding(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=1),
            request_id=123,
            mm_inputs=[],
        )

        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.extend([False, False])
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([2, 1, 5])
        output_pb.output_ids.int32_data = struct.pack(
            "<10i", 10, 11, 128822, 271, 20, 30, 31, 32, 0, 0
        )

        # Row 0 owns two real content rows plus one transport padding row;
        # row 1 owns all three rows.
        output_pb.token_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.token_logprobs.shape.extend([2, 3])
        output_pb.token_logprobs.fp32_data = struct.pack(
            "<6f", -0.13, -0.20, 0.0, -0.30, -0.31, -0.32
        )
        output_pb.top_logprob_token_ids.data_type = TensorPB.DataType.INT32
        output_pb.top_logprob_token_ids.shape.extend([2, 3, 1])
        output_pb.top_logprob_token_ids.int32_data = struct.pack(
            "<6i", 271, 20, 0, 30, 31, 32
        )
        output_pb.top_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.top_logprobs.shape.extend([2, 3, 1])
        output_pb.top_logprobs.fp32_data = struct.pack(
            "<6f", -0.13, -0.20, 0.0, -0.30, -0.31, -0.32
        )
        output_pb.logprobs_offsets.extend([3, 0])
        output_pb.logprobs_counts.extend([2, 3])

        result = trans_output(input_py, outputs_pb, StreamState())

        first, second = result.generate_outputs
        self.assertEqual(first.logprobs_offset, 3)
        self.assertEqual(first.logprobs_count, 2)
        self.assertEqual(first.token_logprobs.shape, torch.Size([2]))
        self.assertTrue(
            torch.allclose(
                first.token_logprobs,
                torch.tensor([-0.13, -0.20], dtype=torch.float32),
            )
        )
        self.assertEqual(first.top_logprob_token_ids.tolist(), [[271], [20]])
        self.assertEqual(second.logprobs_offset, 0)
        self.assertEqual(second.logprobs_count, 3)
        self.assertEqual(second.token_logprobs.shape, torch.Size([3]))

    def test_compact_logprobs_thinking_only_metadata_survives_without_tensors(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=0),
            request_id=123,
            mm_inputs=[],
        )

        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.append(False)
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([1, 3])
        output_pb.output_ids.int32_data = struct.pack("<3i", 10, 11, 128822)
        output_pb.logprobs_offsets.append(3)
        output_pb.logprobs_counts.append(0)

        result = trans_output(input_py, outputs_pb, StreamState())

        output = result.generate_outputs[0]
        self.assertEqual(output.logprobs_offset, 3)
        self.assertEqual(output.logprobs_count, 0)
        self.assertIsNone(output.token_logprobs)
        self.assertIsNone(output.top_logprob_token_ids)
        self.assertIsNone(output.top_logprobs)

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
