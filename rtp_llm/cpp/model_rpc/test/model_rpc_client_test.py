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

from rtp_llm.config.exceptions import FtRuntimeException
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
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
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

        # Typed (not a bare ValueError) so the HTTP layer reports a deterministic
        # client-error code for a caller-assembled mixed batch.
        with self.assertRaises(FtRuntimeException):
            client._select_batch_address(inputs)

    def test_empty_static_addresses_raises_instead_of_dividing_by_zero(self):
        client = self._client([])
        with self.assertRaises(ValueError):
            client._select_batch_address([self._input(7)])

    def test_decode_entrance_honours_decode_role_addr(self):
        client = self._client(["10.9.9.9:1"])
        client._decode_entrance = True
        inputs = [
            self._input(
                8,
                [
                    self._addr("10.0.0.5", RoleType.PREFILL),
                    self._addr("10.0.0.6", RoleType.DECODE, grpc_port=9000),
                ],
            )
        ]

        self.assertEqual("10.0.0.6:9000", client._select_batch_address(inputs))

    def test_role_addr_with_empty_ip_is_skipped_not_selected(self):
        client = self._client(["10.0.0.1:100", "10.0.0.2:100"])
        inputs = [self._input(9, [self._addr("", RoleType.PDFUSION)])]

        # An empty-ip role_addr is a placeholder, not an assignment: the batch must
        # fall back to the static address list exactly like an unrouted input.
        self.assertEqual("10.0.0.2:100", client._select_batch_address(inputs))


class BatchEnqueueRoutingTest(TestCase):
    """A batch RPC is one scheduling unit — one BatchGenerateCall to one backend. If the visitor
    routed each input separately, a multi-worker master could legally stamp a different backend
    on every input, and _select_batch_address would (rightly) reject the batch as having no
    single valid target. So BackendRPCServerVisitor.batch_enqueue must route once and propagate
    the same assignment to every unrouted input.
    """

    @staticmethod
    def _visitor(master_rotation):
        """Visitor stub whose route_ips simulates a round-robin master: each call stamps the
        next address from master_rotation, exactly what a real multi-worker master may do."""
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 1024
        visitor.sp_config = None
        visitor.host_service = SimpleNamespace(service_available=True)
        sent = {}

        async def fake_batch_enqueue(inputs):
            sent["inputs"] = inputs
            return []

        visitor.model_rpc_client = SimpleNamespace(batch_enqueue=fake_batch_enqueue)
        route_calls = []

        async def fake_route_ips(inp, seq_len_hint=None):
            inp.generate_config.role_addrs = [
                master_rotation[len(route_calls) % len(master_rotation)]
            ]
            route_calls.append((inp, seq_len_hint))

        visitor.route_ips = fake_route_ips
        return visitor, route_calls, sent

    @staticmethod
    def _addr(ip):
        return SimpleNamespace(
            role=RoleType.PDFUSION, ip=ip, http_port=8088, grpc_port=8089
        )

    @staticmethod
    def _input(request_id, role_addrs=()):
        return SimpleNamespace(
            request_id=request_id,
            prompt_length=8,
            generate_config=SimpleNamespace(
                role_addrs=list(role_addrs), max_new_tokens=16
            ),
        )

    def test_round_robin_master_cannot_scatter_a_batch_across_backends(self):
        visitor, route_calls, sent = self._visitor(
            [self._addr("10.0.0.1"), self._addr("10.0.0.2")]
        )
        inputs = [self._input(i) for i in range(4)]

        asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(
            1, len(route_calls), "a batch is one scheduling unit: one master round-trip"
        )
        # The single routing call must carry the batch's aggregate weight — otherwise
        # the master accounts one request's load while N inputs land on the worker.
        self.assertEqual(
            sum(inp.prompt_length for inp in inputs), route_calls[0][1]
        )
        # Every input carries the first routing decision, and the whole batch resolves to a
        # single target through the real _select_batch_address — the exact seam that a
        # per-input routing loop breaks under a round-robin master.
        client = ModelRpcClient.__new__(ModelRpcClient)
        client._addresses = ["10.9.9.9:1"]
        client._decode_entrance = False
        self.assertEqual("10.0.0.1:8089", client._select_batch_address(inputs))
        self.assertIs(inputs, sent["inputs"])
        # Propagated as copies: one input's later mutation must not silently retarget siblings.
        self.assertIsNot(
            inputs[0].generate_config.role_addrs,
            inputs[1].generate_config.role_addrs,
        )

    def test_dispatcher_pre_assigned_chunk_never_touches_the_master(self):
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.9")])
        stamped = self._addr("10.0.0.7")
        inputs = [self._input(i, [stamped]) for i in range(3)]

        asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(
            0,
            len(route_calls),
            "pre-assigned chunks carry the dispatcher's decision; re-routing would discard it",
        )
        client = ModelRpcClient.__new__(ModelRpcClient)
        client._addresses = []
        client._decode_entrance = False
        self.assertEqual("10.0.0.7:8089", client._select_batch_address(inputs))

    def test_mixed_batch_converges_when_master_agrees_with_pre_assignment(self):
        # A caller-assembled batch mixing pre-assigned and unrouted inputs is legal
        # exactly when everything ends up on one backend: the unrouted subset gets one
        # routing call, and if the master picks the same worker the batch goes through.
        shared = self._addr("10.0.0.7")
        visitor, route_calls, sent = self._visitor([shared])
        inputs = [self._input(0, [shared]), self._input(1), self._input(2)]

        asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(1, len(route_calls))
        client = ModelRpcClient.__new__(ModelRpcClient)
        client._addresses = []
        client._decode_entrance = False
        self.assertEqual("10.0.0.7:8089", client._select_batch_address(inputs))

    def test_mixed_batch_is_rejected_when_master_disagrees_with_pre_assignment(self):
        # ...and when the master picks a different worker, the batch must fail loudly
        # (typed error) instead of silently shipping the pre-assigned input to the
        # wrong backend.
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.9")])
        inputs = [self._input(0, [self._addr("10.0.0.7")]), self._input(1)]

        asyncio.run(visitor.batch_enqueue(inputs))

        client = ModelRpcClient.__new__(ModelRpcClient)
        client._addresses = []
        client._decode_entrance = False
        with self.assertRaises(FtRuntimeException):
            client._select_batch_address(inputs)


if __name__ == "__main__":
    setup_logging()
    main()
