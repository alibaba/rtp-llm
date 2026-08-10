import asyncio
import json
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
from unittest.mock import patch

import grpc
import torch

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleType
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.response_format_compiler import ReasoningFormat
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

    # The single-request enqueue() and the batch path now share _role_addr_target for the
    # role->address rule; pin the helper directly so a change to either path cannot quietly
    # diverge the two on where a pre-assigned request lands.
    def test_role_addr_target_returns_the_pre_assigned_backend(self):
        client = self._client(["10.0.0.99:100"])
        target = client._role_addr_target(
            self._input(10, [self._addr("10.0.0.7", RoleType.PDFUSION)])
        )
        self.assertEqual("10.0.0.7:8089", target)

    def test_role_addr_target_is_none_when_nothing_pre_assigned(self):
        client = self._client(["10.0.0.1:100"])
        self.assertIsNone(client._role_addr_target(self._input(11)))

    def test_role_addr_target_skips_an_empty_ip_placeholder(self):
        client = self._client(["10.0.0.1:100"])
        self.assertIsNone(
            client._role_addr_target(self._input(12, [self._addr("", RoleType.PDFUSION)]))
        )


class EnqueueTargetSelectionTest(TestCase):
    """enqueue() delegates the role->address decision to _role_addr_target and then wraps it:
    a hit becomes a single-target list, a miss falls back to the static data-parallel list, and
    the request_id indexes into whichever list wins. That wiring never runs in the FakeModelRpcClient
    (it overrides enqueue outright), so drive the real body far enough to capture the chosen target
    and stop before any gRPC I/O.
    """

    class _StopBeforeRpc(Exception):
        pass

    @staticmethod
    def _client(addresses):
        client = ModelRpcClient.__new__(ModelRpcClient)
        client._addresses = addresses
        client._decode_entrance = False
        client._compute_grpc_timeout = lambda timeout_ms: 1.0
        return client

    @staticmethod
    def _addr(ip, role, grpc_port=8089):
        return SimpleNamespace(role=role, ip=ip, http_port=8088, grpc_port=grpc_port)

    @staticmethod
    def _input(request_id, role_addrs=()):
        return SimpleNamespace(
            request_id=request_id,
            generate_config=SimpleNamespace(
                role_addrs=list(role_addrs), timeout_ms=1000
            ),
        )

    def _capture_target(self, client, input_py, calls=None):
        captured = {}

        async def fake_get(address):
            captured["address"] = address
            raise EnqueueTargetSelectionTest._StopBeforeRpc()

        client._channel_pool = SimpleNamespace(get=fake_get)

        async def drive():
            async for _ in client.enqueue(input_py):
                pass

        def counting_trans_input(x):
            if calls is not None:
                calls.append(x)
            return object()

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.trans_input", counting_trans_input
        ):
            with self.assertRaises(EnqueueTargetSelectionTest._StopBeforeRpc):
                asyncio.run(drive())
        return captured["address"]

    def test_enqueue_builds_the_request_pb_exactly_once(self):
        # trans_input validates the generate_config and copies every field into a fresh PB, so
        # building it twice doubles that cost on every streaming request. enqueue once did exactly
        # that (the first result was overwritten); pin the single build so it cannot come back.
        client = self._client(["10.0.0.1:100"])
        calls = []
        self._capture_target(client, self._input(0), calls=calls)
        self.assertEqual(1, len(calls), "enqueue must build the request PB exactly once")

    def test_enqueue_sends_to_the_pre_assigned_backend(self):
        client = self._client(["10.9.9.9:1"])
        input_py = self._input(0, [self._addr("10.0.0.7", RoleType.PDFUSION)])
        self.assertEqual("10.0.0.7:8089", self._capture_target(client, input_py))

    def test_enqueue_falls_back_to_static_addresses_when_unrouted(self):
        client = self._client(["10.0.0.1:100", "10.0.0.2:100"])
        # request_id=1 -> index 1 in the static list, proving the fallback both selects the
        # static list and keeps the request_id modulo indexing.
        self.assertEqual("10.0.0.2:100", self._capture_target(client, self._input(1)))


class BatchEnqueueDecodeSemanticsTest(TestCase):
    """Exercises the REAL ModelRpcClient.batch_enqueue decode/error path.

    The routing tests above stub the client's batch_enqueue wholesale, so the gRPC call, the
    results-decode loop, the inputs[i] <-> results[i] alignment, the raise-on-first-error
    (chunk-level all-or-nothing) and the grpc.RpcError translation had no coverage at all. This
    class drives that method body with a faked stub. trans_input/trans_output are the separately
    tested leaf decoders; here they are spied so a mis-alignment (trans_output fed the wrong
    input) or a dropped/renumbered error index turns the test red.
    """

    @staticmethod
    def _client():
        client = ModelRpcClient.__new__(ModelRpcClient)
        client._addresses = ["10.0.0.1:8089"]
        client._decode_entrance = False
        client._max_rpc_timeout_ms = 30000

        class _Pool:
            async def get(self, addr):
                return object()

        client._channel_pool = _Pool()
        return client

    @staticmethod
    def _input(request_id, role_addrs=()):
        return SimpleNamespace(
            request_id=request_id,
            prompt_length=4,
            generate_config=SimpleNamespace(timeout_ms=1000, role_addrs=list(role_addrs)),
        )

    @staticmethod
    def _addr(ip):
        return SimpleNamespace(role=RoleType.PDFUSION, ip=ip, grpc_port=8089)

    @staticmethod
    def _stub(response=None, error=None):
        class _FakeStub:
            def __init__(self, channel):
                pass

            async def BatchGenerateCall(self, batch_input_pb, timeout=None):
                if error is not None:
                    raise error
                return response

        return _FakeStub

    def _run(self, client, inputs, *, response=None, error=None, seen=None):
        def spy_trans_output(input_py, final_output, stream_state):
            if seen is not None:
                seen.append(input_py)
            return ("OUT", input_py.request_id)

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            new=self._stub(response=response, error=error),
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.trans_input",
            new=lambda inp: GenerateInputPB(),
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.trans_output",
            new=spy_trans_output,
        ):
            return asyncio.run(client.batch_enqueue(inputs))

    def test_empty_batch_short_circuits_without_touching_the_stub(self):
        # error would fire if the stub were reached; an empty batch must not reach it.
        self.assertEqual(
            [], self._run(self._client(), [], error=RuntimeError("stub reached"))
        )

    def test_all_success_returns_one_output_per_input_in_order(self):
        resp = BatchGenerateOutputsPB()
        resp.results.add()  # result[0]: no error_info -> success
        resp.results.add()  # result[1]: success
        inputs = [self._input(0), self._input(1)]
        seen = []

        out = self._run(self._client(), inputs, response=resp, seen=seen)

        self.assertEqual([("OUT", 0), ("OUT", 1)], out)
        # inputs[i] <-> results[i]: decode input 0 then input 1, asserted by identity so that
        # trans_output(inputs[0], results[1]) or a reused index cannot pass.
        self.assertIs(inputs[0], seen[0])
        self.assertIs(inputs[1], seen[1])

    def test_first_item_error_aborts_whole_chunk_and_names_the_index(self):
        resp = BatchGenerateOutputsPB()
        resp.results.add()  # result[0]: success, decoded before the failure is seen ...
        r1 = resp.results.add()  # result[1]: failure
        r1.error_info.error_message = "boom"
        inputs = [self._input(0), self._input(1)]
        seen = []

        with self.assertRaises(FtRuntimeException) as ctx:
            self._run(self._client(), inputs, response=resp, seen=seen)

        # chunk-level all-or-nothing: item 0's success was computed ...
        self.assertIs(inputs[0], seen[0])
        # ... yet the call raised (discarding it) and named the failing index.
        self.assertIn("batch item 1", str(ctx.exception))

    def test_error_index_is_the_failing_position_not_a_constant(self):
        # Guards against a hard-coded index or an off-by-one in the error message.
        resp = BatchGenerateOutputsPB()
        resp.results.add()
        resp.results.add()
        r2 = resp.results.add()
        r2.error_info.error_message = "third one failed"
        inputs = [self._input(0), self._input(1), self._input(2)]

        with self.assertRaises(FtRuntimeException) as ctx:
            self._run(self._client(), inputs, response=resp)
        self.assertIn("batch item 2", str(ctx.exception))

    def test_grpc_error_is_translated_to_ft_runtime_exception(self):
        class _FakeRpcError(grpc.RpcError):
            def trailing_metadata(self):
                return {}

            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self):
                return "backend unavailable"

        # grpc.RpcError must not leak raw; batch_enqueue funnels it through _handle_grpc_error.
        with self.assertRaises(FtRuntimeException):
            self._run(self._client(), [self._input(0)], error=_FakeRpcError())

    def test_error_info_present_but_empty_message_is_treated_as_success(self):
        # The decode guard is `HasField("error_info") AND error_info.error_message`: an error_info
        # sub-message that is set but carries no message must NOT abort the chunk. Mutation guard:
        # drop the second conjunct (raise on any error_info present) and item 0 wrongly raises.
        resp = BatchGenerateOutputsPB()
        r0 = resp.results.add()
        r0.error_info.SetInParent()  # error_info present (HasField True) but error_message == ""
        resp.results.add()  # result[1]: plain success
        inputs = [self._input(0), self._input(1)]
        seen = []

        out = self._run(self._client(), inputs, response=resp, seen=seen)

        self.assertEqual([("OUT", 0), ("OUT", 1)], out)
        self.assertIs(inputs[0], seen[0])

    def test_result_count_mismatch_fails_loudly_instead_of_silently_truncating(self):
        # The C++ contract is 1:1 (one result per input). A short result vector must fail typed,
        # not silently return a truncated list (trailing inputs dropped with no error). Mutation
        # guard: drop the length check and this returns [("OUT", 0)] for two inputs instead of
        # raising.
        resp = BatchGenerateOutputsPB()
        resp.results.add()  # only 1 result for 2 inputs
        inputs = [self._input(0), self._input(1)]

        with self.assertRaises(FtRuntimeException) as ctx:
            self._run(self._client(), inputs, response=resp)
        self.assertIn("1:1 per-item contract", str(ctx.exception))

    def test_result_count_longer_than_inputs_also_raises(self):
        # The guard's other direction: MORE results than inputs (which would otherwise IndexError
        # on inputs[i] for the surplus i) must fail typed with the same 1:1-contract error. The
        # docstring/comment names both failure modes; this pins the longer one. 3 results / 2 inputs.
        resp = BatchGenerateOutputsPB()
        resp.results.add()
        resp.results.add()
        resp.results.add()
        inputs = [self._input(0), self._input(1)]

        with self.assertRaises(FtRuntimeException) as ctx:
            self._run(self._client(), inputs, response=resp)
        self.assertIn("1:1 per-item contract", str(ctx.exception))

    def test_mixed_pre_assigned_batch_raises_before_reaching_the_stub(self):
        # _select_batch_address runs in the real body AFTER trans_input on every input but BEFORE
        # the gRPC call. Two inputs pre-assigned to different backends is a caller assembly error
        # that must fail typed (INVALID_PARAMS) without shipping the batch to either backend. The
        # isolated BatchEnqueueRoutingTest drives _select_batch_address directly; this pins the same
        # guard inside batch_enqueue's real ordering. error= would fire if the stub were reached.
        inputs = [
            self._input(0, [self._addr("10.0.0.7")]),
            self._input(1, [self._addr("10.0.0.8")]),
        ]
        with self.assertRaises(FtRuntimeException) as ctx:
            self._run(
                self._client(), inputs, error=RuntimeError("stub must not be reached")
            )
        self.assertIn("one batch RPC targets one backend", str(ctx.exception))


if __name__ == "__main__":
    setup_logging()
    main()
