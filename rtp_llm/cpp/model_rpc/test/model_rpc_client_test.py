import asyncio
import logging
import os
import struct
import unittest
from contextlib import asynccontextmanager
from dataclasses import asdict
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.config.log_config import setup_logging
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    ModelRpcClient,
    StreamState,
    trans_input,
    trans_output,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    ErrorDetailsPB,
    GenerateInputPB,
    GenerateOutputsPB,
    RoleAddrPB,
    TensorPB,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.base_model_datatypes import GenerateInput, RequestInfo
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


class FakeStub:
    def GenerateStreamCall(self, input: GenerateInputPB, timeout=None):
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

        # 2. 第二个响应：包含累积的两个 token
        outputs_pb2 = GenerateOutputsPB()
        output_pb2 = outputs_pb2.flatten_output
        output_pb2.output_ids.data_type = TensorPB.DataType.INT32
        output_pb2.output_ids.shape.extend([1, 2])
        output_pb2.output_ids.int32_data = struct.pack("<ii", 0, 1)
        aux_info2 = output_pb2.aux_info.add()
        aux_info2.iter_count = 2
        aux_info2.output_len = 2
        aux_info2.speculative_verify_rounds = 3
        aux_info2.speculative_accepted_token_num = 9
        aux_info2.speculative_proposed_draft_tokens = 12
        aux_info2.context_execute_time_us = 100
        aux_info2.generate_execute_time_us = 200
        aux_info2.context_execute_time_with_cache_us = 80
        output_pb2.logits.data_type = TensorPB.DataType.FP32
        output_pb2.logits.shape.extend([1, 1, 2])
        output_pb2.logits.fp32_data = struct.pack("<ff", 0.1, 0.2)
        output_pb2.finished.extend([False])

        # 3. 最终响应：标记结束，并携带最后一个状态
        outputs_pb3 = GenerateOutputsPB()
        output_pb3_item = outputs_pb3.flatten_output
        output_pb3_item.CopyFrom(output_pb2)
        output_pb3_item.finished[0] = True
        return _FakeResponseIterator([outputs_pb1, outputs_pb2, outputs_pb3])


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
        self.active_leases = 0

    @asynccontextmanager
    async def acquire(self, target_address, *, timeout=None):
        self.targets.append(target_address)
        self.active_leases += 1
        try:
            yield object()
        finally:
            self.active_leases -= 1


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


def _make_response(finished=True, output_len=None, output_ids=None):
    outputs_pb = GenerateOutputsPB()
    outputs_pb.flatten_output.finished.extend([finished])
    if output_len is not None:
        outputs_pb.flatten_output.aux_info.add().output_len = output_len
    if output_ids is not None:
        outputs_pb.flatten_output.output_ids.CopyFrom(trans_from_tensor(output_ids))
    return outputs_pb


def _prefill_role_addr(ip="prefill", grpc_port=9000):
    return RoleAddr(role=RoleType.PREFILL, ip=ip, http_port=8000, grpc_port=grpc_port)


def _decode_role_addr(ip="decode", grpc_port=9001):
    return RoleAddr(role=RoleType.DECODE, ip=ip, http_port=8001, grpc_port=grpc_port)


class ModelRpcClientTest(TestCase):

    @staticmethod
    def _make_generate_input(generate_config: GenerateConfig) -> GenerateInput:
        return GenerateInput(
            request_id=1,
            token_ids=torch.tensor([1], dtype=torch.int32),
            mm_inputs=[],
            generate_config=generate_config,
        )

    @staticmethod
    def _make_client(max_rpc_timeout_ms: int = 0) -> ModelRpcClient:
        channel_pool = _FakeChannelPool()
        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.GrpcHostChannelPool",
            return_value=channel_pool,
        ):
            return ModelRpcClient(
                addresses=["localhost:1234"],
                client_config={},
                max_rpc_timeout_ms=max_rpc_timeout_ms,
                decode_entrance=False,
            )

    def _round_trip_generate_config(self, config: GenerateConfig):
        input_pb = trans_input(self._make_generate_input(config))
        round_tripped_input_pb = GenerateInputPB()
        round_tripped_input_pb.ParseFromString(input_pb.SerializeToString())
        return round_tripped_input_pb.generate_config

    def test_capture_round_trips_independently_of_reuse_and_generation(self):
        self.assertFalse(
            self._round_trip_generate_config(GenerateConfig()).capture_hidden_states
        )
        for tokens in (0, 4):
            for reuse in (False, True):
                for capture in (False, True):
                    with self.subTest(tokens=tokens, reuse=reuse, capture=capture):
                        config = GenerateConfig(
                            max_new_tokens=tokens,
                            reuse_cache=reuse,
                            capture_hidden_states=capture,
                        )
                        wire = self._round_trip_generate_config(config)
                        self.assertEqual(wire.capture_hidden_states, capture)
                        self.assertEqual(wire.reuse_cache, reuse)
                        self.assertEqual(wire.max_new_tokens, tokens)
                        self.assertEqual(wire.prefill_only, tokens == 0)

    def test_trans_input_serializes_resolved_positive_default_max_new_tokens(self):
        config = GenerateConfig()

        self.assertGreater(config.max_new_tokens, 0)
        config_pb = self._round_trip_generate_config(config)

        self.assertEqual(config_pb.max_new_tokens, config.max_new_tokens)
        self.assertFalse(config_pb.prefill_only)
        self.assertIn(
            "max_new_tokens", {field.name for field, _ in config_pb.ListFields()}
        )

    def test_trans_input_round_trips_zero_max_new_tokens_with_prefill_flag(self):
        config_pb = self._round_trip_generate_config(GenerateConfig(max_new_tokens=0))

        self.assertEqual(config_pb.max_new_tokens, 0)
        self.assertTrue(config_pb.prefill_only)
        fields = {field.name for field, _ in config_pb.ListFields()}
        self.assertNotIn("max_new_tokens", fields)
        self.assertIn("prefill_only", fields)

    def test_prefill_only_aux_info_is_a_local_output_option(self):
        for aux_info in (False, True):
            with self.subTest(aux_info=aux_info):
                input_py = self._make_generate_input(
                    GenerateConfig(max_new_tokens=0, aux_info=aux_info)
                )
                request = trans_input(input_py)
                self.assertTrue(request.generate_config.prefill_only)
                self.assertNotIn(
                    "aux_info", request.generate_config.DESCRIPTOR.fields_by_name
                )
                response = _make_response(output_len=0)
                outputs = trans_output(input_py, response, StreamState())
                self.assertEqual(
                    outputs.generate_outputs[0].aux_info is not None, aux_info
                )

    def test_prefill_only_uses_existing_stream_rpc_and_releases_channel(self):
        client = self._make_client()
        stub = _RoutingStub(generate_responses=[_make_response(output_len=0)])
        input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))
        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub", return_value=stub
        ):
            outputs = asyncio.run(self._run(client, input_py))
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].finished)
        self.assertEqual(outputs[0].aux_info.output_len, 0)
        self.assertEqual(len(stub.generate_calls), 1)
        request, _ = stub.generate_calls[0]
        self.assertTrue(request.generate_config.prefill_only)
        self.assertEqual(request.generate_config.max_new_tokens, 0)
        self.assertEqual(stub.fetch_calls, [])
        self.assertFalse(stub.generate_iterator.cancelled)
        self.assertEqual(client._channel_pool.active_leases, 0)

    def test_prefill_only_uses_existing_fetch_rpc_for_master_enqueued_request(self):
        client = self._make_client()
        stub = _RoutingStub(fetch_responses=[_make_response(output_len=0)])
        input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))
        input_py.enqueued_by_master = True
        input_py.generate_config.role_addrs = [_prefill_role_addr()]
        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub", return_value=stub
        ):
            outputs = asyncio.run(self._run(client, input_py))
        self.assertEqual(outputs[0].aux_info.output_len, 0)
        self.assertEqual(len(stub.fetch_calls), 1)
        self.assertEqual(stub.fetch_calls[0][0].request_id, input_py.request_id)
        self.assertEqual(stub.generate_calls, [])
        self.assertEqual(client._channel_pool.targets, ["prefill:9000"])
        self.assertEqual(client._channel_pool.active_leases, 0)

    def test_invalid_prefill_config_does_not_acquire_channel_or_send_rpc(self):
        for field in ("return_logits", "return_logprobs"):
            with self.subTest(field=field):
                client = self._make_client()
                input_py = self._make_generate_input(GenerateConfig(max_new_tokens=1))
                input_py.generate_config.max_new_tokens = 0
                setattr(input_py.generate_config, field, True)
                with (
                    patch(
                        "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub"
                    ) as stub,
                    self.assertRaisesRegex(FtRuntimeException, field) as ctx,
                ):
                    asyncio.run(self._run(client, input_py))
                self.assertEqual(
                    ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
                )
                stub.assert_not_called()
                self.assertEqual(client._channel_pool.targets, [])
                self.assertEqual(client._channel_pool.active_leases, 0)

    def test_prefill_stream_failure_and_cancellation_release_channel(self):
        class FailingIterator(_FakeResponseIterator):
            def __init__(self, error):
                super().__init__([])
                self.error = error

            async def __anext__(self):
                raise self.error

        for error in (ValueError("invalid response"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                client = self._make_client()
                stub = _RoutingStub()
                stub.generate_iterator = FailingIterator(error)
                input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))
                with (
                    patch(
                        "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                        return_value=stub,
                    ),
                    self.assertRaises(type(error)),
                ):
                    asyncio.run(self._run(client, input_py))
                self.assertTrue(stub.generate_iterator.cancelled)
                self.assertEqual(client._channel_pool.active_leases, 0)

    def test_prefill_channel_capacity_timeout_is_typed(self):
        class TimeoutPool:
            @asynccontextmanager
            async def acquire(self, target_address, *, timeout=None):
                raise asyncio.TimeoutError()
                yield

        client = self._make_client()
        client._channel_pool = TimeoutPool()
        input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))
        with (
            patch("rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub") as stub,
            self.assertRaises(FtRuntimeException) as ctx,
        ):
            asyncio.run(self._run(client, input_py))
        self.assertEqual(ctx.exception.exception_type, ExceptionType.GENERATE_TIMEOUT)
        stub.assert_not_called()

    def test_trans_input_round_trips_explicit_positive_max_new_tokens(self):
        config_pb = self._round_trip_generate_config(GenerateConfig(max_new_tokens=7))

        self.assertEqual(config_pb.max_new_tokens, 7)
        self.assertFalse(config_pb.prefill_only)
        self.assertIn(
            "max_new_tokens", {field.name for field, _ in config_pb.ListFields()}
        )

    def test_transport_error_mapping_handles_absent_and_unknown_metadata(self):
        import grpc

        client = self._make_client()
        for status, details, metadata, expected in (
            (
                grpc.StatusCode.CANCELLED,
                "cancelled",
                None,
                ExceptionType.CANCELLED_ERROR,
            ),
            (
                grpc.StatusCode.DEADLINE_EXCEEDED,
                "deadline",
                (),
                ExceptionType.GENERATE_TIMEOUT,
            ),
            (
                grpc.StatusCode.UNAVAILABLE,
                "socket closed",
                (),
                ExceptionType.CONNECTION_RESET_BY_PEER,
            ),
            (
                grpc.StatusCode.UNAVAILABLE,
                "timed out",
                (),
                ExceptionType.CONNECT_TIMEOUT,
            ),
            (
                grpc.StatusCode.UNAVAILABLE,
                "unavailable",
                (),
                ExceptionType.CONNECT_FAILED,
            ),
            (
                grpc.StatusCode.INTERNAL,
                "typed",
                (
                    (
                        "grpc-status-details-bin",
                        ErrorDetailsPB(
                            error_code=int(ExceptionType.INVALID_PARAMS),
                            error_message="bad capture config",
                        ).SerializeToString(),
                    ),
                ),
                ExceptionType.INVALID_PARAMS,
            ),
            (
                grpc.StatusCode.INTERNAL,
                "unknown",
                (
                    (
                        "grpc-status-details-bin",
                        ErrorDetailsPB(
                            error_code=2000000000,
                            error_message="unknown backend code",
                        ).SerializeToString(),
                    ),
                ),
                ExceptionType.UNKNOWN_ERROR,
            ),
        ):
            with self.subTest(status=status, details=details):
                error = MagicMock(spec=["code", "details", "trailing_metadata"])
                error.code.return_value = status
                error.details.return_value = details
                error.trailing_metadata.return_value = metadata
                with self.assertRaises(FtRuntimeException) as ctx:
                    client._handle_grpc_error(
                        error, "test request", "private-peer:1234"
                    )
                self.assertEqual(ctx.exception.exception_type, expected)
                self.assertNotIn("private-peer", ctx.exception.message)

    def test_trans_tensor_preserves_scalar_and_zero_numel_shapes(self):
        cases = (
            ("scalar", [], struct.pack("<i", 7), (), 7),
            ("empty_vector", [0], b"", (0,), []),
            ("empty_matrix", [0, 4], b"", (0, 4), []),
            ("prefill_only_output", [1, 1, 0], b"", (1, 1, 0), [[[]]]),
        )

        for name, shape, payload, expected_shape, expected_value in cases:
            with self.subTest(name=name):
                tensor_pb = TensorPB(
                    data_type=TensorPB.DataType.INT32, int32_data=payload
                )
                tensor_pb.shape.extend(shape)

                tensor = trans_tensor(tensor_pb)

                self.assertEqual(tensor.dtype, torch.int32)
                self.assertEqual(tuple(tensor.shape), expected_shape)
                self.assertEqual(tensor.tolist(), expected_value)

    def test_trans_tensor_preserves_default_empty_sentinel(self):
        tensor = trans_tensor(TensorPB())

        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(tensor.shape), (0,))

    def test_trans_tensor_rejects_invalid_shape_and_payload(self):
        cases = (
            ("negative_dimension", [1, -1], b"", "non-negative"),
            ("short_payload", [2], struct.pack("<i", 1), "expected 8 bytes, got 4"),
            (
                "long_scalar_payload",
                [],
                struct.pack("<ii", 1, 2),
                "expected 4 bytes, got 8",
            ),
        )

        for name, shape, payload, expected_message in cases:
            with self.subTest(name=name):
                tensor_pb = TensorPB(
                    data_type=TensorPB.DataType.INT32, int32_data=payload
                )
                tensor_pb.shape.extend(shape)

                with self.assertRaisesRegex(ValueError, expected_message):
                    trans_tensor(tensor_pb)

    def test_trans_output_accepts_prefill_only_empty_output_ids(self):
        input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))
        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.append(True)
        output_pb.aux_info.add().output_len = 0
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([1, 1, 0])

        outputs = trans_output(input_py, outputs_pb, StreamState())

        self.assertEqual(len(outputs.generate_outputs), 1)
        output_ids = outputs.generate_outputs[0].output_ids
        self.assertEqual(output_ids.dtype, torch.int32)
        self.assertEqual(tuple(output_ids.shape), (1, 0))
        self.assertEqual(output_ids.numel(), 0)

    def test_prefill_only_rejects_actual_tokens_independently_of_aux_info(self):
        for use_fetch in (False, True):
            for aux_info in (False, True):
                for output_len in (None, 0):
                    for finished in (None, False, True):
                        for shape in ((), (1, 1, 1), (1, 1, 2)):
                            with self.subTest(
                                use_fetch=use_fetch,
                                aux_info=aux_info,
                                output_len=output_len,
                                finished=finished,
                                shape=shape,
                            ):
                                tokens = torch.full(shape, 123456789, dtype=torch.int32)
                                response = _make_response(
                                    finished=finished is True,
                                    output_len=output_len,
                                    output_ids=tokens,
                                )
                                if finished is None:
                                    response.flatten_output.ClearField("finished")
                                response = GenerateOutputsPB.FromString(
                                    response.SerializeToString()
                                )
                                client = self._make_client()
                                input_py = self._make_generate_input(
                                    GenerateConfig(max_new_tokens=0, aux_info=aux_info)
                                )
                                input_py.enqueued_by_master = use_fetch
                                input_py.generate_config.role_addrs = [
                                    _prefill_role_addr()
                                ]
                                stub = _RoutingStub(
                                    fetch_responses=[response] if use_fetch else [],
                                    generate_responses=[] if use_fetch else [response],
                                )
                                observed_outputs = []

                                async def consume():
                                    async for output in client.enqueue(input_py):
                                        observed_outputs.extend(output.generate_outputs)

                                with (
                                    patch(
                                        "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                                        return_value=stub,
                                    ),
                                    self.assertLogs(
                                        "rtp_llm.cpp.model_rpc.model_rpc_client",
                                        level="ERROR",
                                    ) as logs,
                                    self.assertRaises(FtRuntimeException) as ctx,
                                ):
                                    asyncio.run(consume())
                                self.assertEqual(
                                    ctx.exception.exception_type,
                                    ExceptionType.EXECUTION_EXCEPTION,
                                )
                                self.assertIn(
                                    f"output_ids contains {tokens.numel()} tokens",
                                    ctx.exception.message,
                                )
                                log_text = "\n".join(logs.output)
                                self.assertIn(
                                    "prefill-only response contract violation", log_text
                                )
                                self.assertIn("request_id=1", log_text)
                                self.assertNotIn("123456789", log_text)
                                self.assertNotIn("123456789", ctx.exception.message)
                                self.assertEqual(observed_outputs, [])
                                iterator = (
                                    stub.fetch_iterator
                                    if use_fetch
                                    else stub.generate_iterator
                                )
                                self.assertTrue(iterator.cancelled)
                                self.assertEqual(client._channel_pool.active_leases, 0)
                                self.assertEqual(len(stub.fetch_calls), int(use_fetch))
                                self.assertEqual(
                                    len(stub.generate_calls), int(not use_fetch)
                                )

    def test_prefill_only_accepts_empty_tensors_and_intermediate_frames(self):
        for use_fetch in (False, True):
            for aux_info in (False, True):
                for shape in (None, (), (0,), (0, 4), (1, 1, 0)):
                    with self.subTest(
                        use_fetch=use_fetch, aux_info=aux_info, shape=shape
                    ):
                        terminal = _make_response(output_len=0 if aux_info else None)
                        if shape == ():
                            terminal.flatten_output.output_ids.CopyFrom(TensorPB())
                        elif shape is not None:
                            terminal.flatten_output.output_ids.CopyFrom(
                                trans_from_tensor(torch.empty(shape, dtype=torch.int32))
                            )
                        responses = [
                            GenerateOutputsPB(),
                            _make_response(finished=False),
                            terminal,
                        ]
                        responses = [
                            GenerateOutputsPB.FromString(response.SerializeToString())
                            for response in responses
                        ]
                        client = self._make_client()
                        input_py = self._make_generate_input(
                            GenerateConfig(max_new_tokens=0, aux_info=aux_info)
                        )
                        input_py.enqueued_by_master = use_fetch
                        input_py.generate_config.role_addrs = [_prefill_role_addr()]
                        stub = _RoutingStub(
                            fetch_responses=responses if use_fetch else [],
                            generate_responses=[] if use_fetch else responses,
                        )
                        with patch(
                            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                            return_value=stub,
                        ):
                            outputs = asyncio.run(self._run(client, input_py))
                        self.assertEqual(
                            [output.finished for output in outputs], [False, True]
                        )
                        if shape == (1, 1, 0):
                            self.assertEqual(
                                tuple(outputs[-1].output_ids.shape), (1, 0)
                            )
                            self.assertEqual(outputs[-1].output_ids.numel(), 0)
                        self.assertEqual(outputs[-1].aux_info is not None, aux_info)
                        iterator = (
                            stub.fetch_iterator if use_fetch else stub.generate_iterator
                        )
                        self.assertFalse(iterator.cancelled)
                        self.assertEqual(client._channel_pool.active_leases, 0)
                        self.assertEqual(len(stub.fetch_calls), int(use_fetch))
                        self.assertEqual(len(stub.generate_calls), int(not use_fetch))

    def test_prefill_only_rejects_malformed_output_ids_before_empty_frame_return(self):
        for shape, payload in (
            ([1, -1], b""),
            ([1, 1, 0], struct.pack("<i", 7)),
            ([1, 1, 2], struct.pack("<i", 7)),
            ([], struct.pack("<ii", 7, 8)),
        ):
            with self.subTest(shape=shape):
                response = GenerateOutputsPB()
                response.flatten_output.output_ids.CopyFrom(
                    TensorPB(
                        data_type=TensorPB.DataType.INT32,
                        shape=shape,
                        int32_data=payload,
                    )
                )
                response = GenerateOutputsPB.FromString(response.SerializeToString())
                input_py = self._make_generate_input(
                    GenerateConfig(max_new_tokens=0, aux_info=False)
                )
                with self.assertRaisesRegex(
                    FtRuntimeException, "invalid output_ids"
                ) as ctx:
                    trans_output(input_py, response, StreamState())
                self.assertEqual(
                    ctx.exception.exception_type, ExceptionType.EXECUTION_EXCEPTION
                )

    def test_positive_generation_keeps_nonempty_output_ids(self):
        tokens = torch.tensor([[[7, 8]]], dtype=torch.int32)
        response = _make_response(output_ids=tokens)
        response = GenerateOutputsPB.FromString(response.SerializeToString())
        input_py = self._make_generate_input(
            GenerateConfig(max_new_tokens=2, aux_info=False)
        )
        outputs = trans_output(input_py, response, StreamState())
        self.assertTrue(torch.equal(outputs.generate_outputs[0].output_ids, tokens[0]))
        self.assertTrue(outputs.generate_outputs[0].finished)

    def test_prefill_only_fails_fast_when_backend_reports_generated_tokens(self):
        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=False,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(generate_responses=[_make_response(output_len=1)])
        input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))

        with (
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                return_value=stub,
            ),
            self.assertLogs(
                "rtp_llm.cpp.model_rpc.model_rpc_client", level="ERROR"
            ) as logs,
            self.assertRaises(FtRuntimeException) as context,
        ):
            asyncio.run(self._run(client, input_py))

        self.assertEqual(
            context.exception.exception_type, ExceptionType.EXECUTION_EXCEPTION
        )
        self.assertIn("output_len=1", context.exception.message)
        self.assertIn("may not support", context.exception.message)
        self.assertIn(
            "prefill-only response contract violation",
            "\n".join(logs.output),
        )
        log_text = "\n".join(logs.output)
        self.assertIn("output_len=1", log_text)
        self.assertIn("request_id=1", log_text)
        self.assertIn("aux_info[0]", log_text)
        self.assertTrue(stub.generate_iterator.cancelled)

    def test_prefill_only_does_not_reject_missing_first_packet_aux_info(self):
        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=False,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(
            generate_responses=[
                _make_response(finished=False),
                _make_response(finished=True, output_len=0),
            ]
        )
        input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            responses = asyncio.run(self._run(client, input_py))

        self.assertEqual(len(responses), 2)
        self.assertFalse(stub.generate_iterator.cancelled)

    def test_prefill_only_accepts_missing_stream_aux_info_when_disabled(self):
        client = self._make_client()
        stub = _RoutingStub(
            generate_responses=[
                _make_response(finished=False),
                _make_response(finished=True),
            ]
        )
        input_py = self._make_generate_input(
            GenerateConfig(max_new_tokens=0, aux_info=False)
        )

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            responses = asyncio.run(self._run(client, input_py))

        self.assertEqual(len(responses), 2)
        self.assertEqual([response.finished for response in responses], [False, True])
        self.assertTrue(all(response.aux_info is None for response in responses))
        self.assertFalse(stub.generate_iterator.cancelled)

    def test_prefill_only_fails_when_stream_never_exposes_aux_info(self):
        client = ModelRpcClient(
            addresses=["worker:9000"],
            client_config={},
            max_rpc_timeout_ms=0,
            decode_entrance=False,
        )
        client._channel_pool = _FakeChannelPool()
        stub = _RoutingStub(generate_responses=[_make_response(finished=True)])
        input_py = self._make_generate_input(
            GenerateConfig(max_new_tokens=0, aux_info=True)
        )

        with (
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                return_value=stub,
            ),
            self.assertLogs(
                "rtp_llm.cpp.model_rpc.model_rpc_client", level="ERROR"
            ) as logs,
            self.assertRaises(FtRuntimeException) as context,
        ):
            asyncio.run(self._run(client, input_py))

        self.assertEqual(
            context.exception.exception_type, ExceptionType.EXECUTION_EXCEPTION
        )
        self.assertIn("completed without aux_info", context.exception.message)
        self.assertIn("capability cannot be verified", context.exception.message)
        self.assertIn(
            "prefill-only capability verification failed",
            "\n".join(logs.output),
        )
        log_text = "\n".join(logs.output)
        self.assertIn("completed without aux_info", log_text)
        self.assertIn("request_id=1", log_text)
        self.assertFalse(stub.generate_iterator.cancelled)

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    @staticmethod
    async def _run(client, input):
        responses = []
        async for res in client.enqueue(input):
            responses.extend(res.generate_outputs)
        return responses

    def test_frontend_metric_envelope_stays_out_of_public_aux_info(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            mm_inputs=[],
            generate_config=GenerateConfig(aux_info=True),
        )
        outputs_pb = GenerateOutputsPB()
        outputs_pb.frontend_metric_only = True
        outputs_pb.frontend_context_token_num.value = 11
        outputs_pb.frontend_context_token_num_with_cache.value = 13
        outputs_pb.frontend_context_execute_time_us.value = 101
        outputs_pb.frontend_context_execute_time_with_cache_us.value = 81
        outputs_pb.frontend_generate_token_num.value = 17
        outputs_pb.frontend_generate_execute_time_us.value = 201
        outputs_pb.frontend_sp_tpot_samples.add(sequence_id=3, tpot_us=12345.5)
        outputs_pb.flatten_output.finished.append(False)
        outputs_pb.flatten_output.aux_info.add().output_len = 4

        output = trans_output(input_py, outputs_pb, StreamState())

        self.assertTrue(output.frontend_metric_only)
        self.assertEqual(output.frontend_context_token_num, 11)
        self.assertEqual(output.frontend_context_token_num_with_cache, 13)
        self.assertEqual(output.frontend_context_execute_time_us, 101)
        self.assertEqual(output.frontend_context_execute_time_with_cache_us, 81)
        self.assertEqual(output.frontend_generate_token_num, 17)
        self.assertEqual(output.frontend_generate_execute_time_us, 201)
        self.assertEqual(output.frontend_sp_tpot_samples, [(3, 12345.5)])
        self.assertNotIn(
            "frontend_sp_tpot_samples", asdict(output.generate_outputs[0].aux_info)
        )
        self.assertNotIn(
            "frontend_metric_only",
            asdict(output.generate_outputs[0].aux_info),
        )
        self.assertNotIn(
            "frontend_generate_token_num",
            asdict(output.generate_outputs[0].aux_info),
        )
        self.assertNotIn(
            "frontend_context_token_num",
            asdict(output.generate_outputs[0].aux_info),
        )
        self.assertNotIn(
            "frontend_generate_execute_time_us",
            asdict(output.generate_outputs[0].aux_info),
        )

    @unittest.skip("need fix")
    def test_generate_stream(self):
        client = self._make_client()
        generate_config: GenerateConfig = GenerateConfig(using_hf_sampling=False)
        input = GenerateInput(
            token_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            generate_config=generate_config,
        )
        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=FakeStub(),
        ):
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
        self.assertEqual(res[1].aux_info.speculative_verify_rounds, 3)
        self.assertEqual(res[1].aux_info.speculative_accepted_token_num, 9)
        self.assertEqual(res[1].aux_info.speculative_proposed_draft_tokens, 12)
        self.assertEqual(res[1].aux_info.context_execute_time_us, 100)
        self.assertEqual(res[1].aux_info.generate_execute_time_us, 200)
        self.assertEqual(res[1].aux_info.context_execute_time_with_cache_us, 80)

        self.assertEqual(res[2].finished, True)

    def test_generate_stream_with_logits_index(self):
        client = self._make_client()
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
        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=FakeStub(),
        ):
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

    def test_trans_input_priority_uses_shared_qos_resolution(self):
        cases = (
            ({"x-dashscope-inner-qos-level": "70"}, None, 70),
            ({"X-DashScope-Inner-QoS-Level": "60"}, None, 60),
            ({}, 77, 77),
            ({"x-dashscope-inner-qos-level": "invalid"}, 40, 40),
            ({"x-dashscope-inner-qos-level": "101"}, None, 50),
            ({}, None, 50),
        )

        for headers, config_priority, expected in cases:
            with self.subTest(
                headers=headers,
                config_priority=config_priority,
                expected=expected,
            ):
                input_pb = trans_input(
                    GenerateInput(
                        token_ids=torch.tensor([1, 2, 3]),
                        generate_config=GenerateConfig(qos_priority=config_priority),
                        request_id=123,
                        mm_inputs=[],
                        headers=headers,
                    )
                )
                self.assertEqual(input_pb.priority, expected)

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
        self.assertEqual(client._channel_pool.active_leases, 0)
        self.assertEqual(len(stub.fetch_calls), 1)
        self.assertEqual(stub.fetch_calls[0][0].request_id, 321)
        self.assertGreater(stub.fetch_calls[0][1]["timeout"], 0)
        self.assertLessEqual(stub.fetch_calls[0][1]["timeout"], 1.0)
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
            headers={"x-dashscope-inner-qos-level": "70"},
        )

        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        ):
            responses = asyncio.run(self._run(client, input_py))

        self.assertEqual(len(responses), 1)
        self.assertEqual(len(stub.generate_calls), 1)
        self.assertEqual(stub.generate_calls[0][0].request_id, 322)
        self.assertEqual(stub.generate_calls[0][0].priority, 70)
        self.assertEqual(stub.fetch_calls, [])

    def test_enqueue_cancels_fetch_stream_on_early_close(self):
        async def run_and_close():
            gen = client.enqueue(input_py)
            first = await gen.__anext__()
            self.assertEqual(client._channel_pool.active_leases, 1)
            await gen.aclose()
            self.assertEqual(client._channel_pool.active_leases, 0)
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


if __name__ == "__main__":
    setup_logging()
    main()
