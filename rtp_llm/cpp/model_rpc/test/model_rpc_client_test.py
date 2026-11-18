import asyncio
import inspect
import logging
import os
import struct
from contextlib import asynccontextmanager
from dataclasses import asdict
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import grpc
import torch
from rtp_llm.config.exceptions import (
    ExceptionCategory,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.config.log_config import setup_logging
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    ModelRpcClient,
    StreamState,
    _trans_rpc_error_code,
    trans_input,
    trans_output,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    BatchGenerateOutputsPB,
    ErrorCodePB,
    ErrorDetailsPB,
    GenerateInputPB,
    GenerateOutputsPB,
    RoleAddrPB,
    TensorPB,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
    RequestInfo,
)
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

    def test_trans_input_serializes_positive_max_new_tokens(self):
        for name, config, expected in (
            ("default", GenerateConfig(), None),
            ("explicit", GenerateConfig(max_new_tokens=7), 7),
        ):
            with self.subTest(name=name):
                if expected is None:
                    self.assertGreater(config.max_new_tokens, 0)
                    expected = config.max_new_tokens
                config_pb = self._round_trip_generate_config(config)

                self.assertEqual(config_pb.max_new_tokens, expected)
                self.assertFalse(config_pb.prefill_only)
                self.assertIn(
                    "max_new_tokens",
                    {field.name for field, _ in config_pb.ListFields()},
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

    def test_prefill_only_uses_correct_rpc_and_releases_channel(self):
        for enqueued_by_master in (False, True):
            with self.subTest(enqueued_by_master=enqueued_by_master):
                client = self._make_client()
                stub = _RoutingStub(
                    fetch_responses=(
                        [_make_response(output_len=0)] if enqueued_by_master else []
                    ),
                    generate_responses=(
                        [] if enqueued_by_master else [_make_response(output_len=0)]
                    ),
                )
                input_py = self._make_generate_input(GenerateConfig(max_new_tokens=0))
                input_py.enqueued_by_master = enqueued_by_master
                if enqueued_by_master:
                    input_py.generate_config.role_addrs = [_prefill_role_addr()]

                with patch(
                    "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                    return_value=stub,
                ):
                    outputs = asyncio.run(self._run(client, input_py))

                self.assertEqual(len(outputs), 1)
                self.assertEqual(outputs[0].aux_info.output_len, 0)
                self.assertEqual(len(stub.fetch_calls), int(enqueued_by_master))
                self.assertEqual(len(stub.generate_calls), int(not enqueued_by_master))
                iterator = (
                    stub.fetch_iterator
                    if enqueued_by_master
                    else stub.generate_iterator
                )
                self.assertFalse(iterator.cancelled)
                self.assertEqual(client._channel_pool.active_leases, 0)
                if enqueued_by_master:
                    self.assertEqual(
                        stub.fetch_calls[0][0].request_id, input_py.request_id
                    )
                    self.assertEqual(client._channel_pool.targets, ["prefill:9000"])
                else:
                    self.assertTrue(outputs[0].finished)
                    request, _ = stub.generate_calls[0]
                    self.assertTrue(request.generate_config.prefill_only)
                    self.assertEqual(request.generate_config.max_new_tokens, 0)

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

    def test_prefill_only_rejects_actual_tokens_independently_of_metadata(self):
        cases = (
            ("stream_missing_finished", False, False, None, None, ()),
            ("stream_with_aux_info", False, True, 0, False, (1, 1, 1)),
            ("fetch_finished", True, False, None, True, (1, 1, 2)),
            ("fetch_with_aux_info", True, True, 0, None, (1, 1, 1)),
        )
        for name, use_fetch, aux_info, output_len, finished, shape in cases:
            with self.subTest(name=name):
                tokens = torch.full(shape, 123456789, dtype=torch.int32)
                response = _make_response(
                    finished=finished is True,
                    output_len=output_len,
                    output_ids=tokens,
                )
                if finished is None:
                    response.flatten_output.ClearField("finished")
                response = GenerateOutputsPB.FromString(response.SerializeToString())
                client = self._make_client()
                input_py = self._make_generate_input(
                    GenerateConfig(max_new_tokens=0, aux_info=aux_info)
                )
                input_py.enqueued_by_master = use_fetch
                input_py.generate_config.role_addrs = [_prefill_role_addr()]
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
                        "rtp_llm.cpp.model_rpc.model_rpc_client", level="ERROR"
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
                self.assertIn("prefill-only response contract violation", log_text)
                self.assertIn("request_id=1", log_text)
                self.assertNotIn("123456789", log_text)
                self.assertNotIn("123456789", ctx.exception.message)
                self.assertEqual(observed_outputs, [])
                iterator = stub.fetch_iterator if use_fetch else stub.generate_iterator
                self.assertTrue(iterator.cancelled)
                self.assertEqual(client._channel_pool.active_leases, 0)

    def test_prefill_only_accepts_empty_tensors_and_intermediate_frames(self):
        for use_fetch in (False, True):
            for aux_info in (False, True):
                for shape in (None, (), (0,), (1, 1, 0)):
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
        public_aux_info = asdict(output.generate_outputs[0].aux_info)
        for field in (
            "frontend_sp_tpot_samples",
            "frontend_metric_only",
            "frontend_generate_token_num",
            "frontend_context_token_num",
            "frontend_generate_execute_time_us",
        ):
            self.assertNotIn(field, public_aux_info)

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
        logits_0 = res[0].logits.tolist()
        self.assertAlmostEqual(logits_0[0][0], 0.0, places=6)
        self.assertAlmostEqual(logits_0[0][1], 0.0, places=6)

        # res[1] 是第二个token
        logits_1 = res[1].logits.tolist()
        self.assertAlmostEqual(logits_1[0][0], 0.1, places=6)
        self.assertAlmostEqual(logits_1[0][1], 0.2, places=6)

        # res[2] 是完成标记，包含指定位置token的logits
        self.assertTrue(res[2].finished)
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


class ModelRpcClientBatchGenerateTest(TestCase):
    _make_client = staticmethod(ModelRpcClientTest._make_client)

    @staticmethod
    def _inputs(count, **config):
        values = dict(max_new_tokens=0, capture_hidden_states=True, reuse_cache=False)
        values.update(config)
        return [
            GenerateInput(
                request_id=index,
                token_ids=torch.tensor([index + 1], dtype=torch.int32),
                mm_inputs=[],
                generate_config=GenerateConfig(**values),
            )
            for index in range(count)
        ]

    @staticmethod
    def _response(codes):
        response = BatchGenerateOutputsPB()
        for index, code in enumerate(codes):
            item = response.results.add()
            if code is None:
                item.final_output.CopyFrom(_make_response(output_len=0))
                item.final_output.request_id = index
            else:
                item.error_info.error_code = code
        return response

    def _run_batch(
        self, client, inputs, response, *, error=None, method="batch_generate", **kwargs
    ):
        class Call:
            cancelled = False

            def cancel(self):
                self.cancelled = True

            def __await__(self):
                async def receive():
                    if error is not None:
                        raise error
                    return (
                        BatchGenerateOutputsPB.FromString(response.SerializeToString())
                        if response is not None else None
                    )

                return receive().__await__()

        call = Call()
        stub = MagicMock()
        stub.BatchGenerateCall.return_value = call
        self.last_stub = stub
        self.last_call = call
        with (
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                return_value=stub,
            ),
            patch.object(
                client,
                "enqueue",
                side_effect=AssertionError("batch must not enqueue per item"),
            ),
        ):
            try:
                return asyncio.run(getattr(client, method)(inputs, **kwargs))
            finally:
                self.assertEqual(client._channel_pool.active_leases, 0)
                stub.GenerateStreamCall.assert_not_called()
                stub.FetchResponse.assert_not_called()

    def test_empty_batch_is_noop_and_alias_keeps_default(self):
        client = self._make_client()
        self.assertTrue(inspect.iscoroutinefunction(ModelRpcClient.batch_generate))
        for method in ("batch_generate", "batch_enqueue"):
            self.assertEqual(self._run_batch(client, [], None, method=method), [])
            self.last_stub.BatchGenerateCall.assert_not_called()
        self.assertEqual(client._channel_pool.targets, [])

    def test_native_batch_success_preserves_order_and_caller_configs(self):
        client = self._make_client()
        inputs = self._inputs(3)
        before = [item.generate_config.model_dump() for item in inputs]
        results = self._run_batch(client, inputs, self._response([None] * 3))
        self.assertEqual(
            [r.generate_outputs[0].input_ids.item() for r in results], [1, 2, 3]
        )
        self.assertTrue(all(r.generate_outputs[0].finished for r in results))
        self.assertEqual(before, [item.generate_config.model_dump() for item in inputs])
        self.last_stub.BatchGenerateCall.assert_called_once()
        wire = self.last_stub.BatchGenerateCall.call_args.args[0]
        self.assertEqual([item.request_id for item in wire.inputs], [0, 1, 2])
        self.assertEqual(client._channel_pool.targets, ["localhost:1234"])
        self.assertFalse(self.last_call.cancelled)

    def test_multistep_batch_forces_non_streaming_without_mutating_inputs(self):
        for method in ("batch_generate", "batch_enqueue"):
            for streaming in (False, True):
                with self.subTest(method=method, streaming=streaming):
                    client = self._make_client()
                    inputs = self._inputs(
                        2,
                        max_new_tokens=3,
                        capture_hidden_states=False,
                        is_streaming=streaming,
                    )
                    original_configs = [item.generate_config for item in inputs]
                    before = [config.model_dump() for config in original_configs]
                    tokens = [[7, 8, 9], [10, 11, 12]]
                    response = BatchGenerateOutputsPB()
                    for row in tokens:
                        response.results.add().final_output.CopyFrom(
                            _make_response(
                                output_len=3,
                                output_ids=torch.tensor([row], dtype=torch.int32),
                            )
                        )
                    results = self._run_batch(client, inputs, response, method=method)
                    self.last_stub.BatchGenerateCall.assert_called_once()
                    wire = self.last_stub.BatchGenerateCall.call_args.args[0]
                    self.assertTrue(
                        all(not item.generate_config.is_streaming for item in wire.inputs)
                    )
                    self.assertEqual(
                        [item.generate_outputs[0].output_ids.tolist() for item in results],
                        tokens,
                    )
                    self.assertTrue(
                        all(item.generate_outputs[0].finished for item in results)
                    )
                    self.assertEqual(
                        before, [item.generate_config.model_dump() for item in inputs]
                    )
                    for item, original in zip(inputs, original_configs):
                        self.assertIs(item.generate_config, original)

    def test_capture_backpressure_preserves_exact_positions_and_later_successes(self):
        busy = ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE
        for codes in ([busy, None, busy, None], [None, busy, None, busy], [busy] * 4):
            with self.subTest(codes=codes):
                response = self._response(codes)
                results = self._run_batch(
                    self._make_client(),
                    self._inputs(4),
                    response,
                    preserve_capture_backpressure=True,
                )
                for index, (code, result) in enumerate(zip(codes, results)):
                    if code is None:
                        self.assertIsInstance(result, GenerateOutputs)
                        self.assertEqual(
                            result.generate_outputs[0].input_ids.item(), index + 1
                        )
                    else:
                        self.assertIsInstance(result, FtRuntimeException)
                        self.assertEqual(
                            result.exception_type,
                            ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE,
                        )
                        self.assertIn("HIDDEN_STATE_CAPTURE_BACKPRESSURE", str(result))

    def test_default_and_alias_raise_on_backpressure_without_partial_results(self):
        for method in ("batch_generate", "batch_enqueue"):
            with (
                self.subTest(method=method),
                self.assertRaises(FtRuntimeException) as caught,
            ):
                self._run_batch(
                    self._make_client(),
                    self._inputs(3),
                    self._response(
                        [None, ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE, None]
                    ),
                    method=method,
                )
            self.assertEqual(
                caught.exception.exception_type,
                ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE,
            )

    def test_other_item_errors_are_atomic_even_after_preserved_backpressure(self):
        for code, expected in (
            (ErrorCodePB.INVALID_PARAMS, ExceptionType.INVALID_PARAMS),
            (ErrorCodePB.EXECUTION_EXCEPTION, ExceptionType.EXECUTION_EXCEPTION),
            (ErrorCodePB.CANCELLED, ExceptionType.CANCELLED_ERROR),
            (8107, ExceptionType.UNKNOWN_ERROR),
            (99999, ExceptionType.UNKNOWN_ERROR),
            (ErrorCodePB.NONE_ERROR, ExceptionType.UNKNOWN_ERROR),
        ):
            response = self._response(
                [ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE, code, None]
            )
            response.results[1].error_info.error_message = (
                "HIDDEN_STATE_CAPTURE_BACKPRESSURE busy 8107"
            )
            with (
                self.subTest(code=code),
                self.assertRaises(FtRuntimeException) as caught,
            ):
                self._run_batch(
                    self._make_client(),
                    self._inputs(3),
                    response,
                    preserve_capture_backpressure=True,
                )
            self.assertEqual(caught.exception.exception_type, expected)

    def test_native_item_codes_match_transport_details_and_preserve_reason(self):
        for raw_code, expected in (
            (603, ExceptionType.GENERATE_TIMEOUT),
            (511, ExceptionType.LONG_PROMPT_ERROR),
            (607, ExceptionType.EXCEEDS_KV_CACHE_MAX_LEN),
            (602, ExceptionType.MALLOC_ERROR),
            (8211, ExceptionType.DECODE_MALLOC_FAILED),
            (8212, ExceptionType.LOAD_KV_CACHE_FAILED),
            (8213, ExceptionType.WAIT_TO_RUN_TIMEOUT),
            (605, ExceptionType.INVALID_PARAMS),
            (606, ExceptionType.EXECUTION_EXCEPTION),
            (8100, ExceptionType.CANCELLED),
            (99999, ExceptionType.UNKNOWN_ERROR),
            (28, ExceptionType.UNKNOWN_ERROR),
        ):
            reason = f"original native failure {raw_code}"
            details = ErrorDetailsPB(error_code=raw_code, error_message=reason)
            transport = MagicMock(spec=["code", "details", "trailing_metadata"])
            transport.code.return_value = grpc.StatusCode.INTERNAL
            transport.details.return_value = "transport failure"
            transport.trailing_metadata.return_value = (
                ("grpc-status-details-bin", details.SerializeToString()),
            )
            with self.assertRaises(FtRuntimeException) as caught:
                self._make_client()._handle_grpc_error(transport, "native roundtrip")
            self.assertEqual(caught.exception.exception_type, expected)
            self.assertEqual(caught.exception.message, reason)
            for preserve, legacy in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(raw_code=raw_code, preserve=preserve, legacy=legacy):
                    inputs = self._inputs(3)
                    inputs[1].request_id = 12345
                    response = self._response([None, ErrorCodePB.UNKNOWN_ERROR, None])
                    response.results[1].error_details.CopyFrom(details)
                    response.results[1].error_info.error_message = "legacy must not win"
                    if not legacy:
                        response.results[1].ClearField("error_info")
                    with self.assertRaises(FtRuntimeException) as caught:
                        self._run_batch(
                            self._make_client(), inputs, response,
                            preserve_capture_backpressure=preserve,
                        )
                    self.assertEqual(caught.exception.exception_type, expected)
                    self.assertIn(reason, caught.exception.message)
                    self.assertIn("batch item 1 request [12345]", caught.exception.message)
                    self.assertNotIn("legacy must not win", caught.exception.message)
                    if expected == ExceptionType.UNKNOWN_ERROR:
                        self.assertIn(f"UNKNOWN({raw_code})", caught.exception.message)
                    self.last_stub.BatchGenerateCall.assert_called_once()

    def test_native_backpressure_preserves_siblings_only_when_opted_in(self):
        busy = ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE
        for legacy in (False, True):
            for codes in ([busy, None, busy, None], [None, busy, None, busy], [busy] * 4):
                with self.subTest(legacy=legacy, codes=codes):
                    response = self._response(codes)
                    for code, result in zip(codes, response.results):
                        if code is not None:
                            result.error_details.error_code = 8107
                            result.error_details.error_message = "capture quota occupied"
                            if not legacy:
                                result.ClearField("error_info")
                    inputs = self._inputs(4)
                    results = self._run_batch(
                        self._make_client(), inputs, response,
                        preserve_capture_backpressure=True,
                    )
                    self.assertEqual(len(results), len(inputs))
                    for index, (code, result) in enumerate(zip(codes, results)):
                        if code is None:
                            self.assertIsInstance(result, GenerateOutputs)
                            self.assertEqual(result.generate_outputs[0].input_ids.item(), index + 1)
                        else:
                            self.assertIsInstance(result, FtRuntimeException)
                            self.assertEqual(result.exception_type, ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE)
                            self.assertIn(f"batch item {index} request [{index}]", result.message)
                    for method in ("batch_generate", "batch_enqueue"):
                        with self.assertRaises(FtRuntimeException) as caught:
                            self._run_batch(self._make_client(), inputs, response, method=method)
                        self.assertEqual(caught.exception.exception_type, ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE)

    def test_conflicting_native_and_legacy_errors_never_preserve_unsafe_backpressure(self):
        busy = ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE
        for raw_code, legacy, final_output, expected in (
            (606, busy, False, ExceptionType.EXECUTION_EXCEPTION),
            (603, busy, False, ExceptionType.GENERATE_TIMEOUT),
            (99999, busy, False, ExceptionType.UNKNOWN_ERROR),
            (0, busy, False, ExceptionType.EXECUTION_EXCEPTION),
            (8107, ErrorCodePB.EXECUTION_EXCEPTION, False, ExceptionType.EXECUTION_EXCEPTION),
            (8107, ErrorCodePB.UNKNOWN_ERROR, False, ExceptionType.EXECUTION_EXCEPTION),
            (8107, ErrorCodePB.NONE_ERROR, False, ExceptionType.EXECUTION_EXCEPTION),
            (8107, busy, True, ExceptionType.EXECUTION_EXCEPTION),
        ):
            for preserve in (False, True):
                with self.subTest(raw_code=raw_code, legacy=legacy, final_output=final_output, preserve=preserve):
                    response = self._response([busy, legacy, None])
                    response.results[0].error_details.error_code = 8107
                    item = response.results[1]
                    item.error_details.error_code = raw_code
                    item.error_details.error_message = "authoritative native reason"
                    if final_output:
                        item.final_output.CopyFrom(_make_response(output_len=0))
                    # Default fails at the first BP. Opt-in must then fail at the conflict.
                    if not preserve:
                        response.results[0].CopyFrom(self._response([None]).results[0])
                    with self.assertRaises(FtRuntimeException) as caught:
                        self._run_batch(
                            self._make_client(), self._inputs(3), response,
                            preserve_capture_backpressure=preserve,
                        )
                    self.assertEqual(caught.exception.exception_type, expected)
                    self.assertIn("batch item 1 request [1]", caught.exception.message)
                    self.assertIn("authoritative native reason", caught.exception.message)

    def test_strict_validation_runs_before_rpc_for_every_item(self):
        for config in (
            {"max_new_tokens": 1},
            {"capture_hidden_states": False},
            {"reuse_cache": True},
        ):
            inputs = self._inputs(2)
            inputs[1] = self._inputs(1, **config)[0]
            before = [item.generate_config.model_dump() for item in inputs]
            client = self._make_client()
            with (
                self.subTest(config=config),
                self.assertRaises(FtRuntimeException) as caught,
            ):
                self._run_batch(
                    client, inputs, None, preserve_capture_backpressure=True
                )
            self.assertEqual(
                caught.exception.exception_type, ExceptionType.INVALID_PARAMS
            )
            self.last_stub.BatchGenerateCall.assert_not_called()
            self.assertEqual(client._channel_pool.targets, [])
            self.assertEqual(
                before, [item.generate_config.model_dump() for item in inputs]
            )

    def test_default_allows_non_capture_inputs(self):
        response = self._response([None])
        [result] = self._run_batch(
            self._make_client(),
            self._inputs(1, max_new_tokens=1, capture_hidden_states=False),
            response,
        )
        self.assertTrue(result.generate_outputs[0].finished)

    def test_cardinality_is_checked_before_preserving_any_item(self):
        for count in (0, 1, 3):
            for preserve in (False, True):
                with (
                    self.subTest(count=count, preserve=preserve),
                    self.assertRaisesRegex(FtRuntimeException, "cardinality"),
                ):
                    self._run_batch(
                        self._make_client(),
                        self._inputs(2),
                        self._response(
                            [ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE] * count
                        ),
                        preserve_capture_backpressure=preserve,
                    )

    def test_conversion_and_prefill_completion_failures_remain_atomic(self):
        for output in (
            _make_response(),
            _make_response(output_len=1),
            _make_response(
                output_len=0, output_ids=torch.tensor([[9]], dtype=torch.int32)
            ),
        ):
            response = self._response(
                [ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE, None]
            )
            response.results[1].final_output.CopyFrom(output)
            with self.subTest(output=output), self.assertRaises(FtRuntimeException):
                self._run_batch(
                    self._make_client(),
                    self._inputs(2),
                    response,
                    preserve_capture_backpressure=True,
                )
        error = ValueError("conversion failed")
        with patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.trans_output", side_effect=error
        ):
            with self.assertRaises(ValueError) as caught:
                self._run_batch(
                    self._make_client(),
                    self._inputs(1),
                    self._response([None]),
                    preserve_capture_backpressure=True,
                )
        self.assertIs(caught.exception, error)

    def test_timeout_selection_preserves_per_item_values(self):
        for default, timeouts, expected in (
            (5000, [1000, 2000], 2.0),
            (5000, [1000, -1], 5.0),
            (0, [1000, -1], None),
            (0, [-1, None], None),
        ):
            with self.subTest(default=default, timeouts=timeouts):
                client = self._make_client(max_rpc_timeout_ms=default)
                inputs = self._inputs(2)
                for item, timeout in zip(inputs, timeouts):
                    item.generate_config.timeout_ms = timeout
                before = [item.generate_config.model_dump() for item in inputs]
                self._run_batch(client, inputs, self._response([None, None]))
                args, kwargs = self.last_stub.BatchGenerateCall.call_args
                if expected is None:
                    self.assertNotIn("timeout", kwargs)
                else:
                    self.assertGreater(kwargs["timeout"], 0)
                    self.assertLessEqual(kwargs["timeout"], expected)
                expected_wire = [
                    (
                        value
                        if value is not None and value > 0
                        else (default if default > 0 else value)
                    )
                    for value in timeouts
                ]
                self.assertEqual(
                    [item.generate_config.timeout_ms for item in args[0].inputs],
                    [value if value is not None else 0 for value in expected_wire],
                )
                self.assertEqual(
                    before, [item.generate_config.model_dump() for item in inputs]
                )

    def test_transport_backpressure_is_not_an_item_and_preserves_native_details(self):
        class RpcFailure(grpc.RpcError):
            def __init__(self, raw_code):
                self.raw_code = raw_code

            def code(self):
                return grpc.StatusCode.RESOURCE_EXHAUSTED

            def details(self):
                return "HIDDEN_STATE_CAPTURE_BACKPRESSURE 8107"

            def trailing_metadata(self):
                if self.raw_code is None:
                    return ()
                return (
                    (
                        "grpc-status-details-bin",
                        ErrorDetailsPB(
                            error_code=self.raw_code, error_message="capture busy"
                        ).SerializeToString(),
                    ),
                )

        for raw_code, expected in (
            (8107, ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE),
            (28, ExceptionType.UNKNOWN_ERROR),
            (None, ExceptionType.UNKNOWN_ERROR),
        ):
            for preserve in (False, True):
                with (
                    self.subTest(raw_code=raw_code, preserve=preserve),
                    self.assertRaises(FtRuntimeException) as caught,
                ):
                    self._run_batch(
                        self._make_client(),
                        self._inputs(1),
                        None,
                        error=RpcFailure(raw_code),
                        preserve_capture_backpressure=preserve,
                    )
                self.assertEqual(caught.exception.exception_type, expected)
                self.assertTrue(self.last_call.cancelled)

    def test_channel_acquisition_timeout_does_not_start_rpc(self):
        client = self._make_client()

        @asynccontextmanager
        async def exhausted_pool(*args, **kwargs):
            raise asyncio.TimeoutError()
            yield

        with patch.object(client._channel_pool, "acquire", exhausted_pool):
            with self.assertRaises(FtRuntimeException) as caught:
                self._run_batch(client, self._inputs(1), None)
        self.assertEqual(
            caught.exception.exception_type, ExceptionType.GENERATE_TIMEOUT
        )
        self.last_stub.BatchGenerateCall.assert_not_called()

    def test_cancellation_releases_batch_channel_and_cancels_call(self):
        with self.assertRaises(asyncio.CancelledError):
            self._run_batch(
                self._make_client(),
                self._inputs(3),
                None,
                error=asyncio.CancelledError(),
            )
        self.assertTrue(self.last_call.cancelled)

    def test_rpc_enum_mapping_and_legacy_grammar_number(self):
        self.assertEqual(ExceptionType.GRAMMAR_COMPILE_OVERLOADED.value, 8105)
        self.assertEqual(ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE.value, 8107)
        self.assertEqual(
            ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE.category,
            ExceptionCategory.CAPACITY,
        )
        self.assertEqual(
            _trans_rpc_error_code(28), ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE
        )
        self.assertEqual(_trans_rpc_error_code(8107), ExceptionType.UNKNOWN_ERROR)
        for name, code in ErrorCodePB.items():
            if name == "NONE_ERROR":
                continue
            expected_name = (
                "P2P_CONNECTOR_WORKER_READ_CANCELLED"
                if name == "P2P_CONNECTOR_WORKER_READ_CANCELED"
                else name
            )
            expected = (
                ExceptionType.CANCELLED_ERROR
                if name == "CANCELLED"
                else ExceptionType[expected_name]
            )
            self.assertEqual(_trans_rpc_error_code(code), expected)


if __name__ == "__main__":
    setup_logging()
    main()
