"""Request-scoped SP logprobs compatibility over real localhost aio gRPC.

The peers implement historical wire-decoding policies, not an old model engine.
Visitor validation, serialization, channel pools and output conversion are real.
"""

import asyncio
import os
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace
from unittest.mock import patch

import grpc
import torch
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ReturnAllProbsMode
from rtp_llm.cpp.model_rpc.proto import flexlb_schedule_service_pb2 as flex_pb
from rtp_llm.cpp.model_rpc.proto import flexlb_schedule_service_pb2_grpc as flex_rpc
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc
from rtp_llm.ops import (
    PDSepConfig,
    RoleType,
    SpeculativeExecutionConfig,
    SpeculativeType,
)
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.master_client import FLEXLB_GRPC_PORT_OFFSET
from rtp_llm.utils.base_model_datatypes import GenerateInput
from rtp_llm.utils.grpc_util import trans_from_tensor


def _old_config_type(mode_aware):
    """Model the legacy schema without the speculative probability field."""
    file = descriptor_pb2.FileDescriptorProto(
        name=f"legacy_sp_logprobs_{mode_aware}.proto", syntax="proto3"
    )
    message = file.message_type.add(name="LegacyConfig")
    message.field.add(name="return_all_probs", number=27, type=8, label=1)
    if mode_aware:
        message.field.add(name="return_all_probs_mode", number=60, type=5, label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("LegacyConfig"))


_BOOL_CONFIG = _old_config_type(False)
_MODE_CONFIG = _old_config_type(True)


def _legacy_mode(config, mode_aware):
    old_type = _MODE_CONFIG if mode_aware else _BOOL_CONFIG
    old = old_type.FromString(config.SerializeToString())
    if mode_aware and old.return_all_probs_mode:
        return old.return_all_probs_mode - 1
    return (
        ReturnAllProbsMode.DEFAULT if old.return_all_probs else ReturnAllProbsMode.NONE
    )


def _response(mode=ReturnAllProbsMode.NONE, *, finished=True, empty=False):
    response = pb.GenerateOutputsPB()
    output = response.flatten_output
    output.finished.append(finished)
    if not empty:
        output.output_ids.CopyFrom(
            trans_from_tensor(torch.tensor([[[3]]], dtype=torch.int32))
        )
    if mode != ReturnAllProbsMode.NONE and not empty:
        values = (
            [0.1, 0.2, 0.3, 0.4]
            if mode == ReturnAllProbsMode.DEFAULT
            else [0.4, 0.3, 0.2, 0.1]
        )
        output.all_probs.CopyFrom(
            trans_from_tensor(torch.tensor([[values]], dtype=torch.float32))
        )
    return response


class _Peer(rpc.RpcServiceServicer):
    def __init__(
        self, *, mode_aware=True, new=False, sp=True, fail_first=False, empty=False
    ):
        self.mode_aware, self.new, self.sp = mode_aware, new, sp
        self.fail_first, self.empty = fail_first, empty
        self.inputs = []
        self.saved = {}
        self.unsafe = 0
        self.calls = 0
        self.fetch_calls = 0
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def admit(self, request, context):
        self.inputs.append(request)
        config = request.generate_config
        mode = _legacy_mode(config, self.mode_aware)
        if self.new and config.speculative_return_all_probs_mode in (2, 3):
            mode = config.speculative_return_all_probs_mode - 1
        if self.sp and not self.new and not config.force_disable_sp_run and mode != 0:
            self.unsafe += 1
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "unsafe legacy SP logprobs"
            )
        self.saved[request.request_id] = (request, mode)
        return mode

    async def stream(self, request, mode):
        if self.empty:
            yield pb.GenerateOutputsPB()  # No outputs at all: not a capability acknowledgment.
            yield _response(finished=False, empty=True)  # Metadata without output_ids.
            terminal = _response(empty=True)
            terminal.flatten_output.output_ids.CopyFrom(
                trans_from_tensor(torch.empty((1, 1, 0), dtype=torch.int32))
            )
            yield terminal  # An explicit zero-token tensor also needs no scores.
        elif (
            self.new
            or not self.sp
            or request.generate_config.force_disable_sp_run
            or not request.generate_config.speculative_return_all_probs_mode
        ):
            yield _response(mode)
        else:
            # A safe old peer generated a token without probabilities. The new
            # client must fail before publishing it and cancel this still-live RPC.
            try:
                yield _response(finished=False)
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def GenerateStreamCall(self, request, context):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            await context.abort(
                grpc.StatusCode.UNAVAILABLE, "injected transport unavailable"
            )
        mode = await self.admit(request, context)
        async for response in self.stream(request, mode):
            yield response

    async def FetchResponse(self, request, context):
        self.fetch_calls += 1
        original, mode = self.saved[request.request_id]
        async for response in self.stream(original, mode):
            yield response

    async def BatchGenerateCall(self, request, context):
        response = pb.BatchGenerateOutputsPB()
        for item in request.inputs:
            mode = await self.admit(item, context)
            response.results.add().final_output.CopyFrom(_response(mode))
        return response


class _Master(flex_rpc.FlexlbServiceServicer):
    def __init__(self, peer):
        self.peer = peer
        self.port = 0
        self.payloads = []

    async def Schedule(self, request, context):
        self.payloads.append(request.generate_input)
        # This is the master's enqueue wire boundary. Retain the complete PB;
        # only its selected endpoint changes, as in FlexLB's mergeFrom path.
        original = pb.GenerateInputPB.FromString(request.generate_input)
        await self.peer.admit(original, context)
        return flex_pb.FlexlbScheduleResponsePB(
            success=True,
            code=200,
            enqueued_by_master=True,
            server_status=[
                flex_pb.FlexlbServerStatusPB(
                    role="PREFILL", server_ip="127.0.0.1", grpc_port=self.port
                )
            ],
        )


@asynccontextmanager
async def _serve(peer, master=None):
    server = grpc.aio.server()
    rpc.add_RpcServiceServicer_to_server(peer, server)
    if master is not None:
        flex_rpc.add_FlexlbServiceServicer_to_server(master, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if master is not None:
        master.port = port
    await server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        peer.release.set()
        await server.stop(None)


class SpeculativeLogprobsWireCompatTest(unittest.IsolatedAsyncioTestCase):
    def visitor(self, address, *, sp=True):
        config = SpeculativeExecutionConfig()
        config.model_type = "kimi_k3_mtp" if sp else ""
        config.type = SpeculativeType.MTP if sp else SpeculativeType.NONE
        pd = PDSepConfig()
        pd.role_type = RoleType.PDFUSION
        pd.max_rpc_timeout_ms = 2000
        # Configure discovery off, without replacing validation, serialization,
        # the channel pool or any transport operation.
        with patch.dict(os.environ, {"MODEL_SERVICE_CONFIG": ""}):
            visitor = BackendRPCServerVisitor(64, 8, pd, [address], sp_config=config)
        visitor.pd_route_retry_on_unavailable = 0
        self.addAsyncCleanup(visitor.close)
        return visitor

    @staticmethod
    def request(mode, request_id=1):
        return GenerateInput(
            request_id=request_id,
            token_ids=torch.tensor([1, 2], dtype=torch.int32),
            mm_inputs=[],
            generate_config=GenerateConfig(
                max_new_tokens=1,
                top_k=1,
                timeout_ms=2000,
                is_streaming=True,
                return_all_probs=mode,
            ),
        )

    async def consume(self, visitor, request):
        stream = await visitor.enqueue(request)
        try:
            return [output async for output in stream]
        finally:
            await stream.aclose()

    async def failed_consume(self, visitor, request):
        try:
            await self.consume(visitor, request)
        except FtRuntimeException as error:
            return error
        self.fail(
            "guarded SP probabilities must not silently become a token-only success"
        )

    def assert_guarded_wire(self, request, mode):
        config = request.generate_config
        self.assertFalse(config.return_all_probs)
        self.assertEqual(config.return_all_probs_mode, 1)
        self.assertEqual(config.speculative_return_all_probs_mode, mode + 1)
        for old_type in (_BOOL_CONFIG, _MODE_CONFIG):
            old = old_type.FromString(config.SerializeToString())
            restored = pb.GenerateConfigPB.FromString(old.SerializeToString())
            self.assertEqual(restored.speculative_return_all_probs_mode, mode + 1)

    async def test_legacy_decoders_fail_safely_and_next_request_is_healthy(self):
        for mode_aware in (False, True):
            for mode in (ReturnAllProbsMode.DEFAULT, ReturnAllProbsMode.ORIGINAL):
                with self.subTest(mode_aware=mode_aware, mode=mode):
                    peer = _Peer(mode_aware=mode_aware)
                    async with _serve(peer) as address:
                        visitor = self.visitor(address)
                        request = self.request(mode)
                        error = await self.failed_consume(visitor, request)
                        self.assertEqual(peer.unsafe, 0)
                        self.assertEqual(
                            error.exception_type, ExceptionType.UNSUPPORTED_OPERATION
                        )
                        self.assert_guarded_wire(peer.inputs[0], mode)
                        await asyncio.wait_for(peer.cancelled.wait(), 1)
                        ordinary = await self.consume(
                            visitor, self.request(ReturnAllProbsMode.NONE, 2)
                        )
                        self.assertEqual(
                            ordinary[0].generate_outputs[0].output_ids.tolist(), [[3]]
                        )
                        self.assertEqual(peer.unsafe, 0)

    async def test_new_peer_returns_real_default_and_original_probability_tensors(self):
        peer = _Peer(new=True)
        async with _serve(peer) as address:
            visitor = self.visitor(address)
            for mode in (ReturnAllProbsMode.DEFAULT, ReturnAllProbsMode.ORIGINAL):
                outputs = await self.consume(visitor, self.request(mode, mode))
                expected = (
                    [0.1, 0.2, 0.3, 0.4]
                    if mode == ReturnAllProbsMode.DEFAULT
                    else [0.4, 0.3, 0.2, 0.1]
                )
                self.assertTrue(
                    torch.equal(
                        outputs[0].generate_outputs[0].all_probs,
                        torch.tensor([expected], dtype=torch.float32),
                    )
                )
                self.assert_guarded_wire(peer.inputs[-1], mode)

    async def test_empty_data_and_terminal_frames_do_not_require_token_probabilities(
        self,
    ):
        peer = _Peer(empty=True)
        async with _serve(peer) as address:
            visitor = self.visitor(address)
            outputs = await self.consume(
                visitor, self.request(ReturnAllProbsMode.DEFAULT)
            )
            self.assertEqual(peer.unsafe, 0)
            self.assertEqual(len(outputs), 3)
            self.assertFalse(outputs[0].generate_outputs)
            self.assertIsNone(outputs[1].generate_outputs[0].output_ids)
            self.assertTrue(outputs[2].generate_outputs[0].finished)
            self.assertEqual(outputs[2].generate_outputs[0].output_ids.numel(), 0)

    async def test_non_sp_and_force_disabled_sp_keep_legacy_modes(self):
        for sp, disable in ((False, False), (True, True)):
            for mode in (ReturnAllProbsMode.DEFAULT, ReturnAllProbsMode.ORIGINAL):
                with self.subTest(sp=sp, disable=disable, mode=mode):
                    peer = _Peer(sp=sp)
                    async with _serve(peer) as address:
                        visitor = self.visitor(address, sp=sp)
                        request = self.request(mode)
                        if disable:
                            visitor.check_sp_supported(request)
                            request.generate_config.force_disable_sp_run = True
                        outputs = await self.consume(visitor, request)
                        self.assertIsNotNone(outputs[0].generate_outputs[0].all_probs)
                        self.assertEqual(peer.unsafe, 0)
                        wire = peer.inputs[0].generate_config
                        self.assertTrue(wire.return_all_probs)
                        self.assertEqual(wire.return_all_probs_mode, mode + 1)
                        self.assertEqual(wire.speculative_return_all_probs_mode, 0)

    async def test_copied_batch_requests_preserve_probability_modes(self):
        peer = _Peer(new=True)
        async with _serve(peer) as address:
            visitor = self.visitor(address)
            requests = []
            for mode in (ReturnAllProbsMode.DEFAULT, ReturnAllProbsMode.ORIGINAL):
                request = self.request(mode, mode)
                visitor.check_sp_supported(request)
                requests.append(
                    replace(
                        request,
                        generate_config=request.generate_config.model_copy(deep=True),
                    )
                )

            outputs = await visitor.batch_enqueue(requests)
            self.assertEqual(len(outputs), 2)
            self.assertEqual(len(peer.inputs), 2)
            for request, wire, output in zip(requests, peer.inputs, outputs):
                self.assert_guarded_wire(wire, request.generate_config.return_all_probs)
                self.assertIsNotNone(output.generate_outputs[0].all_probs)

    async def test_old_batch_fails_atomically_without_an_unsafe_mode(self):
        peer = _Peer()
        async with _serve(peer) as address:
            visitor = self.visitor(address)
            error = None
            try:
                await visitor.batch_enqueue([self.request(1, 1), self.request(2, 2)])
            except FtRuntimeException as caught:
                error = caught
            self.assertEqual(peer.unsafe, 0)
            self.assertIsNotNone(error)
            self.assertEqual(error.exception_type, ExceptionType.UNSUPPORTED_OPERATION)
            self.assertEqual(len(peer.inputs), 2)
            for request in peer.inputs:
                self.assert_guarded_wire(request, request.request_id)

    async def test_master_enqueue_serialization_reaches_the_actual_fetch_request(self):
        peer = _Peer()
        master = _Master(peer)
        async with _serve(peer, master) as address:
            visitor = self.visitor(address)
            visitor.host_service.service_available = True
            visitor.host_service.get_master_addr = (
                lambda: f"127.0.0.1:{master.port - FLEXLB_GRPC_PORT_OFFSET}"
            )
            request = self.request(ReturnAllProbsMode.ORIGINAL)
            error = await self.failed_consume(visitor, request)
            self.assertEqual(peer.unsafe, 0)
            self.assertEqual(error.exception_type, ExceptionType.UNSUPPORTED_OPERATION)
            self.assertTrue(request.enqueued_by_master)
            self.assertEqual(peer.fetch_calls, 1)
            self.assertEqual(len(master.payloads), 1)
            self.assert_guarded_wire(
                pb.GenerateInputPB.FromString(master.payloads[0]),
                ReturnAllProbsMode.ORIGINAL,
            )
            await asyncio.wait_for(peer.cancelled.wait(), 1)

    async def test_transport_retry_keeps_marker_and_original_transport_error(self):
        peer = _Peer(fail_first=True)
        async with _serve(peer) as address:
            visitor = self.visitor(address)
            visitor.pd_route_retry_on_unavailable = 1
            visitor.set_request_id_factory(lambda: 2)
            error = await self.failed_consume(
                visitor, self.request(ReturnAllProbsMode.DEFAULT)
            )
            self.assertEqual(peer.calls, 2)
            self.assertEqual(peer.unsafe, 0)
            self.assertEqual(error.exception_type, ExceptionType.CONNECT_FAILED)
            self.assert_guarded_wire(peer.inputs[0], ReturnAllProbsMode.DEFAULT)
            self.assertEqual(peer.inputs[0].request_id, 2)
            await asyncio.wait_for(peer.cancelled.wait(), 1)


if __name__ == "__main__":
    unittest.main()
