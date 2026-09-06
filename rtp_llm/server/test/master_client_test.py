import asyncio
import socket
import time
import unittest
from types import SimpleNamespace
from typing import cast

import grpc
import grpc.aio

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2 import (
    HIGHER_PRIORITY_AHEAD,
    RESOURCE_EXHAUSTED,
    SAME_PRIORITY_AHEAD,
    SCHEDULE_FAILURE_REASON_UNSPECIFIED,
    FlexlbCancelResponsePB,
    FlexlbScheduleResponsePB,
    FlexlbServerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2_grpc import (
    FlexlbServiceServicer,
    add_FlexlbServiceServicer_to_server,
)
from rtp_llm.server.master_client import (
    FLEXLB_GRPC_PORT_OFFSET,
    MasterClient,
    _ScheduleAttemptSucceeded,
)


class _FakeMasterConfig:
    master_connect_timeout_ms = 100
    master_max_connect_pool_size = 4
    master_session_timeout_s = 1
    master_default_timeout_ms = 3600000


class _FakeHostService:
    def get_master_addr(self):
        return "master:1234"

    def get_slave_addr(self):
        return None


class _FakeHostServiceWithSlave(_FakeHostService):
    def get_slave_addr(self):
        return "slave:1234"


class _StaticHostService(_FakeHostService):
    def __init__(self, master_addr, slave_addr=None):
        self.master_addr = master_addr
        self.slave_addr = slave_addr

    def get_master_addr(self):
        return self.master_addr

    def get_slave_addr(self):
        return self.slave_addr


class _FakeGenerateConfig:
    max_new_tokens = 17
    num_beams = 2
    force_disable_sp_run = True
    ttft_timeout_ms = 3000
    timeout_ms = -1
    traffic_reject_priority = 12


class _FakeInput:
    prompt_length = 5

    def __init__(self, headers=None, ttft_timeout_ms=None):
        self.generate_config = _FakeGenerateConfig()
        if ttft_timeout_ms is not None:
            self.generate_config.ttft_timeout_ms = ttft_timeout_ms
        self.headers = {"x-request-id": "req-1"} if headers is None else headers


class _CaptureMasterClient(MasterClient):
    def __init__(self):
        super().__init__(
            host_service=_FakeHostService(),
            master_config=_FakeMasterConfig(),
        )
        self.calls = []

    async def _send_schedule_request(self, addr, request_pb, timeout_s, request_id):
        self.calls.append(
            {
                "addr": addr,
                "request_pb": request_pb,
                "timeout_s": timeout_s,
                "request_id": request_id,
            }
        )
        return _ScheduleAttemptSucceeded(
            FlexlbScheduleResponsePB(
                success=True,
                code=200,
                server_status=[
                    FlexlbServerStatusPB(
                        role="PREFILL",
                        server_ip="10.0.0.7",
                        http_port=8080,
                        grpc_port=9000,
                    )
                ],
                enqueued_by_master=True,
            )
        )


class _DeadlineMasterClient(MasterClient):
    def __init__(self):
        super().__init__(
            host_service=_FakeHostServiceWithSlave(),
            master_config=_FakeMasterConfig(),
        )
        self.calls = []

    async def _send_schedule_request(self, addr, request_pb, timeout_s, request_id):
        self.calls.append(addr)
        raise FtRuntimeException(
            ExceptionType.DEADLINE_EXCEEDED, "schedule deadline exceeded"
        )


class _RejectingMasterClient(MasterClient):
    def __init__(self, code, reason, *, include_reason=True):
        super().__init__(
            host_service=_FakeHostService(),
            master_config=_FakeMasterConfig(),
        )
        self.code = code
        self.reason = reason
        self.include_reason = include_reason

    async def _send_schedule_request(self, addr, request_pb, timeout_s, request_id):
        fields = {
            "code": int(self.code),
            "error_message": "private scheduler diagnostic",
            "queue_length": 3,
        }
        if self.include_reason:
            fields["admission_reject_reason"] = int(self.reason)
        return _ScheduleAttemptSucceeded(
            cast(FlexlbScheduleResponsePB, SimpleNamespace(**fields))
        )


class _FakeInputPB:
    def SerializeToString(self):
        return b"serialized-input"


class _FlexlbService(FlexlbServiceServicer):
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.schedule_received = asyncio.Event()
        self.schedule_requests = []
        self.cancel_requests = []

    async def Schedule(self, request, _context):
        self.schedule_requests.append(request)
        self.schedule_received.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        return FlexlbScheduleResponsePB(
            success=True,
            code=200,
            server_status=[
                FlexlbServerStatusPB(
                    role="PREFILL",
                    server_ip="10.0.0.8",
                    http_port=8000,
                    grpc_port=8001,
                )
            ],
        )

    async def Cancel(self, request, _context):
        self.cancel_requests.append(request)
        return FlexlbCancelResponsePB(found=True)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _start_server(service):
    server = grpc.aio.server()
    add_FlexlbServiceServicer_to_server(service, server)
    grpc_port = server.add_insecure_port("127.0.0.1:0")
    if grpc_port <= FLEXLB_GRPC_PORT_OFFSET:
        raise RuntimeError("failed to bind FlexLB test server")
    await server.start()
    return server, grpc_port


class MasterClientGrpcConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_does_not_cancel_concurrent_rpc_on_shared_channel(self):
        service = _FlexlbService(delay=0.2)
        server, grpc_port = await _start_server(service)
        follower_service = _FlexlbService()
        follower_server, follower_grpc_port = await _start_server(follower_service)

        master_addr = f"127.0.0.1:{grpc_port - FLEXLB_GRPC_PORT_OFFSET}"
        client = MasterClient(
            host_service=_StaticHostService(
                master_addr,
                f"127.0.0.1:{follower_grpc_port - FLEXLB_GRPC_PORT_OFFSET}",
            ),
            master_config=_FakeMasterConfig(),
        )
        long_request_id = 201
        short_request_id = 202
        long_task = None
        try:
            long_task = asyncio.create_task(
                client.get_backend_role_addrs(
                    block_cache_keys=[11, 12],
                    cache_key_block_size=1024,
                    input=_FakeInput(ttft_timeout_ms=500),
                    request_id=long_request_id,
                )
            )
            await asyncio.wait_for(service.schedule_received.wait(), timeout=0.5)

            with self.assertRaises(FtRuntimeException) as raised:
                await client.get_backend_role_addrs(
                    block_cache_keys=[21, 22],
                    cache_key_block_size=1024,
                    input=_FakeInput(ttft_timeout_ms=20),
                    request_id=short_request_id,
                )

            self.assertEqual(
                raised.exception.exception_type, ExceptionType.DEADLINE_EXCEEDED
            )
            long_result = await long_task
            self.assertTrue(long_result.is_ok)
            self.assertEqual(long_result.role_addrs[0].ip, "10.0.0.8")
            self.assertEqual(len(service.schedule_requests), 2)
            self.assertEqual(follower_service.schedule_requests, [])
            self.assertTrue(
                any(
                    request.request_id == short_request_id
                    for request in service.cancel_requests
                )
            )
        finally:
            if long_task is not None and not long_task.done():
                long_task.cancel()
                await asyncio.gather(long_task, return_exceptions=True)
            await client.close()
            await server.stop(None)
            await follower_server.stop(None)

    async def test_transport_failure_retries_follower_with_explicit_attempt(self):
        service = _FlexlbService()
        server, follower_grpc_port = await _start_server(service)
        unavailable_grpc_port = _free_port()
        self.assertGreater(unavailable_grpc_port, FLEXLB_GRPC_PORT_OFFSET)
        client = MasterClient(
            host_service=_StaticHostService(
                f"127.0.0.1:{unavailable_grpc_port - FLEXLB_GRPC_PORT_OFFSET}",
                f"127.0.0.1:{follower_grpc_port - FLEXLB_GRPC_PORT_OFFSET}",
            ),
            master_config=_FakeMasterConfig(),
        )
        try:
            response = await client.get_backend_role_addrs(
                block_cache_keys=[11, 12],
                cache_key_block_size=1024,
                input=_FakeInput(ttft_timeout_ms=500),
                request_id=203,
            )

            self.assertTrue(response.is_ok)
            self.assertEqual(response.role_addrs[0].ip, "10.0.0.8")
            self.assertEqual(len(service.schedule_requests), 1)
        finally:
            await client.close()
            await server.stop(None)

    async def test_channel_readiness_timeout_retries_follower_before_deadline(self):
        connected = asyncio.Event()
        release_connections = asyncio.Event()
        handler_tasks: set[asyncio.Task[None]] = set()

        async def hold_without_http2_handshake(
            _reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            task = asyncio.current_task()
            self.assertIsNotNone(task)
            handler_tasks.add(cast(asyncio.Task[None], task))
            connected.set()
            try:
                await release_connections.wait()
            finally:
                writer.close()
                await writer.wait_closed()
                handler_tasks.discard(cast(asyncio.Task[None], task))

        blackhole_server = await asyncio.start_server(
            hold_without_http2_handshake,
            "127.0.0.1",
            0,
        )
        blackhole_socket = blackhole_server.sockets[0]
        blackhole_grpc_port = int(blackhole_socket.getsockname()[1])

        follower_service = _FlexlbService()
        follower_server, follower_grpc_port = await _start_server(follower_service)
        client = MasterClient(
            host_service=_StaticHostService(
                f"127.0.0.1:{blackhole_grpc_port - FLEXLB_GRPC_PORT_OFFSET}",
                f"127.0.0.1:{follower_grpc_port - FLEXLB_GRPC_PORT_OFFSET}",
            ),
            master_config=_FakeMasterConfig(),
        )
        started_at = time.monotonic()
        try:
            response = await client.get_backend_role_addrs(
                block_cache_keys=[11, 12],
                cache_key_block_size=1024,
                input=_FakeInput(ttft_timeout_ms=1_000),
                request_id=204,
            )
            elapsed_s = time.monotonic() - started_at

            self.assertTrue(connected.is_set())
            self.assertTrue(response.is_ok)
            self.assertEqual(response.role_addrs[0].ip, "10.0.0.8")
            self.assertEqual(len(follower_service.schedule_requests), 1)
            self.assertLess(elapsed_s, 0.8)
        finally:
            await client.close()
            await follower_server.stop(None)
            blackhole_server.close()
            release_connections.set()
            await asyncio.gather(*tuple(handler_tasks), return_exceptions=True)
            await blackhole_server.wait_closed()

    def test_non_positive_connect_timeout_is_rejected(self):
        master_config = _FakeMasterConfig()
        master_config.master_connect_timeout_ms = 0

        with self.assertRaisesRegex(ValueError, "connect timeout must be positive"):
            MasterClient(
                host_service=_StaticHostService("master:1234"),
                master_config=master_config,
            )


class MasterClientBatchPayloadTest(unittest.IsolatedAsyncioTestCase):
    def test_python_reason_enum_matches_schedule_wire_values(self):
        self.assertEqual(
            int(AdmissionRejectReason.UNSPECIFIED),
            SCHEDULE_FAILURE_REASON_UNSPECIFIED,
        )
        self.assertEqual(
            int(AdmissionRejectReason.HIGHER_PRIORITY_AHEAD),
            HIGHER_PRIORITY_AHEAD,
        )
        self.assertEqual(
            int(AdmissionRejectReason.SAME_PRIORITY_AHEAD),
            SAME_PRIORITY_AHEAD,
        )
        self.assertEqual(
            int(AdmissionRejectReason.RESOURCE_EXHAUSTED),
            RESOURCE_EXHAUSTED,
        )

    async def test_schedule_payload_contains_batch_fields_and_pb(self):
        client = _CaptureMasterClient()

        response = await client.get_backend_role_addrs(
            block_cache_keys=[1, 2, 3],
            cache_key_block_size=1024,
            input=_FakeInput(),
            request_id=99,
            input_pb=_FakeInputPB(),
        )

        self.assertTrue(response.is_ok)
        self.assertTrue(response.enqueued_by_master)
        self.assertEqual(response.role_addrs[0].ip, "10.0.0.7")

        call = client.calls[0]
        request_pb = call["request_pb"]
        self.assertEqual(call["addr"], "master:1234")
        self.assertEqual(call["timeout_s"], 3.0)
        self.assertEqual(call["request_id"], 99)
        self.assertEqual(list(request_pb.block_cache_keys), [1, 2, 3])
        self.assertEqual(request_pb.seq_len, 5)
        self.assertEqual(request_pb.generate_timeout, 3000)
        self.assertEqual(request_pb.request_id, 99)
        self.assertEqual(request_pb.max_new_tokens, 17)
        self.assertEqual(request_pb.num_beams, 2)
        self.assertTrue(request_pb.force_disable_sp_run)
        self.assertEqual(request_pb.generate_input, b"serialized-input")
        self.assertEqual(request_pb.cache_key_block_size, 1024)
        self.assertEqual(request_pb.priority, 50)

    async def test_schedule_payload_priority_from_qos_header(self):
        client = _CaptureMasterClient()

        await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=_FakeInput(headers={"x-dashscope-inner-qos-level": "70"}),
            request_id=101,
            input_pb=_FakeInputPB(),
        )

        self.assertEqual(client.calls[0]["request_pb"].priority, 70)

    async def test_schedule_payload_priority_defaults_when_header_missing(self):
        client = _CaptureMasterClient()

        await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=_FakeInput(headers={}),
            request_id=102,
            input_pb=_FakeInputPB(),
        )

        self.assertEqual(client.calls[0]["request_pb"].priority, 50)

    async def test_schedule_payload_priority_invalid_header_no_raise(self):
        client = _CaptureMasterClient()

        response = await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=_FakeInput(headers={"x-dashscope-inner-qos-level": "high"}),
            request_id=103,
            input_pb=_FakeInputPB(),
        )

        self.assertTrue(response.is_ok)
        self.assertEqual(client.calls[0]["request_pb"].priority, 50)

    async def test_schedule_payload_priority_falls_back_to_generate_config(self):
        client = _CaptureMasterClient()
        input = _FakeInput(headers={})
        input.generate_config.qos_priority = 77

        await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=input,
            request_id=104,
            input_pb=_FakeInputPB(),
        )

        self.assertEqual(client.calls[0]["request_pb"].priority, 77)

    async def test_schedule_payload_priority_out_of_range_defaults(self):
        client = _CaptureMasterClient()

        await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=_FakeInput(headers={"x-dashscope-inner-qos-level": "101"}),
            request_id=105,
            input_pb=_FakeInputPB(),
        )

        self.assertEqual(client.calls[0]["request_pb"].priority, 50)

    async def test_schedule_deadline_does_not_retry_slave(self):
        client = _DeadlineMasterClient()

        with self.assertRaises(FtRuntimeException) as raised:
            await client.get_backend_role_addrs(
                block_cache_keys=[1],
                cache_key_block_size=1024,
                input=_FakeInput(),
                request_id=100,
                input_pb=_FakeInputPB(),
            )

        self.assertEqual(
            raised.exception.exception_type, ExceptionType.DEADLINE_EXCEEDED
        )
        self.assertEqual(client.calls, ["master:1234"])

    async def test_schedule_failure_preserves_typed_admission_reason(self):
        cases = (
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
            ),
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD,
            ),
            (
                ExceptionType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
            ),
            (
                ExceptionType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED,
            ),
        )
        for exception_type, reason in cases:
            with self.subTest(exception_type=exception_type, reason=reason):
                client = _RejectingMasterClient(exception_type, reason)
                with self.assertRaises(FtRuntimeException) as raised:
                    await client.get_backend_role_addrs(
                        block_cache_keys=[1],
                        cache_key_block_size=1024,
                        input=_FakeInput(),
                        request_id=104,
                        input_pb=_FakeInputPB(),
                    )

                self.assertEqual(exception_type, raised.exception.exception_type)
                self.assertEqual(
                    reason,
                    raised.exception.admission_reject_reason,
                )
                self.assertEqual(
                    "private scheduler diagnostic",
                    raised.exception.message,
                )

    async def test_missing_reason_field_falls_back_to_unspecified(self):
        client = _RejectingMasterClient(
            ExceptionType.ADMISSION_UNAVAILABLE,
            AdmissionRejectReason.UNSPECIFIED,
            include_reason=False,
        )

        with self.assertRaises(FtRuntimeException) as raised:
            await client.get_backend_role_addrs(
                block_cache_keys=[1],
                cache_key_block_size=1024,
                input=_FakeInput(),
                request_id=105,
                input_pb=_FakeInputPB(),
            )

        self.assertEqual(
            AdmissionRejectReason.UNSPECIFIED,
            raised.exception.admission_reject_reason,
        )

    async def test_unknown_reason_is_preserved_as_invalid(self):
        client = _RejectingMasterClient(
            ExceptionType.PRIORITY_PREEMPTED,
            999,
        )

        with self.assertRaises(FtRuntimeException) as raised:
            await client.get_backend_role_addrs(
                block_cache_keys=[1],
                cache_key_block_size=1024,
                input=_FakeInput(),
                request_id=106,
                input_pb=_FakeInputPB(),
            )

        self.assertEqual(
            AdmissionRejectReason.INVALID,
            raised.exception.admission_reject_reason,
        )


if __name__ == "__main__":
    unittest.main()
