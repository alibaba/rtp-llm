import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import grpc

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2 import (
    CANCEL_REASON_CLIENT_CANCELLED,
    CANCEL_REASON_DEADLINE_EXCEEDED,
    HIGHER_PRIORITY_AHEAD,
    RESOURCE_EXHAUSTED,
    SAME_PRIORITY_AHEAD,
    SCHEDULE_FAILURE_REASON_UNSPECIFIED,
    FlexlbCancelResponsePB,
    FlexlbScheduleRequestPB,
    FlexlbScheduleResponsePB,
    FlexlbServerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2_grpc import (
    FlexlbServiceServicer,
    add_FlexlbServiceServicer_to_server,
)
from rtp_llm.server.master_client import FLEXLB_GRPC_PORT_OFFSET, MasterClient


class _FakeMasterConfig:
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


class _FakeGenerateConfig:
    max_new_tokens = 17
    num_beams = 2
    force_disable_sp_run = True
    ttft_timeout_ms = 3000
    timeout_ms = -1
    traffic_reject_priority = 12


class _FakeInput:
    prompt_length = 5

    def __init__(self, headers=None):
        self.generate_config = _FakeGenerateConfig()
        self.headers = {"x-request-id": "req-1"} if headers is None else headers


class _CaptureMasterClient(MasterClient):
    def __init__(self, host_service=None, responses=None):
        super().__init__(
            host_service=host_service or _FakeHostService(),
            master_config=_FakeMasterConfig(),
        )
        self.calls = []
        self.responses = iter(responses) if responses is not None else None

    async def _send_schedule_request(self, addr, request_pb, timeout_s, request_id):
        self.calls.append(
            {
                "addr": addr,
                "request_pb": FlexlbScheduleRequestPB.FromString(
                    request_pb.SerializeToString()
                ),
                "timeout_s": timeout_s,
                "request_id": request_id,
            }
        )
        if self.responses is not None:
            return next(self.responses)
        return FlexlbScheduleResponsePB(
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
        return SimpleNamespace(**fields)


class _FakeInputPB:
    def SerializeToString(self):
        return b"serialized-input"


class _ConcurrentScheduleService(FlexlbServiceServicer):
    def __init__(self):
        self.started = {1: asyncio.Event(), 2: asyncio.Event()}
        self.release = asyncio.Event()
        self.failure_status = None
        self.schedule_requests = []
        self.cancel_requests = []
        self.response = FlexlbScheduleResponsePB(
            code=200,
            success=True,
            server_status=[
                FlexlbServerStatusPB(
                    role="PREFILL",
                    server_ip="10.0.0.8",
                    http_port=8000,
                    grpc_port=8001,
                )
            ],
        )

    async def Schedule(self, request, context):
        self.schedule_requests.append(request.request_id)
        if request.request_id in self.started:
            self.started[request.request_id].set()
            if request.request_id == 1 and self.failure_status is not None:
                await context.abort(self.failure_status, "schedule rejected")
            await self.release.wait()
        return self.response

    async def Cancel(self, request, context):
        self.cancel_requests.append((request.request_id, request.reason))
        return FlexlbCancelResponsePB(found=True)


class MasterClientChannelIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = _ConcurrentScheduleService()
        self.server = grpc.aio.server()
        add_FlexlbServiceServicer_to_server(self.service, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        await self.server.start()
        self.address = f"127.0.0.1:{port - FLEXLB_GRPC_PORT_OFFSET}"
        self.target = f"127.0.0.1:{port}"
        self.client = MasterClient(master_config=_FakeMasterConfig())
        self.channel = self.client._get_channel(self.target)
        self.requests = []

    async def asyncTearDown(self):
        await self.client.close()
        await asyncio.gather(*self.requests, return_exceptions=True)
        await self.server.stop(0)

    async def _start_pending_request(self, request_id, timeout_s=None):
        pending = asyncio.create_task(
            self.client._send_schedule_request(
                self.address,
                FlexlbScheduleRequestPB(request_id=request_id),
                timeout_s,
                request_id,
            )
        )
        self.requests.append(pending)
        await asyncio.wait_for(self.service.started[request_id].wait(), timeout=5)
        self.assertFalse(pending.done())
        return pending

    async def _start_slave(self):
        service = _ConcurrentScheduleService()
        service.release.set()
        server = grpc.aio.server()
        add_FlexlbServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        self.addAsyncCleanup(server.stop, 0)
        await server.start()
        self.client.host_service = SimpleNamespace(
            get_master_addr=lambda: self.address,
            get_slave_addr=lambda: f"127.0.0.1:{port - FLEXLB_GRPC_PORT_OFFSET}",
        )
        return service

    async def _assert_peer_completes_and_channel_is_reused(self, peer):
        self.service.release.set()
        response = await asyncio.wait_for(peer, timeout=5)
        self.assertEqual(response.code, 200)
        self.assertIs(self.client._get_channel(self.target), self.channel)
        response = await self.client._send_schedule_request(
            self.address, FlexlbScheduleRequestPB(request_id=3), 5, 3
        )
        self.assertEqual(response.code, 200)
        self.assertIs(self.client._get_channel(self.target), self.channel)

    async def _assert_rpc_failure_is_isolated(self, status):
        peer = await self._start_pending_request(2)
        self.service.failure_status = status
        with self.assertLogs("route_logger", level="ERROR"):
            response = await self.client._send_schedule_request(
                self.address, FlexlbScheduleRequestPB(request_id=1), 5, 1
            )
        self.assertIsNone(response)
        await self._assert_peer_completes_and_channel_is_reused(peer)
        self.assertEqual(self.service.cancel_requests, [])

    async def test_resource_exhausted_does_not_cancel_shared_channel_peer(self):
        await self._assert_rpc_failure_is_isolated(grpc.StatusCode.RESOURCE_EXHAUSTED)

    async def test_unavailable_does_not_cancel_shared_channel_peer(self):
        await self._assert_rpc_failure_is_isolated(grpc.StatusCode.UNAVAILABLE)

    async def test_transport_failure_retries_slave_without_cancelling_peer(self):
        slave = await self._start_slave()
        peer = await self._start_pending_request(2)
        self.service.failure_status = grpc.StatusCode.RESOURCE_EXHAUSTED
        with self.assertLogs("route_logger", level="ERROR"):
            response = await self.client.get_backend_role_addrs(
                [11, 12], 1024, _FakeInput(), 1
            )
        self.assertTrue(response.is_ok)
        self.assertFalse(response.connection_failed)
        self.assertEqual(response.role_addrs[0].ip, "10.0.0.8")
        self.assertEqual(response.role_addrs[0].grpc_port, 8001)
        self.assertEqual(self.service.schedule_requests, [2, 1])
        self.assertEqual(slave.schedule_requests, [1])
        self.assertFalse(peer.done())
        await self._assert_peer_completes_and_channel_is_reused(peer)
        self.assertEqual(self.service.cancel_requests, [])
        self.assertEqual(slave.cancel_requests, [])

    async def test_both_transport_failures_return_connection_failed(self):
        slave = await self._start_slave()
        peer = await self._start_pending_request(2)
        self.service.failure_status = grpc.StatusCode.UNAVAILABLE
        slave.failure_status = grpc.StatusCode.UNAVAILABLE
        with self.assertLogs("route_logger", level="ERROR"):
            response = await self.client.get_backend_role_addrs(
                [11, 12], 1024, _FakeInput(), 1
            )
        self.assertTrue(response.connection_failed)
        self.assertFalse(response.is_ok)
        self.assertEqual(self.service.schedule_requests, [2, 1])
        self.assertEqual(slave.schedule_requests, [1])
        self.assertFalse(peer.done())
        await self._assert_peer_completes_and_channel_is_reused(peer)
        self.assertEqual(self.service.cancel_requests, [])
        self.assertEqual(slave.cancel_requests, [])

    async def test_business_rejection_does_not_retry_slave(self):
        slave = await self._start_slave()
        self.service.release.set()
        self.service.response = FlexlbScheduleResponsePB(
            code=int(ExceptionType.PRIORITY_ADMISSION_REJECTED),
            admission_reject_reason=int(AdmissionRejectReason.SAME_PRIORITY_AHEAD),
            error_message="same-priority request ahead",
        )
        with self.assertLogs("route_logger", level="ERROR"):
            with self.assertRaises(FtRuntimeException) as raised:
                await self.client.get_backend_role_addrs(
                    [11, 12], 1024, _FakeInput(), 1
                )
        self.assertEqual(
            raised.exception.exception_type, ExceptionType.PRIORITY_ADMISSION_REJECTED
        )
        self.assertEqual(
            raised.exception.admission_reject_reason,
            AdmissionRejectReason.SAME_PRIORITY_AHEAD,
        )
        self.assertEqual(self.service.schedule_requests, [1])
        self.assertEqual(slave.schedule_requests, [])
        self.assertIs(self.client._get_channel(self.target), self.channel)
        self.assertEqual(self.service.cancel_requests, [])

    async def test_deadline_cancels_only_expired_request(self):
        peer = await self._start_pending_request(2)
        with self.assertLogs("route_logger", level="ERROR"):
            expired = await self._start_pending_request(1, timeout_s=0.1)
            with self.assertRaises(FtRuntimeException) as raised:
                await asyncio.wait_for(expired, timeout=5)
        self.assertEqual(
            raised.exception.exception_type, ExceptionType.DEADLINE_EXCEEDED
        )
        self.assertEqual(
            self.service.cancel_requests, [(1, CANCEL_REASON_DEADLINE_EXCEEDED)]
        )
        await self._assert_peer_completes_and_channel_is_reused(peer)

    async def test_task_cancel_cancels_only_requested_rpc(self):
        peer = await self._start_pending_request(2)
        cancelled = await self._start_pending_request(1)
        self.assertTrue(cancelled.cancel())
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled, timeout=5)
        self.assertEqual(
            self.service.cancel_requests, [(1, CANCEL_REASON_CLIENT_CANCELLED)]
        )
        await self._assert_peer_completes_and_channel_is_reused(peer)

    async def test_local_exception_does_not_cancel_shared_channel_peer(self):
        peer = await self._start_pending_request(2)
        with patch(
            "rtp_llm.server.master_client.start_client_span",
            side_effect=RuntimeError("span setup failed"),
        ), self.assertLogs("route_logger", level="ERROR"):
            response = await self.client._send_schedule_request(
                self.address, FlexlbScheduleRequestPB(request_id=1), 5, 1
            )
        self.assertIsNone(response)
        await self._assert_peer_completes_and_channel_is_reused(peer)
        self.assertEqual(self.service.cancel_requests, [])

    async def test_owner_close_cancels_pending_rpcs_and_closes_channel(self):
        first = await self._start_pending_request(1)
        second = await self._start_pending_request(2)
        await self.client.close()
        results = await asyncio.wait_for(
            asyncio.gather(first, second, return_exceptions=True), timeout=5
        )
        self.assertTrue(
            all(isinstance(result, asyncio.CancelledError) for result in results)
        )
        self.assertEqual(self.channel.get_state(), grpc.ChannelConnectivity.SHUTDOWN)
        self.assertEqual(self.client._channels, {})


class MasterClientBatchPayloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_vit_deadline_uses_existing_request_budget_and_explicit_cap(self):
        for explicit, expected in ((None, 3.0), (0.2, 0.2), (5.0, 3.0)):
            with self.subTest(explicit=explicit):
                client = _CaptureMasterClient()
                await client.get_backend_role_addrs(
                    [], 1024, _FakeInput(), 99, vit_only=True, timeout_s=explicit
                )
                self.assertEqual(client.calls[0]["timeout_s"], expected)
                self.assertEqual(client.calls[0]["request_pb"].generate_timeout, 3000)
                await client.close()

    async def test_vit_deadline_falls_back_to_master_default(self):
        request = _FakeInput()
        request.generate_config.ttft_timeout_ms = -1
        client = _CaptureMasterClient()
        await client.get_backend_role_addrs([], 1024, request, 99, vit_only=True)
        self.assertEqual(client.calls[0]["timeout_s"], 3600.0)
        await client.close()

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

    async def test_omitted_input_stays_omitted_on_slave_fallback(self):
        client = _CaptureMasterClient(
            _FakeHostServiceWithSlave(),
            [None, FlexlbScheduleResponsePB(code=200, success=True)],
        )
        with patch("rtp_llm.server.master_client.trans_input") as build:
            response = await client.get_backend_role_addrs([1], 1024, _FakeInput(), 99)
        self.assertTrue(response.is_ok)
        build.assert_not_called()
        self.assertEqual(
            [c["request_pb"].generate_input for c in client.calls], [b"", b""]
        )

    async def test_provided_input_is_serialized_once_across_slave_fallback(self):
        client = _CaptureMasterClient(
            _FakeHostServiceWithSlave(),
            [None, FlexlbScheduleResponsePB(code=200, success=True)],
        )
        payload = Mock(wraps=_FakeInputPB())
        await client.get_backend_role_addrs(
            [1], 1024, _FakeInput(), 99, input_pb=payload
        )
        payload.SerializeToString.assert_called_once_with()
        self.assertEqual(
            [c["request_pb"].generate_input for c in client.calls],
            [b"serialized-input", b"serialized-input"],
        )

    async def test_vit_only_never_serializes_payload(self):
        client = _CaptureMasterClient()
        payload = Mock(wraps=_FakeInputPB())
        await client.get_backend_role_addrs(
            [1], 1024, _FakeInput(), 99, input_pb=payload, vit_only=True
        )
        payload.SerializeToString.assert_not_called()
        self.assertEqual(client.calls[0]["request_pb"].generate_input, b"")

    async def test_explicit_missing_input_rebuilds_once_without_failure_reporting(self):
        for code, message in (
            (8409, "generate input required"),
            (8406, "missing serialized generate_input for batch dispatch"),
        ):
            for failover in (False, True):
                with self.subTest(code=code, failover=failover):
                    responses = [
                        FlexlbScheduleResponsePB(code=code, error_message=message),
                        FlexlbScheduleResponsePB(code=200, success=True),
                    ]
                    if failover:
                        responses.insert(0, None)
                    client = _CaptureMasterClient(
                        _FakeHostServiceWithSlave(), responses
                    )
                    original = _FakeInput()
                    with patch(
                        "rtp_llm.server.master_client.trans_input",
                        return_value=_FakeInputPB(),
                    ) as build, patch(
                        "rtp_llm.server.master_client.kmonitor.report"
                    ) as report:
                        response = await client.get_backend_role_addrs(
                            [1], 1024, original, 99
                        )
                    self.assertTrue(response.is_ok)
                    build.assert_called_once_with(original)
                    report.assert_not_called()
                    first, second = client.calls[-2:]
                    self.assertEqual(first["addr"], second["addr"])
                    self.assertEqual(first["request_pb"].generate_input, b"")
                    self.assertEqual(
                        second["request_pb"].generate_input, b"serialized-input"
                    )
                    second["request_pb"].ClearField("generate_input")
                    self.assertEqual(first["request_pb"], second["request_pb"])
                    self.assertGreater(second["timeout_s"], 0)
                    self.assertLess(second["timeout_s"], first["timeout_s"])

    async def test_ambiguous_or_admitted_rejection_does_not_rebuild(self):
        with_lifecycle = FlexlbScheduleResponsePB(code=8409)
        with_lifecycle.lifecycle.SetInParent()
        for response in (
            FlexlbScheduleResponsePB(
                code=8406, error_message="another invalid request"
            ),
            FlexlbScheduleResponsePB(code=8409, success=True),
            FlexlbScheduleResponsePB(code=8409, enqueued_by_master=True),
            FlexlbScheduleResponsePB(
                code=8409, server_status=[FlexlbServerStatusPB(role="PREFILL")]
            ),
            with_lifecycle,
        ):
            with self.subTest(response=str(response)):
                client = _CaptureMasterClient(responses=[response])
                with patch("rtp_llm.server.master_client.trans_input") as build:
                    with self.assertRaises(FtRuntimeException):
                        await client.get_backend_role_addrs([1], 1024, _FakeInput(), 99)
                build.assert_not_called()
                self.assertEqual(len(client.calls), 1)

    async def test_provided_input_and_vit_only_do_not_trigger_rebuild(self):
        for kwargs in ({"input_pb": _FakeInputPB()}, {"vit_only": True}):
            with self.subTest(kwargs=kwargs):
                client = _CaptureMasterClient(
                    responses=[FlexlbScheduleResponsePB(code=8409)]
                )
                with patch("rtp_llm.server.master_client.trans_input") as build:
                    with self.assertRaises(FtRuntimeException):
                        await client.get_backend_role_addrs(
                            [1], 1024, _FakeInput(), 99, **kwargs
                        )
                build.assert_not_called()
                self.assertEqual(len(client.calls), 1)

    async def test_repeated_missing_input_rejection_stops_after_one_rebuild(self):
        client = _CaptureMasterClient(
            responses=[FlexlbScheduleResponsePB(code=8409)] * 2
        )
        with patch(
            "rtp_llm.server.master_client.trans_input", return_value=_FakeInputPB()
        ) as build:
            with self.assertRaises(FtRuntimeException) as raised:
                await client.get_backend_role_addrs([1], 1024, _FakeInput(), 99)
        build.assert_called_once()
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            raised.exception.exception_type,
            ExceptionType.MASTER_GENERATE_INPUT_REQUIRED,
        )

    async def test_expired_deadline_does_not_resend(self):
        client = _CaptureMasterClient(responses=[FlexlbScheduleResponsePB(code=8409)])
        clock = SimpleNamespace(
            time=lambda: 0, monotonic=Mock(side_effect=[10.0, 13.1])
        )
        with patch("rtp_llm.server.master_client.time", clock), patch(
            "rtp_llm.server.master_client.trans_input", return_value=_FakeInputPB()
        ):
            with self.assertRaises(FtRuntimeException) as raised:
                await client.get_backend_role_addrs([1], 1024, _FakeInput(), 99)
        self.assertEqual(
            raised.exception.exception_type, ExceptionType.DEADLINE_EXCEEDED
        )
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
