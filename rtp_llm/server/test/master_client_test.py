import asyncio
import os
import unittest
import grpc
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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
    FlexlbScheduleRequestPB,
    FlexlbScheduleResponsePB,
    FlexlbServerStatusPB,
)
from rtp_llm.server.master_client import MasterClient, _flexlb_max_message_bytes


class FlexlbMessageLimitConfigTest(unittest.TestCase):
    def test_default_and_configured_limits(self):
        for value, mib in [
            (None, 512),
            ("", 512),
            ("  ", 512),
            ("256", 256),
            (" 768 ", 768),
            ("1", 1),
            ("2047", 2047),
        ]:
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["FLEXLB_MAX_MESSAGE_SIZE_MB"] = value
                self.assertEqual(_flexlb_max_message_bytes(), mib * 1024 * 1024)

    def test_invalid_limits_fail_fast(self):
        for value in [
            "0",
            "-1",
            "2048",
            "2147483648",
            "1.5",
            "512MiB",
            "1_024",
            "+512",
        ]:
            with self.subTest(value=value), patch.dict(
                os.environ, {"FLEXLB_MAX_MESSAGE_SIZE_MB": value}
            ):
                with self.assertRaisesRegex(ValueError, "FLEXLB_MAX_MESSAGE_SIZE_MB"):
                    _flexlb_max_message_bytes()


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


class MasterClientBatchPayloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_large_image_payload_is_not_limited_to_256_mib(self):
        payload = b"x" * (300 * 1024 * 1024)
        client = _CaptureMasterClient()
        await client.get_backend_role_addrs(
            [],
            512,
            _FakeInput(),
            100,
            input_pb=SimpleNamespace(SerializeToString=lambda: payload),
        )
        self.assertEqual(client.calls[0]["request_pb"].generate_input, payload)

    async def test_oversized_payload_is_rejected_before_rpc(self):
        client = _CaptureMasterClient()
        with patch(
            "rtp_llm.server.master_client.FLEXLB_MAX_MESSAGE_BYTES", 4
        ), self.assertLogs("route_logger", level="WARNING") as logs:
            with self.assertRaises(FtRuntimeException) as error:
                await client.get_backend_role_addrs(
                    [],
                    512,
                    _FakeInput(),
                    100,
                    input_pb=_FakeInputPB(),
                )
        self.assertEqual(error.exception.exception_type, ExceptionType.INVALID_PARAMS)
        self.assertNotIn("FlexLB", str(error.exception))
        self.assertIn("FlexLB", logs.output[0])
        self.assertIn("request_id=100", logs.output[0])
        self.assertIn("limit_bytes=4", logs.output[0])
        self.assertEqual(client.calls, [])

    async def test_channel_uses_configured_limit_for_send_and_receive(self):
        limit = 768 * 1024 * 1024
        with patch(
            "rtp_llm.server.master_client.FLEXLB_MAX_MESSAGE_BYTES", limit
        ), patch("rtp_llm.server.master_client.grpc.aio.insecure_channel") as channel:
            client = _CaptureMasterClient()
            client._get_channel("master:1234")
        options = dict(channel.call_args.kwargs["options"])
        self.assertEqual(options["grpc.max_send_message_length"], limit)
        self.assertEqual(options["grpc.max_receive_message_length"], limit)

    async def test_message_limit_reserves_actual_follower_forwarding_overhead(self):
        with patch("rtp_llm.server.master_client.time.time", return_value=1234567890):
            client = _CaptureMasterClient()
            await client.get_backend_role_addrs(
                [], 512, _FakeInput(), 100, input_pb=_FakeInputPB()
            )
            request = client.calls[0]["request_pb"]
            request.forward_hop = 1
            forwarded_size = request.ByteSize()
            for limit in (forwarded_size, forwarded_size - 1):
                with self.subTest(limit=limit), patch(
                    "rtp_llm.server.master_client.FLEXLB_MAX_MESSAGE_BYTES", limit
                ):
                    candidate = _CaptureMasterClient()
                    if limit == forwarded_size:
                        await candidate.get_backend_role_addrs(
                            [], 512, _FakeInput(), 100, input_pb=_FakeInputPB()
                        )
                        self.assertEqual(len(candidate.calls), 1)
                    else:
                        with self.assertRaises(FtRuntimeException):
                            await candidate.get_backend_role_addrs(
                                [], 512, _FakeInput(), 100, input_pb=_FakeInputPB()
                            )
                        self.assertEqual(candidate.calls, [])

    async def test_resource_exhausted_does_not_enable_domain_fallback(self):
        client = _CaptureMasterClient()
        client._get_channel = Mock()
        stub = SimpleNamespace(
            Schedule=AsyncMock(
                side_effect=grpc.aio.AioRpcError(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    grpc.aio.Metadata(),
                    grpc.aio.Metadata(),
                    "message too large",
                )
            )
        )
        with patch("rtp_llm.server.master_client.FlexlbServiceStub", return_value=stub):
            with self.assertRaises(FtRuntimeException) as error:
                await MasterClient._send_schedule_request(
                    client,
                    "master:1234",
                    FlexlbScheduleRequestPB(request_id=100),
                    1.0,
                    100,
                )
        self.assertEqual(
            error.exception.exception_type, ExceptionType.TRAFFIC_LIMIT_ERROR
        )

    async def test_vit_selection_has_no_enqueue_payload_or_pd_queue_state(self):
        client = _CaptureMasterClient()
        client.latest_queue_length = 9
        client._send_schedule_request = AsyncMock(
            return_value=FlexlbScheduleResponsePB(
                code=200,
                server_status=[
                    FlexlbServerStatusPB(role="VIT", server_ip="vit", grpc_port=8011)
                ],
            )
        )
        response = await client.get_backend_role_addrs(
            [],
            256,
            _FakeInput(),
            101,
            input_pb=_FakeInputPB(),
            vit_only=True,
            timeout_s=0.25,
        )
        self.assertTrue(response.is_ok)
        request = client._send_schedule_request.await_args.args[1]
        self.assertTrue(request.vit_only)
        self.assertFalse(request.generate_input)
        self.assertEqual(request.request_id, 101)
        self.assertEqual(request.generate_timeout, 250)
        self.assertEqual(client.latest_queue_length, 9)

    async def test_vit_cancellation_does_not_cancel_later_pd_schedule(self):
        client = _CaptureMasterClient()
        stub = SimpleNamespace(
            Schedule=AsyncMock(side_effect=asyncio.CancelledError()),
            Cancel=AsyncMock(),
        )
        client._get_channel = Mock()
        with patch("rtp_llm.server.master_client.FlexlbServiceStub", return_value=stub):
            with self.assertRaises(asyncio.CancelledError):
                await MasterClient._send_schedule_request(
                    client,
                    "master:1234",
                    FlexlbScheduleRequestPB(request_id=101, vit_only=True),
                    1.0,
                    101,
                )
            stub.Cancel.assert_not_awaited()
            with self.assertRaises(asyncio.CancelledError):
                await MasterClient._send_schedule_request(
                    client,
                    "master:1234",
                    FlexlbScheduleRequestPB(request_id=101),
                    1.0,
                    101,
                )
            stub.Cancel.assert_awaited_once()

    async def test_vit_selection_rejects_master_enqueue_response(self):
        client = _CaptureMasterClient()
        with self.assertRaises(FtRuntimeException):
            await client.get_backend_role_addrs(
                [], 256, _FakeInput(), 101, vit_only=True
            )

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
