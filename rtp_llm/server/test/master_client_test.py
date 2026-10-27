import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
from rtp_llm.server.master_client import MasterClient


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
