import asyncio
import unittest
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
from rtp_llm.server.master_client import MasterClient


class _FakeMasterConfig:
    master_max_connect_pool_size = 4
    master_session_timeout_s = 1
    master_default_timeout_ms = 3600000


class _FakeHostService:
    def __init__(self, requirements=None):
        self.requirements = requirements or {}

    def requires_generate_input(self, addr):
        return self.requirements.get(addr)

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
            input_pb_factory=_FakeInputPB,
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

    async def test_selected_vit_preserves_optional_logical_engine_index(self):
        for engine_index in (None, 0, 1):
            with self.subTest(engine_index=engine_index):
                client = _CaptureMasterClient()
                await client.get_backend_role_addrs(
                    block_cache_keys=[],
                    cache_key_block_size=1024,
                    input=_FakeInput(),
                    request_id=99,
                    selected_vit={
                        "role": "vit",
                        "server_ip": "10.0.0.1",
                        "http_port": 8000,
                        "grpc_port": 8001,
                        "worker_generation": 42,
                        "engine_index": engine_index,
                    },
                )
                selected = client.calls[0]["request_pb"].selected_vit
                self.assertEqual(
                    selected.HasField("engine_index"), engine_index is not None
                )
                if engine_index is not None:
                    self.assertEqual(selected.engine_index, engine_index)
                self.assertEqual(selected.worker_generation, 42)

    async def test_vit_response_keeps_engine_index_for_subsequent_route(self):
        for engine_index in (None, 0, 1):
            with self.subTest(engine_index=engine_index):
                client = _CaptureMasterClient()
                status = FlexlbServerStatusPB(
                    role="VIT",
                    server_ip="10.0.0.1",
                    http_port=8000,
                    grpc_port=8001,
                    worker_generation=42,
                )
                if engine_index is not None:
                    status.engine_index = engine_index
                client._send_schedule_request = AsyncMock(
                    return_value=FlexlbScheduleResponsePB(
                        success=True,
                        code=200,
                        server_status=[status],
                    )
                )
                response = await client.get_backend_role_addrs(
                    block_cache_keys=[],
                    cache_key_block_size=1024,
                    input=_FakeInput(),
                    request_id=99,
                    vit_only=True,
                )
                selected = response.result["server_status"][0]
                self.assertEqual(selected["engine_index"], engine_index)
                await client.get_backend_role_addrs(
                    block_cache_keys=[],
                    cache_key_block_size=1024,
                    input=_FakeInput(),
                    request_id=99,
                    selected_vit=selected,
                )
                request = client._send_schedule_request.call_args.args[1]
                self.assertEqual(
                    request.selected_vit.HasField("engine_index"),
                    engine_index is not None,
                )
                if engine_index is not None:
                    self.assertEqual(request.selected_vit.engine_index, engine_index)

    async def test_schedule_payload_priority_from_qos_header(self):
        client = _CaptureMasterClient()

        await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=_FakeInput(headers={"x-dashscope-inner-qos-level": "70"}),
            request_id=101,
            input_pb_factory=_FakeInputPB,
        )

        self.assertEqual(client.calls[0]["request_pb"].priority, 70)

    async def test_schedule_payload_priority_defaults_when_header_missing(self):
        client = _CaptureMasterClient()

        await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=_FakeInput(headers={}),
            request_id=102,
            input_pb_factory=_FakeInputPB,
        )

        self.assertEqual(client.calls[0]["request_pb"].priority, 50)

    async def test_schedule_payload_priority_invalid_header_no_raise(self):
        client = _CaptureMasterClient()

        response = await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=_FakeInput(headers={"x-dashscope-inner-qos-level": "high"}),
            request_id=103,
            input_pb_factory=_FakeInputPB,
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
                input_pb_factory=_FakeInputPB,
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
                        input_pb_factory=_FakeInputPB,
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
                input_pb_factory=_FakeInputPB,
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
                input_pb_factory=_FakeInputPB,
            )

        self.assertEqual(
            AdmissionRejectReason.INVALID,
            raised.exception.admission_reject_reason,
        )

    async def test_non_batch_omits_input_without_constructing_it(self):
        client = _CaptureMasterClient(
            _FakeHostService({"master:1234": False}),
            [FlexlbScheduleResponsePB(success=True, code=200)],
        )
        factory = Mock(side_effect=AssertionError("NON_BATCH built the payload"))
        selected_vit = {"role": "VIT", "server_ip": "10.0.0.8", "grpc_port": 9000}

        response = await client.get_backend_role_addrs(
            block_cache_keys=[1, 2, 3],
            cache_key_block_size=1024,
            input=_FakeInput(headers={"x-dashscope-inner-qos-level": "70"}),
            request_id=99,
            input_pb_factory=factory,
            cache_affinity_keys=["image-key"],
            selected_vit=selected_vit,
            seq_len=42,
        )

        factory.assert_not_called()
        self.assertTrue(response.is_ok)
        self.assertFalse(response.enqueued_by_master)
        request = client.calls[0]["request_pb"]
        self.assertEqual(request.generate_input, b"")
        self.assertEqual(request.request_id, 99)
        self.assertEqual(list(request.block_cache_keys), [1, 2, 3])
        self.assertEqual(request.cache_key_block_size, 1024)
        self.assertEqual(request.seq_len, 42)
        self.assertEqual(request.max_new_tokens, 17)
        self.assertEqual(request.num_beams, 2)
        self.assertTrue(request.force_disable_sp_run)
        self.assertEqual(request.priority, 70)
        self.assertEqual(list(request.cache_affinity_keys), ["image-key"])
        self.assertEqual(request.selected_vit.server_ip, "10.0.0.8")

    async def test_batch_and_unknown_capabilities_build_payload_once(self):
        for required in (True, None):
            with self.subTest(required=required):
                client = _CaptureMasterClient(
                    _FakeHostService({"master:1234": required})
                )
                payload = Mock(wraps=_FakeInputPB())
                factory = Mock(return_value=payload)

                await client.get_backend_role_addrs(
                    [1], 1024, _FakeInput(), 99, input_pb_factory=factory
                )

                factory.assert_called_once_with()
                payload.SerializeToString.assert_called_once_with()
                self.assertEqual(
                    client.calls[0]["request_pb"].generate_input, b"serialized-input"
                )

    async def test_vit_only_never_constructs_payload(self):
        client = _CaptureMasterClient(_FakeHostService({"master:1234": True}))
        factory = Mock(side_effect=AssertionError("ViT route built the payload"))

        await client.get_backend_role_addrs(
            [1], 1024, _FakeInput(), 99, input_pb_factory=factory, vit_only=True
        )

        factory.assert_not_called()
        self.assertEqual(client.calls[0]["request_pb"].generate_input, b"")
        self.assertTrue(client.calls[0]["request_pb"].vit_route_only)

    async def test_slave_uses_its_own_capability_and_reuses_serialized_payload(self):
        for master_required, slave_required in (
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(master=master_required, slave=slave_required):
                client = _CaptureMasterClient(
                    _FakeHostServiceWithSlave(
                        {
                            "master:1234": master_required,
                            "slave:1234": slave_required,
                        }
                    ),
                    [None, FlexlbScheduleResponsePB(success=True, code=200)],
                )
                payload = Mock(wraps=_FakeInputPB())
                factory = Mock(return_value=payload)

                response = await client.get_backend_role_addrs(
                    [1], 1024, _FakeInput(), 99, input_pb_factory=factory
                )

                self.assertTrue(response.is_ok)
                self.assertEqual(
                    [call["addr"] for call in client.calls],
                    ["master:1234", "slave:1234"],
                )
                self.assertEqual(
                    [call["request_pb"].generate_input for call in client.calls],
                    [
                        b"serialized-input" if master_required else b"",
                        b"serialized-input" if slave_required else b"",
                    ],
                )
                factory.assert_called_once_with()
                payload.SerializeToString.assert_called_once_with()

    async def test_missing_payload_rejection_resends_once_without_failure_reporting(
        self,
    ):
        for code, message in (
            (8409, "generate input required"),
            (8406, "missing serialized generate_input for batch dispatch"),
        ):
            with self.subTest(code=code):
                client = _CaptureMasterClient(
                    _FakeHostService({"master:1234": False}),
                    [
                        FlexlbScheduleResponsePB(code=code, error_message=message),
                        FlexlbScheduleResponsePB(
                            success=True, code=200, enqueued_by_master=True
                        ),
                    ],
                )
                payload = Mock(wraps=_FakeInputPB())
                factory = Mock(return_value=payload)
                with (
                    patch("rtp_llm.server.master_client.kmonitor.report") as report,
                    patch("rtp_llm.server.master_client.route_logger.error") as error,
                ):
                    response = await client.get_backend_role_addrs(
                        [1], 1024, _FakeInput(), 99, input_pb_factory=factory
                    )

                self.assertTrue(response.is_ok)
                self.assertTrue(response.enqueued_by_master)
                factory.assert_called_once_with()
                payload.SerializeToString.assert_called_once_with()
                report.assert_not_called()
                error.assert_not_called()
                self.assertEqual(
                    [call["addr"] for call in client.calls], ["master:1234"] * 2
                )
                first, second = (call["request_pb"] for call in client.calls)
                self.assertEqual(first.generate_input, b"")
                self.assertEqual(second.generate_input, b"serialized-input")
                second.ClearField("generate_input")
                self.assertEqual(first, second)

    async def test_payload_resend_uses_remaining_timeout(self):
        class SlowFirstResponseClient(_CaptureMasterClient):
            async def _send_schedule_request(self, *args):
                if not self.calls:
                    await asyncio.sleep(0.02)
                return await super()._send_schedule_request(*args)

        client = SlowFirstResponseClient(
            _FakeHostService({"master:1234": False}),
            [
                FlexlbScheduleResponsePB(code=8409),
                FlexlbScheduleResponsePB(success=True, code=200),
            ],
        )
        await client.get_backend_role_addrs(
            [1], 1024, _FakeInput(), 99, input_pb_factory=_FakeInputPB
        )

        self.assertEqual(len(client.calls), 2)
        first_timeout, second_timeout = (call["timeout_s"] for call in client.calls)
        self.assertGreater(second_timeout, 0)
        self.assertLess(second_timeout, first_timeout - 0.01)

    async def test_payload_rejection_after_deadline_does_not_send_again(self):
        client = _CaptureMasterClient(
            _FakeHostServiceWithSlave({"master:1234": False}),
            [FlexlbScheduleResponsePB(code=8409)],
        )
        clock = SimpleNamespace(
            time=lambda: 0,
            monotonic=Mock(side_effect=[10.0, 13.1]),
        )

        with patch("rtp_llm.server.master_client.time", clock):
            with self.assertRaises(FtRuntimeException) as raised:
                await client.get_backend_role_addrs(
                    [1], 1024, _FakeInput(), 99, input_pb_factory=_FakeInputPB
                )

        self.assertEqual(
            raised.exception.exception_type, ExceptionType.DEADLINE_EXCEEDED
        )
        self.assertEqual([call["addr"] for call in client.calls], ["master:1234"])
        self.assertEqual(client.calls[0]["request_pb"].generate_input, b"")

    async def test_ambiguous_or_admitted_rejection_does_not_resend_payload(self):
        with_lifecycle = FlexlbScheduleResponsePB(code=8409)
        with_lifecycle.lifecycle.SetInParent()
        responses = (
            FlexlbScheduleResponsePB(
                code=8406, error_message="another invalid request"
            ),
            FlexlbScheduleResponsePB(code=8408),
            FlexlbScheduleResponsePB(code=8409, success=True),
            FlexlbScheduleResponsePB(code=8409, enqueued_by_master=True),
            FlexlbScheduleResponsePB(
                code=8409, server_status=[FlexlbServerStatusPB(role="PREFILL")]
            ),
            with_lifecycle,
        )
        for rejected in responses:
            with self.subTest(response=str(rejected)):
                client = _CaptureMasterClient(
                    _FakeHostService({"master:1234": False}), [rejected]
                )
                factory = Mock(return_value=_FakeInputPB())
                with self.assertRaises(FtRuntimeException):
                    await client.get_backend_role_addrs(
                        [1], 1024, _FakeInput(), 99, input_pb_factory=factory
                    )

                self.assertEqual(len(client.calls), 1)
                factory.assert_not_called()

    async def test_missing_payload_rejection_with_payload_already_sent_does_not_retry(
        self,
    ):
        client = _CaptureMasterClient(
            _FakeHostService({"master:1234": True}),
            [FlexlbScheduleResponsePB(code=8409)],
        )
        factory = Mock(return_value=_FakeInputPB())

        with self.assertRaises(FtRuntimeException):
            await client.get_backend_role_addrs(
                [1], 1024, _FakeInput(), 99, input_pb_factory=factory
            )

        self.assertEqual(len(client.calls), 1)
        factory.assert_called_once_with()

    async def test_missing_payload_rejection_without_factory_does_not_retry(self):
        client = _CaptureMasterClient(
            _FakeHostService({"master:1234": False}),
            [FlexlbScheduleResponsePB(code=8409)],
        )

        with self.assertRaises(FtRuntimeException) as raised:
            await client.get_backend_role_addrs([1], 1024, _FakeInput(), 99)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            raised.exception.exception_type,
            ExceptionType.MASTER_GENERATE_INPUT_REQUIRED,
        )

    async def test_repeated_payload_rejection_is_not_retried_again(self):
        client = _CaptureMasterClient(
            _FakeHostService({"master:1234": False}),
            [FlexlbScheduleResponsePB(code=8409)] * 2,
        )
        factory = Mock(return_value=_FakeInputPB())

        with self.assertRaises(FtRuntimeException):
            await client.get_backend_role_addrs(
                [1], 1024, _FakeInput(), 99, input_pb_factory=factory
            )

        self.assertEqual(len(client.calls), 2)
        factory.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
