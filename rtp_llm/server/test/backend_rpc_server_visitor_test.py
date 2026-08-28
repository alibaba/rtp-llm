import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics
from rtp_llm.server.backend_rpc_server_visitor import (
    BackendRPCServerVisitor,
    get_role_names,
)
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.master_client import FlexlbResponse, MasterClient


class _FakeTokenIds:
    shape = (3,)


class _FakeGenerateConfig:
    def __init__(
        self,
        is_streaming=False,
        calculate_loss=0,
        return_hidden_states=False,
        return_all_hidden_states=False,
    ):
        self.role_addrs = []
        self.is_streaming = is_streaming
        self.max_new_tokens = 16
        self.num_beams = 1
        self.force_disable_sp_run = False
        self.calculate_loss = calculate_loss
        self.return_hidden_states = return_hidden_states
        self.return_all_hidden_states = return_all_hidden_states

    def validate(self):
        return None

    def model_copy(self, update=None):
        copied = _FakeGenerateConfig(self.is_streaming)
        copied.role_addrs = list(self.role_addrs)
        for key, value in (update or {}).items():
            setattr(copied, key, value)
        return copied


@dataclass
class _FakeInput:
    generate_config: _FakeGenerateConfig = field(default_factory=_FakeGenerateConfig)
    request_id: int = 123
    token_ids: _FakeTokenIds = field(default_factory=_FakeTokenIds)
    headers = None
    enqueued_by_master: bool = False
    prompt_length: int = 17

    def __init__(
        self,
        is_streaming=False,
        generate_config=None,
        request_id=123,
        token_ids=None,
        enqueued_by_master=False,
        prompt_length=17,
        **generate_config_kwargs,
    ):
        if generate_config is not None:
            self.generate_config = generate_config
        elif isinstance(is_streaming, _FakeGenerateConfig):
            self.generate_config = is_streaming
        else:
            self.generate_config = _FakeGenerateConfig(
                is_streaming=is_streaming, **generate_config_kwargs
            )
        self.request_id = request_id
        self.token_ids = token_ids or _FakeTokenIds()
        self.headers = None
        self.enqueued_by_master = enqueued_by_master
        self.prompt_length = prompt_length


class _FakeRouteTokenIds:
    shape = (3,)

    def tolist(self):
        return [1, 2, 3]


class _FakeRouteInput:
    request_id = 456
    token_ids = _FakeRouteTokenIds()
    prompt_length = 3

    def __init__(self):
        self.generate_config = _FakeGenerateConfig()
        self.enqueued_by_master = False


class _FakeHostService:
    service_available = False

    def get_master_addr(self):
        return "master:1234"


class _FakeMasterConfig:
    master_default_timeout_ms = 3000


class _FakeInputPB:
    def SerializeToString(self):
        return b"serialized-input"


class _FakeMasterClient:
    def __init__(self):
        self.calls = []

    async def get_backend_role_addrs(
        self,
        block_cache_keys,
        cache_key_block_size,
        input,
        request_id,
        input_pb=None,
    ):
        self.calls.append(
            {
                "block_cache_keys": block_cache_keys,
                "input": input,
                "request_id": request_id,
                "input_pb": input_pb,
            }
        )
        return FlexlbResponse.ok(["prefill-role"], enqueued_by_master=True)


class BackendRPCServerVisitorRouteCacheKeysTest(unittest.TestCase):
    def test_get_role_names(self):
        role_addrs = [
            RoleAddr(role=RoleType.PREFILL, ip="127.0.0.1", http_port=1, grpc_port=2),
            RoleAddr(role=RoleType.DECODE, ip="127.0.0.2", http_port=3, grpc_port=4),
        ]

        self.assertEqual(get_role_names(role_addrs), {"PREFILL", "DECODE"})

    def test_route_cache_keys_passthrough_when_page_rr_disabled(self):
        self.assertEqual(
            route_cache_keys_for_page_rr([10, 11, 12, 13], False, 4),
            [10, 11, 12, 13],
        )

    def test_route_cache_keys_use_last_rank_canonical_keys_under_page_rr(self):
        self.assertEqual(
            route_cache_keys_for_page_rr(
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], True, 4
            ),
            [13, 17, 21],
        )

    def test_route_cache_keys_short_prompt_has_no_complete_virtual_block(self):
        self.assertEqual(route_cache_keys_for_page_rr([10, 11, 12], True, 4), [])

    def test_cache_key_block_size_tracks_routed_key_granularity(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.seq_size_per_block = 256
        visitor._page_rr_route_cache_keys = False
        visitor._page_rr_cp_size = 4
        self.assertEqual(visitor._cache_key_block_size(), 256)

        visitor._page_rr_route_cache_keys = True
        self.assertEqual(visitor._cache_key_block_size(), 1024)


class BackendRPCServerVisitorRouteIpsTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _master_route_visitor(master_client):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.seq_size_per_block = 16
        visitor.master_client = master_client
        visitor._route_cache_keys = lambda keys: keys
        visitor._report_recent_cache_key_metrics = lambda keys: None
        visitor._page_rr_route_cache_keys = False
        visitor._page_rr_cp_size = 1
        return visitor

    async def test_get_master_route_addrs_passes_pb_and_marks_master_enqueue(self):
        visitor = self._master_route_visitor(_FakeMasterClient())

        input = _FakeRouteInput()

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.get_block_cache_keys",
            return_value=[11, 22],
        ), patch(
            "rtp_llm.server.backend_rpc_server_visitor.trans_input",
            return_value=_FakeInputPB(),
        ), patch(
            "rtp_llm.server.backend_rpc_server_visitor.kmonitor"
        ) as mock_kmonitor:
            result = await visitor.get_master_route_addrs(input)

        self.assertIsNone(result)
        self.assertEqual(input.generate_config.role_addrs, ["prefill-role"])
        self.assertTrue(input.enqueued_by_master)
        self.assertEqual(visitor.master_client.calls[0]["block_cache_keys"], [11, 22])
        self.assertEqual(visitor.master_client.calls[0]["request_id"], 456)
        self.assertEqual(
            visitor.master_client.calls[0]["input_pb"].SerializeToString(),
            b"serialized-input",
        )
        mock_kmonitor.report.assert_called_once_with(
            AccMetrics.MASTER_ROUTE_QPS_METRIC, 1
        )

    async def test_non_200_master_response_reports_route_error_once(self):
        master_client = MasterClient(
            host_service=_FakeHostService(), master_config=_FakeMasterConfig()
        )
        master_client._send_schedule_request = AsyncMock(
            return_value=SimpleNamespace(
                code=int(ExceptionType.PRIORITY_ADMISSION_REJECTED),
                error_message="same-priority request ahead",
                admission_reject_reason=int(AdmissionRejectReason.SAME_PRIORITY_AHEAD),
                queue_length=7,
            )
        )
        visitor = self._master_route_visitor(master_client)
        input = _FakeRouteInput()

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.get_block_cache_keys",
            return_value=[11, 22],
        ), patch(
            "rtp_llm.server.backend_rpc_server_visitor.trans_input",
            return_value=_FakeInputPB(),
        ), patch(
            "rtp_llm.metrics.kmonitor.report"
        ) as report:
            with self.assertRaises(FtRuntimeException) as ctx:
                await visitor.get_master_route_addrs(input)

        self.assertEqual(
            ctx.exception.exception_type,
            ExceptionType.PRIORITY_ADMISSION_REJECTED,
        )
        report.assert_called_once_with(
            AccMetrics.MASTER_ROUTE_ERROR_QPS_METRIC,
            1,
            {"error_code": "8430_PRIORITY_ADMISSION_REJECTED"},
        )

    async def test_master_connection_failure_reports_route_error_once(self):
        master_client = MasterClient(
            host_service=_FakeHostService(), master_config=_FakeMasterConfig()
        )
        master_client._send_schedule_request = AsyncMock(return_value=None)
        visitor = self._master_route_visitor(master_client)

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.get_block_cache_keys",
            return_value=[11, 22],
        ), patch(
            "rtp_llm.server.backend_rpc_server_visitor.trans_input",
            return_value=_FakeInputPB(),
        ), patch(
            "rtp_llm.metrics.kmonitor.report"
        ) as report:
            result = await visitor.get_master_route_addrs(_FakeRouteInput())

        self.assertTrue(result.connection_failed)
        report.assert_called_once_with(
            AccMetrics.MASTER_ROUTE_ERROR_QPS_METRIC,
            1,
            {"error_code": "8201_GET_CONNECTION_FAILED"},
        )

    async def test_master_exceptions_report_once_and_propagate_unchanged(self):
        cases = (
            (
                FtRuntimeException(
                    ExceptionType.DEADLINE_EXCEEDED, "schedule deadline exceeded"
                ),
                "8204_DEADLINE_EXCEEDED",
            ),
            (RuntimeError("unexpected master client failure"), "514_UNKNOWN_ERROR"),
        )
        for error, expected_error_code in cases:
            with self.subTest(error=type(error).__name__):
                master_client = MasterClient(
                    host_service=_FakeHostService(), master_config=_FakeMasterConfig()
                )
                master_client._send_schedule_request = AsyncMock(side_effect=error)
                visitor = self._master_route_visitor(master_client)

                with patch(
                    "rtp_llm.server.backend_rpc_server_visitor.get_block_cache_keys",
                    return_value=[11, 22],
                ), patch(
                    "rtp_llm.server.backend_rpc_server_visitor.trans_input",
                    return_value=_FakeInputPB(),
                ), patch(
                    "rtp_llm.metrics.kmonitor.report"
                ) as report:
                    with self.assertRaises(type(error)) as ctx:
                        await visitor.get_master_route_addrs(_FakeRouteInput())

                self.assertIs(ctx.exception, error)
                report.assert_called_once_with(
                    AccMetrics.MASTER_ROUTE_ERROR_QPS_METRIC,
                    1,
                    {"error_code": expected_error_code},
                )

    async def test_route_ips_preserves_master_route_error_code_on_route_error(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.master_config = None
        visitor.host_service = _FakeHostService()
        visitor.backend_role_list = ["PREFILL"]

        async def get_master_route_addrs(_input):
            return FlexlbResponse.error_response(
                int(ExceptionType.MASTER_NO_AVAILABLE_WORKER), "no worker"
            )

        visitor.get_master_route_addrs = get_master_route_addrs

        with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
            with self.assertRaises(FtRuntimeException) as ctx:
                await visitor.route_ips(_FakeInput())

        self.assertEqual(ctx.exception.exception_type, ExceptionType.ROUTE_ERROR)
        self.assertEqual(
            ctx.exception.rtp_error_code,
            int(ExceptionType.MASTER_NO_AVAILABLE_WORKER),
        )

    async def test_route_ips_falls_back_on_master_connection_failure(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.master_config = None
        visitor.host_service = _FakeHostService()
        visitor.backend_role_list = ["PREFILL"]
        domain_route_called = False

        async def get_master_route_addrs(_input):
            return FlexlbResponse.connection_failed_response()

        async def get_domain_route_addrs(input):
            nonlocal domain_route_called
            domain_route_called = True
            input.generate_config.role_addrs.append("domain-role")

        visitor.get_master_route_addrs = get_master_route_addrs
        visitor.get_domain_route_addrs = get_domain_route_addrs
        input = _FakeInput()

        with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
            await visitor.route_ips(input)

        self.assertTrue(domain_route_called)
        self.assertEqual(input.generate_config.role_addrs, ["domain-role"])


class _RetryingModelRpcClient:
    def __init__(self):
        self.attempts = 0
        self.inputs = []

    async def enqueue(self, input):
        self.attempts += 1
        self.inputs.append(input)
        attempt = self.attempts
        if attempt == 1:
            yield "partial-output-from-failed-attempt"
            raise RuntimeError("StatusCode.UNAVAILABLE recvmsg:Connection timed out")
        yield "successful-output"


class _SuccessfulModelRpcClient:
    def __init__(self, outputs):
        self.outputs = outputs
        self.attempts = 0

    async def enqueue(self, _input):
        self.attempts += 1
        for output in self.outputs:
            yield output


class _AlwaysFailingModelRpcClient:
    def __init__(self, error):
        self.error = error
        self.attempts = 0

    async def enqueue(self, _input):
        self.attempts += 1
        yield "partial-output-from-failed-attempt"
        raise self.error


class _EscalatingErrorModelRpcClient:
    """Raises a CAPACITY FtRuntimeException on the first attempt, then a
    non-retryable RuntimeError on the second attempt.

    Verifies that stream_with_aux_info re-raises the ORIGINAL exception
    (CAPACITY) after a retry encounters a different, non-retryable error,
    so the caller sees the correct error category (429, not 500)."""

    def __init__(self):
        self.attempts = 0

    async def enqueue(self, _input):
        self.attempts += 1
        # Production ModelRpcClient.enqueue() is an async generator. Keep this
        # fake's call shape identical so failures are raised while iterating.
        if False:
            yield None
        if self.attempts == 1:
            raise FtRuntimeException(
                ExceptionType.MASTER_NO_AVAILABLE_WORKER,
                "no available worker",
            )
        raise RuntimeError("unexpected downstream error")


class _CapacityThenPreemptedModelRpcClient:
    def __init__(self):
        self.attempts = 0

    async def enqueue(self, _input):
        self.attempts += 1
        # Keep the fake aligned with ModelRpcClient.enqueue(), which returns an
        # async iterator rather than an awaitable coroutine.
        if False:
            yield None
        if self.attempts == 1:
            raise FtRuntimeException(
                ExceptionType.MASTER_NO_AVAILABLE_WORKER,
                "no available worker",
            )
        raise FtRuntimeException(
            ExceptionType.PRIORITY_PREEMPTED,
            "preempted by higher-priority request",
        )


class _CapacityThenBatchSloExpiredModelRpcClient:
    def __init__(self):
        self.attempts = 0

    async def enqueue(self, _input):
        self.attempts += 1
        if False:
            yield None
        if self.attempts == 1:
            raise FtRuntimeException(
                ExceptionType.MASTER_NO_AVAILABLE_WORKER,
                "no available worker",
            )
        raise FtRuntimeException(
            ExceptionType.BATCH_SLO_EXPIRED,
            "admission deadline exceeded",
        )


class BackendRPCServerVisitorRetryTest(unittest.IsolatedAsyncioTestCase):
    def _visitor(self, model_rpc_client) -> BackendRPCServerVisitor:
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 1024
        visitor.model_rpc_client = model_rpc_client
        visitor.host_service = _FakeHostService()
        visitor.pd_route_retry_on_unavailable = 3
        visitor._prefill_cp_active = False
        visitor.request_id_factory = None
        visitor.fill_request_info = lambda _input: None
        visitor.check_sp_supported = lambda _input: None
        return visitor

    async def test_prefill_cp_rejects_full_sequence_outputs_before_rpc(self):
        client = _SuccessfulModelRpcClient(["unexpected-output"])
        visitor = self._visitor(client)
        visitor._prefill_cp_active = True

        for option in ("calculate_loss", "return_all_hidden_states"):
            request = _FakeInput(**{option: True})
            with self.assertRaisesRegex(
                FtRuntimeException,
                f"prefill context parallelism does not support request option\\(s\\): {option}",
            ) as ctx:
                await visitor.enqueue(request)
            self.assertEqual(ctx.exception.exception_type, ExceptionType.INVALID_PARAMS)

        self.assertEqual(client.attempts, 0)

    def test_prefill_cp_allows_return_hidden_states(self):
        visitor = self._visitor(_SuccessfulModelRpcClient([]))
        visitor._prefill_cp_active = True

        visitor.check_prefill_cp_supported(_FakeInput(return_hidden_states=True))

    async def test_non_streaming_discards_partial_attempt_before_retry(self):
        client = _RetryingModelRpcClient()
        visitor = self._visitor(client)
        input = _FakeInput(_FakeGenerateConfig(is_streaming=False))
        visitor.set_request_id_factory(lambda: 456)

        stream = await visitor.enqueue(input)
        outputs = [output async for output in stream]

        self.assertEqual(outputs, ["successful-output"])
        self.assertEqual(client.attempts, 2)
        self.assertEqual([item.request_id for item in client.inputs], [123, 456])
        self.assertIs(client.inputs[0], input)
        self.assertIsNot(client.inputs[1], input)
        self.assertIsNot(client.inputs[1].generate_config, input.generate_config)
        self.assertIs(client.inputs[1].token_ids, input.token_ids)
        self.assertEqual(input.request_id, 123)

    async def test_non_streaming_replays_successful_outputs_in_order(self):
        client = _SuccessfulModelRpcClient(["first-output", "second-output"])
        visitor = self._visitor(client)

        stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(False)))
        outputs = [output async for output in stream]

        self.assertEqual(outputs, ["first-output", "second-output"])
        self.assertEqual(client.attempts, 1)

    async def test_non_streaming_raises_after_retry_budget_exhausted(self):
        client = _AlwaysFailingModelRpcClient(
            RuntimeError("StatusCode.UNAVAILABLE recvmsg:Connection timed out")
        )
        visitor = self._visitor(client)
        visitor.pd_route_retry_on_unavailable = 1
        visitor.set_request_id_factory(lambda: 456)

        stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(False)))
        outputs = []
        with self.assertRaisesRegex(RuntimeError, "StatusCode.UNAVAILABLE"):
            async for output in stream:
                outputs.append(output)

        self.assertEqual(outputs, [])
        self.assertEqual(client.attempts, 2)

    async def test_retry_without_request_id_factory_is_disabled(self):
        client = _RetryingModelRpcClient()
        visitor = self._visitor(client)

        stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(False)))
        with self.assertRaisesRegex(RuntimeError, "StatusCode.UNAVAILABLE"):
            [output async for output in stream]

        self.assertEqual(client.attempts, 1)

    async def test_non_streaming_non_retryable_error_does_not_retry(self):
        client = _AlwaysFailingModelRpcClient(ValueError("bad output"))
        visitor = self._visitor(client)

        stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(False)))
        outputs = []
        with self.assertRaisesRegex(ValueError, "bad output"):
            async for output in stream:
                outputs.append(output)

        self.assertEqual(outputs, [])
        self.assertEqual(client.attempts, 1)

    async def test_streaming_does_not_retry_after_partial_output_yielded(self):
        client = _RetryingModelRpcClient()
        visitor = self._visitor(client)

        stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(True)))
        outputs = []
        with self.assertRaisesRegex(RuntimeError, "StatusCode.UNAVAILABLE"):
            async for output in stream:
                outputs.append(output)

        self.assertEqual(outputs, ["partial-output-from-failed-attempt"])
        self.assertEqual(client.attempts, 1)

    async def test_retry_preserves_original_capacity_exception(self):
        """When a retryable CAPACITY error (e.g. MASTER_NO_AVAILABLE_WORKER)
        triggers a retry and the next attempt hits a different, non-retryable
        error, the ORIGINAL exception must be re-raised so the caller maps
        it to 429, not 500."""
        client = _EscalatingErrorModelRpcClient()
        visitor = self._visitor(client)
        visitor.set_request_id_factory(lambda: 456)

        stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(False)))
        with self.assertRaises(FtRuntimeException) as ctx:
            [output async for output in stream]

        self.assertEqual(
            ctx.exception.exception_type,
            ExceptionType.MASTER_NO_AVAILABLE_WORKER,
        )
        self.assertEqual(client.attempts, 2)

    async def test_priority_preempted_is_terminal_and_never_retried(self):
        client = _AlwaysFailingModelRpcClient(
            FtRuntimeException(
                ExceptionType.PRIORITY_PREEMPTED,
                "preempted by higher-priority request",
            )
        )
        visitor = self._visitor(client)
        visitor.set_request_id_factory(lambda: 456)

        stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(False)))
        with self.assertRaises(FtRuntimeException) as ctx:
            [output async for output in stream]

        self.assertEqual(
            ctx.exception.exception_type,
            ExceptionType.PRIORITY_PREEMPTED,
        )
        self.assertEqual(client.attempts, 1)

    async def test_batch_slo_expired_is_terminal_and_keeps_request_identity(self):
        client = _AlwaysFailingModelRpcClient(
            FtRuntimeException(
                ExceptionType.BATCH_SLO_EXPIRED,
                "admission deadline exceeded",
            )
        )
        visitor = self._visitor(client)
        request_id_factory = Mock(return_value=456)
        visitor.set_request_id_factory(request_id_factory)

        stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(False)))
        with self.assertRaises(FtRuntimeException) as ctx:
            [output async for output in stream]

        self.assertEqual(
            ctx.exception.exception_type,
            ExceptionType.BATCH_SLO_EXPIRED,
        )
        self.assertEqual(client.attempts, 1)
        request_id_factory.assert_not_called()

    async def test_batch_slo_expired_overrides_earlier_retryable_capacity(self):
        client = _CapacityThenBatchSloExpiredModelRpcClient()
        visitor = self._visitor(client)
        visitor.set_request_id_factory(lambda: 456)

        stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(False)))
        with self.assertRaises(FtRuntimeException) as ctx:
            [output async for output in stream]

        self.assertEqual(
            ctx.exception.exception_type,
            ExceptionType.BATCH_SLO_EXPIRED,
        )
        self.assertEqual(client.attempts, 2)

    async def test_admission_rejections_are_terminal_and_keep_typed_reason(self):
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
                client = _AlwaysFailingModelRpcClient(
                    FtRuntimeException(
                        exception_type,
                        "typed admission rejection",
                        admission_reject_reason=reason,
                    )
                )
                visitor = self._visitor(client)
                visitor.set_request_id_factory(lambda: 456)

                stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(False)))
                with self.assertRaises(FtRuntimeException) as ctx:
                    [output async for output in stream]

                self.assertEqual(exception_type, ctx.exception.exception_type)
                self.assertEqual(reason, ctx.exception.admission_reject_reason)
                self.assertEqual(1, client.attempts)

    async def test_priority_preempted_overrides_earlier_retryable_capacity(self):
        client = _CapacityThenPreemptedModelRpcClient()
        visitor = self._visitor(client)
        visitor.set_request_id_factory(lambda: 456)

        stream = await visitor.enqueue(_FakeInput(_FakeGenerateConfig(False)))
        with self.assertRaises(FtRuntimeException) as ctx:
            [output async for output in stream]

        self.assertEqual(
            ctx.exception.exception_type,
            ExceptionType.PRIORITY_PREEMPTED,
        )
        self.assertEqual(client.attempts, 2)


if __name__ == "__main__":
    unittest.main()
