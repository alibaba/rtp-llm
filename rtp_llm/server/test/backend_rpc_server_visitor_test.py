import asyncio
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import torch

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.server.backend_rpc_server_visitor import (
    BackendRPCServerVisitor,
    disable_token_only_reuse_for_input_embeddings,
    get_role_names,
)
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.host_service import HostService, HostServiceArgs
from rtp_llm.server.master_client import FlexlbResponse
from rtp_llm.server.recent_cache_key_window import RecentCacheKeyWindow
from rtp_llm.telemetry import attributes as trace_attrs
from rtp_llm.utils.base_model_datatypes import GenerateInput, InputEmbeddings


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
        # Also accepts the full dataclass field set so dataclasses.replace()
        # (used by the visitor's retry path to stamp a fresh request_id) works.
        if isinstance(is_streaming, _FakeGenerateConfig) and generate_config is None:
            generate_config = is_streaming
        if generate_config is None:
            generate_config = _FakeGenerateConfig(
                is_streaming=is_streaming, **generate_config_kwargs
            )
        self.generate_config = generate_config
        self.request_id = request_id
        self.token_ids = token_ids if token_ids is not None else _FakeTokenIds()
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

    def __init__(self):
        self.generate_config = _FakeGenerateConfig()
        self.enqueued_by_master = False


class _FakeHostService:
    service_available = False

    def get_master_addr(self):
        return "master:1234"


class _FakeRouteSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def finish(self, **kwargs):
        pass


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
    async def test_get_master_route_addrs_passes_pb_and_marks_master_enqueue(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.seq_size_per_block = 16
        visitor.master_client = _FakeMasterClient()
        visitor._route_cache_keys = lambda keys: keys
        visitor._report_recent_cache_key_metrics = lambda keys: None
        visitor._page_rr_route_cache_keys = False
        visitor._page_rr_cp_size = 1

        input = _FakeRouteInput()

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.get_block_cache_keys",
            return_value=[11, 22],
        ), patch(
            "rtp_llm.server.backend_rpc_server_visitor.trans_input",
            return_value=_FakeInputPB(),
        ), patch(
            "rtp_llm.server.backend_rpc_server_visitor.kmonitor"
        ):
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

    async def test_route_ips_records_master_success_when_domain_completes_roles(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.master_config = None
        visitor.host_service = _FakeHostService()
        visitor.backend_role_list = ["PREFILL", "DECODE"]
        route_span = _FakeRouteSpan()

        async def get_master_route_addrs(route_input):
            route_input.generate_config.role_addrs.append(
                RoleAddr(role=RoleType.PREFILL, ip="master", http_port=1, grpc_port=2)
            )
            return None

        async def get_domain_route_addrs(route_input):
            route_input.generate_config.role_addrs.append(
                RoleAddr(role=RoleType.DECODE, ip="domain", http_port=3, grpc_port=4)
            )

        visitor.get_master_route_addrs = get_master_route_addrs
        visitor.get_domain_route_addrs = get_domain_route_addrs

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.start_internal_span",
            return_value=route_span,
        ), patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
            await visitor.route_ips(_FakeInput())

        self.assertEqual(
            route_span.attributes["rtp_llm.route.source"],
            "master+domain_fallback",
        )

    async def test_route_ips_preserves_request_source_when_domain_completes_roles(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.master_config = None
        visitor.host_service = _FakeHostService()
        visitor.backend_role_list = ["PREFILL", "DECODE"]
        route_span = _FakeRouteSpan()
        input = _FakeInput()
        input.generate_config.role_addrs.append(
            RoleAddr(role=RoleType.PREFILL, ip="requested", http_port=1, grpc_port=2)
        )

        async def get_domain_route_addrs(route_input):
            route_input.generate_config.role_addrs.append(
                RoleAddr(role=RoleType.DECODE, ip="domain", http_port=3, grpc_port=4)
            )

        visitor.get_domain_route_addrs = get_domain_route_addrs

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.start_internal_span",
            return_value=route_span,
        ), patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
            await visitor.route_ips(input)

        self.assertEqual(
            route_span.attributes["rtp_llm.route.source"],
            "request+domain_fallback",
        )

    async def test_route_failure_uses_stable_error_type_and_code(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.master_config = None
        visitor.host_service = _FakeHostService()
        visitor.backend_role_list = ["PREFILL"]
        route_span = _FakeRouteSpan()

        async def get_master_route_addrs(_input):
            return FlexlbResponse.error_response(
                int(ExceptionType.MASTER_NO_AVAILABLE_WORKER), "scheduler details"
            )

        visitor.get_master_route_addrs = get_master_route_addrs

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.start_internal_span",
            return_value=route_span,
        ), patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
            with self.assertRaises(FtRuntimeException):
                await visitor.route_ips(_FakeInput())

        self.assertEqual(
            route_span.attributes["rtp_llm.error.code"],
            int(ExceptionType.MASTER_NO_AVAILABLE_WORKER),
        )


class TestBackendRouteTrace(unittest.TestCase):
    def test_proactive_rejection_finishes_after_all_attributes(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.master_config = SimpleNamespace(master_queue_reject_threshold=-1)
        visitor.host_service = MagicMock()
        visitor.host_service.get_queue_length.return_value = 0
        route_span = MagicMock()
        request = SimpleNamespace(request_id=123)

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.start_internal_span",
            return_value=route_span,
        ), patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor.report"):
            with self.assertRaisesRegex(
                Exception, "queue length 0 exceeds threshold -1"
            ):
                asyncio.run(visitor.route_ips(request))

        route_span.set_attribute.assert_any_call(trace_attrs.REQUEST_ID, "123")
        route_span.set_attribute.assert_any_call(trace_attrs.RTP_LLM_REQUEST_ID, 123)
        route_span.set_attribute.assert_any_call(
            trace_attrs.RTP_LLM_ROUTE_QUEUE_LENGTH, 0
        )
        route_span.set_attribute.assert_any_call(
            trace_attrs.RTP_LLM_ROUTE_QUEUE_REJECT_THRESHOLD, -1
        )
        route_span.set_attribute.assert_any_call(
            trace_attrs.RTP_LLM_ROUTE_SOURCE, "none"
        )
        route_span.set_attribute.assert_any_call(
            trace_attrs.RTP_LLM_ERROR_CODE, int(ExceptionType.TRAFFIC_LIMIT_ERROR)
        )
        route_span.finish.assert_called_once()
        self.assertEqual(
            route_span.finish.call_args.kwargs["error_type"], "TrafficLimit"
        )
        self.assertEqual(route_span.method_calls[-1][0], "finish")
        self.assertIn("none", trace_attrs.RTP_LLM_ROUTE_SOURCE_VALUES)

    def test_route_cancellation_uses_stable_span_error_type(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.master_config = None
        visitor.host_service = MagicMock()
        visitor.host_service.get_master_addr.return_value = "master:9000"
        visitor.backend_role_list = ["PREFILL"]
        route_span = MagicMock()

        async def cancel_master_route(_input):
            raise asyncio.CancelledError("request cancelled")

        visitor.get_master_route_addrs = cancel_master_route

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.start_internal_span",
            return_value=route_span,
        ), patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor.report"):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(visitor.route_ips(_FakeInput()))

        route_span.set_attribute.assert_any_call(
            trace_attrs.RTP_LLM_ROUTE_SOURCE, "none"
        )
        route_span.finish.assert_called_once()
        self.assertEqual(route_span.finish.call_args.kwargs["error_type"], "Cancelled")


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


def make_generate_input(input_embeddings=None):
    return GenerateInput(
        request_id=123,
        token_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        mm_inputs=[],
        generate_config=GenerateConfig(max_new_tokens=1),
        input_embeddings=input_embeddings,
    )


class BackendRPCServerVisitorTest(unittest.IsolatedAsyncioTestCase):
    def make_visitor(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.source_role = "frontend"
        visitor.source_ip = "127.0.0.1"
        visitor._page_rr_route_cache_keys = False
        visitor._page_rr_cp_size = 1
        visitor._prefill_cp_active = False
        visitor.recent_cache_key_window = RecentCacheKeyWindow()
        return visitor

    async def test_master_route_uses_token_cache_keys_without_input_embeddings(self):
        visitor = self.make_visitor()
        visitor.seq_size_per_block = 2
        visitor.master_client = Mock()
        visitor.master_client.get_backend_role_addrs = AsyncMock(
            return_value=FlexlbResponse.ok(
                [
                    RoleAddr(
                        role=RoleType.PREFILL,
                        ip="127.0.0.1",
                        http_port=1,
                        grpc_port=2,
                    )
                ]
            )
        )
        input = make_generate_input()

        await visitor.get_master_route_addrs(input)

        kwargs = visitor.master_client.get_backend_role_addrs.call_args.kwargs
        self.assertGreater(len(kwargs["block_cache_keys"]), 0)

    def make_domain_visitor(self, roles=None):
        visitor = self.make_visitor()
        visitor.master_config = SimpleNamespace(master_queue_reject_threshold=10)
        visitor.pd_sep_config = SimpleNamespace(role_type=RoleType.FRONTEND)
        visitor.host_service = Mock()
        visitor.host_service.get_queue_length.return_value = 0
        visitor.host_service.get_master_addr.return_value = "batch-master:9000"
        visitor.host_service.get_backend_role_addrs.return_value = roles or []
        visitor.backend_role_list = [RoleType.PDFUSION]
        visitor.master_client = Mock()
        visitor.master_client.get_backend_role_addrs = AsyncMock()
        return visitor

    def custom_input(self):
        return make_generate_input(InputEmbeddings([torch.zeros(1, 8)], [1]))

    async def test_embeddings_select_one_topology_with_real_host_service(self):
        p = RoleAddr(
            role=RoleType.PREFILL, ip="127.0.0.1", http_port=9100, grpc_port=9101
        )
        d = RoleAddr(
            role=RoleType.DECODE, ip="127.0.0.1", http_port=9200, grpc_port=9201
        )
        f = RoleAddr(
            role=RoleType.PDFUSION, ip="127.0.0.1", http_port=9300, grpc_port=9301
        )
        cases = [
            ([p, d], RoleType.PDFUSION, [p, d]),
            (
                [p],
                RoleType.PDFUSION,
                [
                    p,
                    RoleAddr(
                        role=RoleType.DECODE,
                        ip="127.0.0.1",
                        http_port=8200,
                        grpc_port=8201,
                    ),
                ],
            ),
            ([f], RoleType.DECODE, [f]),
            ([p, d, f], RoleType.PDFUSION, [p, d]),
            (
                [],
                RoleType.PDFUSION,
                [
                    RoleAddr(
                        role=RoleType.DECODE,
                        ip="127.0.0.1",
                        http_port=8200,
                        grpc_port=8201,
                    ),
                    RoleAddr(
                        role=RoleType.PREFILL,
                        ip="127.0.0.1",
                        http_port=8100,
                        grpc_port=8101,
                    ),
                ],
            ),
        ]
        for explicit, unavailable, expected in cases:
            with self.subTest(explicit=explicit, unavailable=unavailable):
                visitor = self.make_domain_visitor()
                visitor.pd_sep_config = SimpleNamespace(
                    role_type=RoleType.FRONTEND, to_string=lambda: "FRONTEND"
                )
                args = HostServiceArgs(
                    pdfusion_domain="127.0.0.1:8000",
                    prefill_domain="127.0.0.1:8100",
                    decode_domain="127.0.0.1:8200",
                    use_local=True,
                )
                visitor.host_service = HostService(args)
                visitor.backend_role_list = visitor.get_backend_role_list(
                    visitor.pd_sep_config, args
                )
                request = self.custom_input()
                request.generate_config.role_addrs = list(explicit)
                with patch.object(
                    visitor.host_service,
                    "get_master_addr",
                    return_value="batch-master:9000",
                ), patch.object(
                    visitor.host_service.role_vip_map[unavailable],
                    "get_host",
                    return_value=None,
                ), patch(
                    "rtp_llm.server.backend_rpc_server_visitor.kmonitor"
                ):
                    await visitor.route_ips(request)
                self.assertCountEqual(request.generate_config.role_addrs, expected)
                self.assertFalse(request.enqueued_by_master)
                visitor.master_client.get_backend_role_addrs.assert_not_called()

    async def test_embeddings_topology_selection_keeps_required_auxiliary_roles(self):
        from rtp_llm.ops import VitSeparation

        visitor = self.make_domain_visitor()
        visitor.pd_sep_config = SimpleNamespace(
            role_type=RoleType.FRONTEND, to_string=lambda: "FRONTEND"
        )
        args = HostServiceArgs(
            pdfusion_domain="127.0.0.1:8000",
            prefill_domain="127.0.0.1:8100",
            decode_domain="127.0.0.1:8200",
            vit_domain="127.0.0.1:8300",
            use_local=True,
        )
        visitor.host_service = HostService(args)
        visitor.backend_role_list = visitor.get_backend_role_list(
            visitor.pd_sep_config, args, VitSeparation.VIT_SEPARATION_REMOTE
        )
        request = self.custom_input()
        request.generate_config.role_addrs = [
            RoleAddr(
                role=RoleType.PDFUSION, ip="127.0.0.1", http_port=9300, grpc_port=9301
            )
        ]
        with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
            await visitor.route_ips(request)
        self.assertEqual(
            {addr.role for addr in request.generate_config.role_addrs},
            {RoleType.PDFUSION, RoleType.VIT},
        )

    async def test_embeddings_bypass_batch_master_and_control_plane_serialization(self):
        address = RoleAddr(
            role=RoleType.PDFUSION, ip="127.0.0.1", http_port=9000, grpc_port=9001
        )
        visitor = self.make_domain_visitor([address])
        for rows in (1, 2047, 2048, 2049):
            with self.subTest(rows=rows):
                request = make_generate_input(
                    InputEmbeddings(
                        [torch.zeros(rows, 4096, dtype=torch.bfloat16)], [1]
                    )
                )
                request.token_ids = torch.zeros(rows + 2, dtype=torch.int32)
                with patch(
                    "rtp_llm.server.backend_rpc_server_visitor.trans_input"
                ) as serialize:
                    await visitor.route_ips(request)
                serialize.assert_not_called()
                visitor.master_client.get_backend_role_addrs.assert_not_called()
                self.assertEqual(request.generate_config.role_addrs, [address])
                self.assertFalse(request.enqueued_by_master)

    async def test_embeddings_do_not_enter_master_even_when_called_directly(self):
        visitor = self.make_domain_visitor()
        with self.assertRaisesRegex(FtRuntimeException, "backend domain routing"):
            await visitor.get_master_route_addrs(self.custom_input())
        visitor.master_client.get_backend_role_addrs.assert_not_called()

    async def test_embeddings_retain_cached_queue_rejection(self):
        visitor = self.make_domain_visitor()
        visitor.host_service.get_queue_length.return_value = 11
        with self.assertRaises(FtRuntimeException) as error:
            await visitor.route_ips(self.custom_input())
        self.assertEqual(
            error.exception.exception_type, ExceptionType.TRAFFIC_LIMIT_ERROR
        )
        visitor.host_service.get_backend_role_addrs.assert_not_called()
        visitor.master_client.get_backend_role_addrs.assert_not_called()

    async def test_embeddings_reject_missing_or_incomplete_backend_roles(self):
        prefill = RoleAddr(
            role=RoleType.PREFILL, ip="127.0.0.1", http_port=9000, grpc_port=9001
        )
        invalid = RoleAddr(role=RoleType.PDFUSION, ip="", http_port=0, grpc_port=0)
        valid = RoleAddr(
            role=RoleType.PDFUSION, ip="127.0.0.1", http_port=9000, grpc_port=9001
        )
        bad_first = RoleAddr(
            role=RoleType.PDFUSION, ip="127.0.0.1", http_port=0, grpc_port=0
        )
        for addresses in ([], [prefill], [invalid], [bad_first, valid]):
            with self.subTest(addresses=addresses):
                visitor = self.make_domain_visitor(addresses)
                with self.assertRaises(FtRuntimeException) as error:
                    await visitor.route_ips(self.custom_input())
                self.assertEqual(
                    error.exception.exception_type, ExceptionType.ROUTE_ERROR
                )
                visitor.master_client.get_backend_role_addrs.assert_not_called()

    async def test_embeddings_preserve_explicit_roles_and_fill_missing_decode(self):
        prefill = RoleAddr(
            role=RoleType.PREFILL, ip="127.0.0.1", http_port=9000, grpc_port=9001
        )
        decode = RoleAddr(
            role=RoleType.DECODE, ip="127.0.0.1", http_port=9001, grpc_port=9002
        )
        visitor = self.make_domain_visitor([decode])
        visitor.backend_role_list = [RoleType.PREFILL, RoleType.DECODE]
        request = self.custom_input()
        request.generate_config.role_addrs = [prefill]
        await visitor.route_ips(request)
        self.assertEqual(request.generate_config.role_addrs, [prefill, decode])
        visitor.host_service.get_backend_role_addrs.assert_called_once_with(
            [RoleType.DECODE]
        )
        visitor.master_client.get_backend_role_addrs.assert_not_called()

    async def test_embeddings_reject_stale_master_enqueue_flag(self):
        visitor = self.make_domain_visitor()
        request = self.custom_input()
        request.enqueued_by_master = True
        request.generate_config.role_addrs = [
            RoleAddr(
                role=RoleType.PDFUSION, ip="127.0.0.1", http_port=9000, grpc_port=9001
            )
        ]
        with self.assertRaisesRegex(FtRuntimeException, "cannot fetch"):
            await visitor.route_ips(request)
        visitor.host_service.get_backend_role_addrs.assert_not_called()

    async def test_embeddings_propagate_domain_cancellation(self):
        visitor = self.make_domain_visitor()
        visitor.get_domain_route_addrs = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await visitor.route_ips(self.custom_input())
        visitor.master_client.get_backend_role_addrs.assert_not_called()

    async def test_enqueue_disables_token_only_reuse_with_input_embeddings(self):
        visitor = self.make_visitor()
        visitor.max_seq_len = 16
        visitor.sp_config = None
        visitor.host_service = Mock(service_available=False)
        visitor.model_rpc_client = Mock()

        async def stream():
            yield "result"

        visitor.model_rpc_client.enqueue = Mock(return_value=stream())
        input = make_generate_input(
            InputEmbeddings(
                embeddings=[torch.zeros((1, 8), dtype=torch.float32)],
                embedding_locs=[1],
            )
        )

        self.assertTrue(input.generate_config.reuse_cache)
        output = await visitor.enqueue(input)

        self.assertEqual([item async for item in output], ["result"])
        self.assertFalse(input.generate_config.reuse_cache)
        self.assertFalse(input.generate_config.enable_device_cache)
        self.assertFalse(input.generate_config.enable_memory_cache)
        self.assertFalse(input.generate_config.enable_remote_cache)

    def test_check_sp_supported_rejects_input_embeddings(self):
        visitor = self.make_visitor()
        visitor.sp_config = Mock(model_type="mtp")
        input = make_generate_input(
            InputEmbeddings(
                embeddings=[torch.zeros((1, 8), dtype=torch.float32)],
                embedding_locs=[1],
            )
        )

        for disabled in (False, True):
            with self.subTest(force_disable_sp_run=disabled):
                input.generate_config.force_disable_sp_run = disabled
                with self.assertRaisesRegex(
                    FtRuntimeException, "unsupported by speculative"
                ) as error:
                    visitor.check_sp_supported(input)
                self.assertEqual(
                    error.exception.exception_type, ExceptionType.UNSUPPORTED_OPERATION
                )

    async def test_batch_enqueue_disables_token_only_reuse_with_input_embeddings(self):
        visitor = self.make_visitor()
        visitor.max_seq_len = 16
        visitor.sp_config = None
        visitor.host_service = Mock(service_available=False)
        visitor.model_rpc_client = Mock()
        visitor.model_rpc_client.batch_enqueue = AsyncMock(return_value=[])
        text_input = make_generate_input()
        embedding_input = make_generate_input(
            InputEmbeddings(
                embeddings=[torch.zeros((1, 8), dtype=torch.float32)],
                embedding_locs=[1],
            )
        )

        await visitor.batch_enqueue([text_input, embedding_input])

        self.assertTrue(text_input.generate_config.reuse_cache)
        self.assertFalse(embedding_input.generate_config.reuse_cache)
        self.assertFalse(embedding_input.generate_config.enable_device_cache)
        self.assertFalse(embedding_input.generate_config.enable_memory_cache)
        self.assertFalse(embedding_input.generate_config.enable_remote_cache)

    def test_empty_input_embeddings_keeps_reuse_flags(self):
        input = make_generate_input(
            InputEmbeddings(
                embeddings=[],
                embedding_locs=[],
            )
        )

        disable_token_only_reuse_for_input_embeddings(input)

        self.assertTrue(input.generate_config.reuse_cache)
        self.assertTrue(input.generate_config.enable_device_cache)
        self.assertTrue(input.generate_config.enable_memory_cache)
        self.assertTrue(input.generate_config.enable_remote_cache)


if __name__ == "__main__":
    unittest.main()
