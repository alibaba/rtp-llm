import asyncio
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.server.backend_rpc_server_visitor import (
    BackendRPCServerVisitor,
    get_role_names,
)
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.master_client import FlexlbResponse
from rtp_llm.telemetry import attributes as trace_attrs


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

    def is_prefill_only(self):
        return self.max_new_tokens == 0

    def model_copy(self, update=None):
        copied = _FakeGenerateConfig(
            is_streaming=self.is_streaming,
            calculate_loss=self.calculate_loss,
            return_hidden_states=self.return_hidden_states,
            return_all_hidden_states=self.return_all_hidden_states,
        )
        copied.max_new_tokens = self.max_new_tokens
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
    # The visitor routes v41 vision inputs through the master schedule path;
    # non-vision requests carry no v41 payload.
    v41_inputs = None

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
        visitor.cache_key_seed = 0
        visitor.remote_vit = False
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
        self.assertNotIn(
            "rtp_llm.request_id",
            [call.args[0] for call in route_span.set_attribute.call_args_list],
        )
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
        generate_config = _FakeGenerateConfig(
            is_streaming=False,
            calculate_loss=1,
            return_hidden_states=True,
            return_all_hidden_states=True,
        )
        input = _FakeInput(generate_config)
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
        self.assertFalse(client.inputs[1].generate_config.is_prefill_only())
        self.assertEqual(client.inputs[1].generate_config.calculate_loss, 1)
        self.assertTrue(client.inputs[1].generate_config.return_hidden_states)
        self.assertTrue(client.inputs[1].generate_config.return_all_hidden_states)
        self.assertEqual(client.inputs[1].generate_config.role_addrs, [])

    async def test_prefill_only_retry_preserves_legal_config(self):
        client = _RetryingModelRpcClient()
        visitor = self._visitor(client)
        generate_config = _FakeGenerateConfig(is_streaming=False)
        generate_config.max_new_tokens = 0
        input = _FakeInput(generate_config)
        visitor.set_request_id_factory(lambda: 456)

        stream = await visitor.enqueue(input)
        outputs = [output async for output in stream]

        self.assertEqual(outputs, ["successful-output"])
        self.assertEqual(client.attempts, 2)
        retried_config = client.inputs[1].generate_config
        self.assertTrue(retried_config.is_prefill_only())
        self.assertEqual(retried_config.calculate_loss, 0)
        self.assertFalse(retried_config.return_hidden_states)
        self.assertFalse(retried_config.return_all_hidden_states)
        self.assertEqual(retried_config.role_addrs, [])

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


class _OwnedAttempt:
    """Retained iterator: cleanup must be explicit, never a GC side effect."""

    def __init__(self, outputs=(), error=None, close_error=None, block=False):
        self.outputs = iter(outputs)
        self.error = error
        self.close_error = close_error
        self.block = block
        self.waiting = asyncio.Event()
        self.closes = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.outputs)
        except StopIteration:
            pass
        if self.error is not None:
            raise self.error
        if self.block:
            self.waiting.set()
            await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self):
        self.closes += 1
        if self.close_error is not None:
            raise self.close_error


class BackendRPCServerVisitorCloseTest(unittest.IsolatedAsyncioTestCase):
    _visitor = BackendRPCServerVisitorRetryTest._visitor

    async def test_close_propagates_to_retained_inner_and_is_request_local(self):
        closed = [0, 0]

        async def inner(index):
            try:
                yield index
                await asyncio.Event().wait()
            finally:
                closed[index] += 1

        retained = [inner(0), inner(1)]
        client = SimpleNamespace(enqueue=Mock(side_effect=retained))
        visitor = self._visitor(client)
        wrappers = [
            await visitor.enqueue(_FakeInput(is_streaming=True)) for _ in retained
        ]
        try:
            self.assertEqual(await wrappers[0].__anext__(), 0)
            self.assertEqual(await wrappers[1].__anext__(), 1)
            await wrappers[0].aclose()
            self.assertEqual(closed, [1, 0])
            await wrappers[1].aclose()
            self.assertEqual(closed, [1, 1])
            await wrappers[0].aclose()
            self.assertEqual(closed, [1, 1])
        finally:
            for stream in wrappers + retained:
                await stream.aclose()

    async def test_attempt_closes_before_retry_and_preserves_buffered_outputs(self):
        first = _OwnedAttempt(
            ["discard"],
            RuntimeError("StatusCode.UNAVAILABLE"),
            RuntimeError("close failed"),
        )
        second = _OwnedAttempt(["one", "two"])

        def enqueue(_input):
            if first.closes:
                return second
            return first

        client = SimpleNamespace(enqueue=Mock(side_effect=enqueue))
        visitor = self._visitor(client)
        visitor.set_request_id_factory(lambda: 456)
        stream = await visitor.enqueue(_FakeInput())
        self.assertEqual([x async for x in stream], ["one", "two"])
        self.assertEqual(client.enqueue.call_count, 2)
        self.assertEqual((first.closes, second.closes), (1, 1))

    async def test_cleanup_error_preserves_primary_error_and_terminal_priority(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                original = FtRuntimeException(
                    ExceptionType.MASTER_NO_AVAILABLE_WORKER, "capacity"
                )
                later = (
                    FtRuntimeException(ExceptionType.BATCH_SLO_EXPIRED, "expired")
                    if terminal
                    else RuntimeError("downstream failed")
                )
                retained = [
                    _OwnedAttempt(
                        error=e,
                        close_error=RuntimeError("StatusCode.UNAVAILABLE cleanup"),
                    )
                    for e in (original, later)
                ]
                client = SimpleNamespace(enqueue=Mock(side_effect=retained))
                visitor = self._visitor(client)
                visitor.set_request_id_factory(lambda: 456)
                stream = await visitor.enqueue(_FakeInput())
                with self.assertRaises(FtRuntimeException) as caught:
                    [x async for x in stream]
                self.assertIs(caught.exception, later if terminal else original)
                self.assertEqual([x.closes for x in retained], [1, 1])
                self.assertEqual(client.enqueue.call_count, 2)

    async def test_streaming_partial_error_does_not_retry_or_get_masked_by_close(self):
        error = RuntimeError("StatusCode.UNAVAILABLE business")
        retained = _OwnedAttempt(["one"], error, RuntimeError("close failed"))
        client = SimpleNamespace(enqueue=Mock(return_value=retained))
        visitor = self._visitor(client)
        visitor.set_request_id_factory(lambda: 456)
        stream = await visitor.enqueue(_FakeInput(is_streaming=True))
        self.assertEqual(await stream.__anext__(), "one")
        with self.assertRaises(RuntimeError) as caught:
            await stream.__anext__()
        self.assertIs(caught.exception, error)
        self.assertEqual(retained.closes, 1)
        self.assertEqual(client.enqueue.call_count, 1)

    async def test_task_cancel_closes_attempt_and_preserves_cancellation(self):
        retained = _OwnedAttempt(block=True, close_error=RuntimeError("close failed"))
        client = SimpleNamespace(enqueue=Mock(return_value=retained))
        visitor = self._visitor(client)
        visitor.set_request_id_factory(lambda: 456)
        stream = await visitor.enqueue(_FakeInput(is_streaming=True))
        task = asyncio.create_task(stream.__anext__())
        await asyncio.wait_for(retained.waiting.wait(), 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(retained.closes, 1)
        self.assertEqual(client.enqueue.call_count, 1)

    async def test_optional_close_and_successful_cleanup_error(self):
        class IteratorWithoutClose:
            def __init__(self):
                self.outputs = iter(["one", "two"])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self.outputs)
                except StopIteration:
                    raise StopAsyncIteration

        for retained in (
            IteratorWithoutClose(),
            _OwnedAttempt(["one", "two"], close_error=RuntimeError("close failed")),
        ):
            with self.subTest(iterator=type(retained).__name__):
                client = SimpleNamespace(enqueue=Mock(return_value=retained))
                visitor = self._visitor(client)
                stream = await visitor.enqueue(_FakeInput())
                self.assertEqual([x async for x in stream], ["one", "two"])
                self.assertEqual(client.enqueue.call_count, 1)
                if isinstance(retained, _OwnedAttempt):
                    self.assertEqual(retained.closes, 1)


class _CloseTraceHandle:
    """Observe the real client's final status without a global SDK dependency."""

    def __init__(self):
        self.attributes = {}
        self.finished = asyncio.Event()
        self.status = None
        self.end_count = 0

    def set_attribute(self, key, value):
        if not self.finished.is_set():
            self.attributes[key] = value

    def finish(self, error=None, error_type=""):
        if not self.finished.is_set():
            self.end_count += 1
            self.status = "ERROR" if error is not None or error_type else "OK"
            self.finished.set()


class BackendRPCServerVisitorRpcCloseTest(unittest.IsolatedAsyncioTestCase):
    _visitor = BackendRPCServerVisitorRetryTest._visitor

    async def _check_rpc(self, *, fetch, finished, action, terminal, trace_enabled):
        import grpc
        import torch

        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.cpp.model_rpc import model_rpc_client as client_module
        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc
        from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
            GenerateOutputsPB,
            TensorPB,
        )
        from rtp_llm.utils.base_model_datatypes import GenerateInput

        release = asyncio.Event()
        server_waiting = asyncio.Event()
        server_closed = asyncio.Event()

        class Service(rpc.RpcServiceServicer):
            async def responses(self, request, context):
                try:
                    response = GenerateOutputsPB()
                    output = response.flatten_output
                    output.output_ids.data_type = TensorPB.DataType.INT32
                    output.output_ids.shape.extend([1, 1])
                    output.output_ids.int32_data = (7).to_bytes(4, "little")
                    output.finished.append(finished)
                    aux = output.aux_info.add()
                    aux.input_len, aux.output_len = 3, 1
                    yield response
                    server_waiting.set()
                    await release.wait()
                    if terminal != grpc.StatusCode.OK:
                        await context.abort(terminal, "injected terminal error")
                finally:
                    server_closed.set()

            FetchResponse = responses
            GenerateStreamCall = responses

        server = grpc.aio.server()
        rpc.add_RpcServiceServicer_to_server(Service(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        client = client_module.ModelRpcClient([f"127.0.0.1:{port}"], {}, 0)

        async def get_channel(_address):
            return channel

        client._channel_pool = SimpleNamespace(get=get_channel)
        retained = []

        def enqueue(request):
            inner = client.enqueue(request)
            retained.append(inner)
            return inner

        visitor = self._visitor(SimpleNamespace(enqueue=Mock(side_effect=enqueue)))
        request = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(
                max_new_tokens=16,
                is_streaming=True,
                timeout_ms=0,
                role_addrs=[
                    RoleAddr(
                        role=RoleType.PREFILL,
                        ip="127.0.0.1",
                        http_port=8000,
                        grpc_port=port,
                    )
                ],
            ),
            request_id=731,
            mm_inputs=[],
            enqueued_by_master=fetch,
        )
        calls = []
        stub = rpc.RpcServiceStub(channel)
        method = stub.FetchResponse if fetch else stub.GenerateStreamCall

        def call(*args, **kwargs):
            result = method(*args, **kwargs)
            calls.append(result)
            return result

        wrapped_stub = SimpleNamespace(
            **{"FetchResponse" if fetch else "GenerateStreamCall": call}
        )
        span = _CloseTraceHandle()
        wrapper = None
        task = None
        try:
            with (
                patch.object(
                    client_module, "RpcServiceStub", return_value=wrapped_stub
                ),
                patch.object(
                    client_module,
                    "start_client_span",
                    return_value=(span if trace_enabled else None, []),
                ),
                patch.object(client_module, "RPC_SETTLE_TIMEOUT_SECONDS", 0.05),
            ):
                wrapper = await visitor.enqueue(request)
                output = await wrapper.__anext__()
                self.assertEqual(output.generate_outputs[0].finished, finished)
                await asyncio.wait_for(server_waiting.wait(), 5)
                if action == "cancel":
                    task = asyncio.create_task(wrapper.__anext__())
                    await asyncio.sleep(0)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    await asyncio.wait_for(wrapper.aclose(), 1)
                # Retaining the generator prevents GC from repairing a missing close.
                self.assertIsNone(retained[0].ag_frame)
                if finished:
                    self.assertFalse(calls[0].cancelled())
                if action != "stall":
                    release.set()
                expected = (
                    terminal
                    if finished and action != "stall"
                    else grpc.StatusCode.CANCELLED
                )
                self.assertEqual(await asyncio.wait_for(calls[0].code(), 5), expected)
                await asyncio.wait_for(server_closed.wait(), 5)
                if trace_enabled:
                    await asyncio.wait_for(span.finished.wait(), 5)
                    self.assertEqual(span.attributes["request_id"], "731")
                    self.assertEqual(
                        span.attributes["rpc.response.status_code"], expected.name
                    )
                    self.assertEqual(
                        span.status, "OK" if expected == grpc.StatusCode.OK else "ERROR"
                    )
                    self.assertEqual(span.end_count, 1)
                self.assertEqual(visitor.model_rpc_client.enqueue.call_count, 1)
        finally:
            release.set()
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if wrapper is not None:
                await wrapper.aclose()
            for inner in retained:
                await inner.aclose()
            await channel.close()
            await server.stop(None)

    async def test_real_rpc_wrapper_close_cancel_and_late_status(self):
        import grpc

        cases = [
            (False, False, "close", grpc.StatusCode.OK),
            (True, False, "close", grpc.StatusCode.OK),
            (False, False, "cancel", grpc.StatusCode.OK),
            (True, False, "cancel", grpc.StatusCode.OK),
            (True, True, "close", grpc.StatusCode.OK),
            (True, True, "cancel", grpc.StatusCode.OK),
            (True, True, "close", grpc.StatusCode.INTERNAL),
            (True, True, "close", grpc.StatusCode.CANCELLED),
            (True, True, "close", grpc.StatusCode.DEADLINE_EXCEEDED),
            (True, True, "stall", grpc.StatusCode.OK),
        ]
        for trace_enabled in (False, True):
            for fetch, finished, action, terminal in cases:
                with self.subTest(
                    trace=trace_enabled,
                    fetch=fetch,
                    finished=finished,
                    action=action,
                    terminal=terminal,
                ):
                    await self._check_rpc(
                        fetch=fetch,
                        finished=finished,
                        action=action,
                        terminal=terminal,
                        trace_enabled=trace_enabled,
                    )


if __name__ == "__main__":
    unittest.main()
