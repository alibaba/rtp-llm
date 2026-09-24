import asyncio
import gc
import unittest
import weakref
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import grpc
import torch

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    GenerateInputPB,
    MultimodalOutputPB,
    MultimodalOutputsPB,
)
from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics
from rtp_llm.ops import PDSepConfig, SpecialTokens, VitSeparation
from rtp_llm.server.backend_rpc_server_visitor import (
    BackendRPCServerVisitor,
    get_role_names,
)
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.master_client import FlexlbResponse, MasterClient
from rtp_llm.utils.base_model_datatypes import GenerateInput
from rtp_llm.utils.multimodal_util import MultimodalInput


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
    async def test_close_failure_releases_failed_request_without_cyclic_gc(self):
        class Stream:
            def __init__(self, request, partial):
                self.request = request
                self.partial = partial

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.partial:
                    return self.request.token_ids
                raise RuntimeError("iteration failed")

            async def aclose(self):
                raise RuntimeError("close failed")

        class Client:
            def __init__(self, partial):
                self.partial = partial

            def enqueue(self, request):
                return Stream(request, self.partial)

        async def run(partial):
            request = _FakeInput(is_streaming=True, token_ids=torch.zeros(32))
            ref = weakref.ref(request.token_ids)
            stream = await self._visitor(Client(partial)).enqueue(request)
            try:
                await stream.__anext__()
                await stream.aclose()
            except RuntimeError as error:
                self.assertEqual(str(error), "close failed")
            else:
                self.fail("close failure must propagate")
            return ref

        enabled = gc.isenabled()
        gc.disable()
        try:
            for partial in (False, True):
                with self.subTest(partial=partial):
                    ref = await run(partial)
                    self.assertIsNone(ref())
        finally:
            if enabled:
                gc.enable()
            gc.collect()

    async def test_completed_requests_release_tensors_without_cyclic_gc(self):
        async def run(mode, is_streaming):
            class Client:
                attempts = 0

                async def enqueue(self, request):
                    self.attempts += 1
                    try:
                        if mode.startswith("retry") and self.attempts == 1:
                            raise FtRuntimeException(
                                ExceptionType.MASTER_NO_AVAILABLE_WORKER, "capacity"
                            )
                        if mode in ("error", "retry_error"):
                            raise RuntimeError("failed")
                        if mode in ("cancel", "retry_cancel"):
                            raise asyncio.CancelledError()
                        yield request.token_ids
                    finally:
                        if mode == "close_error":
                            raise RuntimeError("close failed")

            client = Client()
            visitor = self._visitor(client)
            visitor.request_id_factory = lambda: 456
            request = _FakeInput(is_streaming=is_streaming, token_ids=torch.zeros(32))
            refs = [weakref.ref(request), weakref.ref(request.token_ids)]
            stream = await visitor.enqueue(request)
            try:
                async for output in stream:
                    del output
                    if mode == "early_close":
                        break
            except (RuntimeError, FtRuntimeException, asyncio.CancelledError):
                pass
            finally:
                await stream.aclose()
            return refs

        enabled = gc.isenabled()
        gc.disable()
        try:
            for mode in (
                "success",
                "error",
                "retry_success",
                "retry_error",
                "cancel",
                "retry_cancel",
                "close_error",
                "early_close",
            ):
                for is_streaming in (False, True):
                    with self.subTest(mode=mode, is_streaming=is_streaming):
                        refs = await run(mode, is_streaming)
                        self.assertTrue(all(ref() is None for ref in refs))
        finally:
            if enabled:
                gc.enable()
            gc.collect()

    async def test_close_after_partial_output_closes_model_rpc(self):
        closed = []

        class Client:
            async def enqueue(self, _input):
                try:
                    yield "partial"
                    yield "unread"
                finally:
                    closed.append(True)

        visitor = self._visitor(Client())
        stream = await visitor.enqueue(_FakeInput(is_streaming=True))
        self.assertEqual(await stream.__anext__(), "partial")
        await stream.aclose()
        self.assertEqual(closed, [True])

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


class DeepSeekVisionMasterRoutingTest(unittest.IsolatedAsyncioTestCase):
    async def test_openai_frontend_pipeline_enables_two_stage_routing(self):
        env = SimpleNamespace(
            server_config=None,
            distribute_config=None,
            parallelism_config=None,
            sp_config=None,
            grpc_config=None,
            master_config=None,
            prefill_cp_config=None,
            generate_env_config=None,
            vit_config=SimpleNamespace(
                vit_separation=VitSeparation.VIT_SEPARATION_REMOTE
            ),
        )
        config = SimpleNamespace(
            ckpt_path="unused",
            tokenizer_path="unused",
            model_type="deepseek_v4",
            max_seq_len=100,
            mm_model_config=SimpleNamespace(mm_padding_size=4),
            attn_config=SimpleNamespace(tokens_per_block=4),
            mm_related_params=SimpleNamespace(
                config={"vision_n_layers": 1},
                special_token_ids={"image_token_id": 99},
            ),
        )
        with patch("rtp_llm.frontend.frontend_worker.TokenizerFactory.create"), patch(
            "rtp_llm.frontend.frontend_worker.EngineConfig.create",
            return_value=SimpleNamespace(
                pd_sep_config=PDSepConfig(), parallelism_config=None
            ),
        ), patch("rtp_llm.frontend.frontend_worker.get_world_info"), patch(
            "rtp_llm.frontend.frontend_worker.get_dp_addrs_from_world_info",
            return_value=[],
        ):
            worker = FrontendWorker(env, config, SpecialTokens())
        visitor = worker.backend_rpc_server_visitor
        try:
            self.assertEqual(visitor.dsv4_image_token_id, 99)
            control = self.make_visitor()
            visitor.master_client.get_backend_role_addrs = (
                control.master_client.get_backend_role_addrs
            )
            visitor._get_vit_token_ids = control._get_vit_token_ids
            request = self.make_input()
            with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor.report"):
                self.assertIsNone(await visitor.get_master_route_addrs(request))
            calls = visitor.master_client.get_backend_role_addrs.await_args_list
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0].kwargs["vit_only"])
            self.assertEqual(
                list(calls[1].kwargs["input_pb"].token_ids), request.token_ids.tolist()
            )
            self.assertEqual(request.generate_config.role_addrs[0], self.vit)
        finally:
            await worker.close()

    def make_visitor(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.dsv4_image_token_id = 99
        visitor.mm_padding_size = 4
        visitor.max_seq_len = 100
        visitor.seq_size_per_block = 4
        visitor._page_rr_route_cache_keys = False
        visitor._page_rr_cp_size = 1
        visitor._report_recent_cache_key_metrics = Mock()
        self.vit = RoleAddr(
            role=RoleType.VIT, ip="chosen-vit", http_port=81, grpc_port=82
        )
        self.prefill = RoleAddr(
            role=RoleType.PREFILL, ip="prefill", http_port=91, grpc_port=92
        )
        other_vit = self.vit.model_copy(update={"ip": "other-vit"})
        visitor.master_client = SimpleNamespace(
            master_config=SimpleNamespace(master_default_timeout_ms=1000),
            get_backend_role_addrs=AsyncMock(
                side_effect=[
                    FlexlbResponse.ok([self.vit]),
                    FlexlbResponse.ok(
                        [self.prefill, other_vit], enqueued_by_master=True
                    ),
                ]
            ),
        )
        visitor._get_vit_token_ids = AsyncMock(
            return_value=[[41, 42, 43, 44], [51, 52, 53, 54, 55, 56]]
        )
        return visitor

    def make_input(self):
        return GenerateInput(
            request_id=88,
            token_ids=torch.tensor([7, 99, 8, 9, 99, 10], dtype=torch.int32),
            mm_inputs=[MultimodalInput("first.png"), MultimodalInput("second.png")],
            generate_config=GenerateConfig(timeout_ms=1000, ttft_timeout_ms=1000),
        )

    async def test_two_stage_hint_keeps_raw_enqueue_and_selected_vit(self):
        visitor = self.make_visitor()
        request = self.make_input()
        raw_ids = request.token_ids.clone()
        with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor.report"):
            self.assertIsNone(await visitor.get_master_route_addrs(request))

        calls = visitor.master_client.get_backend_role_addrs.await_args_list
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].kwargs["vit_only"])
        self.assertEqual(calls[0].kwargs["request_id"], calls[1].kwargs["request_id"])
        second = calls[1].kwargs
        self.assertEqual(second["input"].prompt_length, 14)
        self.assertEqual(
            second["input"].token_ids.tolist(),
            [7, 41, 42, 43, 44, 8, 9, 51, 52, 53, 54, 55, 56, 10],
        )
        self.assertEqual(list(second["input_pb"].token_ids), raw_ids.tolist())
        self.assertTrue(torch.equal(request.token_ids, raw_ids))
        self.assertEqual(
            [
                item.mm_preprocess_config.mm_padding_size
                for item in second["input_pb"].multimodal_inputs
            ],
            [2, 0],
        )
        self.assertIsNot(request.mm_inputs[0].config, request.mm_inputs[1].config)
        self.assertEqual(
            second["input_pb"].generate_config.role_addrs[0].ip, "chosen-vit"
        )
        self.assertEqual(request.generate_config.role_addrs, [self.vit, self.prefill])
        self.assertTrue(request.enqueued_by_master)
        self.assertLessEqual(second["timeout_s"], calls[0].kwargs["timeout_s"])

    async def test_vit_failure_never_schedules_pd(self):
        visitor = self.make_visitor()
        request = self.make_input()
        visitor._get_vit_token_ids.side_effect = RuntimeError("vit unavailable")
        with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor.report"):
            with self.assertRaisesRegex(RuntimeError, "vit unavailable"):
                await visitor.get_master_route_addrs(request)
        self.assertEqual(visitor.master_client.get_backend_role_addrs.await_count, 1)
        self.assertEqual(request.generate_config.role_addrs, [])
        self.assertFalse(request.enqueued_by_master)

    async def test_configured_padding_two_and_disabled(self):
        for padding_size in (0, 2):
            visitor = self.make_visitor()
            visitor.mm_padding_size = padding_size
            request = self.make_input()
            request.token_ids = torch.tensor([7, 8, 99, 9, 99, 10], dtype=torch.int32)
            with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor.report"):
                await visitor.get_master_route_addrs(request)
            actual = [item.config.mm_padding_size for item in request.mm_inputs]
            self.assertEqual(actual, [1, 1] if padding_size else [0, 0])

    async def test_metadata_rpc_requests_no_features_and_checks_image_count(self):
        visitor = self.make_visitor()
        visitor._vit_channel_pool = SimpleNamespace(
            get=AsyncMock(return_value=object())
        )
        input_pb = GenerateInputPB()
        input_pb.multimodal_inputs.add(multimodal_url="first.png")
        rpc = AsyncMock(
            return_value=MultimodalOutputsPB(
                multimodal_outputs=[MultimodalOutputPB(token_ids=[41, 42])]
            )
        )
        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.MultimodalRpcServiceStub",
            return_value=SimpleNamespace(RemoteMultimodalEmbedding=rpc),
        ):
            result = await BackendRPCServerVisitor._get_vit_token_ids(
                visitor, self.vit, input_pb, 0.5
            )
            self.assertEqual(result, [[41, 42]])
            self.assertTrue(rpc.await_args.args[0].metadata_only)
            self.assertEqual(rpc.await_args.kwargs["timeout"], 0.5)
            input_pb.multimodal_inputs.add(multimodal_url="second.png")
            with self.assertRaises(FtRuntimeException):
                await BackendRPCServerVisitor._get_vit_token_ids(
                    visitor, self.vit, input_pb, 0.5
                )

    async def test_metadata_rpc_preserves_failure_categories(self):
        visitor = self.make_visitor()
        visitor._vit_channel_pool = SimpleNamespace(
            get=AsyncMock(return_value=object())
        )
        cases = {
            grpc.StatusCode.DEADLINE_EXCEEDED: ExceptionType.GENERATE_TIMEOUT,
            grpc.StatusCode.CANCELLED: ExceptionType.CANCELLED_ERROR,
            grpc.StatusCode.RESOURCE_EXHAUSTED: ExceptionType.TRAFFIC_LIMIT_ERROR,
            grpc.StatusCode.INVALID_ARGUMENT: ExceptionType.MM_WRONG_FORMAT_ERROR,
            grpc.StatusCode.UNAVAILABLE: ExceptionType.CONNECT_FAILED,
            grpc.StatusCode.INTERNAL: ExceptionType.MM_PROCESS_ERROR,
        }
        for status, expected in cases.items():
            with self.subTest(status=status):
                rpc = AsyncMock(
                    side_effect=grpc.aio.AioRpcError(status, (), (), "detail")
                )
                with patch(
                    "rtp_llm.server.backend_rpc_server_visitor.MultimodalRpcServiceStub",
                    return_value=SimpleNamespace(RemoteMultimodalEmbedding=rpc),
                ):
                    with self.assertRaises(FtRuntimeException) as caught:
                        await BackendRPCServerVisitor._get_vit_token_ids(
                            visitor, self.vit, GenerateInputPB(), 0.5
                        )
                self.assertEqual(caught.exception.exception_type, expected)


class V41RoutingCacheKeyTest(unittest.TestCase):
    def test_image_identity_matches_engine_block_hash_input(self):
        from rtp_llm.ops import cpp_get_block_cache_keys, get_block_cache_keys

        tokens = [1, 2, 3, 129264, 129264, 129264, 129264, 8, 9, 10, 11, 12]
        image = SimpleNamespace(
            start=3,
            types=torch.zeros(4),
            n_vit_h=2,
            n_vit_w=2,
            content_sha256="a" * 64,
            processor_identity="b" * 64,
        )
        prepared = SimpleNamespace(images=[image])
        # CompleteTokenIds::imageCacheIdentity appends these int32 words to
        # each overlapping block before continuing the rolling hash.
        identity = [-41, 3, 7, 2, 2] + [0xAAAAAAAA] * 8 + [0xBBBBBBBB] * 8
        expected = cpp_get_block_cache_keys(
            [
                tokens[:4] + identity,
                tokens[4:8] + identity,
                tokens[8:],
            ]
        )
        actual = get_block_cache_keys(tokens, 4, prepared)
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual, get_block_cache_keys(tokens, 4))
        image.content_sha256 = "c" * 64
        self.assertNotEqual(actual, get_block_cache_keys(tokens, 4, prepared))
        self.assertEqual(tokens[3], 129264)
        self.assertEqual(len(tokens), 12)

    def test_text_prefix_keys_unchanged_until_first_image_block(self):
        from rtp_llm.ops import get_block_cache_keys

        tokens = list(range(12))
        image = SimpleNamespace(
            start=8,
            types=torch.zeros(4),
            n_vit_h=2,
            n_vit_w=2,
            content_sha256="a" * 64,
            processor_identity="b" * 64,
        )
        text = get_block_cache_keys(tokens, 4)
        self.assertEqual(
            text, get_block_cache_keys(tokens, 4, SimpleNamespace(images=[]))
        )
        with_image = get_block_cache_keys(tokens, 4, SimpleNamespace(images=[image]))
        self.assertEqual(text[:2], with_image[:2])
        self.assertNotEqual(text[2], with_image[2])


class V41PreparedRoutingTest(unittest.IsolatedAsyncioTestCase):
    async def test_local_and_remote_preserve_expanded_metadata(self):
        from rtp_llm.models.multimodal.deepseek_v41_processor import (
            V41ImageInput,
            V41PreparedInputs,
        )
        from rtp_llm.ops import get_block_cache_keys

        for remote in (False, True):
            with self.subTest(remote=remote):
                tokens = (7, 129264, 129264, 129264, 129264, 8, 9, 10)
                image = V41ImageInput(
                    1,
                    torch.ones((4, 3, 14, 14), dtype=torch.bfloat16),
                    2,
                    2,
                    torch.tensor([0, 1, 2, 3]),
                    "a" * 64,
                    "b" * 64,
                )
                prepared = V41PreparedInputs(
                    "", tokens, (-1, 0, 1, 2, 3, -1, -1, -1), (image,)
                )
                request = GenerateInput(
                    request_id=700,
                    token_ids=torch.tensor(tokens, dtype=torch.int32),
                    mm_inputs=[],
                    generate_config=GenerateConfig(timeout_ms=1000),
                    v41_inputs=prepared,
                )
                visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
                visitor.remote_vit = remote
                visitor.dsv4_image_token_id = None
                visitor.mm_padding_size = 0
                visitor.seq_size_per_block = 4
                visitor._page_rr_route_cache_keys = False
                visitor._page_rr_cp_size = 1
                visitor._report_recent_cache_key_metrics = Mock()
                visitor._get_vit_token_ids = AsyncMock()
                vit = RoleAddr(
                    role=RoleType.VIT, ip="vit", grpc_port=8001, http_port=8000
                )
                prefill = RoleAddr(
                    role=RoleType.PREFILL, ip="prefill", grpc_port=9001, http_port=9000
                )
                responses = ([FlexlbResponse.ok([vit])] if remote else []) + [
                    FlexlbResponse.ok([prefill], enqueued_by_master=True)
                ]
                visitor.master_client = SimpleNamespace(
                    get_backend_role_addrs=AsyncMock(side_effect=responses),
                    master_config=SimpleNamespace(master_default_timeout_ms=1000),
                )
                with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor.report"):
                    self.assertIsNone(await visitor.get_master_route_addrs(request))
                calls = visitor.master_client.get_backend_role_addrs.await_args_list
                self.assertEqual(len(calls), 2 if remote else 1)
                if remote:
                    self.assertTrue(calls[0].kwargs["vit_only"])
                visitor._get_vit_token_ids.assert_not_awaited()
                last = calls[-1].kwargs
                self.assertEqual(last["input"].prompt_length, len(tokens))
                self.assertEqual(
                    last["block_cache_keys"],
                    get_block_cache_keys(list(tokens), 4, prepared),
                )
                self.assertEqual(list(last["input_pb"].token_ids), list(tokens))
                self.assertEqual(
                    last["input_pb"].v41_inputs.images[0].content_sha256,
                    image.content_sha256,
                )
                self.assertEqual(
                    len(last["input_pb"].v41_inputs.images[0].patches.bf16_data),
                    image.patches.numel() * 2,
                )
                self.assertEqual(
                    request.generate_config.role_addrs,
                    ([vit] if remote else []) + [prefill],
                )


if __name__ == "__main__":
    unittest.main()
