import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.master_client import FlexlbResponse


class _FakeTokenIds:
    shape = (3,)


class _FakeGenerateConfig:
    def __init__(self):
        self.role_addrs = []
        self.max_new_tokens = 1
        self.is_streaming = True

    def validate(self):
        pass


class _FakeInput:
    request_id = 123
    token_ids = _FakeTokenIds()

    def __init__(self):
        self.generate_config = _FakeGenerateConfig()
        self.allow_pd_route_retry = True

    @property
    def prompt_length(self):
        return 1


class _FakeHostService:
    def get_master_addr(self):
        return "master:1234"


class BackendRPCServerVisitorRouteCacheKeysTest(unittest.TestCase):
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


class BackendRPCServerVisitorRouteIpsTest(unittest.IsolatedAsyncioTestCase):
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


class BackendRPCServerVisitorRetryTest(unittest.TestCase):
    def test_pd_route_retry_can_be_disabled_per_request(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.pd_route_retry_on_unavailable = 3

        self.assertEqual(
            visitor._pd_route_retry_limit(SimpleNamespace(allow_pd_route_retry=True)),
            3,
        )
        self.assertEqual(
            visitor._pd_route_retry_limit(SimpleNamespace(allow_pd_route_retry=False)),
            0,
        )

    def test_retryable_route_rpc_error_uses_transport_whitelist(self):
        self.assertTrue(
            BackendRPCServerVisitor._is_retryable_route_rpc_error(
                FtRuntimeException(ExceptionType.CONNECT_FAILED, "connect failed")
            )
        )
        self.assertTrue(
            BackendRPCServerVisitor._is_retryable_route_rpc_error(
                FtRuntimeException(
                    ExceptionType.GET_HOST_FAILED,
                    "decode host disappeared during restart",
                )
            )
        )
        self.assertTrue(
            BackendRPCServerVisitor._is_retryable_route_rpc_error(
                FtRuntimeException(
                    ExceptionType.REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED,
                    "stale decode allocate stream",
                )
            )
        )
        self.assertTrue(
            BackendRPCServerVisitor._is_retryable_route_rpc_error(
                FtRuntimeException(
                    ExceptionType.CACHE_STORE_LOAD_RDMA_WRITE_FAILED,
                    "stale decode rdma write failed",
                )
            )
        )
        self.assertTrue(
            BackendRPCServerVisitor._is_retryable_route_rpc_error(
                FtRuntimeException(
                    ExceptionType.P2P_CONNECTOR_WORKER_READ_TIMEOUT,
                    "stale prefill p2p read timed out",
                )
            )
        )
        self.assertTrue(
            BackendRPCServerVisitor._is_retryable_route_rpc_error(
                FtRuntimeException(
                    ExceptionType.MASTER_NO_AVAILABLE_WORKER,
                    "master has no healthy worker during restart",
                )
            )
        )

        for error_type in (
            ExceptionType.ROUTE_ERROR,
            ExceptionType.OUT_OF_VOCAB_RANGE,
            ExceptionType.REMOTE_GENERATE_FAILED,
            ExceptionType.CANCELLED,
            ExceptionType.P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED,
            ExceptionType.P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH,
        ):
            with self.subTest(error_type=error_type):
                self.assertFalse(
                    BackendRPCServerVisitor._is_retryable_route_rpc_error(
                        FtRuntimeException(error_type, "not a stale endpoint error")
                    )
                )

    def test_route_error_retry_uses_underlying_error_code_whitelist(self):
        connection_error = FtRuntimeException(ExceptionType.ROUTE_ERROR, "route failed")
        connection_error.rtp_error_code = int(ExceptionType.CONNECT_FAILED)
        self.assertTrue(
            BackendRPCServerVisitor._is_retryable_route_rpc_error(connection_error)
        )

        no_worker = FtRuntimeException(ExceptionType.ROUTE_ERROR, "route failed")
        no_worker.rtp_error_code = int(ExceptionType.MASTER_NO_AVAILABLE_WORKER)
        self.assertTrue(BackendRPCServerVisitor._is_retryable_route_rpc_error(no_worker))


class BackendRPCServerVisitorEnqueueRetryTest(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_does_not_retry_when_request_disables_pd_route_retry(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 10
        visitor.sp_config = None
        visitor.pd_route_retry_on_unavailable = 3
        visitor.host_service = SimpleNamespace(service_available=False)
        visitor.fill_request_info = lambda _input: None

        class _FailingModelRpcClient:
            calls = 0

            def enqueue(self, _input):
                self.calls += 1
                raise RuntimeError("StatusCode.UNAVAILABLE")

        client = _FailingModelRpcClient()
        visitor.model_rpc_client = client

        input = _FakeInput()
        input.allow_pd_route_retry = False

        stream = await visitor.enqueue(input)
        with self.assertRaisesRegex(RuntimeError, "StatusCode.UNAVAILABLE"):
            async for _ in stream:
                pass

        self.assertEqual(client.calls, 1)

    async def test_enqueue_does_not_retry_after_stream_yields_output(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 10
        visitor.sp_config = None
        visitor.pd_route_retry_on_unavailable = 3
        visitor.host_service = SimpleNamespace(service_available=False)
        visitor.fill_request_info = lambda _input: None

        class _YieldThenFailModelRpcClient:
            calls = 0

            def enqueue(self, _input):
                self.calls += 1

                async def stream():
                    yield "first"
                    raise FtRuntimeException(
                        ExceptionType.CONNECT_FAILED, "stale backend"
                    )

                return stream()

        client = _YieldThenFailModelRpcClient()
        visitor.model_rpc_client = client

        outputs = []
        stream = await visitor.enqueue(_FakeInput())
        with self.assertRaises(FtRuntimeException) as ctx:
            async for output in stream:
                outputs.append(output)

        self.assertEqual(outputs, ["first"])
        self.assertEqual(ctx.exception.exception_type, ExceptionType.CONNECT_FAILED)
        self.assertEqual(client.calls, 1)

    async def test_enqueue_retries_non_streaming_after_buffered_backend_output(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 10
        visitor.sp_config = None
        visitor.pd_route_retry_on_unavailable = 1
        visitor.host_service = SimpleNamespace(service_available=False)
        visitor.fill_request_info = lambda _input: None

        class _YieldThenFailOnceModelRpcClient:
            calls = 0

            def enqueue(self, _input):
                self.calls += 1

                async def stream():
                    if self.calls == 1:
                        yield "partial-from-failed-attempt"
                        raise FtRuntimeException(
                            ExceptionType.CONNECT_FAILED, "stale backend"
                        )
                    yield "final-from-retry"

                return stream()

        client = _YieldThenFailOnceModelRpcClient()
        visitor.model_rpc_client = client

        input = _FakeInput()
        input.generate_config.is_streaming = False
        stream = await visitor.enqueue(input)
        outputs = []
        async for output in stream:
            outputs.append(output)

        self.assertEqual(outputs, ["final-from-retry"])
        self.assertEqual(client.calls, 2)

    async def test_enqueue_retries_non_client_streaming_internal_streaming_failure(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 10
        visitor.sp_config = None
        visitor.pd_route_retry_on_unavailable = 1
        visitor.host_service = SimpleNamespace(service_available=False)
        visitor.fill_request_info = lambda _input: None

        class _YieldThenFailOnceModelRpcClient:
            calls = 0

            def enqueue(self, _input):
                self.calls += 1

                async def stream():
                    if self.calls == 1:
                        yield "internal-partial-from-failed-attempt"
                        raise FtRuntimeException(
                            ExceptionType.CONNECT_FAILED, "stale backend"
                        )
                    yield "final-from-retry"

                return stream()

        client = _YieldThenFailOnceModelRpcClient()
        visitor.model_rpc_client = client

        input = _FakeInput()
        input.generate_config.is_streaming = True
        input.client_streaming = False
        stream = await visitor.enqueue(input)
        outputs = []
        async for output in stream:
            outputs.append(output)

        self.assertEqual(outputs, ["final-from-retry"])
        self.assertEqual(client.calls, 2)

    async def test_enqueue_retry_uses_refreshed_domain_route(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 10
        visitor.sp_config = None
        visitor.pd_route_retry_on_unavailable = 1
        visitor.fill_request_info = lambda _input: None
        visitor.backend_role_list = [RoleType.PREFILL]

        stale_addr = RoleAddr(
            role=RoleType.PREFILL, ip="10.0.0.1", http_port=100, grpc_port=101
        )
        fresh_addr = RoleAddr(
            role=RoleType.PREFILL, ip="10.0.0.2", http_port=100, grpc_port=101
        )

        async def route_ips(input):
            input.generate_config.role_addrs = [stale_addr]

        visitor.route_ips = route_ips

        class _HostService:
            service_available = True

            def __init__(self):
                self.refresh_calls = []

            def get_backend_role_addrs(self, _roles, refresh=False):
                self.refresh_calls.append(refresh)
                return [fresh_addr]

        host_service = _HostService()
        visitor.host_service = host_service

        class _FlakyModelRpcClient:
            calls = 0

            def enqueue(self, _input):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("StatusCode.UNAVAILABLE")

                async def empty_stream():
                    if False:
                        yield None

                return empty_stream()

        client = _FlakyModelRpcClient()
        visitor.model_rpc_client = client

        input = _FakeInput()
        stream = await visitor.enqueue(input)
        async for _ in stream:
            pass

        self.assertEqual(client.calls, 2)
        self.assertEqual(host_service.refresh_calls, [True])
        self.assertEqual(input.generate_config.role_addrs, [fresh_addr])

    async def test_enqueue_retry_filters_failed_domain_route_addr(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 10
        visitor.sp_config = None
        visitor.pd_route_retry_on_unavailable = 1
        visitor.fill_request_info = lambda _input: None
        visitor.backend_role_list = [RoleType.PREFILL]

        stale_addr = RoleAddr(
            role=RoleType.PREFILL, ip="10.0.0.1", http_port=100, grpc_port=101
        )
        fresh_addr = RoleAddr(
            role=RoleType.PREFILL, ip="10.0.0.2", http_port=100, grpc_port=101
        )

        async def route_ips(input):
            input.generate_config.role_addrs = [stale_addr]

        visitor.route_ips = route_ips

        class _HostService:
            service_available = True

            def get_backend_role_addrs(self, _roles, refresh=False):
                return [stale_addr, fresh_addr]

        visitor.host_service = _HostService()

        class _FlakyModelRpcClient:
            def __init__(self):
                self.calls = 0
                self.seen_role_addrs = []

            def enqueue(self, input):
                self.calls += 1
                self.seen_role_addrs.append(list(input.generate_config.role_addrs))
                if self.calls == 1:
                    raise RuntimeError("StatusCode.UNAVAILABLE")

                async def empty_stream():
                    if False:
                        yield None

                return empty_stream()

        client = _FlakyModelRpcClient()
        visitor.model_rpc_client = client

        input = _FakeInput()
        stream = await visitor.enqueue(input)
        async for _ in stream:
            pass

        self.assertEqual(client.calls, 2)
        self.assertEqual(client.seen_role_addrs, [[stale_addr], [fresh_addr]])
        self.assertEqual(input.generate_config.role_addrs, [fresh_addr])

    async def test_enqueue_retry_rejects_incomplete_filtered_domain_route(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 10
        visitor.sp_config = None
        visitor.pd_route_retry_on_unavailable = 1
        visitor.fill_request_info = lambda _input: None
        visitor.backend_role_list = [RoleType.PREFILL, RoleType.DECODE]

        stale_prefill = RoleAddr(
            role=RoleType.PREFILL, ip="10.0.0.1", http_port=100, grpc_port=101
        )
        stale_decode = RoleAddr(
            role=RoleType.DECODE, ip="10.0.0.3", http_port=300, grpc_port=301
        )
        fresh_decode = RoleAddr(
            role=RoleType.DECODE, ip="10.0.0.4", http_port=400, grpc_port=401
        )

        async def route_ips(input):
            input.generate_config.role_addrs = [stale_prefill, stale_decode]

        visitor.route_ips = route_ips

        class _HostService:
            service_available = True

            def get_backend_role_addrs(self, _roles, refresh=False):
                return [stale_prefill, fresh_decode]

        visitor.host_service = _HostService()

        class _FailOnceModelRpcClient:
            calls = 0

            def enqueue(self, _input):
                self.calls += 1
                raise RuntimeError("StatusCode.UNAVAILABLE")

        client = _FailOnceModelRpcClient()
        visitor.model_rpc_client = client

        stream = await visitor.enqueue(_FakeInput())
        with self.assertRaises(FtRuntimeException) as ctx:
            async for _ in stream:
                pass

        self.assertEqual(ctx.exception.exception_type, ExceptionType.ROUTE_ERROR)
        self.assertIn("missing backend role addresses", str(ctx.exception))
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
