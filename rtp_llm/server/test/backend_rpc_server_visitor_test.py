import unittest
from unittest.mock import patch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.master_client import FlexlbResponse


class _FakeTokenIds:
    shape = (3,)


class _FakeGenerateConfig:
    def __init__(self, is_streaming=False):
        self.role_addrs = []
        self.is_streaming = is_streaming
        self.max_new_tokens = 16

    def validate(self):
        return None


class _FakeInput:
    request_id = 123
    prompt_length = 17
    token_ids = _FakeTokenIds()
    headers = None

    def __init__(self, is_streaming=False):
        self.generate_config = _FakeGenerateConfig(is_streaming=is_streaming)


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
                "cache_key_block_size": cache_key_block_size,
                "input": input,
                "request_id": request_id,
                "input_pb": input_pb,
            }
        )
        return FlexlbResponse.ok(["prefill-role"], enqueued_by_master=True)


class _FakeRoleAddr:
    def __init__(self, role, ip):
        self.role = role
        self.ip = ip


class _FallbackHostService:
    def __init__(self):
        self.requested_roles = []

    def get_master_addr(self):
        return "master:1234"

    def get_backend_role_addrs(self, roles):
        self.requested_roles.append(roles)
        return [_FakeRoleAddr("PREFILL", "vipserver-prefill")]


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
        visitor._page_rr_route_cache_keys = False
        visitor._page_rr_cp_size = 1
        visitor.master_client = _FakeMasterClient()
        visitor._route_cache_keys = lambda keys: keys
        visitor._report_recent_cache_key_metrics = lambda keys: None

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
        self.assertEqual(visitor.master_client.calls[0]["cache_key_block_size"], 16)
        self.assertEqual(visitor.master_client.calls[0]["request_id"], 456)
        self.assertEqual(
            visitor.master_client.calls[0]["input_pb"].SerializeToString(),
            b"serialized-input",
        )

    async def test_connection_failure_falls_back_to_vipserver(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.master_config = None
        visitor.host_service = _FallbackHostService()
        visitor.backend_role_list = ["PREFILL"]

        async def get_master_route_addrs(_input):
            return FlexlbResponse.connection_failed_response()

        visitor.get_master_route_addrs = get_master_route_addrs
        input = _FakeInput()

        with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
            await visitor.route_ips(input)

        self.assertEqual(visitor.host_service.requested_roles, [["PREFILL"]])
        self.assertEqual(len(input.generate_config.role_addrs), 1)
        self.assertEqual(input.generate_config.role_addrs[0].ip, "vipserver-prefill")

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


class _RetryingModelRpcClient:
    def __init__(self):
        self.attempts = 0

    async def enqueue(self, _input):
        self.attempts += 1
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


class BackendRPCServerVisitorRetryTest(unittest.IsolatedAsyncioTestCase):
    def _visitor(self, model_rpc_client) -> BackendRPCServerVisitor:
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 1024
        visitor.model_rpc_client = model_rpc_client
        visitor.host_service = _FakeHostService()
        visitor.pd_route_retry_on_unavailable = 3
        visitor.fill_request_info = lambda _input: None
        visitor.check_sp_supported = lambda _input: None
        return visitor

    async def test_non_streaming_discards_partial_attempt_before_retry(self):
        client = _RetryingModelRpcClient()
        visitor = self._visitor(client)

        stream = await visitor.enqueue(_FakeInput(is_streaming=False))
        outputs = [output async for output in stream]

        self.assertEqual(outputs, ["successful-output"])
        self.assertEqual(client.attempts, 2)

    async def test_non_streaming_replays_successful_outputs_in_order(self):
        client = _SuccessfulModelRpcClient(["first-output", "second-output"])
        visitor = self._visitor(client)

        stream = await visitor.enqueue(_FakeInput(is_streaming=False))
        outputs = [output async for output in stream]

        self.assertEqual(outputs, ["first-output", "second-output"])
        self.assertEqual(client.attempts, 1)

    async def test_non_streaming_raises_after_retry_budget_exhausted(self):
        client = _AlwaysFailingModelRpcClient(
            RuntimeError("StatusCode.UNAVAILABLE recvmsg:Connection timed out")
        )
        visitor = self._visitor(client)
        visitor.pd_route_retry_on_unavailable = 1

        stream = await visitor.enqueue(_FakeInput(is_streaming=False))
        outputs = []
        with self.assertRaisesRegex(RuntimeError, "StatusCode.UNAVAILABLE"):
            async for output in stream:
                outputs.append(output)

        self.assertEqual(outputs, [])
        self.assertEqual(client.attempts, 2)

    async def test_non_streaming_non_retryable_error_does_not_retry(self):
        client = _AlwaysFailingModelRpcClient(ValueError("bad output"))
        visitor = self._visitor(client)

        stream = await visitor.enqueue(_FakeInput(is_streaming=False))
        outputs = []
        with self.assertRaisesRegex(ValueError, "bad output"):
            async for output in stream:
                outputs.append(output)

        self.assertEqual(outputs, [])
        self.assertEqual(client.attempts, 1)

    async def test_streaming_does_not_retry_after_partial_output_yielded(self):
        client = _RetryingModelRpcClient()
        visitor = self._visitor(client)

        stream = await visitor.enqueue(_FakeInput(is_streaming=True))
        outputs = []
        with self.assertRaisesRegex(RuntimeError, "StatusCode.UNAVAILABLE"):
            async for output in stream:
                outputs.append(output)

        self.assertEqual(outputs, ["partial-output-from-failed-attempt"])
        self.assertEqual(client.attempts, 1)


if __name__ == "__main__":
    unittest.main()
