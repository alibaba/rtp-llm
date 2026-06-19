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
    mm_inputs = []
    tokenizer = None
    token_type_ids = []
    headers = None

    def __init__(self, is_streaming=False):
        self.generate_config = _FakeGenerateConfig(is_streaming=is_streaming)


class _FakeHostService:
    service_available = False

    def get_master_addr(self):
        return "master:1234"

    def get_slave_addr(self):
        return None


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
    async def test_route_ips_preserves_master_route_error_code_on_route_error(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.master_config = None
        visitor.host_service = _FakeHostService()
        visitor.backend_role_list = ["PREFILL"]

        async def get_master_route_addrs(_input, excluded_role_addrs=None):
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


class BackendRPCServerVisitorRetryTest(unittest.IsolatedAsyncioTestCase):
    def _visitor(self, model_rpc_client) -> BackendRPCServerVisitor:
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 1024
        visitor.model_rpc_client = model_rpc_client
        visitor.host_service = _FakeHostService()
        visitor.pd_route_retry_on_unavailable = 3
        visitor.fill_request_info = lambda _input: None
        visitor.check_sp_supported = lambda _input: None
        visitor._remember_failed_route_addrs = lambda failed, addrs: None
        return visitor

    async def test_non_streaming_discards_partial_attempt_before_retry(self):
        client = _RetryingModelRpcClient()
        visitor = self._visitor(client)

        stream = await visitor.enqueue(_FakeInput(is_streaming=False))
        outputs = [output async for output in stream]

        self.assertEqual(outputs, ["successful-output"])
        self.assertEqual(client.attempts, 2)

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
