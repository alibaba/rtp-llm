import unittest
from unittest.mock import patch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.master_client import FlexlbResponse


class _FakeTokenIds:
    shape = (3,)


class _FakeGenerateConfig:
    def __init__(self):
        self.role_addrs = []


class _FakeInput:
    request_id = 123
    token_ids = _FakeTokenIds()

    def __init__(self):
        self.generate_config = _FakeGenerateConfig()


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


if __name__ == "__main__":
    unittest.main()
