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
    def get_master_addr(self):
        return "master:1234"


class _FakeInputPB:
    def SerializeToString(self):
        return b"serialized-input"


class _FakeMasterClient:
    def __init__(self):
        self.calls = []

    async def get_backend_role_addrs(
        self, block_cache_keys, input, request_id, input_pb_bytes=None
    ):
        self.calls.append(
            {
                "block_cache_keys": block_cache_keys,
                "input": input,
                "request_id": request_id,
                "input_pb_bytes": input_pb_bytes,
            }
        )
        return FlexlbResponse.ok(["prefill-role"], enqueued_by_master=True)


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
    async def test_get_master_route_addrs_passes_pb_and_marks_master_enqueue(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.seq_size_per_block = 16
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
        self.assertEqual(visitor.master_client.calls[0]["request_id"], 456)
        self.assertEqual(
            visitor.master_client.calls[0]["input_pb_bytes"], b"serialized-input"
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


if __name__ == "__main__":
    unittest.main()
