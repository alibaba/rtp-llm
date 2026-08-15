import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.master_client import FlexlbResponse


class _FakeTokenIds:
    shape = (3,)

    def tolist(self):
        return [1, 2, 3]


class _FakeGenerateConfig:
    def __init__(self):
        self.role_addrs = []


class _FakeInput:
    request_id = 123
    prompt_length = 3
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

    @staticmethod
    def _make_visitor(tp_size, kv_cache_sharded, prefill_cp_size=0):
        pd_sep_config = SimpleNamespace(max_rpc_timeout_ms=1000, decode_entrance=False)
        parallelism_config = SimpleNamespace(tp_size=tp_size)
        prefill_cp_config = SimpleNamespace(
            kv_cache_sharded=kv_cache_sharded,
            prefill_cp_size=prefill_cp_size,
        )
        with patch("rtp_llm.server.backend_rpc_server_visitor.ModelRpcClient"), patch(
            "rtp_llm.server.backend_rpc_server_visitor.HostServiceArgs.create_from_env",
            return_value=SimpleNamespace(),
        ), patch("rtp_llm.server.backend_rpc_server_visitor.HostService"), patch(
            "rtp_llm.server.backend_rpc_server_visitor.MasterClient"
        ), patch.object(
            BackendRPCServerVisitor, "get_backend_role_list", return_value=[]
        ):
            return BackendRPCServerVisitor(
                max_seq_len=4096,
                seq_size_per_block=128,
                pd_sep_config=pd_sep_config,
                addresses=[],
                parallelism_config=parallelism_config,
                prefill_cp_config=prefill_cp_config,
            )

    def test_visitor_enables_page_rr_route_keys_for_cp4_sharded_cache(self):
        visitor = self._make_visitor(tp_size=4, kv_cache_sharded=True)

        self.assertTrue(visitor._page_rr_route_cache_keys)
        self.assertEqual(visitor._page_rr_cp_size, 4)

    def test_visitor_keeps_legacy_route_keys_when_sharding_is_disabled(self):
        visitor = self._make_visitor(tp_size=4, kv_cache_sharded=False)

        self.assertFalse(visitor._page_rr_route_cache_keys)
        self.assertEqual(visitor._page_rr_cp_size, 1)

    def test_visitor_keeps_legacy_route_keys_for_tp1(self):
        visitor = self._make_visitor(tp_size=1, kv_cache_sharded=True)

        self.assertFalse(visitor._page_rr_route_cache_keys)
        self.assertEqual(visitor._page_rr_cp_size, 1)

    def test_visitor_prefers_explicit_prefill_cp_size_for_routing(self):
        visitor = self._make_visitor(
            tp_size=8, kv_cache_sharded=True, prefill_cp_size=4
        )

        self.assertTrue(visitor._page_rr_route_cache_keys)
        self.assertEqual(visitor._page_rr_cp_size, 4)


class BackendRPCServerVisitorRouteIpsTest(unittest.IsolatedAsyncioTestCase):
    async def test_page_rr_canonical_keys_are_sent_to_flexlb_client(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.seq_size_per_block = 128
        visitor._page_rr_route_cache_keys = True
        visitor._page_rr_cp_size = 4
        visitor._report_recent_cache_key_metrics = lambda _keys: None

        calls = []

        class _MasterClient:
            async def get_backend_role_addrs(
                self, *, block_cache_keys, input, request_id
            ):
                calls.append((block_cache_keys, input, request_id))
                return FlexlbResponse.ok(["prefill-route"])

        visitor.master_client = _MasterClient()
        generate_input = _FakeInput()

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.get_block_cache_keys",
            return_value=list(range(10, 22)),
        ), patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
            result = await visitor.get_master_route_addrs(generate_input)

        self.assertIsNone(result)
        self.assertEqual(calls, [([13, 17, 21], generate_input, 123)])
        self.assertEqual(generate_input.generate_config.role_addrs, ["prefill-route"])

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
