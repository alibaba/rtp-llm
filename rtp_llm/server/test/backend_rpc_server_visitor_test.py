import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.master_client import FlexlbResponse
from rtp_llm.utils.base_model_datatypes import GenerateOutput, GenerateOutputs


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


class BackendRPCServerVisitorFrontendStopIdsTest(unittest.TestCase):
    def test_set_frontend_stop_word_ids_merges_eos_and_renderer_stops(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)

        visitor.set_frontend_stop_word_ids(
            eos_token_id=151329,
            stop_word_ids_list=[[151336], [151338]],
        )

        self.assertEqual(
            visitor.frontend_stop_word_ids_list,
            [[151329], [151336], [151338]],
        )

    def test_strip_finished_output_ids_at_stop_word(self):
        output = GenerateOutput(
            output_ids=torch.tensor([[10, 151338]], dtype=torch.int32),
            finished=True,
        )

        BackendRPCServerVisitor.strip_frontend_stop_word_ids(
            GenerateOutputs(generate_outputs=[output]),
            [[151338]],
            {},
        )

        self.assertEqual(output.output_ids.tolist(), [[10]])

    def test_strip_streaming_multi_token_stop_across_chunks(self):
        pending = {}
        first = GenerateOutput(
            output_ids=torch.tensor([[7, 8]], dtype=torch.int32),
            finished=False,
        )
        second = GenerateOutput(
            output_ids=torch.tensor([[9]], dtype=torch.int32),
            finished=True,
        )

        BackendRPCServerVisitor.strip_frontend_stop_word_ids(
            GenerateOutputs(generate_outputs=[first]),
            [[8, 9]],
            pending,
        )
        BackendRPCServerVisitor.strip_frontend_stop_word_ids(
            GenerateOutputs(generate_outputs=[second]),
            [[8, 9]],
            pending,
        )

        self.assertEqual(first.output_ids.tolist(), [[7]])
        self.assertEqual(second.output_ids.tolist(), [[]])


if __name__ == "__main__":
    unittest.main()
