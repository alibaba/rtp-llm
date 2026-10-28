import pickle
import unittest
from unittest.mock import AsyncMock, Mock, patch

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.ops import PDSepConfig, RoleType
from rtp_llm.server.backend_rpc_server_visitor import (
    BackendRPCServerVisitor,
    disable_token_only_reuse_for_input_embeddings,
)
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.host_service import HostServiceArgs
from rtp_llm.server.master_client import FlexlbResponse
from rtp_llm.server.misc import format_exception
from rtp_llm.utils.base_model_datatypes import GenerateInput, InputEmbeddings


def make_visitor():
    """Exercise constructor defaults without opening clients or discovering hosts."""
    module = "rtp_llm.server.backend_rpc_server_visitor"
    with (
        patch(f"{module}.ModelRpcClient"),
        patch(f"{module}.MasterClient"),
        patch(f"{module}.HostService") as host_service,
        patch(
            f"{module}.HostServiceArgs.create_from_env", return_value=HostServiceArgs()
        ),
    ):
        host_service.return_value.service_available = False
        host_service.return_value.get_master_addr.return_value = "master:1234"
        host_service.return_value.get_backend_role_addrs.return_value = []
        return BackendRPCServerVisitor(
            max_seq_len=16,
            seq_size_per_block=2,
            pd_sep_config=PDSepConfig(),
            addresses=["worker:9000"],
        )


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


class _FakeHostService:
    service_available = False

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
        for error_code in (int(ExceptionType.MASTER_NO_AVAILABLE_WORKER), 8499):
            with self.subTest(error_code=error_code):
                visitor = make_visitor()
                visitor.get_master_route_addrs = AsyncMock(
                    return_value=FlexlbResponse.error_response(error_code, "no worker")
                )
                with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
                    with self.assertRaises(FtRuntimeException) as ctx:
                        await visitor.route_ips(make_generate_input())

                error = ctx.exception
                self.assertEqual(error.exception_type, ExceptionType.ROUTE_ERROR)
                self.assertEqual(error.rtp_error_code, error_code)
                self.assertIn("request_id=123", error.message)
                self.assertIn("no worker", error.message)
                visitor.host_service.get_backend_role_addrs.assert_not_called()
                restored = pickle.loads(pickle.dumps(error))
                self.assertEqual(restored.exception_type, ExceptionType.ROUTE_ERROR)
                self.assertEqual(restored.message, error.message)
                self.assertEqual(restored.rtp_error_code, error_code)
                formatted = format_exception(restored)
                self.assertEqual(formatted["error_code"], error_code)
                self.assertEqual(
                    formatted["error_code_str"],
                    "8400_MASTER_NO_AVAILABLE_WORKER" if error_code == 8400 else "8499",
                )

    async def test_connection_failure_falls_back_to_domain(self):
        visitor = make_visitor()
        visitor.backend_role_list = [RoleType.PREFILL]
        visitor.get_master_route_addrs = AsyncMock(
            return_value=FlexlbResponse.connection_failed_response()
        )
        role_addr = RoleAddr(
            role=RoleType.PREFILL, ip="127.0.0.1", http_port=1, grpc_port=2
        )
        visitor.host_service.get_backend_role_addrs.return_value = [role_addr]
        input = make_generate_input()
        with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
            await visitor.route_ips(input)
        self.assertEqual(input.generate_config.role_addrs, [role_addr])
        visitor.host_service.get_backend_role_addrs.assert_called_once_with(
            [RoleType.PREFILL]
        )

    async def test_empty_domain_route_keeps_generic_error(self):
        visitor = make_visitor()
        visitor.host_service.get_master_addr.return_value = ""
        with patch("rtp_llm.server.backend_rpc_server_visitor.kmonitor"):
            with self.assertRaises(FtRuntimeException) as ctx:
                await visitor.route_ips(make_generate_input())
        self.assertIsNone(ctx.exception.rtp_error_code)
        self.assertEqual(format_exception(ctx.exception)["error_code"], 8500)

    def test_default_exception_format_and_pickle_remain_compatible(self):
        error = FtRuntimeException(ExceptionType.INVALID_PARAMS, "bad input")
        restored = pickle.loads(pickle.dumps(error))
        self.assertEqual(restored.exception_type, error.exception_type)
        self.assertIsNone(restored.rtp_error_code)
        restored.aux_info = {"input_len": 4}
        self.assertEqual(
            format_exception(restored),
            {
                "error_code": 605,
                "error_code_str": "605_INVALID_PARAMS",
                "message": "bad input",
                "aux_info": {"input_len": 4},
            },
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


def make_generate_input(input_embeddings=None):
    return GenerateInput(
        request_id=123,
        token_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        mm_inputs=[],
        generate_config=GenerateConfig(max_new_tokens=1),
        input_embeddings=input_embeddings,
    )


class BackendRPCServerVisitorTest(unittest.IsolatedAsyncioTestCase):
    async def test_master_route_uses_token_cache_keys_without_input_embeddings(self):
        visitor = make_visitor()
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

    async def test_master_route_skips_token_cache_keys_with_input_embeddings(self):
        visitor = make_visitor()
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
        input = make_generate_input(
            InputEmbeddings(
                embeddings=[torch.zeros((1, 8), dtype=torch.float32)],
                embedding_locs=[1],
            )
        )

        await visitor.get_master_route_addrs(input)

        kwargs = visitor.master_client.get_backend_role_addrs.call_args.kwargs
        self.assertEqual(kwargs["block_cache_keys"], [])

    async def test_enqueue_disables_token_only_reuse_with_input_embeddings(self):
        visitor = make_visitor()
        visitor.model_rpc_client = _SuccessfulModelRpcClient(["output"])
        input = make_generate_input(
            InputEmbeddings(
                embeddings=[torch.zeros((1, 8), dtype=torch.float32)],
                embedding_locs=[1],
            )
        )

        self.assertTrue(input.generate_config.reuse_cache)
        stream = await visitor.enqueue(input)
        outputs = [output async for output in stream]

        self.assertEqual(outputs, ["output"])
        self.assertEqual(visitor.model_rpc_client.attempts, 1)
        self.assertEqual(input.request_info.source_role, "frontend")
        self.assertFalse(input.generate_config.reuse_cache)
        self.assertFalse(input.generate_config.enable_device_cache)
        self.assertFalse(input.generate_config.enable_memory_cache)
        self.assertFalse(input.generate_config.enable_remote_cache)

    def test_check_sp_supported_disables_sp_with_input_embeddings(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.sp_config = Mock(model_type="mtp")
        input = make_generate_input(
            InputEmbeddings(
                embeddings=[torch.zeros((1, 8), dtype=torch.float32)],
                embedding_locs=[1],
            )
        )

        visitor.check_sp_supported(input)

        self.assertTrue(input.generate_config.force_disable_sp_run)

    async def test_batch_enqueue_disables_token_only_reuse_with_input_embeddings(self):
        visitor = make_visitor()
        visitor.model_rpc_client.batch_enqueue = AsyncMock(return_value=[])
        text_input = make_generate_input()
        embedding_input = make_generate_input(
            InputEmbeddings(
                embeddings=[torch.zeros((1, 8), dtype=torch.float32)],
                embedding_locs=[1],
            )
        )

        outputs = await visitor.batch_enqueue([text_input, embedding_input])

        self.assertEqual(outputs, [])
        visitor.model_rpc_client.batch_enqueue.assert_awaited_once_with(
            [text_input, embedding_input]
        )
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
