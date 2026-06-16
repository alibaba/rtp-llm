import unittest
from unittest.mock import AsyncMock, Mock

import torch

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.ops import RoleType
from rtp_llm.server.backend_rpc_server_visitor import (
    BackendRPCServerVisitor,
    disable_token_only_reuse_for_input_embeddings,
)
from rtp_llm.server.master_client import FlexlbResponse
from rtp_llm.utils.base_model_datatypes import GenerateInput, InputEmbeddings


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
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.seq_size_per_block = 2
        visitor.master_client = Mock()
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
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.seq_size_per_block = 2
        visitor.master_client = Mock()
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
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 16
        visitor.sp_config = None
        visitor.host_service = Mock(service_available=False)
        visitor.model_rpc_client = Mock()
        visitor.model_rpc_client.enqueue = Mock(return_value="stream")
        input = make_generate_input(
            InputEmbeddings(
                embeddings=[torch.zeros((1, 8), dtype=torch.float32)],
                embedding_locs=[1],
            )
        )

        self.assertTrue(input.generate_config.reuse_cache)
        output = await visitor.enqueue(input)

        self.assertEqual(output, "stream")
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
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 16
        visitor.sp_config = None
        visitor.host_service = Mock(service_available=False)
        visitor.model_rpc_client = Mock()
        visitor.model_rpc_client.batch_enqueue = AsyncMock(return_value=[])
        text_input = make_generate_input()
        embedding_input = make_generate_input(
            InputEmbeddings(
                embeddings=[torch.zeros((1, 8), dtype=torch.float32)],
                embedding_locs=[1],
            )
        )

        await visitor.batch_enqueue([text_input, embedding_input])

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
