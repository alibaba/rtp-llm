"""Regressions for Qwen3-VL embedding configuration and prefill positions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.qwen3_vl import QWen3_VL
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeKVCachePrefillOpBase,
    RopeContractError,
)


class MropeContractTest(unittest.TestCase):
    def test_checkpoint_rope_parameters_and_legacy_config(self):
        for key in ("rope_parameters", "rope_scaling"):
            with self.subTest(key=key):
                config = ModelConfig()
                QWen3_VL._from_config_json(
                    config,
                    {
                        "vision_start_token_id": 1,
                        "vision_end_token_id": 2,
                        "text_config": {
                            "intermediate_size": 256,
                            "num_attention_heads": 2,
                            "num_key_value_heads": 1,
                            "head_dim": 128,
                            "hidden_size": 256,
                            "num_hidden_layers": 2,
                            "vocab_size": 1024,
                            "rope_theta": 5000000,
                            key: {
                                "rope_theta": 5000000,
                                "mrope_section": [24, 20, 20],
                                "mrope_interleaved": True,
                            },
                        },
                    },
                )
                rope = config.attn_config.rope_config
                self.assertEqual(rope.base, 5000000)
                self.assertEqual(rope.index_factor, 3)
                self.assertEqual(
                    [rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3], [24, 20, 20]
                )
                self.assertTrue(rope.mrope_interleaved)

    def test_embedding_chat_converts_optional_preprocess_fields(self):
        import threading
        from unittest.mock import Mock

        from rtp_llm.models.downstream_modules.common_input_generator import (
            CommonInputGenerator,
        )
        from rtp_llm.models.downstream_modules.embedding.api_datatype import ChatMessage

        generator = CommonInputGenerator.__new__(CommonInputGenerator)
        generator.lock = threading.Lock()
        generator.tokenizer_ = object()
        generator.config_ = SimpleNamespace(max_seq_len=2048, position_ids_style=0)
        renderer = Mock()
        renderer.render_chat.return_value = SimpleNamespace(
            input_ids=[1, 2], multimodal_inputs=[]
        )
        generator.openai_render_info = SimpleNamespace(chat_renderer=renderer)
        message = ChatMessage(
            role="user",
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/image.png"},
                    "preprocess_config": {"mm_timeout_ms": None},
                }
            ],
        )
        with patch("rtp_llm.models.downstream_modules.common_input_generator.kmonitor"):
            generator.generate([message])
        request = renderer.render_chat.call_args.args[0]
        self.assertIsNone(request.messages[0].tool_calls)
        self.assertIsNone(request.messages[0].tool_call_id)
        self.assertIsNotNone(
            request.messages[0].content[0].preprocess_config.mm_timeout_ms
        )
        self.assertIsNone(request.get_chat_template_kwargs())

    def test_url_image_rpc_omits_empty_tensor(self):
        import asyncio
        from unittest.mock import AsyncMock, Mock

        from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint

        async def check(tensor):
            endpoint = EmbeddingEndpoint.__new__(EmbeddingEndpoint)
            endpoint.address, endpoint.options, endpoint.host_service = (
                "unused",
                [],
                None,
            )
            config = SimpleNamespace(
                width=-1,
                height=-1,
                min_pixels=-1,
                max_pixels=-1,
                fps=-1,
                min_frames=-1,
                max_frames=-1,
                crop_positions=[],
                mm_timeout_ms=1000,
            )
            feature = SimpleNamespace(
                url="https://example.test/image.png",
                mm_type=1,
                tensor=tensor,
                mm_preprocess_config=config,
            )
            inputs = SimpleNamespace(
                token_ids=torch.tensor([1]),
                token_type_ids=torch.tensor([0]),
                input_lengths=torch.tensor([1]),
                input_length=1,
                multimodal_inputs=[feature],
            )
            stub = Mock()
            stub.embedding = AsyncMock(
                return_value=SimpleNamespace(output_is_tensor=False, output_map=[])
            )
            channel = Mock(close=AsyncMock())
            with patch(
                "rtp_llm.embedding.embedding_endpoint.grpc.aio.insecure_channel",
                return_value=channel,
            ), patch(
                "rtp_llm.embedding.embedding_endpoint.pb2_grpc.EmbeddingRpcServiceStub",
                return_value=stub,
            ):
                await endpoint.generate_embeddings_grpc(inputs, SimpleNamespace())
            feature_pb = stub.embedding.call_args.args[0].multimodal_features[0]
            self.assertEqual(
                feature_pb.HasField("multimodal_tensor"),
                tensor is not None and tensor.numel() > 0,
            )
            channel.close.assert_awaited_once()

        for tensor in (None, torch.empty(0), torch.ones(2)):
            asyncio.run(check(tensor))

    def test_cuda_rejects_one_axis_positions_before_kernel(self):
        op = FusedRopeKVCachePrefillOpBase(
            SimpleNamespace(
                rope_config=SimpleNamespace(style=RopeStyle.Mrope, index_factor=3)
            )
        )
        for positions in (None, torch.arange(7, dtype=torch.int32)):
            inputs = SimpleNamespace(
                kv_cache_kernel_block_id_device=None,
                combo_position_ids=positions,
                context_parallel_info=None,
                padding_offset=torch.zeros(7, dtype=torch.int32),
            )
            with self.subTest(positions=positions), patch(
                "rtp_llm.ops.fused_rope_kvcache_op._get_fused_rope_kvcache"
            ) as kernel:
                with self.assertRaisesRegex(RopeContractError, "expected at least 21"):
                    op.prepare(inputs)
                kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
