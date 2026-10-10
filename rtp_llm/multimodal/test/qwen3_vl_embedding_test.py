from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import Qwen3_VLImageEmbedding
from rtp_llm.utils.base_model_datatypes import MMUrlType, VitParameters


class Qwen3VLEmbeddingTest(TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        config = Qwen3VLVisionConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=24,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1],
        )
        config._attn_implementation = "eager"
        # Instantiate the real vision model locally; no checkpoint or processor
        # download is needed to exercise the production output contract.
        params = VitParameters()
        params.config["ckpt_path"] = "unused-local-test-config"
        with patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin.AutoProcessor.from_pretrained",
            return_value=Mock(),
        ), patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin.Qwen2VLImageProcessor.from_pretrained",
            return_value=Mock(),
        ), patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin.Qwen3VLConfig.from_pretrained",
            return_value=SimpleNamespace(vision_config=config),
        ), patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin.default_attn_impl",
            "eager",
        ), torch.random.fork_rng(
            devices=[]
        ):
            torch.manual_seed(42)
            self.embedding = Qwen3_VLImageEmbedding(params)
        self.embedding.visual.eval()

    def media(self, height, width):
        generator = torch.Generator().manual_seed(height * 10 + width)
        pixels = torch.randn(height * width, 24, generator=generator)
        return pixels, torch.tensor([[1, height, width]], dtype=torch.int64)

    def test_embedding_uses_pooled_output_and_preserves_deepstack(self):
        data = self.media(4, 4)
        with torch.inference_mode():
            reference = self.embedding.visual(data[0], grid_thw=data[1])
        features, positions, deepstack = self.embedding.embedding(data)
        self.assertEqual(features.shape, (4, 24))
        self.assertEqual(positions.shape, (4, 3))
        self.assertEqual(deepstack.shape, (2 * 4 * 24,))
        torch.testing.assert_close(features, reference.pooler_output)
        torch.testing.assert_close(
            deepstack, torch.stack(reference.deepstack_features).flatten()
        )

    def test_batch_preserves_each_image_embedding_and_deepstack(self):
        data = [self.media(4, 4), self.media(2, 4)]
        singles = [self.embedding.embedding(item) for item in data]
        batched = self.embedding.batched_embedding(
            data, [MMUrlType.IMAGE, MMUrlType.IMAGE]
        )
        self.assertEqual(len(batched), 2)
        self.assertEqual([result[0].shape[0] for result in batched], [4, 2])
        for actual, expected in zip(batched, singles):
            for actual_tensor, expected_tensor in zip(actual, expected):
                torch.testing.assert_close(actual_tensor, expected_tensor)


if __name__ == "__main__":
    main()
