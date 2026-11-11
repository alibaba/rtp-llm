from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch

from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import qwen3_5_moe_mixin
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    Qwen3_5MoeVisionConfig,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType, VitParameters


class Qwen35EmbeddingTest(TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        config = Qwen3_5MoeVisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=24,
            num_position_embeddings=16,
        )
        params = VitParameters()
        params.config["ckpt_path"] = "unused-local-test-config"
        # Mock downloads only: execute the real Qwen3.5 ViT and its real
        # BaseModelOutputWithPooling contract, which has no DeepStack fields.
        with patch.object(
            qwen3_5_moe_mixin.AutoProcessor, "from_pretrained", return_value=Mock()
        ), patch.object(
            qwen3_5_moe_mixin.Qwen2VLImageProcessor,
            "from_pretrained",
            return_value=Mock(),
        ), patch.object(
            Qwen3_5MoeVisionConfig, "from_pretrained", return_value=config
        ), patch.object(
            qwen3_5_moe_mixin, "default_attn_impl", "sdpa"
        ), torch.random.fork_rng(
            devices=[]
        ):
            torch.manual_seed(42)
            self.embedding = qwen3_5_moe_mixin.Qwen3_5MoeImageEmbedding(params)
        self.embedding.visual.eval()

    def media(self, frames=1):
        generator = torch.Generator().manual_seed(frames)
        pixels = torch.randn(frames * 16, 24, generator=generator)
        grid = torch.tensor([[frames, 4, 4]], dtype=torch.int64)
        return pixels, grid

    def test_default_video_and_mixed_batches_match_real_single_media_forward(self):
        cases = (
            [MMUrlType.IMAGE, MMUrlType.IMAGE],
            [MMUrlType.DEFAULT, MMUrlType.DEFAULT],
            [MMUrlType.VIDEO, MMUrlType.VIDEO],
            [MMUrlType.IMAGE, MMUrlType.VIDEO],
            [MMUrlType.DEFAULT, MMUrlType.IMAGE],
        )
        for mm_types in cases:
            with self.subTest(mm_types=mm_types):
                data = [
                    self.media(2 if kind == MMUrlType.VIDEO else 1) for kind in mm_types
                ]
                expected = [
                    self.embedding.embedding(item, mm_type=kind)
                    for item, kind in zip(data, mm_types)
                ]
                actual = self.embedding.batched_embedding(data, mm_types)
                self.assertEqual(len(actual), len(expected))
                for result, single in zip(actual, expected):
                    self.assertEqual(len(result), 2)
                    self.assertEqual(result[0].shape[0], result[1].shape[0])
                    for tensor, reference in zip(result, single):
                        torch.testing.assert_close(tensor, reference)

    def test_empty_batch(self):
        self.assertEqual(self.embedding.batched_embedding([], []), [])

    def test_mismatched_batch_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            self.embedding.batched_embedding([self.media()], [])


if __name__ == "__main__":
    main()
