from types import SimpleNamespace
from unittest import TestCase, main

import torch
from PIL import Image

from rtp_llm.models.deepseek_v4_vision import (
    IMAGE_END,
    IMAGE_START,
    DeepSeekV4VisionEmbedding,
    build_image_block,
)


class DeepSeekV4VisionTest(TestCase):
    def test_image_block_alignment(self):
        for start_pos in (0, 1, 2, 3, 18, 24, 405):
            types, permutation = build_image_block(8, 12, start_pos)
            image_start = int((types == IMAGE_START).nonzero().item())
            self.assertEqual((start_pos + image_start) % 4, 3)
            self.assertEqual(int(types[-1]), IMAGE_END)
            self.assertEqual(permutation.numel(), 8 * 12)

    def test_tiny_encoder_uses_multimodal_config(self):
        vision_config = {
            "hidden_size": 8,
            "vision_n_layers": 1,
            "vision_dim": 16,
            "vision_n_heads": 2,
            "vision_inter_dim": 32,
            "vision_patch_size": 2,
            "vision_rope_theta": 10000.0,
            "vision_downsample_ratio": 1,
            "vision_max_n_token": 32,
            "vision_min_pixels": 16,
            "vision_max_wh_ratio": 8,
        }
        encoder = DeepSeekV4VisionEmbedding(
            SimpleNamespace(config=vision_config),
            SimpleNamespace(compute_dtype=torch.float32),
        )
        output = encoder.image_embedding([Image.new("RGB", (4, 4))], start_pos=3)[0]
        self.assertEqual(output.shape, (10, 8))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    main()
