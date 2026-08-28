import unittest
from types import SimpleNamespace

import torch

from rtp_llm.multimodal.multimodal_mixins.glm5_next.glm5_next_mixin import (
    Glm5NextVisionAttention,
    Glm5NextVisionModel,
    glm5_smart_resize,
)


class Glm5NextVitTest(unittest.TestCase):
    def test_resize_is_upward_aligned(self):
        height, width = glm5_smart_resize(
            101,
            203,
            temporal_patch_size=2,
            factor=28,
            min_pixels=2 * 28 * 28,
            max_pixels=2 * 280 * 280,
        )
        self.assertEqual(height % 28, 0)
        self.assertEqual(width % 28, 0)
        self.assertLessEqual(2 * height * width, 2 * 280 * 280)

    def test_tiny_vision_forward(self):
        config = SimpleNamespace(
            attention_bias=True,
            depth=1,
            hidden_size=32,
            in_channels=3,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=32,
            patch_size=2,
            projection_intermediate_size=64,
            rms_norm_eps=1e-5,
            spatial_merge_size=2,
            swiglu_limit=10.0,
            temporal_patch_size=2,
        )
        model = Glm5NextVisionModel(config)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
        pixel_values = torch.randn(16, 3 * 2 * 2 * 2)
        self.assertEqual(model._rotary_freqs(grid_thw).shape, (16, 4))
        output = model(pixel_values, grid_thw)
        self.assertEqual(output.shape, (4, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_rope_preserves_bfloat16(self):
        config = SimpleNamespace(
            attention_bias=True,
            hidden_size=32,
            num_heads=4,
        )
        attention = Glm5NextVisionAttention(config)
        q = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        freqs = torch.randn(3, 4)
        q, k = attention._apply_rope(q, k, freqs)
        self.assertEqual(q.dtype, torch.bfloat16)
        self.assertEqual(k.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
