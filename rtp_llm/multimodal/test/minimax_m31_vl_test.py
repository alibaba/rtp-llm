import unittest
from types import SimpleNamespace

import torch

from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.multimodal.multimodal_mixins.minimax_m31_vl.minimax_m31_vl_mixin import (
    MiniMaxM31VLImageEmbedding,
    MiniMaxM31VLMixin,
    _MiniMaxM31VLVisionGraphCache,
)
from rtp_llm.multimodal.multimodal_mixins.minimax_m31_vl.minimax_m31_vl_vit import (
    MiniMaxM31VLVisionTower,
    get_fused_qkv_checkpoint_names,
)


class MiniMaxM31VLTest(unittest.TestCase):
    def test_m31_only_mixin_registration(self):
        self.assertIs(get_multimodal_mixin_cls("minimax_m31_vl"), MiniMaxM31VLMixin)
        self.assertEqual(
            get_fused_qkv_checkpoint_names(
                "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj.weight"
            ),
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
                "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight",
                "vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.weight",
            ),
        )

    def test_vision_graph_cache_cpu_fallback(self):
        class Visual(torch.nn.Module):
            def forward(self, pixels, grid):
                return pixels.sum(dim=0)

        cache = _MiniMaxM31VLVisionGraphCache(Visual())
        output = cache.run(torch.ones((4, 3)), torch.tensor([[1, 2, 2]]))
        self.assertEqual(output.tolist(), [4.0, 4.0, 4.0])
        self.assertEqual(cache.stats()["fallback"], 1)

    def test_small_vision_forward_cpu(self):
        config = {
            "vision_config": {
                "hidden_size": 24,
                "num_attention_heads": 3,
                "num_hidden_layers": 1,
                "intermediate_size": 48,
                "patch_size": 2,
                "image_size": 4,
            },
            "text_config": {"hidden_size": 32},
            "projector_hidden_size": 32,
        }
        model = MiniMaxM31VLVisionTower(config).to(torch.bfloat16).eval()
        self.assertEqual(
            MiniMaxM31VLImageEmbedding._data_type.fget(SimpleNamespace(visual=model)),
            torch.bfloat16,
        )
        pixels = torch.randn(4, 24, dtype=torch.bfloat16)
        grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
        with torch.no_grad():
            output = model(pixels, grid)
        self.assertEqual(output.shape, (1, 32))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
