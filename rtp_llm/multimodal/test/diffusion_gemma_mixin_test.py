from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch
from transformers import Gemma4VisionConfig

from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.multimodal.multimodal_mixins.diffusion_gemma_mixin import (
    DiffusionGemmaImageEmbedding,
    DiffusionGemmaMixin,
    DiffusionGemmaVitDeployWeightInfo,
    DiffusionGemmaVitWeight,
)
from rtp_llm.utils.base_model_datatypes import VitParameters


class DiffusionGemmaMixinTest(TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        vision_config = Gemma4VisionConfig(
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            patch_size=2,
            position_embedding_size=64,
            pooling_kernel_size=2,
            rope_theta=100.0,
            standardize=True,
        )
        vision_config._attn_implementation = "eager"
        text_config = SimpleNamespace(hidden_size=24, rms_norm_eps=1e-6)
        params = VitParameters()
        params.config["ckpt_path"] = "unused-local-test-config"
        with patch(
            "rtp_llm.multimodal.multimodal_mixins.diffusion_gemma_mixin.AutoProcessor.from_pretrained",
            return_value=Mock(),
        ), patch(
            "rtp_llm.multimodal.multimodal_mixins.diffusion_gemma_mixin.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(
                vision_config=vision_config, text_config=text_config
            ),
        ), patch(
            "rtp_llm.multimodal.multimodal_mixins.diffusion_gemma_mixin.default_attn_impl",
            "eager",
        ), torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.embedding = DiffusionGemmaImageEmbedding(params)
        self.embedding.visual.eval()
        self.vision_config = vision_config

    def patches_and_positions(self, num_patches=16):
        grid = 4
        pixels = torch.randn(1, num_patches, 3 * 2 * 2)
        positions = torch.zeros(1, num_patches, 2, dtype=torch.long)
        for i in range(num_patches):
            positions[0, i, 0] = i % grid
            positions[0, i, 1] = i // grid
        return pixels, positions

    def test_embedding_returns_text_hidden_features(self):
        data = self.patches_and_positions()
        with torch.inference_mode():
            features, pos_id = self.embedding.embedding(data)
        # pooling_kernel_size=2: 16 patches -> 4 soft tokens.
        self.assertIsNone(pos_id)
        self.assertEqual(tuple(features.shape), (4, 24))
        self.assertEqual(features.dtype, self.embedding._data_type)

    def test_vit_weight_checkpoint_key_mapping(self):
        vit_weight = DiffusionGemmaVitWeight(
            {
                "visual": self.embedding.visual,
                "embed_vision": self.embedding.embed_vision,
            }
        )
        names = vit_weight.weight_names
        self.assertIn("visual.patch_embedder.input_proj.weight", names)
        self.assertIn("visual.std_bias", names)
        self.assertIn("embed_vision.embedding_projection.weight", names)
        self.assertEqual(
            DiffusionGemmaVitWeight.ckpt_key("visual.patch_embedder.input_proj.weight"),
            "model.encoder.vision_tower.patch_embedder.input_proj.weight",
        )
        self.assertEqual(
            DiffusionGemmaVitWeight.ckpt_key("visual.encoder.layers.0.mlp.gate_proj.linear.weight"),
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight",
        )
        self.assertEqual(
            DiffusionGemmaVitWeight.ckpt_key("embed_vision.embedding_projection.weight"),
            "model.encoder.embed_vision.embedding_projection.weight",
        )

    def test_deploy_weight_info_covers_all_vit_weights(self):
        vit_weight = DiffusionGemmaVitWeight(
            {
                "visual": self.embedding.visual,
                "embed_vision": self.embedding.embed_vision,
            }
        )
        deploy_info = DiffusionGemmaVitDeployWeightInfo(
            vit_weights=vit_weight, vit_config=Mock()
        )
        weight_info = deploy_info.get_weight_info()
        declared = {w.name for w in weight_info.weights}
        self.assertEqual(declared, set(vit_weight.weight_names))
        for module_weight in weight_info.weights:
            ckpt_names = [info.name for info in module_weight.weights_info]
            self.assertEqual(len(ckpt_names), 1)
            self.assertEqual(
                ckpt_names[0], DiffusionGemmaVitWeight.ckpt_key(module_weight.name)
            )

    def test_mixin_registered_for_diffusion_gemma(self):
        self.assertIs(get_multimodal_mixin_cls("diffusion_gemma"), DiffusionGemmaMixin)


if __name__ == "__main__":
    main()
