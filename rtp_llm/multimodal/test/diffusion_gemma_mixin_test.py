from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch

from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.multimodal.multimodal_mixins.diffusion_gemma.image_processor import (
    DiffusionGemmaImageProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.diffusion_gemma.vision_modules import (
    DiffusionGemmaMultimodalEmbedder,
    DiffusionGemmaVisionConfig,
    DiffusionGemmaVisionTower,
)
from rtp_llm.multimodal.multimodal_mixins.diffusion_gemma_mixin import (
    DiffusionGemmaImageEmbedding,
    DiffusionGemmaMixin,
    DiffusionGemmaVitDeployWeightInfo,
    DiffusionGemmaVitWeight,
)
from rtp_llm.utils.base_model_datatypes import VitParameters


def _small_vision_config():
    return DiffusionGemmaVisionConfig(
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


class DiffusionGemmaMixinTest(TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.vision_config = _small_vision_config()
        config_json = {
            "vision_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 8,
                "patch_size": 2,
                "position_embedding_size": 64,
                "pooling_kernel_size": 2,
                "rope_parameters": {"rope_theta": 100.0},
                "standardize": True,
            },
            "text_config": {"hidden_size": 24},
        }
        params = VitParameters()
        params.config["ckpt_path"] = "unused-local-test-config"
        with patch(
            "builtins.open",
            lambda *a, **k: __import__("io").StringIO(__import__("json").dumps(config_json)),
        ), patch(
            "os.path.exists", lambda p: True
        ), torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.embedding = DiffusionGemmaImageEmbedding(params)
        self.embedding.visual.eval()
        self.embedding.embed_vision.eval()

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

    def test_state_dict_layout_matches_checkpoint_prefix(self):
        tower = DiffusionGemmaVisionTower(self.vision_config)
        names = set(tower.state_dict().keys())
        self.assertIn("patch_embedder.input_proj.weight", names)
        self.assertIn("patch_embedder.position_embedding_table", names)
        self.assertIn("encoder.layers.0.self_attn.q_proj.linear.weight", names)
        self.assertIn("encoder.layers.0.self_attn.q_norm.weight", names)
        self.assertIn("encoder.layers.0.mlp.gate_proj.linear.weight", names)
        self.assertIn("encoder.layers.0.input_layernorm.weight", names)
        self.assertIn("encoder.layers.0.post_feedforward_layernorm.weight", names)
        self.assertIn("std_bias", names)
        self.assertIn("std_scale", names)
        embed = DiffusionGemmaMultimodalEmbedder(32, 24, 1e-6)
        self.assertEqual(
            set(embed.state_dict().keys()),
            {"embedding_projection.weight"},
        )

    def test_image_processor_output_contract(self):
        # 70 is the smallest supported soft-token budget (max_patches=280).
        processor = DiffusionGemmaImageProcessor(
            patch_size=2, max_soft_tokens=70, pooling_kernel_size=2
        )
        image = Mock()
        image.convert.return_value = image
        tensor = torch.randint(0, 256, (3, 8, 16), dtype=torch.uint8)
        with patch(
            "torch.from_numpy", return_value=tensor
        ):
            res = processor(image)
        self.assertEqual(res["pixel_values"].shape[1], 280)
        self.assertEqual(res["image_position_ids"].shape[1], 280)
        self.assertLessEqual(res["pixel_values"].max().item(), 1.0)
        padding = (res["image_position_ids"][0, :, 0] == -1)
        self.assertGreater(int(padding.sum()), 0)
        self.assertTrue(
            torch.all(res["pixel_values"][0][padding] == 0)
        )

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
            DiffusionGemmaVitWeight.ckpt_key(
                "visual.encoder.layers.0.mlp.gate_proj.linear.weight"
            ),
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
            ckpt_names = [info.name for info in module_weight.weights]
            self.assertEqual(len(ckpt_names), 1)
            self.assertEqual(
                ckpt_names[0], DiffusionGemmaVitWeight.ckpt_key(module_weight.name)
            )

    def test_mixin_registered_for_diffusion_gemma(self):
        self.assertIs(get_multimodal_mixin_cls("diffusion_gemma"), DiffusionGemmaMixin)

    def test_vision_tower_pooler_averages_pooled_patches(self):
        # Hand-computed pooling: k=2, one pooled group per 2x2 patch block.
        config = _small_vision_config()
        config.num_hidden_layers = 0
        tower = DiffusionGemmaVisionTower(config).to(torch.float32)
        # Zero out all weights so hidden == patch embeddings + positions;
        # use a delta on position table row 0 to give each patch a distinct value.
        with torch.no_grad():
            for p in tower.parameters():
                p.zero_()
            tower.patch_embedder.position_embedding_table.zero_()
            tower.patch_embedder.position_embedding_table[0, 3, 0] = 1.0
            tower.patch_embedder.position_embedding_table[1, 2, 0] = 1.0
            tower.std_bias.zero_()
            tower.std_scale.fill_(1.0)
        pixels = torch.zeros(1, 16, 12)
        positions = torch.zeros(1, 16, 2, dtype=torch.long)
        for i in range(16):
            positions[0, i, 0] = i % 4
            positions[0, i, 1] = i // 4
        with torch.inference_mode():
            soft = tower(pixel_values=pixels, pixel_position_ids=positions)
        # Patch value (channel 0) = (x == 3) + (y == 2); each pooled token is
        # the 2x2 block mean times sqrt(hidden):
        #   block (0,0) -> 0, (1,0) -> 0.5, (0,1) -> 0.5, (1,1) -> 1.0
        expect = torch.tensor([0.0, 0.5, 0.5, 1.0]) * (32**0.5)
        self.assertEqual(tuple(soft.shape), (4, 32))
        self.assertTrue(
            torch.allclose(soft[:, 0], expect, atol=1e-4),
            f"channel-0 mismatch: {soft[:, 0]} vs {expect}",
        )
        self.assertTrue(torch.all(soft[:, 1:] == 0))


if __name__ == "__main__":
    main()
