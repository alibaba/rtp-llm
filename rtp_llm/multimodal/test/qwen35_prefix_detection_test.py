import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen35MoeWeight
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
    Qwen3_5MoeVitWeight,
)


class Qwen35PrefixDetectionTest(unittest.TestCase):
    def test_image_position_ids_follow_thw_order_and_handle_empty_grid(self):
        embedding = SimpleNamespace(visual=SimpleNamespace(spatial_merge_size=2))
        grid_thw = torch.tensor([[2, 4, 2], [0, 4, 2]], dtype=torch.int32)

        position_ids = Qwen3_5MoeImageEmbedding.get_position_ids(embedding, grid_thw)

        self.assertEqual((4, 3), tuple(position_ids[0].shape))
        torch.testing.assert_close(
            position_ids[0],
            torch.tensor(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                dtype=torch.int32,
            ),
        )
        self.assertEqual((0, 3), tuple(position_ids[1].shape))

    def test_detects_legacy_checkpoint_prefixes(self):
        llm_weight = object.__new__(Qwen35MoeWeight)
        llm_weight._has_stacked_ckpt = False
        llm_weight._process_meta(
            {},
            {
                "mtp.layers.0.input_layernorm.weight",
                "model.language_model.layers.0.input_layernorm.weight",
            },
        )
        self.assertEqual("model.language_model.", llm_weight.prefix)

        vit_weight = object.__new__(Qwen3_5MoeVitWeight)
        vit_weight.detect_ckpt_prefix(
            ["model.visual.blocks.0.norm1.weight"],
        )
        self.assertEqual("model.visual.", vit_weight._ckpt_prefix)

    def test_detects_transformers_540_converted_checkpoint_prefixes(self):
        llm_weight = object.__new__(Qwen35MoeWeight)
        llm_weight._has_stacked_ckpt = False
        llm_weight._process_meta(
            {},
            {
                "model.language_model.language_model.language_model.layers.0.input_layernorm.weight",
                "model.language_model.language_model.language_model.layers.0.mlp.experts.gate_up_proj",
            },
        )
        self.assertEqual(
            "model.language_model.language_model.language_model.",
            llm_weight.prefix,
        )
        self.assertTrue(llm_weight._has_stacked_ckpt)

        vit_weight = object.__new__(Qwen3_5MoeVitWeight)
        vit_weight.detect_ckpt_prefix(
            ["model.language_model.visual.blocks.0.norm1.weight"],
        )
        self.assertEqual("model.language_model.visual.", vit_weight._ckpt_prefix)

    def test_detects_visual_prefix_from_nearest_visual_anchor(self):
        vit_weight = object.__new__(Qwen3_5MoeVitWeight)
        vit_weight.detect_ckpt_prefix(
            ["model.visual.language_model.visual.blocks.0.norm1.weight"],
        )
        self.assertEqual(
            "model.visual.language_model.visual.",
            vit_weight._ckpt_prefix,
        )

    def test_raises_when_required_prefix_anchor_is_missing(self):
        llm_weight = object.__new__(Qwen35MoeWeight)
        llm_weight._has_stacked_ckpt = False
        with self.assertRaisesRegex(ValueError, "cannot determine prefix"):
            llm_weight._process_meta({}, {"mtp.layers.0.input_layernorm.weight"})

        vit_weight = object.__new__(Qwen3_5MoeVitWeight)
        with self.assertRaisesRegex(ValueError, "cannot determine visual prefix"):
            vit_weight.detect_ckpt_prefix(["model.text.blocks.0.norm1.weight"])


if __name__ == "__main__":
    unittest.main()
