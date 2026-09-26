"""K3 image-token and MoonViT contracts without a text-model forward."""

import json
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_config import (
    configure_kimi_k3_multimodal,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    KimiK3VisionProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_mixin import KimiK3Mixin
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit import (
    KimiK3ImageEmbedding,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


_CONFIG_PATH = Path(__file__).parent / "testdata/kimi_k3/preprocessor_config.json"


def _media_config():
    return json.loads(_CONFIG_PATH.read_text())["media_proc_cfg"]


class KimiK3MultimodalTest(unittest.TestCase):
    def test_checkpoint_geometry_and_pixel_values(self) -> None:
        processor = KimiK3VisionProcessor(_media_config())
        image = Image.new("RGB", (28, 28), (128, 64, 32))
        processed = processor.preprocess({"image": image}, return_tensors="pt")

        self.assertEqual(tuple(processed.pixel_values.shape), (4, 3, 14, 14))
        self.assertEqual(processed.grid_thws.tolist(), [[1, 2, 2]])
        self.assertEqual(processor.media_tokens_calculator({"image": image}), 1)
        expected = torch.tensor([2 * value / 255 - 1 for value in (128, 64, 32)])
        torch.testing.assert_close(processed.pixel_values[0, :, 0, 0], expected)
        self.assertEqual(
            processor.make_image_prompt(28, 28),
            "<|media_begin|>image 28x28<|media_content|><|media_pad|><|media_end|>",
        )

    def test_transparent_pixels_use_configured_background(self) -> None:
        media_config = _media_config()
        media_config["transparent_bg_config"] = {"pattern": "black"}
        media_config["transparent_bg_fill_stage"] = "before_resize"
        image = Image.new("RGBA", (28, 28), (255, 0, 0, 0))
        processed = KimiK3VisionProcessor(media_config).preprocess({"image": image})
        torch.testing.assert_close(
            processed.pixel_values,
            torch.full((4, 3, 14, 14), -1.0),
        )

    def test_config_binds_single_placeholder_and_registered_mixin(self) -> None:
        self.assertIs(get_multimodal_mixin_cls("kimi_k3"), KimiK3Mixin)
        config = ModelConfig()
        config.ckpt_path = str(_CONFIG_PATH.parent)
        configure_kimi_k3_multimodal(
            config,
            {
                "vision_config": {"vt_hidden_size": 8, "_name_or_path": "unused"},
                "media_placeholder_token_id": 163605,
                "image_placeholder": "<|kimi_image_placeholder|>",
            },
        )
        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[163605]])
        self.assertEqual(
            config.mm_related_params.special_token_ids["image_token_index"], 163605
        )
        self.assertNotIn("_name_or_path", config.mm_related_params.config["vision_config"])
        self.assertEqual(config.mm_related_params.config["media_proc_cfg"]["patch_size"], 14)

    def test_image_prompt_embeds_text_on_both_sides_of_vision(self) -> None:
        embedding = object.__new__(KimiK3ImageEmbedding)
        embedding.image_processor = KimiK3VisionProcessor(_media_config())
        embedding._word_embedding_weight = torch.arange(80, dtype=torch.float32).reshape(20, 4)
        prompt = embedding.image_processor.make_image_prompt(28, 28)
        embedding._tokenizer = SimpleNamespace(
            encode=lambda text: [3, 4, 7, 5] if text == prompt else [7]
        )
        features = torch.tensor([[100.0] * 4, [200.0] * 4])

        result = embedding._assemble_image(Image.new("RGB", (28, 28)), features)
        expected = torch.cat(
            [embedding._word_embedding_weight[[3, 4]], features, embedding._word_embedding_weight[[5]]]
        )
        torch.testing.assert_close(result, expected)

    def test_direct_image_bytes_honor_size_cap_and_media_type(self) -> None:
        output = BytesIO()
        Image.new("RGB", (28, 28)).save(output, format="PNG")
        image_bytes = output.getvalue()
        mm_input = SimpleNamespace(
            mm_type=MMUrlType.IMAGE,
            tensor=torch.tensor(list(image_bytes), dtype=torch.uint8),
            url="",
        )
        vit_config = VitConfig()
        vit_config.mm_image_max_file_size_kb = 1
        image = KimiK3ImageEmbedding.preprocess_input([mm_input], vit_config)
        self.assertEqual(image.size, (28, 28))
        mm_input.tensor = torch.tensor(
            list(image_bytes + b"0" * 1024), dtype=torch.uint8
        )
        with self.assertRaisesRegex(ValueError, "per-image limit"):
            KimiK3ImageEmbedding.preprocess_input([mm_input], vit_config)
        mm_input.mm_type = MMUrlType.VIDEO
        with self.assertRaisesRegex(ValueError, "only supports image"):
            KimiK3ImageEmbedding.preprocess_input([mm_input], vit_config)

    def test_tiny_moonvit_projects_one_merged_patch(self) -> None:
        torch.manual_seed(1)
        embedding = KimiK3ImageEmbedding(
            SimpleNamespace(
                config={
                    "vision_config": {
                        "vt_hidden_size": 8,
                        "qkv_hidden_size": 8,
                        "vt_intermediate_size": 16,
                        "vt_num_attention_heads": 1,
                        "vt_num_hidden_layers": 1,
                        "text_hidden_size": 16,
                        "init_pos_emb_height": 2,
                        "init_pos_emb_width": 2,
                    },
                    "media_proc_cfg": _media_config(),
                }
            )
        )
        embedding.vision_tower.eval()
        embedding.mm_projector.eval()
        image = Image.new("RGB", (28, 28))
        features = embedding.image_embedding([image, image])
        serial = embedding.image_embedding([image])[0]
        self.assertEqual(len(features), 2)
        for one_image in features:
            self.assertEqual(tuple(one_image.shape), (1, 16))
            self.assertTrue(torch.isfinite(one_image).all().item())
            torch.testing.assert_close(one_image, serial)


if __name__ == "__main__":
    unittest.main()
