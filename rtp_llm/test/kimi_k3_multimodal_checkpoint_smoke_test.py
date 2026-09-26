"""Checkpoint-backed K3 image prompt assembly without text-model execution."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_config import (
    configure_kimi_k3_multimodal,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    KimiK3VisionProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit import (
    KimiK3ImageEmbedding,
)


class KimiK3MultimodalCheckpointSmokeTest(unittest.TestCase):
    def test_native_image_prompt_and_text_embedding_shard(self) -> None:
        checkpoint = Path(os.environ["K3_CKPT_PATH"])
        top_config = json.loads((checkpoint / "config.json").read_text())
        model_config = ModelConfig()
        model_config.ckpt_path = str(checkpoint)
        configure_kimi_k3_multimodal(model_config, top_config)
        self.assertTrue(model_config.mm_model_config.is_multimodal)

        embedding = object.__new__(KimiK3ImageEmbedding)
        embedding.vision_config = SimpleNamespace(
            text_hidden_size=top_config["vision_config"]["text_hidden_size"]
        )
        embedding.image_processor = KimiK3VisionProcessor(
            model_config.mm_related_params.config["media_proc_cfg"]
        )
        embedding._ckpt_path = str(checkpoint)
        embedding._tokenizer = None
        embedding._word_embedding_weight = None
        embedding._ensure_text_embeddings()

        image = Image.new("RGB", (28, 28))
        expected_visual_tokens = embedding.image_processor.media_tokens_calculator(
            {"image": image}
        )
        self.assertEqual(expected_visual_tokens, 1)
        hidden_size = embedding.vision_config.text_hidden_size
        features = torch.full(
            (expected_visual_tokens, hidden_size),
            0.125,
            dtype=embedding._word_embedding_weight.dtype,
        )
        prompt = embedding.image_processor.make_image_prompt(*image.size)
        prompt_ids = embedding._tokenizer.encode(prompt)
        pad_ids = embedding._tokenizer.encode("<|media_pad|>")
        self.assertEqual(len(pad_ids), 1)
        self.assertEqual(prompt_ids.count(pad_ids[0]), 1)

        assembled = embedding._assemble_image(image, features)
        pad_index = prompt_ids.index(pad_ids[0])
        self.assertEqual(tuple(assembled.shape), (len(prompt_ids), hidden_size))
        torch.testing.assert_close(assembled[pad_index : pad_index + 1], features)
        torch.testing.assert_close(
            assembled[:pad_index], embedding._word_embedding_weight[prompt_ids[:pad_index]]
        )
        torch.testing.assert_close(
            assembled[pad_index + 1 :],
            embedding._word_embedding_weight[prompt_ids[pad_index + 1 :]],
        )


if __name__ == "__main__":
    unittest.main()
