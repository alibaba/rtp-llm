"""Checkpoint-backed K3 image prompt assembly without text-model execution."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from safetensors import safe_open

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
    def test_real_moonvit_and_projector_batch_match_serial(self) -> None:
        checkpoint = Path(os.environ["K3_CKPT_PATH"])
        top_config = json.loads((checkpoint / "config.json").read_text())
        model_config = ModelConfig()
        model_config.ckpt_path = str(checkpoint)
        configure_kimi_k3_multimodal(model_config, top_config)
        embedding = KimiK3ImageEmbedding(model_config.mm_related_params)

        weight_map = json.loads(
            (checkpoint / "model.safetensors.index.json").read_text()
        )["weight_map"]
        vision_state = {}
        projector_state = {}
        vision_shards = {
            name
            for key, name in weight_map.items()
            if key.startswith(("vision_tower.", "mm_projector."))
        }
        for shard_name in sorted(vision_shards):
            with safe_open(
                checkpoint / shard_name, framework="pt", device="cpu"
            ) as shard:
                for key in shard.keys():
                    if key.startswith("vision_tower."):
                        vision_state[key.removeprefix("vision_tower.")] = (
                            shard.get_tensor(key)
                        )
                    elif key.startswith("mm_projector."):
                        projector_state[key.removeprefix("mm_projector.")] = (
                            shard.get_tensor(key)
                        )
        embedding.vision_tower.load_state_dict(vision_state, strict=True)
        embedding.mm_projector.load_state_dict(projector_state, strict=True)
        self.assertEqual(len(vision_state), 165)
        self.assertEqual(len(projector_state), 3)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding.vision_tower.to(device=device, dtype=torch.bfloat16).eval()
        embedding.mm_projector.to(device=device, dtype=torch.bfloat16).eval()
        image_a = Image.new("RGB", (28, 28), (128, 64, 32))
        image_b = Image.new("RGB", (56, 28), (32, 64, 128))
        batched = embedding.image_embedding([image_a, image_b])
        separate = [
            embedding.image_embedding([image])[0] for image in (image_a, image_b)
        ]
        self.assertEqual(
            [tuple(value.shape) for value in batched], [(1, 7168), (2, 7168)]
        )
        for actual, expected in zip(batched, separate):
            self.assertTrue(torch.isfinite(actual).all().item())
            # BF16 matmul accumulates in a different order for batched and
            # single-image shapes; constrain both typical and worst-case error.
            error = (actual.float() - expected.float()).abs()
            self.assertLess(error.mean().item(), 0.005)
            self.assertLess(error.max().item(), 0.06)

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
