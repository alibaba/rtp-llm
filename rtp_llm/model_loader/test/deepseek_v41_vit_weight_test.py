"""V41 vision descriptors must follow the execution role before checkpoint I/O."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.models.deepseek_v4 import DeepSeekV4, DeepSeekV4Weight
from rtp_llm.models.deepseek_v41 import (
    DeepSeekV41,
    DeepSeekV41DSpark,
    DeepSeekV41Weight,
)
from rtp_llm.ops import VitSeparation
from rtp_llm.utils.model_weight import W


class V41VitWeightTest(unittest.TestCase):
    def descriptor(self, separation):
        descriptor = object.__new__(DeepSeekV41Weight)
        descriptor.vit_separation = separation
        descriptor.vit_weights = None
        descriptor.model_config = SimpleNamespace(
            deepseek_v41_config={"vision_config": {"num_hidden_layers": 2}}
        )
        return descriptor

    def test_role_never_constructs_language_weights(self):
        descriptor = self.descriptor(VitSeparation.VIT_SEPARATION_ROLE)
        with patch.object(
            DeepSeekV4Weight,
            "_get_weight_info",
            side_effect=AssertionError("language descriptors requested by ViT ROLE"),
        ):
            info = descriptor.get_weight_info()
        self.assertEqual(info.layer_weights, [])
        names = {weight.name for weight in info.weights}
        self.assertIn("v41.image_start", names)
        self.assertIn("v41.aligner.w2.bias", names)
        self.assertIn("v41.vision.blocks.1.attn.wqkv.weight", names)
        self.assertTrue(all(name.startswith("v41.") for name in names))
        for weight in info.weights:
            norm = ".norm" in weight.name
            self.assertEqual(
                weight.data_type, torch.float32 if norm else torch.bfloat16
            )

    def test_remote_omits_vision_and_local_preserves_it(self):
        for separation in (
            VitSeparation.VIT_SEPARATION_LOCAL,
            VitSeparation.VIT_SEPARATION_REMOTE,
        ):
            with self.subTest(separation=separation):
                descriptor = self.descriptor(separation)
                language = SimpleNamespace(name="language")
                obsolete = SimpleNamespace(name=W.v4_hc_head_base)
                layers = [[SimpleNamespace(name="layer")]]
                with patch.object(
                    DeepSeekV4Weight,
                    "_get_weight_info",
                    return_value=ModelWeightInfo([language, obsolete], layers),
                ):
                    info = descriptor._get_weight_info()
                self.assertIs(info.layer_weights, layers)
                self.assertIs(info.weights[0], language)
                self.assertNotIn(obsolete, info.weights)
                self.assertEqual(
                    any(weight.name.startswith("v41.") for weight in info.weights),
                    separation == VitSeparation.VIT_SEPARATION_LOCAL,
                )

    def test_frontend_config_is_multimodal_before_model_construction(self):
        config = {
            "max_position_embeddings": 1024,
            "kv_source_layer_ids": [0],
            "vision_config": {"num_hidden_layers": 2},
        }
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(config))
            with patch.object(DeepSeekV4, "_from_hf"):
                actual = DeepSeekV41._create_config(directory)
        self.assertTrue(actual.mm_model_config.is_multimodal)
        self.assertTrue(actual.is_deepseek_v41)

    def test_draft_config_and_model_never_advertise_vision(self):
        config = {
            "max_position_embeddings": 1024,
            "kv_source_layer_ids": [0],
            "vision_config": {"num_hidden_layers": 2},
            "num_nextn_predict_layers": 1,
            "compress_ratios": [1, 0],
            "dspark_n_routed_experts": 4,
            "dspark_num_experts_per_tok": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(config))
            with patch.object(
                DeepSeekV4,
                "_from_hf",
                side_effect=lambda config, *args, **kwargs: setattr(
                    config, "num_layers", 1
                ),
            ):
                target = DeepSeekV41._create_config(directory)
                draft = DeepSeekV41DSpark._create_config(directory)
        self.assertTrue(target.mm_model_config.is_multimodal)
        self.assertFalse(draft.mm_model_config.is_multimodal)
        self.assertTrue(draft.is_mtp)
        target_model = object.__new__(DeepSeekV41)
        draft_model = object.__new__(DeepSeekV41DSpark)
        self.assertIs(target_model._as_multimodal_model(), target_model)
        self.assertIsNone(draft_model._as_multimodal_model())
        with patch.object(DeepSeekV4, "_may_init_multimodal") as initialize:
            draft_model._may_init_multimodal()
        initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
