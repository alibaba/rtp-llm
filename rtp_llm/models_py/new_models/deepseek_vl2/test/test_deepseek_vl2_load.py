"""Mixed language/vision checkpoint: both owners must consume the service keys."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.deepseek_vl2.deepseek_vl2 import DeepSeekVLV2
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.deepseek_vl2.language import DeepSeekVLV2ForCausalLM
from rtp_llm.models_py.new_models.glm4_moe.test.checkpoint_fixture import checkpoint
from rtp_llm.models_py.new_models.glm4_moe.test.load_config_fixture import make_load_config


def language_config(path):
    cfg = ModelConfig()
    cfg.model_type = "deepseek_vl_v2"
    cfg.ckpt_path = str(path)
    cfg.data_type = "fp32"
    cfg.enable_fp32_lm_head = False
    cfg.activation_type = "SiGLU"
    raw = {"language_config": dict(
        hidden_size=4, intermediate_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=8,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        moe_intermediate_size=4, first_k_dense_replace=1, use_mla=False,
    )}
    DeepSeekVLV2._from_hf(cfg, raw)
    (path / "config.json").write_text(json.dumps(raw))
    return cfg


def weights():
    result = {
        "language." + key: tensor
        for key, tensor in checkpoint(qk_norm=False, correction=False).items()
        if not key.endswith(".bias")
    }
    result.update({
        "vision.weight": torch.eye(4),
        "projector.layers.weight": torch.eye(4) * 2,
        "projector.layers.bias": torch.zeros(4),
        "image_newline": torch.arange(4, dtype=torch.float32) + 0.1234,
        "view_seperator": torch.arange(4, dtype=torch.float32) + 4.1234,
    })
    return result


class DeepSeekVLV2LoadTest(unittest.TestCase):
    def test_tiny_language_loads_bias_free_qkv_experts_and_injects_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = language_config(path)
            tensors = weights()
            save_file(tensors, str(path / "model.safetensors"))
            owner = NewModelLoader(config, make_load_config(), model_path=directory).load()
            self.assertIsInstance(owner, DeepSeekVLV2ForCausalLM)
            stem = "language.model.layers.1."
            torch.testing.assert_close(
                owner.layers[1].self_attn.qkv_proj.weight,
                torch.cat([tensors[stem + f"self_attn.{p}_proj.weight"] for p in "qkv"]),
            )
            torch.testing.assert_close(
                owner.layers[0].mlp.gate_up_proj.weight,
                torch.cat([
                    tensors[f"language.model.layers.0.mlp.{p}_proj.weight"]
                    for p in ("gate", "up")
                ]),
            )
            torch.testing.assert_close(
                owner.layers[1].mlp.experts.w2[2],
                tensors[stem + "mlp.experts.2.down_proj.weight"],
            )
            image = torch.tensor([[0.0, 8.0, 13.0, 21.0]])
            inputs = SimpleNamespace(
                input_ids=torch.tensor([1, -100, 0]),
                embedding_inputs=SimpleNamespace(text_tokens_mask=torch.tensor([1, 0, 1])),
                multimodal_inputs=SimpleNamespace(
                    multimodal_features=[image], mm_features_locs=torch.tensor([1])
                ),
            )
            word = tensors["language.model.embed_tokens.weight"]
            torch.testing.assert_close(
                owner._input_embeddings(inputs), torch.stack((word[1], image[0], word[0]))
            )

    def test_vision_factory_streams_the_same_file_and_uses_both_separators(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_vl2 import deepseek_vl2_mixin as vit

        class SmallVision(torch.nn.Linear):
            def forward_features(self, images):
                return self(images)

        params = SimpleNamespace(config={
            "vision_config": {}, "tile_tag": "2D",
            "projector_config": {"projector_type": "linear", "input_dim": 4, "n_embed": 4},
        })
        tensors = weights()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            vit.timm, "create_model", side_effect=lambda *a, **kw: SmallVision(4, 4, bias=False)
        ):
            path = Path(directory) / "model.safetensors"
            save_file(tensors, str(path))
            def load():
                return vit.DeepSeekVLV2Mixin.from_new_loader(
                    torch.float32, "cpu", params, "scratch", None, directory
                ).mm_part
            part = load()
            images = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
            features, _ = part.embedding([images, 1, 1], vit.MMUrlType.IMAGE)
            newline = tensors["image_newline"].half().float()
            separator = tensors["view_seperator"].half().float()
            torch.testing.assert_close(features, torch.stack((
                images[0, 0] * 2, newline, separator, images[1, 0] * 2, newline,
            )))
            del tensors["view_seperator"]
            save_file(tensors, str(path))
            with self.assertRaisesRegex((ValueError, RuntimeError), "view_seperator"):
                load()


if __name__ == "__main__":
    unittest.main()
