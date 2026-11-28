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
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.deepseek_vl2.language import DeepSeekVLV2ForCausalLM
from rtp_llm.models_py.new_models.glm4_moe.test.checkpoint_fixture import checkpoint
from rtp_llm.models_py.new_models.glm4_moe.test.load_config_fixture import (
    make_load_config,
)


def language_config(path):
    cfg = ModelConfig()
    cfg.model_type = "deepseek_vl_v2"
    cfg.ckpt_path = str(path)
    cfg.data_type = "fp32"
    cfg.enable_fp32_lm_head = False
    cfg.activation_type = "SiGLU"
    raw = {
        "language_config": dict(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=8,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=4,
            first_k_dense_replace=1,
            use_mla=False,
        )
    }
    DeepSeekVLV2._from_hf(cfg, raw)
    (path / "config.json").write_text(json.dumps(raw))
    return cfg


def weights():
    result = {
        "language." + key: tensor
        for key, tensor in checkpoint(qk_norm=False, correction=False).items()
        if not key.endswith(".bias")
    }
    result.update(
        {
            "vision.weight": torch.eye(4),
            "projector.layers.weight": torch.eye(4) * 2,
            "projector.layers.bias": torch.zeros(4),
            "image_newline": torch.arange(4, dtype=torch.float32) + 0.1234,
            "view_seperator": torch.arange(4, dtype=torch.float32) + 4.1234,
        }
    )
    return result


class DeepSeekVLV2LoadTest(unittest.TestCase):
    def test_declared_top_level_classifier_survives_language_filter(self):
        from dataclasses import replace

        from rtp_llm.models_py.downstream_heads import handler_weight_mappings
        from rtp_llm.ops import TaskType

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = language_config(path)
            config.task_type = TaskType.SEQ_CLASSIFICATION
            tensors = weights()
            heads = {
                "classifier.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "classifier.bias": torch.tensor([1., 5., -1.]),
            }
            tensors.update(heads)
            # This unowned spelling maps to the same logical role, but is not
            # the declared physical source; it must not be read a second time.
            tensors["model.classifier.weight"] = torch.zeros(3, 4)
            save_file(tensors, str(path / "model.safetensors"))
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    load = replace(
                        make_load_config(), load_method=method,
                        custom_weight_mappings=handler_weight_mappings(tuple(heads)),
                    )
                    owner = NewModelLoader(config, load, model_path=directory).load()
                    exports = owner.runtime_weight_view()
                    self.assertIsNone(owner.lm_head)
                    self.assertNotIn("lm_head", exports)
                    for name, expected in heads.items():
                        torch.testing.assert_close(exports["__custom__." + name], expected)
                    owns = owner.checkpoint_weight_name_filter()
                    self.assertFalse(owns("model.classifier.weight"))
                    self.assertFalse(owns("vision.weight"))

    def test_canonical_updates_reach_only_the_owning_pipeline_stage(self):
        from rtp_llm.models_py.new_weight_manager import NewLoaderWeightManager

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = language_config(path)
            save_file(weights(), str(path / "model.safetensors"))
            for stage in (0, 1):
                with self.subTest(stage=stage):
                    load = make_load_config()
                    pc = load.parallelism_config
                    pc.world_size = pc.local_world_size = pc.pp_size = 2
                    pc.world_rank = pc.local_rank = pc.pp_rank = stage
                    pc.pp_stage_layer_counts = [1, 1]
                    loader = NewModelLoader(config, load, model_path=directory)
                    owner = loader.load()
                    manager = NewLoaderWeightManager(owner, loader)
                    norm = owner.layers[str(stage)].input_layernorm.weight
                    before = norm.clone()
                    other = f"model.layers.{1-stage}.input_layernorm.weight"
                    manager.update_tensors({other: torch.full_like(norm, 77)})
                    self.assertEqual(manager.version, 0)
                    torch.testing.assert_close(norm, before)
                    for version, prefix in enumerate(("model.", "language.model."), 1):
                        value = torch.full_like(norm, 100 + version)
                        manager.update_tensors({
                            f"{prefix}layers.{stage}.input_layernorm.weight": value
                        })
                        self.assertEqual(manager.version, version)
                        torch.testing.assert_close(norm, value)

    def test_writer_ft_router_bias_is_not_in_the_hf_language_namespace(self):
        from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index
        from rtp_llm.tools.convert.ft_export_writer import write_ft_shards

        records = [
            (None, "embedding", torch.ones(8, 4)),
            (None, "lm_head", torch.ones(8, 4)),
            (None, "final_layernorm.gamma", torch.ones(4)),
        ]
        for layer, ffn_width in enumerate((8, 4)):
            for name, shape in (
                ("pre_layernorm_weights.gamma", (4,)),
                ("post_layernorm_weights.gamma", (4,)),
                ("self_attention_weights.query_weight.kernel", (4, 8)),
                ("self_attention_weights.attention_output_weight.kernel", (4, 4)),
                ("ffn_weights.intermediate_weight13.kernel", (4, 2 * ffn_width)),
                ("ffn_weights.intermediate_weight2.kernel", (ffn_width, 4)),
            ):
                records.append((layer, name, torch.ones(shape)))
        bias = torch.tensor([-3., 7., 4., -1.])
        for name, tensor in (
            ("gate.kernel", torch.ones(4, 4)),
            ("intermediate_weight.kernel", torch.ones(4, 8, 4)),
            ("intermediate_weight2.kernel", torch.ones(4, 4, 4)),
            ("e_score_correction_bias", bias),
        ):
            records.append((1, "partial_moe_weights." + name, tensor))
        with tempfile.TemporaryDirectory() as directory:
            config = language_config(Path(directory))
            write_ft_shards(records, directory)
            generate_ft_index(directory, {"TP_SIZE": 1, "DP_SIZE": 1, "EP_SIZE": 1})
            owner = NewModelLoader(config, make_load_config(), model_path=directory).load()
            torch.testing.assert_close(owner.layers[1].mlp.gate.e_score_correction_bias, bias)
            self.assertEqual(config.scoring_func, 1)
            self.assertEqual(owner.layers[1].mlp.scoring_func.value, 1)

    def test_tiny_language_loads_bias_free_qkv_experts_and_injects_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = language_config(path)
            tensors = weights()
            save_file(tensors, str(path / "model.safetensors"))
            owner = NewModelLoader(
                config, make_load_config(), model_path=directory
            ).load()
            self.assertIsInstance(owner, DeepSeekVLV2ForCausalLM)
            stem = "language.model.layers.1."
            torch.testing.assert_close(
                owner.layers[1].self_attn.qkv_proj.weight,
                torch.cat(
                    [tensors[stem + f"self_attn.{p}_proj.weight"] for p in "qkv"]
                ),
            )
            torch.testing.assert_close(
                owner.layers[0].mlp.gate_up_proj.weight,
                torch.cat(
                    [
                        tensors[f"language.model.layers.0.mlp.{p}_proj.weight"]
                        for p in ("gate", "up")
                    ]
                ),
            )
            torch.testing.assert_close(
                owner.layers[1].mlp.experts.w2[2],
                tensors[stem + "mlp.experts.2.down_proj.weight"],
            )
            image = torch.tensor([[0.0, 8.0, 13.0, 21.0]])
            inputs = SimpleNamespace(
                input_ids=torch.tensor([1, -100, 0]),
                embedding_inputs=SimpleNamespace(
                    text_tokens_mask=torch.tensor([1, 0, 1])
                ),
                multimodal_inputs=SimpleNamespace(
                    multimodal_features=[image], mm_features_locs=torch.tensor([1])
                ),
            )
            word = tensors["language.model.embed_tokens.weight"]
            torch.testing.assert_close(
                owner._input_embeddings(inputs),
                torch.stack((word[1], image[0], word[0])),
            )

    def test_vision_factory_streams_the_same_file_and_uses_both_separators(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_vl2 import (
            deepseek_vl2_mixin as vit,
        )

        class SmallVision(torch.nn.Linear):
            def forward_features(self, images):
                return self(images)

        params = SimpleNamespace(
            config={
                "vision_config": {},
                "tile_tag": "2D",
                "projector_config": {
                    "projector_type": "linear",
                    "input_dim": 4,
                    "n_embed": 4,
                },
            }
        )
        tensors = weights()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            vit.timm,
            "create_model",
            side_effect=lambda *a, **kw: SmallVision(4, 4, bias=False),
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
            torch.testing.assert_close(
                features,
                torch.stack(
                    (
                        images[0, 0] * 2,
                        newline,
                        separator,
                        images[1, 0] * 2,
                        newline,
                    )
                ),
            )
            del tensors["view_seperator"]
            save_file(tensors, str(path))
            with self.assertRaisesRegex((ValueError, RuntimeError), "view_seperator"):
                load()


if __name__ == "__main__":
    unittest.main()
