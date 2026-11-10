import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_factory_register import get_lazy_model_module_path
from rtp_llm.models.deepseek_v2 import DeepSeekV2, DeepSeekV32Mtp
from rtp_llm.utils.model_weight import W


class DeepSeekV32MtpTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config_json = {
            "architectures": ["DeepseekV32ForCausalLM"],
            "num_hidden_layers": 61,
            "num_nextn_predict_layers": 1,
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "num_attention_heads": 128,
            "vocab_size": 129280,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "rope_theta": 10000,
            "rope_scaling": {
                "factor": 40,
                "original_max_position_embeddings": 4096,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
            },
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 3,
            "scoring_func": "sigmoid",
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": True,
            "n_group": 8,
            "topk_group": 4,
            "index_head_dim": 128,
            "index_n_heads": 64,
            "index_topk": 2048,
        }

    def config(self):
        Path(self.directory.name, "config.json").write_text(
            json.dumps(self.config_json)
        )
        return DeepSeekV32Mtp._create_config(self.directory.name)

    def test_full_checkpoint_preserves_dsa_yarn_and_grouped_routing(self):
        config = self.config()
        self.assertEqual(config.num_layers, 1)
        self.assertEqual(config.mtp_layer_offset, 61)
        self.assertEqual(config.moe_layer_index, [0])
        self.assertTrue(config.is_mtp)
        self.assertTrue(config.attn_config.is_sparse)
        self.assertEqual(config.attn_config.indexer_topk, 2048)
        self.assertEqual(config.attn_config.indexer_head_num, 64)
        self.assertEqual(config.hidden_size, 7168)
        self.assertEqual((config.moe_n_group, config.moe_topk_group), (8, 4))
        self.assertEqual(config.routed_scaling_factor, 2.5)
        self.assertFalse(config.attn_config.rope_config.is_neox_style)
        self.assertTrue(config.attn_config.rope_config.indexer_is_neox_style)
        self.assertEqual(config.attn_config.rope_config.scale, 40)
        self.assertGreater(config.attn_config.softmax_extra_scale, 1.0)
        self.assertFalse(config.index_share_for_mtp_iteration)
        self.assertFalse(config._is_glm52_architecture)
        main_config = DeepSeekV2._create_config(self.directory.name)
        self.assertEqual(main_config.num_layers, 61)
        self.assertEqual(main_config.moe_layer_index, list(range(3, 61)))

    def test_pruned_main_layer_count_controls_mtp_offset(self):
        for layers in (1, 4, 8):
            with self.subTest(layers=layers):
                self.config_json["num_hidden_layers"] = layers
                config = self.config()
                self.assertEqual(config.mtp_layer_offset, layers)
                self.assertEqual(config.num_layers, 1)
                main = DeepSeekV2._create_config(self.directory.name)
                self.assertEqual(main.num_layers, layers)

    def test_extracted_checkpoint_uses_tensor_layout_not_layer_count(self):
        self.config_json["num_hidden_layers"] = 1
        loader = object.__new__(DeepSeekV32Mtp.get_weight_cls())
        loader.model_config = self.config()
        self.assertEqual(loader.model_config.mtp_layer_offset, 1)
        loader._process_meta({}, {"model.layers.0.enorm.weight"})
        self.assertEqual(loader.model_config.mtp_layer_offset, 0)

    def test_rejects_missing_or_multiple_nextn_layers(self):
        for count in (0, 2):
            with self.subTest(count=count):
                self.config_json["num_nextn_predict_layers"] = count
                with self.assertRaisesRegex(ValueError, "exactly one nextn layer"):
                    self.config()

    def test_requires_dsa(self):
        del self.config_json["index_topk"]
        with self.assertRaisesRegex(ValueError, "DSA indexer"):
            self.config()

    def test_lazy_registration(self):
        self.assertEqual(
            get_lazy_model_module_path("deepseek_v32_mtp"),
            "rtp_llm.models.deepseek_v2",
        )

    def test_mega_fp8_native_scales_preserve_weights_exactly(self):
        for kind in ("deepseek_v32", "deepseek_v32_mtp"):
            model = object.__new__(DeepSeekV2)
            model.model_config = SimpleNamespace(model_type=kind)
            weight = torch.tensor([[1.0, -2.0]], dtype=torch.float8_e4m3fn)
            scale = torch.tensor([[0.5, 2.0]])
            layer = {W.moe_w1: weight, W.moe_s1: scale, W.moe_s2: scale.clone()}
            model.weight = SimpleNamespace(weights=[{}, layer])
            with patch.dict(os.environ, {"MOE_STRATEGY": "mega_moe_fp8"}):
                model._prepare_native_moe_scales()
            self.assertIs(layer[W.moe_w1], weight)
            for key in (W.moe_s1, W.moe_s2):
                self.assertEqual(layer[key].dtype, torch.float8_e8m0fnu)
                self.assertTrue(torch.equal(layer[key].float(), scale))

    def test_moe_scale_encoding_keeps_nonrepresentable_pairs_and_other_models(self):
        for kind, strategy, value in (
            ("deepseek_v32", "mega_moe_fp8", 1.5),
            ("deepseek_v32", "mega_moe", 2.0),
            ("glm_5", "mega_moe_fp8", 2.0),
        ):
            model = object.__new__(DeepSeekV2)
            model.model_config = SimpleNamespace(model_type=kind)
            scale1, scale2 = torch.tensor([[value]]), torch.tensor([[2.0]])
            layer = {W.moe_s1: scale1, W.moe_s2: scale2}
            model.weight = SimpleNamespace(weights=[layer])
            with patch.dict(os.environ, {"MOE_STRATEGY": strategy}):
                model._prepare_native_moe_scales()
            self.assertIs(layer[W.moe_s1], scale1)
            self.assertIs(layer[W.moe_s2], scale2)

    def test_native_fp8_scale_metadata_for_main_and_mtp(self):
        for model_cls, model_type in (
            (DeepSeekV2, "deepseek_v32"),
            (DeepSeekV32Mtp, "deepseek_v32_mtp"),
        ):
            with self.subTest(model_type=model_type):
                config = self.config()
                config.model_type = model_type
                config.quant_config = Fp8BlockWiseQuantConfig(
                    bits=8, group_size=128, is_quanted=True, scale_fmt="ue8m0"
                )
                loader = object.__new__(model_cls.get_weight_cls())
                loader.model_config = config
                loader._num_layers = 1
                loader._process_meta(
                    {}, {"model.layers.61.self_attn.q_a_proj.weight_scale_inv"}
                )
                self.assertEqual(config.quant_config.scale_fmt, "float32")

    def test_converted_ue8m0_and_glm_scale_metadata_unchanged(self):
        for model_type, suffix in (
            ("deepseek_v32_mtp", ".scale"),
            ("glm_5", ".weight_scale_inv"),
        ):
            with self.subTest(model_type=model_type, suffix=suffix):
                config = self.config()
                config.model_type = model_type
                config.quant_config = Fp8BlockWiseQuantConfig(
                    bits=8, group_size=128, is_quanted=True, scale_fmt="ue8m0"
                )
                loader = object.__new__(DeepSeekV32Mtp.get_weight_cls())
                loader.model_config = config
                loader._process_meta(
                    {}, {"model.layers.61.self_attn.q_a_proj" + suffix}
                )
                self.assertEqual(config.quant_config.scale_fmt, "ue8m0")

    def test_weight_mapping_uses_draft_layer_and_shared_globals(self):
        weight_cls = DeepSeekV32Mtp.get_weight_cls()
        for main_layers, offset in ((61, 61), (4, 4), (1, 1), (1, 0)):
            with self.subTest(main_layers=main_layers):
                self.config_json["num_hidden_layers"] = main_layers
                loader = object.__new__(weight_cls)
                loader.model_config = self.config()
                loader._num_layers = 1
                loader._hidden_size = 7168
                keys = {
                    f"model.layers.{offset}.enorm.weight",
                    f"model.layers.{offset}.self_attn.q_a_proj.weight",
                    f"model.layers.{offset}.mlp.gate.e_score_correction_bias",
                }
                loader._process_meta({}, keys)
                self.assertTrue(loader.q_use_lora)
                self.assertTrue(loader.has_e_score_correction_bias)
                # Include attention, indexer and expert/scale components so the
                # full-checkpoint offset reaches every nested checkpoint key.
                components = [
                    SimpleNamespace(weights=[SimpleNamespace(name=name)])
                    for name in (
                        "model.layers.{i}.self_attn.q_a_proj.weight",
                        "model.layers.{i}.self_attn.indexer.wq_b.weight",
                        "model.layers.{i}.mlp.experts.0.gate_proj.weight",
                        "model.layers.{i}.mlp.experts.0.gate_proj.weight_scale_inv",
                    )
                ]
                layer = SimpleNamespace(get_components=lambda: components)
                with patch.object(
                    loader, "_get_hf_layer_weight_info", return_value=[layer]
                ):
                    info = loader._get_weight_info()
                names = {
                    weight.name: weight.weights[0].name
                    for weight in info.weights
                    if weight.weights
                }
                self.assertEqual(
                    names[W.embedding],
                    (
                        "model.embed_tokens.weight"
                        if offset
                        else "model.layers.0.embed_tokens.weight"
                    ),
                )
                self.assertEqual(
                    names[W.lm_head],
                    (
                        "lm_head.weight"
                        if offset
                        else "model.layers.0.shared_head.head.weight"
                    ),
                )
                self.assertEqual(
                    names[W.multi_tokens_predict_final_ln_gamma],
                    f"model.layers.{offset}.shared_head.norm.weight",
                )
                for component in components:
                    name = component.weights[0].name.format(i=0)
                    self.assertTrue(name.startswith(f"model.layers.{offset}."))


if __name__ == "__main__":
    unittest.main()
