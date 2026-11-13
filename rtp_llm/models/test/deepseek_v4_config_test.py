import copy
import json
import os
import tempfile
import unittest

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, QuantizationConfig
from rtp_llm.models.deepseek_v4 import (
    DeepSeekV4,
    DeepSeekV4DSpark,
    _get_hf_text_config,
)


class DeepSeekV4ConfigTest(unittest.TestCase):
    @staticmethod
    def _text_config():
        return {
            "num_hidden_layers": 40,
            "hidden_size": 5120,
            "vocab_size": 129280,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            "o_groups": 8,
            "o_lora_rank": 1024,
            "sliding_window": 128,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 512,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "num_experts_per_tok": 6,
            "n_routed_experts": 384,
            "moe_intermediate_size": 2304,
            "n_shared_experts": 1,
        }

    @staticmethod
    def _quantization_config():
        return {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }

    def _nested_config(self):
        return {
            "model_type": "deepseek_v41",
            "dtype": "bfloat16",
            "quantization_config": self._quantization_config(),
            "text_config": self._text_config(),
            "vision_config": {"num_hidden_layers": 32, "hidden_size": 1024},
        }

    def _load(self, raw, model_cls=DeepSeekV4):
        with tempfile.TemporaryDirectory() as ckpt_path:
            config_path = os.path.join(ckpt_path, "config.json")
            with open(config_path, "w", encoding="utf-8") as writer:
                json.dump(raw, writer)
            config = model_cls._create_config(ckpt_path)
            with open(config_path, encoding="utf-8") as reader:
                self.assertEqual(json.load(reader), raw)
            return config

    def _assert_text_geometry(self, config):
        self.assertEqual(config.num_layers, 40)
        self.assertEqual(config.hidden_size, 5120)
        self.assertEqual(config.vocab_size, 129280)
        self.assertEqual(config.attn_config.head_num, 64)
        self.assertEqual(config.attn_config.kv_head_num, 1)
        self.assertEqual(config.attn_config.size_per_head, 512)
        self.assertEqual(
            config.attn_config.layer_compress_ratios,
            self._text_config()["compress_ratios"][:40],
        )
        self.assertEqual(config.expert_num, 384)
        self.assertEqual(config.inter_size, 2304)

    def test_legacy_flat_config(self):
        raw = self._text_config()
        raw.update(torch_dtype="bfloat16", expert_dtype="fp4")
        config = self._load(raw)
        self._assert_text_geometry(config)
        self.assertEqual(config.config_dtype, "bfloat16")
        self.assertEqual(config.expert_weight_dtype, "fp4")

    def test_nested_config_uses_text_geometry_and_root_metadata(self):
        config = self._load(self._nested_config())
        self._assert_text_geometry(config)
        self.assertEqual(config.config_dtype, "bfloat16")
        self.assertEqual(config.expert_weight_dtype, "fp4")

    def test_text_geometry_overrides_top_level_geometry(self):
        raw = self._nested_config()
        raw.update(num_hidden_layers=99, hidden_size=128, num_attention_heads=1)
        self._assert_text_geometry(self._load(raw))

    def test_root_metadata_overrides_text_defaults(self):
        raw = self._nested_config()
        raw["text_config"].update(
            dtype="float16",
            torch_dtype="float32",
            quantization_config={"quant_method": "fp8", "expert_dtype": "fp8"},
        )
        config = self._load(raw)
        self.assertEqual(config.config_dtype, "bfloat16")
        self.assertEqual(config.expert_weight_dtype, "fp4")
        normalized = _get_hf_text_config(raw)
        self.assertEqual(normalized["quantization_config"], raw["quantization_config"])
        self.assertEqual(normalized["dtype"], "bfloat16")

    def test_dtype_aliases_and_scope_precedence(self):
        cases = [
            ({"dtype": "bfloat16"}, {}, "bfloat16"),
            ({"torch_dtype": "float16"}, {}, "float16"),
            ({"torch_dtype": "bfloat16"}, {"dtype": "float16"}, "bfloat16"),
            ({"dtype": "bfloat16"}, {"torch_dtype": "float16"}, "bfloat16"),
            ({"dtype": "bfloat16", "torch_dtype": "float16"}, {}, "bfloat16"),
            ({}, {"dtype": "bfloat16"}, "bfloat16"),
            ({}, {"torch_dtype": "float16"}, "float16"),
            ({}, {}, None),
        ]
        for root, text, expected in cases:
            with self.subTest(root=root, text=text):
                raw = {**root, "text_config": {**self._text_config(), **text}}
                self.assertEqual(self._load(raw).config_dtype, expected)

    def test_normalization_does_not_mutate_caller_config(self):
        raw = self._nested_config()
        before = copy.deepcopy(raw)
        normalized = _get_hf_text_config(raw)
        self.assertEqual(raw, before)
        self.assertIsNot(normalized, raw)
        self.assertNotIn("text_config", normalized)
        config = ModelConfig()
        DeepSeekV4._from_hf(config, "unused", config_json=raw)
        self._assert_text_geometry(config)
        self.assertEqual(raw, before)

    def test_null_text_config_keeps_flat_config_supported(self):
        raw = {**self._text_config(), "text_config": None, "torch_dtype": "float16"}
        config = self._load(raw)
        self._assert_text_geometry(config)
        self.assertEqual(config.config_dtype, "float16")

    def test_invalid_text_config_reports_schema_error(self):
        for invalid in ([], "invalid", 1, False):
            with self.subTest(text_config=invalid):
                raw = {**self._text_config(), "text_config": invalid}
                with self.assertRaisesRegex(
                    ValueError, "text_config.*must be an object"
                ):
                    self._load(raw)

    def test_explicit_expert_dtype_remains_supported(self):
        raw = self._nested_config()
        raw["expert_dtype"] = "fp8"
        self.assertEqual(self._load(raw).expert_weight_dtype, "fp8")

    def test_checkpoint_quantization_remains_loadable(self):
        raw = self._nested_config()
        with tempfile.TemporaryDirectory() as ckpt_path:
            config_path = os.path.join(ckpt_path, "config.json")
            with open(config_path, "w", encoding="utf-8") as writer:
                json.dump(raw, writer)
            model = DeepSeekV4._create_config(ckpt_path)
            quantization = QuantizationConfig.load_from_ckpt(ckpt_path)
        self.assertEqual(model.expert_weight_dtype, "fp4")
        self.assertIsInstance(quantization, Fp8BlockWiseQuantConfig)
        self.assertEqual(quantization.group_size(), 32)
        self.assertTrue(quantization.is_quanted())

    def test_dspark_supports_nested_and_flat_configs(self):
        for raw in (
            self._nested_config(),
            {**self._text_config(), "torch_dtype": "bfloat16"},
        ):
            with self.subTest(nested="text_config" in raw):
                config = self._load(raw, DeepSeekV4DSpark)
                self.assertEqual(config.num_layers, 3)
                self.assertEqual(config.attn_config.layer_compress_ratios, [0, 0, 0])
                self.assertEqual(config.moe_layer_index, [0, 1, 2])
                self.assertTrue(config.is_mtp)
                self.assertEqual(config.config_dtype, "bfloat16")

    def test_nested_dspark_keeps_draft_layout_validation(self):
        for ratios, message in (
            ([0] * 40, "trailing draft"),
            ([0] * 40 + [2], "all use SWA"),
        ):
            with self.subTest(ratios=ratios):
                raw = self._nested_config()
                raw["text_config"]["compress_ratios"] = ratios
                with self.assertRaisesRegex(ValueError, message):
                    self._load(raw, DeepSeekV4DSpark)


if __name__ == "__main__":
    unittest.main()
