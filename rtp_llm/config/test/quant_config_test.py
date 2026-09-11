import json
import os
import tempfile
import unittest

from rtp_llm.config.quant_config import ModelOptFp4Config, QuantizationConfig


class QuantConfigTest(unittest.TestCase):
    def _load_config(self, config_json):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            with open(
                os.path.join(checkpoint_path, "config.json"), "w", encoding="utf-8"
            ) as writer:
                json.dump(config_json, writer)
            return QuantizationConfig.load_from_ckpt(checkpoint_path)

    @staticmethod
    def _grouped_modelopt_config():
        return {
            "quant_method": "modelopt",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "input_activations": {"num_bits": 4},
                }
            },
        }

    @staticmethod
    def _flat_modelopt_config():
        return {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": 16,
            "with_input_scale": True,
        }

    def test_grouped_modelopt_config_remains_supported(self):
        quantization_config = self._grouped_modelopt_config()
        quantization_config["ignore"] = ["model.layers.*.self_attn"]
        quantization_config["exclude"] = ["lm_head"]

        config = self._load_config({"quantization_config": quantization_config})

        self.assertIsInstance(config, ModelOptFp4Config)
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.group_size(), 16)
        self.assertTrue(config.is_quanted())
        self.assertFalse(config.mixed_attention)
        self.assertEqual(
            config.exclude_modules, {"model.layers.*.self_attn", "lm_head"}
        )

    def test_flat_modelopt_config_is_normalized(self):
        quantization_config = self._flat_modelopt_config()
        quantization_config["quant_algo"] = "nvFp4"
        quantization_config["ignore"] = ["model.layers.*.mlp"]
        quantization_config["exclude"] = ["lm_head"]

        config = self._load_config({"quantization_config": quantization_config})

        self.assertIsInstance(config, ModelOptFp4Config)
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.group_size(), 16)
        self.assertTrue(config.is_quanted())
        self.assertEqual(config.exclude_modules, {"model.layers.*.mlp", "lm_head"})

    def test_mixed_attention_prefers_text_config_and_supports_top_level(self):
        cases = [
            ({"full_attention_interval": 4}, True),
            (
                {
                    "full_attention_interval": 4,
                    "text_config": {"full_attention_interval": 0},
                },
                False,
            ),
            (
                {
                    "full_attention_interval": 0,
                    "text_config": {"full_attention_interval": 4},
                },
                True,
            ),
        ]
        schemas = [
            self._grouped_modelopt_config(),
            self._flat_modelopt_config(),
        ]
        for quantization_config in schemas:
            for extra_config, expected in cases:
                with self.subTest(
                    quantization_config=quantization_config,
                    extra_config=extra_config,
                ):
                    config_json = {
                        "quantization_config": quantization_config,
                        **extra_config,
                    }
                    config = self._load_config(config_json)
                    self.assertEqual(config.mixed_attention, expected)

    def test_nested_text_config_modelopt_uses_mixed_attention(self):
        config = self._load_config(
            {
                "full_attention_interval": 0,
                "text_config": {
                    "full_attention_interval": 4,
                    "quantization_config": self._flat_modelopt_config(),
                },
            }
        )

        self.assertIsInstance(config, ModelOptFp4Config)
        self.assertTrue(config.mixed_attention)

    def test_invalid_flat_modelopt_config_raises_value_error(self):
        invalid_cases = [
            ({"quant_method": "other"}, "method"),
            ({"quant_method": None}, "quant_method"),
            ({"quant_algo": "FP8"}, "quant_algo"),
            ({"quant_algo": None}, "quant_algo"),
            ({"group_size": 32}, "group_size"),
            ({"group_size": True}, "group_size"),
            ({"with_input_scale": False}, "with_input_scale"),
            ({"with_input_scale": 1}, "with_input_scale"),
        ]
        for overrides, expected_message in invalid_cases:
            with self.subTest(overrides=overrides):
                quantization_config = self._flat_modelopt_config()
                quantization_config.update(overrides)
                with self.assertRaisesRegex(ValueError, expected_message):
                    self._load_config({"quantization_config": quantization_config})

        for missing_key in (
            "quant_method",
            "quant_algo",
            "group_size",
            "with_input_scale",
        ):
            with self.subTest(missing_key=missing_key):
                quantization_config = self._flat_modelopt_config()
                del quantization_config[missing_key]
                with self.assertRaisesRegex(ValueError, missing_key):
                    self._load_config({"quantization_config": quantization_config})


if __name__ == "__main__":
    unittest.main()
