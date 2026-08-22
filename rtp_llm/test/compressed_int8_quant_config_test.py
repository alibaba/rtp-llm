import copy
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional

from rtp_llm.config.quant_config import (
    CompressedTensorsQuantConfig,
    CompressedW4A8Int4PerChannelQuantConfig,
    Fp8PerChannelCompressedQuantConfig,
    QuantizationConfig,
    W8A8Int8PerChannelCompressedQuantConfig,
    init_quant_config,
    preset_quant_config,
)

# The authoritative GLM-4.7 INT8 W8A8 shape: per-channel symmetric int8 weights
# with dynamic per-token int8 activations.
INT8_W8A8_QUANT_CONFIG: Dict[str, Any] = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "ignore": ["lm_head", "model.layers.0.mlp.gate"],
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "int",
                "strategy": "channel",
                "symmetric": True,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "int",
                "strategy": "token",
                "dynamic": True,
                "symmetric": True,
            },
        }
    },
}

FP8_PER_CHANNEL_QUANT_CONFIG: Dict[str, Any] = {
    "quant_method": "compressed-tensors",
    "config_groups": {
        "group_0": {
            "weights": {"num_bits": 8, "type": "float", "strategy": "channel"},
            "input_activations": {"num_bits": 8, "type": "float", "dynamic": True},
        }
    },
}

INT4_GROUP_QUANT_CONFIG: Dict[str, Any] = {
    "quant_method": "compressed-tensors",
    "ignore": ["lm_head"],
    "config_groups": {
        "group_0": {
            "weights": {
                "num_bits": 4,
                "type": "int",
                "strategy": "group",
                "group_size": 32,
                "symmetric": True,
            },
            "input_activations": {"num_bits": 8, "type": "float", "dynamic": True},
        }
    },
}


def _quant_config_with(
    weights: Optional[Dict[str, Any]] = None,
    input_activations: Optional[Dict[str, Any]] = None,
    drop: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Clone the reference config, overriding/removing group_0 fields."""
    config = copy.deepcopy(INT8_W8A8_QUANT_CONFIG)
    group = config["config_groups"]["group_0"]
    group["weights"].update(weights or {})
    group["input_activations"].update(input_activations or {})
    for key in drop or []:
        group.pop(key, None)
    return config


class CompressedInt8QuantConfigTest(unittest.TestCase):
    def _load_from_ckpt(self, config_json: Dict[str, Any]):
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as f:
                json.dump(config_json, f)
            return QuantizationConfig.load_from_ckpt(ckpt_path)

    def test_parse_reference_config(self):
        config = W8A8Int8PerChannelCompressedQuantConfig.from_checkpoint_quant_config(
            INT8_W8A8_QUANT_CONFIG
        )
        self.assertIsInstance(config, W8A8Int8PerChannelCompressedQuantConfig)
        self.assertIsInstance(config, CompressedTensorsQuantConfig)
        self.assertEqual(config.bits, 8)
        self.assertEqual(config.group_size(), 0)
        self.assertTrue(config.is_quanted())
        self.assertTrue(config.is_dynamic())
        self.assertEqual(config.weight_scale_suffix, ".weight_scale")

    def test_ignore_list_becomes_ignore_patterns_and_exclude_modules(self):
        config = W8A8Int8PerChannelCompressedQuantConfig.from_checkpoint_quant_config(
            INT8_W8A8_QUANT_CONFIG
        )
        self.assertEqual(
            config.ignore_patterns, ["lm_head", "model.layers.0.mlp.gate"]
        )
        self.assertEqual(
            config.exclude_modules, {"lm_head", "model.layers.0.mlp.gate"}
        )

    def test_missing_ignore_list_is_allowed(self):
        quant_config = copy.deepcopy(INT8_W8A8_QUANT_CONFIG)
        quant_config.pop("ignore")
        config = W8A8Int8PerChannelCompressedQuantConfig.from_checkpoint_quant_config(
            quant_config
        )
        self.assertEqual(config.ignore_patterns, [])
        self.assertEqual(config.exclude_modules, set())

    def test_load_from_ckpt_detects_int8_w8a8(self):
        config = self._load_from_ckpt(
            {"quantization_config": INT8_W8A8_QUANT_CONFIG}
        )
        self.assertIsInstance(config, W8A8Int8PerChannelCompressedQuantConfig)
        self.assertEqual(config.ignore_patterns, ["lm_head", "model.layers.0.mlp.gate"])

    def test_load_from_ckpt_detects_nested_text_config(self):
        config = self._load_from_ckpt(
            {"text_config": {"quantization_config": INT8_W8A8_QUANT_CONFIG}}
        )
        self.assertIsInstance(config, W8A8Int8PerChannelCompressedQuantConfig)

    def test_registry_selects_config_by_method(self):
        config = QuantizationConfig.from_config(
            {
                "bits": 8,
                "method": "W8A8_INT8_PER_CHANNEL_COMPRESSED",
                "group_size": 0,
                "is_quanted": True,
            }
        )
        self.assertIsInstance(config, W8A8Int8PerChannelCompressedQuantConfig)

    def test_preset_quant_config_entry(self):
        preset = preset_quant_config["W8A8_INT8_PER_CHANNEL_COMPRESSED"]
        self.assertIsInstance(preset, W8A8Int8PerChannelCompressedQuantConfig)
        self.assertIs(
            init_quant_config("w8a8_int8_per_channel_compressed"), preset
        )
        self.assertEqual(
            W8A8Int8PerChannelCompressedQuantConfig.get_method(),
            "W8A8_INT8_PER_CHANNEL_COMPRESSED",
        )

    def test_reject_non_quantized_checkpoint(self):
        with self.assertRaises(ValueError):
            W8A8Int8PerChannelCompressedQuantConfig(bits=8, is_quanted=False)

    def test_reject_non_8bit_weights(self):
        with self.assertRaises(ValueError):
            W8A8Int8PerChannelCompressedQuantConfig(bits=4)

    def test_reject_asymmetric_weights(self):
        quant_config = _quant_config_with(weights={"symmetric": False})
        with self.assertRaises(ValueError) as ctx:
            self._load_from_ckpt({"quantization_config": quant_config})
        self.assertIn("symmetric", str(ctx.exception))

    def test_reject_static_activations(self):
        quant_config = _quant_config_with(input_activations={"dynamic": False})
        with self.assertRaises(ValueError) as ctx:
            self._load_from_ckpt({"quantization_config": quant_config})
        self.assertIn("dynamic", str(ctx.exception))
        self.assertIn("input_activations", str(ctx.exception))

    def test_reject_per_tensor_activations(self):
        quant_config = _quant_config_with(input_activations={"strategy": "tensor"})
        with self.assertRaises(ValueError) as ctx:
            self._load_from_ckpt({"quantization_config": quant_config})
        self.assertIn("strategy", str(ctx.exception))

    def test_reject_non_8bit_activations(self):
        quant_config = _quant_config_with(input_activations={"num_bits": 4})
        with self.assertRaises(ValueError) as ctx:
            self._load_from_ckpt({"quantization_config": quant_config})
        self.assertIn("num_bits", str(ctx.exception))

    def test_reject_float_activations(self):
        quant_config = _quant_config_with(input_activations={"type": "float"})
        with self.assertRaises(ValueError) as ctx:
            self._load_from_ckpt({"quantization_config": quant_config})
        self.assertIn("type", str(ctx.exception))

    def test_reject_incomplete_quant_config(self):
        # load_from_ckpt already requires group_0.input_activations before any
        # scheme is picked, so exercise the parser directly here.
        for quant_config in (
            _quant_config_with(drop=["input_activations"]),
            _quant_config_with(drop=["weights"]),
            {"quant_method": "compressed-tensors"},
            {"quant_method": "compressed-tensors", "config_groups": {}},
            {"quant_method": "compressed-tensors", "config_groups": None},
        ):
            with self.subTest(quant_config=quant_config):
                with self.assertRaises(ValueError) as ctx:
                    W8A8Int8PerChannelCompressedQuantConfig.from_checkpoint_quant_config(
                        quant_config
                    )
                self.assertIn(
                    "is not a W8A8_INT8_PER_CHANNEL_COMPRESSED", str(ctx.exception)
                )

    def test_fp8_per_channel_still_wins(self):
        config = self._load_from_ckpt(
            {"quantization_config": FP8_PER_CHANNEL_QUANT_CONFIG}
        )
        self.assertIsInstance(config, Fp8PerChannelCompressedQuantConfig)

    def test_int4_group_still_wins(self):
        config = self._load_from_ckpt({"quantization_config": INT4_GROUP_QUANT_CONFIG})
        self.assertIsInstance(config, CompressedW4A8Int4PerChannelQuantConfig)

    def test_matches_weights_rejects_other_schemes(self):
        # Weight dtype/width/granularity is the branch discriminator. A scheme
        # this config does not own must not be claimed here -- including
        # per-tensor int8, for which main has no branch at all.
        for weights in (
            {"num_bits": 8, "type": "float", "strategy": "channel"},
            {"num_bits": 8, "type": "float", "strategy": "tensor"},
            {"num_bits": 8, "type": "int", "strategy": "tensor"},
            {"num_bits": 4, "type": "int", "strategy": "channel"},
            {"num_bits": 8, "type": "int", "strategy": "group"},
            {},
        ):
            with self.subTest(weights=weights):
                self.assertFalse(
                    W8A8Int8PerChannelCompressedQuantConfig.matches_weights(weights)
                )


if __name__ == "__main__":
    unittest.main()
