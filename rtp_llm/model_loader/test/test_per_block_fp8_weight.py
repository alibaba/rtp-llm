import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, QuantizationConfig
from rtp_llm.model_loader.attn_weight import (
    AttnAtomicWeight,
    AttnConfig,
    MlaAttnAtomicWeight,
    MlaConfig,
)
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    PerBlockFp8Weight,
    per_block_cast_to_fp8,
)
from rtp_llm.utils.model_weight import FP8_E4M3_MAX, CkptWeightInfo, W


def _compressed_fp8_config():
    return {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "block",
                    "block_structure": [128, 128],
                    "dynamic": False,
                    "symmetric": True,
                    "weight_scale_suffix": ".scale",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 128,
                    "dynamic": True,
                    "symmetric": True,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": ["lm_head"],
    }


class CompressedFp8BlockConfigTest(unittest.TestCase):
    def _load(self, quantization_config):
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({"quantization_config": quantization_config})
            )
            return QuantizationConfig.load_from_ckpt(model_dir)

    def test_accepts_supported_schema(self):
        config = self._load(_compressed_fp8_config())
        self.assertIsInstance(config, Fp8BlockWiseQuantConfig)
        self.assertEqual(config.group_size(), 128)
        self.assertEqual(config.weight_scale_suffix, ".scale")
        self.assertEqual(config.exclude_modules, {"lm_head"})

    def test_rejects_unsupported_weight_schema(self):
        cases = {
            "small block": ("block_structure", [64, 64]),
            "zero block": ("block_structure", [0, 0]),
            "asymmetric block": ("block_structure", [128, 64]),
            "dynamic weight": ("dynamic", True),
            "asymmetric weight": ("symmetric", False),
        }
        for name, (field, value) in cases.items():
            with self.subTest(name=name):
                config = copy.deepcopy(_compressed_fp8_config())
                config["config_groups"]["group_0"]["weights"][field] = value
                with self.assertRaises(ValueError):
                    self._load(config)

    def test_rejects_unsupported_activation_schema(self):
        cases = {
            "missing activation": None,
            "static activation": {"dynamic": False},
            "wrong group size": {"group_size": 64},
            "asymmetric activation": {"symmetric": False},
        }
        for name, override in cases.items():
            with self.subTest(name=name):
                config = copy.deepcopy(_compressed_fp8_config())
                group = config["config_groups"]["group_0"]
                if override is None:
                    group["input_activations"] = None
                else:
                    group["input_activations"].update(override)
                with self.assertRaises(ValueError):
                    self._load(config)


class PerBlockCastToFp8Test(unittest.TestCase):
    def test_non_aligned_dimensions_preserve_layout(self):
        block_size = 4
        for shape in ((8, 7), (7, 8), (7, 6)):
            with self.subTest(shape=shape):
                weight = torch.arange(
                    1, shape[0] * shape[1] + 1, dtype=torch.float32
                ).reshape(shape)
                actual, actual_scales = per_block_cast_to_fp8(weight, block_size)
                for block_row, row in enumerate(range(0, shape[0], block_size)):
                    for block_col, col in enumerate(range(0, shape[1], block_size)):
                        tile = weight[row : row + block_size, col : col + block_size]
                        amax = tile.abs().amax().clamp(1e-4)
                        expected = (tile * (FP8_E4M3_MAX / amax)).to(
                            torch.float8_e4m3fn
                        )
                        actual_tile = actual[
                            row : row + block_size, col : col + block_size
                        ]
                        self.assertTrue(torch.equal(actual_tile, expected))
                        self.assertEqual(
                            actual_scales[block_row, block_col].item(),
                            (amax / FP8_E4M3_MAX).item(),
                        )


class PerBlockFp8WeightDescriptorTest(unittest.TestCase):
    @staticmethod
    def _quant_config(**kwargs):
        return Fp8BlockWiseQuantConfig(is_quanted=True, group_size=128, **kwargs)

    @staticmethod
    def _qkv_weight():
        return AttnAtomicWeight(
            name=W.attn_qkv_w,
            weights=[
                CkptWeightInfo("model.layers.{i}.self_attn.q_proj.weight"),
                CkptWeightInfo("model.layers.{i}.self_attn.k_proj.weight"),
                CkptWeightInfo("model.layers.{i}.self_attn.v_proj.weight"),
            ],
            config=AttnConfig(),
        )

    def test_rejects_concrete_layer_exclusion(self):
        config = self._quant_config(ignore_patterns=["model.layers.7.self_attn.q_proj"])
        with self.assertRaisesRegex(ValueError, "partial per-layer"):
            PerBlockFp8Weight.support(config, self._qkv_weight())

    def test_rejects_partial_fused_weight_exclusion(self):
        config = self._quant_config(
            ignore_patterns=["model.layers.{i}.self_attn.q_proj"]
        )
        with self.assertRaisesRegex(ValueError, "fused weight"):
            PerBlockFp8Weight.support(config, self._qkv_weight())

    def test_accepts_whole_fused_weight_exclusion(self):
        config = self._quant_config(
            ignore_patterns=[
                "model.layers.{i}.self_attn.q_proj",
                "model.layers.{i}.self_attn.k_proj",
                "model.layers.{i}.self_attn.v_proj",
            ]
        )
        self.assertFalse(PerBlockFp8Weight.support(config, self._qkv_weight()))

    def test_rewrites_mla_embedded_scale_suffix(self):
        source = MlaAttnAtomicWeight(
            name=W.mla_kc,
            weights=[CkptWeightInfo("model.layers.{i}.self_attn.kv_b_proj.weight")],
            config=MlaConfig(
                head_num=2,
                nope_head_dim=64,
                rope_head_dim=64,
                kv_lora_rank=128,
                v_head_dim=64,
                use_mla=True,
            ),
        )
        quantized = PerBlockFp8Weight(
            source,
            self._quant_config(weight_scale_suffix=".scale"),
            name=source.name,
        )
        self.assertEqual(
            [weight.name for weight in quantized.kernel.weights],
            [
                "model.layers.{i}.self_attn.kv_b_proj.weight",
                "model.layers.{i}.self_attn.kv_b_proj.scale",
            ],
        )


if __name__ == "__main__":
    unittest.main()
