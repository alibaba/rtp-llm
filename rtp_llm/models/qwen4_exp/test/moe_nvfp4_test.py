import copy
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models.qwen4_exp.moe_nvfp4 import (
    is_qwen4_nvfp4_checkpoint,
    validate_moe_nvfp4_checkpoint,
    validate_moe_nvfp4_manifest,
)


class Qwen4MoeNvfp4Test(unittest.TestCase):
    def setUp(self):
        self.targets = [
            f"model.language_model.layers.{layer}.mlp.experts" for layer in range(2)
        ]
        self.draft = "mtp.layers.0.mlp.experts"
        self.ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
        layers = {
            name: {"quant_algo": "NVFP4", "group_size": 16} for name in self.targets
        }
        layers[self.draft] = {"quant_algo": "FP8_PB_WO", "group_size": 128}
        layers[self.ple] = {"quant_algo": "FP8"}
        self.config = {
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 2,
                "num_experts": 2,
                "hidden_size": 32,
                "moe_intermediate_size": 16,
            },
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "ignore": ["lm_head"],
                "quantized_layers": layers,
                "config_groups": {
                    "group_0": {
                        "targets": self.targets,
                        "weights": {
                            "dynamic": False,
                            "num_bits": 4,
                            "type": "float",
                            "group_size": 16,
                        },
                        "input_activations": {
                            "dynamic": False,
                            "num_bits": 4,
                            "type": "float",
                            "group_size": 16,
                        },
                    },
                    "group_1": {
                        "targets": [self.draft],
                        "weights": {
                            "dynamic": False,
                            "num_bits": 8,
                            "type": "float",
                            "group_size": 128,
                        },
                        "input_activations": {
                            "dynamic": True,
                            "num_bits": 8,
                            "type": "float",
                            "group_size": 128,
                        },
                    },
                    "group_2": {
                        "targets": [self.ple],
                        "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                    },
                },
            },
        }
        self.keys = {}
        for module in self.targets + [self.draft]:
            suffixes = (
                ("weight", "weight_scale_inv")
                if module == self.draft
                else ("weight", "weight_scale", "weight_scale_2", "input_scale")
            )
            for expert in range(2):
                for projection in ("gate", "up", "down"):
                    for suffix in suffixes:
                        name = f"{module}.{expert}.{projection}_proj.{suffix}"
                        self.keys[name] = "model.safetensors"
        for shard in range(2):
            self.keys[f"{self.ple}.shard_{shard}.weight"] = "model.safetensors"
        self.keys[self.ple + ".weight_scale"] = "model.safetensors"
        self.keys["lm_head.weight"] = "model.safetensors"

    def test_recognizes_family_before_strict_validation(self):
        self.assertTrue(is_qwen4_nvfp4_checkpoint(self.config))
        config = copy.deepcopy(self.config)
        config["quantization_config"]["quantized_layers"][self.targets[0]][
            "quant_algo"
        ] = "UNKNOWN"
        self.assertTrue(is_qwen4_nvfp4_checkpoint(config))
        with self.assertRaisesRegex(ValueError, "unsupported quantization"):
            validate_moe_nvfp4_manifest(config, self.keys)

    def test_does_not_recognize_other_quantization_families(self):
        for config in ({}, None, {"quantization_config": None}):
            with self.subTest(config=config):
                self.assertFalse(is_qwen4_nvfp4_checkpoint(config))
        for field, value in (("quant_method", "fp8"), ("quant_algo", "NVFP4")):
            config = copy.deepcopy(self.config)
            config["quantization_config"][field] = value
            self.assertFalse(is_qwen4_nvfp4_checkpoint(config))
            with self.assertRaisesRegex(ValueError, "ModelOpt MIXED_PRECISION"):
                validate_moe_nvfp4_manifest(config, self.keys)

    def test_accepts_complete_three_format_manifest(self):
        validate_moe_nvfp4_manifest(self.config, self.keys)

    def test_real_offline_checkpoint_contract(self):
        directory = os.getenv("QWEN4_NVFP4_CKPT_DIR")
        if not directory:
            self.skipTest("set QWEN4_NVFP4_CKPT_DIR to validate an offline checkpoint")
        root = Path(directory)
        config = json.loads((root / "config.json").read_text())
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.assertTrue(is_qwen4_nvfp4_checkpoint(config))
        validate_moe_nvfp4_checkpoint(config, index["weight_map"], root)

    def test_requires_both_metadata_representations(self):
        for key in ("quantized_layers", "config_groups"):
            for value in (None, {}, []):
                with self.subTest(key=key, value=value):
                    config = copy.deepcopy(self.config)
                    config["quantization_config"][key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        validate_moe_nvfp4_manifest(config, self.keys)

    def test_requires_valid_text_geometry(self):
        config = copy.deepcopy(self.config)
        del config["text_config"]
        with self.assertRaisesRegex(ValueError, "text_config geometry"):
            validate_moe_nvfp4_manifest(config, self.keys)
        for key in ("num_hidden_layers", "num_experts"):
            for value in (0, -1, True, "2"):
                config = copy.deepcopy(self.config)
                config["text_config"][key] = value
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    validate_moe_nvfp4_manifest(config, self.keys)

    def test_rejects_missing_target_projection_or_each_companion(self):
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            with self.subTest(suffix=suffix):
                keys = dict(self.keys)
                del keys[f"{self.targets[0]}.0.gate_proj.{suffix}"]
                with self.assertRaisesRegex(ValueError, "missing target"):
                    validate_moe_nvfp4_manifest(self.config, keys)

    def test_rejects_partial_or_nvfp4_draft(self):
        for suffix in ("weight", "weight_scale_inv"):
            keys = dict(self.keys)
            del keys[f"{self.draft}.0.up_proj.{suffix}"]
            with self.assertRaisesRegex(ValueError, "missing draft FP8"):
                validate_moe_nvfp4_manifest(self.config, keys)
        config = copy.deepcopy(self.config)
        config["quantization_config"]["quantized_layers"][self.draft] = {
            "quant_algo": "NVFP4",
            "group_size": 16,
        }
        with self.assertRaisesRegex(ValueError, "unsupported quantization"):
            validate_moe_nvfp4_manifest(config, self.keys)

    def test_rejects_unknown_modules_and_target_holes(self):
        for replacement in (
            "model.language_model.layers.9.mlp.experts",
            "mtp.layers.1.mlp.experts",
        ):
            config = copy.deepcopy(self.config)
            layers = config["quantization_config"]["quantized_layers"]
            layers[replacement] = layers.pop(self.targets[0])
            with self.assertRaisesRegex(ValueError, "unsupported quantized module"):
                validate_moe_nvfp4_manifest(config, self.keys)
        config = copy.deepcopy(self.config)
        del config["quantization_config"]["quantized_layers"][self.targets[0]]
        with self.assertRaisesRegex(ValueError, "every target layer"):
            validate_moe_nvfp4_manifest(config, self.keys)

    def test_rejects_wrong_group_size_and_unknown_recipe_fields(self):
        for change in ({"group_size": 32}, {"quant_algo": "FP8"}, {"unknown": True}):
            config = copy.deepcopy(self.config)
            config["quantization_config"]["quantized_layers"][self.targets[0]].update(
                change
            )
            with self.assertRaisesRegex(ValueError, "unsupported quantization"):
                validate_moe_nvfp4_manifest(config, self.keys)

    def test_rejects_group_disagreement_duplicate_and_missing_target(self):
        for mutation in ("wrong_spec", "duplicate", "missing", "unknown"):
            with self.subTest(mutation=mutation):
                config = copy.deepcopy(self.config)
                groups = config["quantization_config"]["config_groups"]
                if mutation == "wrong_spec":
                    groups["group_0"]["input_activations"]["dynamic"] = True
                elif mutation == "duplicate":
                    groups["group_0"]["targets"].append(self.targets[0])
                elif mutation == "missing":
                    groups["group_0"]["targets"].pop()
                else:
                    groups["group_0"]["targets"].append("unknown")
                with self.assertRaisesRegex(ValueError, "config_groups"):
                    validate_moe_nvfp4_manifest(config, self.keys)

    def test_rejects_unknown_scale_or_expert_namespace(self):
        for name in (
            f"{self.targets[0]}.0.gate_proj.weight_scale_inv",
            f"{self.draft}.0.gate_proj.input_scale",
            "model.language_model.layers.0.self_attn.q_proj.weight_scale",
            "mtp.layers.1.mlp.experts.0.up_proj.weight",
            f"{self.targets[0]}.0.gate_up_proj.weight",
        ):
            with self.subTest(name=name):
                keys = {**self.keys, name: "model.safetensors"}
                with self.assertRaisesRegex(
                    ValueError, "unexpected expert or quantization"
                ):
                    validate_moe_nvfp4_manifest(self.config, keys)

    def test_rejects_missing_ple_scale_or_noncontiguous_shards(self):
        keys = dict(self.keys)
        del keys[self.ple + ".weight_scale"]
        with self.assertRaisesRegex(ValueError, "missing its global scale"):
            validate_moe_nvfp4_manifest(self.config, keys)
        keys = dict(self.keys)
        del keys[self.ple + ".shard_0.weight"]
        with self.assertRaisesRegex(ValueError, "contiguous from zero"):
            validate_moe_nvfp4_manifest(self.config, keys)

    def test_allows_omitted_optional_namespaces_only_when_not_declared(self):
        config = copy.deepcopy(self.config)
        quant = config["quantization_config"]
        for name in (self.draft, self.ple):
            del quant["quantized_layers"][name]
        del quant["config_groups"]["group_1"]
        del quant["config_groups"]["group_2"]
        keys = {
            name: value
            for name, value in self.keys.items()
            if not name.startswith((self.draft, self.ple))
        }
        validate_moe_nvfp4_manifest(config, keys)
        with self.assertRaises(ValueError):
            validate_moe_nvfp4_manifest(config, self.keys)

    def _checkpoint_tensors(self, fp8_scale_dtype=torch.bfloat16):
        tensors = {}
        for name in self.keys:
            suffix = name.rsplit(".", 1)[-1]
            n, k = (32, 16) if ".down_proj." in name else (16, 32)
            if name == "lm_head.weight":
                value = torch.ones(4, 32, dtype=torch.bfloat16)
            elif name.startswith(self.ple):
                value = (
                    torch.tensor([0.25], dtype=fp8_scale_dtype)
                    if suffix == "weight_scale"
                    else torch.ones(2, 32).to(torch.float8_e4m3fn)
                )
            elif name.startswith(self.draft):
                value = (
                    torch.ones(1, 1, dtype=fp8_scale_dtype)
                    if suffix == "weight_scale_inv"
                    else torch.ones(n, k).to(torch.float8_e4m3fn)
                )
            elif suffix == "weight":
                value = torch.zeros(n, k // 2, dtype=torch.uint8)
            elif suffix == "weight_scale":
                value = torch.ones(n, k // 16).to(torch.float8_e4m3fn)
            else:
                value = torch.tensor([0.5], dtype=torch.float32)
            tensors[name] = value
        return tensors

    def _validate_checkpoint(self, tensors, keys=None, config=None):
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, str(Path(directory) / "model.safetensors"))
            validate_moe_nvfp4_checkpoint(
                self.config if config is None else config,
                self.keys if keys is None else keys,
                directory,
            )

    def test_checkpoint_accepts_supported_fp8_scale_dtypes(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self._validate_checkpoint(self._checkpoint_tensors(dtype))

    def test_checkpoint_rejects_packed_weight_and_block_scale_contract_mismatch(self):
        base = self.targets[0] + ".0.gate_proj."
        for suffix, value in (
            ("weight", torch.zeros(16, 16, dtype=torch.int8)),
            ("weight", torch.zeros(16, 32, dtype=torch.uint8)),
            ("weight_scale", torch.ones(16, 2, dtype=torch.float32)),
            ("weight_scale", torch.ones(2, 16).to(torch.float8_e4m3fn)),
            ("weight_scale_2", torch.ones(1, dtype=torch.bfloat16)),
            ("input_scale", torch.ones((), dtype=torch.float32)),
        ):
            with self.subTest(suffix=suffix, dtype=value.dtype, shape=value.shape):
                tensors = self._checkpoint_tensors()
                tensors[base + suffix] = value
                with self.assertRaisesRegex(ValueError, "dtype/shape mismatch"):
                    self._validate_checkpoint(tensors)

    def test_checkpoint_rejects_mtp_and_ple_dtype_shape_mismatch(self):
        for name, value in (
            (self.draft + ".0.up_proj.weight", torch.zeros(16, 16, dtype=torch.uint8)),
            (self.draft + ".0.up_proj.weight_scale_inv", torch.ones(1, 2)),
            (self.ple + ".shard_0.weight", torch.ones(32, dtype=torch.bfloat16)),
            (self.ple + ".weight_scale", torch.ones(2)),
        ):
            with self.subTest(name=name):
                tensors = self._checkpoint_tensors()
                tensors[name] = value
                with self.assertRaisesRegex(ValueError, "dtype/shape mismatch"):
                    self._validate_checkpoint(tensors)

    def test_checkpoint_rejects_invalid_scalar_scale_values(self):
        for name in (
            self.targets[0] + ".0.gate_proj.weight_scale_2",
            self.targets[0] + ".0.down_proj.input_scale",
            self.ple + ".weight_scale",
        ):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    tensors = self._checkpoint_tensors()
                    tensors[name] = torch.tensor([value], dtype=torch.float32)
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        self._validate_checkpoint(tensors)

    def test_checkpoint_materializes_only_scalar_companions(self):
        reads = []

        @contextmanager
        def guarded_open(*args, **kwargs):
            with safe_open(*args, **kwargs) as tensors:
                proxy = mock.Mock(wraps=tensors)

                def get_tensor(name):
                    reads.append(name)
                    self.assertTrue(
                        name.endswith((".weight_scale_2", ".input_scale"))
                        or name == self.ple + ".weight_scale"
                    )
                    return tensors.get_tensor(name)

                proxy.get_tensor.side_effect = get_tensor
                yield proxy

        with mock.patch("rtp_llm.models.qwen4_exp.moe_nvfp4.safe_open", guarded_open):
            self._validate_checkpoint(self._checkpoint_tensors())
        self.assertEqual(len(reads), 2 * 2 * 3 * 2 + 1)

    def test_checkpoint_rejects_unindexed_and_undeclared_quantized_tensors(self):
        tensors = self._checkpoint_tensors()
        name = "model.language_model.layers.0.self_attn.q_proj.weight"
        tensors[name] = torch.zeros(32, 32).to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "shard/index keys disagree"):
            self._validate_checkpoint(tensors)
        with self.assertRaisesRegex(ValueError, "undeclared quantized tensor"):
            self._validate_checkpoint(tensors, {**self.keys, name: "model.safetensors"})

    def test_checkpoint_rejects_invalid_shard_paths_before_opening(self):
        for shard in (
            "/tmp/model.safetensors",
            "../model.safetensors",
            "model.bin",
            None,
        ):
            with self.subTest(shard=shard):
                keys = {name: shard for name in self.keys}
                with self.assertRaisesRegex(ValueError, "relative safetensors"):
                    self._validate_checkpoint(self._checkpoint_tensors(), keys)

    def test_checkpoint_rejects_invalid_dimensions(self):
        for value in (0, -16, 17, "32", True):
            config = copy.deepcopy(self.config)
            config["text_config"]["hidden_size"] = value
            with self.assertRaises(ValueError):
                self._validate_checkpoint(self._checkpoint_tensors(), config=config)


if __name__ == "__main__":
    unittest.main()
