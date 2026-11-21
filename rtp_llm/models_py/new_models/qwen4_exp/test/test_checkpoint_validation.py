"""Real safetensors preflight tests, not a full Qwen4 runtime/golden test."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.new_models.qwen4_exp.checkpoint_validation import (
    validate_checkpoint_directory,
    validate_moe_fp8_checkpoint,
    validate_moe_fp8_manifest,
    validate_nvfp4_manifest,
)


class Qwen4CheckpointValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "text_config": {
                "num_hidden_layers": 1,
                "num_experts": 2,
                "hidden_size": 4,
                "moe_intermediate_size": 8,
            },
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            },
        }

    def tensors(self, draft=False):
        result = {}
        for prefix in (
            ["model.language_model", "mtp"] if draft else ["model.language_model"]
        ):
            for expert in range(2):
                for projection in ("gate", "up", "down"):
                    name = f"{prefix}.layers.0.mlp.experts.{expert}.{projection}_proj.weight"
                    shape = (4, 8) if projection == "down" else (8, 4)
                    result[name] = torch.ones(shape, dtype=torch.float8_e4m3fn)
                    result[name + "_scale_inv"] = torch.ones(1, 1, dtype=torch.bfloat16)
        return result

    def write(self, tensors, config=None):
        save_file(tensors, str(self.root / "weights.safetensors"))
        weight_map = {name: "weights.safetensors" for name in tensors}
        (self.root / "config.json").write_text(json.dumps(config or self.config))
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        return weight_map

    def test_target_and_draft_real_fp8_headers(self):
        self.write(self.tensors(draft=True))
        self.assertEqual(validate_checkpoint_directory(self.root), self.config)

    def test_bf16_source_does_not_require_quant_manifest(self):
        config = {"text_config": self.config["text_config"]}
        (self.root / "config.json").write_text(json.dumps(config))
        self.assertEqual(validate_checkpoint_directory(self.root), config)

    def test_partial_target_and_draft_rejected(self):
        for draft in (False, True):
            tensors = self.tensors(draft)
            prefix = "mtp" if draft else "model.language_model"
            key = f"{prefix}.layers.0.mlp.experts.0.up_proj.weight"
            del tensors[key]
            del tensors[key + "_scale_inv"]
            with self.subTest(draft=draft), self.assertRaisesRegex(
                ValueError, "missing.*expert"
            ):
                validate_moe_fp8_manifest(self.config, self.write(tensors))

    def test_missing_extra_and_wrong_block_scales_rejected(self):
        for mode in ("missing", "extra", "shape", "dtype"):
            tensors = self.tensors()
            name = next(k for k in tensors if k.endswith("_scale_inv"))
            if mode == "missing":
                del tensors[name]
            elif mode == "extra":
                tensors["model.embed_tokens.weight_scale_inv"] = torch.ones(1)
            elif mode == "shape":
                tensors[name] = torch.ones(2, 1)
            else:
                tensors[name] = torch.ones(1, 1, dtype=torch.int64)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_moe_fp8_checkpoint(self.config, self.write(tensors), self.root)

    def test_fp8_expert_wrong_dtype_shape_rejected(self):
        for tensor in (torch.ones(8, 4), torch.ones(7, 4).to(torch.float8_e4m3fn)):
            tensors = self.tensors()
            tensors[next(iter(tensors))] = tensor
            with self.assertRaisesRegex(ValueError, "dtype/shape"):
                validate_moe_fp8_checkpoint(self.config, self.write(tensors), self.root)

    def test_file_index_mismatch_rejected(self):
        weights = self.write(self.tensors())
        weights["not_in_file"] = "weights.safetensors"
        with self.assertRaisesRegex(ValueError, "shard/index"):
            validate_moe_fp8_checkpoint(self.config, weights, self.root)

    def test_invalid_shard_paths_rejected(self):
        for path in (
            "../weights.safetensors",
            "/tmp/weights.safetensors",
            "weights.bin",
        ):
            weights = {name: path for name in self.tensors()}
            with self.subTest(path=path), self.assertRaisesRegex(
                ValueError, "relative safetensors"
            ):
                validate_moe_fp8_checkpoint(self.config, weights, self.root)

    def test_ple_fp8_global_scale_required_and_positive(self):
        name = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
        for scale in (
            None,
            torch.tensor([0.0]),
            torch.tensor([float("nan")]),
            torch.ones(2),
            torch.ones(1),
        ):
            tensors = self.tensors()
            tensors[name + ".shard_0.weight"] = torch.ones(4, 4).to(torch.float8_e4m3fn)
            if scale is not None:
                tensors[name + ".weight_scale"] = scale
            weights = self.write(tensors)
            if scale is not None and scale.shape == (1,) and scale.item() == 1.0:
                validate_moe_fp8_checkpoint(self.config, weights, self.root)
            else:
                with self.assertRaisesRegex(ValueError, "PLE"):
                    validate_moe_fp8_checkpoint(self.config, weights, self.root)

    def test_nonexpert_fp8_not_silently_treated_as_bf16(self):
        tensors = self.tensors()
        tensors["model.language_model.layers.0.self_attn.q_proj.weight"] = torch.ones(
            4, 4
        ).to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "non-expert"):
            validate_moe_fp8_checkpoint(self.config, self.write(tensors), self.root)

    def nvfp4(self, draft=False):
        config = copy.deepcopy(self.config)
        config["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": 16,
            "with_input_scale": True,
        }
        weights = {}
        for name in self.tensors(draft):
            if name.endswith("_scale_inv"):
                continue
            weights[name] = "weights.safetensors"
            for suffix in (".weight_scale", ".weight_scale_2", ".input_scale"):
                weights[name.removesuffix(".weight") + suffix] = "weights.safetensors"
        return config, weights

    def test_nvfp4_target_draft_manifest(self):
        for draft in (False, True):
            config, weights = self.nvfp4(draft)
            validate_nvfp4_manifest(config, weights)

    def test_nvfp4_incomplete_experts_scales_and_recipe(self):
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            config, weights = self.nvfp4(True)
            del weights[f"mtp.layers.0.mlp.experts.0.up_proj.{suffix}"]
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                validate_nvfp4_manifest(config, weights)
        config, weights = self.nvfp4()
        config["quantization_config"]["group_size"] = True
        with self.assertRaisesRegex(ValueError, "group-16"):
            validate_nvfp4_manifest(config, weights)


if __name__ == "__main__":
    unittest.main()
