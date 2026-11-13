import copy
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models.qwen4_exp.moe_fp8 import (
    validate_moe_fp8_checkpoint,
    validate_moe_fp8_manifest,
)


class Qwen4MoeFp8Test(unittest.TestCase):
    def test_missing_or_null_quantization_metadata_rejects_with_value_error(self):
        for config in ({}, {"quantization_config": None}):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "requires dynamic"):
                    validate_moe_fp8_manifest(config, {})

    def setUp(self):
        self.config = {
            "text_config": {
                "num_hidden_layers": 2,
                "num_experts": 2,
                "hidden_size": 256,
                "moe_intermediate_size": 128,
            },
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["lm_head"],
            },
        }
        self.keys = {}
        for prefix, layers in (("model.language_model", 2), ("mtp", 1)):
            for layer in range(layers):
                for expert in range(2):
                    for projection in ("gate", "up", "down"):
                        name = (
                            f"{prefix}.layers.{layer}.mlp.experts.{expert}."
                            f"{projection}_proj.weight"
                        )
                        self.keys[name] = "model.safetensors"
                        self.keys[name + "_scale_inv"] = "model.safetensors"

    def test_accepts_complete_target_and_draft(self):
        validate_moe_fp8_manifest(self.config, self.keys)

    def test_rejects_missing_projection_and_scale(self):
        keys = dict(self.keys)
        name = next(iter(keys))
        del keys[name]
        del keys[name + "_scale_inv"]
        with self.assertRaisesRegex(ValueError, "missing target"):
            validate_moe_fp8_manifest(self.config, keys)

    def test_rejects_non_expert_quantization(self):
        keys = dict(self.keys)
        keys["model.language_model.layers.0.self_attn.q_proj.weight_scale_inv"] = "x"
        with self.assertRaisesRegex(ValueError, "exactly the split MoE"):
            validate_moe_fp8_manifest(self.config, keys)

    def test_rejects_scale_coverage_mismatch(self):
        keys = dict(self.keys)
        del keys[next(name for name in keys if name.endswith("_scale_inv"))]
        with self.assertRaisesRegex(ValueError, "exactly the split MoE"):
            validate_moe_fp8_manifest(self.config, keys)

    def test_rejects_wrong_layer_with_same_key_count(self):
        keys = {
            name.replace("layers.1.", "layers.9."): file
            for name, file in self.keys.items()
        }
        with self.assertRaisesRegex(ValueError, "missing target"):
            validate_moe_fp8_manifest(self.config, keys)

    def test_rejects_partial_draft(self):
        keys = dict(self.keys)
        name = "mtp.layers.0.mlp.experts.0.gate_proj.weight"
        del keys[name]
        del keys[name + "_scale_inv"]
        with self.assertRaisesRegex(ValueError, "missing draft"):
            validate_moe_fp8_manifest(self.config, keys)

    def test_rejects_unsupported_quantization(self):
        for change in (
            {"activation_scheme": "static"},
            {"weight_block_size": [64, 128]},
            {"modules_to_not_convert": []},
        ):
            config = copy.deepcopy(self.config)
            config["quantization_config"].update(change)
            with self.assertRaises(ValueError):
                validate_moe_fp8_manifest(config, self.keys)

    def _checkpoint_tensors(self, scale_dtype=torch.bfloat16):
        tensors = {}
        for name in self.keys:
            shape = (256, 128) if "down_proj" in name else (128, 256)
            if name.endswith("_scale_inv"):
                shape = tuple(size // 128 for size in shape)
                tensors[name] = torch.ones(shape, dtype=scale_dtype)
            else:
                tensors[name] = torch.zeros(shape).to(torch.float8_e4m3fn)
        return tensors

    def _validate_checkpoint(self, tensors, keys=None):
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, str(Path(directory) / "model.safetensors"))
            validate_moe_fp8_checkpoint(
                self.config, self.keys if keys is None else keys, directory
            )

    def test_checkpoint_accepts_official_scale_dtypes(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(scale_dtype=dtype):
                self._validate_checkpoint(self._checkpoint_tensors(dtype))

    def test_checkpoint_rejects_expert_dtype_and_shape_mismatch(self):
        weight = next(name for name in self.keys if not name.endswith("_scale_inv"))
        for name, value in (
            (weight, torch.zeros(128, 256, dtype=torch.bfloat16)),
            (weight, torch.zeros(129, 256).to(torch.float8_e4m3fn)),
            (weight + "_scale_inv", torch.ones(1, 2, dtype=torch.int64)),
            (weight + "_scale_inv", torch.ones(2, 2, dtype=torch.bfloat16)),
        ):
            with self.subTest(tensor=name, dtype=value.dtype, shape=value.shape):
                tensors = self._checkpoint_tensors()
                tensors[name] = value
                with self.assertRaisesRegex(ValueError, "dtype/shape mismatch"):
                    self._validate_checkpoint(tensors)

    def test_checkpoint_rejects_non_expert_fp8_and_unindexed_tensor(self):
        tensors = self._checkpoint_tensors()
        name = "model.language_model.layers.0.self_attn.q_proj.weight"
        tensors[name] = torch.zeros(256, 256).to(torch.float8_e4m3fn)
        keys = {**self.keys, name: "model.safetensors"}
        with self.assertRaisesRegex(ValueError, "non-expert tensor"):
            self._validate_checkpoint(tensors, keys)
        with self.assertRaisesRegex(ValueError, "shard/index keys disagree"):
            self._validate_checkpoint(tensors)

    def test_checkpoint_rejects_nonrelative_shard(self):
        for shard in ("/tmp/model.safetensors", "../model.safetensors"):
            with self.subTest(shard=shard):
                keys = {name: shard for name in self.keys}
                with self.assertRaisesRegex(ValueError, "relative safetensors"):
                    self._validate_checkpoint(self._checkpoint_tensors(), keys)

    def test_checkpoint_ple_requires_finite_global_scale(self):
        name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
        scale_name = name.rsplit(".shard_", 1)[0] + ".weight_scale"
        tensors = self._checkpoint_tensors()
        tensors[name] = torch.ones(2, 160).to(torch.float8_e4m3fn)
        keys = {**self.keys, name: "model.safetensors"}
        with self.assertRaisesRegex(ValueError, "missing its global scale"):
            self._validate_checkpoint(tensors, keys)
        keys[scale_name] = "model.safetensors"
        for value in (0.00019931793212890625, 0.0, -1.0, float("nan")):
            with self.subTest(value=value):
                tensors[scale_name] = torch.tensor([value], dtype=torch.bfloat16)
                if value > 0:
                    self._validate_checkpoint(tensors, keys)
                else:
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        self._validate_checkpoint(tensors, keys)


if __name__ == "__main__":
    unittest.main()
