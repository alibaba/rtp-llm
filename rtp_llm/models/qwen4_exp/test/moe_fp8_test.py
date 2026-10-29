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

    def test_accepts_ignored_layers_exclusion_dialect(self):
        # ModelOpt-style exports record the same non-expert exclusion list as
        # "ignored_layers" and omit "modules_to_not_convert" entirely.
        config = copy.deepcopy(self.config)
        quant = config["quantization_config"]
        quant["ignored_layers"] = quant.pop("modules_to_not_convert")
        validate_moe_fp8_manifest(config, self.keys)

    def test_ignored_layers_fallback_used_when_official_list_empty(self):
        config = copy.deepcopy(self.config)
        quant = config["quantization_config"]
        quant["ignored_layers"] = quant["modules_to_not_convert"]
        quant["modules_to_not_convert"] = []
        validate_moe_fp8_manifest(config, self.keys)

    def test_official_exclusions_take_precedence_over_ignored_layers(self):
        # A non-empty official list validates on its own even when the alias
        # dialect key is present but empty.
        config = copy.deepcopy(self.config)
        config["quantization_config"]["ignored_layers"] = []
        validate_moe_fp8_manifest(config, self.keys)

    def test_rejects_missing_exclusions_in_both_dialects(self):
        for change in (
            {},
            {"ignored_layers": []},
        ):
            with self.subTest(change=change):
                config = copy.deepcopy(self.config)
                del config["quantization_config"]["modules_to_not_convert"]
                config["quantization_config"].update(change)
                with self.assertRaisesRegex(
                    ValueError, "explicit non-expert exclusions"
                ):
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


class Qwen4NativeFp8LoaderTest(unittest.TestCase):
    def test_native_tp_shards_preserve_bytes_scales_and_moe_math(self):
        from types import SimpleNamespace
        from unittest import mock

        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
        from rtp_llm.model_loader.tensor_source import TensorSource
        from rtp_llm.models.qwen4_exp.native_fp8 import Qwen4NativeFp8MoeWeight
        from rtp_llm.utils.model_weight import CkptWeightInfo, W

        # 5 intermediate blocks expose the real TP4/TP8 boundary bug; non-square,
        # distinct up/gate/expert/block values expose permutations and scale loss.
        tensors = {}
        for expert in range(2):
            for projection in ("up", "gate", "down"):
                shape = (256, 640) if projection == "down" else (640, 256)
                n, k = shape
                values = (torch.arange(n * k).reshape(shape) % 13 - 6).float()
                values = values + expert + (2 if projection == "gate" else 0)
                name = f"layers.0.experts.{expert}.{projection}.weight"
                tensors[name] = values.to(torch.float8_e4m3fn)
                tensors[name + "_scale_inv"] = (
                    torch.arange(n // 128 * (k // 128)).reshape(n // 128, k // 128) + 1
                ).float() / (512 if projection == "down" else 128)

        class Source(TensorSource):
            def load_tensor(self, name, data_type=torch.float16):
                return [tensors[name].to(data_type)]

            def has_tensor(self, name):
                return name in tensors

            def get_database(self):
                return None

        def dequant(weight, scale):
            return weight.double() * scale.double().repeat_interleave(
                128, -2
            ).repeat_interleave(128, -1)

        def source_value(expert, projection):
            name = f"layers.0.experts.{expert}.{projection}.weight"
            return dequant(tensors[name], tensors[name + "_scale_inv"])

        x = (torch.arange(3 * 256).reshape(3, 256) % 11 - 5).double() / 32
        expected = []
        for expert in range(2):
            up = x @ source_value(expert, "up").T
            gate = x @ source_value(expert, "gate").T
            expected.append(
                (up * torch.nn.functional.silu(gate)) @ source_value(expert, "down").T
            )
        expected = torch.stack(expected)

        for tp in (1, 2, 4, 8):
            actual = torch.zeros_like(expected)
            shards = []
            for rank in range(tp):
                config = SimpleNamespace(
                    tp_size=tp,
                    tp_rank=rank,
                    ep_size=1,
                    ep_rank=0,
                    dp_size=1,
                    dp_rank=0,
                    ffn_tp_size=tp,
                    ffn_tp_rank=rank,
                    hidden_size=256,
                    head_num=2,
                    head_num_kv=2,
                    size_per_head=128,
                    moe_pure_tp_mode=True,
                    moe_pure_tp_preshard=True,
                    bit=8,
                    merge_lora=False,
                    compute_dtype=torch.bfloat16,
                    use_swizzleA=False,
                    get_selected_experts=lambda layer, count: range(count),
                    exported_device=SimpleNamespace(
                        shuffle_moe_weight=lambda value, *args: value,
                        maybe_rewrite_weight_by_key=lambda name, value, **kwargs: value,
                    ),
                )
                local = {}
                for name, projections in (
                    (W.moe_w1, ("up", "gate")),
                    (W.moe_w2, ("down",)),
                ):
                    raw = MoeAtomicWeight(
                        name,
                        [
                            CkptWeightInfo(
                                f"layers.{{i}}.experts.{{expert_id}}.{p}.weight"
                            )
                            for p in projections
                        ],
                        config=MoeConfig(expert_num=2),
                    )
                    descriptor = Qwen4NativeFp8MoeWeight(
                        raw, Fp8BlockWiseQuantConfig(is_quanted=True)
                    )
                    with mock.patch(
                        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                        return_value=False,
                    ):
                        local.update(descriptor.load(Source(), 0, "cpu", config))
                self.assertEqual(local[W.moe_w1].dtype, torch.float8_e4m3fn)
                self.assertEqual(local[W.moe_w2].dtype, torch.float8_e4m3fn)
                upgate = dequant(local[W.moe_w1], local[W.moe_s1])
                down = dequant(local[W.moe_w2], local[W.moe_s2])
                up, gate = (x @ upgate.transpose(-1, -2)).chunk(2, -1)
                actual += (up * torch.nn.functional.silu(gate)) @ down.transpose(-1, -2)
                shards.append(local)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            for projection_index, projection in enumerate(("up", "gate")):
                restored = torch.cat(
                    [
                        part[W.moe_w1].float().chunk(2, 1)[projection_index]
                        for part in shards
                    ],
                    1,
                )[:, :640]
                source = torch.stack(
                    [
                        tensors[
                            f"layers.0.experts.{expert}.{projection}.weight"
                        ].float()
                        for expert in range(2)
                    ]
                )
                torch.testing.assert_close(restored, source, rtol=0, atol=0)
            restored = torch.cat([part[W.moe_w2].float() for part in shards], 2)[
                ..., :640
            ]
            source = torch.stack(
                [
                    tensors[f"layers.0.experts.{expert}.down.weight"].float()
                    for expert in range(2)
                ]
            )
            torch.testing.assert_close(restored, source, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
