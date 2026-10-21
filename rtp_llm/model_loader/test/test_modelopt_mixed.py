import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.config.modelopt_mixed import ModelOptMixedPrecisionConfig
from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig
from rtp_llm.model_loader.linear_attn_weight import LinearAttnAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.modelopt_mixed_weight import ModelOptMixedWeight
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose


class MemorySource:
    def __init__(self, tensors):
        self.tensors = tensors

    def load_tensor(self, name, dtype=None):
        t = self.tensors[name]
        return [t if dtype is None else t.to(dtype)]


class ModelOptMixedTest(unittest.TestCase):
    def config(self, algorithm="FP8", prefix="model.layers.0.proj"):
        recipe = {"quant_algo": algorithm}
        if algorithm == "NVFP4":
            recipe["group_size"] = 16
        return ModelOptMixedPrecisionConfig({prefix: recipe}, ignore=["mtp*"])

    def test_parse_real_group_order(self):
        q = {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.proj": {"quant_algo": "NVFP4", "group_size": 16}
            },
            "config_groups": {"group_0": {"weights": {"num_bits": 8}}},
            "ignore": ["mtp*"],
        }
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({"quantization_config": q}))
            cfg = QuantizationConfig.load_from_ckpt(d)
        self.assertIsInstance(cfg, ModelOptMixedPrecisionConfig)
        self.assertEqual(cfg.get_algo(), "modelopt_fp4")
        self.assertEqual(
            cfg.recipe_for("model.layers.{i}.proj.weight")["quant_algo"], "NVFP4"
        )

    def test_excluded_mtp_is_unchanged(self):
        cfg = self.config(prefix="mtp.layers.0.proj")
        source = AtomicWeight(
            W.ffn_w2, [CkptWeightInfo("mtp.layers.{i}.proj.weight", identity)]
        )
        self.assertIs(WeightModule.create(source, cfg), source)

    def test_unknown_scheme_fails(self):
        with self.assertRaisesRegex(ValueError, "Unsupported ModelOpt recipe"):
            self.config("INT4")

    def test_block32_is_not_nvfp4(self):
        with self.assertRaisesRegex(ValueError, "Unsupported ModelOpt recipe"):
            ModelOptMixedPrecisionConfig(
                {"proj": {"quant_algo": "NVFP4", "group_size": 32}}
            )

    def test_layer_dependent_scheme_rejected(self):
        with self.assertRaisesRegex(ValueError, "Layer-dependent"):
            ModelOptMixedPrecisionConfig(
                {
                    "model.layers.0.proj": {"quant_algo": "FP8"},
                    "model.layers.1.proj": {"quant_algo": "NVFP4", "group_size": 16},
                }
            )

    def test_fp8_codes_and_scales_survive_loading(self):
        cfg = self.config()
        descriptor = AtomicWeight(
            W.attn_o_w,
            [CkptWeightInfo("model.layers.{i}.proj.weight", identity)],
            transpose,
        )
        loader = WeightModule.create(descriptor, cfg)
        self.assertIsInstance(loader, ModelOptMixedWeight)
        codes = (
            torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.float8_e4m3fn)
        )
        source = MemorySource(
            {
                "model.layers.0.proj.weight": codes,
                "model.layers.0.proj.weight_scale": torch.tensor(0.25),
                "model.layers.0.proj.input_scale": torch.tensor(0.125),
            }
        )
        raw = loader._load_raw_tensor(source, 0, "cpu", None)
        split = loader._split(raw, SimpleNamespace(tp_size=1, dp_size=1, ep_size=1))
        result = loader._postprocess(split, "cpu", None)
        self.assertTrue(
            torch.equal(result[W.attn_o_w].view(torch.uint8), codes.view(torch.uint8))
        )
        torch.testing.assert_close(
            result[W.attn_o_s], torch.tensor([[0.25] * 4, [0.125] * 4])
        )

    def test_fused_gdn_fp8_scales_follow_tp_shards(self):
        names = ["model.layers.0.in_proj_qkv", "model.layers.0.in_proj_z"]
        cfg = ModelOptMixedPrecisionConfig(
            {name: {"quant_algo": "FP8"} for name in names}
        )
        # Two key heads, four value heads, dimensions 2: Q/K/V/Z = 4/4/8/8.
        linear = SimpleNamespace(
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
        )
        descriptor = LinearAttnAtomicWeight(
            W.linear_attn_qkvz_w,
            [
                CkptWeightInfo(name.replace(".0.", ".{i}.") + ".weight", identity)
                for name in names
            ],
            lambda values: torch.cat(values, dim=0).T,
            config=linear,
        )
        loader = WeightModule.create(descriptor, cfg)
        qkv = torch.arange(16 * 8).reshape(16, 8).to(torch.float8_e4m3fn)
        z = torch.arange(8 * 8).reshape(8, 8).to(torch.float8_e4m3fn)
        tensors = {}
        for name, codes, ws, xs in zip(names, [qkv, z], [0.25, 0.5], [0.125, 0.0625]):
            tensors.update(
                {
                    name + ".weight": codes,
                    name + ".weight_scale": torch.tensor(ws),
                    name + ".input_scale": torch.tensor(xs),
                }
            )
        raw = loader._load_raw_tensor(MemorySource(tensors), 0, "cpu", None)
        codes = torch.cat([qkv, z], dim=0)
        scales = torch.tensor([[0.25] * 16 + [0.5] * 8, [0.125] * 16 + [0.0625] * 8])
        for rank, rows in [
            (0, [0, 1, 4, 5, 8, 9, 10, 11, 16, 17, 18, 19]),
            (1, [2, 3, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23]),
        ]:
            with self.subTest(rank=rank):
                split = loader._split(raw, SimpleNamespace(tp_size=2, tp_rank=rank))
                result = loader._postprocess(split, "cpu", None)
                self.assertTrue(
                    torch.equal(
                        result[W.linear_attn_qkvz_w].view(torch.uint8),
                        codes.view(torch.uint8)[rows],
                    )
                )
                torch.testing.assert_close(
                    result[W.linear_attn_qkvz_s], scales[:, rows]
                )

    def test_nvfp4_head_uses_lm_head_tp_for_codes_and_block_scales(self):
        cfg = self.config("NVFP4", "lm_head")
        loader = WeightModule.create(
            AtomicWeight(W.lm_head, [CkptWeightInfo("lm_head.weight", identity)]), cfg
        )
        codes = torch.arange(32 * 16).reshape(32, 16).to(torch.uint8)
        blocks = torch.arange(64).reshape(32, 2).to(torch.float8_e4m3fn)
        source = MemorySource(
            {
                "lm_head.weight": codes,
                "lm_head.weight_scale": blocks,
                "lm_head.weight_scale_2": torch.tensor(0.5),
                "lm_head.input_scale": torch.tensor(0.25),
            }
        )
        for rank in (0, 1):
            with self.subTest(rank=rank):
                config = LoadConfig.model_construct(
                    tp_size=1,
                    tp_rank=0,
                    ep_size=1,
                    ep_rank=0,
                    dp_size=1,
                    dp_rank=0,
                    lm_head_tp_size=2,
                    lm_head_tp_rank=rank,
                    ffn_tp_size=1,
                    ffn_tp_rank=0,
                    hidden_size=32,
                    head_num=2,
                    head_num_kv=2,
                    size_per_head=16,
                    moe_pure_tp_mode=False,
                )
                raw = loader._load_raw_tensor(source, None, "cpu", config)
                result = loader._split(raw, config)
                rows = slice(rank * 16, (rank + 1) * 16)
                self.assertTrue(torch.equal(result[W.lm_head], codes[rows]))
                self.assertTrue(
                    torch.equal(
                        result[W.lm_head_s].view(torch.uint8),
                        blocks[rows].view(torch.uint8),
                    )
                )
                self.assertEqual(result[W.lm_head_s2].item(), 0.5)
                self.assertEqual(result[W.lm_head_i_s].item(), 0.25)

    def test_nvfp4_fusion_rejects_unequal_global_or_input_scales(self):
        names = ["model.layers.0.gate", "model.layers.0.up"]
        cfg = ModelOptMixedPrecisionConfig(
            {name: {"quant_algo": "NVFP4", "group_size": 16} for name in names}
        )
        descriptor = FfnAtomicWeight(
            W.ffn_w13,
            [
                CkptWeightInfo(name.replace(".0.", ".{i}.") + ".weight", identity)
                for name in names
            ],
            config=FfnConfig(),
        )
        loader = WeightModule.create(descriptor, cfg)
        for unequal_suffix in (".weight_scale_2", ".input_scale"):
            with self.subTest(suffix=unequal_suffix):
                tensors = {
                    name + suffix: torch.tensor(1.0)
                    for name in names
                    for suffix in (".weight_scale_2", ".input_scale")
                }
                tensors[names[1] + unequal_suffix] = torch.tensor(2.0)
                with self.assertRaisesRegex(ValueError, "Cannot fuse unequal NVFP4"):
                    loader._load_raw_tensor(MemorySource(tensors), 0, "cpu", None)

    def test_fp8_rejects_cast_quantized_codes(self):
        cfg = self.config()
        descriptor = AtomicWeight(
            W.attn_o_w,
            [CkptWeightInfo("model.layers.{i}.proj.weight", identity)],
            transpose,
        )
        loader = WeightModule.create(descriptor, cfg)
        with self.assertRaisesRegex(ValueError, "Expected FP8 checkpoint"):
            loader._load_raw_tensor(
                MemorySource({"model.layers.0.proj.weight": torch.zeros(4, 8)}),
                0,
                "cpu",
                None,
            )

    def test_discovery_contains_companion_scales(self):
        cfg = self.config()
        descriptor = AtomicWeight(
            W.attn_o_w,
            [CkptWeightInfo("model.layers.{i}.proj.weight", identity)],
            transpose,
        )
        loader = WeightModule.create(descriptor, cfg)
        self.assertEqual(
            loader.get_tensor_names(3, None),
            {
                "model.layers.3.proj.weight",
                "model.layers.3.proj.weight_scale",
                "model.layers.3.proj.input_scale",
            },
        )


if __name__ == "__main__":
    unittest.main()
