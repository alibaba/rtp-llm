import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.modelopt_mixed import ModelOptMixedPrecisionConfig
from rtp_llm.config.quant_config import ModelOptFp4Config, QuantizationConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    MoeAtomicWeight,
    MoeConfig,
)
from rtp_llm.model_loader.linear_attn_weight import LinearAttnAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.modelopt_mixed_weight import ModelOptMixedWeight
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    sp_moe_neg1,
    stack_,
    transpose,
)


class MemorySource:
    def __init__(self, tensors):
        self.tensors = tensors

    def load_tensor(self, name, dtype=None):
        t = self.tensors[name]
        return [t if dtype is None else t.to(dtype)]

    def has_tensor(self, name):
        return name in self.tensors


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
            Path(d, "config.json").write_text(
                json.dumps({"num_hidden_layers": 1, "quantization_config": q})
            )
            cfg = QuantizationConfig.load_from_ckpt(d)
        self.assertIsInstance(cfg, ModelOptMixedPrecisionConfig)
        self.assertEqual(cfg.get_algo(), "none")
        self.assertEqual(
            cfg.recipe_for("model.layers.{i}.proj.weight")["quant_algo"], "NVFP4"
        )

    def test_mixed_is_not_a_homogeneous_fp4_config(self):
        from rtp_llm.model_loader.mixed_fp4_quant_weight import MixedFp4Weight
        from rtp_llm.model_loader.per_group_fp4_quant_weight import PerGroupFp4Weight
        from rtp_llm.ops import QuantAlgo

        config = self.config("NVFP4")
        self.assertNotIsInstance(config, ModelOptFp4Config)
        weight = AtomicWeight(
            W.ffn_w2, [CkptWeightInfo("model.layers.{i}.proj.weight", identity)]
        )
        self.assertFalse(MixedFp4Weight.support(config, weight))
        self.assertFalse(PerGroupFp4Weight.support(config, weight))
        native = QuantAlgo()
        native.setQuantAlgo("modelopt_fp4", 4, 16)
        native.setQuantAlgo(config.get_algo(), config.bits, config.group_size())
        self.assertFalse(native.isQuant())
        self.assertFalse(native.isModelOptFP4())
        self.assertEqual(native.getWeightBits(), 16)
        self.assertEqual(config.get_supported_compute_dtypes(), [torch.bfloat16])
        with self.assertRaises(ValueError):
            config.verify_compute_dtype_and_kv_cache_dtype(
                torch.float16, torch.bfloat16
            )

    def test_recipe_keys_accept_weight_suffix(self):
        cfg = self.config(prefix="model.layers.0.proj.weight")
        self.assertEqual(
            cfg.recipe_for("model.layers.{i}.proj.weight")["quant_algo"], "FP8"
        )

    def test_expert_indices_share_one_descriptor_recipe(self):
        recipe = {"quant_algo": "NVFP4", "group_size": 16}
        cfg = ModelOptMixedPrecisionConfig(
            {
                "model.layers.0.mlp.experts.0.down_proj.weight": recipe,
                "model.layers.0.mlp.experts.1.down_proj.weight": recipe,
            },
            num_layers=1,
        )
        self.assertEqual(
            cfg.recipe_for("model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight"),
            recipe,
        )

    def test_unsupported_nvfp4_descriptor_reports_weight_name(self):
        cfg = self.config("NVFP4")
        descriptor = AtomicWeight(
            W.attn_gate_w,
            [CkptWeightInfo("model.layers.{i}.proj.weight", identity)],
        )
        with self.assertRaisesRegex(ValueError, "Unsupported mixed NVFP4 weight"):
            WeightModule.create(descriptor, cfg)

    def test_unmatched_quantized_tensor_fails_before_bf16_conversion(self):
        cfg = self.config(prefix="model.layers.0.other")
        descriptor = AtomicWeight(
            W.attn_o_w,
            [CkptWeightInfo("model.layers.{i}.proj.weight", identity)],
        )
        raw = WeightModule.create(descriptor, cfg)
        for dtype in (torch.float8_e4m3fn, torch.uint8):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(
                ValueError, "Missing mixed ModelOpt recipe"
            ):
                source = MemorySource(
                    {"model.layers.0.proj.weight": torch.zeros(4, 8).to(dtype)}
                )
                raw._load_raw_tensor(
                    source,
                    0,
                    "cpu",
                    SimpleNamespace(compute_dtype=torch.bfloat16),
                )

    def unmatched_moe(self, stacked=False):
        name = "model.layers.{i}.mlp.experts."
        name += "down_proj.weight" if stacked else "{expert_id}.down_proj.weight"
        return WeightModule.create(
            MoeAtomicWeight(
                W.moe_w2,
                [CkptWeightInfo(name, identity)],
                stack_,
                config=MoeConfig(expert_num=2),
                stacked_ckpt_keys=stacked,
                split_func=sp_moe_neg1,
                enable_pure_tp_preshard=True,
            ),
            self.config(prefix="model.layers.0.other"),
        )

    def test_unmatched_moe_rejects_quantized_codes_before_serial_or_gpu_cast(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        config = LoadConfig.model_construct(
            compute_dtype=torch.bfloat16,
            ep_size=1,
            ep_rank=0,
            phy2log=None,
            moe_pure_tp_preshard=False,
        )
        for stacked in (False, True):
            loader = self.unmatched_moe(stacked)
            for dtype in (torch.float8_e4m3fn, torch.uint8, torch.bfloat16):
                if stacked:
                    tensors = {
                        loader.weights[0]
                        .tensor_name(0): torch.ones(2, 16, 16)
                        .to(dtype)
                    }
                else:
                    tensors = {
                        loader._ckpt_name(loader.weights[0], 0, expert): torch.ones(
                            16, 16
                        ).to(dtype if expert else torch.bfloat16)
                        for expert in range(2)
                    }
                for device in devices:
                    with self.subTest(stacked=stacked, dtype=dtype, device=device):
                        if dtype == torch.bfloat16:
                            result = loader._load_raw_tensor(
                                MemorySource(tensors), 0, device, config
                            )
                            torch.testing.assert_close(
                                result[W.moe_w2].cpu(),
                                torch.ones(2, 16, 16, dtype=dtype),
                            )
                        else:
                            with self.assertRaisesRegex(
                                ValueError, "Missing mixed ModelOpt recipe"
                            ):
                                loader._load_raw_tensor(
                                    MemorySource(tensors), 0, device, config
                                )

    def test_unmatched_moe_preshard_checks_raw_safetensor_slices(self):
        config = LoadConfig.model_construct(
            compute_dtype=torch.bfloat16,
            ep_size=1,
            ep_rank=0,
            phy2log=None,
            moe_pure_tp_preshard=True,
            moe_pure_tp_mode=True,
            tp_size=2,
            tp_rank=0,
            merge_lora=False,
        )
        for stacked in (False, True):
            loader = self.unmatched_moe(stacked)
            for dtype in (torch.float8_e4m3fn, torch.uint8, torch.bfloat16):
                shape = (2, 16, 16) if stacked else (16, 16)
                tensors = {
                    loader._ckpt_name(loader.weights[0], 0, expert): torch.ones(
                        shape
                    ).to(dtype)
                    for expert in ([0] if stacked else range(2))
                }
                with self.subTest(
                    stacked=stacked, dtype=dtype
                ), tempfile.TemporaryDirectory() as directory:
                    save_file(tensors, str(Path(directory, "model.safetensors")))
                    database = CkptDatabase(directory)
                    try:
                        source = DatabaseTensorSource(database)
                        with patch.object(
                            database,
                            "load_tensor_slice",
                            wraps=database.load_tensor_slice,
                        ) as sliced:
                            if dtype == torch.bfloat16:
                                result = loader._load_pure_tp(source, 0, "cpu", config)
                                torch.testing.assert_close(
                                    result, torch.ones(2, 16, 8, dtype=dtype)
                                )
                            else:
                                with self.assertRaisesRegex(
                                    ValueError, "Missing mixed ModelOpt recipe"
                                ):
                                    loader._load_pure_tp(source, 0, "cpu", config)
                            self.assertTrue(sliced.called)
                    finally:
                        for checkpoint in database.pretrain_file_list:
                            checkpoint.close_safetensor_handle()

    def test_deployment_loader_uses_module_recipes_without_global_quant_algo(self):
        from rtp_llm.model_loader.model_weight_info import (
            ModelDeployWeightInfo,
            ModelWeightInfo,
        )
        from rtp_llm.ops import QuantAlgo
        from rtp_llm.utils.model_weight import WeightStyle

        source = AtomicWeight(
            W.attn_o_w, [CkptWeightInfo("model.layers.{i}.proj.weight", identity)]
        )
        owner = ModelDeployWeightInfo.__new__(ModelDeployWeightInfo)
        owner._get_weight_info = lambda: ModelWeightInfo([], [[source]])
        owner._quant_algo = QuantAlgo()
        owner._quant_config = self.config()
        owner.model_config = SimpleNamespace(enable_w4a16_sm120_dense_ffn=False)
        owner.output_vocab_ids = ()
        owner.weight_style = WeightStyle.NONE
        owner.gen_dummy_reciprocal = False
        owner.tie_word_embeddings = False
        owner.enable_fp32_lm_head = False
        result = owner.get_weight_info()
        self.assertIsInstance(result.layer_weights[0][0], ModelOptMixedWeight)
        self.assertFalse(owner._quant_algo.isQuant())

    def test_nvfp4_head_discovers_prefixed_companion_tensors(self):
        for prefix in ("lm_head", "language_model.lm_head", "transformer.output_layer"):
            with self.subTest(prefix=prefix):
                loader = WeightModule.create(
                    AtomicWeight(
                        W.lm_head, [CkptWeightInfo(prefix + ".weight", identity)]
                    ),
                    self.config("NVFP4", prefix),
                )
                names = {
                    prefix + suffix
                    for suffix in (
                        ".weight",
                        ".weight_scale",
                        ".weight_scale_2",
                        ".input_scale",
                    )
                }
                self.assertEqual(loader.get_tensor_names(None, None), names)
                raw = loader._load_raw_tensor(
                    MemorySource(
                        {
                            prefix + ".weight": torch.ones(16, 16, dtype=torch.uint8),
                            prefix
                            + ".weight_scale": torch.ones(16, 2).to(
                                torch.float8_e4m3fn
                            ),
                            prefix + ".weight_scale_2": torch.tensor(0.5),
                            prefix + ".input_scale": torch.tensor(0.25),
                        }
                    ),
                    None,
                    "cpu",
                    SimpleNamespace(compute_dtype=torch.bfloat16),
                )
                self.assertEqual(raw[W.lm_head].dtype, torch.uint8)
                self.assertEqual(raw[W.lm_head_s].dtype, torch.float8_e4m3fn)
                self.assertEqual(raw[W.lm_head_s2].item(), 0.5)
                self.assertEqual(raw[W.lm_head_i_s].item(), 0.25)

    def test_missing_layer_fails_during_config_parsing(self):
        recipe = {"quant_algo": "FP8"}
        with self.assertRaisesRegex(ValueError, "Incomplete.*missing layers.*1"):
            ModelOptMixedPrecisionConfig(
                {"model.layers.0.proj": recipe, "model.layers.2.proj": recipe},
                num_layers=3,
            )

    def test_missing_projection_is_found_without_architecture_metadata(self):
        recipe = {"quant_algo": "FP8"}
        with self.assertRaisesRegex(ValueError, "Incomplete.*other"):
            ModelOptMixedPrecisionConfig(
                {
                    "model.layers.0.proj": recipe,
                    "model.layers.1.proj": recipe,
                    "model.layers.0.other": recipe,
                }
            )

    def test_hybrid_layer_coverage_and_auxiliary_ignores(self):
        recipe = {"quant_algo": "FP8"}
        layers = {
            "model.layers.0.linear_attn.out_proj": recipe,
            "model.layers.2.linear_attn.out_proj": recipe,
            "model.layers.1.self_attn.o_proj": recipe,
        }
        config = ModelOptMixedPrecisionConfig(
            layers,
            ignore=["mtp*", "mtp.layers.0*"],
            num_layers=3,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
        )
        self.assertEqual(
            config.recipe_for("model.layers.{i}.self_attn.o_proj.weight"), recipe
        )
        del layers["model.layers.2.linear_attn.out_proj"]
        with self.assertRaisesRegex(ValueError, "Incomplete.*missing layers.*2"):
            ModelOptMixedPrecisionConfig(
                layers,
                num_layers=3,
                layer_types=["linear_attention", "full_attention", "linear_attention"],
            )

    def test_numbered_main_layer_ignore_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Layer-specific ignore"):
            ModelOptMixedPrecisionConfig(
                {
                    "model.layers.0.proj": {"quant_algo": "FP8"},
                    "model.layers.1.proj": {"quant_algo": "FP8"},
                },
                ignore=["model.layers.1.*"],
            )

    def test_ignored_only_layer_cannot_leak_a_template_recipe(self):
        for pattern in ("model.layers.0.*", "*.layers.0.*", "model.layers.[0].*"):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(
                ValueError, "Layer-specific ignore"
            ):
                ModelOptMixedPrecisionConfig(
                    {"model.layers.0.proj": {"quant_algo": "FP8"}}, ignore=[pattern]
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

    def test_nvfp4_moe_loads_only_selected_experts_with_per_expert_scales(self):
        for weight, projections, scale_key in (
            (W.moe_w1, ["gate_proj", "up_proj"], W.moe_w1_s2),
            (W.moe_w2, ["down_proj"], W.moe_w2_s2),
        ):
            with self.subTest(weight=weight):
                names = [
                    f"model.layers.{{i}}.mlp.experts.{{expert_id}}.{p}.weight"
                    for p in projections
                ]
                cfg = ModelOptMixedPrecisionConfig(
                    {
                        name.format(i=0, expert_id=e): {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                        }
                        for name in names
                        for e in range(4)
                    }
                )
                loader = WeightModule.create(
                    MoeAtomicWeight(
                        weight,
                        [CkptWeightInfo(name, identity) for name in names],
                        config=MoeConfig(expert_num=4),
                    ),
                    cfg,
                )
                load_config = LoadConfig.model_construct(
                    ep_size=2,
                    ep_rank=1,
                    phy2log=[[1, 3, 2, 0]],
                    moe_pure_tp_mode=False,
                )
                tensors = {}
                for name in names:
                    for expert in (2, 0):
                        prefix = name.format(i=0, expert_id=expert).removesuffix(
                            ".weight"
                        )
                        tensors[prefix + ".weight"] = torch.full(
                            (16, 16), expert + 1, dtype=torch.uint8
                        )
                        tensors[prefix + ".weight_scale"] = torch.ones(16, 2).to(
                            torch.float8_e4m3fn
                        )
                        tensors[prefix + ".weight_scale_2"] = torch.tensor(
                            float(expert + 1)
                        )
                        tensors[prefix + ".input_scale"] = torch.tensor(
                            float(expert + 1) / 2
                        )
                raw = loader._load_raw_tensor(
                    MemorySource(tensors), 0, "cpu", load_config
                )
                self.assertEqual(raw[weight].dtype, torch.uint8)
                self.assertEqual(
                    tuple(raw[weight].shape), (2, 16 * len(projections), 16)
                )
                self.assertTrue(torch.all(raw[weight][0] == 3))
                self.assertTrue(torch.all(raw[weight][1] == 1))
                self.assertEqual(raw[scale_key].flatten().tolist(), [3.0, 1.0])
                if weight == W.moe_w1:
                    for suffix in (".weight_scale_2", ".input_scale"):
                        broken = dict(tensors)
                        prefix = (
                            names[1].format(i=0, expert_id=2).removesuffix(".weight")
                        )
                        broken[prefix + suffix] = torch.tensor(7.0)
                        with self.assertRaisesRegex(
                            ValueError, "Cannot fuse unequal NVFP4"
                        ):
                            loader._load_raw_tensor(
                                MemorySource(broken), 0, "cpu", load_config
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
