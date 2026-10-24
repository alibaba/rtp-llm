import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import (
    QWEN4_FP8_NVFP4_RECIPE,
    Fp8BlockWiseQuantConfig,
    ModelOptFp4Config,
    init_quant_config,
)
from rtp_llm.model_factory_register import ModelDict, _model_factory
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import Qwen4ExpWeight
from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeMixin,
)
from rtp_llm.ops import (
    DataType,
    HWKernelConfig,
    HybridAttentionType,
    KvCacheDataType,
    ParallelismConfig,
    RoleType,
    RopeStyle,
)
from rtp_llm.utils.model_weight import W

_NUM_LAYERS = 48

# Key patterns taken from the real Qwen/Qwen3.8-Flash-Next weight map; see
# docs/design/qwen3.8_flash_next_support_design.md appendix C.
_CKPT_KEYS = (
    [
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    ]
    + [
        f"model.language_model.layers.{i}.{module}.{suffix}"
        for i in range(_NUM_LAYERS)
        for module in ("attn_hyper_connection", "mlp_hyper_connection")
        for suffix in (
            "hc_norm.weight",
            "input_mix_weight_down.weight",
            "input_mix_weight_up.weight",
            "block_inject_weight.weight",
        )
    ]
    + [
        f"model.language_model.layers.{i}.mlp.experts.{proj}"
        for i in range(_NUM_LAYERS)
        for proj in ("gate_up_proj", "down_proj")
    ]
)


def _ckpt_keys(modules):
    """Collect ckpt tensor names, descending into composite weight modules."""
    keys = set()
    for module in modules:
        sub_weights = getattr(module, "sub_weights", None)
        if sub_weights:
            keys |= _ckpt_keys(sub_weights.values())
        else:
            keys |= {info.name for info in module.weights}
    return keys


class Qwen4ExpTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        Path(self._temp_dir.name, "config.json").write_text(json.dumps(self._config()))
        self.config = Qwen4Exp.create_config(self._temp_dir.name)

    def _loaded_model(self, **changes):
        model = Qwen4Exp.__new__(Qwen4Exp)
        config = dict(
            hidden_size=2560,
            hc_mult=4,
            qwen4_ple_prefetch_mb=256,
            _qwen4_ple_layer_ids=(1,),
            _qwen4_indexer_head_dim=128,
            _qwen4_indexer_budget=1024,
            _qwen4_indexer_kv_heads=1,
            enable_qwen4_qsa=True,
            enable_qwen4_ple=True,
            is_mtp=False,
            data_type=DataType.TYPE_BF16,
            quant_config=None,
            gen_num_per_cycle=0,
            _qwen4_indexer_compress_ratio=4,
            _qwen4_split_ngram_parts=128,
            attn_config=SimpleNamespace(
                is_sparse=True,
                use_sparse_gqa_fmha=True,
                kv_cache_dtype=KvCacheDataType.BASE,
            ),
        )
        config.update(changes)
        model.model_config = SimpleNamespace(**config)
        model.parallelism_config = SimpleNamespace(
            role_type=RoleType.PDFUSION,
            get_attn_tp_size=lambda: 1,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False, is_prefill_enabled=lambda: False
            ),
        )
        return model

    def test_checkpoint_capabilities_are_required_and_read_only(self):
        self.assertTrue(self.config.enable_qwen4_ple)
        self.assertTrue(self.config.enable_qwen4_qsa)
        self.assertTrue(self.config.attn_config.is_sparse)
        self.assertTrue(self.config.attn_config.use_sparse_gqa_fmha)
        for field in ("enable_qwen4_ple", "enable_qwen4_qsa", "qwen4_ple_cpu_offload"):
            with self.assertRaises(AttributeError):
                setattr(self.config, field, False)
        self.assertTrue(self.config.getAttentionConfigs(8).use_sparse_gqa_fmha)

    def test_incomplete_model_is_rejected_before_loading(self):
        for changes in (dict(enable_qwen4_qsa=False), dict(enable_qwen4_ple=False)):
            model = self._loaded_model(**changes)
            with mock.patch.object(Qwen35Moe, "load") as parent:
                with self.assertRaisesRegex(RuntimeError, "complete checkpoint"):
                    model.load()
            parent.assert_not_called()

    def test_complete_model_and_pd_roles_need_no_model_permission(self):
        for role in (RoleType.PDFUSION, RoleType.PREFILL, RoleType.DECODE):
            model = self._loaded_model()
            model.parallelism_config.role_type = role
            with mock.patch.object(Qwen35Moe, "load", return_value="loaded") as parent:
                self.assertEqual(model.load(skip_python_model=True), "loaded")
            parent.assert_called_once_with(skip_python_model=True)

    def test_full_checkpoint_owns_all_side_regions(self):
        self.assertEqual(len(self.config.kv_cache_spec_descs), self.config.num_layers)
        for idx, regions in enumerate(self.config.kv_cache_spec_descs):
            tags = {r.tag for r in regions}
            self.assertEqual("indexer_kv" in tags, (idx + 1) % 4 == 0)
            self.assertEqual("indexer_state" in tags, (idx + 1) % 4 == 0)
            self.assertEqual("ple_conv_state" in tags, idx == 1)
            self.assertEqual("ple_ngram_ctx" in tags, idx == 1)

    def test_cpu_ple_residency_and_gpu_vocab_are_fixed(self):
        info = self._weight()._get_weight_info()
        for weight in info.weights:
            self.assertIsNone(getattr(weight, "resident_device", None))
        ple = next(
            w for w in info.layer_weights[1] if w.name == W.qwen4_ple_ngram_shards
        )
        self.assertEqual(ple.resident_device, "cpu")
        self.assertEqual(self.config._tp_sharded_extra_weight_bytes, 0)
        self.assertGreater(self.config._replicated_extra_weight_bytes, 0)

    def test_indexer_weights_only_belong_to_full_attention(self):
        info = self._weight()._get_weight_info()
        self.assertFalse(
            [k for k in _ckpt_keys(info.layer_weights[0]) if ".indexer." in k]
        )
        self.assertEqual(
            len([k for k in _ckpt_keys(info.layer_weights[3]) if ".indexer." in k]), 3
        )

    def test_scaled_fp8_kv_is_a_common_format_choice(self):
        for kind in (KvCacheDataType.BASE, KvCacheDataType.FP8):
            model = self._loaded_model()
            model.model_config.attn_config.kv_cache_dtype = kind
            with mock.patch.object(Qwen35Moe, "load", return_value="loaded"):
                self.assertEqual(model.load(), "loaded")

    def test_qsa_bounds_and_context_parallelism_still_fail_before_load(self):
        model = self._loaded_model(gen_num_per_cycle=4)
        with mock.patch.object(Qwen35Moe, "load") as parent:
            with self.assertRaisesRegex(RuntimeError, "gen_num_per_cycle"):
                model.load()
        parent.assert_not_called()
        model = self._loaded_model()
        model.parallelism_config.prefill_cp_config.is_enabled = lambda: True
        with mock.patch.object(Qwen35Moe, "load") as parent:
            with self.assertRaisesRegex(RuntimeError, "parallelism"):
                model.load()
        parent.assert_not_called()

    def test_ple_shard_topology_and_activation_dtype_still_fail_before_load(self):
        for changes, pattern in (
            (dict(data_type=DataType.TYPE_FP16), "BF16"),
            (dict(_qwen4_split_ngram_parts=0), "shard split"),
        ):
            model = self._loaded_model(**changes)
            with mock.patch.object(Qwen35Moe, "load") as parent:
                with self.assertRaisesRegex(RuntimeError, pattern):
                    model.load()
            parent.assert_not_called()

    def test_graph_permission_is_common_and_model_has_capture_capability(self):
        model = self._loaded_model()
        self.assertTrue(model.support_cuda_graph())

    def test_prefetch_budget_is_the_only_model_resource_override(self):
        self.assertEqual(self.config.qwen4_ple_prefetch_mb, 256)
        self.config.apply_override_args('{"qwen4_ple_prefetch_mb":64}')
        self.assertEqual(self.config.qwen4_ple_prefetch_mb, 64)
        for budget in (0, -1, True, 1.5, "256", None):
            with (
                self.subTest(budget=budget),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                self.config.apply_override_args(
                    json.dumps({"qwen4_ple_prefetch_mb": budget})
                )

    def test_bf16_checkpoint_rejects_partial_quantization_recipes(self):
        for choice in ("MODELOPT_FP4", "FP8_PER_BLOCK", "QWEN4_FP8_NVFP4"):
            self.config.quantization = choice
            with (
                self.subTest(choice=choice),
                self.assertRaisesRegex(ValueError, "recipe"),
            ):
                Qwen4Exp._post_build_model_config(self.config)

    def test_official_recipe_survives_source_precision_initialization_and_cache_override(
        self,
    ):
        data = self._config()
        data["quantization_config"] = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
        Path(self._temp_dir.name, "config.json").write_text(json.dumps(data))
        Path(self._temp_dir.name, "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {}})
        )
        with mock.patch(
            "rtp_llm.models.qwen4_exp.moe_fp8.validate_moe_fp8_checkpoint"
        ) as validate:
            cfg = Qwen4Exp.create_config(self._temp_dir.name)
        validate.assert_called_once()
        cfg.quant_config = Fp8BlockWiseQuantConfig(is_quanted=True)
        cfg.quantization = json.dumps(
            {
                "method": "modelopt_fp4",
                "bits": 4,
                "group_size": 16,
                "is_quanted": False,
                "model_recipe": QWEN4_FP8_NVFP4_RECIPE,
                "online_cache_dir": "/tmp/qwen4-test-cache",
            }
        )
        Qwen4Exp._post_build_model_config(cfg)
        self.assertIsInstance(cfg.quant_config, ModelOptFp4Config)
        self.assertEqual(cfg.quant_config.model_recipe, QWEN4_FP8_NVFP4_RECIPE)
        self.assertEqual(cfg.quant_config.source_fp8_block_size, 128)
        self.assertEqual(cfg.quant_config.online_cache_dir, "/tmp/qwen4-test-cache")
        self.assertEqual(cfg.quant_config.group_size(), 16)
        self.assertTrue(cfg._qwen4_full_fp8_nvfp4)

    def test_quantization_preset_carries_the_complete_recipe(self):
        cfg = init_quant_config("QWEN4_FP8_NVFP4")
        self.assertEqual(cfg.model_recipe, QWEN4_FP8_NVFP4_RECIPE)
        self.assertEqual(cfg.activation_amax, 32.0)
        self.assertTrue(cfg.require_contiguous_experts)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_full_recipe_rejects_partial_or_incompatible_fields(self):
        self.config._qwen4_fp8_source = True
        base = dict(
            method="modelopt_fp4",
            bits=4,
            group_size=16,
            is_quanted=False,
            model_recipe=QWEN4_FP8_NVFP4_RECIPE,
            activation_amax=32.0,
            source_fp8_block_size=128,
            require_contiguous_experts=True,
        )
        for changes in (
            dict(group_size=32),
            dict(is_quanted=True),
            dict(source_fp8_block_size=64),
            dict(activation_amax=16),
            dict(require_contiguous_experts=False),
            dict(mixed_attention=True),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config.quantization = json.dumps(dict(base, **changes))
                Qwen4Exp._configure_precision_recipe(self.config)

    def test_qwen4_recipe_rejected_for_other_models(self):
        config = ModelConfig()
        config.model_type = "qwen3_moe"
        config.quantization = "QWEN4_FP8_NVFP4"
        with self.assertRaisesRegex(ValueError, "only valid for qwen4_exp"):
            config.init_precision_config(None, "BF16")

    def test_registration(self):
        self.assertIs(_model_factory["qwen4_exp"], Qwen4Exp)
        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(
                {"architectures": ["Qwen4ExpForConditionalGeneration"]}
            ),
            "qwen4_exp",
        )
        self.assertIs(Qwen4Exp.get_weight_cls(), Qwen4ExpWeight)
        self.assertIs(get_multimodal_mixin_cls("qwen4_exp"), Qwen3_5MoeMixin)

    def test_basic_config(self):
        config = self.config
        self.assertEqual(config.num_layers, _NUM_LAYERS)
        self.assertEqual(config.hidden_size, 2560)
        self.assertEqual(config.vocab_size, 248320)
        self.assertEqual(config.max_seq_len, 262144)
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual(config.attn_config.head_num, 24)
        self.assertEqual(config.attn_config.kv_head_num, 2)
        self.assertEqual(config.attn_config.size_per_head, 256)
        self.assertEqual(config.layernorm_eps, 1e-6)
        self.assertTrue(config.qk_norm)

    def test_rope_config(self):
        rope_config = self.config.attn_config.rope_config
        self.assertEqual(rope_config.style, RopeStyle.Mrope)
        self.assertEqual(rope_config.base, 10_000_000)
        self.assertEqual(self.config.partial_rotary_factor, 0.25)
        # partial rotary: 256 * 0.25 == 64, so mrope_section must sum to 32 pairs.
        self.assertEqual(rope_config.dim, 64)
        self.assertEqual(rope_config.index_factor, 3)
        self.assertEqual(
            [rope_config.mrope_dim1, rope_config.mrope_dim2, rope_config.mrope_dim3],
            [11, 11, 10],
        )
        self.assertTrue(rope_config.mrope_interleaved)
        self.assertEqual(self.config.mm_model_config.mm_position_ids_style, 2)

    def test_linear_attn_norm_activation(self):
        # Upstream falls back to hidden_act when output_gate_type is unset
        # (this trimmed config has neither -> generic silu default).
        self.assertEqual(self.config.linear_attn_norm_activation, "silu")
        config_json = self._config()
        config_json["text_config"]["output_gate_type"] = "sigmoid"
        Path(self._temp_dir.name, "config.json").write_text(json.dumps(config_json))
        config = Qwen4Exp.create_config(self._temp_dir.name)
        self.assertEqual(config.linear_attn_norm_activation, "sigmoid")

    def test_moe_config(self):
        config = self.config
        self.assertEqual(config.expert_num, 512)
        self.assertEqual(config.moe_k, 10)
        self.assertEqual(config.moe_inter_size, 640)
        self.assertEqual(config.inter_size, 640)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.moe_style, 2)
        # No decoder_sparse_step in the checkpoint config: every layer is MoE.
        self.assertEqual(list(config.moe_layer_index), list(range(_NUM_LAYERS)))

    def test_layer_types_drive_hybrid_topology(self):
        hybrid_config = self.config.hybrid_attention_config
        self.assertTrue(hybrid_config.enable_hybrid_attention)
        layer_types = list(hybrid_config.hybrid_attention_types)
        self.assertEqual(len(layer_types), _NUM_LAYERS)
        self.assertEqual(layer_types.count(HybridAttentionType.NONE), 12)
        self.assertEqual(layer_types.count(HybridAttentionType.LINEAR), 36)
        for idx, layer_type in enumerate(layer_types):
            expected = (
                HybridAttentionType.NONE
                if (idx + 1) % 4 == 0
                else HybridAttentionType.LINEAR
            )
            self.assertEqual(layer_type, expected, f"layer {idx}")

    def test_linear_attention_config(self):
        linear_config = self.config.linear_attention_config
        self.assertEqual(linear_config.linear_num_key_heads, 16)
        self.assertEqual(linear_config.linear_num_value_heads, 48)
        self.assertEqual(linear_config.linear_key_head_dim, 128)
        self.assertEqual(linear_config.linear_value_head_dim, 128)
        self.assertEqual(linear_config.linear_conv_kernel_dim, 4)

    def test_multimodal_config(self):
        mm_config = self.config.mm_model_config
        self.assertTrue(mm_config.is_multimodal)
        self.assertEqual(
            [list(pair) for pair in mm_config.mm_sep_tokens], [[248053, 248054]]
        )
        self.assertEqual(
            self.config.mm_related_params.config["ckpt_path"], self._temp_dir.name
        )

    def test_hc_mult_is_plumbed_from_hc_count(self):
        self.assertEqual(self.config.hc_mult, 4)
        self.assertEqual(
            self.config.mtp_input_hidden_size,
            self.config.hc_mult * self.config.hidden_size,
        )
        self.assertEqual(
            self.config.getMtpInputHiddenSize(),
            self.config.hc_mult * self.config.hidden_size,
        )

    def _weight(self):
        weight = Qwen4ExpWeight(
            model_config=self.config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        weight._process_meta({}, _CKPT_KEYS)
        return weight

    def test_prefix_is_detected_from_gated_residual_anchor(self):
        # The Qwen3.5 anchor (layers.0.input_layernorm.weight) does not exist here.
        self.assertEqual(self._weight().prefix, "model.language_model.")
        self.assertTrue(self._weight()._has_stacked_ckpt)

    def test_prefix_detection_rejects_a_checkpoint_without_the_anchor(self):
        weight = Qwen4ExpWeight(
            model_config=self.config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )

        with self.assertRaisesRegex(ValueError, "cannot determine prefix"):
            weight._process_meta({}, ["model.language_model.embed_tokens.weight"])

    def test_gated_residual_weights_cover_every_hc_tensor(self):
        weight = self._weight()

        info = weight._get_weight_info()

        global_keys = _ckpt_keys(info.weights)
        self.assertEqual(
            global_keys,
            {
                "lm_head.weight",
                "model.language_model.embed_tokens.weight",
                "model.language_model.hyper_connection_mixer.hc_norm.weight",
                "model.language_model.hyper_connection_mixer."
                "input_mix_weight_down.weight",
                "model.language_model.hyper_connection_mixer."
                "input_mix_weight_up.weight",
            },
        )

        self.assertEqual(len(info.layer_weights), _NUM_LAYERS)
        for layer_id in (0, 3):
            layer_keys = _ckpt_keys(info.layer_weights[layer_id])
            for module_name in ("attn_hyper_connection", "mlp_hyper_connection"):
                prefix = f"model.language_model.layers.{{i}}.{module_name}."
                for suffix in (
                    "hc_norm.weight",
                    "input_mix_weight_down.weight",
                    "input_mix_weight_up.weight",
                    "block_inject_weight.weight",
                ):
                    self.assertIn(prefix + suffix, layer_keys)

    def test_no_layer_norm_keys_are_requested(self):
        info = self._weight()._get_weight_info()

        requested = _ckpt_keys(info.weights)
        for layer in info.layer_weights:
            requested |= _ckpt_keys(layer)

        # These three exist in Qwen3.5 but not in this checkpoint; requesting any
        # of them would fail the load.
        for absent in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "model.language_model.norm.weight",
        ):
            self.assertFalse(
                [key for key in requested if key.endswith(absent)],
                f"mapping must not request {absent}",
            )

    def test_linear_and_full_attention_layers_use_the_right_branch(self):
        info = self._weight()._get_weight_info()

        linear_keys = _ckpt_keys(info.layer_weights[0])
        full_keys = _ckpt_keys(info.layer_weights[3])

        self.assertIn(
            "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight",
            linear_keys,
        )
        self.assertNotIn(
            "model.language_model.layers.{i}.self_attn.q_proj.weight", linear_keys
        )
        self.assertIn(
            "model.language_model.layers.{i}.self_attn.q_proj.weight", full_keys
        )
        self.assertNotIn(
            "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight", full_keys
        )

    @staticmethod
    def _config():
        """Trimmed Qwen/Qwen3.8-Flash-Next config.json."""
        return {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "image_token_id": 248056,
            "model_type": "qwen4_exp",
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "tie_word_embeddings": False,
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "num_hidden_layers": _NUM_LAYERS,
                "hidden_size": 2560,
                "vocab_size": 248320,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
                "full_attention_interval": 4,
                "layer_types": [
                    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                    for i in range(_NUM_LAYERS)
                ],
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "hc_count": 4,
                "hc_lowrank": 320,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "indexer_head_dim": 128,
                "indexer_kv_heads": 1,
                "indexer_n_heads": 4,
                "ngram_size": 3,
                "ngram_vocab_size_base": 20000000,
                "ple_conv_kernel_size": 4,
                "ple_embed_dim": 2560,
                "ple_layer_ids": [2],
                "split_ngram_parts": 128,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {
                    "mrope_interleaved": True,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default",
                },
            },
            "vision_config": {
                "deepstack_visual_indexes": [],
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2560,
                "patch_size": 16,
            },
        }


if __name__ == "__main__":
    unittest.main()
