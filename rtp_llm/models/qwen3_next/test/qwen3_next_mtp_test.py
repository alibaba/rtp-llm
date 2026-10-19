import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rtp_llm.model_factory_register as model_registry
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig, resolve_ssm_state_dtype
from rtp_llm.model_factory_register import ModelDict, _model_factory
from rtp_llm.model_loader.ffn_weight import FfnWeight, MoeWeight
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Dense, Qwen35Moe
from rtp_llm.models.qwen3_next.qwen3_next_mtp import (
    Qwen35DenseMTP,
    Qwen35DenseMTPWeight,
)
from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeMixin,
)
from rtp_llm.ops import (
    DataType,
    HWKernelConfig,
    HybridAttentionType,
    ParallelismConfig,
    RopeStyle,
)


class Qwen35DenseMTPTest(unittest.TestCase):
    def test_text_only_dense_export_matches_composite_backbone_config(self):
        for tied_embeddings in (False, True):
            with self.subTest(tied_embeddings=tied_embeddings):
                composite_json = self._config()
                composite_json["text_config"]["tie_word_embeddings"] = tied_embeddings
                text_json = dict(composite_json["text_config"])
                text_json.update(
                    model_type="qwen3_5_text",
                    architectures=["Qwen3_5ForCausalLM"],
                )
                # A saved HF text CausalLM has no composite or vision fields.
                self.assertNotIn("text_config", text_json)
                self.assertNotIn("vision_start_token_id", text_json)
                with tempfile.TemporaryDirectory() as directory:
                    text_path = Path(directory, "text")
                    composite_path = Path(directory, "composite")
                    text_path.mkdir()
                    composite_path.mkdir()
                    (text_path / "config.json").write_text(json.dumps(text_json))
                    (composite_path / "config.json").write_text(
                        json.dumps(composite_json)
                    )
                    text = Qwen35Dense.create_config(str(text_path))
                    composite = Qwen35Dense.create_config(str(composite_path))

                self.assertIsInstance(text, ModelConfig)
                for field, expected in (
                    ("num_layers", 64),
                    ("hidden_size", 5120),
                    ("vocab_size", 248320),
                    ("max_seq_len", 262144),
                    ("inter_size", 17408),
                    ("moe_style", 0),
                    ("tie_word_embeddings", tied_embeddings),
                ):
                    self.assertEqual(getattr(text, field), expected, field)
                    self.assertEqual(
                        getattr(text, field), getattr(composite, field), field
                    )
                for field, expected in (
                    ("head_num", 24),
                    ("kv_head_num", 4),
                    ("size_per_head", 256),
                ):
                    self.assertEqual(getattr(text.attn_config, field), expected, field)
                    self.assertEqual(
                        getattr(text.attn_config, field),
                        getattr(composite.attn_config, field),
                        field,
                    )
                for field in (
                    "linear_conv_kernel_dim",
                    "linear_key_head_dim",
                    "linear_num_key_heads",
                    "linear_num_value_heads",
                    "linear_value_head_dim",
                    "supports_history_state_reclaim",
                ):
                    self.assertEqual(
                        getattr(text.linear_attention_config, field),
                        getattr(composite.linear_attention_config, field),
                        field,
                    )
                expected_types = (
                    [HybridAttentionType.LINEAR] * 3 + [HybridAttentionType.NONE]
                ) * 16
                self.assertTrue(text.hybrid_attention_config.enable_hybrid_attention)
                self.assertEqual(
                    list(text.hybrid_attention_config.hybrid_attention_types),
                    expected_types,
                )
                self.assertEqual(
                    list(composite.hybrid_attention_config.hybrid_attention_types),
                    expected_types,
                )
                self.assertEqual(len(text.kv_cache_spec_descs), 64)
                self.assertEqual(
                    [
                        [(spec.tag, spec.cache_type) for spec in layer]
                        for layer in text.kv_cache_spec_descs
                    ],
                    [
                        [(spec.tag, spec.cache_type) for spec in layer]
                        for layer in composite.kv_cache_spec_descs
                    ],
                )
                self.assertFalse(text.mm_model_config.is_multimodal)
                self.assertTrue(composite.mm_model_config.is_multimodal)
                self.assertEqual(
                    [list(pair) for pair in composite.mm_model_config.mm_sep_tokens],
                    [[248053, 248054]],
                )

    def test_text_only_dense_architecture_routes_without_eager_model_import(self):
        # Recreate the cold architecture map; the model module imported by this
        # test file must not hide a missing built-in lazy registration.
        with (
            mock.patch.dict(model_registry._hf_architecture_2_ft, {}, clear=True),
            mock.patch.object(model_registry, "_load_internal_legacy_models"),
        ):
            model_registry._register_builtin_lazy_models()
            for architecture in (
                "Qwen3_5ForCausalLM",
                "Qwen3_5ForConditionalGeneration",
            ):
                with self.subTest(architecture=architecture):
                    self.assertEqual(
                        ModelDict.get_ft_model_type_by_config(
                            {"architectures": [architecture]}
                        ),
                        "qwen35_dense",
                    )
        self.assertIs(_model_factory["qwen35_dense"], Qwen35Dense)

    def test_history_state_reclaim_is_a_qwen35_target_capability(self):
        for model, expected in (
            (Qwen3Next, False),
            (Qwen35Moe, True),
            (Qwen35Dense, True),
        ):
            with self.subTest(model=model.__name__):
                config = ModelConfig()
                model._parse_linear_attention_config(
                    self._config()["text_config"], config
                )
                self.assertEqual(
                    config.linear_attention_config.supports_history_state_reclaim,
                    expected,
                )

    def test_auto_ssm_state_dtype_preserves_model_value_without_remote_cache(self):
        self.assertEqual(
            resolve_ssm_state_dtype("auto", DataType.TYPE_FP32, False),
            DataType.TYPE_FP32,
        )

    def test_auto_ssm_state_dtype_resolves_to_bf16_for_remote_cache(self):
        self.assertEqual(
            resolve_ssm_state_dtype("auto", DataType.TYPE_FP32, True),
            DataType.TYPE_BF16,
        )

    def test_explicit_ssm_state_dtype_remains_authoritative_for_remote_cache(self):
        self.assertEqual(
            resolve_ssm_state_dtype("fp32", DataType.TYPE_BF16, True),
            DataType.TYPE_FP32,
        )

    def test_target_uses_model_ssm_state_dtype_with_independent_hybrid_pools(self):
        config_json = self._config()
        config_json["text_config"]["mamba_ssm_dtype"] = "float32"
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text(json.dumps(config_json))
            config = Qwen35Dense.create_config(temp_dir)

        # The model keeps MHA pages and recurrent GDN states in their native
        # per-group physical layouts.
        self.assertTrue(config.hybrid_attention_config.enable_hybrid_attention)
        self.assertEqual(
            list(config.hybrid_attention_config.hybrid_attention_types),
            ([HybridAttentionType.LINEAR] * 3 + [HybridAttentionType.NONE]) * 16,
        )
        self.assertEqual(len(config.kv_cache_spec_descs), config.num_layers)
        self.assertEqual(
            config.linear_attention_config.ssm_state_dtype, DataType.TYPE_FP32
        )

    def test_remote_cache_preserves_hybrid_layout_for_bf16_ssm_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text(json.dumps(self._config()))
            config = Qwen35Dense.create_config(temp_dir)

        config.linear_attention_config.ssm_state_dtype = DataType.TYPE_BF16
        kv_cache_config = KVCacheConfig()
        kv_cache_config.reuse_cache = True
        kv_cache_config.enable_remote_cache = True

        expected_types = list(config.hybrid_attention_config.hybrid_attention_types)
        expected_specs = [
            [(spec.tag, spec.cache_type) for spec in layer]
            for layer in config.kv_cache_spec_descs
        ]
        Qwen35Dense._apply_kv_cache_config(config, kv_cache_config)

        self.assertEqual(
            list(config.hybrid_attention_config.hybrid_attention_types), expected_types
        )
        self.assertEqual(
            [
                [(spec.tag, spec.cache_type) for spec in layer]
                for layer in config.kv_cache_spec_descs
            ],
            expected_specs,
        )

    def test_remote_cache_rejects_explicit_fp32_ssm_state(self):
        config_json = self._config()
        config_json["text_config"]["mamba_ssm_dtype"] = "float32"
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text(json.dumps(config_json))
            config = Qwen35Dense.create_config(temp_dir)

        kv_cache_config = KVCacheConfig()
        kv_cache_config.reuse_cache = True
        kv_cache_config.enable_remote_cache = True

        with self.assertRaisesRegex(
            ValueError, "remote cache requires BF16 SSM state storage"
        ):
            Qwen35Dense._apply_kv_cache_config(config, kv_cache_config)

    def test_dense_mtp_config_and_registration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text(json.dumps(self._config()))

            config = Qwen35DenseMTP.create_config(temp_dir)

        self.assertIs(_model_factory["qwen35_dense_mtp"], Qwen35DenseMTP)
        self.assertEqual(config.num_layers, 1)
        self.assertTrue(config.is_mtp)
        self.assertFalse(config.linear_attention_config.supports_history_state_reclaim)
        self.assertEqual(config.moe_style, 0)
        self.assertEqual(list(config.moe_layer_index), [])
        self.assertEqual(
            list(config.hybrid_attention_config.hybrid_attention_types),
            [HybridAttentionType.NONE],
        )
        self.assertEqual(config.attn_config.rope_config.style, RopeStyle.Base)
        self.assertEqual(len(config.kv_cache_spec_descs), 1)

    def test_dense_mtp_uses_dense_ffn_checkpoint_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text(json.dumps(self._config()))
            config = Qwen35DenseMTP.create_config(temp_dir)
            weight = Qwen35DenseMTPWeight(
                model_config=config,
                parallelism_config=ParallelismConfig(),
                hw_kernel_config=HWKernelConfig(),
                kv_cache_config=KVCacheConfig(),
            )

        modules = weight._create_ffn_weight()

        self.assertEqual(weight.prefix, "mtp.")
        self.assertEqual(weight.model_prefix, "model.language_model.")
        self.assertEqual(len(modules), 1)
        self.assertIsInstance(modules[0], FfnWeight)
        self.assertNotIsInstance(modules[0], MoeWeight)
        ffn = modules[0]
        checkpoint_keys = {
            info.name
            for atomic_weight in (ffn.origin_w1, ffn.origin_w3, ffn.w2)
            for info in atomic_weight.weights
        }
        self.assertEqual(
            checkpoint_keys,
            {
                "mtp.layers.{i}.mlp.gate_proj.weight",
                "mtp.layers.{i}.mlp.up_proj.weight",
                "mtp.layers.{i}.mlp.down_proj.weight",
            },
        )

    def test_dense_mtp_uses_qwen35_multimodal_mixin(self):
        self.assertIs(get_multimodal_mixin_cls("qwen35_dense_mtp"), Qwen3_5MoeMixin)

    @staticmethod
    def _config():
        return {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "text_config": {
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "vocab_size": 248320,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "intermediate_size": 17408,
                "full_attention_interval": 4,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "rope_parameters": {
                    "mrope_interleaved": True,
                    "mrope_section": [11, 11, 10],
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25,
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
