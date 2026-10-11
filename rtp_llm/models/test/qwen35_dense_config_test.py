import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Dense, Qwen35Moe
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen35DenseWeight
from rtp_llm.ops import (
    DataType,
    HWKernelConfig,
    HybridAttentionType,
    ParallelismConfig,
    RopeStyle,
)
from rtp_llm.utils.model_weight import W


def _tiny_text_config():
    # 2 linear + 1 full layer (full_attention_interval=3); dims mirror autojev-27b.
    return {
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "num_hidden_layers": 3,
        "hidden_size": 5120,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "intermediate_size": 17408,
        "full_attention_interval": 3,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "tie_word_embeddings": False,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    }


def _nested_dense_config():
    return {
        "architectures": ["Qwen3_5Model"],
        "model_type": "qwen3_5",
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "image_token_id": 248056,
        "tie_word_embeddings": False,
        "text_config": _tiny_text_config(),
        "vision_config": {
            "model_type": "qwen3_5_vision",
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 5120,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 2304,
        },
    }


def _nested_moe_config():
    config = _nested_dense_config()
    config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    text_config = config["text_config"]
    text_config.pop("intermediate_size")
    text_config.update(
        {
            "num_experts_per_tok": 2,
            "num_experts": 8,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 128,
            "decoder_sparse_step": 1,
            "norm_topk_prob": True,
        }
    )
    return config


def _write_config(config_json):
    temp_dir = tempfile.TemporaryDirectory()
    Path(temp_dir.name, "config.json").write_text(json.dumps(config_json))
    return temp_dir


# Mirrors the decision checkpoints (autojev/kev) served through the downstream
# decision module; their config carries this block and no lm_head tensor.
_DECISION_BLOCK = {
    "format_version": 1,
    "head_type": "linear",
    "prompt_format": "chat",
    "head_weight": "decision_head.weight",
    "num_labels": 255,
}


def _assert_common_text_fields(test, config):
    test.assertEqual(config.attn_config.head_num, 24)
    test.assertEqual(config.attn_config.kv_head_num, 4)
    test.assertEqual(config.attn_config.size_per_head, 256)
    test.assertEqual(config.num_layers, 3)
    test.assertEqual(config.hidden_size, 5120)
    test.assertEqual(config.vocab_size, 248320)
    test.assertFalse(config.tie_word_embeddings)

    # hybrid pattern: every 3rd layer is full attention
    test.assertTrue(config.hybrid_attention_config.enable_hybrid_attention)
    test.assertEqual(
        list(config.hybrid_attention_config.hybrid_attention_types),
        [
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
        ],
    )
    test.assertEqual(len(config.kv_cache_spec_descs), config.num_layers)

    # linear attention params
    linear_config = config.linear_attention_config
    test.assertEqual(linear_config.linear_conv_kernel_dim, 4)
    test.assertEqual(linear_config.linear_key_head_dim, 128)
    test.assertEqual(linear_config.linear_num_key_heads, 16)
    test.assertEqual(linear_config.linear_num_value_heads, 48)
    test.assertEqual(linear_config.linear_value_head_dim, 128)
    test.assertEqual(linear_config.ssm_state_dtype, DataType.TYPE_FP32)

    # mrope fields
    rope_config = config.attn_config.rope_config
    test.assertEqual(rope_config.style, RopeStyle.Mrope)
    test.assertEqual(rope_config.base, 10000000)
    test.assertEqual(config.partial_rotary_factor, 0.25)
    test.assertEqual(rope_config.dim, int(256 * 0.25))
    test.assertEqual(
        (rope_config.mrope_dim1, rope_config.mrope_dim2, rope_config.mrope_dim3),
        (11, 11, 10),
    )
    test.assertTrue(rope_config.mrope_interleaved)
    test.assertEqual(rope_config.index_factor, 3)
    test.assertEqual(config.mm_model_config.mm_position_ids_style, 2)


class Qwen35DenseConfigTest(unittest.TestCase):
    def test_nested_vlm_config_parses_text_and_mm(self):
        with _write_config(_nested_dense_config()) as temp_dir:
            config = Qwen35Dense.create_config(temp_dir)

        _assert_common_text_fields(self, config)
        self.assertEqual(config.inter_size, 17408)
        self.assertTrue(config.is_multimodal)
        self.assertEqual(
            [list(pair) for pair in config.mm_model_config.mm_sep_tokens],
            [[248053, 248054]],
        )
        self.assertEqual(
            config.mm_related_params.config["ckpt_path"], config.ckpt_path
        )

    def test_flat_text_only_config_parses_without_mm(self):
        with _write_config(_tiny_text_config()) as temp_dir:
            config = Qwen35Dense.create_config(temp_dir)

        _assert_common_text_fields(self, config)
        self.assertFalse(config.mm_model_config.is_multimodal)
        self.assertEqual(
            [list(pair) for pair in config.mm_model_config.mm_sep_tokens], []
        )

    def test_moe_config_parses_experts(self):
        with _write_config(_nested_moe_config()) as temp_dir:
            config = Qwen35Moe.create_config(temp_dir)

        _assert_common_text_fields(self, config)
        self.assertEqual(config.moe_k, 2)
        self.assertEqual(config.expert_num, 8)
        self.assertEqual(config.moe_inter_size, 64)
        self.assertEqual(config.inter_size, 128)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.moe_style, 2)
        self.assertTrue(config.has_moe_norm)
        self.assertEqual(list(config.moe_layer_index), [0, 1, 2])
        self.assertTrue(config.is_multimodal)


class Qwen35DenseWeightTest(unittest.TestCase):
    def _build_weight(self, config):
        return Qwen35DenseWeight(
            model_config=config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )

    def _dense_config(self):
        with _write_config(_nested_dense_config()) as temp_dir:
            return Qwen35Dense.create_config(temp_dir)

    def _dense_config_with_dir(self, extra=None):
        config_json = _nested_dense_config()
        if extra:
            config_json.update(extra)
        return _write_config(config_json)

    def test_autojev_keys_detect_bare_prefix_and_missing_lm_head(self):
        with self._dense_config_with_dir(
            {"decision": _DECISION_BLOCK}
        ) as temp_dir:
            weight = self._build_weight(Qwen35Dense.create_config(temp_dir))
            weight_keys = {
                "language_model.embed_tokens.weight",
                "language_model.norm.weight",
                "language_model.layers.0.input_layernorm.weight",
            }
            weight._process_meta([{}], weight_keys)

            self.assertEqual(weight.prefix, "language_model.")
            self.assertFalse(weight._has_lm_head)
            lm_head = weight._create_lm_head_weight()
            self.assertEqual(lm_head.name, W.lm_head)
            self.assertEqual(
                [info.name for info in lm_head.weights],
                ["language_model.embed_tokens.weight"],
            )

    def test_missing_lm_head_without_decision_head_is_rejected(self):
        with self._dense_config_with_dir() as temp_dir:
            weight = self._build_weight(Qwen35Dense.create_config(temp_dir))
            weight._process_meta(
                [{}],
                {
                    "language_model.embed_tokens.weight",
                    "language_model.layers.0.input_layernorm.weight",
                },
            )

            with self.assertRaisesRegex(ValueError, "no lm_head.weight"):
                weight._create_lm_head_weight()

    def test_tied_embeddings_missing_lm_head_reuse_embedding(self):
        config_json = _nested_dense_config()
        config_json["text_config"]["tie_word_embeddings"] = True
        with _write_config(config_json) as temp_dir:
            weight = self._build_weight(Qwen35Dense.create_config(temp_dir))
            weight._process_meta(
                [{}],
                {
                    "language_model.embed_tokens.weight",
                    "language_model.layers.0.input_layernorm.weight",
                },
            )

            lm_head = weight._create_lm_head_weight()
            self.assertEqual(
                [info.name for info in lm_head.weights],
                ["language_model.embed_tokens.weight"],
            )

    def test_official_keys_detect_wrapped_prefix_and_root_lm_head(self):
        weight = self._build_weight(self._dense_config())
        weight_keys = {
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "lm_head.weight",
        }
        weight._process_meta([{}], weight_keys)

        self.assertEqual(weight.prefix, "model.language_model.")
        self.assertTrue(weight._has_lm_head)
        lm_head = weight._create_lm_head_weight()
        self.assertEqual([info.name for info in lm_head.weights], ["lm_head.weight"])

    def test_nested_lm_head_is_used_and_mtp_keys_are_ignored(self):
        with self._dense_config_with_dir(
            {"decision": _DECISION_BLOCK}
        ) as temp_dir:
            weight = self._build_weight(Qwen35Dense.create_config(temp_dir))
            weight._process_meta(
                [{}],
                {
                    "model.layers.0.input_layernorm.weight",
                    "model.lm_head.weight",
                    "mtp.lm_head.weight",
                },
            )
            self.assertEqual(weight.prefix, "model.")
            self.assertTrue(weight._has_lm_head)
            self.assertEqual(
                [info.name for info in weight._create_lm_head_weight().weights],
                ["model.lm_head.weight"],
            )

            # an MTP draft lm_head alone must not count as the target lm_head
            weight = self._build_weight(Qwen35Dense.create_config(temp_dir))
            weight._process_meta(
                [{}],
                {
                    "language_model.embed_tokens.weight",
                    "language_model.layers.0.input_layernorm.weight",
                    "mtp.lm_head.weight",
                },
            )
            self.assertFalse(weight._has_lm_head)
            self.assertEqual(
                [info.name for info in weight._create_lm_head_weight().weights],
                ["language_model.embed_tokens.weight"],
            )

    def test_dense_ffn_uses_detected_prefix(self):
        weight = self._build_weight(self._dense_config())
        weight._process_meta(
            [{}],
            {
                "language_model.layers.0.input_layernorm.weight",
                "language_model.embed_tokens.weight",
            },
        )
        modules = weight._create_ffn_weight()
        self.assertEqual(len(modules), 1)
        ckpt_names = {
            info.name
            for component in modules[0].get_components()
            for info in component.weights
        }
        self.assertEqual(
            ckpt_names,
            {
                "language_model.layers.{i}.mlp.gate_proj.weight",
                "language_model.layers.{i}.mlp.up_proj.weight",
                "language_model.layers.{i}.mlp.down_proj.weight",
            },
        )

    def test_full_weight_info_matches_autojev_key_layout(self):
        temp_dir = self._dense_config_with_dir({"decision": _DECISION_BLOCK})
        self.addCleanup(temp_dir.cleanup)
        weight = self._build_weight(Qwen35Dense.create_config(temp_dir.name))
        prefix = "language_model."
        expected = {prefix + "embed_tokens.weight", prefix + "norm.weight"}
        for i in range(3):
            layer = f"{prefix}layers.{i}."
            expected.add(layer + "input_layernorm.weight")
            expected.add(layer + "post_attention_layernorm.weight")
            if i == 2:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    expected.add(layer + f"self_attn.{name}.weight")
                expected.add(layer + "self_attn.q_norm.weight")
                expected.add(layer + "self_attn.k_norm.weight")
            else:
                for name in (
                    "in_proj_qkv.weight",
                    "in_proj_z.weight",
                    "in_proj_b.weight",
                    "in_proj_a.weight",
                    "norm.weight",
                    "dt_bias",
                    "conv1d.weight",
                    "A_log",
                    "out_proj.weight",
                ):
                    expected.add(layer + f"linear_attn.{name}")
            for name in ("gate_proj", "up_proj", "down_proj"):
                expected.add(layer + f"mlp.{name}.weight")
        # NOTE: no lm_head.weight, mirroring the autojev decision checkpoint.
        weight._process_meta([{}], set(expected))

        weight_info = weight._get_weight_info()
        declared = set()
        for module in weight_info.weights:
            for component in module.get_components():
                declared.update(info.name for info in component.weights)
        for layer_modules in weight_info.layer_weights:
            for module in layer_modules:
                for component in module.get_components():
                    declared.update(info.name for info in component.weights)

        # CkptWeightInfo names are per-layer templates ("layers.{i}."): rebuild
        # the expectation in template form (union of linear/full layer branches).
        layer = "language_model.layers.{i}."
        expected_templates = {
            "language_model.embed_tokens.weight",
            "language_model.norm.weight",
            layer + "input_layernorm.weight",
            layer + "post_attention_layernorm.weight",
        }
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            expected_templates.add(layer + f"self_attn.{name}.weight")
        expected_templates.add(layer + "self_attn.q_norm.weight")
        expected_templates.add(layer + "self_attn.k_norm.weight")
        for name in (
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_b.weight",
            "in_proj_a.weight",
            "norm.weight",
            "dt_bias",
            "conv1d.weight",
            "A_log",
            "out_proj.weight",
        ):
            expected_templates.add(layer + f"linear_attn.{name}")
        for name in ("gate_proj", "up_proj", "down_proj"):
            expected_templates.add(layer + f"mlp.{name}.weight")

        # lm_head falls back to the embedding tensor; q_proj feeds both the
        # qkv and the output-gate descriptors. Everything else is 1:1.
        self.assertEqual(declared, expected_templates)
        self.assertEqual(len(weight_info.layer_weights), 3)


if __name__ == "__main__":
    unittest.main()
