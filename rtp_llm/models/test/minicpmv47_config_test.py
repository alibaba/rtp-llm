import json
import os
import tempfile
import unittest

from rtp_llm.models.minicpmv4_7 import MiniCPMV4_7, MiniCPMV4_7Weight
from rtp_llm.ops import HybridAttentionType

# Trimmed mirrors of the real openbmb/MiniCPM-V-4.7 config.json files
# (fetched 2026-10-13; see runs/minicpmv47_h20_20261013/adaptation-plan.md).
_TEXT_CONFIG_MOE = {
    "attention_bias": False,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_size": 2048,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ] * 2,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "moe_intermediate_size": 512,
    "num_attention_heads": 16,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 8,
    "num_key_value_heads": 2,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
    "shared_expert_intermediate_size": 512,
    "vocab_size": 248144,
    "model_type": "qwen3_5_moe_text",
}

_TEXT_CONFIG_DENSE = {
    "attention_bias": False,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "num_attention_heads": 8,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
    },
    "vocab_size": 248094,
    "model_type": "qwen3_5_text",
}

_VISION_CONFIG = {
    "hidden_size": 1152,
    "image_size": 980,
    "intermediate_size": 4304,
    "layer_norm_eps": 1e-06,
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 27,
    "patch_size": 14,
    "window_kernel_size": [2, 2],
}


def _write_ckpt(text_config: dict) -> str:
    tmp = tempfile.mkdtemp()
    config = {
        "architectures": ["MiniCPMV4_7ForConditionalGeneration"],
        "model_type": "minicpmv4_7",
        "dtype": "bfloat16",
        "insert_layer_id": 6,
        "downsample_mode": "16x",
        "mrope_mode": "canvas",
        "merge_kernel_size": [2, 2],
        "merger_times": 1,
        "image_token_id": 248056,
        "video_token_id": 248057,
        "image_start_id": 248078,
        "image_end_id": 248079,
        "slice_start_id": 248088,
        "slice_end_id": 248089,
        "text_config": text_config,
        "vision_config": _VISION_CONFIG,
    }
    with open(os.path.join(tmp, "config.json"), "w") as writer:
        json.dump(config, writer)
    return tmp


class MiniCPMV47ConfigTest(unittest.TestCase):
    def test_moe_config_parses_hybrid_moe_and_mm_contract(self):
        ckpt = _write_ckpt(_TEXT_CONFIG_MOE)
        config = MiniCPMV4_7._create_config(ckpt)

        self.assertEqual(config.num_layers, 8)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.vocab_size, 248144)
        self.assertEqual(config.max_seq_len, 262144)
        self.assertTrue(config.hybrid_attention_config.enable_hybrid_attention)
        self.assertEqual(
            config.hybrid_attention_config.hybrid_attention_types,
            [HybridAttentionType.LINEAR] * 3 + [HybridAttentionType.NONE]
            + [HybridAttentionType.LINEAR] * 3 + [HybridAttentionType.NONE],
        )
        self.assertEqual(config.expert_num, 256)
        self.assertEqual(config.moe_k, 8)
        self.assertEqual(config.moe_inter_size, 512)
        self.assertEqual(config.inter_size, 512)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.moe_style, 2)
        self.assertTrue(config.has_moe_norm)
        self.assertEqual(config.moe_layer_index, list(range(8)))

        self.assertEqual(config.attn_config.head_num, 16)
        self.assertEqual(config.attn_config.kv_head_num, 2)
        self.assertEqual(config.attn_config.size_per_head, 256)
        self.assertEqual(int(config.attn_config.rope_config.style), 7)
        self.assertEqual(config.attn_config.rope_config.dim, 64)
        self.assertEqual(config.partial_rotary_factor, 0.25)
        self.assertEqual(config.attn_config.rope_config.mrope_dim1, 11)
        self.assertEqual(config.attn_config.rope_config.mrope_dim3, 10)
        self.assertTrue(config.attn_config.rope_config.mrope_interleaved)
        self.assertTrue(config.qk_norm)
        self.assertEqual(config.layernorm_eps, 1e-06)

        linear = config.linear_attention_config
        self.assertEqual(linear.linear_conv_kernel_dim, 4)
        self.assertEqual(linear.linear_key_head_dim, 128)
        self.assertEqual(linear.linear_num_key_heads, 16)
        self.assertEqual(linear.linear_num_value_heads, 32)
        self.assertEqual(linear.linear_value_head_dim, 128)

        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[248078, 248079]])
        self.assertFalse(config.mm_model_config.include_sep_tokens)
        self.assertEqual(config.mm_model_config.mm_position_ids_style, 2)
        self.assertEqual(
            config.mm_related_params.special_tokens.get("default_mm_token"),
            "<|image_pad|>",
        )
        self.assertEqual(config.mm_related_params.config["ckpt_path"], ckpt)

    def test_dense_config_parses_dense_ffn(self):
        ckpt = _write_ckpt(_TEXT_CONFIG_DENSE)
        config = MiniCPMV4_7._create_config(ckpt)

        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.inter_size, 3584)
        self.assertEqual(config.expert_num, 0)
        self.assertEqual(config.attn_config.head_num, 8)
        self.assertEqual(
            config.hybrid_attention_config.hybrid_attention_types,
            [HybridAttentionType.LINEAR] * 3 + [HybridAttentionType.NONE],
        )
        self.assertTrue(config.mm_model_config.is_multimodal)

    def test_hybrid_kv_cache_spec_descs_cover_both_layer_types(self):
        ckpt = _write_ckpt(_TEXT_CONFIG_DENSE)
        config = MiniCPMV4_7._create_config(ckpt)
        MiniCPMV4_7._post_build_model_config(config)

        self.assertEqual(len(config.kv_cache_spec_descs), 4)
        for layer_idx, descs in enumerate(config.kv_cache_spec_descs):
            self.assertEqual(len(descs), 1)

    def test_layer_types_length_mismatch_is_rejected(self):
        bad = dict(_TEXT_CONFIG_DENSE)
        bad["layer_types"] = ["linear_attention"] * 3
        ckpt = _write_ckpt(bad)
        with self.assertRaises(ValueError):
            MiniCPMV4_7._create_config(ckpt)

    def test_weight_prefix_detection_from_language_model_keys(self):
        weight = MiniCPMV4_7Weight.__new__(MiniCPMV4_7Weight)
        weight._has_stacked_ckpt = False
        keys = [
            "model.language_model.layers.0.input_layernorm.weight",
            "model.vision_tower.embeddings.patch_embedding.weight",
            "lm_head.weight",
        ]
        weight._process_meta({}, keys)
        self.assertEqual(weight.prefix, "model.language_model.")

    def test_weight_prefix_detection_rejects_missing_text_key(self):
        weight = MiniCPMV4_7Weight.__new__(MiniCPMV4_7Weight)
        with self.assertRaises(ValueError):
            weight._process_meta({}, ["model.vision_tower.embeddings.patch_embedding.weight"])


if __name__ == "__main__":
    unittest.main()
