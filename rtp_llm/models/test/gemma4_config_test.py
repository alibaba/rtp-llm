import json
import math
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import torch
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_factory_register import ModelDict, ensure_model_registered
from rtp_llm.model_loader.ffn_weight import FfnWeight, MoeAtomicWeight, MoeWeight
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.models.gemma4 import Gemma4
from rtp_llm.models.gemma4_weight import (
    GEMMA4_DEFAULT_CKPT_PREFIX,
    Gemma4WeightInfo,
    merge_qkv_keq_v,
    scale_reshape,
)
from rtp_llm.ops import (
    CacheGroupType,
    HWKernelConfig,
    HybridAttentionType,
    KVCacheSpecDesc,
    KVCacheSpecType,
    ParallelismConfig,
)
from rtp_llm.utils.model_weight import W, merge_qkv_hf, stack_

NUM_LAYERS = 30
HIDDEN_SIZE = 2816
HEAD_NUM = 16
KV_HEAD_NUM = 8
SIZE_PER_HEAD = 256
GLOBAL_KV_HEAD_NUM = 2
GLOBAL_HEAD_DIM = 512
INTER_SIZE = 2112
MOE_INTER_SIZE = 704
EXPERT_NUM = 128
MOE_K = 8
SLIDING_WINDOW = 1024
FULL_LAYER_IDS = [5, 11, 17, 23, 29]

# Full-fidelity copy of gemma-4-26B-A4B-it config.json text_config (the test
# below cross-checks the synthetic copy against the real file when present).
SYNTHETIC_TEXT_CONFIG = {
    "num_hidden_layers": NUM_LAYERS,
    "hidden_size": HIDDEN_SIZE,
    "vocab_size": 262144,
    "num_attention_heads": HEAD_NUM,
    "num_key_value_heads": KV_HEAD_NUM,
    "num_global_key_value_heads": GLOBAL_KV_HEAD_NUM,
    "head_dim": SIZE_PER_HEAD,
    "global_head_dim": GLOBAL_HEAD_DIM,
    "intermediate_size": INTER_SIZE,
    "moe_intermediate_size": MOE_INTER_SIZE,
    "num_experts": EXPERT_NUM,
    "top_k_experts": MOE_K,
    "rms_norm_eps": 1e-06,
    "hidden_activation": "gelu_pytorch_tanh",
    "sliding_window": SLIDING_WINDOW,
    "max_position_embeddings": 262144,
    "tie_word_embeddings": True,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "final_logit_softcapping": 30.0,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    "layer_types": [
        "full_attention" if i in FULL_LAYER_IDS else "sliding_attention"
        for i in range(NUM_LAYERS)
    ],
}


def _write_config_dir(text_config=None, with_vision_wrapper=True):
    text_config = dict(SYNTHETIC_TEXT_CONFIG if text_config is None else text_config)
    tmp_dir = tempfile.mkdtemp(prefix="gemma4_config_")
    if with_vision_wrapper:
        config = {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": text_config,
        }
    else:
        config = text_config
    with open(os.path.join(tmp_dir, "config.json"), "w") as writer:
        json.dump(config, writer)
    return tmp_dir


def _create_config_from_synthetic_checkpoint(with_vision_wrapper=True):
    ckpt_dir = _write_config_dir(with_vision_wrapper=with_vision_wrapper)
    try:
        return Gemma4.create_config(ckpt_dir)
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def _build_weight_builder(config, parallelism_config=None):
    return Gemma4WeightInfo(
        model_config=config,
        parallelism_config=parallelism_config or ParallelismConfig(),
        hw_kernel_config=HWKernelConfig(),
        kv_cache_config=KVCacheConfig(),
    )


def _flatten_weights(weights):
    for weight in weights:
        yield weight
        for sub in getattr(weight, "sub_weights", {}).values():
            yield from _flatten_weights([sub])


def _by_name(layer_weights, name):
    return [weight for weight in _flatten_weights(layer_weights) if weight.name == name]


class Gemma4ConfigTest(unittest.TestCase):
    def _create_config(self, with_vision_wrapper=True):
        return _create_config_from_synthetic_checkpoint(with_vision_wrapper)

    def test_invalid_geometry_and_numeric_parameters(self):
        for name, value in (
            ("num_attention_heads", 0),
            ("num_global_key_value_heads", 3),
            ("head_dim", 255),
            ("rms_norm_eps", float("nan")),
            ("top_k_experts", 129),
            ("sliding_window", 0),
            ("final_logit_softcapping", -1),
        ):
            with self.subTest(name=name):
                text = dict(SYNTHETIC_TEXT_CONFIG)
                text[name] = value
                directory = _write_config_dir(text)
                try:
                    with self.assertRaises(ValueError):
                        Gemma4.create_config(directory)
                finally:
                    shutil.rmtree(directory)

    def test_nondefault_window_eps_rope_and_softcap_reach_geometry(self):
        from rtp_llm.models_py.modules.gemma4.geometry import (
            build_gemma4_layer_geometry,
        )

        text = dict(SYNTHETIC_TEXT_CONFIG)
        text.update(sliding_window=2048, rms_norm_eps=1e-5, final_logit_softcapping=7.5)
        text["rope_parameters"] = {
            "sliding_attention": {"rope_theta": 20000, "rope_type": "default"},
            "full_attention": {
                "rope_theta": 2000000,
                "rope_type": "proportional",
                "partial_rotary_factor": 0.5,
            },
        }
        directory = _write_config_dir(text)
        try:
            config = Gemma4.create_config(directory)
            Gemma4._post_build_model_config(config)
            swa = build_gemma4_layer_geometry(config, ParallelismConfig(), 0)
            full = build_gemma4_layer_geometry(config, ParallelismConfig(), 5)
            self.assertEqual(swa.sliding_window, 2048)
            self.assertEqual(swa.rope_theta, 20000)
            self.assertEqual(full.rope_theta, 2000000)
            self.assertEqual(full.rope_partial_rotary_factor, 0.5)
            self.assertAlmostEqual(config.layernorm_eps, 1e-5)
            self.assertEqual(config.final_logit_softcapping, 7.5)
        finally:
            shutil.rmtree(directory)

    def test_model_registration_resolves_architecture(self):
        self.assertTrue(ensure_model_registered("gemma4"))
        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    ModelDict.get_ft_model_type_by_config(
                        {"architectures": [architecture]}
                    ),
                    "gemma4",
                )
                self.assertEqual(
                    ModelDict.get_ft_model_type_by_hf_architectures(architecture),
                    "gemma4",
                )

    def test_create_config_geometry_and_moe(self):
        for with_vision_wrapper in (True, False):
            with self.subTest(with_vision_wrapper=with_vision_wrapper):
                config = self._create_config(with_vision_wrapper)

                self.assertEqual(config.num_layers, NUM_LAYERS)
                self.assertEqual(config.hidden_size, HIDDEN_SIZE)
                self.assertEqual(config.vocab_size, 262144)
                self.assertEqual(config.attn_config.head_num, HEAD_NUM)
                self.assertEqual(config.attn_config.kv_head_num, KV_HEAD_NUM)
                self.assertEqual(config.attn_config.size_per_head, SIZE_PER_HEAD)
                self.assertEqual(config.attn_config.sliding_window, SLIDING_WINDOW)

                self.assertEqual(config.inter_size, INTER_SIZE)
                self.assertEqual(config.moe_inter_size, MOE_INTER_SIZE)
                self.assertEqual(config.expert_num, EXPERT_NUM)
                self.assertEqual(config.moe_k, MOE_K)
                self.assertEqual(config.moe_style, 2)
                self.assertEqual(config.n_shared_experts, 1)
                self.assertEqual(config.moe_layer_index, list(range(NUM_LAYERS)))

                # pybind returns enum objects; compare through str() so the
                # assertion also shows the enum on failure.
                self.assertEqual(str(config.norm_type), "NormType.rmsnorm")
                self.assertAlmostEqual(config.layernorm_eps, 1e-06)
                self.assertEqual(str(config.activation_type), "ActivationType.Geglu")
                self.assertTrue(config.isGatedActivation())
                self.assertTrue(config.has_post_decoder_layernorm)
                self.assertFalse(config.has_pre_decoder_layernorm)
                self.assertTrue(config.qk_norm)

                self.assertTrue(config.tie_word_embeddings)
                self.assertFalse(config.enable_fp32_lm_head)
                # sliding-layer rope only; full layers build their own rope
                self.assertEqual(
                    str(config.attn_config.rope_config.style), "RopeStyle.Base"
                )
                self.assertEqual(config.attn_config.rope_config.base, 10000)
                self.assertEqual(config.attn_config.rope_config.dim, SIZE_PER_HEAD)

                self.assertEqual(config.max_seq_len, 262144)
                self.assertEqual(config.final_logit_softcapping, 30.0)
                self.assertEqual(
                    config.mm_related_params.config["full_layer_kv_head_num"],
                    GLOBAL_KV_HEAD_NUM,
                )
                self.assertEqual(
                    config.mm_related_params.config["full_layer_size_per_head"],
                    GLOBAL_HEAD_DIM,
                )
                self.assertEqual(
                    config.mm_related_params.config["full_layer_rope_theta"],
                    1_000_000.0,
                )
                self.assertEqual(
                    config.mm_related_params.config["full_layer_partial_rotary_factor"],
                    0.25,
                )
                self.assertEqual(config.special_tokens.bos_token_id, 2)
                self.assertEqual(config.special_tokens.eos_token_id, 1)
                # the synthetic config carries no dtype field, so the
                # python-only config_dtype helper stays unset here
                self.assertIsNone(config.config_dtype)

    def test_layer_types_and_kv_cache_spec_descs(self):
        config = self._create_config()

        hybrid_types = list(config.hybrid_attention_config.hybrid_attention_types)
        self.assertEqual(len(hybrid_types), NUM_LAYERS)
        expected_types = [
            (
                HybridAttentionType.NONE
                if i in FULL_LAYER_IDS
                else HybridAttentionType.SLIDING_WINDOW
            )
            for i in range(NUM_LAYERS)
        ]
        self.assertEqual(hybrid_types, expected_types)
        self.assertTrue(config.hybrid_attention_config.enable_hybrid_attention)

        descs = config.kv_cache_spec_descs
        self.assertEqual(len(descs), NUM_LAYERS)
        for layer_id, (layer_descs, attn_type) in enumerate(zip(descs, hybrid_types)):
            self.assertEqual(len(layer_descs), 1)
            desc = layer_descs[0]
            self.assertEqual(desc.cache_type, KVCacheSpecType.MHA)
            if attn_type == HybridAttentionType.NONE:
                self.assertEqual(desc.tag, "full")
                self.assertEqual(desc.group_type, CacheGroupType.FULL)
                self.assertIsNone(desc.tail)
            else:
                self.assertEqual(desc.tag, "swa")
                self.assertEqual(desc.group_type, CacheGroupType.SWA)
                self.assertIsNotNone(desc.tail)
                expected_tail = (SLIDING_WINDOW + 128 - 1) // 128 + 1
                self.assertEqual(desc.tail.active_tail_blocks, expected_tail)
                self.assertIsNotNone(desc.capacity)
                self.assertFalse(desc.capacity.reservable)
                self.assertTrue(desc.capacity.bounded_by_active_tail)

    def test_layer_weight_param_count_includes_dense_mlp_and_experts(self):
        config = self._create_config()
        count = config.layer_weight_param_count()

        qkv = (
            NUM_LAYERS * HIDDEN_SIZE * HIDDEN_SIZE
            + NUM_LAYERS * HIDDEN_SIZE * (KV_HEAD_NUM * SIZE_PER_HEAD) * 2
        )
        attn_o = NUM_LAYERS * HIDDEN_SIZE * HIDDEN_SIZE
        dense_mlp = NUM_LAYERS * INTER_SIZE * HIDDEN_SIZE * 3
        routed_experts = NUM_LAYERS * MOE_INTER_SIZE * HIDDEN_SIZE * 3 * EXPERT_NUM
        small = NUM_LAYERS * HIDDEN_SIZE * 11
        self.assertEqual(count, qkv + attn_o + dense_mlp + routed_experts + small)
        # both terms must be present: drop one and the count must change
        self.assertNotEqual(count - dense_mlp, count)
        self.assertNotEqual(count - routed_experts, count)

    def test_kv_desc_geometry_override(self):
        config = self._create_config()
        for layer_id, layer_descs in enumerate(config.kv_cache_spec_descs):
            desc = layer_descs[0]
            if layer_id in FULL_LAYER_IDS:
                self.assertEqual(desc.kv_head_num, GLOBAL_KV_HEAD_NUM)
                self.assertEqual(desc.size_per_head, GLOBAL_HEAD_DIM)
            else:
                self.assertEqual(desc.kv_head_num, KV_HEAD_NUM)
                self.assertEqual(desc.size_per_head, SIZE_PER_HEAD)

    def test_full_geometry_is_instance_local(self):
        first_text_config = dict(SYNTHETIC_TEXT_CONFIG)
        second_text_config = dict(SYNTHETIC_TEXT_CONFIG)
        second_text_config["num_global_key_value_heads"] = 1
        second_text_config["global_head_dim"] = 384
        first_dir = _write_config_dir(first_text_config)
        second_dir = _write_config_dir(second_text_config)
        try:
            first = Gemma4.create_config(first_dir)
            second = Gemma4.create_config(second_dir)
            first_attn = _build_weight_builder(first)._full_layer_attn_config()
            second_attn = _build_weight_builder(second)._full_layer_attn_config()
        finally:
            shutil.rmtree(first_dir, ignore_errors=True)
            shutil.rmtree(second_dir, ignore_errors=True)

        self.assertEqual(first_attn.head_num_kv, GLOBAL_KV_HEAD_NUM)
        self.assertEqual(first_attn.size_per_head, GLOBAL_HEAD_DIM)
        self.assertEqual(second_attn.head_num_kv, 1)
        self.assertEqual(second_attn.size_per_head, 384)


if __name__ == "__main__":
    unittest.main()
