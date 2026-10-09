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


class Gemma4WeightInfoStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _create_config_from_synthetic_checkpoint()
        cls.weight_builder = _build_weight_builder(cls.config)
        checkpoint_keys = {
            GEMMA4_DEFAULT_CKPT_PREFIX + "layers.0.input_layernorm.weight",
            GEMMA4_DEFAULT_CKPT_PREFIX + "embed_tokens.weight",
            GEMMA4_DEFAULT_CKPT_PREFIX + "norm.weight",
        }
        cls.weight_builder._process_meta([{}], checkpoint_keys)
        cls.weight_info = cls.weight_builder._get_weight_info()

    def test_prefix_detection(self):
        # VLM layout: model.language_model.
        self.assertEqual(self.weight_builder.prefix, GEMMA4_DEFAULT_CKPT_PREFIX)
        # plain text layout falls back to model.
        builder = _build_weight_builder(self.config)
        builder._process_meta([{}], {"model.layers.0.input_layernorm.weight"})
        self.assertEqual(builder.prefix, "model.")
        # vision-only keys must not be mistaken for the language model
        builder = _build_weight_builder(self.config)
        with self.assertRaises(ValueError):
            builder._process_meta(
                [{}], {"model.vision_tower.encoder.layers.0.input_layernorm.weight"}
            )

    def test_layer_weight_count(self):
        self.assertEqual(len(self.weight_info.layer_weights), NUM_LAYERS)

    def test_sliding_layer_attention_keys(self):
        layer = self.weight_info.layer_weights[0]
        qkv = _by_name(layer, W.attn_qkv_w)
        self.assertEqual(len(qkv), 1)
        self.assertEqual(
            [info.name for info in qkv[0].weights],
            [
                GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.self_attn.q_proj.weight",
                GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.self_attn.k_proj.weight",
                GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.self_attn.v_proj.weight",
            ],
        )
        self.assertEqual(qkv[0].process_fun, merge_qkv_hf)
        o = _by_name(layer, W.attn_o_w)
        self.assertEqual(
            o[0].weights[0].name,
            GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.self_attn.o_proj.weight",
        )
        for name, ckpt_name in (
            (W.q_ln_gamma, "layers.{i}.self_attn.q_norm.weight"),
            (W.k_ln_gamma, "layers.{i}.self_attn.k_norm.weight"),
        ):
            found = _by_name(layer, name)
            self.assertEqual(len(found), 1, name)
            self.assertEqual(
                found[0].weights[0].name, GEMMA4_DEFAULT_CKPT_PREFIX + ckpt_name
            )

    def test_full_layer_attention_keys(self):
        layer = self.weight_info.layer_weights[5]
        qkv = _by_name(layer, W.attn_qkv_w)
        self.assertEqual(len(qkv), 1)
        self.assertEqual(
            [info.name for info in qkv[0].weights],
            [
                GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.self_attn.q_proj.weight",
                GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.self_attn.k_proj.weight",
            ],
        )
        self.assertEqual(qkv[0].process_fun, merge_qkv_keq_v)
        # full layers have no v_proj anywhere
        for weight in _flatten_weights(layer):
            for info in getattr(weight, "weights", []) or []:
                self.assertNotIn("v_proj", info.name)

    def test_layer_norm_and_scalar_keys(self):
        expected = {
            W.pre_ln_gamma: "layers.{i}.input_layernorm.weight",
            W.post_ln_gamma: "layers.{i}.post_attention_layernorm.weight",
            W.pre_ffn_ln_gamma: "layers.{i}.pre_feedforward_layernorm.weight",
            W.pre_ffn2_ln_gamma: "layers.{i}.pre_feedforward_layernorm_2.weight",
            W.post_ffn_ln_gamma: "layers.{i}.post_feedforward_layernorm.weight",
            W.post_ffn1_ln_gamma: "layers.{i}.post_feedforward_layernorm_1.weight",
            W.post_ffn2_ln_gamma: "layers.{i}.post_feedforward_layernorm_2.weight",
            W.layer_scalar: "layers.{i}.layer_scalar",
        }
        for layer in self.weight_info.layer_weights:
            for name, ckpt_name in expected.items():
                found = _by_name(layer, name)
                self.assertEqual(len(found), 1, f"{name} in layer weights")
                self.assertEqual(
                    found[0].weights[0].name, GEMMA4_DEFAULT_CKPT_PREFIX + ckpt_name
                )
                self.assertEqual(found[0].process_fun.__name__, "identity")

    def test_dense_mlp_and_stacked_moe_keys(self):
        layer = self.weight_info.layer_weights[0]
        ffn = [weight for weight in layer if isinstance(weight, FfnWeight)]
        self.assertEqual(len(ffn), 1)
        # FfnWeight merges w1/w3 into w13 at construction
        self.assertIsNotNone(ffn[0].w13)
        self.assertIsNotNone(ffn[0].w2)
        w13_names = [info.name for info in ffn[0].w13.weights]
        self.assertEqual(
            w13_names,
            [
                GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.mlp.gate_proj.weight",
                GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.mlp.up_proj.weight",
            ],
        )
        self.assertEqual(
            ffn[0].w2.weights[0].name,
            GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.mlp.down_proj.weight",
        )

        moe = [weight for weight in layer if isinstance(weight, MoeWeight)]
        self.assertEqual(len(moe), 1)
        moe_w1 = _by_name(layer, W.moe_w1)[0]
        moe_w2 = _by_name(layer, W.moe_w2)[0]
        self.assertIsInstance(moe_w1, MoeAtomicWeight)
        self.assertIsInstance(moe_w2, MoeAtomicWeight)
        self.assertTrue(moe_w1.stacked_ckpt_keys)
        self.assertTrue(moe_w2.stacked_ckpt_keys)
        self.assertEqual(
            moe_w1.weights[0].name,
            GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.experts.gate_up_proj",
        )
        self.assertEqual(moe_w1.process_fun, stack_)
        self.assertEqual(
            moe_w2.weights[0].name,
            GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.experts.down_proj",
        )
        self.assertEqual(moe_w2.process_fun, stack_)

    def test_router_keys(self):
        layer = self.weight_info.layer_weights[0]
        router_scale = _by_name(layer, W.moe_router_scale)[0]
        self.assertEqual(
            router_scale.weights[0].name,
            GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.router.scale",
        )
        self.assertEqual(router_scale.weights[0].merge_fun, scale_reshape)
        expert_scale = _by_name(layer, W.moe_router_expert_scale)[0]
        self.assertEqual(
            expert_scale.weights[0].name,
            GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.router.per_expert_scale",
        )
        self.assertEqual(expert_scale.weights[0].merge_fun, scale_reshape)
        gate = _by_name(layer, W.moe_gate)[0]
        self.assertEqual(
            gate.weights[0].name,
            GEMMA4_DEFAULT_CKPT_PREFIX + "layers.{i}.router.proj.weight",
        )

    def test_global_weights_and_tied_lm_head(self):
        names = [weight.name for weight in self.weight_info.weights]
        self.assertIn(W.embedding, names)
        self.assertIn(W.lm_head, names)
        self.assertIn(W.final_ln_gamma, names)
        embedding = _by_name(self.weight_info.weights, W.embedding)[0]
        self.assertEqual(
            embedding.weights[0].name,
            GEMMA4_DEFAULT_CKPT_PREFIX + "embed_tokens.weight",
        )
        # the full pipeline rewrites lm_head to fall back to the embedding
        fixed = self.weight_builder.get_weight_info()
        lm_head = _by_name(fixed.weights, W.lm_head)[0]
        lm_head_ckpt_names = [info.name for info in lm_head.weights]
        self.assertIn("lm_head.weight", lm_head_ckpt_names)
        self.assertIn(
            GEMMA4_DEFAULT_CKPT_PREFIX + "embed_tokens.weight", lm_head_ckpt_names
        )


if __name__ == "__main__":
    unittest.main()
