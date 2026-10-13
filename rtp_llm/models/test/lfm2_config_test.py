import json
import os
import tempfile
from types import SimpleNamespace
from unittest import TestCase, main

import torch
from safetensors.torch import save_file

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.lfm2.lfm2 import Lfm2Vl
from rtp_llm.models.lfm2.lfm2_weight import Lfm2Weight
from rtp_llm.ops import (
    DataType,
    HybridAttentionType,
    HWKernelConfig,
    KVCacheSpecType,
    ParallelismConfig,
    RopeStyle,
)
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W

HIDDEN = 64
HEAD_NUM = 4
KV_HEAD_NUM = 2
INTER = 128
VOCAB = 96
CONV_KERNEL = 3

# attention-first mix: exercises dispatch on a pattern different from d1's
LAYER_TYPES = [
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
]


def _lfm2_vl_config(layer_types, conv_kernel=CONV_KERNEL):
    return {
        "architectures": ["Lfm2VlForConditionalGeneration"],
        "dtype": "bfloat16",
        "eos_token_id": 124900,
        "image_token_id": 124907,
        "text_config": {
            "hidden_size": HIDDEN,
            "num_attention_heads": HEAD_NUM,
            "num_key_value_heads": KV_HEAD_NUM,
            "num_hidden_layers": len(layer_types),
            "layer_types": layer_types,
            "vocab_size": VOCAB,
            "intermediate_size": INTER,
            "norm_eps": 1e-05,
            "conv_dim": HIDDEN,
            "conv_L_cache": conv_kernel,
            "tie_word_embeddings": True,
            "max_position_embeddings": 32768,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        },
        "vision_config": {"hidden_size": 1152},
    }


def _write_config(directory, config):
    with open(os.path.join(directory, "config.json"), "w") as writer:
        json.dump(config, writer)


class Lfm2ConfigTest(TestCase):
    def _create(self, layer_types=LAYER_TYPES):
        with tempfile.TemporaryDirectory() as ckpt:
            _write_config(ckpt, _lfm2_vl_config(layer_types))
            return Lfm2Vl._create_config(ckpt)

    def test_basic_config_fields(self):
        config = self._create()
        self.assertEqual(config.hidden_size, HIDDEN)
        self.assertEqual(config.num_layers, len(LAYER_TYPES))
        self.assertEqual(config.vocab_size, VOCAB)
        self.assertEqual(config.inter_size, INTER)
        self.assertEqual(config.attn_config.head_num, HEAD_NUM)
        self.assertEqual(config.attn_config.kv_head_num, KV_HEAD_NUM)
        self.assertEqual(config.attn_config.size_per_head, HIDDEN // HEAD_NUM)
        self.assertEqual(config.layernorm_eps, 1e-05)
        # string assignment is coerced into the NormType enum
        self.assertIn("rmsnorm", str(config.norm_type))
        self.assertIn("Swiglu", str(config.activation_type))
        self.assertTrue(config.tie_word_embeddings)
        self.assertTrue(config.qk_norm)
        self.assertEqual(config.max_seq_len, 32768)
        self.assertEqual(config.special_tokens.eos_token_id, 124900)
        self.assertEqual(config.config_dtype, "bfloat16")

    def test_rope_config(self):
        config = self._create()
        self.assertEqual(config.attn_config.rope_config.base, 1000000)
        # half-split rotary over the full head dim (llama style)
        self.assertEqual(config.attn_config.rope_config.style, RopeStyle.Base)
        self.assertEqual(config.attn_config.rope_config.dim, HIDDEN // HEAD_NUM)
        self.assertFalse(config.attn_config.use_mla)

    def test_hybrid_attention_types(self):
        config = self._create()
        expected = [
            HybridAttentionType.NONE if t == "full_attention" else HybridAttentionType.LINEAR
            for t in LAYER_TYPES
        ]
        self.assertEqual(
            config.hybrid_attention_config.hybrid_attention_types, expected
        )
        self.assertTrue(config.hybrid_attention_config.enable_hybrid_attention)

    def test_conv_state_layout_contract(self):
        config = self._create()
        linear = config.linear_attention_config
        self.assertEqual(linear.linear_conv_kernel_dim, CONV_KERNEL)
        self.assertEqual(linear.linear_key_head_dim, 1)
        self.assertEqual(linear.linear_value_head_dim, 1)
        self.assertEqual(linear.linear_num_key_heads, 1)
        self.assertEqual(linear.linear_num_value_heads, 2046)
        self.assertEqual(linear.ssm_state_dtype, DataType.TYPE_FP32)
        # The layout targets the real d1-3B hidden size 2048:
        # qkv (conv state width) = 1*1*2 + 1*2046 = 2048 = hidden
        qkv_real = 1 * 1 * 2 + 1 * 2046
        self.assertEqual(qkv_real, 2048)
        conv_state = (CONV_KERNEL - 1) * qkv_real
        self.assertEqual(conv_state, 2 * 2048)

    def test_kv_cache_spec_descs(self):
        config = self._create()
        Lfm2Vl._post_build_model_config(config)
        descs = config.kv_cache_spec_descs
        self.assertEqual(len(descs), len(LAYER_TYPES))
        for layer_type, layer_descs in zip(LAYER_TYPES, descs):
            self.assertEqual(len(layer_descs), 1)
            desc = layer_descs[0]
            if layer_type == "full_attention":
                self.assertEqual(desc.tag, "full")
                self.assertEqual(desc.cache_type, KVCacheSpecType.MHA)
            else:
                self.assertEqual(desc.tag, "linear")
                self.assertEqual(desc.cache_type, KVCacheSpecType.LINEAR)

    def test_mm_config(self):
        config = self._create()
        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[124907]])
        self.assertEqual(config.mm_model_config.mm_position_ids_style, 0)
        self.assertEqual(
            config.mm_related_params.special_token_ids["image_token_index"], 124907
        )
        self.assertEqual(
            config.mm_related_params.special_tokens.get("default_mm_token"), "<image>"
        )
        self.assertEqual(
            config.mm_related_params.config["ckpt_path"], config.ckpt_path
        )
        self.assertIn("vision_config", config.mm_related_params.config)
        self.assertEqual(
            config.mm_related_params.config["vision_config"]["hidden_size"], 1152
        )

    def test_unknown_layer_type_rejected(self):
        with self.assertRaises(ValueError):
            self._create(["full_attention", "mamba"])

    def test_layer_types_length_mismatch_rejected(self):
        bad = _lfm2_vl_config(LAYER_TYPES)
        bad["text_config"]["num_hidden_layers"] = len(LAYER_TYPES) + 1
        with tempfile.TemporaryDirectory() as ckpt:
            _write_config(ckpt, bad)
            with self.assertRaises(ValueError):
                Lfm2Vl._create_config(ckpt)

    def test_flat_text_config_fallback(self):
        # text-only Lfm2ForCausalLM-style flat config still parses
        flat = _lfm2_vl_config(LAYER_TYPES)["text_config"]
        with tempfile.TemporaryDirectory() as ckpt:
            _write_config(ckpt, flat)
            config = Lfm2Vl._create_config(ckpt)
        self.assertEqual(config.num_layers, len(LAYER_TYPES))
        self.assertEqual(config.hidden_size, HIDDEN)


class Lfm2WeightTest(TestCase):
    def _make_weight(self, layer_types=LAYER_TYPES):
        with tempfile.TemporaryDirectory() as ckpt:
            _write_config(ckpt, _lfm2_vl_config(layer_types))
            config = Lfm2Vl._create_config(ckpt)
        Lfm2Vl._post_build_model_config(config)
        return Lfm2Weight(
            model_config=config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        ), config

    def test_checkpoint_tensor_names(self):
        weight, _ = self._make_weight()
        weight._process_meta([{}], ["model.language_model.layers.0.conv.conv.weight"])
        weight_info = weight.get_weight_info()

        def _ckpt_names(module):
            subs = getattr(module, "sub_weights", None)
            if subs:
                names = set()
                for sub in subs.values():
                    names.update(_ckpt_names(sub))
                return names
            return set(module.get_ckpt_tensor_names())

        conv_layer = weight_info.layer_weights[LAYER_TYPES.index("conv")]
        conv_names = set()
        for module in conv_layer:
            conv_names.update(_ckpt_names(module))
        self.assertIn("model.language_model.layers.{i}.conv.in_proj.weight", conv_names)
        self.assertIn("model.language_model.layers.{i}.conv.conv.weight", conv_names)
        self.assertIn("model.language_model.layers.{i}.conv.out_proj.weight", conv_names)
        self.assertIn("model.language_model.layers.{i}.operator_norm.weight", conv_names)
        self.assertIn("model.language_model.layers.{i}.ffn_norm.weight", conv_names)
        self.assertIn("model.language_model.layers.{i}.feed_forward.w1.weight", conv_names)

        attn_layer = weight_info.layer_weights[0]
        attn_names = set()
        for module in attn_layer:
            attn_names.update(_ckpt_names(module))
        self.assertIn("model.language_model.layers.{i}.self_attn.q_proj.weight", attn_names)
        self.assertIn("model.language_model.layers.{i}.self_attn.k_proj.weight", attn_names)
        self.assertIn("model.language_model.layers.{i}.self_attn.v_proj.weight", attn_names)
        self.assertIn(
            "model.language_model.layers.{i}.self_attn.q_layernorm.weight", attn_names
        )
        self.assertIn(
            "model.language_model.layers.{i}.self_attn.k_layernorm.weight", attn_names
        )
        self.assertIn("model.language_model.layers.{i}.self_attn.out_proj.weight", attn_names)

        global_names = set()
        for module in weight_info.weights:
            global_names.update(_ckpt_names(module))
        self.assertIn("model.language_model.embed_tokens.weight", global_names)
        self.assertIn("model.language_model.embedding_norm.weight", global_names)
        self.assertIn("lm_head.weight", global_names)

    def _fake_checkpoint(self, directory, layer_types):
        tensors = {}
        for layer in range(len(layer_types)):
            p = f"model.language_model.layers.{layer}"
            tensors[f"{p}.operator_norm.weight"] = torch.full((HIDDEN,), 1.0)
            tensors[f"{p}.ffn_norm.weight"] = torch.full((HIDDEN,), 1.0)
            tensors[f"{p}.feed_forward.w1.weight"] = torch.full((INTER, HIDDEN), 3.0)
            tensors[f"{p}.feed_forward.w3.weight"] = torch.full((INTER, HIDDEN), 4.0)
            tensors[f"{p}.feed_forward.w2.weight"] = torch.full((HIDDEN, INTER), 5.0)
            if layer_types[layer] == "conv":
                # distinct markers per gate: B=1, C=2, x=3 rows of in_proj
                in_proj = torch.cat(
                    [
                        torch.full((HIDDEN, HIDDEN), 1.0),
                        torch.full((HIDDEN, HIDDEN), 2.0),
                        torch.full((HIDDEN, HIDDEN), 3.0),
                    ],
                    dim=0,
                )
                tensors[f"{p}.conv.in_proj.weight"] = in_proj
                tensors[f"{p}.conv.conv.weight"] = torch.full((HIDDEN, 1, CONV_KERNEL), 6.0)
                tensors[f"{p}.conv.out_proj.weight"] = torch.full((HIDDEN, HIDDEN), 7.0)
            else:
                head_dim = HIDDEN // HEAD_NUM
                tensors[f"{p}.self_attn.q_proj.weight"] = torch.full(
                    (HEAD_NUM * head_dim, HIDDEN), 1.0
                )
                tensors[f"{p}.self_attn.k_proj.weight"] = torch.full(
                    (KV_HEAD_NUM * head_dim, HIDDEN), 2.0
                )
                tensors[f"{p}.self_attn.v_proj.weight"] = torch.full(
                    (KV_HEAD_NUM * head_dim, HIDDEN), 3.0
                )
                tensors[f"{p}.self_attn.out_proj.weight"] = torch.full(
                    (HIDDEN, HIDDEN), 7.0
                )
                tensors[f"{p}.self_attn.q_layernorm.weight"] = torch.full(
                    (head_dim,), 1.0
                )
                tensors[f"{p}.self_attn.k_layernorm.weight"] = torch.full(
                    (head_dim,), 1.0
                )
        tensors["model.language_model.embed_tokens.weight"] = torch.full(
            (VOCAB, HIDDEN), 8.0
        )
        tensors["model.language_model.embedding_norm.weight"] = torch.full(
            (HIDDEN,), 9.0
        )
        save_file(tensors, os.path.join(directory, "model.safetensors"))

    def test_conv_gate_order_and_layout(self):
        weight, config = self._make_weight()
        with tempfile.TemporaryDirectory() as ckpt:
            self._fake_checkpoint(ckpt, LAYER_TYPES)
            weight._process_meta(
                [{}], ["model.language_model.layers.1.conv.conv.weight"]
            )
            weight_info = weight.get_weight_info()
            database = CkptDatabase(ckpt)
            from rtp_llm.model_loader.tensor_source import DatabaseTensorSource

            source = DatabaseTensorSource(database)
            exported_device = SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda _name, tensor, use_swizzle_a=None: tensor,
            )
            load_config = weight.create_load_config(
                torch.float32, database, exported_device=exported_device
            )

            conv_layer_id = LAYER_TYPES.index("conv")
            loaded = {}
            for module in weight_info.layer_weights[conv_layer_id]:
                loaded.update(module.load(source, conv_layer_id, "cpu", load_config))

            in_proj = loaded[W.conv_in_proj_w]
            self.assertEqual(tuple(in_proj.shape), (HIDDEN, 3 * HIDDEN))
            b, c, x = in_proj.chunk(3, dim=1)
            # HF chunk order: B (input gate), C (output gate), x (value)
            self.assertTrue(torch.all(b == 1.0))
            self.assertTrue(torch.all(c == 2.0))
            self.assertTrue(torch.all(x == 3.0))

            conv_w = loaded[W.conv_w]
            self.assertEqual(tuple(conv_w.shape), (HIDDEN, CONV_KERNEL))

            out_proj = loaded[W.conv_out_proj_w]
            self.assertEqual(tuple(out_proj.shape), (HIDDEN, HIDDEN))

            # gated FFN weights arrive merged: w13 = [gate | up]
            self.assertEqual(tuple(loaded[W.ffn_w13].shape), (HIDDEN, 2 * INTER))
            self.assertEqual(tuple(loaded[W.ffn_w2].shape), (INTER, HIDDEN))
            gate, up = loaded[W.ffn_w13].chunk(2, dim=1)
            self.assertTrue(torch.all(gate == 3.0))
            self.assertTrue(torch.all(up == 4.0))

    def test_attn_qkv_merge(self):
        weight, config = self._make_weight()
        with tempfile.TemporaryDirectory() as ckpt:
            self._fake_checkpoint(ckpt, LAYER_TYPES)
            weight._process_meta(
                [{}], ["model.language_model.layers.0.self_attn.q_proj.weight"]
            )
            weight_info = weight.get_weight_info()
            database = CkptDatabase(ckpt)
            from rtp_llm.model_loader.tensor_source import DatabaseTensorSource

            source = DatabaseTensorSource(database)
            exported_device = SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda _name, tensor, use_swizzle_a=None: tensor,
            )
            load_config = weight.create_load_config(
                torch.float32, database, exported_device=exported_device
            )

            loaded = {}
            for module in weight_info.layer_weights[0]:
                loaded.update(module.load(source, 0, "cpu", load_config))

            head_dim = HIDDEN // HEAD_NUM
            qkv = loaded[W.attn_qkv_w]
            self.assertEqual(
                tuple(qkv.shape),
                (HIDDEN, HEAD_NUM * head_dim + 2 * KV_HEAD_NUM * head_dim),
            )
            # q markers (1.0) span the head-major q region of the merged layout
            values = set(torch.unique(qkv).tolist())
            self.assertEqual(values, {1.0, 2.0, 3.0})
            self.assertEqual(
                tuple(loaded[W.q_ln_gamma].shape), (head_dim,)
            )
            self.assertEqual(
                tuple(loaded[W.attn_o_w].shape), (HIDDEN, HIDDEN)
            )


if __name__ == "__main__":
    main()
