import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.load_config import LoadConfig, LoadMethod
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.qwen3_next.qwen3_next_weight import (
    Qwen3NextBaseWeight,
    Qwen35DenseWeight,
    Qwen35MoeWeight,
)
from rtp_llm.ops import HybridAttentionType
from rtp_llm.utils.model_weight import W, WeightStyle


def make_weight_info(*, attention_rank: bool, expert_rank: bool, stacked=False):
    info = Qwen35MoeWeight.__new__(Qwen35MoeWeight)
    info.prefix = "model.language_model."
    info._has_stacked_ckpt = stacked
    info._num_layers = 2
    info._hidden_size = 8
    info._head_num = 1
    info._head_num_kv = 1
    info._size_per_head = 8
    info._is_gated_activation = True
    info._align_size = 0
    info.expert_num_ = 4
    info.is_attn_model = attention_rank
    info.is_ffn_service = expert_rank
    info._quant_algo = SimpleNamespace(isQuant=lambda: False)
    info._quant_config = None
    info.gen_dummy_reciprocal = False
    info.output_vocab_ids = ()
    info.weight_style = WeightStyle.NONE
    info.tie_word_embeddings = False
    info.enable_fp32_lm_head = False
    info.model_config = SimpleNamespace(
        n_shared_experts=1,
        attn_config=SimpleNamespace(head_num=1, size_per_head=8),
        linear_attention_config=SimpleNamespace(
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        ),
        hybrid_attention_config=SimpleNamespace(
            hybrid_attention_types=[
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
            ]
        ),
    )
    return info


def component_names(weights):
    return {
        component.name for weight in weights for component in weight.get_components()
    }


def checkpoint_names(weights):
    return {
        ckpt.name
        for weight in weights
        for component in weight.get_components()
        for ckpt in component.weights
    }


class Qwen35AfdWeightTest(unittest.TestCase):
    def test_expert_rank_loads_all_experts_despite_union_world_ep(self):
        def initialize_base(instance):
            instance.expert_num_ = 256
            instance.is_attn_model = False
            instance.is_ffn_service = True
            instance.ep_size = 3
            instance.ep_rank = 2
            instance.dp_size = 3
            instance.dp_rank = 2
            instance.num_nodes = 3

        with patch.object(Qwen3NextBaseWeight, "__init__", initialize_base):
            info = Qwen35MoeWeight()

        self.assertEqual((info.ep_size, info.ep_rank), (1, 0))
        self.assertEqual((info.dp_size, info.dp_rank), (1, 0))
        self.assertEqual(info.num_nodes, 1)
        load_config = SimpleNamespace(
            ep_size=info.ep_size, ep_rank=info.ep_rank, phy2log=None
        )
        self.assertEqual(
            list(LoadConfig.get_selected_experts(load_config, 0, 256)),
            list(range(256)),
        )

    def test_attention_rank_keeps_gdn_attention_router_and_shared_expert(self):
        info = make_weight_info(attention_rank=True, expert_rank=False)
        weight_info = info._get_weight_info()
        self.assertEqual(
            component_names(weight_info.weights),
            {W.embedding, W.lm_head, W.final_ln_gamma},
        )

        gdn_names = component_names(weight_info.layer_weights[0])
        mha_names = component_names(weight_info.layer_weights[1])
        for names in (gdn_names, mha_names):
            self.assertTrue(
                {
                    W.pre_ln_gamma,
                    W.post_ln_gamma,
                    W.moe_gate,
                    W.ffn_w13,
                    W.ffn_w2,
                    W.shared_expert_gate,
                }
                <= names
            )
            self.assertFalse({W.moe_w1, W.moe_w2} & names)
        self.assertIn(W.linear_attn_qkvz_w, gdn_names)
        self.assertIn(W.linear_attn_ba_w, gdn_names)
        self.assertIn(W.attn_qkv_w, mha_names)
        self.assertIn(W.attn_o_w, mha_names)

        sources = checkpoint_names(weight_info.layer_weights[0])
        self.assertIn("model.language_model.layers.{i}.mlp.gate.weight", sources)
        self.assertIn(
            "model.language_model.layers.{i}.mlp.shared_expert.gate_proj.weight",
            sources,
        )
        self.assertIn(
            "model.language_model.layers.{i}.mlp.shared_expert.up_proj.weight",
            sources,
        )
        self.assertFalse(any(".mlp.experts." in name for name in sources))

    def test_attention_rank_without_shared_expert_keeps_only_router(self):
        info = make_weight_info(attention_rank=True, expert_rank=False)
        info.model_config.n_shared_experts = 0
        layer = info._get_weight_info().layer_weights[0]
        self.assertEqual(
            component_names(layer)
            & {
                W.moe_gate,
                W.ffn_w1,
                W.ffn_w2,
                W.ffn_w3,
                W.shared_expert_gate,
                W.moe_w1,
                W.moe_w2,
            },
            {W.moe_gate},
        )

    def test_expert_rank_loads_only_routed_experts_for_split_and_stacked_ckpt(self):
        for stacked in (False, True):
            with self.subTest(stacked=stacked):
                info = make_weight_info(
                    attention_rank=False, expert_rank=True, stacked=stacked
                )
                weight_info = info.get_weight_info()
                self.assertEqual(weight_info.weights, [])
                for layer in weight_info.layer_weights:
                    self.assertEqual(component_names(layer), {W.moe_w1, W.moe_w2})
                    self.assertTrue(
                        all(".mlp.experts." in key for key in checkpoint_names(layer))
                    )

    def test_normal_qwen35_keeps_both_router_and_routed_experts(self):
        info = make_weight_info(attention_rank=False, expert_rank=False)
        weight_info = info._get_weight_info()
        self.assertTrue(
            {W.moe_gate, W.moe_w1, W.moe_w2, W.ffn_w13, W.ffn_w2}
            <= component_names(weight_info.layer_weights[0])
        )
        self.assertEqual(len(weight_info.weights), 3)
        self.assertFalse(Qwen35DenseWeight.load_layer_weights_on_attn_rank)

    def test_expert_only_descriptors_survive_fp8_quant_conversion(self):
        info = make_weight_info(attention_rank=False, expert_rank=True)
        info._quant_algo = SimpleNamespace(isQuant=lambda: True)
        info._quant_config = Fp8BlockWiseQuantConfig(is_quanted=True)
        weight_info = info.get_weight_info()
        self.assertEqual(weight_info.weights, [])
        for layer in weight_info.layer_weights:
            self.assertEqual(len(layer), 1)
            self.assertEqual(set(layer[0].sub_weights), {W.moe_w1, W.moe_w2})

    def test_loader_reads_qwen35_attention_rank_layers(self):
        loader = ModelLoader.__new__(ModelLoader)
        loader._is_attn_model = True
        loader._weights_info = SimpleNamespace(load_layer_weights_on_attn_rank=True)
        loader._model_weights_info = SimpleNamespace(weights=[])
        loader._misc_weights_info = []
        loader._load_config = SimpleNamespace(num_layers=2)
        with patch.object(
            loader, "_load_layer_weights", side_effect=[{"a": 1}, {"b": 2}]
        ) as read:
            self.assertEqual(
                list(loader.prepare_weights("cpu")), [(0, "a", 1), (1, "b", 2)]
            )
        self.assertEqual(read.call_count, 2)

    def test_expert_rank_does_not_derive_lm_head_from_absent_embedding(self):
        loader = ModelLoader.__new__(ModelLoader)
        loader._weights_info = SimpleNamespace(
            uses_fastafd_weight_partition=True, is_ffn_service=True
        )
        weights = ModelWeights(1, "cpu", torch.bfloat16)
        loader._load_dynamic_weights(weights, "cpu")
        self.assertEqual(weights.global_weights, {})

    def test_ft_checkpoint_and_eplb_fail_before_loading(self):
        model_config = SimpleNamespace(task_type=None)
        weight_info = SimpleNamespace(
            uses_fastafd_weight_partition=True,
            enable_eplb_=False,
            phy_exp_num_=4,
            expert_num_=4,
            output_vocab_ids=(),
            weight_style=WeightStyle.NONE,
            is_ffn_service=True,
        )
        with self.assertRaisesRegex(ValueError, "pre-sharded checkpoints"):
            ModelLoader(
                model_config, weight_info, None, SimpleNamespace(is_ft_style=True)
            )
        weight_info.enable_eplb_ = True
        with self.assertRaisesRegex(ValueError, "does not support EPLB"):
            ModelLoader(
                model_config, weight_info, None, SimpleNamespace(is_ft_style=False)
            )
        weight_info.enable_eplb_ = False
        weight_info.output_vocab_ids = (1, 2)
        with self.assertRaisesRegex(ValueError, "output vocabulary pruning"):
            ModelLoader(
                model_config, weight_info, None, SimpleNamespace(is_ft_style=False)
            )
        weight_info.output_vocab_ids = ()
        with self.assertRaisesRegex(ValueError, "cannot load custom module weights"):
            ModelLoader(
                model_config,
                weight_info,
                [object()],
                SimpleNamespace(is_ft_style=False),
            )

    def test_fastsafetensors_is_rejected_before_loading(self):
        weight_info = SimpleNamespace(uses_fastafd_weight_partition=True)
        with self.assertRaisesRegex(ValueError, "requires scratch weight loading"):
            ModelLoader(
                SimpleNamespace(task_type=None),
                weight_info,
                None,
                SimpleNamespace(is_ft_style=False),
                load_method=LoadMethod.FASTSAFETENSORS,
            )


if __name__ == "__main__":
    unittest.main()
