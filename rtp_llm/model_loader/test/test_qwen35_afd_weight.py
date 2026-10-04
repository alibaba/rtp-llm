import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, QuantizationConfig
from rtp_llm.model_loader.load_config import LoadConfig, LoadMethod
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.models.qwen3_next.qwen3_next_weight import (
    Qwen3NextBaseWeight,
    Qwen35DenseWeight,
    Qwen35MoeWeight,
)
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
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
    def _partitioned_weight_info(self, rank, expert_parallel_size=1, expert_num=256):
        parallelism = ParallelismConfig()
        parallelism.world_size = 2 + expert_parallel_size
        parallelism.world_rank = rank
        ffn = parallelism.ffn_disaggregate_config
        ffn.attention_dp_size = 2
        ffn.ffn_tp_size = expert_parallel_size
        ffn.is_ffn_rank = rank >= 2

        def initialize_base(instance, model_config, parallelism_config):
            instance.expert_num_ = expert_num
            instance.is_attn_model = rank < 2
            instance.is_ffn_service = rank >= 2
            instance.ep_size = parallelism.world_size
            instance.ep_rank = rank
            instance.dp_size = parallelism.world_size
            instance.dp_rank = rank
            instance.num_nodes = 3

        with patch.object(Qwen3NextBaseWeight, "__init__", initialize_base):
            return Qwen35MoeWeight(None, parallelism)

    def test_expert_rank_loads_all_experts_despite_union_world_ep(self):
        info = self._partitioned_weight_info(2)

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

    def test_expert_group_loads_disjoint_complete_expert_ranges(self):
        selected = []
        for rank in (2, 3):
            info = self._partitioned_weight_info(rank, expert_parallel_size=2)
            self.assertEqual((info.ep_size, info.ep_rank), (2, rank - 2))
            self.assertEqual((info.dp_size, info.dp_rank, info.num_nodes), (1, 0, 1))
            self.assertFalse(info._moe_pure_tp_mode)
            selected.append(
                list(
                    LoadConfig.get_selected_experts(
                        SimpleNamespace(
                            ep_size=info.ep_size, ep_rank=info.ep_rank, phy2log=None
                        ),
                        0,
                        256,
                    )
                )
            )
        self.assertEqual(selected, [list(range(128)), list(range(128, 256))])
        for rank in (0, 1):
            info = self._partitioned_weight_info(rank, expert_parallel_size=2)
            self.assertEqual((info.ep_size, info.ep_rank), (1, 0))

    def test_expert_group_requires_divisible_expert_count(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            self._partitioned_weight_info(2, expert_parallel_size=3)

    def test_fp8_expert_group_reads_matching_weight_and_scale_shards(self):
        info = make_weight_info(attention_rank=False, expert_rank=True)
        info._quant_algo = SimpleNamespace(isQuant=lambda: True)
        info._quant_config = Fp8BlockWiseQuantConfig(is_quanted=True)
        components = [
            atomic
            for quant_weight in info.get_weight_info()
            .layer_weights[0][0]
            .get_components()
            for atomic in quant_weight.sub_weights.values()
        ]
        self.assertEqual(
            {weight.name for weight in components},
            {W.moe_w1, W.moe_s1, W.moe_w2, W.moe_s2},
        )

        class Source(TensorSource):
            def __init__(self, tensors):
                self.tensors = tensors
                self.reads = []

            def load_tensor(self, name, data_type=torch.float16):
                self.reads.append(name)
                return [self.tensors[name].to(data_type)]

            def has_tensor(self, name):
                return name in self.tensors

        def load_config(ep_size, ep_rank):
            return LoadConfig.model_construct(
                ep_size=ep_size,
                ep_rank=ep_rank,
                tp_size=1,
                tp_rank=0,
                dp_size=1,
                dp_rank=0,
                ffn_tp_size=1,
                ffn_tp_rank=0,
                hidden_size=128,
                head_num=1,
                head_num_kv=1,
                size_per_head=128,
                moe_pure_tp_mode=False,
                moe_pure_tp_preshard=False,
                compute_dtype=torch.bfloat16,
            )

        for weight in components:
            with self.subTest(weight=weight.name):
                shape = (1, 1) if weight.name in (W.moe_s1, W.moe_s2) else (128, 128)
                tensors = {
                    ckpt.name.format(i=0, i_1=1, expert_id=expert): torch.full(
                        shape, float(1 + expert + 4 * index)
                    )
                    for index, ckpt in enumerate(weight.weights)
                    for expert in range(4)
                }
                full = weight._load_raw_tensor(
                    Source(tensors), 0, "cpu", load_config(1, 0)
                )[weight.name]
                shards = []
                for rank in (0, 1):
                    source = Source(tensors)
                    config = load_config(2, rank)
                    raw = weight._load_raw_tensor(source, 0, "cpu", config)
                    shard = weight._split(raw, config)[weight.name]
                    self.assertEqual(shard.shape, (2, *full.shape[1:]))
                    self.assertTrue(source.reads)
                    self.assertTrue(
                        all(
                            any(
                                f".experts.{expert}." in key
                                for expert in range(rank * 2, rank * 2 + 2)
                            )
                            for key in source.reads
                        )
                    )
                    shards.append(shard.float())
                torch.testing.assert_close(
                    torch.cat(shards), full.float(), rtol=0, atol=0
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


class Qwen35MixedFp8WeightTest(unittest.TestCase):
    def _load_quant_config(self, exclusions, legacy=(), *, nested=False):
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": list(exclusions),
                "exclude": list(legacy),
            }
        }
        if nested:
            config = {"text_config": config}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(config))
            return QuantizationConfig.load_from_ckpt(directory)

    def _mixed_exclusions(self):
        # Each module is excluded in every layer where it exists, as in the
        # Qwen3.5-397B checkpoint (45 GDN layers and 15 full-attention layers).
        exclusions = {
            f"model.language_model.layers.0.linear_attn.{name}"
            for name in ("in_proj_qkv", "in_proj_z", "out_proj")
        }
        exclusions.update(
            f"model.language_model.layers.1.self_attn.{name}_proj"
            for name in ("q", "k", "v", "o")
        )
        exclusions.update(
            f"model.language_model.layers.{layer}.mlp.shared_expert.{name}_proj"
            for layer in (0, 1)
            for name in ("gate", "up", "down")
        )
        return exclusions

    def _weight_info(self, quant_config, *, attention_rank=False, expert_rank=False):
        info = make_weight_info(attention_rank=attention_rank, expert_rank=expert_rank)
        info._quant_algo = SimpleNamespace(isQuant=lambda: True)
        info._quant_config = quant_config
        return info.get_weight_info()

    def test_checkpoint_modules_to_not_convert_preserves_legacy_exclusions(self):
        exclusions = self._mixed_exclusions()
        for nested in (False, True):
            with self.subTest(nested=nested):
                quant_config = self._load_quant_config(
                    exclusions, legacy=("lm_head",), nested=nested
                )
                self.assertIsInstance(quant_config, Fp8BlockWiseQuantConfig)
                self.assertTrue(quant_config.is_quanted())
                self.assertEqual(quant_config.group_size(), 128)
                self.assertEqual(quant_config.exclude_modules, exclusions | {"lm_head"})

    def test_baseline_and_afd_keep_only_routed_experts_fp8(self):
        quant_config = self._load_quant_config(self._mixed_exclusions())
        for attention_rank, expert_rank in (
            (False, False),
            (True, False),
            (False, True),
        ):
            with self.subTest(attention_rank=attention_rank, expert_rank=expert_rank):
                info = self._weight_info(
                    quant_config,
                    attention_rank=attention_rank,
                    expert_rank=expert_rank,
                )
                for layer_index, layer in enumerate(info.layer_weights):
                    components = [
                        component
                        for weight in layer
                        for component in weight.get_components()
                    ]
                    quantized = [
                        component
                        for component in components
                        if isinstance(component, PerBlockFp8Weight)
                    ]
                    self.assertEqual(
                        {weight.kernel.name for weight in quantized},
                        set() if attention_rank else {W.moe_w1, W.moe_w2},
                    )
                    for weight in quantized:
                        self.assertIsNotNone(weight.scale)
                        self.assertTrue(
                            all(
                                ".mlp.experts." in ckpt.name
                                and ckpt.name.endswith(".weight_scale_inv")
                                for ckpt in weight.scale.weights
                            )
                        )
                    if not expert_rank:
                        unquantized = {
                            component.name: component
                            for component in components
                            if not isinstance(component, PerBlockFp8Weight)
                        }
                        expected = {W.ffn_w13, W.ffn_w2}
                        if layer_index == 0:
                            expected.update({W.linear_attn_qkvz_w, W.linear_attn_out_w})
                        else:
                            expected.update({W.attn_qkv_w, W.attn_o_w, W.attn_gate_w})
                        self.assertTrue(expected <= unquantized.keys())
                        for name in expected:
                            self.assertTrue(
                                all(
                                    ckpt.name.endswith(".weight")
                                    for ckpt in unquantized[name].weights
                                )
                            )

    def test_partial_fused_exclusion_is_rejected(self):
        for module in ("linear_attn.in_proj_qkv", "mlp.shared_expert.gate_proj"):
            with self.subTest(module=module):
                quant_config = self._load_quant_config(
                    {f"model.language_model.layers.0.{module}"}
                )
                with self.assertRaisesRegex(
                    ValueError, "mixes excluded and quantized checkpoint modules"
                ):
                    self._weight_info(quant_config)

    def test_exclusion_applies_only_to_the_named_layer(self):
        info = make_weight_info(attention_rank=False, expert_rank=False)
        info.model_config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
        ]
        quant_config = self._load_quant_config(
            {
                f"model.language_model.layers.0.{module}"
                for module in (
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "mlp.shared_expert.gate_proj",
                    "mlp.shared_expert.up_proj",
                )
            }
        )
        info._quant_algo = SimpleNamespace(isQuant=lambda: True)
        info._quant_config = quant_config
        weight_info = info.get_weight_info()
        for layer_id, layer in enumerate(weight_info.layer_weights):
            quantized_names = {
                component.kernel.name
                for weight in layer
                for component in weight.get_components()
                if isinstance(component, PerBlockFp8Weight)
            }
            for name in (W.linear_attn_qkvz_w, W.ffn_w13):
                self.assertEqual(name in quantized_names, layer_id == 1)
            self.assertTrue({W.moe_w1, W.moe_w2} <= quantized_names)
        self.assertIsNone(quant_config._weight_layer_id)

    def test_concrete_exclusion_requires_layer_context_for_direct_conversion(self):
        info = make_weight_info(attention_rank=False, expert_rank=False)
        weight = info._create_linear_attn_qkvz_weight()
        quant_config = self._load_quant_config(
            {
                f"model.language_model.layers.0.linear_attn.{name}"
                for name in ("in_proj_qkv", "in_proj_z")
            }
        )
        with self.assertRaisesRegex(ValueError, "requires a layer index"):
            PerBlockFp8Weight.support(quant_config, weight)
        # An explicit template exclusion unambiguously applies to all layers.
        quant_config.exclude_modules = {
            key.replace("layers.0.", "layers.{i}.")
            for key in quant_config.exclude_modules
        }
        self.assertFalse(PerBlockFp8Weight.support(quant_config, weight))

    def test_no_exclusions_preserves_attention_and_shared_fp8_scales(self):
        info = self._weight_info(self._load_quant_config(()))
        for layer_index, layer in enumerate(info.layer_weights):
            quantized = {
                component.kernel.name: component
                for weight in layer
                for component in weight.get_components()
                if isinstance(component, PerBlockFp8Weight)
            }
            expected = {W.ffn_w13, W.ffn_w2, W.moe_w1, W.moe_w2}
            if layer_index == 0:
                expected.update({W.linear_attn_qkvz_w, W.linear_attn_out_w})
            else:
                expected.update({W.attn_qkv_w, W.attn_o_w, W.attn_gate_w})
            self.assertTrue(expected <= quantized.keys())
            for name in expected:
                self.assertIsNotNone(quantized[name].scale)


if __name__ == "__main__":
    unittest.main()
