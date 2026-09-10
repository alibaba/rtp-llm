import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.ffn_weight import FfnWeight, MoeWeight
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource, TensorSource
from rtp_llm.models.deepseek_v2 import DeepSeekV2, DeepSeekV2Weight
from rtp_llm.models.kimi_k25.kimi_k25 import KimiK25
from rtp_llm.models.kimi_k25.kimi_k25_weight import KimiK25Weight
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Moe
from rtp_llm.models.qwen_v2_moe import Qwen2Moe
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W


def _deepseek_text_config(n_shared_experts: int = 2) -> dict:
    return {
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 4,
        "vocab_size": 32000,
        "hidden_size": 4096,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "routed_scaling_factor": 2.5,
        "num_experts_per_tok": 6,
        "n_routed_experts": 64,
        "moe_intermediate_size": 1408,
        "n_shared_experts": n_shared_experts,
        "first_k_dense_replace": 1,
    }


def _assert_moe_config_is_constructible(
    test: unittest.TestCase, config, expected_n_shared_experts: int = 2
) -> None:
    adapter = MoEConfigAdapter(
        model_config=config,
        parallelism_config=ParallelismConfig(),
        moe_config=MoeConfig(),
    )
    test.assertEqual(adapter.moe_inter_dim, 1408)
    test.assertEqual(adapter.n_shared_experts, expected_n_shared_experts)
    test.assertEqual(adapter.route_scale, 2.5)


class _RoutedOnlyTensorSource(TensorSource):
    def __init__(self):
        self.requested_names = []

    def load_tensor(self, name, data_type=torch.float16):
        if "shared_experts" in name:
            raise AssertionError(f"unexpected shared-expert lookup: {name}")
        self.requested_names.append(name)
        return [torch.ones((2, 2), dtype=data_type)]

    def has_tensor(self, name):
        return False

    def get_database(self):
        return None


class MoeConfigPropagationTest(unittest.TestCase):
    def test_scheduler_prefill_batch_capacity_reaches_generic_moe(self):
        for configured_cap, expected_cap in ((0, 12288), (7000, 7000)):
            with self.subTest(configured_cap=configured_cap):
                model_config = ModelConfig()
                model_config.max_seq_len = 4096
                model_config.model_name = "test-model"
                scheduler_config = SimpleNamespace(
                    max_context_batch_size=3,
                    max_batch_tokens_size=configured_cap,
                )
                engine_config = SimpleNamespace(
                    runtime_config=SimpleNamespace(
                        fifo_scheduler_config=scheduler_config,
                        model_name="",
                    )
                )

                ModelFactory.update_engine_config_from_model_config(
                    engine_config, model_config
                )
                adapter = MoEConfigAdapter(
                    model_config=model_config,
                    parallelism_config=ParallelismConfig(),
                    moe_config=MoeConfig(),
                )

                self.assertEqual(
                    model_config.moe_prefill_max_tokens_per_rank, expected_cap
                )
                self.assertEqual(adapter.prefill_max_tokens_per_rank, expected_cap)
                self.assertEqual(adapter.max_tokens_per_rank, expected_cap)

    def test_declared_prefill_capacity_reaches_adapter(self):
        config = ModelConfig()
        self.assertIsNone(config.moe_prefill_max_tokens_per_rank)
        config.max_seq_len = 4096
        moe_config = MoeConfig()
        moe_config.ll_num_max_token = 32
        for capacity, expected in ((None, 4096), (0, 0), (7000, 7000)):
            with self.subTest(capacity=capacity):
                config.moe_prefill_max_tokens_per_rank = capacity
                adapter = MoEConfigAdapter(
                    model_config=config,
                    parallelism_config=ParallelismConfig(),
                    moe_config=moe_config,
                )
                self.assertEqual(adapter.prefill_max_tokens_per_rank, expected)
                self.assertEqual(adapter.max_tokens_per_rank, max(32, expected))

    def test_declared_moe_w1_layout_reaches_adapter(self):
        config = ModelConfig()
        self.assertEqual(config.moe_w1_layout, "up_gate")
        for layout in ("gate_up", "up_gate"):
            with self.subTest(layout=layout):
                config.moe_w1_layout = layout
                adapter = MoEConfigAdapter(
                    model_config=config,
                    parallelism_config=ParallelismConfig(),
                    moe_config=MoeConfig(),
                )
                self.assertEqual(adapter.moe_w1_layout, layout)

    def test_physical_experts_must_partition_evenly_across_ep_ranks(self):
        config = ModelConfig()
        config.expert_num = 64
        parallelism = ParallelismConfig()
        parallelism.ep_size = 8
        for redundant in (0, 5, 8):
            with self.subTest(redundant=redundant):
                config.eplb_config.redundant_expert = redundant
                if redundant == 5:
                    with self.assertRaisesRegex(ValueError, "69.*divisible.*8"):
                        MoEConfigAdapter(config, parallelism, MoeConfig())
                else:
                    ranges = []
                    for rank in range(8):
                        parallelism.ep_rank = rank
                        adapter = MoEConfigAdapter(config, parallelism, MoeConfig())
                        ranges.extend(
                            range(adapter.local_expert_start, adapter.local_expert_end)
                        )
                    self.assertEqual(ranges, list(range(64 + redundant)))

    def _assert_independently_sized_qwen_shared_expert(
        self, config, model_cls, shared_width, routed_width, *, stacked=False
    ):
        # Keep the real expert counts and independently specified FFN widths;
        # a small hidden dimension makes the checkpoint inexpensive on CPU.
        config.hidden_size = 8
        parallelism_config = ParallelismConfig()
        adapter = MoEConfigAdapter(
            model_config=config,
            parallelism_config=parallelism_config,
            moe_config=MoeConfig(),
        )

        has_shared = shared_width > 0
        self.assertEqual(config.moe_style, 2 if has_shared else 1)
        self.assertEqual(config.n_shared_experts, int(has_shared))
        self.assertEqual(adapter.n_shared_experts, int(has_shared))
        self.assertEqual(config.inter_size, shared_width)
        self.assertEqual(config.moe_inter_size, routed_width)
        self.assertEqual(adapter.moe_inter_dim, routed_width)

        prefix = "model.language_model." if model_cls is Qwen35Moe else "model."
        mlp = prefix + "layers.0.mlp."
        checkpoint = {
            prefix + "layers.0.input_layernorm.weight": torch.ones(8),
            mlp + "gate.weight": torch.ones(config.expert_num, 8),
        }
        if has_shared:
            checkpoint.update(
                {
                    mlp + "shared_expert_gate.weight": torch.ones(1, 8),
                    mlp
                    + "shared_expert.gate_proj.weight": torch.full(
                        (shared_width, 8), 1.0
                    ),
                    mlp
                    + "shared_expert.up_proj.weight": torch.full(
                        (shared_width, 8), 3.0
                    ),
                    mlp
                    + "shared_expert.down_proj.weight": torch.full(
                        (8, shared_width), 2.0
                    ),
                }
            )
        if stacked:
            checkpoint[mlp + "experts.gate_up_proj"] = torch.cat(
                (
                    torch.full((config.expert_num, routed_width, 8), 1.0),
                    torch.full((config.expert_num, routed_width, 8), 3.0),
                ),
                dim=1,
            )
            checkpoint[mlp + "experts.down_proj"] = torch.full(
                (config.expert_num, 8, routed_width), 2.0
            )
        else:
            for expert_id in range(config.expert_num):
                expert = mlp + f"experts.{expert_id}."
                checkpoint[expert + "gate_proj.weight"] = torch.full(
                    (routed_width, 8), 1.0
                )
                checkpoint[expert + "up_proj.weight"] = torch.full(
                    (routed_width, 8), 3.0
                )
                checkpoint[expert + "down_proj.weight"] = torch.full(
                    (8, routed_width), 2.0
                )

        weight_builder = model_cls.get_weight_cls()(
            model_config=config,
            parallelism_config=parallelism_config,
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        weight_builder._process_meta([{}], set(checkpoint))
        layer_weights = (
            weight_builder._get_hf_ffn_layer_weight_info(0)
            if model_cls is Qwen2Moe
            else weight_builder._create_ffn_weight()
        )
        # Exercise production checkpoint lookup, merge/transpose and load.
        # Device-specific packing is outside this CPU loader contract.
        exported_device = SimpleNamespace(
            shuffle_moe_weight=lambda tensor, *_args: tensor,
            maybe_rewrite_weight_by_key=lambda _name, tensor: tensor,
        )
        with tempfile.TemporaryDirectory() as checkpoint_path:
            save_file(checkpoint, os.path.join(checkpoint_path, "model.safetensors"))
            database = CkptDatabase(checkpoint_path)
            source = DatabaseTensorSource(database)
            load_config = weight_builder.create_load_config(
                torch.float32, database, exported_device=exported_device
            )
            loaded = {}
            for weight in layer_weights:
                loaded.update(weight.load(source, 0, "cpu", load_config))

            if has_shared:
                self.assertEqual(
                    source.load_tensor(mlp + "shared_expert.gate_proj.weight")[0].shape,
                    (config.inter_size, config.hidden_size),
                )
            routed_key = mlp + (
                "experts.gate_up_proj" if stacked else "experts.0.gate_proj.weight"
            )
            self.assertEqual(
                source.load_tensor(routed_key)[0].shape,
                (
                    (config.expert_num, 2 * config.moe_inter_size, config.hidden_size)
                    if stacked
                    else (config.moe_inter_size, config.hidden_size)
                ),
            )

        self.assertEqual(
            loaded[W.moe_w1].shape, (config.expert_num, 2 * routed_width, 8)
        )
        self.assertEqual(loaded[W.moe_w2].shape, (config.expert_num, 8, routed_width))
        self.assertEqual(loaded[W.moe_gate].shape, (8, config.expert_num))
        if has_shared:
            self.assertEqual(loaded[W.ffn_w13].shape, (8, 2 * shared_width))
            self.assertEqual(loaded[W.ffn_w2].shape, (shared_width, 8))
            self.assertEqual(loaded[W.shared_expert_gate].shape, (8, 1))
            # Shared W13 merges gate|up; generic routed W1 loads up|gate.
            self.assertTrue(torch.all(loaded[W.ffn_w13][:, :shared_width] == 1))
            self.assertTrue(torch.all(loaded[W.ffn_w13][:, shared_width:] == 3))
        else:
            self.assertEqual(set(loaded), {W.moe_gate, W.moe_w1, W.moe_w2})
        self.assertTrue(torch.all(loaded[W.moe_w1][:, :routed_width] == 3))
        self.assertTrue(torch.all(loaded[W.moe_w1][:, routed_width:] == 1))

    def _assert_routed_only_weight_descriptors(self, weight_cls, config):
        weight = weight_cls.__new__(weight_cls)
        weight._align_size = 0
        weight._is_gated_activation = True
        weight.model_config = config
        weight.moe_layer_index_ = {1}
        weight.expert_num_ = config.expert_num
        weight.has_e_score_correction_bias = False

        layer_weights = weight._get_hf_ffn_layer_weight_info(1)

        self.assertEqual(len(layer_weights), 1)
        self.assertIsInstance(layer_weights[0], MoeWeight)
        self.assertFalse(any(isinstance(item, FfnWeight) for item in layer_weights))

        exported_device = SimpleNamespace(
            shuffle_moe_weight=lambda tensor, *_args: tensor,
            maybe_rewrite_weight_by_key=lambda _name, tensor: tensor,
        )
        load_config = SimpleNamespace(
            compute_dtype=torch.float32,
            merge_lora=False,
            moe_pure_tp_preshard=False,
            tp_size=1,
            dp_size=1,
            ep_size=1,
            exported_device=exported_device,
            get_selected_experts=lambda _layer_id, expert_num: range(expert_num),
        )
        source = _RoutedOnlyTensorSource()
        loaded = layer_weights[0].load(source, 1, "cpu", load_config)

        self.assertEqual(set(loaded), {W.moe_gate, W.moe_w1, W.moe_w2})
        self.assertTrue(source.requested_names)
        self.assertFalse(
            any("shared_experts" in name for name in source.requested_names)
        )

    def test_deepseek_v2_propagates_moe_dimensions(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as writer:
                json.dump(_deepseek_text_config(), writer)
            config = ModelConfig()
            DeepSeekV2._from_hf(config, ckpt_path)

        self.assertEqual(config.moe_inter_size, 1408)
        self.assertEqual(config.n_shared_experts, 2)
        self.assertEqual(config.inter_size, 2816)
        _assert_moe_config_is_constructible(self, config)

    def test_kimi_k25_propagates_moe_dimensions(self):
        config = ModelConfig()
        KimiK25._populate_text_config(config, _deepseek_text_config())

        self.assertEqual(config.moe_inter_size, 1408)
        self.assertEqual(config.n_shared_experts, 2)
        self.assertEqual(config.inter_size, 2816)
        _assert_moe_config_is_constructible(self, config)

    def test_deepseek_v2_routed_only_config_is_constructible(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as writer:
                json.dump(_deepseek_text_config(n_shared_experts=0), writer)
            config = ModelConfig()
            DeepSeekV2._from_hf(config, ckpt_path)

        self.assertEqual(config.moe_style, 1)
        self.assertEqual(config.n_shared_experts, 0)
        self.assertEqual(config.inter_size, 0)
        self.assertEqual(config.dense_inter_size, 11008)
        _assert_moe_config_is_constructible(self, config, 0)

    def test_routed_only_param_count_keeps_preceding_dense_layers(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as writer:
                json.dump(_deepseek_text_config(n_shared_experts=0), writer)
            config = ModelConfig()
            DeepSeekV2._from_hf(config, ckpt_path)

        moe_layer_count = len(config.moe_layer_index)
        dense_layer_count = config.num_layers - moe_layer_count
        ffn_w_count = 3 if config.isGatedActivation() else 2
        expected_moe_weights = (
            moe_layer_count
            * config.moe_inter_size
            * config.hidden_size
            * ffn_w_count
            * config.expert_num
        )
        self.assertEqual(config.moe_weight_param_count(), expected_moe_weights)

        count_with_dense_layers = config.layer_weight_param_count()
        config.dense_inter_size = 0
        count_without_dense_layers = config.layer_weight_param_count()
        self.assertEqual(
            count_with_dense_layers - count_without_dense_layers,
            dense_layer_count * 11008 * config.hidden_size * ffn_w_count,
        )

    def test_kimi_k25_routed_only_config_is_constructible(self):
        config = ModelConfig()
        KimiK25._populate_text_config(config, _deepseek_text_config(n_shared_experts=0))

        self.assertEqual(config.moe_style, 1)
        self.assertEqual(config.n_shared_experts, 0)
        self.assertEqual(config.inter_size, 0)
        self.assertEqual(config.dense_inter_size, 11008)
        _assert_moe_config_is_constructible(self, config, 0)

    def test_deepseek_v2_routed_only_weights_skip_shared_expert_keys(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as writer:
                json.dump(_deepseek_text_config(n_shared_experts=0), writer)
            config = ModelConfig()
            DeepSeekV2._from_hf(config, ckpt_path)

        self._assert_routed_only_weight_descriptors(DeepSeekV2Weight, config)

    def test_kimi_k25_routed_only_weights_skip_shared_expert_keys(self):
        config = ModelConfig()
        KimiK25._populate_text_config(config, _deepseek_text_config(n_shared_experts=0))

        self._assert_routed_only_weight_descriptors(KimiK25Weight, config)

    def test_qwen3_next_propagates_independently_sized_shared_expert(self):
        config = ModelConfig()
        config.num_layers = 2
        Qwen3Next._parse_moe_config(
            {
                "num_experts_per_tok": 8,
                "num_experts": 512,
                "moe_intermediate_size": 768,
                "shared_expert_intermediate_size": 2048,
                "decoder_sparse_step": 1,
            },
            config,
        )

        self._assert_independently_sized_qwen_shared_expert(
            config, Qwen3Next, 2048, 768
        )

    def test_qwen35_propagates_independently_sized_shared_expert(self):
        config = ModelConfig()
        config.num_layers = 2
        Qwen35Moe._parse_moe_config(
            {
                "num_experts_per_tok": 8,
                "num_experts": 256,
                "moe_intermediate_size": 768,
                "shared_expert_intermediate_size": 2048,
                "decoder_sparse_step": 1,
            },
            config,
        )

        for stacked in (False, True):
            with self.subTest(stacked=stacked):
                self._assert_independently_sized_qwen_shared_expert(
                    config, Qwen35Moe, 2048, 768, stacked=stacked
                )

    def test_qwen2_moe_propagates_independently_sized_shared_expert(self):
        config = ModelConfig()
        config.num_layers = 2
        Qwen2Moe.load_moe_config(
            config,
            {
                "num_experts_per_tok": 4,
                "num_experts": 60,
                "moe_intermediate_size": 1408,
                "shared_expert_intermediate_size": 5632,
                "decoder_sparse_step": 1,
            },
        )

        self._assert_independently_sized_qwen_shared_expert(
            config, Qwen2Moe, 5632, 1408
        )

    def test_qwen_routed_only_checkpoint_has_no_shared_expert_lookups(self):
        for model_cls in (Qwen2Moe, Qwen3Next, Qwen35Moe):
            for shared_width in (None, 0):
                with self.subTest(model=model_cls.__name__, shared_width=shared_width):
                    config = ModelConfig()
                    config.num_layers = 1
                    hf_config = {
                        "num_experts_per_tok": 1,
                        "num_experts": 2,
                        "moe_intermediate_size": 16,
                        "decoder_sparse_step": 1,
                    }
                    if shared_width is not None:
                        hf_config["shared_expert_intermediate_size"] = shared_width
                    if model_cls is Qwen2Moe:
                        model_cls.load_moe_config(config, hf_config)
                    else:
                        model_cls._parse_moe_config(hf_config, config)
                    layouts = (False, True) if model_cls is Qwen35Moe else (False,)
                    for stacked in layouts:
                        self._assert_independently_sized_qwen_shared_expert(
                            config, model_cls, 0, 16, stacked=stacked
                        )


if __name__ == "__main__":
    unittest.main()
