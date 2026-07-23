import types
import unittest
from unittest import mock

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewModelLoader,
    _ExpertRangeFilter,
)
from rtp_llm.models_py.modules import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.new_models.qwen3_vl_moe import (
    Qwen3VLMoeForCausalLM as ExportedQwen3VLMoeForCausalLM,
)
from rtp_llm.models_py.new_models.qwen3_vl_moe.model import Qwen3VLMoeForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class


def _parallelism(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0):
    return types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        dp_size=1,
        dp_rank=0,
        world_size=max(tp_size, ep_size),
        local_rank=tp_rank,
        get_attn_tp_size=lambda: tp_size,
        get_attn_tp_rank=lambda: tp_rank,
        get_ffn_tp_size=lambda: tp_size,
        get_ffn_tp_rank=lambda: tp_rank,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False,
            is_prefill_enabled=lambda: False,
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
    )


def _moe_config():
    return types.SimpleNamespace(
        ll_num_max_token=1,
        masked_max_token_num=1,
        moe_strategy=0,
        use_mori_ep=False,
        use_deepep_moe=False,
        use_deepep_low_latency=False,
        use_all_gather=True,
        fake_balance_expert=False,
    )


def _model_config(num_experts=2, tie_word_embeddings=False):
    config = ModelConfig()
    config.model_type = "qwen3_vl_moe"
    config.num_layers = 1
    config.vocab_size = 8
    config.hidden_size = 4
    config.inter_size = 0
    config.expert_num = num_experts
    config.moe_inter_size = 3
    config.moe_k = 1
    config.moe_topk_group = 1
    config.attn_config.head_num = 2
    config.attn_config.kv_head_num = 1
    config.attn_config.size_per_head = 2
    config.layernorm_eps = 1e-6
    config.enable_fp32_lm_head = False
    config.tie_word_embeddings = tie_word_embeddings
    config.data_type = "fp32"
    config.quant_config = None
    config.activation_type = "SiGLU"
    return config


def _load_config(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0):
    return NewLoaderConfig(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        compute_dtype=torch.float32,
        device="cpu",
        quant_config=QuantizationConfig("none"),
        parallelism_config=_parallelism(tp_size, tp_rank, ep_size, ep_rank),
        moe_config=_moe_config(),
    )


def _weights(num_experts=2):
    prefix = "model.language_model."
    weights = {
        prefix
        + "embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        prefix + "layers.0.input_layernorm.weight": torch.ones(4),
        prefix + "layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
        prefix + "layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
        prefix + "layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
        prefix + "layers.0.self_attn.q_norm.weight": torch.ones(2),
        prefix + "layers.0.self_attn.k_norm.weight": torch.ones(2),
        prefix + "layers.0.self_attn.o_proj.weight": torch.eye(4),
        prefix + "layers.0.post_attention_layernorm.weight": torch.ones(4),
        prefix + "layers.0.mlp.gate.weight": torch.ones(num_experts, 4),
        prefix
        + "layers.0.mlp.experts.gate_up_proj": torch.arange(
            num_experts * 6 * 4, dtype=torch.float32
        ).reshape(num_experts, 6, 4),
        prefix
        + "layers.0.mlp.experts.down_proj": torch.arange(
            num_experts * 4 * 3, dtype=torch.float32
        ).reshape(num_experts, 4, 3),
        prefix + "norm.weight": torch.ones(4),
        "lm_head.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
    }
    return weights


class Qwen3VLMoeNewLoaderTest(unittest.TestCase):
    def test_package_export_and_registry_resolve_language_model(self):
        self.assertIs(ExportedQwen3VLMoeForCausalLM, Qwen3VLMoeForCausalLM)
        self.assertIs(get_model_class("qwen3_vl_moe"), Qwen3VLMoeForCausalLM)

    def test_prefixed_stacked_checkpoint_loads_complete_model(self):
        model = Qwen3VLMoeForCausalLM(_model_config(), _load_config())
        checkpoint = _weights()
        checkpoint["model.visual.unused.weight"] = torch.ones(1)
        name_filter = model.checkpoint_weight_name_filter()
        model.load_weights(
            (name, tensor) for name, tensor in checkpoint.items() if name_filter(name)
        )
        NewModelLoader._validate_loaded_weights(model)

        self.assertEqual(model.layers[0].mlp.experts._loaded_count, 6)
        torch.testing.assert_close(
            model.lm_head.weight,
            checkpoint["lm_head.weight"],
        )

    def test_unknown_and_missing_language_tensors_are_not_ignored(self):
        model = Qwen3VLMoeForCausalLM(_model_config(), _load_config())
        unknown = _weights()
        unknown["model.language_model.layers.0.typo.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "typo"):
            model.load_weights(unknown)

        model = Qwen3VLMoeForCausalLM(_model_config(), _load_config())
        missing = _weights()
        del missing["model.language_model.layers.0.self_attn.q_proj.weight"]
        model.load_weights(missing)
        with self.assertRaisesRegex(RuntimeError, r"qkv_proj.*weight"):
            NewModelLoader._validate_loaded_weights(model)

    def test_ep_expands_only_prefixed_language_experts(self):
        model = Qwen3VLMoeForCausalLM(
            _model_config(num_experts=4),
            _load_config(ep_size=2, ep_rank=1),
        )
        model_filter = model.checkpoint_weight_name_filter()
        expert_filter = _ExpertRangeFilter(4, 2, 1)
        combined_filter = NewModelLoader._checkpoint_name_filter(
            model_filter, expert_filter
        )
        checkpoint_name = "model.language_model.layers.0.mlp.experts.gate_up_proj"

        self.assertTrue(combined_filter(checkpoint_name))
        self.assertTrue(expert_filter.handles_safetensor(checkpoint_name))
        expanded = expert_filter.expand_safetensor(checkpoint_name, (4, 6, 4))
        self.assertEqual(
            [
                (model.WEIGHTS_MAPPER.map_name(name), expert_id)
                for name, expert_id in expanded
            ],
            [
                ("layers.0.mlp.experts.2.gate_up_proj.weight", 2),
                ("layers.0.mlp.experts.3.gate_up_proj.weight", 3),
            ],
        )
        self.assertFalse(
            combined_filter("model.visual.layers.0.mlp.experts.gate_up_proj")
        )

    def test_tp_and_ep_owners_follow_moe_topology(self):
        model = Qwen3VLMoeForCausalLM(
            _model_config(num_experts=4),
            _load_config(tp_size=2, tp_rank=1, ep_size=2, ep_rank=1),
        )

        self.assertEqual(
            (model.embed_tokens.tp_size, model.embed_tokens.tp_rank), (2, 1)
        )
        self.assertEqual(
            (
                model.layers[0].self_attn.qkv_proj.tp_size,
                model.layers[0].self_attn.qkv_proj.tp_rank,
            ),
            (2, 1),
        )
        experts = model.layers[0].mlp.experts
        self.assertEqual(
            (experts.moe_expert_tp_size, experts.moe_expert_tp_rank), (1, 0)
        )
        self.assertEqual((experts.ep_size, experts.ep_rank), (2, 1))

    def test_forward_masks_image_token_ids_and_preserves_residual_contract(self):
        class ResidualLayer(torch.nn.Module):
            def __init__(self, hidden_delta, residual_delta):
                super().__init__()
                self.hidden_delta = hidden_delta
                self.residual_delta = residual_delta

            def forward(self, hidden_states, residual, fmha_impl, kv_cache=None):
                return (
                    hidden_states + self.hidden_delta,
                    residual + self.residual_delta,
                )

        class FinalNorm(torch.nn.Module):
            def forward(self, hidden_states, residual):
                return hidden_states + residual, residual

        model = Qwen3VLMoeForCausalLM.__new__(Qwen3VLMoeForCausalLM)
        torch.nn.Module.__init__(model)
        model.embed_tokens = torch.nn.Embedding(8, 4)
        model.embed_tokens.weight.data.copy_(
            torch.arange(32, dtype=torch.float32).reshape(8, 4)
        )
        model.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        model.multimodal_deepstack_injector = MultimodalDeepstackInjector()
        model.layers = torch.nn.ModuleList(
            [ResidualLayer(1.0, 10.0), ResidualLayer(2.0, 20.0)]
        )
        model.norm = FinalNorm()
        model.kv_cache = None

        input_ids = torch.tensor([1, 100, 2])
        image_feature = torch.full((1, 4), 50.0)
        deepstack = torch.stack((torch.full((1, 4), 100.0), torch.full((1, 4), 200.0)))
        inputs = types.SimpleNamespace(
            input_ids=input_ids,
            embedding_inputs=types.SimpleNamespace(
                text_tokens_mask=torch.tensor([True, False, True])
            ),
            multimodal_inputs=types.SimpleNamespace(
                multimodal_features=[image_feature],
                mm_features_locs=torch.tensor([1]),
                mm_extra_input=[deepstack.flatten()],
            ),
            attention_inputs=object(),
        )
        fmha_impl = types.SimpleNamespace(fmha_params=None)

        with mock.patch(
            "rtp_llm.models_py.new_models.qwen3_vl_moe.model."
            "select_block_map_for_layer"
        ) as select_block:
            output = model(inputs, fmha_impl)

        expected = model.embed_tokens(torch.tensor([1, 0, 2]))
        expected[1] = image_feature[0]
        expected = expected + 1.0
        expected[1] += 100.0
        expected = expected + 2.0
        expected[1] += 200.0
        expected = expected + 30.0
        torch.testing.assert_close(output.hidden_states, expected)
        self.assertEqual(
            [call.args for call in select_block.call_args_list],
            [(inputs.attention_inputs, 0), (inputs.attention_inputs, 1)],
        )

    def test_forward_handles_text_only_input(self):
        class IdentityLayer(torch.nn.Module):
            def forward(self, hidden_states, residual, fmha_impl, kv_cache=None):
                return hidden_states, residual

        class FinalNorm(torch.nn.Module):
            def forward(self, hidden_states, residual):
                return hidden_states + residual, residual

        model = Qwen3VLMoeForCausalLM.__new__(Qwen3VLMoeForCausalLM)
        torch.nn.Module.__init__(model)
        model.embed_tokens = torch.nn.Embedding(8, 4)
        model.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        model.multimodal_deepstack_injector = MultimodalDeepstackInjector()
        model.layers = torch.nn.ModuleList([IdentityLayer()])
        model.norm = FinalNorm()
        model.kv_cache = None
        input_ids = torch.tensor([1, 2, 3])
        inputs = types.SimpleNamespace(
            input_ids=input_ids,
            embedding_inputs=types.SimpleNamespace(text_tokens_mask=None),
            multimodal_inputs=types.SimpleNamespace(
                multimodal_features=[],
                mm_features_locs=torch.empty(0, dtype=torch.long),
                mm_extra_input=[],
            ),
            attention_inputs=object(),
        )
        fmha_impl = types.SimpleNamespace(fmha_params=None)

        with mock.patch(
            "rtp_llm.models_py.new_models.qwen3_vl_moe.model."
            "select_block_map_for_layer"
        ):
            output = model(inputs, fmha_impl)

        torch.testing.assert_close(output.hidden_states, model.embed_tokens(input_ids))


if __name__ == "__main__":
    unittest.main()
