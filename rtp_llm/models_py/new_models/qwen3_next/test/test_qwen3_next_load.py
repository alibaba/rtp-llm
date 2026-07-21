import os
import tempfile
import types
import unittest

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.ops import DataType, HybridAttentionType

from rtp_llm.models_py.new_models.qwen3_next.language import (
    Qwen3NextForCausalLM,
    Qwen3NextGatedDeltaNet,
    reorder_ba,
    reorder_qkvz,
    reorder_qkvz_scale,
)


def _parallelism(
    tp_size=1,
    tp_rank=0,
    ep_size=1,
    ep_rank=0,
    attn_tp_size=None,
    attn_tp_rank=None,
    ffn_tp_size=None,
    ffn_tp_rank=None,
):
    attn_tp_size = tp_size if attn_tp_size is None else attn_tp_size
    attn_tp_rank = tp_rank if attn_tp_rank is None else attn_tp_rank
    ffn_tp_size = tp_size if ffn_tp_size is None else ffn_tp_size
    ffn_tp_rank = tp_rank if ffn_tp_rank is None else ffn_tp_rank
    return types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        dp_size=1,
        dp_rank=0,
        world_size=max(tp_size, ep_size),
        local_rank=tp_rank,
        get_attn_tp_size=lambda: attn_tp_size,
        get_attn_tp_rank=lambda: attn_tp_rank,
        get_ffn_tp_size=lambda: ffn_tp_size,
        get_ffn_tp_rank=lambda: ffn_tp_rank,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False,
            is_prefill_enabled=lambda: False,
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
    )


def _moe_config():
    return types.SimpleNamespace(
        fake_balance_expert=False,
        ll_num_max_token=1,
        masked_max_token_num=1,
        moe_strategy=0,
        use_mori_ep=False,
        use_deepep_moe=False,
        use_deepep_low_latency=False,
        use_all_gather=True,
    )


def _config(layer_type=HybridAttentionType.NONE, tie=True):
    config = ModelConfig()
    config.model_type = "qwen3_next"
    config.num_layers = 1
    config.vocab_size = 8
    config.hidden_size = 4
    config.inter_size = 4
    config.expert_num = 2
    config.moe_inter_size = 4
    config.moe_k = 1
    config.moe_style = 2
    config.moe_layer_index = []
    config.has_moe_norm = True
    config.attn_config.head_num = 2
    config.attn_config.kv_head_num = 1
    config.attn_config.size_per_head = 2
    config.layernorm_eps = 1e-6
    config.partial_rotary_factor = 1.0
    config.enable_fp32_lm_head = False
    config.tie_word_embeddings = tie
    config.data_type = "fp32"
    config.quant_config = None
    config.activation_type = "SiGLU"
    config.hybrid_attention_config.enable_hybrid_attention = True
    config.hybrid_attention_config.hybrid_attention_types = [layer_type]
    linear = config.linear_attention_config
    linear.linear_conv_kernel_dim = 2
    linear.linear_key_head_dim = 2
    linear.linear_num_key_heads = 2
    linear.linear_num_value_heads = 2
    linear.linear_value_head_dim = 2
    linear.ssm_state_dtype = DataType.TYPE_FP32
    linear.conv_state_dtype = DataType.TYPE_FP32
    return config


def _load_config(
    tp_size=1,
    tp_rank=0,
    ep_size=1,
    ep_rank=0,
    attn_tp_size=None,
    attn_tp_rank=None,
    ffn_tp_size=None,
    ffn_tp_rank=None,
    quant_config=None,
):
    attn_tp_size = tp_size if attn_tp_size is None else attn_tp_size
    attn_tp_rank = tp_rank if attn_tp_rank is None else attn_tp_rank
    ffn_tp_size = tp_size if ffn_tp_size is None else ffn_tp_size
    ffn_tp_rank = tp_rank if ffn_tp_rank is None else ffn_tp_rank
    parallelism = _parallelism(
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
        attn_tp_size,
        attn_tp_rank,
        ffn_tp_size,
        ffn_tp_rank,
    )
    return NewLoaderConfig(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
        ffn_tp_size=ffn_tp_size,
        ffn_tp_rank=ffn_tp_rank,
        compute_dtype=torch.float32,
        device="cpu",
        quant_config=quant_config or QuantizationConfig("none"),
        parallelism_config=parallelism,
        moe_config=_moe_config(),
    )


def _dense_weights(include_lm_head=False):
    weights = {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
            8, 4
        ),
        "model.layers.0.input_layernorm.weight": torch.ones(4),
        # Q and gate are interleaved per head: [head, q/gate, head_dim, hidden].
        "model.layers.0.self_attn.q_proj.weight": torch.arange(
            32, dtype=torch.float32
        ).reshape(8, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
        "model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(2),
        "model.layers.0.self_attn.k_norm.weight": torch.ones(2),
        "model.layers.0.self_attn.o_proj.weight": torch.eye(4),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
        "model.layers.0.mlp.gate_proj.weight": torch.ones(4, 4),
        "model.layers.0.mlp.up_proj.weight": torch.full((4, 4), 2.0),
        "model.layers.0.mlp.down_proj.weight": torch.eye(4),
        "model.norm.weight": torch.ones(4),
    }
    if include_lm_head:
        weights["lm_head.weight"] = torch.full((8, 4), 3.0)
    return weights


def _moe_weights():
    weights = _dense_weights()
    for name in (
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ):
        del weights[name]
    weights.update(
        {
            "model.layers.0.mlp.gate.weight": torch.ones(2, 4),
            "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.ones(4, 4),
            "model.layers.0.mlp.shared_expert.up_proj.weight": torch.full((4, 4), 2.0),
            "model.layers.0.mlp.shared_expert.down_proj.weight": torch.eye(4),
            "model.layers.0.mlp.shared_expert_gate.weight": torch.ones(1, 4),
        }
    )
    for expert_id in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert_id}"
        weights[f"{prefix}.gate_proj.weight"] = torch.full((4, 4), float(expert_id + 1))
        weights[f"{prefix}.up_proj.weight"] = torch.full((4, 4), float(expert_id + 2))
        weights[f"{prefix}.down_proj.weight"] = torch.eye(4)
    return weights


class Qwen3NextLoadTest(unittest.TestCase):
    def test_public_loader_streams_pytorch_checkpoint_and_runs_postprocess(self):
        config = _config(tie=False)
        config.logit_scale = 2.0
        weights = _dense_weights(include_lm_head=True)
        with tempfile.TemporaryDirectory() as model_path:
            torch.save(weights, os.path.join(model_path, "model.pt"))
            model = NewModelLoader(
                config,
                _load_config(),
                model_path=model_path,
            ).load()

        self.assertFalse(model.training)
        torch.testing.assert_close(
            model.lm_head.weight,
            torch.full_like(model.lm_head.weight, 6.0),
        )
        torch.testing.assert_close(model.norm.weight, torch.full((4,), 2.0))

    def test_typed_model_loads_q_gate_and_ties_lm_head(self):
        model = Qwen3NextForCausalLM(_config(), _load_config())
        weights = _dense_weights()
        model.load_weights(weights)
        NewModelLoader._validate_loaded_weights(model)

        q_gate = weights["model.layers.0.self_attn.q_proj.weight"].reshape(2, 2, 2, 4)
        expected_q = q_gate[:, 0].reshape(4, 4)
        expected_gate = q_gate[:, 1].reshape(4, 4)
        torch.testing.assert_close(
            model.layers[0].self_attn.qkv_proj.weight[:4], expected_q
        )
        torch.testing.assert_close(model.layers[0].self_attn.gate.weight, expected_gate)
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)
        torch.testing.assert_close(model.norm.weight, torch.full((4,), 2.0))

    def test_model_prefixed_lm_head_is_not_replaced_when_tied(self):
        model = Qwen3NextForCausalLM(_config(), _load_config())
        weights = _dense_weights()
        expected = torch.full((8, 4), 7.0)
        weights["model.lm_head.weight"] = expected
        model.load_weights(weights)
        NewModelLoader._validate_loaded_weights(model)

        torch.testing.assert_close(model.lm_head.weight, expected)
        self.assertFalse(torch.equal(model.lm_head.weight, model.embed_tokens.weight))

    def test_real_moe_model_tree_loads_router_shared_and_routed_experts(self):
        config = _config()
        config.moe_layer_index = [0]
        model = Qwen3NextForCausalLM(config, _load_config())
        model.load_weights(_moe_weights())
        NewModelLoader._validate_loaded_weights(model)

        moe = model.layers[0].mlp
        self.assertEqual(moe.experts._loaded_count, 6)
        self.assertIsNotNone(moe.select_topk)
        torch.testing.assert_close(moe.shared_expert_gate.weight, torch.ones(1, 4))
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)

    def test_untied_missing_lm_head_fails_integrity(self):
        model = Qwen3NextForCausalLM(_config(tie=False), _load_config())
        model.load_weights(_dense_weights())
        with self.assertRaisesRegex(RuntimeError, "ParallelLMHead.*weight"):
            NewModelLoader._validate_loaded_weights(model)

    def test_unknown_checkpoint_tensor_fails(self):
        model = Qwen3NextForCausalLM(_config(), _load_config())
        weights = _dense_weights()
        weights["model.layers.0.self_attn.q_projj.weight"] = torch.ones(4, 4)
        with self.assertRaisesRegex(RuntimeError, "q_projj"):
            model.load_weights(weights)

    def test_raw_hf_dict_is_rejected_before_model_construction(self):
        with self.assertRaisesRegex(TypeError, "typed ModelConfig"):
            Qwen3NextForCausalLM({}, _load_config())

    def test_hybrid_layer_count_must_match_model(self):
        config = _config()
        config.hybrid_attention_config.hybrid_attention_types = []
        with self.assertRaisesRegex(ValueError, "exactly one entry per layer"):
            Qwen3NextForCausalLM(config, _load_config())

    def test_unsupported_parallel_topologies_fail_before_loading(self):
        with self.assertRaisesRegex(ValueError, "Context parallelism"):
            Qwen3NextForCausalLM(
                _config(),
                _load_config(
                    tp_size=2,
                    attn_tp_size=1,
                    attn_tp_rank=0,
                ),
            )
        with self.assertRaisesRegex(ValueError, "Independent FFN TP"):
            Qwen3NextForCausalLM(
                _config(),
                _load_config(
                    tp_size=2,
                    ffn_tp_size=1,
                    ffn_tp_rank=0,
                ),
            )

    def test_invalid_ep_partition_and_mtp_fail_before_loading(self):
        config = _config()
        config.expert_num = 3
        with self.assertRaisesRegex(ValueError, "divisible by ep_size"):
            Qwen3NextForCausalLM(config, _load_config(ep_size=2))

        config = _config()
        config.is_mtp = True
        with self.assertRaisesRegex(ValueError, "MTP is not part"):
            Qwen3NextForCausalLM(config, _load_config())

    def test_q_gate_per_channel_scale_accepts_vector_layout(self):
        cfg = {"num_heads": 2, "head_dim": 2, "quant_config": None}
        transformed = dict(
            Qwen3NextForCausalLM._split_q_gate_yield(
                "model.layers.0.self_attn.q_proj.weight_scale",
                torch.arange(1, 9, dtype=torch.float32),
                cfg,
            )
        )
        torch.testing.assert_close(
            transformed["model.layers.0.self_attn.qkv_proj.q_proj.weight_scale"],
            torch.tensor([[1.0], [2.0], [5.0], [6.0]]),
        )
        torch.testing.assert_close(
            transformed["model.layers.0.self_attn.gate.weight_scale"],
            torch.tensor([[3.0], [4.0], [7.0], [8.0]]),
        )

    def test_q_gate_block_scale_requires_head_alignment(self):
        cfg = {
            "num_heads": 2,
            "head_dim": 64,
            "quant_config": types.SimpleNamespace(weight_block_size=[128, 128]),
        }
        with self.assertRaisesRegex(ValueError, "must align"):
            list(
                Qwen3NextForCausalLM._split_q_gate_yield(
                    "model.layers.0.self_attn.q_proj.weight_scale_inv",
                    torch.ones(2, 1),
                    cfg,
                )
            )

    def test_q_gate_block_scale_preserves_per_head_interleave(self):
        cfg = {
            "num_heads": 2,
            "head_dim": 128,
            "quant_config": types.SimpleNamespace(weight_block_size=[128, 128]),
        }
        scale = torch.arange(8, dtype=torch.float32).reshape(4, 2).add(1)
        transformed = dict(
            Qwen3NextForCausalLM._split_q_gate_yield(
                "model.layers.0.self_attn.q_proj.weight_scale_inv",
                scale,
                cfg,
            )
        )
        torch.testing.assert_close(
            transformed["model.layers.0.self_attn.qkv_proj.q_proj.weight_scale_inv"],
            scale[[0, 2]],
        )
        torch.testing.assert_close(
            transformed["model.layers.0.self_attn.gate.weight_scale_inv"],
            scale[[1, 3]],
        )


class Qwen3NextLinearAttentionLoadTest(unittest.TestCase):
    def _layer(self, tp_rank):
        config = _config(HybridAttentionType.LINEAR)
        return Qwen3NextGatedDeltaNet(
            linear_attn_config=config.linear_attention_config,
            hidden_size=4,
            rms_norm_eps=config.layernorm_eps,
            attn_tp_size=2,
            attn_tp_rank=tp_rank,
            params_dtype=torch.float32,
            quant_config=QuantizationConfig("none"),
            prefix="layers.0.linear_attn",
        )

    def _weights(self):
        return {
            "in_proj_qkvz.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4),
            "in_proj_ba.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "conv1d.weight": torch.arange(24, dtype=torch.float32).reshape(12, 2),
            "dt_bias": torch.tensor([1.0, 2.0]),
            "A_log": torch.tensor([3.0, 4.0]),
            "norm.weight": torch.tensor([5.0, 6.0]),
            "out_proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        }

    def test_tp_ranks_receive_qkvz_ba_and_out_slices(self):
        config = _config(HybridAttentionType.LINEAR).linear_attention_config
        weights = self._weights()
        reordered = reorder_qkvz([weights["in_proj_qkvz.weight"]], config)
        q, k, v, z = torch.split(reordered, [4, 4, 4, 4], dim=0)
        ba = reorder_ba([weights["in_proj_ba.weight"]], config)
        b, a = torch.split(ba, [2, 2], dim=0)

        for rank in range(2):
            layer = self._layer(rank)
            layer.load_weights(weights)
            NewModelLoader._validate_loaded_weights(layer)
            expected_qkvz = torch.cat(
                [
                    q[rank * 2 : rank * 2 + 2],
                    k[rank * 2 : rank * 2 + 2],
                    v[rank * 2 : rank * 2 + 2],
                    z[rank * 2 : rank * 2 + 2],
                ],
                dim=0,
            )
            expected_ba = torch.cat([b[rank : rank + 1], a[rank : rank + 1]], dim=0).t()
            torch.testing.assert_close(layer.in_proj_qkvz.weight, expected_qkvz)
            torch.testing.assert_close(layer.in_proj_ba_w, expected_ba)
            torch.testing.assert_close(
                layer.out_proj.weight,
                weights["out_proj.weight"][:, rank * 2 : rank * 2 + 2],
            )
            torch.testing.assert_close(
                layer.dt_bias, weights["dt_bias"][rank : rank + 1]
            )
            torch.testing.assert_close(layer.a_log, weights["A_log"][rank : rank + 1])

    def test_unknown_linear_tensor_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            self._layer(0).load_weights({"in_proj_qkvz.typo": torch.ones(1)})

    def test_linear_attention_rejects_cpu_runtime_before_migration(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA or ROCm"):
            self._layer(0).validate_runtime_device(torch.device("cpu"))

    def test_linear_attention_uses_configured_rms_norm_epsilon(self):
        config = _config(HybridAttentionType.LINEAR)
        config.layernorm_eps = 3e-5
        layer = Qwen3NextGatedDeltaNet(
            linear_attn_config=config.linear_attention_config,
            hidden_size=4,
            rms_norm_eps=config.layernorm_eps,
            attn_tp_size=2,
            attn_tp_rank=0,
            params_dtype=torch.float32,
            quant_config=QuantizationConfig("none"),
            prefix="layers.0.linear_attn",
        )
        self.assertEqual(layer.rms_norm_eps, 3e-5)

    def test_qkvz_per_channel_scale_accepts_vector_layout(self):
        config = _config(HybridAttentionType.LINEAR)
        layer = Qwen3NextGatedDeltaNet(
            linear_attn_config=config.linear_attention_config,
            hidden_size=4,
            rms_norm_eps=config.layernorm_eps,
            attn_tp_size=2,
            attn_tp_rank=1,
            params_dtype=torch.float32,
            quant_config=QuantizationConfig("fp8_per_channel"),
            prefix="layers.0.linear_attn",
        )
        scale = torch.arange(1, 17, dtype=torch.float32)
        layer.load_weights({"in_proj_qkvz.weight_scale": scale})
        reordered = reorder_qkvz([scale], config.linear_attention_config)
        expected = layer._split_qkvz_rows(reordered, 1)
        torch.testing.assert_close(layer.in_proj_qkvz.weight_scale, expected)

    def test_fp8_block_scales_follow_qkvz_and_out_projection_tp_layouts(self):
        config = _config(HybridAttentionType.LINEAR)
        linear = config.linear_attention_config
        linear.linear_key_head_dim = 128
        linear.linear_value_head_dim = 128
        quant = QuantizationConfig(
            "fp8_block",
            source_config=types.SimpleNamespace(weight_block_size=[128, 128]),
        )
        layer = Qwen3NextGatedDeltaNet(
            linear_attn_config=linear,
            hidden_size=256,
            rms_norm_eps=config.layernorm_eps,
            attn_tp_size=2,
            attn_tp_rank=1,
            params_dtype=torch.float32,
            quant_config=quant,
            prefix="layers.0.linear_attn",
        )
        qkvz_scale = torch.arange(16, dtype=torch.float32).reshape(8, 2).add(1)
        out_scale = torch.arange(4, dtype=torch.float32).reshape(2, 2).add(1)
        layer.load_weights(
            {
                "in_proj_qkvz.weight_scale_inv": qkvz_scale,
                "out_proj.weight_scale_inv": out_scale,
            }
        )

        reordered = reorder_qkvz_scale(qkvz_scale, linear, block_n=128)
        expected_qkvz = layer._split_qkvz_rows(reordered, 128)
        torch.testing.assert_close(
            layer.in_proj_qkvz.weight_scale_inv,
            expected_qkvz,
        )
        torch.testing.assert_close(
            layer.out_proj.weight_scale_inv,
            out_scale[:, 1:2],
        )

    def test_single_tp_out_scale_allows_partial_final_block(self):
        config = _config(HybridAttentionType.LINEAR)
        linear = config.linear_attention_config
        linear.linear_key_head_dim = 96
        linear.linear_value_head_dim = 96
        layer = Qwen3NextGatedDeltaNet(
            linear_attn_config=linear,
            hidden_size=192,
            rms_norm_eps=config.layernorm_eps,
            attn_tp_size=1,
            attn_tp_rank=0,
            params_dtype=torch.float32,
            quant_config=QuantizationConfig("none"),
            prefix="layers.0.linear_attn",
        )
        layer.out_proj.quant_config = types.SimpleNamespace(
            weight_block_size=[128, 128]
        )
        scale = torch.arange(4, dtype=torch.float32).reshape(2, 2).add(1)
        torch.testing.assert_close(layer._split_out_block_scale(scale), scale)


if __name__ == "__main__":
    unittest.main()
