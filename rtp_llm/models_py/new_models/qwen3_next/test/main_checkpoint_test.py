import os
import tempfile
import types
import unittest
import torch
from safetensors.torch import save_file
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.qwen3_next.language import Qwen35DenseMTPForCausalLM
from rtp_llm.models_py.new_models.qwen3_next.language import Qwen3NextMoEBlock
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.ops import DataType, HybridAttentionType


def _parallelism(
    tp_size=1,
    tp_rank=0,
    ep_size=1,
    ep_rank=0,
    attn_tp_size=None,
    attn_tp_rank=None,
    ffn_tp_size=None,
    ffn_tp_rank=None,
    cp_enabled=False,
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
        pp_size=1,
        pp_rank=0,
        pp_stage_layer_counts=[],
        world_rank=tp_rank,
        dp_size=1,
        dp_rank=0,
        world_size=max(tp_size, ep_size),
        local_rank=tp_rank,
        get_attn_tp_size=lambda: attn_tp_size,
        get_attn_tp_rank=lambda: attn_tp_rank,
        get_ffn_tp_size=lambda: ffn_tp_size,
        get_ffn_tp_rank=lambda: ffn_tp_rank,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: cp_enabled,
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
    cp_enabled=False,
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
        cp_enabled,
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
        load_method="scratch",
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


def _mtp_config(model_type="qwen35_moe_mtp"):
    config = _config(tie=False)
    config.model_type = model_type
    config.is_mtp = True
    config.moe_layer_index = [0]
    config.hybrid_attention_config.hybrid_attention_types = [HybridAttentionType.NONE]
    return config


class Qwen3NextLoadTest(unittest.TestCase):
    def test_shared_expert_fuses_gate_and_finalizes_only_once(self):
        # Compare BF16 intermediate rounding with the one-store gate of the
        # inference runtime. Use CPU stand-ins for just the CUDA kernel and TP
        # collective so both precision and reduction order are observable.
        block = object.__new__(Qwen3NextMoEBlock)
        torch.nn.Module.__init__(block)
        block.tp_size, block.ep_size = 2, 1
        block.use_ep_shared_allreduce = False
        block.experts = types.SimpleNamespace(fused_moe=types.SimpleNamespace(
            router=types.SimpleNamespace(tp_collective_size=2, supports_skip_tp_allreduce=True)))
        calls = []
        hidden = torch.linspace(-2, 2, 64).reshape(8, 8).bfloat16()
        gates = hidden[:, :1] * 0.625
        shared = hidden * 2.1
        routed = hidden.flip(1)
        remote = hidden * 0.31
        def route(_):
            calls.append(("routed", block.use_unified_tp_allreduce))
            return routed.clone()
        def shared_expert(_, skip_allreduce=False):
            calls.append(("shared", skip_allreduce))
            return shared
        def fused(gate, shard, out):
            calls.append("merge")
            out.copy_(torch.addcmul(out.float(), shard.float(), gate.float().sigmoid()))
        def reduce(partial, group):
            calls.append("allreduce")
            return partial + remote
        block._forward_routed = route
        block.shared_expert = shared_expert
        block.shared_expert_gate = lambda _: gates
        block.sigmoid_gate_scale_add = fused
        block.tp_all_reduce = reduce
        expected_partial = torch.addcmul(routed.float(), shared.float(), gates.float().sigmoid()).bfloat16()
        self.assertFalse(torch.equal(expected_partial, routed + gates.sigmoid() * shared))
        self.assertTrue(torch.equal(block(hidden), expected_partial + remote))
        self.assertEqual(calls, [("routed", True), ("shared", True), "merge", "allreduce"])
        block.experts.fused_moe.router.supports_skip_tp_allreduce = False
        calls.clear()
        self.assertTrue(torch.equal(block(hidden), expected_partial))
        self.assertEqual(calls, [("routed", False), ("shared", False), "merge"])

    def test_dense_mtp_real_checkpoint_load_uses_dense_mlp(self):
        config = _mtp_config("qwen35_dense_mtp")
        config.moe_layer_index = []
        config.expert_num = 0
        config.moe_k = 0
        config.moe_style = 0
        weights = {}
        for name, tensor in _dense_weights(include_lm_head=True).items():
            if name == "model.embed_tokens.weight":
                weights["model.language_model.embed_tokens.weight"] = tensor
            elif name.startswith("model."):
                weights["mtp." + name[len("model.") :]] = tensor
            else:
                weights[name] = tensor
        weights.update(
            {
                "mtp.pre_fc_norm_embedding.weight": torch.ones(4),
                "mtp.pre_fc_norm_hidden.weight": torch.full((4,), 2.0),
                "mtp.fc.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            save_file(weights, os.path.join(directory, "model.safetensors"))
            model = NewModelLoader(config, _load_config(), directory).load()
        self.assertIsInstance(model, Qwen35DenseMTPForCausalLM)
        self.assertEqual(model._cfg["moe_layer_index"], [])
        self.assertFalse(any("experts" in name for name, _ in model.named_parameters()))
        torch.testing.assert_close(model.fc.weight, weights["mtp.fc.weight"])
        torch.testing.assert_close(
            model.pre_fc_norm_embedding.weight, torch.full((4,), 2.0)
        )
        torch.testing.assert_close(
            model.pre_fc_norm_hidden.weight, torch.full((4,), 3.0)
        )
        self.assertIs(
            model.runtime_weight_view()["embedding"], model.embed_tokens.weight
        )

    def test_qwen35_dense_safetensors_loads_without_synthetic_moe_config(self):
        config = _config(tie=False)
        config._checkpoint_num_layers_before_override = 2
        config.model_type = "qwen35_dense"
        config.expert_num = config.moe_k = config.moe_inter_size = config.moe_style = 0
        weights = {
            name.replace("model.", "model.language_model.", 1): tensor
            for name, tensor in _dense_weights(include_lm_head=True).items()
        }
        with tempfile.TemporaryDirectory() as model_path:
            weights["model.language_model.layers.1.input_layernorm.weight"] = torch.ones(4)
            weights["model.language_model.visual.weight"] = torch.arange(8).float().reshape(2, 4)
            save_file(weights, os.path.join(model_path, "model.safetensors"))
            from rtp_llm.models_py.new_models.qwen_service_vision import load_service_vision

            for method in ("scratch", "fastsafetensors"):
                visual = load_service_vision(
                    torch.nn.Linear(4, 2, bias=False), model_path, torch.float32,
                    "cpu", method, prefix=None,
                )
                torch.testing.assert_close(visual.weight, weights["model.language_model.visual.weight"])
            for rank in range(2):
                with self.subTest(rank=rank):
                    model = NewModelLoader(
                        config,
                        _load_config(tp_size=2, tp_rank=rank),
                        model_path=model_path,
                    ).load()
                    NewModelLoader._validate_loaded_weights(model)
                    self.assertTrue(model.owns_lora_checkpoint_target("layers.0.self_attn.q_proj.weight"))
                    self.assertFalse(model.owns_lora_checkpoint_target("layers.1.self_attn.q_proj.weight"))
                    # Keys outside even the source geometry must still reach strict validation.
                    self.assertTrue(model.checkpoint_weight_name_filter()("model.language_model.layers.9.bad"))
                    self.assertTrue(model.checkpoint_weight_name_filter()("model.language_model.visual_adapter.unknown"))
                    self.assertEqual(model._cfg["expert_num"], 0)
                    self.assertFalse(hasattr(model.layers[0].mlp, "experts"))
                    q_gate = weights[
                        "model.language_model.layers.0.self_attn.q_proj.weight"
                    ].reshape(2, 2, 2, 4)
                    torch.testing.assert_close(
                        model.layers[0].self_attn.qkv_proj.weight[:2], q_gate[rank, 0]
                    )
                    torch.testing.assert_close(
                        model.layers[0].self_attn.gate.weight, q_gate[rank, 1]
                    )
                    torch.testing.assert_close(model.norm.weight, torch.full((4,), 2.0))
                    torch.testing.assert_close(
                        model.lm_head.weight, torch.full((4, 4), 3.0)
                    )


if __name__ == "__main__":
    unittest.main()
