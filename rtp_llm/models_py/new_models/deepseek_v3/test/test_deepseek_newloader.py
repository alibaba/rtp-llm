import json
import math
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type, is_cuda, is_hip
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v3.attention import (
    DeepSeekV32MlaAttention,
    _CudaRuntimeFusedFp8Linear,
    _kernel_fp8_weight_and_scale,
    _linear_weight_bf16,
    _prepare_fused_fp8_runtime_weight,
)
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    DeepSeekV32ForCausalLM,
    MlaRuntimeLayoutMixin,
    build_rope_cache,
    extract_config_values,
)
from rtp_llm.models_py.new_models.deepseek_v3.mlp import DeepSeekV32MLP
from rtp_llm.models_py.new_models.deepseek_v3.model import (
    DeepSeekV32DecoderLayer,
    DeepSeekV32Indexer,
)
from rtp_llm.models_py.new_models.deepseek_v3.moe import (
    DeepSeekV32MoEBlock,
    _select_deepseek_noaux_topk,
    _select_deepseek_topk,
)
from rtp_llm.models_py.new_models.deepseek_v3.rotary_embedding import (
    DeepseekV3RotaryEmbedding,
)
from rtp_llm.models_py.new_models.deepseek_v3_mtp.language import (
    DeepSeekV32MTPForCausalLM,
    _draft_checkpoint_layer,
    _remap_key,
)
from rtp_llm.models_py.new_models.mtp import MTPBlock
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.ops import EplbMode, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs
from rtp_llm.utils.model_weight import W


def _model_config(*, sparse=False, q_lora_rank=4):
    config = ModelConfig()
    config.model_type = "deepseek3"
    config.hidden_size = 8
    config.num_layers = 4
    config.vocab_size = 16
    config.max_seq_len = 128
    config.layernorm_eps = 1e-6
    config.expert_num = 8
    config.moe_k = 2
    config.scoring_func = 1
    config.routed_scaling_factor = 1.0
    config.moe_n_group = 2
    config.moe_topk_group = 1
    config.has_moe_norm = True
    config.enable_fp32_lm_head = False
    config.tie_word_embeddings = False
    config.attn_config.head_num = 4
    config.attn_config.q_lora_rank = q_lora_rank
    config.attn_config.kv_lora_rank = 4
    config.attn_config.nope_head_dim = 2
    config.attn_config.rope_head_dim = 2
    config.attn_config.v_head_dim = 2
    config.attn_config.use_mla = True
    config.attn_config.is_sparse = sparse
    config.attn_config.indexer_head_dim = 4
    config.attn_config.indexer_head_num = 2
    config.attn_config.indexer_topk = 8
    config.attn_config.kernel_tokens_per_block = 64
    return config


def _raw_config():
    return {
        "architectures": ["DeepseekV3ForCausalLM"],
        "num_hidden_layers": 4,
        "num_nextn_predict_layers": 1,
        "intermediate_size": 16,
        "moe_intermediate_size": 8,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
        "n_group": 2,
        "topk_group": 1,
        "norm_topk_prob": True,
        "topk_method": "greedy",
        "rope_interleave": True,
        "indexer_rope_interleave": False,
        "qk_rope_head_dim": 2,
    }


def _load_config():
    return NewLoaderConfig(
        tp_size=2,
        tp_rank=1,
        ep_size=2,
        ep_rank=1,
        attn_tp_size=1,
        attn_tp_rank=0,
        ffn_tp_size=2,
        ffn_tp_rank=1,
        lm_head_tp_size=2,
        lm_head_tp_rank=1,
        compute_dtype=torch.float32,
        device="cpu",
    )


def _router_model_config(
    *,
    hidden_size=8,
    expert_num=4,
    moe_k=2,
    has_moe_norm=False,
    moe_n_group=1,
    moe_topk_group=1,
    scoring_func=1,
    routed_scaling_factor=1.0,
):
    config = ModelConfig()
    config.attn_config.head_num = 1
    config.attn_config.size_per_head = 128
    config.num_layers = 1
    config.max_seq_len = 128
    config.vocab_size = 16
    config.hidden_size = hidden_size
    config.expert_num = expert_num
    config.moe_k = moe_k
    config.has_moe_norm = has_moe_norm
    config.moe_n_group = moe_n_group
    config.moe_topk_group = moe_topk_group
    config.scoring_func = scoring_func
    config.routed_scaling_factor = routed_scaling_factor
    config.quant_config = None
    return config


def _dense_loader_model_config(checkpoint_path):
    config = _model_config()
    config.ckpt_path = checkpoint_path
    config.num_layers = 1
    config.expert_num = 0
    config.moe_k = 0
    config.tie_word_embeddings = True
    return config


def _dense_loader_config_json():
    config = _raw_config()
    config.update(
        {
            "num_hidden_layers": 1,
            "num_nextn_predict_layers": 0,
            "first_k_dense_replace": 1,
            "n_shared_experts": 0,
            "tie_word_embeddings": True,
        }
    )
    return config


def _deterministic_values(shape, offset):
    count = math.prod(shape)
    return (
        torch.arange(offset, offset + count, dtype=torch.float32).reshape(shape)
        / max(count, 1)
        - 0.5
    )


def _manual_noaux_reference(
    logits,
    correction_bias,
    *,
    top_k,
    n_group,
    topk_group,
    renormalize,
    routed_scaling_factor,
):
    """Scalar reference intentionally independent of the vectorized router."""
    scores = logits.float().sigmoid().tolist()
    biases = correction_bias.float().tolist()
    group_size = len(biases) // n_group
    all_weights = []
    all_ids = []
    for token_scores in scores:
        choice_scores = [score + bias for score, bias in zip(token_scores, biases)]
        group_scores = []
        for group_id in range(n_group):
            begin = group_id * group_size
            group_values = choice_scores[begin : begin + group_size]
            group_scores.append(sum(sorted(group_values, reverse=True)[:2]))
        selected_groups = set(
            sorted(range(n_group), key=group_scores.__getitem__, reverse=True)[
                :topk_group
            ]
        )
        candidates = [
            expert_id
            for expert_id in range(len(token_scores))
            if expert_id // group_size in selected_groups
        ]
        selected_experts = sorted(
            candidates,
            key=choice_scores.__getitem__,
            reverse=True,
        )[:top_k]
        token_weights = [token_scores[expert_id] for expert_id in selected_experts]
        if renormalize and top_k > 1:
            denominator = max(sum(token_weights), 1e-20)
            token_weights = [weight / denominator for weight in token_weights]
        all_ids.append(selected_experts)
        all_weights.append([weight * routed_scaling_factor for weight in token_weights])
    return (
        torch.tensor(all_weights, dtype=torch.float32, device=logits.device),
        torch.tensor(all_ids, dtype=torch.int64, device=logits.device),
    )


def _dense_checkpoint_weights():
    values = _deterministic_values
    return {
        "model.embed_tokens.weight": values((16, 8), 0),
        "model.layers.0.input_layernorm.weight": values((8,), 3) + 1.0,
        "model.layers.0.self_attn.q_a_proj.weight": values((4, 8), 5),
        "model.layers.0.self_attn.q_a_layernorm.weight": values((4,), 7) + 1.0,
        "model.layers.0.self_attn.q_b_proj.weight": values((16, 4), 11),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": values((6, 8), 13),
        "model.layers.0.self_attn.kv_a_layernorm.weight": values((4,), 17) + 1.0,
        "model.layers.0.self_attn.kv_b_proj.weight": values((16, 4), 19),
        "model.layers.0.self_attn.o_proj.weight": values((8, 8), 23),
        "model.layers.0.post_attention_layernorm.weight": values((8,), 29) + 1.0,
        "model.layers.0.mlp.gate_proj.weight": values((16, 8), 31),
        "model.layers.0.mlp.up_proj.weight": values((16, 8), 37),
        "model.layers.0.mlp.down_proj.weight": values((8, 16), 41),
        "model.norm.weight": values((8,), 43) + 1.0,
    }


def _mtp_loader_model_config(checkpoint_path):
    config = _model_config()
    config.model_type = "deepseek-v3-mtp"
    config.ckpt_path = checkpoint_path
    config.num_layers = 1
    config.expert_num = 2
    config.moe_k = 1
    config.moe_n_group = 1
    config.moe_topk_group = 1
    config.data_type = "fp32"
    config.activation_type = "SiGLU"
    return config


def _single_rank_parallelism_config():
    config = ParallelismConfig()
    config.tp_size = 1
    config.tp_rank = 0
    config.ep_size = 1
    config.ep_rank = 0
    config.dp_size = 1
    config.dp_rank = 0
    config.world_size = 1
    config.world_rank = 0
    config.local_rank = 0
    config.local_world_size = 1
    return config


def _mtp_loader_config_json():
    config = _raw_config()
    config.update(
        {
            "num_hidden_layers": 1,
            "num_nextn_predict_layers": 1,
            "n_shared_experts": 1,
            "n_group": 1,
            "topk_group": 1,
        }
    )
    return config


def _mtp_checkpoint_weights():
    values = _deterministic_values
    prefix = "model.layers.1."
    weights = {
        prefix + "embed_tokens.weight": values((16, 8), 0),
        prefix + "enorm.weight": values((8,), 3) + 1.0,
        prefix + "hnorm.weight": values((8,), 5) + 1.0,
        prefix + "eh_proj.weight": values((8, 16), 7),
        prefix + "input_layernorm.weight": values((8,), 11) + 1.0,
        prefix + "self_attn.q_a_proj.weight": values((4, 8), 13),
        prefix + "self_attn.q_a_layernorm.weight": values((4,), 17) + 1.0,
        prefix + "self_attn.q_b_proj.weight": values((16, 4), 19),
        prefix + "self_attn.kv_a_proj_with_mqa.weight": values((6, 8), 23),
        prefix + "self_attn.kv_a_layernorm.weight": values((4,), 29) + 1.0,
        prefix + "self_attn.kv_b_proj.weight": values((16, 4), 31),
        prefix + "self_attn.o_proj.weight": values((8, 8), 37),
        prefix + "post_attention_layernorm.weight": values((8,), 41) + 1.0,
        prefix + "mlp.gate.weight": values((2, 8), 43),
        prefix + "mlp.shared_experts.gate_proj.weight": values((8, 8), 47),
        prefix + "mlp.shared_experts.up_proj.weight": values((8, 8), 53),
        prefix + "mlp.shared_experts.down_proj.weight": values((8, 8), 59),
        prefix + "shared_head.norm.weight": values((8,), 61) + 1.0,
        prefix + "shared_head.head.weight": values((16, 8), 67),
    }
    for expert_id in range(2):
        expert_prefix = prefix + f"mlp.experts.{expert_id}."
        offset = 71 + expert_id * 17
        weights[expert_prefix + "gate_proj.weight"] = values((8, 8), offset)
        weights[expert_prefix + "up_proj.weight"] = values((8, 8), offset + 3)
        weights[expert_prefix + "down_proj.weight"] = values((8, 8), offset + 5)
    return weights


def _uninitialized_model(model_type, *, layer_count=0, checkpoint_prefix=None):
    model = object.__new__(model_type)
    torch.nn.Module.__init__(model)
    if layer_count:
        model.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(layer_count)]
        )
    if checkpoint_prefix is not None:
        model._checkpoint_prefix = checkpoint_prefix
    return model


class DeepSeekNewloaderTest(unittest.TestCase):
    def test_local_rope_cache_preserves_reference_numerics(self):
        max_seq_len = 32
        rope_dim = 8
        base = 10000.0
        plain = build_rope_cache(
            {"qk_rope_head_dim": rope_dim, "rope_theta": base},
            max_seq_len,
            torch.device("cpu"),
        )
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=inv_freq.dtype), inv_freq)
        self.assertTrue(
            torch.equal(
                plain,
                torch.cat([freqs.cos(), freqs.sin()], dim=-1),
            )
        )

        factor = 8.0
        original_max = 16
        beta_fast = 32.0
        beta_slow = 1.0
        mscale = 1.0
        mscale_all_dim = 1.0
        yarn = build_rope_cache(
            {
                "qk_rope_head_dim": rope_dim,
                "rope_theta": base,
                "rope_scaling": {
                    "factor": factor,
                    "original_max_position_embeddings": original_max,
                    "beta_fast": beta_fast,
                    "beta_slow": beta_slow,
                    "mscale": mscale,
                    "mscale_all_dim": mscale_all_dim,
                },
            },
            max_seq_len,
            torch.device("cpu"),
        )
        freq_extra = 1.0 / (
            base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )
        freq_inter = freq_extra / factor

        def correction_dim(num_rotations):
            return (
                rope_dim * math.log(original_max / (num_rotations * 2 * math.pi))
            ) / (2 * math.log(base))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), rope_dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(rope_dim // 2, dtype=torch.float32) - low) / (high - low),
            0,
            1,
        )
        mask = 1.0 - ramp
        yarn_inv_freq = freq_inter * (1 - mask) + freq_extra * mask
        yarn_freqs = torch.outer(
            torch.arange(max_seq_len, dtype=torch.float32),
            yarn_inv_freq,
        )

        def yarn_mscale(value):
            if factor <= 1:
                return 1.0
            return 0.1 * value * math.log(factor) + 1.0

        scale = yarn_mscale(mscale) / yarn_mscale(mscale_all_dim)
        self.assertTrue(
            torch.equal(
                yarn,
                torch.cat(
                    [
                        yarn_freqs.cos() * scale,
                        yarn_freqs.sin() * scale,
                    ],
                    dim=-1,
                ),
            )
        )

        rope_parameters_yarn = build_rope_cache(
            {
                "qk_rope_head_dim": rope_dim,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": base,
                    "factor": factor,
                    "original_max_position_embeddings": original_max,
                    "beta_fast": beta_fast,
                    "beta_slow": beta_slow,
                    "mscale": mscale,
                    "mscale_all_dim": mscale_all_dim,
                },
            },
            max_seq_len,
            torch.device("cpu"),
        )
        self.assertTrue(torch.equal(rope_parameters_yarn, yarn))

        zero_all_dim = build_rope_cache(
            {
                "qk_rope_head_dim": rope_dim,
                "rope_theta": base,
                "rope_scaling": {
                    "factor": factor,
                    "original_max_position_embeddings": original_max,
                    "beta_fast": beta_fast,
                    "beta_slow": beta_slow,
                    "mscale": mscale,
                    "mscale_all_dim": 0.0,
                },
            },
            max_seq_len,
            torch.device("cpu"),
        )
        zero_all_dim_scale = yarn_mscale(mscale)
        torch.testing.assert_close(
            zero_all_dim,
            torch.cat(
                [
                    yarn_freqs.cos() * zero_all_dim_scale,
                    yarn_freqs.sin() * zero_all_dim_scale,
                ],
                dim=-1,
            ),
        )

    def test_rope_cache_rejects_ambiguous_or_incomplete_scaling(self):
        base_config = {"qk_rope_head_dim": 8}
        with self.assertRaisesRegex(ValueError, "unsupported.*rope_type"):
            build_rope_cache(
                {
                    **base_config,
                    "rope_parameters": {
                        "rope_type": "linear",
                        "factor": 2.0,
                    },
                },
                32,
                torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "missing keys"):
            build_rope_cache(
                {
                    **base_config,
                    "rope_parameters": {
                        "rope_type": "yarn",
                        "factor": 2.0,
                    },
                },
                32,
                torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            build_rope_cache(
                {
                    **base_config,
                    "rope_scaling": {
                        "factor": 2.0,
                        "original_max_position_embeddings": 16,
                        "mscale": 1.0,
                        "mscale_all_dim": 1.0,
                    },
                    "rope_parameters": {
                        "rope_type": "yarn",
                        "factor": 4.0,
                        "original_max_position_embeddings": 16,
                        "mscale": 1.0,
                        "mscale_all_dim": 1.0,
                    },
                },
                32,
                torch.device("cpu"),
            )

    def test_registry_aliases_are_typed(self):
        for model_type in (
            "deepseek2",
            "deepseek3",
            "deepseek_v31",
            "deepseek_v32",
            "glm_5",
            "kimi_k2",
        ):
            with self.subTest(model_type=model_type):
                self.assertIs(get_model_class(model_type), DeepSeekV32ForCausalLM)
        self.assertIs(
            get_model_class("deepseek-v3-mtp"),
            DeepSeekV32MTPForCausalLM,
        )

    def test_registry_aliases_complete_a_real_checkpoint_load(self):
        weights = _dense_checkpoint_weights()
        with tempfile.TemporaryDirectory() as checkpoint_path:
            with open(
                f"{checkpoint_path}/config.json",
                "w",
                encoding="utf-8",
            ) as config_file:
                json.dump(_dense_loader_config_json(), config_file)
            save_file(weights, f"{checkpoint_path}/model.safetensors")

            for model_type in (
                "deepseek2",
                "deepseek3",
                "deepseek_v31",
                "deepseek_v32",
                "glm_5",
                "kimi_k2",
            ):
                with self.subTest(model_type=model_type):
                    model_config = _dense_loader_model_config(checkpoint_path)
                    model_config.model_type = model_type
                    model = NewModelLoader(
                        model_config=model_config,
                        load_config=NewLoaderConfig(
                            compute_dtype=torch.float32,
                            device="cpu",
                        ),
                        model_path=checkpoint_path,
                    ).load()
                    self.assertIsInstance(model, DeepSeekV32ForCausalLM)
                    torch.testing.assert_close(
                        model.embed_tokens.weight,
                        weights["model.embed_tokens.weight"],
                    )

    def test_new_model_loader_safetensors_forward_matches_reference(self):
        weights = _dense_checkpoint_weights()
        with tempfile.TemporaryDirectory() as checkpoint_path:
            with open(
                f"{checkpoint_path}/config.json",
                "w",
                encoding="utf-8",
            ) as config_file:
                json.dump(_dense_loader_config_json(), config_file)
            save_file(weights, f"{checkpoint_path}/model.safetensors")
            model = NewModelLoader(
                model_config=_dense_loader_model_config(checkpoint_path),
                load_config=NewLoaderConfig(
                    tp_size=1,
                    tp_rank=0,
                    ep_size=1,
                    ep_rank=0,
                    compute_dtype=torch.float32,
                    device="cpu",
                ),
                model_path=checkpoint_path,
            ).load()

        self.assertIsInstance(model, DeepSeekV32ForCausalLM)
        self.assertFalse(model.training)
        torch.testing.assert_close(
            model.lm_head.weight,
            model.embed_tokens.weight,
        )
        self.assertIsNone(model._mla_kernel_layout)
        model._ensure_mla_kernel_layout()
        self.assertIsNotNone(model._mla_kernel_layout)
        self.assertGreater(model.layers[0].self_attn.q_a_proj.weight.numel(), 0)
        self.assertGreater(
            model.layers[0].self_attn.kv_a_proj_with_mqa.weight.numel(),
            0,
        )
        self.assertGreater(model.layers[0].self_attn.kv_b_proj.weight.numel(), 0)

        class TorchSiluAndMul(torch.nn.Module):
            @staticmethod
            def forward(gate_up):
                gate, up = gate_up.chunk(2, dim=-1)
                return F.silu(gate) * up

        # The production activation is a CUDA/HIP kernel and does not dispatch
        # FP32 CPU tensors. Keep this loader/forward parity test CPU-runnable;
        # GPU activation dispatch is covered by the focused GPU suites.
        model.layers[0].mlp.act_fn = TorchSiluAndMul()

        class ZeroAttention:
            fmha_params = None

            @staticmethod
            def forward(q_view, compressed_kv, k_pe, kv_cache, layer_idx, topk):
                del compressed_kv, k_pe, kv_cache, layer_idx, topk
                return torch.zeros(
                    (*q_view.shape[:-1], 2),
                    dtype=q_view.dtype,
                    device=q_view.device,
                )

        input_ids = torch.tensor([0, 3, 7], dtype=torch.int32)
        inputs = PyModelInputs(
            input_ids=input_ids,
            attention_inputs=PyAttentionInputs(),
        )
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_v3.language."
            "select_block_map_for_layer"
        ) as select_block_map:
            outputs = model(inputs, fmha_impl=ZeroAttention())
        select_block_map.assert_called_once_with(inputs.attention_inputs, 0)

        def rms_norm(x, weight):
            normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
            return normalized * weight

        residual = F.embedding(
            input_ids.long(),
            weights["model.embed_tokens.weight"],
        )
        mlp_input = rms_norm(
            residual,
            weights["model.layers.0.post_attention_layernorm.weight"],
        )
        gate = F.linear(
            mlp_input,
            weights["model.layers.0.mlp.gate_proj.weight"],
        )
        up = F.linear(
            mlp_input,
            weights["model.layers.0.mlp.up_proj.weight"],
        )
        mlp_output = F.linear(
            F.silu(gate) * up,
            weights["model.layers.0.mlp.down_proj.weight"],
        )
        expected = rms_norm(
            residual + mlp_output,
            weights["model.norm.weight"],
        )
        torch.testing.assert_close(outputs.hidden_states, expected)

    def test_weight_validation_preserves_error_type_and_adds_module_context(self):
        class ValueErrorLoader(RtpModule):
            def load_weights(self, weights):
                del weights

            def validate_weights_loaded(self, loaded_tensor_ids):
                del loaded_tensor_ids
                raise ValueError("invalid child state")

        root = RtpModule()
        root.child = ValueErrorLoader()
        with self.assertRaisesRegex(
            ValueError,
            "Weight validation failed for child.*invalid child state",
        ):
            NewModelLoader._validate_loaded_weights(root)

    def test_mtp_new_model_loader_reads_only_appended_draft_weights(self):
        weights = _mtp_checkpoint_weights()
        # The test intentionally loads CPU tensors. Preserve their layout while
        # still executing the real MoE post-load validation/factory hook; device
        # shuffling is covered by the GPU loader/smoke suites.
        cpu_device = types.SimpleNamespace(
            shuffle_moe_weight=lambda tensor, data_type, name: tensor,
        )
        with tempfile.TemporaryDirectory() as checkpoint_path:
            with open(
                f"{checkpoint_path}/config.json",
                "w",
                encoding="utf-8",
            ) as config_file:
                json.dump(_mtp_loader_config_json(), config_file)
            save_file(weights, f"{checkpoint_path}/model.safetensors")
            with mock.patch(
                "rtp_llm.models_py.layers.moe_experts.get_current_device",
                return_value=cpu_device,
            ):
                model = NewModelLoader(
                    model_config=_mtp_loader_model_config(checkpoint_path),
                    load_config=NewLoaderConfig(
                        tp_size=1,
                        tp_rank=0,
                        ep_size=1,
                        ep_rank=0,
                        compute_dtype=torch.float32,
                        device="cpu",
                        parallelism_config=_single_rank_parallelism_config(),
                    ),
                    model_path=checkpoint_path,
                ).load()

        self.assertIsInstance(model, DeepSeekV32MTPForCausalLM)
        self.assertEqual(model._checkpoint_prefix, "model.layers.1.")
        self.assertFalse(model.training)
        torch.testing.assert_close(
            model.embed_tokens.weight,
            weights["model.layers.1.embed_tokens.weight"],
        )
        torch.testing.assert_close(
            model.mtp_block.fc.weight,
            weights["model.layers.1.eh_proj.weight"],
        )
        torch.testing.assert_close(
            model.lm_head.weight,
            weights["model.layers.1.shared_head.head.weight"],
        )
        self.assertIsNotNone(model.layers[0].mlp.experts.fused_moe)
        model._ensure_mla_kernel_layout()
        self.assertGreater(model.layers[0].self_attn.q_a_proj.weight.numel(), 0)
        self.assertGreater(
            model.layers[0].self_attn.kv_a_proj_with_mqa.weight.numel(),
            0,
        )

    def test_rotary_cache_is_valid_immediately_after_construction(self):
        rotary = DeepseekV3RotaryEmbedding(
            dim=8,
            max_position_embeddings=16,
            device=torch.device("cpu"),
        )
        cache_ptr = rotary.cos_cached.data_ptr()
        self.assertEqual(rotary.max_seq_len_cached, 16)
        rotary(torch.zeros(1, 8), seq_len=8)
        self.assertEqual(rotary.cos_cached.data_ptr(), cache_ptr)

    def test_core_filter_excludes_appended_draft_and_unrelated_tensors(self):
        model = _uninitialized_model(DeepSeekV32ForCausalLM, layer_count=4)
        should_load = model.checkpoint_weight_name_filter()
        accepted = (
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.3.mlp.experts.7.down_proj.weight",
            "model.layers.bad.weight",
            "model.norm.weight",
            "model.unknown.weight",
            "model.visual.weight",
            "lm_head.weight",
        )
        rejected = (
            "model.layers.4.self_attn.q_a_proj.weight",
            "model.layers.40.self_attn.q_a_proj.weight",
            "mtp.layers.0.weight",
        )
        for name in accepted:
            with self.subTest(name=name):
                self.assertTrue(should_load(name))
        for name in rejected:
            with self.subTest(name=name):
                self.assertFalse(should_load(name))

    def test_mtp_prefix_supports_standalone_and_full_checkpoints(self):
        standalone = {
            "architectures": ["DeepseekV3ForCausalLMNextN"],
            "num_hidden_layers": 1,
            "num_nextn_predict_layers": 1,
        }
        self.assertEqual(_draft_checkpoint_layer(standalone), 0)
        with self.assertRaisesRegex(ValueError, "at most one draft"):
            _draft_checkpoint_layer(
                {
                    "architectures": ["DeepseekV3ForCausalLMNextN"],
                    "num_hidden_layers": 1,
                    "num_nextn_predict_layers": 2,
                }
            )
        self.assertEqual(_draft_checkpoint_layer(_raw_config()), 4)
        with self.assertRaisesRegex(ValueError, "exactly one appended"):
            _draft_checkpoint_layer(
                {
                    "architectures": ["DeepseekV3ForCausalLM"],
                    "num_hidden_layers": 4,
                    "num_nextn_predict_layers": 0,
                }
            )
        with self.assertRaisesRegex(ValueError, "exactly one appended"):
            _draft_checkpoint_layer(
                {
                    "architectures": ["DeepseekV3ForCausalLM"],
                    "num_hidden_layers": 4,
                    "num_nextn_predict_layers": 2,
                }
            )

    def test_mtp_real_constructor_uses_appended_layer_from_full_checkpoint(self):
        config_json = _raw_config()
        model_config = _model_config()
        # ModelFactory creates the draft config from the full checkpoint, so
        # this remains the main-model layer count at construction time.
        model_config.num_layers = config_json["num_hidden_layers"]
        with tempfile.TemporaryDirectory() as checkpoint_path:
            model_config.ckpt_path = checkpoint_path
            with open(
                f"{checkpoint_path}/config.json", "w", encoding="utf-8"
            ) as config_file:
                json.dump(config_json, config_file)
            with torch.device("cpu"):
                model = DeepSeekV32MTPForCausalLM(
                    model_config,
                    _load_config(),
                )

        self.assertEqual(model._checkpoint_layer, 4)
        self.assertEqual(model._checkpoint_prefix, "model.layers.4.")
        self.assertEqual(model.layer_num, 1)
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(model.cos_sin_cache.device.type, "cpu")

    def test_mtp_filter_is_exact_and_rank_invariant(self):
        model = _uninitialized_model(
            DeepSeekV32MTPForCausalLM,
            checkpoint_prefix="model.layers.4.",
        )
        should_load = model.checkpoint_weight_name_filter()
        self.assertTrue(should_load("model.layers.4.embed_tokens.weight"))
        self.assertTrue(should_load("model.layers.4.mlp.experts.7.gate_proj.weight"))
        self.assertFalse(should_load("model.layers.3.embed_tokens.weight"))
        self.assertFalse(should_load("model.layers.40.embed_tokens.weight"))
        self.assertFalse(should_load("model.layers.4."))
        self.assertFalse(should_load("lm_head.weight"))

    def test_mtp_key_remap_is_explicit(self):
        expected = {
            "embed_tokens.weight": "embed_tokens.weight",
            "shared_head.head.weight": "lm_head.weight",
            "shared_head.norm.weight": "norm.weight",
            "enorm.weight": "mtp_block.e_norm.weight",
            "hnorm.weight": "mtp_block.h_norm.weight",
            "eh_proj.weight": "mtp_block.fc.weight",
            "self_attn.q_a_proj.weight": "layers.0.self_attn.q_a_proj.weight",
            "mlp.gate.weight": "layers.0.mlp.gate.weight",
        }
        for name, remapped in expected.items():
            with self.subTest(name=name):
                self.assertEqual(_remap_key(name), remapped)

    def test_mtp_load_weights_maps_real_modules_and_rejects_extra_tensors(self):
        model = _uninitialized_model(
            DeepSeekV32MTPForCausalLM,
            checkpoint_prefix="model.layers.4.",
        )
        model.embed_tokens = torch.nn.Embedding(3, 2)
        model.lm_head = torch.nn.Linear(2, 3, bias=False)
        model.norm = torch.nn.LayerNorm(2, elementwise_affine=True, bias=False)
        model.mtp_block = MTPBlock(hidden_size=2, params_dtype=torch.float32)

        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_a_proj = torch.nn.Linear(2, 2, bias=False)
        model.layers = torch.nn.ModuleList([layer])

        values = {
            "model.layers.4.embed_tokens.weight": torch.arange(
                6, dtype=torch.float32
            ).reshape(3, 2),
            "model.layers.4.shared_head.head.weight": torch.arange(
                6, 12, dtype=torch.float32
            ).reshape(3, 2),
            "model.layers.4.shared_head.norm.weight": torch.tensor([12.0, 13.0]),
            "model.layers.4.enorm.weight": torch.tensor([14.0, 15.0]),
            "model.layers.4.hnorm.weight": torch.tensor([16.0, 17.0]),
            "model.layers.4.eh_proj.weight": torch.arange(
                18, 26, dtype=torch.float32
            ).reshape(2, 4),
            "model.layers.4.self_attn.q_a_proj.weight": torch.arange(
                26, 30, dtype=torch.float32
            ).reshape(2, 2),
        }
        model.load_weights(values)

        self.assertTrue(
            torch.equal(
                model.embed_tokens.weight,
                values["model.layers.4.embed_tokens.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.lm_head.weight,
                values["model.layers.4.shared_head.head.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.norm.weight,
                values["model.layers.4.shared_head.norm.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.mtp_block.e_norm.weight,
                values["model.layers.4.enorm.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.mtp_block.h_norm.weight,
                values["model.layers.4.hnorm.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.mtp_block.fc.weight,
                values["model.layers.4.eh_proj.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.layers[0].self_attn.q_a_proj.weight,
                values["model.layers.4.self_attn.q_a_proj.weight"],
            )
        )

        with self.assertRaisesRegex(RuntimeError, "non-draft"):
            model.load_weights(
                {
                    "model.layers.3.embed_tokens.weight": torch.zeros(
                        3, 2, dtype=torch.float32
                    )
                }
            )
        with self.assertRaisesRegex(RuntimeError, "could not dispatch"):
            model.load_weights(
                {"model.layers.4.unknown.weight": torch.zeros(1, dtype=torch.float32)}
            )

    def test_config_keeps_attention_and_ffn_topologies_distinct(self):
        cfg = extract_config_values(_model_config(), _load_config(), _raw_config())
        self.assertEqual((cfg["attn_tp_size"], cfg["attn_tp_rank"]), (1, 0))
        self.assertEqual((cfg["ffn_tp_size"], cfg["ffn_tp_rank"]), (2, 1))
        self.assertEqual((cfg["lm_head_tp_size"], cfg["lm_head_tp_rank"]), (2, 1))
        self.assertEqual((cfg["ep_size"], cfg["ep_rank"]), (2, 1))
        self.assertEqual(cfg["moe_layer_index"], [3])
        self.assertEqual(cfg["topk_method"], "greedy")

    def test_config_rejects_eplb_before_module_construction(self):
        config = _model_config()
        config.eplb_config.eplb_mode = EplbMode.EPLB
        with self.assertRaisesRegex(ValueError, "EPLB is not supported"):
            extract_config_values(config, _load_config(), _raw_config())

    def test_tied_embeddings_require_matching_tp_partitions(self):
        config = _model_config()
        config.tie_word_embeddings = True
        with self.assertRaisesRegex(ValueError, "matching attention and LM-head"):
            extract_config_values(config, _load_config(), _raw_config())

    def test_zero_explicit_shared_expert_width_uses_checkpoint_topology(self):
        raw = _raw_config()
        raw["shared_expert_intermediate_size"] = 0
        cfg = extract_config_values(_model_config(), _load_config(), raw)
        self.assertEqual(cfg["shared_expert_intermediate_size"], 8)

    def test_new_loader_rejects_unsupported_attention_and_ffn_tp_groups(self):
        with self.assertRaisesRegex(ValueError, "independent TP subgroups"):
            NewLoaderConfig(
                tp_size=4,
                tp_rank=2,
                attn_tp_size=2,
                attn_tp_rank=1,
            )
        with self.assertRaisesRegex(ValueError, "expected rank=2"):
            NewLoaderConfig(
                tp_size=4,
                tp_rank=2,
                ffn_tp_size=4,
                ffn_tp_rank=1,
            )
        config = NewLoaderConfig(
            tp_size=4,
            tp_rank=2,
            attn_tp_size=1,
            attn_tp_rank=0,
            ffn_tp_size=4,
            ffn_tp_rank=2,
        )
        self.assertEqual((config.attn_tp_size, config.attn_tp_rank), (1, 0))
        self.assertEqual((config.ffn_tp_size, config.ffn_tp_rank), (4, 2))

    def test_raw_router_config_is_canonical(self):
        raw = _raw_config()
        raw.update(
            {
                "scoring_func": "softmax",
                "routed_scaling_factor": 2.5,
                "n_group": 4,
                "topk_group": 2,
                "norm_topk_prob": False,
                "topk_method": "group_limited_greedy",
            }
        )
        cfg = extract_config_values(_model_config(), _load_config(), raw)
        self.assertEqual(cfg["scoring_func"], 0)
        self.assertEqual(cfg["routed_scaling_factor"], 2.5)
        self.assertEqual(cfg["n_group"], 4)
        self.assertEqual(cfg["topk_group"], 2)
        self.assertFalse(cfg["has_moe_norm"])
        self.assertEqual(cfg["topk_method"], "group_limited_greedy")

    def test_sparse_indexer_rejects_no_q_lora(self):
        with self.assertRaisesRegex(ValueError, "requires q_lora_rank"):
            extract_config_values(
                _model_config(sparse=True, q_lora_rank=0),
                _load_config(),
                _raw_config(),
            )

    def test_config_rejects_legacy_expanded_mha_fallback(self):
        config = _model_config()
        config.mla_ops_type = "MHA"
        with self.assertRaisesRegex(ValueError, "requires an MLA attention backend"):
            extract_config_values(config, _load_config(), _raw_config())

    def test_sparse_indexer_fast_and_sparse_call_sequences(self):
        class FakeIndexerOp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def apply_rope_and_rotate_k(self, key, positions):
                self.calls.append(("rotate_k", positions))
                return key + 1

            def quant_k_only(self, key, kv_cache, slot_mapping):
                self.calls.append(("quant_k_only", key, kv_cache, slot_mapping))

            def apply_rope_and_rotate_q_k(self, query, key, positions):
                self.calls.append(("rotate_q_k", positions))
                return query + 2, key + 3

            def quant_q_k(self, query, key, kv_cache, slot_mapping):
                self.calls.append(("quant_q_k", kv_cache, slot_mapping))
                q_scale = torch.ones(
                    query.shape[0],
                    query.shape[1],
                    1,
                    dtype=query.dtype,
                )
                return query, q_scale

            def _get_topk_ragged(
                self,
                q_fp8,
                weights,
                kv_cache,
                fmha_params,
                attention_inputs,
            ):
                self.calls.append(("topk_ragged", kv_cache, attention_inputs))
                return torch.tensor([[1], [0]], dtype=torch.int32)

        fake_op = FakeIndexerOp()
        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.model.IndexerOp",
                return_value=fake_op,
            ),
            torch.device("cpu"),
        ):
            indexer = DeepSeekV32Indexer(
                index_n_heads=1,
                index_head_dim=2,
                index_topk=1,
                rope_head_dim=2,
                hidden_size=2,
                q_lora_rank=2,
                layer_idx=0,
                layernorm_eps=1e-6,
                blocksize=64,
                is_neox_style=False,
                params_dtype=torch.float32,
            )
        indexer.wq_b = torch.nn.Identity()
        indexer.wk = torch.nn.Identity()
        indexer.k_norm = torch.nn.Identity()
        indexer.weights_proj = torch.nn.Linear(2, 1, bias=False)
        indexer.weights_proj.weight.data.fill_(1.0)

        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        q_lora = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
        positions = torch.tensor([0, 1], dtype=torch.int32)
        slot_mapping = torch.tensor([4, 5], dtype=torch.int32)
        fmha_params = types.SimpleNamespace(
            positions_d=positions,
            slot_mapping=slot_mapping,
        )
        kv_cache = object()

        result = indexer(
            hidden_states,
            q_lora,
            kv_cache,
            fmha_params,
            types.SimpleNamespace(is_prefill=False),
            use_fast_path=True,
        )
        self.assertIsNone(result)
        self.assertEqual(
            [call[0] for call in fake_op.calls], ["rotate_k", "quant_k_only"]
        )
        self.assertIs(fake_op.calls[1][2], kv_cache)

        fake_op.calls.clear()
        attention_inputs = types.SimpleNamespace(is_prefill=True)
        result = indexer(
            hidden_states,
            q_lora,
            kv_cache,
            fmha_params,
            attention_inputs,
            use_fast_path=False,
        )
        self.assertTrue(
            torch.equal(result, torch.tensor([[1], [0]], dtype=torch.int32))
        )
        self.assertEqual(
            [call[0] for call in fake_op.calls],
            ["rotate_q_k", "quant_q_k", "topk_ragged"],
        )
        self.assertIs(fake_op.calls[-1][1], kv_cache)
        self.assertIs(fake_op.calls[-1][2], attention_inputs)

    @unittest.skipIf(
        get_device_type() == DeviceType.ROCm,
        "IndexerOp is not implemented on ROCm",
    )
    def test_sparse_indexer_rope_cache_is_a_device_tracked_buffer(self):
        cache = torch.linspace(-0.97, 0.91, 32, dtype=torch.float32).reshape(8, 4)
        with torch.device("cpu"):
            indexer = DeepSeekV32Indexer(
                index_n_heads=1,
                index_head_dim=2,
                index_topk=1,
                rope_head_dim=2,
                hidden_size=2,
                q_lora_rank=2,
                layer_idx=0,
                layernorm_eps=1e-6,
                blocksize=64,
                is_neox_style=False,
                params_dtype=torch.float32,
                cos_sin_cache=cache,
            )
        self.assertIs(
            dict(indexer.indexer_op.named_buffers())["cos_sin_cache"],
            indexer.indexer_op.cos_sin_cache,
        )
        rebound_cache = cache + 0.25
        indexer.bind_rope_cache(rebound_cache)
        self.assertIs(indexer.indexer_op.cos_sin_cache, rebound_cache)
        with self.assertRaisesRegex(TypeError, "must use torch.float32"):
            indexer.bind_rope_cache(rebound_cache.to(torch.bfloat16))
        indexer.indexer_op.to(dtype=torch.bfloat16)
        self.assertEqual(indexer.indexer_op.cos_sin_cache.dtype, torch.float32)
        torch.testing.assert_close(
            indexer.indexer_op.cos_sin_cache,
            rebound_cache,
            rtol=0,
            atol=0,
        )

    def test_model_dtype_migration_preserves_shared_fp32_rope_cache(self):
        cache = torch.linspace(-0.97, 0.91, 16, dtype=torch.float32).reshape(4, 4)

        # A plain consumer isolates the top-level alias-restoration contract;
        # the real CUDA IndexerOp's standalone _apply behavior is covered by
        # test_sparse_indexer_rope_cache_is_a_device_tracked_buffer above.
        class CacheConsumer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cos_sin_cache", cache, persistent=False)

            def bind_rope_cache(self, rope_cache):
                if rope_cache.dtype != torch.float32:
                    raise TypeError("RoPE cache must remain FP32")
                self.cos_sin_cache = rope_cache

        indexer = CacheConsumer()
        attention = torch.nn.Module()
        attention.indexer = indexer
        layer = torch.nn.Module()
        layer.self_attn = attention

        class CacheOwner(MlaRuntimeLayoutMixin, RtpModule):
            def __init__(self):
                super().__init__()
                self.register_buffer("cos_sin_cache", cache, persistent=False)
                self.layers = torch.nn.ModuleList([layer])
                self._mla_kernel_layout = None

        owner = CacheOwner().to(dtype=torch.bfloat16)
        self.assertEqual(owner.cos_sin_cache.dtype, torch.float32)
        self.assertIs(
            owner.layers[0].self_attn.indexer.cos_sin_cache,
            owner.cos_sin_cache,
        )
        torch.testing.assert_close(owner.cos_sin_cache, cache, rtol=0, atol=0)

    def test_rotary_cache_is_fp32_even_when_default_dtype_is_bfloat16(self):
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            cache = build_rope_cache(
                {"qk_rope_head_dim": 4, "rope_theta": 10000.0},
                max_seq_len=16,
                device=torch.device("cpu"),
            )
        finally:
            torch.set_default_dtype(original_dtype)
        reference = build_rope_cache(
            {"qk_rope_head_dim": 4, "rope_theta": 10000.0},
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        self.assertEqual(cache.dtype, torch.float32)
        torch.testing.assert_close(cache, reference, rtol=0, atol=0)

    def test_linear_prefixes_preserve_layer_qualified_quantization_rules(self):
        quant_config = QuantizationConfig(
            "FP8_PER_BLOCK",
            ignored_layers=[
                "model.layers.3.self_attn.o_proj",
                "model.layers.3.mlp.down_proj",
            ],
        )
        layer = DeepSeekV32DecoderLayer(
            hidden_size=128,
            num_heads=2,
            q_lora_rank=128,
            kv_lora_rank=128,
            nope_head_dim=64,
            rope_head_dim=64,
            v_head_dim=64,
            layer_idx=3,
            attn_tp_size=1,
            attn_tp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.bfloat16,
            layernorm_eps=1e-6,
            quant_config=quant_config,
            model_config=_router_model_config(hidden_size=128),
            parallelism_config=None,
            moe_config=None,
            is_moe_layer=False,
            dense_intermediate_size=128,
            prefix="layers.3",
        )
        self.assertEqual(layer.self_attn.q_a_proj.prefix, "layers.3.self_attn.q_a_proj")
        self.assertEqual(layer.self_attn.o_proj.prefix, "layers.3.self_attn.o_proj")
        self.assertEqual(layer.mlp.gate_up_proj.prefix, "layers.3.mlp.gate_up_proj")
        self.assertEqual(layer.mlp.down_proj.prefix, "layers.3.mlp.down_proj")
        self.assertTrue(quant_config.is_layer_ignored(layer.self_attn.o_proj.prefix))
        self.assertTrue(quant_config.is_layer_ignored(layer.mlp.down_proj.prefix))

        fake_indexer_op = torch.nn.Identity()
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_v3.model.IndexerOp",
            return_value=fake_indexer_op,
        ):
            indexer = DeepSeekV32Indexer(
                index_n_heads=1,
                index_head_dim=2,
                index_topk=1,
                rope_head_dim=2,
                hidden_size=2,
                q_lora_rank=2,
                layer_idx=3,
                layernorm_eps=1e-6,
                blocksize=64,
                is_neox_style=False,
                params_dtype=torch.float32,
                prefix="layers.3.self_attn.indexer",
            )
        self.assertEqual(
            indexer.wq_b.prefix,
            "layers.3.self_attn.indexer.wq_b",
        )
        self.assertEqual(
            indexer.weights_proj.prefix,
            "layers.3.self_attn.indexer.weights_proj",
        )

    def test_grouped_router_rejects_invalid_expert_partition(self):
        raw = _raw_config()
        raw.update(
            {
                "n_group": 3,
                "topk_method": "noaux_tc",
                "topk_group": 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "divisible by n_group"):
            extract_config_values(_model_config(), _load_config(), raw)

    def test_no_q_lora_has_no_empty_checkpoint_parameters(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=8,
            num_heads=2,
            q_lora_rank=0,
            kv_lora_rank=4,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            params_dtype=torch.float32,
        )
        self.assertIsNone(attention.q_a_layernorm)
        self.assertIsNone(attention.q_b_proj)
        self.assertFalse(
            any(
                name.startswith("q_b_proj.") for name, _ in attention.named_parameters()
            )
        )
        attention.process_weights_after_loading()
        derived = {
            "_fused_qkv_a_w",
            "_fused_qkv_b_w",
            "_kv_b_w",
            "_kc_w",
            "_vc_w",
        }
        self.assertTrue(derived.isdisjoint(dict(attention.named_parameters())))
        self.assertTrue(derived.isdisjoint(attention.state_dict()))

    def test_no_q_lora_tp2_forward_uses_row_parallel_reduction(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=8,
            num_heads=4,
            q_lora_rank=0,
            kv_lora_rank=4,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            tp_size=2,
            tp_rank=1,
            params_dtype=torch.float32,
        )
        for parameter in attention.parameters():
            parameter.data.zero_()
        attention.kv_a_layernorm.weight.data.fill_(1.0)
        attention.process_weights_after_loading()

        class ZeroFmha:
            def forward(self, q, compressed_kv, k_pe, *args):
                self.shapes = (
                    tuple(q.shape),
                    tuple(compressed_kv.shape),
                    tuple(k_pe.shape),
                )
                return torch.zeros(
                    q.shape[0],
                    q.shape[1] * 2,
                    dtype=q.dtype,
                    device=q.device,
                )

        fmha_impl = ZeroFmha()
        hidden_states = torch.randn(3, 8)
        with mock.patch(
            "rtp_llm.models_py.layers.linear.all_reduce",
            side_effect=lambda tensor, group: tensor,
        ) as reduce_mock:
            output = attention(hidden_states, fmha_impl)

        self.assertEqual(fmha_impl.shapes, ((3, 2, 4), (3, 4), (3, 2)))
        self.assertEqual(tuple(output.shape), (3, 8))
        reduce_mock.assert_called_once()
        self.assertIs(reduce_mock.call_args.kwargs["group"], Group.TP)

    def test_mla_output_projection_owns_tp_reduction(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=8,
            num_heads=2,
            q_lora_rank=4,
            kv_lora_rank=4,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            tp_size=2,
            tp_rank=0,
            params_dtype=torch.float32,
        )
        self.assertTrue(attention.o_proj.reduce_output)

    def test_mla_fp8_derived_weights_support_tensor_and_channel_scales(self):
        class Fp8LinearStub(torch.nn.Module):
            def fp8_scale_block_size(self) -> tuple[int, int]:
                return (128, 128)

        weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float8_e4m3fn)
        for scale in (
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([[0.5], [2.0]], dtype=torch.float32),
        ):
            with self.subTest(scale_shape=tuple(scale.shape)):
                linear = Fp8LinearStub()
                linear.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
                linear.weight_scale = torch.nn.Parameter(
                    scale.clone(), requires_grad=False
                )
                expected_scale = scale.to(torch.bfloat16).float()
                if expected_scale.numel() == 1:
                    expected_scale = expected_scale.reshape(1, 1)
                expected = (weight.to(torch.bfloat16).float() * expected_scale).to(
                    torch.bfloat16
                )
                self.assertTrue(torch.equal(_linear_weight_bf16(linear), expected))

    def test_mla_fused_fp8_runtime_uses_configured_input_group_size(self):
        runtime = _CudaRuntimeFusedFp8Linear(
            torch.ones(4, 4),
            torch.ones(2, 1),
            (2, 4),
        )
        quantized = torch.zeros(1, 4)
        input_scales = torch.ones(1, 1)
        quantizer = mock.Mock(return_value=(quantized, input_scales))

        def fake_gemm(_a, _b, output, **_kwargs):
            output.zero_()

        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.attention."
                "_resolve_sgl_per_token_group_quant",
                return_value=quantizer,
            ),
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.attention."
                "_resolve_fp8_gemm_nt",
                return_value=fake_gemm,
            ),
        ):
            output = runtime(torch.ones(1, 4))

        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertEqual(quantizer.call_args.kwargs["group_size"], 4)

    def test_mla_kernel_fp8_layout_preserves_ue8m0_contract(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        fp32_scale = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        normal_weight, normal_scale = _kernel_fp8_weight_and_scale(weight, fp32_scale)
        self.assertEqual(tuple(normal_weight.shape), (4, 6))
        self.assertEqual(tuple(normal_scale.shape), (2, 3))
        self.assertEqual(normal_weight.data_ptr(), weight.data_ptr())
        self.assertEqual(normal_scale.data_ptr(), fp32_scale.data_ptr())

        ue8m0_scale = torch.ones(6, 1, dtype=torch.int32)
        ue8m0_weight, preserved_scale = _kernel_fp8_weight_and_scale(
            weight, ue8m0_scale
        )
        self.assertIs(ue8m0_weight, weight)
        self.assertIs(preserved_scale, ue8m0_scale)
        self.assertEqual(tuple(ue8m0_weight.shape), (6, 4))
        self.assertEqual(tuple(preserved_scale.shape), (6, 1))

    def test_mla_fused_runtime_requants_before_child_post_load(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        scale = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        requant_weight = weight + 1
        requant_scale = torch.ones(6, 1, dtype=torch.int32)
        requant = mock.Mock(return_value=(requant_weight, requant_scale))

        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.attention."
                "is_deep_gemm_e8m0_used",
                return_value=True,
            ) as e8m0_used,
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.attention."
                "_resolve_requant_weight_ue8m0",
                return_value=requant,
            ),
        ):
            runtime_weight, runtime_scale = _prepare_fused_fp8_runtime_weight(
                weight, scale
            )

        e8m0_used.assert_called_once_with(weight.device)
        requant.assert_called_once_with(weight, scale)
        self.assertIs(runtime_weight, requant_weight)
        self.assertIs(runtime_scale, requant_scale)
        self.assertEqual(runtime_scale.dtype, torch.int32)

    def test_mla_fused_runtime_keeps_standard_block_scales(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        scale = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_v3.attention."
            "is_deep_gemm_e8m0_used",
            return_value=False,
        ):
            runtime_weight, runtime_scale = _prepare_fused_fp8_runtime_weight(
                weight, scale
            )
        self.assertIs(runtime_weight, weight)
        self.assertIs(runtime_scale, scale)

    def test_mla_bf16_kernel_layout_contains_only_backend_consumers(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=8,
            num_heads=2,
            q_lora_rank=4,
            kv_lora_rank=4,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            params_dtype=torch.float32,
        )
        # LinearFactory allocates parameters with torch.empty.  Initialize
        # them deterministically so this layout assertion cannot depend on
        # allocator contents (in particular, NaNs make torch.equal(x, x)
        # return False).
        with torch.no_grad():
            for parameter_index, parameter in enumerate(attention.parameters()):
                values = torch.arange(
                    parameter.numel(),
                    dtype=parameter.dtype,
                    device=parameter.device,
                ).reshape_as(parameter)
                parameter.copy_(values + parameter_index)
        attention.process_weights_after_loading()
        weights = attention._build_mla_kernel_weights()

        self.assertFalse(hasattr(attention, "_fused_qkv_b_w"))
        self.assertEqual(set(weights), {W.mla_kv_b_w, W.mla_kc, W.mla_vc})
        kv_b_ptr = weights[W.mla_kv_b_w].data_ptr()
        attention.release_checkpoint_only_weights()
        self.assertEqual(attention.q_a_proj.weight.numel(), 0)
        self.assertEqual(attention.kv_a_proj_with_mqa.weight.numel(), 0)
        self.assertEqual(attention.kv_b_proj.weight.numel(), 0)
        self.assertGreater(attention.q_b_proj.weight.numel(), 0)
        self.assertGreater(attention.o_proj.weight.numel(), 0)
        self.assertEqual(weights[W.mla_kv_b_w].data_ptr(), kv_b_ptr)
        with self.assertRaisesRegex(RuntimeError, "rebuild the model"):
            attention.load_weights({})

    def test_mla_fp8_kernel_layout_keeps_only_kv_b_scale(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=128,
            num_heads=2,
            q_lora_rank=128,
            kv_lora_rank=128,
            nope_head_dim=64,
            rope_head_dim=64,
            v_head_dim=64,
            layer_idx=0,
            quant_config=QuantizationConfig("FP8_PER_BLOCK"),
            params_dtype=torch.bfloat16,
        )
        fused_weight = torch.zeros(
            128 + 128 + 64,
            128,
            dtype=torch.float8_e4m3fn,
        )
        attention._fused_qkv_a_w = fused_weight
        del attention.q_b_proj.weight_scale_inv
        attention.q_b_proj.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.ones(2, 1, dtype=torch.float32),
                requires_grad=False,
            ),
        )
        attention._kv_b_w = torch.empty(128, 256)
        attention._kc_w = torch.empty(2, 64, 128)
        attention._vc_w = torch.empty(2, 128, 64)

        weights = attention._build_mla_kernel_weights()

        self.assertEqual(
            set(weights),
            {W.mla_kv_b_w, W.mla_kv_b_s, W.mla_kc, W.mla_vc},
        )
        attention.release_checkpoint_only_weights()
        self.assertEqual(attention.q_a_proj.weight.numel(), 0)
        self.assertEqual(attention.kv_a_proj_with_mqa.weight.numel(), 0)
        self.assertGreater(attention.kv_b_proj.weight.numel(), 0)

    def test_non_noaux_router_preserves_scoring_grouping_and_scaling(self):
        logits = torch.tensor(
            [
                [1.0, 2.0, -1.0, 0.0],
                [-2.0, -1.0, 3.0, 2.0],
            ],
            dtype=torch.float32,
        )
        weights, ids = _select_deepseek_topk(
            logits,
            top_k=2,
            scoring_func=1,
            n_group=2,
            topk_group=1,
            group_limited=True,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        scores = logits.sigmoid()
        group_scores = scores.view(2, 2, 2).amax(dim=-1)
        selected_groups = group_scores.topk(1, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        masked = scores.masked_fill(
            ~group_mask.unsqueeze(-1).expand(-1, -1, 2).reshape_as(scores),
            float("-inf"),
        )
        expected_weights, expected_ids = masked.topk(2, dim=-1, sorted=False)
        expected_weights = expected_weights / expected_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-20)
        self.assertTrue(torch.equal(ids, expected_ids))
        self.assertTrue(torch.equal(weights, expected_weights))

        softmax_weights, softmax_ids = _select_deepseek_topk(
            logits,
            top_k=2,
            scoring_func=0,
            n_group=2,
            topk_group=1,
            group_limited=False,
            renormalize=False,
            routed_scaling_factor=1.5,
        )
        reference_weights, reference_ids = logits.softmax(dim=-1).topk(
            2, dim=-1, sorted=False
        )
        self.assertTrue(torch.equal(softmax_ids, reference_ids))
        self.assertTrue(torch.equal(softmax_weights, reference_weights * 1.5))

    def test_noaux_router_uses_bias_only_for_selection(self):
        logits = torch.tensor(
            [
                [2.0, 1.0, 0.0, -1.0, -2.0, 3.0, 0.5, -0.5],
                [-1.0, 0.0, 1.0, 2.0, 3.0, -2.0, -0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        correction_bias = torch.tensor(
            [-0.3, 0.2, 0.1, -0.2, 0.4, -0.4, 0.3, -0.1],
            dtype=torch.float32,
        )
        weights, ids = _select_deepseek_noaux_topk(
            logits,
            correction_bias,
            top_k=2,
            n_group=4,
            topk_group=2,
            renormalize=True,
            routed_scaling_factor=2.5,
        )

        expected_weights, expected_ids = _manual_noaux_reference(
            logits,
            correction_bias,
            top_k=2,
            n_group=4,
            topk_group=2,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        actual_order = ids.argsort(dim=-1)
        expected_order = expected_ids.argsort(dim=-1)
        self.assertTrue(
            torch.equal(
                ids.gather(1, actual_order),
                expected_ids.gather(1, expected_order),
            )
        )
        torch.testing.assert_close(
            weights.gather(1, actual_order),
            expected_weights.gather(1, expected_order),
        )
        scores = logits.sigmoid()
        choice_scores = scores + correction_bias
        self.assertFalse(
            torch.equal(
                weights,
                choice_scores.gather(1, ids)
                / choice_scores.gather(1, ids).sum(dim=-1, keepdim=True)
                * 2.5,
            )
        )

    def test_noaux_router_matches_hand_computed_golden(self):
        probabilities = torch.tensor([[0.5, 0.75, 0.25, 0.6]], dtype=torch.float32)
        logits = torch.logit(probabilities)
        weights, ids = _select_deepseek_noaux_topk(
            logits,
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            top_k=2,
            n_group=2,
            topk_group=1,
            renormalize=False,
            routed_scaling_factor=2.0,
        )
        order = ids.argsort(dim=-1)
        self.assertTrue(torch.equal(ids.gather(1, order), torch.tensor([[2, 3]])))
        torch.testing.assert_close(
            weights.gather(1, order),
            torch.tensor([[0.5, 1.2]]),
        )

    def test_noaux_router_device_selection_is_explicit(self):
        parallelism_config = _single_rank_parallelism_config()
        moe_config = types.SimpleNamespace(fake_balance_expert=False)
        model_config = _router_model_config(moe_n_group=2, moe_topk_group=1)
        for device_type, expect_fast in (
            (DeviceType.Cuda, True),
            (DeviceType.ROCm, False),
        ):
            with (
                self.subTest(device_type=device_type),
                mock.patch(
                    "rtp_llm.models_py.new_models.deepseek_v3.moe.get_device_type",
                    return_value=device_type,
                ),
                mock.patch(
                    "rtp_llm.models_py.new_models.deepseek_v3.moe.GroupTopK"
                ) as group_topk,
                torch.device("cpu"),
            ):
                block = DeepSeekV32MoEBlock(
                    hidden_size=8,
                    moe_intermediate_size=4,
                    num_experts=4,
                    top_k=2,
                    layer_idx=0,
                    tp_size=1,
                    tp_rank=0,
                    ep_size=1,
                    ep_rank=0,
                    model_config=model_config,
                    parallelism_config=parallelism_config,
                    moe_config=moe_config,
                    quant_config=None,
                    params_dtype=torch.float32,
                    has_shared_expert=False,
                    scoring_func=1,
                    routed_scaling_factor=1.0,
                    n_group=2,
                    topk_group=1,
                    topk_method="noaux_tc",
                    correction_bias=True,
                )
                self.assertEqual(block._use_fast_group_topk, expect_fast)
                self.assertEqual(block.group_topk is not None, expect_fast)
                self.assertEqual(group_topk.call_count, int(expect_fast))

    def test_noaux_top1_normalization_uses_reference_path_on_cuda(self):
        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.get_device_type",
                return_value=DeviceType.Cuda,
            ),
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.GroupTopK"
            ) as group_topk,
            torch.device("cpu"),
        ):
            block = DeepSeekV32MoEBlock(
                hidden_size=8,
                moe_intermediate_size=4,
                num_experts=4,
                top_k=1,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=_router_model_config(
                    moe_k=1,
                    has_moe_norm=True,
                    moe_n_group=2,
                    moe_topk_group=1,
                ),
                parallelism_config=_single_rank_parallelism_config(),
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=1,
                routed_scaling_factor=1.0,
                n_group=2,
                topk_group=1,
                topk_method="noaux_tc",
                has_moe_norm=True,
                correction_bias=True,
            )
        self.assertFalse(block._use_fast_group_topk)
        self.assertIsNone(block.group_topk)
        group_topk.assert_not_called()

    def test_noaux_router_exceeding_warp_capacity_uses_reference_path(self):
        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.get_device_type",
                return_value=DeviceType.Cuda,
            ),
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.GroupTopK"
            ) as group_topk,
            torch.device("cpu"),
        ):
            block = DeepSeekV32MoEBlock(
                hidden_size=8,
                moe_intermediate_size=4,
                num_experts=64,
                top_k=2,
                layer_idx=3,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=_router_model_config(
                    expert_num=64,
                    moe_n_group=64,
                    moe_topk_group=2,
                ),
                parallelism_config=_single_rank_parallelism_config(),
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=1,
                routed_scaling_factor=1.0,
                n_group=64,
                topk_group=2,
                topk_method="noaux_tc",
                correction_bias=True,
            )
        self.assertFalse(block._use_fast_group_topk)
        self.assertIsNone(block.group_topk)
        group_topk.assert_not_called()

    def test_greedy_top1_normalization_uses_reference_path_on_cuda(self):
        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.get_device_type",
                return_value=DeviceType.Cuda,
            ),
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.SelectTopk"
            ) as select_topk,
            torch.device("cpu"),
        ):
            block = DeepSeekV32MoEBlock(
                hidden_size=8,
                moe_intermediate_size=4,
                num_experts=4,
                top_k=1,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=_router_model_config(
                    moe_k=1,
                    has_moe_norm=True,
                    scoring_func=0,
                ),
                parallelism_config=_single_rank_parallelism_config(),
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=0,
                routed_scaling_factor=1.0,
                n_group=1,
                topk_group=1,
                topk_method="greedy",
                has_moe_norm=True,
                correction_bias=False,
            )
        self.assertFalse(block._use_fast_select_topk)
        self.assertIsNone(block.select_topk)
        select_topk.assert_not_called()

    def test_non_noaux_router_avoids_cuda_select_topk_on_other_devices(self):
        parallelism_config = _single_rank_parallelism_config()
        moe_config = types.SimpleNamespace(fake_balance_expert=False)
        model_config = _router_model_config(scoring_func=0)
        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.get_device_type",
                return_value=DeviceType.ROCm,
            ),
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.SelectTopk"
            ) as select_topk,
            torch.device("cpu"),
        ):
            block = DeepSeekV32MoEBlock(
                hidden_size=8,
                moe_intermediate_size=4,
                num_experts=4,
                top_k=2,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=model_config,
                parallelism_config=parallelism_config,
                moe_config=moe_config,
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=0,
                routed_scaling_factor=1.0,
                n_group=1,
                topk_group=1,
                topk_method="greedy",
                correction_bias=False,
            )
        self.assertFalse(block._use_fast_select_topk)
        self.assertIsNone(block.select_topk)
        select_topk.assert_not_called()

    def test_moe_rejects_model_config_routing_mismatch(self):
        mismatch_configs = {
            "hidden_size": _router_model_config(hidden_size=16, scoring_func=0),
            "expert_num": _router_model_config(expert_num=8, scoring_func=0),
            "moe_k": _router_model_config(moe_k=1, scoring_func=0),
            "has_moe_norm": _router_model_config(
                has_moe_norm=True,
                scoring_func=0,
            ),
            "moe_n_group": _router_model_config(moe_n_group=2, scoring_func=0),
            "moe_topk_group": _router_model_config(
                moe_topk_group=2,
                scoring_func=0,
            ),
            "scoring_func": _router_model_config(scoring_func=1),
            "routed_scaling_factor": _router_model_config(
                scoring_func=0,
                routed_scaling_factor=2.0,
            ),
        }
        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.get_device_type",
                return_value=DeviceType.Cuda,
            ),
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.SelectTopk"
            ) as select_topk,
            torch.device("cpu"),
        ):
            for field, model_config in mismatch_configs.items():
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(
                        ValueError,
                        field,
                    ),
                ):
                    DeepSeekV32MoEBlock(
                        hidden_size=8,
                        moe_intermediate_size=4,
                        num_experts=4,
                        top_k=2,
                        layer_idx=0,
                        tp_size=1,
                        tp_rank=0,
                        ep_size=1,
                        ep_rank=0,
                        model_config=model_config,
                        parallelism_config=_single_rank_parallelism_config(),
                        moe_config=types.SimpleNamespace(fake_balance_expert=False),
                        quant_config=None,
                        params_dtype=torch.float32,
                        has_shared_expert=False,
                        scoring_func=0,
                        routed_scaling_factor=1.0,
                        n_group=1,
                        topk_group=1,
                        topk_method="greedy",
                        has_moe_norm=False,
                        correction_bias=False,
                    )
        select_topk.assert_not_called()

    def test_moe_forward_uses_reference_noaux_routing_on_cpu(self):
        parallelism_config = _single_rank_parallelism_config()
        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.get_device_type",
                return_value=DeviceType.ROCm,
            ),
            torch.device("cpu"),
        ):
            block = DeepSeekV32MoEBlock(
                hidden_size=4,
                moe_intermediate_size=2,
                num_experts=4,
                top_k=2,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=_router_model_config(
                    hidden_size=4,
                    has_moe_norm=True,
                    moe_n_group=2,
                    moe_topk_group=1,
                    routed_scaling_factor=2.0,
                ),
                parallelism_config=parallelism_config,
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=1,
                routed_scaling_factor=2.0,
                n_group=2,
                topk_group=1,
                topk_method="noaux_tc",
                has_moe_norm=True,
                correction_bias=True,
            )

        class CapturingExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fused_moe = types.SimpleNamespace(topk_ids_dtype=torch.int64)
                self.topk_ids_dtype = torch.int64
                self.weights = None
                self.ids = None

            def forward(self, hidden_states, topk_weights, topk_ids):
                self.weights = topk_weights.clone()
                self.ids = topk_ids.clone()
                return hidden_states + topk_weights.sum(dim=-1, keepdim=True)

        capturing_experts = CapturingExperts()
        block.experts = capturing_experts
        hidden_states = torch.tensor([[1.0, 0.5, -1.0, 2.0], [-0.5, 1.5, 0.25, -1.0]])
        with torch.no_grad():
            block.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )
            block.gate.e_score_correction_bias.copy_(
                torch.tensor([0.1, -0.2, 0.3, -0.4])
            )

        expected_weights, expected_ids = _manual_noaux_reference(
            block.gate(hidden_states).float(),
            block.gate.e_score_correction_bias,
            top_k=2,
            n_group=2,
            topk_group=1,
            renormalize=True,
            routed_scaling_factor=2.0,
        )
        output = block(hidden_states)

        self.assertTrue(torch.equal(capturing_experts.ids, expected_ids))
        self.assertTrue(torch.equal(capturing_experts.weights, expected_weights))
        self.assertTrue(
            torch.equal(
                output,
                hidden_states + expected_weights.sum(dim=-1, keepdim=True),
            )
        )

    def test_moe_rejects_invalid_fake_balance_and_input_rank(self):
        kwargs = dict(
            hidden_size=4,
            moe_intermediate_size=2,
            num_experts=4,
            top_k=2,
            layer_idx=0,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            model_config=_router_model_config(
                hidden_size=4,
                scoring_func=0,
                routed_scaling_factor=2.0,
            ),
            parallelism_config=_single_rank_parallelism_config(),
            quant_config=None,
            params_dtype=torch.float32,
            has_shared_expert=False,
            scoring_func=0,
            routed_scaling_factor=2.0,
            n_group=1,
            topk_group=1,
            topk_method="greedy",
            correction_bias=False,
        )
        with self.assertRaisesRegex(TypeError, "fake_balance_expert"):
            DeepSeekV32MoEBlock(
                moe_config=types.SimpleNamespace(fake_balance_expert=1),
                **kwargs,
            )
        no_parallelism = dict(kwargs)
        no_parallelism["parallelism_config"] = None
        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.get_device_type",
                return_value=DeviceType.Cuda,
            ),
            self.assertRaisesRegex(ValueError, "parallelism_config"),
        ):
            DeepSeekV32MoEBlock(
                moe_config=types.SimpleNamespace(fake_balance_expert=True),
                **no_parallelism,
            )
        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.get_device_type",
                return_value=DeviceType.ROCm,
            ),
            self.assertRaisesRegex(RuntimeError, "only on CUDA"),
        ):
            DeepSeekV32MoEBlock(
                moe_config=types.SimpleNamespace(fake_balance_expert=True),
                **kwargs,
            )

        mismatched_shared = dict(kwargs)
        mismatched_shared["has_shared_expert"] = True
        with self.assertRaisesRegex(ValueError, "has_shared_expert"):
            DeepSeekV32MoEBlock(
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                **mismatched_shared,
            )

        block = DeepSeekV32MoEBlock(
            moe_config=types.SimpleNamespace(fake_balance_expert=False),
            **kwargs,
        )
        block.experts.fused_moe = types.SimpleNamespace(topk_ids_dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            block(torch.zeros(1, 2, 4))

    @unittest.skipUnless(is_cuda(), "CUDA GroupTopK is required")
    def _gpu_moe_cuda_noaux_router_matches_reference(self):
        with torch.device("cuda"):
            block = DeepSeekV32MoEBlock(
                hidden_size=4,
                moe_intermediate_size=2,
                num_experts=8,
                top_k=2,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=_router_model_config(
                    hidden_size=4,
                    expert_num=8,
                    has_moe_norm=True,
                    moe_n_group=4,
                    moe_topk_group=2,
                    routed_scaling_factor=2.5,
                ),
                parallelism_config=_single_rank_parallelism_config(),
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=1,
                routed_scaling_factor=2.5,
                n_group=4,
                topk_group=2,
                topk_method="noaux_tc",
                has_moe_norm=True,
                correction_bias=True,
            )
        self.assertTrue(block._use_fast_group_topk)

        class CapturingExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fused_moe = types.SimpleNamespace(topk_ids_dtype=torch.int32)
                self.topk_ids_dtype = torch.int32
                self.weights = None
                self.ids = None

            def forward(self, hidden_states, topk_weights, topk_ids):
                self.weights = topk_weights.clone()
                self.ids = topk_ids.clone()
                return hidden_states

        capturing_experts = CapturingExperts()
        block.experts = capturing_experts
        hidden_states = torch.tensor(
            [
                [1.0, 0.5, -1.0, 2.0],
                [-0.5, 1.5, 0.25, -1.0],
                [0.75, -0.25, 1.25, 0.5],
            ],
            device="cuda",
        )
        with torch.no_grad():
            block.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.5, -0.5, 0.25, 0.0],
                        [-0.25, 0.75, 0.0, 0.5],
                        [0.0, 0.25, 0.75, -0.5],
                        [0.5, 0.0, -0.25, 0.75],
                    ],
                    device="cuda",
                )
            )
            block.gate.e_score_correction_bias.copy_(
                torch.tensor(
                    [0.1, -0.2, 0.3, -0.4, 0.05, 0.15, -0.1, 0.2],
                    device="cuda",
                )
            )

        router_logits = block.gate(hidden_states).float()
        expected_weights, expected_ids = _manual_noaux_reference(
            router_logits,
            block.gate.e_score_correction_bias,
            top_k=2,
            n_group=4,
            topk_group=2,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        output = block(hidden_states)

        actual_order = capturing_experts.ids.argsort(dim=-1)
        expected_order = expected_ids.argsort(dim=-1)
        actual_ids = capturing_experts.ids.gather(1, actual_order)
        expected_ids = expected_ids.gather(1, expected_order).to(actual_ids.dtype)
        actual_weights = capturing_experts.weights.gather(1, actual_order)
        expected_weights = expected_weights.gather(1, expected_order)
        self.assertTrue(torch.equal(actual_ids, expected_ids))
        self.assertTrue(torch.allclose(actual_weights, expected_weights, atol=1e-6))
        self.assertTrue(torch.equal(output, hidden_states))

    @unittest.skipUnless(is_cuda(), "CUDA SelectTopk is required")
    def _gpu_moe_cuda_fast_select_topk_matches_reference(self):
        model_config = ModelConfig()
        model_config.attn_config.head_num = 1
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 1
        model_config.max_seq_len = 1
        model_config.vocab_size = 16
        model_config.hidden_size = 4
        model_config.expert_num = 8
        model_config.moe_k = 2
        model_config.has_moe_norm = True
        model_config.scoring_func = 0
        model_config.routed_scaling_factor = 1.0
        model_config.moe_n_group = 1
        model_config.moe_topk_group = 1
        with torch.device("cuda"):
            block = DeepSeekV32MoEBlock(
                hidden_size=4,
                moe_intermediate_size=2,
                num_experts=8,
                top_k=2,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=model_config,
                parallelism_config=_single_rank_parallelism_config(),
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=0,
                routed_scaling_factor=1.0,
                n_group=1,
                topk_group=1,
                topk_method="greedy",
                has_moe_norm=True,
                correction_bias=False,
            )
        self.assertTrue(block._use_fast_select_topk)

        class CapturingExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fused_moe = types.SimpleNamespace(topk_ids_dtype=torch.int32)
                self.topk_ids_dtype = torch.int32
                self.weights = None
                self.ids = None

            def forward(self, hidden_states, topk_weights, topk_ids):
                self.weights = topk_weights.clone()
                self.ids = topk_ids.clone()
                return hidden_states

        capturing_experts = CapturingExperts()
        block.experts = capturing_experts
        generator = torch.Generator(device="cuda").manual_seed(20260728)
        hidden_states = torch.randn(
            (17, 4),
            generator=generator,
            device="cuda",
        )
        with torch.no_grad():
            block.gate.weight.copy_(
                torch.randn(
                    block.gate.weight.shape,
                    generator=generator,
                    device="cuda",
                )
            )

        expected_weights, expected_ids = (
            block.gate(hidden_states)
            .float()
            .softmax(dim=-1)
            .topk(2, dim=-1, sorted=False)
        )
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
        block(hidden_states)

        actual_order = capturing_experts.ids.argsort(dim=-1)
        expected_order = expected_ids.argsort(dim=-1)
        actual_ids = capturing_experts.ids.gather(1, actual_order)
        expected_ids = expected_ids.gather(1, expected_order).to(actual_ids.dtype)
        actual_weights = capturing_experts.weights.gather(1, actual_order)
        expected_weights = expected_weights.gather(1, expected_order)
        self.assertTrue(torch.equal(actual_ids, expected_ids))
        torch.testing.assert_close(actual_weights, expected_weights)

    @unittest.skipUnless(is_hip(), "ROCm is required")
    def _gpu_moe_rocm_noaux_reference_router_matches_expected(self):
        with torch.device("cuda"):
            block = DeepSeekV32MoEBlock(
                hidden_size=4,
                moe_intermediate_size=2,
                num_experts=8,
                top_k=2,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=_router_model_config(
                    hidden_size=4,
                    expert_num=8,
                    has_moe_norm=True,
                    moe_n_group=4,
                    moe_topk_group=2,
                    routed_scaling_factor=2.5,
                ),
                parallelism_config=_single_rank_parallelism_config(),
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=1,
                routed_scaling_factor=2.5,
                n_group=4,
                topk_group=2,
                topk_method="noaux_tc",
                has_moe_norm=True,
                correction_bias=True,
            )
        self.assertFalse(block._use_fast_group_topk)
        self.assertIsNone(block.group_topk)

        class CapturingExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fused_moe = types.SimpleNamespace(topk_ids_dtype=torch.int32)
                self.topk_ids_dtype = torch.int32
                self.weights = None
                self.ids = None

            def forward(self, hidden_states, topk_weights, topk_ids):
                self.weights = topk_weights.clone()
                self.ids = topk_ids.clone()
                return hidden_states

        capturing_experts = CapturingExperts()
        block.experts = capturing_experts
        hidden_states = torch.tensor(
            [
                [1.0, 0.5, -1.0, 2.0],
                [-0.5, 1.5, 0.25, -1.0],
            ],
            device="cuda",
        )
        with torch.no_grad():
            block.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 1.0],
                    ],
                    device="cuda",
                )
            )
            block.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.2, 0.2, 8, device="cuda")
            )

        expected_weights, expected_ids = _manual_noaux_reference(
            block.gate(hidden_states).float(),
            block.gate.e_score_correction_bias,
            top_k=2,
            n_group=4,
            topk_group=2,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        block(hidden_states)
        self.assertTrue(torch.equal(capturing_experts.ids, expected_ids))
        torch.testing.assert_close(capturing_experts.weights, expected_weights)

    def test_mla_misaligned_fp8_a_projection_uses_bf16_fallback(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=128,
            num_heads=2,
            q_lora_rank=129,
            kv_lora_rank=128,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            quant_config=QuantizationConfig("FP8_PER_BLOCK"),
            params_dtype=torch.bfloat16,
        )
        for parameter in attention.parameters():
            parameter.data.zero_()
        attention.process_weights_after_loading()
        self.assertIsNone(attention._fused_qkv_a_runtime)
        self.assertEqual(
            tuple(attention._fused_qkv_a_w.shape),
            (129 + 128 + 2, 128),
        )

    def test_mla_non_block_fp8_scales_use_bf16_fallback(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=128,
            num_heads=2,
            q_lora_rank=128,
            kv_lora_rank=128,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            quant_config=QuantizationConfig("FP8_PER_BLOCK"),
            params_dtype=torch.bfloat16,
        )
        for parameter in attention.parameters():
            parameter.data.zero_()
        attention.q_a_proj.weight_scale_inv = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )
        attention.kv_a_proj_with_mqa.weight_scale_inv = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )
        attention.process_weights_after_loading()
        self.assertIsNone(attention._fused_qkv_a_runtime)
        self.assertEqual(
            tuple(attention._fused_qkv_a_w.shape),
            (128 + 128 + 2, 128),
        )

    @unittest.skipUnless(is_cuda(), "requires a CUDA GPU")
    def _gpu_mla_online_fp8_uses_models_py_quantizer(self):
        with torch.device("cuda"):
            attention = DeepSeekV32MlaAttention(
                hidden_size=128,
                num_heads=2,
                q_lora_rank=128,
                kv_lora_rank=128,
                nope_head_dim=2,
                rope_head_dim=2,
                v_head_dim=2,
                layer_idx=0,
                quant_config=QuantizationConfig("fp8_block_online"),
                params_dtype=torch.bfloat16,
            )
        for parameter in attention.parameters():
            parameter.data.zero_()
        attention.process_weights_after_loading()
        self.assertIsNotNone(attention._fused_qkv_a_runtime)
        self.assertEqual(attention._fused_qkv_a_w.dtype, torch.float8_e4m3fn)
        self.assertEqual(
            tuple(attention._fused_qkv_a_runtime.weight.shape),
            (128 + 128 + 2, 128),
        )

    @unittest.skipUnless(is_hip(), "requires a ROCm GPU")
    def _gpu_mla_online_fp8_rocm_keeps_exact_bf16_kv_b_views(self):
        device = torch.device("cuda")
        with torch.device(device):
            attention = DeepSeekV32MlaAttention(
                hidden_size=128,
                num_heads=2,
                q_lora_rank=128,
                kv_lora_rank=128,
                nope_head_dim=64,
                rope_head_dim=128,
                v_head_dim=64,
                layer_idx=0,
                quant_config=QuantizationConfig("fp8_block_online"),
                params_dtype=torch.bfloat16,
            )
        with torch.no_grad():
            for parameter in attention.parameters():
                parameter.zero_()
            checkpoint_weight = torch.arange(
                attention.kv_b_proj.weight.numel(),
                device=device,
                dtype=torch.float32,
            ).reshape_as(attention.kv_b_proj.weight)
            checkpoint_weight = (checkpoint_weight.remainder(97) - 48).to(
                torch.bfloat16
            )
            attention.kv_b_proj.weight.copy_(checkpoint_weight)
        expected_kv_b = checkpoint_weight.t().contiguous()

        NewModelLoader._validate_runtime_backends(attention, "cuda")
        NewModelLoader._migrate_staged_modules(attention, "cuda")
        attention.to(device)
        NewModelLoader._run_post_load_hooks(attention)

        self.assertIsNone(attention._fused_qkv_a_runtime)
        self.assertEqual(attention._fused_qkv_a_w.dtype, torch.bfloat16)
        self.assertIsNone(attention._kv_b_runtime_w)
        self.assertIsNone(attention._kv_b_runtime_s)
        self.assertTrue(torch.equal(attention._kv_b_w, expected_kv_b))
        kernel_weights = attention._build_mla_kernel_weights()
        self.assertTrue(torch.equal(kernel_weights[W.mla_kv_b_w], expected_kv_b))
        self.assertNotIn(W.mla_kv_b_s, kernel_weights)

    @unittest.skipUnless(is_cuda(), "requires a CUDA GPU")
    def _gpu_mla_online_fused_fp8_projection_matches_bf16_reference(self):
        device = torch.device("cuda")
        with torch.device(device):
            attention = DeepSeekV32MlaAttention(
                hidden_size=128,
                num_heads=2,
                q_lora_rank=128,
                kv_lora_rank=128,
                nope_head_dim=64,
                rope_head_dim=128,
                v_head_dim=64,
                layer_idx=0,
                quant_config=QuantizationConfig("fp8_block_online"),
                params_dtype=torch.bfloat16,
            )
        generator = torch.Generator(device=device).manual_seed(20260727)
        q_a_weight = torch.randn(
            attention.q_a_proj.weight.shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        kv_a_weight = torch.randn(
            attention.kv_a_proj_with_mqa.weight.shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        attention.q_a_proj.weight.data.copy_(q_a_weight)
        attention.kv_a_proj_with_mqa.weight.data.copy_(kv_a_weight)
        reference_weight = torch.cat([q_a_weight, kv_a_weight], dim=0)

        attention.process_weights_after_loading()
        self.assertIsNotNone(attention._fused_qkv_a_runtime)
        x = torch.randn(
            (7, 128),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        actual = attention._fused_qkv_a_runtime(x)
        expected = torch.nn.functional.linear(x, reference_weight)
        difference = actual.float() - expected.float()
        relative_rmse = (
            difference.square().mean().sqrt() / expected.float().square().mean().sqrt()
        )
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            expected.float().flatten(),
            dim=0,
        )
        self.assertLess(float(relative_rmse), 0.05)
        self.assertGreater(float(cosine), 0.998)

    @unittest.skipUnless(is_cuda(), "requires a CUDA GPU")
    def _gpu_mla_online_fp8_kc_vc_use_bf16_checkpoint_source(self):
        from rtp_llm.models_py.modules.factory import LinearFactory

        device = torch.device("cuda")
        with torch.device(device):
            attention = DeepSeekV32MlaAttention(
                hidden_size=128,
                num_heads=2,
                q_lora_rank=128,
                kv_lora_rank=128,
                nope_head_dim=64,
                rope_head_dim=128,
                v_head_dim=64,
                layer_idx=0,
                quant_config=QuantizationConfig("fp8_block_online"),
                params_dtype=torch.bfloat16,
            )
        generator = torch.Generator(device=device).manual_seed(20260730)
        checkpoint_weight = torch.randn(
            attention.kv_b_proj.weight.shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        attention.kv_b_proj.weight.data.copy_(checkpoint_weight)
        expected = checkpoint_weight.transpose(0, 1).contiguous().view(128, 2, 128)

        attention.process_weights_after_loading()

        self.assertTrue(
            torch.equal(
                attention._kc_w,
                expected[:, :, :64].permute(1, 2, 0).contiguous(),
            )
        )
        self.assertTrue(
            torch.equal(
                attention._vc_w,
                expected[:, :, 64:].transpose(0, 1).contiguous(),
            )
        )
        self.assertIsNone(attention._kv_b_w)
        self.assertIsNotNone(attention._kv_b_runtime_w)
        self.assertIsNotNone(attention._kv_b_runtime_s)

        kernel_weights = attention._build_mla_kernel_weights()
        linear = LinearFactory.create_linear_from_weights(
            kernel_weights,
            W.mla_kv_b_w,
            W.mla_kv_b_s,
            None,
            quant_config=types.SimpleNamespace(get_method=lambda: "FP8_PER_BLOCK"),
        )
        x = torch.randn(
            (11, checkpoint_weight.shape[1]),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        actual = linear(x)
        reference = F.linear(x, checkpoint_weight)
        difference = actual.float() - reference.float()
        relative_rmse = (
            difference.square().mean().sqrt() / reference.float().square().mean().sqrt()
        )
        cosine = F.cosine_similarity(
            actual.float().flatten(),
            reference.float().flatten(),
            dim=0,
        )
        self.assertLess(float(relative_rmse), 0.05)
        self.assertGreater(float(cosine), 0.998)

    @unittest.skipUnless(is_cuda(), "requires a CUDA GPU")
    def _gpu_mla_prequantized_fp8_kernel_views_execute_numerically(self):
        from rtp_llm.models_py.kernels.cuda.fp8_quant import per_block_cast_to_fp8
        from rtp_llm.models_py.modules.factory import LinearFactory

        device = torch.device("cuda")
        quant_config = QuantizationConfig("FP8_PER_BLOCK")
        with torch.device(device):
            attention = DeepSeekV32MlaAttention(
                hidden_size=128,
                num_heads=2,
                q_lora_rank=128,
                kv_lora_rank=128,
                nope_head_dim=64,
                rope_head_dim=128,
                v_head_dim=64,
                layer_idx=0,
                quant_config=quant_config,
                params_dtype=torch.bfloat16,
            )

        generator = torch.Generator(device=device).manual_seed(20260728)
        reference_weights = {}
        for name, linear in (
            ("q_a", attention.q_a_proj),
            ("kv_a", attention.kv_a_proj_with_mqa),
            ("q_b", attention.q_b_proj),
            ("kv_b", attention.kv_b_proj),
            ("o", attention.o_proj),
        ):
            reference = torch.randn(
                linear.weight.shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            fp8_weight, scale = per_block_cast_to_fp8(
                reference,
                use_ue8m0=False,
            )
            linear.weight.data.copy_(fp8_weight)
            linear.weight_scale_inv.data.copy_(scale)
            reference_weights[name] = reference

        NewModelLoader._run_post_load_hooks(attention)
        kernel_weights = attention._build_mla_kernel_weights()
        self.assertIsNone(attention._kv_b_w)

        cases = ((W.mla_kv_b_w, W.mla_kv_b_s, reference_weights["kv_b"]),)
        runtime_quant_config = types.SimpleNamespace(get_method=lambda: "FP8_PER_BLOCK")
        for weight_key, scale_key, reference in cases:
            with self.subTest(weight_key=weight_key):
                linear = LinearFactory.create_linear_from_weights(
                    kernel_weights,
                    weight_key,
                    scale_key,
                    None,
                    quant_config=runtime_quant_config,
                )
                x = torch.randn(
                    (11, reference.shape[1]),
                    generator=generator,
                    device=device,
                    dtype=torch.bfloat16,
                )
                actual = linear(x)
                expected = F.linear(x, reference)
                difference = actual.float() - expected.float()
                relative_rmse = (
                    difference.square().mean().sqrt()
                    / expected.float().square().mean().sqrt()
                )
                cosine = F.cosine_similarity(
                    actual.float().flatten(),
                    expected.float().flatten(),
                    dim=0,
                )
                self.assertLess(float(relative_rmse), 0.05)
                self.assertGreater(float(cosine), 0.998)

    def test_mlp_tp_reduction_is_owned_by_row_parallel_linear(self):
        dense = DeepSeekV32MLP(
            hidden_size=4,
            intermediate_size=4,
            tp_size=2,
            tp_rank=0,
            params_dtype=torch.float32,
            reduce_output=True,
        )
        shared = DeepSeekV32MLP(
            hidden_size=4,
            intermediate_size=4,
            tp_size=2,
            tp_rank=0,
            params_dtype=torch.float32,
            reduce_output=False,
        )
        self.assertTrue(dense.down_proj.reduce_output)
        self.assertFalse(shared.down_proj.reduce_output)

    def test_fp8_mlp_pads_weights_and_scales_before_tp_split(self):
        hidden_size = 128
        intermediate_size = 257
        block_size = 128
        mlp = DeepSeekV32MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=2,
            tp_rank=1,
            quant_config=QuantizationConfig("FP8_PER_BLOCK"),
            params_dtype=torch.bfloat16,
        )
        self.assertEqual(mlp.padded_intermediate_size, 512)

        gate = (
            torch.arange(intermediate_size * hidden_size, dtype=torch.float32)
            .reshape(intermediate_size, hidden_size)
            .remainder(17)
            .sub(8)
            .to(torch.float8_e4m3fn)
        )
        up = (
            torch.arange(intermediate_size * hidden_size, dtype=torch.float32)
            .reshape(intermediate_size, hidden_size)
            .remainder(13)
            .sub(6)
            .to(torch.float8_e4m3fn)
        )
        down = (
            torch.arange(hidden_size * intermediate_size, dtype=torch.float32)
            .reshape(hidden_size, intermediate_size)
            .remainder(11)
            .sub(5)
            .to(torch.float8_e4m3fn)
        )
        scale_blocks = (intermediate_size + block_size - 1) // block_size
        gate_scale = torch.arange(1, scale_blocks + 1, dtype=torch.float32).reshape(
            scale_blocks, 1
        )
        up_scale = gate_scale + 10
        down_scale = gate_scale.t() + 20

        mlp.load_weights(
            {
                "gate_proj.weight": gate,
                "gate_proj.weight_scale_inv": gate_scale,
                "up_proj.weight": up,
                "up_proj.weight_scale_inv": up_scale,
                "down_proj.weight": down,
                "down_proj.weight_scale_inv": down_scale,
            }
        )

        expected_gate = torch.zeros(256, hidden_size, dtype=torch.float8_e4m3fn)
        expected_gate[0] = gate[-1]
        expected_up = torch.zeros(256, hidden_size, dtype=torch.float8_e4m3fn)
        expected_up[0] = up[-1]
        self.assertTrue(torch.equal(mlp.gate_up_proj.weight[:256], expected_gate))
        self.assertTrue(torch.equal(mlp.gate_up_proj.weight[256:], expected_up))

        expected_down = torch.zeros(hidden_size, 256, dtype=torch.float8_e4m3fn)
        expected_down[:, 0] = down[:, -1]
        self.assertTrue(torch.equal(mlp.down_proj.weight, expected_down))

        expected_gate_scale = torch.tensor([[3.0], [0.0]])
        expected_up_scale = torch.tensor([[13.0], [0.0]])
        self.assertTrue(
            torch.equal(
                mlp.gate_up_proj.weight_scale_inv[:2],
                expected_gate_scale,
            )
        )
        self.assertTrue(
            torch.equal(
                mlp.gate_up_proj.weight_scale_inv[2:],
                expected_up_scale,
            )
        )
        self.assertTrue(
            torch.equal(
                mlp.down_proj.weight_scale_inv,
                torch.tensor([[23.0, 0.0]]),
            )
        )

    def test_runtime_weight_view_exports_only_runtime_globals(self):
        for model_type in (DeepSeekV32ForCausalLM, DeepSeekV32MTPForCausalLM):
            with self.subTest(model_type=model_type.__name__):
                model = _uninitialized_model(model_type)
                model.embed_tokens = torch.nn.Embedding(4, 2)
                model.norm = torch.nn.LayerNorm(2)
                model.lm_head = torch.nn.Linear(2, 4, bias=False)
                view = model.runtime_weight_view()
                self.assertEqual(
                    set(view),
                    {"embedding", "final_layernorm.gamma", "lm_head"},
                )
                self.assertIs(view["embedding"], model.embed_tokens.weight)
                self.assertIs(view["lm_head"], model.lm_head.weight)

    def test_initialize_failure_does_not_build_layout_or_release_weights(self):
        model = _uninitialized_model(DeepSeekV32ForCausalLM)
        model._mla_kernel_layout = None
        model._keep_mla_checkpoint_weights = False
        with (
            mock.patch(
                "rtp_llm.models_py.model_desc.module_base.GptModelBase.initialize",
                return_value=False,
            ),
            mock.patch.object(model, "_ensure_mla_kernel_layout") as ensure_layout,
        ):
            self.assertFalse(model.initialize(None))
        ensure_layout.assert_not_called()

    def test_keep_mla_checkpoint_weights_is_typed_and_prevents_release(self):
        with self.assertRaisesRegex(TypeError, "keep_mla_checkpoint_weights"):
            NewLoaderConfig(keep_mla_checkpoint_weights=1)

        model = _uninitialized_model(DeepSeekV32ForCausalLM)
        model._mla_kernel_layout = None
        model._keep_mla_checkpoint_weights = True
        attention = mock.Mock()
        model.layers = [types.SimpleNamespace(self_attn=attention)]
        with (
            mock.patch(
                "rtp_llm.models_py.model_desc.module_base.GptModelBase.initialize",
                return_value=True,
            ),
            mock.patch.object(model, "_ensure_mla_kernel_layout") as ensure_layout,
        ):
            self.assertTrue(model.initialize(None))
        ensure_layout.assert_called_once_with()
        attention.release_checkpoint_only_weights.assert_not_called()

    def test_mtp_block_rejects_inconsistent_inputs(self):
        block = MTPBlock(hidden_size=4, params_dtype=torch.float32)
        self.assertEqual(block.fc.prefix, "mtp_block.fc")
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            block(torch.zeros(2, 4), torch.zeros(1, 4))
        with self.assertRaisesRegex(TypeError, "share a dtype"):
            block(
                torch.zeros(2, 4, dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float64),
            )

    def test_mtp_block_concat_order_is_numerically_explicit(self):
        inputs_embeds = torch.tensor([[1.0, 2.0]])
        last_hidden_states = torch.tensor([[3.0, 5.0]])
        projection = torch.tensor(
            [
                [1.0, 10.0, 100.0, 1000.0],
                [-2.0, 3.0, -5.0, 7.0],
            ]
        )
        for reverse_concat, expected_concat in (
            (False, torch.tensor([[1.0, 2.0, 3.0, 5.0]])),
            (True, torch.tensor([[3.0, 5.0, 1.0, 2.0]])),
        ):
            with self.subTest(reverse_concat=reverse_concat):
                block = MTPBlock(
                    hidden_size=2,
                    reverse_concat=reverse_concat,
                    params_dtype=torch.float32,
                )
                block.e_norm = torch.nn.Identity()
                block.h_norm = torch.nn.Identity()
                block.fc.weight.data.copy_(projection)
                actual = block(inputs_embeds, last_hidden_states)
                expected = F.linear(expected_concat, projection)
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
