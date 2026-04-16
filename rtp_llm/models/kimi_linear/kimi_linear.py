import json
import logging
import os
from typing import List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.kimi_linear.kimi_linear_weight import KimiLinearWeight
from rtp_llm.ops import HybridAttentionType


class KimiLinear(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        cls._parse_basic_config(config_json, config)
        cls._parse_mla_config(config_json, config)
        cls._parse_rope_config(config_json, config)
        cls._parse_moe_config(config_json, config)
        cls._parse_hybrid_attention_config(config_json, config)
        cls._parse_linear_attention_config(config_json, config)

        return config

    @staticmethod
    def _parse_basic_config(config_json: dict, config: ModelConfig):
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config_json["num_attention_heads"]
        )
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.inter_size = config_json["intermediate_size"]
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-05)
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "SiGLU"
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.config_dtype = config_json.get("dtype", None)

    @staticmethod
    def _parse_mla_config(config_json: dict, config: ModelConfig):
        config.attn_config.use_mla = True
        q_lora_rank = config_json.get("q_lora_rank")
        config.attn_config.q_lora_rank = (
            int(q_lora_rank) if q_lora_rank is not None else 0
        )
        kv_lora_rank = config_json.get("kv_lora_rank")
        config.attn_config.kv_lora_rank = (
            int(kv_lora_rank) if kv_lora_rank is not None else 0
        )
        config.attn_config.nope_head_dim = config_json["qk_nope_head_dim"]
        config.attn_config.rope_head_dim = config_json["qk_rope_head_dim"]
        config.attn_config.v_head_dim = config_json["v_head_dim"]
        config.attn_config.size_per_head = (
            config.attn_config.nope_head_dim + config.attn_config.rope_head_dim
        )
        config.attn_config.rope_config.dim = config.attn_config.rope_head_dim
        config.attn_config.rope_config.offset = config.attn_config.nope_head_dim

    @staticmethod
    def _parse_rope_config(config_json: dict, config: ModelConfig):
        from rtp_llm.ops import MlaOpsType

        config.attn_config.rope_config.base = int(config_json.get("rope_theta", 10000))
        # No YaRN scaling for Kimi Linear (rope_scaling is null)
        if config.mla_ops_type != MlaOpsType.MHA:
            config.attn_config.rope_config.style = 0
        else:
            config.attn_config.rope_config.style = 5

        rope_scaling = config_json.get("rope_scaling")
        if rope_scaling is not None:
            from rtp_llm.utils.model_weight import yarn_get_mscale

            config.attn_config.rope_config.scale = rope_scaling["factor"]
            config.attn_config.rope_config.factor1 = float(
                rope_scaling.get("beta_slow", 1)
            )
            config.attn_config.rope_config.factor2 = float(
                rope_scaling.get("beta_fast", 32)
            )
            config.attn_config.rope_config.max_pos = rope_scaling[
                "original_max_position_embeddings"
            ]
            scaling_factor = rope_scaling["factor"]
            mscale = rope_scaling["mscale"]
            mscale_all_dim = rope_scaling["mscale_all_dim"]
            config.deepseek_rope_mscale = mscale
            config.deepseek_mscale_all_dim = mscale_all_dim
            config.attn_config.rope_config.mscale = yarn_get_mscale(
                scaling_factor, mscale
            ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
            softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.attn_config.softmax_extra_scale = softmax_mscale * softmax_mscale

    @staticmethod
    def _parse_moe_config(config_json: dict, config: ModelConfig):
        # Kimi Linear uses sigmoid routing like DeepSeek-V3
        moe_router_func = config_json.get("moe_router_activation_func", "sigmoid")
        if moe_router_func == "sigmoid":
            config.scoring_func = 1
        elif moe_router_func == "softmax":
            config.scoring_func = 0
        else:
            raise ValueError(f"Unknown moe_router_activation_func: {moe_router_func}")

        config.routed_scaling_factor = config_json["routed_scaling_factor"]
        config.moe_k = config_json["num_experts_per_token"]
        config.expert_num = config_json["num_experts"]
        moe_intermediate_size = config_json["moe_intermediate_size"]
        config.moe_n_group = config_json.get("num_expert_group", 1)
        config.moe_topk_group = config_json.get("topk_group", 1)

        n_shared_experts = config_json.get("num_shared_experts", 1)
        config.moe_inter_size = moe_intermediate_size
        config.inter_size = n_shared_experts * moe_intermediate_size
        config.has_moe_norm = config_json.get("moe_renormalize", False)
        config.moe_style = 2  # shared + expert

        moe_step = config_json.get("moe_layer_freq", 1)
        first_k_dense_replace = config_json.get("first_k_dense_replace", 1)
        config.moe_layer_index = [
            i
            for i in range(config.num_layers)
            if i >= first_k_dense_replace and i % moe_step == 0
        ]

    @staticmethod
    def _parse_hybrid_attention_config(config_json: dict, config: ModelConfig):
        linear_attn_config = config_json.get("linear_attn_config", {})
        kda_layers = set(linear_attn_config.get("kda_layers", []))
        full_attn_layers = set(linear_attn_config.get("full_attn_layers", []))

        config.hybrid_attention_config.enable_hybrid_attention = True
        hybrid_layer_types: List[HybridAttentionType] = []
        for i in range(config.num_layers):
            # config.json uses 1-based layer indices
            layer_1based = i + 1
            if layer_1based in kda_layers:
                hybrid_layer_types.append(HybridAttentionType.LINEAR)
            elif layer_1based in full_attn_layers:
                hybrid_layer_types.append(HybridAttentionType.NONE)
            else:
                # Default: treat as full attention if not in either list
                hybrid_layer_types.append(HybridAttentionType.NONE)
        config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types

    @staticmethod
    def _parse_linear_attention_config(config_json: dict, config: ModelConfig):
        linear_attn_config = config_json.get("linear_attn_config", {})
        head_dim = linear_attn_config["head_dim"]
        num_heads = linear_attn_config["num_heads"]
        conv_kernel_size = linear_attn_config.get("short_conv_kernel_size", 4)

        config.linear_attention_config.linear_key_head_dim = head_dim
        config.linear_attention_config.linear_value_head_dim = head_dim
        config.linear_attention_config.linear_num_key_heads = num_heads
        config.linear_attention_config.linear_num_value_heads = num_heads
        config.linear_attention_config.linear_conv_kernel_dim = conv_kernel_size

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.utils.arch import is_cuda

        if not is_cuda():
            raise RuntimeError("KimiLinear is only supported in cuda arch")
        from rtp_llm.models_py.model_desc.kimi_linear import KimiLinearModel

        self.py_model = KimiLinearModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return KimiLinearWeight


register_model("kimi_linear", KimiLinear, ["KimiLinearForCausalLM"])
