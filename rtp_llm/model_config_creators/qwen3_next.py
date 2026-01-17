"""Configuration creators for Qwen3 Next models."""

import logging
from typing import List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator
from rtp_llm.ops import HybridAttentionType

logger = logging.getLogger(__name__)


@register_config_creator("qwen3_next")
def create_qwen3_next_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen3 Next model configuration."""
    config_json = require_config_json(ckpt_path)

    config = ModelConfig()
    config.ckpt_path = ckpt_path
    # Basic model structure
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.kv_head_num = config_json["num_key_value_heads"]
    config.attn_config.size_per_head = config_json["head_dim"]
    config.num_layers = config_json["num_hidden_layers"]
    config.hidden_size = config_json["hidden_size"]
    config.vocab_size = config_json["vocab_size"]
    config.max_seq_len = config_json["max_position_embeddings"]

    # RoPE configuration
    config.attn_config.rope_config.style = 1
    config.attn_config.rope_config.base = config_json["rope_theta"]
    config.partial_rotary_factor = config_json["partial_rotary_factor"]
    config.attn_config.rope_config.dim = int(
        config.attn_config.size_per_head * config.partial_rotary_factor
    )

    # Normalization
    config.layernorm_eps = config_json["rms_norm_eps"]
    config.norm_type = "rmsnorm"
    config.has_pre_decoder_layernorm = False
    config.has_post_decoder_layernorm = True
    config.qk_norm = True

    # Activation
    config.activation_type = "SiGLU"

    # MoE configuration
    config.moe_k = config_json["num_experts_per_tok"]
    config.expert_num = config_json["num_experts"]
    config.moe_inter_size = config_json["moe_intermediate_size"]
    config.inter_size = config_json["shared_expert_intermediate_size"]
    config.has_moe_norm = config_json.get("norm_topk_prob", False)
    config.moe_style = 2  # shared + expert

    # MoE layer indices
    moe_step = config_json["decoder_sparse_step"]
    moe_layer_index = []
    for i in range(config.num_layers):
        if (i + 1) % moe_step == 0:
            moe_layer_index.append(i)
    config.moe_layer_index = moe_layer_index

    # Hybrid attention configuration
    attention_step = config_json["full_attention_interval"]
    config.hybrid_attention_config.enable_hybrid_attention = True
    hybrid_layer_types: List[HybridAttentionType] = []
    for i in range(config.num_layers):
        if (i + 1) % attention_step == 0:
            hybrid_layer_types.append(HybridAttentionType.NONE)
        else:
            hybrid_layer_types.append(HybridAttentionType.LINEAR)
    config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types

    # Linear attention configuration
    config.linear_attention_config.linear_conv_kernel_dim = config_json[
        "linear_conv_kernel_dim"
    ]
    config.linear_attention_config.linear_key_head_dim = config_json[
        "linear_key_head_dim"
    ]
    config.linear_attention_config.linear_num_key_heads = config_json[
        "linear_num_key_heads"
    ]
    config.linear_attention_config.linear_num_value_heads = config_json[
        "linear_num_value_heads"
    ]
    config.linear_attention_config.linear_value_head_dim = config_json[
        "linear_value_head_dim"
    ]

    return config
