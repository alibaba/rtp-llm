"""Configuration creators for Mixtbstars models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("mixtbstars")
@register_config_creator("mixtbstars-mtp")
def create_mixtbstars_config(ckpt_path: str) -> ModelConfig:
    """Create Mixtbstars model configuration."""
    config_json = require_config_json(ckpt_path)
    size_per_head = config_json["hidden_size"] // config_json["num_attention_heads"]

    # Determine inter_size and moe_inter_size based on configuration
    moe_intermediate_size = config_json.get("intermediate_size", 8192)
    num_shared_experts = config_json.get("num_shared_experts", 0)

    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.size_per_head = size_per_head
    config.inter_size = (
        8192
        if config_json.get("first_k_dense_layers") is None
        else num_shared_experts * moe_intermediate_size
    )
    config.num_layers = config_json["num_hidden_layers"]
    config.max_seq_len = config_json.get("max_sequence_length", 2048)
    config.vocab_size = config_json["vocab_size"]
    config.attn_config.kv_head_num = config_json.get(
        "num_key_value_heads", config_json["num_attention_heads"]
    )
    config.activation_type = "SiGLU"
    config.norm_type = "rmsnorm"
    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-05)
    config.has_moe_norm = True
    config.attn_config.rope_config.dim = size_per_head
    config.attn_config.rope_config.style = 1
    config.has_post_decoder_layernorm = True
    config.has_pre_decoder_layernorm = False
    config.attn_config.rope_config.base = config_json.get("rope_theta", 10000)
    config.expert_num = config_json.get(
        "num_local_experts", config_json.get("num_routed_experts")
    )
    config.moe_k = config_json["num_experts_per_tok"]
    config.moe_normalize_expert_scale = (
        False if config_json.get("first_k_dense_layers") is None else True
    )
    config.moe_layer_index = [
        i
        for i in range(
            config_json.get("first_k_dense_layers", 0),
            config_json["num_hidden_layers"],
        )
    ]
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

    # Set moe_inter_size for MOE expert intermediate size
    config.moe_inter_size = moe_intermediate_size

    if (
        config_json.get("first_k_dense_layers") is None
        or config_json.get("first_k_dense_layers") == 0
    ):
        if (
            config_json.get("num_shared_experts") is not None
            and config_json.get("num_shared_experts") != 0
        ):
            config.moe_style = 2
        else:
            config.moe_style = 1
    else:
        config.moe_style = 2

    return config
