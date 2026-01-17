"""Configuration creator for TBStars2_5 model."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def create_tbstars2_5_config(ckpt_path: str) -> ModelConfig:
    """Create TBStars2_5 model configuration from checkpoint path.

    This function extracts the configuration creation logic from TBStars2_5._create_config,
    allowing configuration creation without instantiating the model class.

    Args:
        ckpt_path: Path to the TBStars2_5 model checkpoint directory

    Returns:
        ModelConfig instance with TBStars2_5-specific settings

    Raises:
        FileNotFoundError: If config.json doesn't exist
        json.JSONDecodeError: If config.json is invalid JSON
    """
    config_json = require_config_json(ckpt_path)

    # Check for MTP model
    num_mtp_layers = config_json.get("num_mtp_layers", 0)
    if num_mtp_layers > 0:
        logger.warning(
            f"Detected MTP model with num_mtp_layers={num_mtp_layers}. "
            f"Please use TBStars2_5Mtp class instead of TBStars2_5 for proper MTP support."
        )

    hidden_size = config_json["hidden_size"]
    num_attention_heads = config_json["num_attention_heads"]
    size_per_head = config_json.get("head_dim", hidden_size // num_attention_heads)

    # MoE configuration
    num_routed_experts = config_json.get("num_routed_experts", 160)
    num_shared_experts = config_json.get("num_shared_experts", 2)
    num_experts_per_tok = config_json.get("num_experts_per_tok", 8)
    first_k_dense_layers = config_json.get("first_k_dense_layers", 0)
    intermediate_size = config_json.get("intermediate_size", 768)

    # MLA (Multi-head Latent Attention) configuration
    q_lora_rank = config_json.get("q_lora_rank", 1536)
    kv_lora_rank = config_json.get("kv_lora_rank", 384)
    qk_nope_head_dim = config_json.get("qk_nope_head_dim", 128)
    qk_rope_head_dim = config_json.get("qk_rope_head_dim", 64)
    v_head_dim = config_json.get("v_head_dim", 128)
    qk_layernorm = config_json.get("qk_layernorm", True)

    size_per_head = qk_nope_head_dim + qk_rope_head_dim

    # Create and populate config
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = num_attention_heads
    config.attn_config.size_per_head = size_per_head
    config.inter_size = num_shared_experts * intermediate_size
    config.moe_inter_size = intermediate_size
    config.num_layers = config_json["num_hidden_layers"]
    config.max_seq_len = config_json.get("max_position_embeddings", 8192)
    config.vocab_size = config_json["vocab_size"]
    num_key_value_heads = config_json.get("num_key_value_heads")
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    config.attn_config.kv_head_num = num_key_value_heads
    config.activation_type = "SiGLU"
    config.norm_type = "rmsnorm"
    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-6)
    config.has_moe_norm = True
    config.attn_config.rope_config.dim = qk_rope_head_dim
    config.attn_config.rope_config.style = 0
    config.has_post_decoder_layernorm = True
    config.has_pre_decoder_layernorm = False
    config.attn_config.rope_config.base = config_json.get("rope_theta", 10000)
    config.expert_num = num_routed_experts
    config.moe_k = num_experts_per_tok
    config.moe_layer_index = [
        i for i in range(first_k_dense_layers, config_json["num_hidden_layers"])
    ]
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    config.hidden_size = hidden_size

    # MLA config
    config.attn_config.use_mla = True
    config.attn_config.q_lora_rank = q_lora_rank
    config.attn_config.kv_lora_rank = kv_lora_rank
    config.attn_config.nope_head_dim = qk_nope_head_dim
    config.attn_config.rope_head_dim = qk_rope_head_dim
    config.attn_config.v_head_dim = v_head_dim
    # Store qk_layernorm as a custom attribute
    config.qk_norm = qk_layernorm

    config.attn_config.rope_config.offset = qk_nope_head_dim

    # Set MoE style based on shared experts
    if num_shared_experts > 0:
        config.moe_style = 2
    else:
        config.moe_style = 1

    return config


# Register the configuration creator
register_config_creator("tbstars2_5", create_tbstars2_5_config)
