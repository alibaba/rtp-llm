"""Configuration creators for GLM4 MoE models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _apply_glm4_moe_config_json(config: ModelConfig, config_json: Dict[str, Any]):
    """Apply GLM4 MoE configuration from config.json."""
    config.inter_size = config_json["intermediate_size"]
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.kv_head_num = config_json.get(
        "num_key_value_heads", config.attn_config.head_num
    )
    config.attn_config.size_per_head = (
        int(config_json.get("head_dim", 0))
        if "head_dim" in config_json
        else config_json["hidden_size"] // config.attn_config.head_num
    )
    if config_json.get("hidden_size") is not None:
        config.hidden_size = config_json["hidden_size"]
    config.num_layers = config_json["num_hidden_layers"]
    config.attn_config.rope_config.base = int(
        config_json.get("rope_theta", config.attn_config.rope_config.base)
    )
    config.vocab_size = config_json["vocab_size"]
    config.attn_config.rope_config.dim = config.attn_config.size_per_head
    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    config.config_dtype = config_json.get("torch_dtype", None)

    # MoE config
    config.expert_num = config_json.get("num_routed_experts", 0)
    config.moe_k = config_json.get("num_experts_per_tok", 8)
    config.moe_inter_size = config_json.get("moe_intermediate_size", config.inter_size)
    first_k_dense_layers = config_json.get("first_k_dense_layers", 0)
    config.moe_layer_index = [i for i in range(first_k_dense_layers, config.num_layers)]
    config.moe_style = config_json.get("moe_style", 1)
    config.routed_scaling_factor = config_json.get("routed_scaling_factor", 1.0)
    config.qk_norm = config_json.get("qk_norm", False)

    logger.info(
        f"glm4 moe config use_qk_norm: {config.qk_norm}, routed_scaling_factor: {config.routed_scaling_factor}"
    )


@register_config_creator("glm4_moe")
def create_glm4_moe_config(ckpt_path: str) -> ModelConfig:
    """Create GLM4 MoE model configuration."""
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = 0
    config.attn_config.kv_head_num = 0
    config.attn_config.size_per_head = 0
    config.num_layers = 0
    config.inter_size = 0
    config.vocab_size = 152064
    config.max_seq_len = 8192
    config.attn_config.rope_config.style = 1
    config.activation_type = "SiGLU"
    config.has_pre_decoder_layernorm = False
    config.has_post_decoder_layernorm = True
    config.norm_type = "rmsnorm"

    config_json = require_config_json(ckpt_path)
    _apply_glm4_moe_config_json(config, config_json)

    assert (
        config.attn_config.head_num > 0
        and config.attn_config.kv_head_num > 0
        and config.attn_config.size_per_head > 0
        and config.num_layers > 0
        and config.inter_size > 0
    ), f"error config config.attn_config.head_num={config.attn_config.head_num} config.attn_config.kv_head_num={config.attn_config.kv_head_num} config.attn_config.size_per_head={config.attn_config.size_per_head} config.num_layers={config.num_layers} config.inter_size={config.inter_size}"

    return config
