"""Configuration creators for DeepSeek V2 models."""

import json
import logging
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _apply_deepseek_v2_hf_config(config: ModelConfig, config_json: Dict[str, Any]):
    """Apply DeepSeek V2 configuration from HuggingFace config.json."""
    config.inter_size = config_json["intermediate_size"]
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.kv_head_num = config_json.get(
        "num_key_value_heads", config.attn_config.head_num
    )
    config.num_layers = config_json["num_hidden_layers"]
    config.attn_config.rope_config.base = int(
        config_json.get("rope_theta", config.attn_config.rope_config.base)
    )
    config.vocab_size = config_json["vocab_size"]
    config.hidden_size = config_json["hidden_size"]
    config.attn_config.size_per_head = config.hidden_size // config.attn_config.head_num
    config.attn_config.rope_config.dim = config_json.get(
        "rope_dim", config.attn_config.size_per_head
    )
    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    config.config_dtype = config_json.get("torch_dtype", None)

    # MLA config
    config.attn_config.use_mla = config_json.get("use_mla", False)
    if config.attn_config.use_mla:
        config.attn_config.q_lora_rank = config_json.get("q_lora_rank", 0)
        config.attn_config.kv_lora_rank = config_json.get("kv_lora_rank", 0)
        config.attn_config.nope_head_dim = config_json.get("nope_head_dim", 0)
        config.attn_config.rope_head_dim = config_json.get("rope_head_dim", 0)
        config.attn_config.v_head_dim = config_json.get("v_head_dim", 0)
        config.qk_norm = config_json.get("qk_norm", False)
        config.attn_config.rope_config.offset = config.attn_config.nope_head_dim

    # MOE config
    if "num_routed_experts" in config_json or "num_local_experts" in config_json:
        config.expert_num = config_json.get(
            "num_routed_experts", config_json.get("num_local_experts", 0)
        )
        config.moe_k = config_json.get("num_experts_per_tok", 8)
        config.moe_inter_size = config_json.get(
            "moe_intermediate_size", config.inter_size
        )
        config.moe_style = config_json.get("moe_style", 1)
        first_k_dense_layers = config_json.get("first_k_dense_layers", 0)
        config.moe_layer_index = [
            i for i in range(first_k_dense_layers, config.num_layers)
        ]
        config.has_moe_norm = config_json.get("has_moe_norm", False)
        config.routed_scaling_factor = config_json.get("routed_scaling_factor", 1.0)

        # Scoring function
        scoring_func = config_json.get("scoring_func", "softmax")
        if scoring_func == "softmax":
            config.scoring_func = 0
        elif scoring_func == "sigmoid":
            config.scoring_func = 1
        else:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")


@register_config_creator("deepseek2")
@register_config_creator("deepseek3")
@register_config_creator("kimi_k2")
@register_config_creator("deepseek_v31")
def create_deepseek_v2_config(ckpt_path: str) -> ModelConfig:
    """Create DeepSeek V2 model configuration."""
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = 0
    config.attn_config.kv_head_num = 0
    config.attn_config.size_per_head = 0
    config.num_layers = 0
    config.inter_size = 0
    config.vocab_size = 102400
    config.max_seq_len = 8192
    config.norm_type = "rmsnorm"
    config.has_post_decoder_layernorm = True
    config.activation_type = "SiGLU"

    config_json = get_config_json(ckpt_path)
    if config_json:
        _apply_deepseek_v2_hf_config(config, config_json)

    return config


@register_config_creator("deepseek-v3-mtp")
def create_deepseek_v3_mtp_config(ckpt_path: str) -> ModelConfig:
    """Create DeepSeek V3 MTP model configuration."""
    config = create_deepseek_v2_config(ckpt_path)
    config.moe_layer_index = [i for i in range(config.num_layers)]
    config.reverse_e_h_norm = True
    config.is_mtp = True
    return config
