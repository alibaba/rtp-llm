"""Configuration creators for DeepSeek VL V2 models."""

import json
import logging
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _apply_deepseek_vl_v2_hf_config(
    config: ModelConfig, top_config_json: Dict[str, Any]
):
    """Apply DeepSeek VL V2 configuration from HuggingFace config.json."""
    config.model_name = top_config_json.get("model_type", "deepseek_vl_v2")

    config_json = top_config_json["language_config"]
    config.hidden_size = config_json["hidden_size"]
    config.inter_size = config_json["intermediate_size"]
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.kv_head_num = config_json.get(
        "num_key_value_heads", config.attn_config.head_num
    )
    config.num_layers = config_json["num_hidden_layers"]
    config.vocab_size = config_json["vocab_size"]
    config.layernorm_eps = config_json.get(
        "rms_norm_eps", config_json.get("layer_norm_eps", 1e-05)
    )
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

    config.attn_config.size_per_head = config.hidden_size // config.attn_config.head_num
    config.attn_config.rope_config.base = int(config_json.get("rope_theta", 10000))
    # Note: rope_config.dim is set but not assigned a value in original code

    # MOE config
    if "scoring_func" in config_json:
        scoring_func = config_json["scoring_func"]
        if scoring_func == "softmax":
            config.scoring_func = 0
        elif scoring_func == "sigmoid":
            config.scoring_func = 1
        else:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")

    config.routed_scaling_factor = config_json.get("routed_scaling_factor", 1.0)
    config.moe_k = config_json["num_experts_per_tok"]
    config.expert_num = config_json["n_routed_experts"]
    # MOE architecture: shared experts + routed experts
    config.moe_n_group = config_json.get("n_group", 1)
    config.moe_topk_group = config_json.get("topk_group", 1)
    config.moe_inter_size = config_json["moe_intermediate_size"]

    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
    config.has_moe_norm = config_json.get("norm_topk_prob", False)
    config.moe_style = 2  # shared + expert

    moe_step = config_json.get("moe_layer_freq", 1)
    first_k_dense_replace = config_json["first_k_dense_replace"]
    config.moe_layer_index = [
        i
        for i in range(config.num_layers)
        if i >= first_k_dense_replace and i % moe_step == 0
    ]


@register_config_creator("deepseek2-vl")
def create_deepseek_vl_v2_config(ckpt_path: str) -> ModelConfig:
    """Create DeepSeek VL V2 model configuration."""
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = 0
    config.attn_config.kv_head_num = 0
    config.attn_config.size_per_head = 0
    config.num_layers = 0
    config.inter_size = 0
    config.moe_inter_size = 0
    config.vocab_size = 102400
    config.max_seq_len = 8192
    config.norm_type = "rmsnorm"
    config.has_post_decoder_layernorm = True
    config.activation_type = "gated-silu"

    config_json = get_config_json(ckpt_path)
    if config_json:
        _apply_deepseek_vl_v2_hf_config(config, config_json)

    return config
