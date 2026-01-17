"""Configuration creators for Falcon models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("falcon")
def create_falcon_config(ckpt_path: str) -> ModelConfig:
    """Create Falcon model configuration."""
    config_json = require_config_json(ckpt_path)
    head_num = config_json.get("n_head", config_json.get("num_attention_heads"))

    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = head_num
    config.attn_config.kv_head_num = config_json.get(
        "n_head_kv", config_json.get("num_kv_heads", 1)
    )
    config.attn_config.size_per_head = config_json["hidden_size"] // head_num
    config.inter_size = config_json["hidden_size"] * 4
    config.num_layers = config_json.get("n_layer", config_json.get("num_hidden_layers"))
    config.max_seq_len = 2048
    config.vocab_size = config_json["vocab_size"]
    config.activation_type = "gelu-none-approximate"
    config.has_post_decoder_layernorm = True
    config.attn_config.rope_config.style = 1
    config.special_tokens.bos_token_id = config_json.get("bos_token_id", -1)
    config.special_tokens.eos_token_id = config_json.get("eos_token_id", 0)
    config.config_dtype = config_json.get("torch_dtype", None)

    return config
