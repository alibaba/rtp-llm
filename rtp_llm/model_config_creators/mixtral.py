"""Configuration creators for Mixtral models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("mixtral")
def create_mixtral_config(ckpt_path: str) -> ModelConfig:
    """Create Mixtral model configuration."""
    config_json = require_config_json(ckpt_path)
    size_per_head = config_json["hidden_size"] // config_json["num_attention_heads"]

    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.size_per_head = size_per_head
    config.moe_inter_size = config_json["intermediate_size"]
    config.num_layers = config_json["num_hidden_layers"]
    config.max_seq_len = config_json.get("max_sequence_length", 2048)
    config.vocab_size = config_json["vocab_size"]
    config.attn_config.kv_head_num = config_json["num_key_value_heads"]
    config.attn_config.rope_config.dim = size_per_head
    config.has_moe_norm = True
    config.attn_config.rope_config.style = 1
    config.attn_config.rope_config.base = int(config_json.get("rope_theta", 10000))
    config.expert_num = config_json["num_local_experts"]
    config.moe_k = config_json["num_experts_per_tok"]
    config.moe_style = 1
    config.moe_layer_index = [i for i in range(config_json["num_hidden_layers"])]
    config.special_tokens.eos_token_id = 2
    config.special_tokens.bos_token_id = 1
    config.config_dtype = config_json.get("torch_dtype", None)

    return config
