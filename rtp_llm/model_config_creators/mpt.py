"""Configuration creators for MPT models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("mpt")
def create_mpt_config(ckpt_path: str) -> ModelConfig:
    """Create MPT model configuration."""
    config_json = require_config_json(ckpt_path)
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = config_json["n_heads"]
    config.attn_config.size_per_head = config_json["d_model"] // config_json["n_heads"]
    config.inter_size = config_json["d_model"] * 4
    config.num_layers = config_json["n_layers"]
    config.max_seq_len = 8192
    config.vocab_size = config_json["vocab_size"]
    config.activation_type = "gelu-none-approximate"
    config.has_post_decoder_layernorm = True
    config.use_attention_linear_bias = True
    config.config_dtype = config_json.get("torch_dtype", None)

    return config
