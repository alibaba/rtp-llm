"""Configuration creators for Phi models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("phi")
def create_phi_config(ckpt_path: str) -> ModelConfig:
    """Create Phi model configuration."""
    config_json = get_config_json(ckpt_path)
    if config_json is None:
        config_json = {}

    size_per_head = int(config_json.get("n_embd", 2048) / config_json.get("n_head", 32))

    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = config_json.get("n_head", 32)
    config.attn_config.size_per_head = size_per_head
    config.inter_size = 4 * config_json.get("n_embd", 2048)
    config.num_layers = config_json.get("n_layer", 24)
    config.max_seq_len = config_json.get("n_positions", 2048)
    config.vocab_size = config_json.get("vocab_size", 32)
    config.attn_config.rope_config.dim = config_json.get("rotary_dim", size_per_head)
    config.attn_config.rope_config.style = 1
    config.attn_config.kv_head_num = config_json.get("n_head", 32)
    config.norm_type = "layernorm"
    config.activation_type = "gelu"
    config.has_positional_encoding = False
    config.has_post_decoder_layernorm = True
    config.has_lm_head_bias = True
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    config.config_dtype = config_json.get("torch_dtype", None)

    return config
