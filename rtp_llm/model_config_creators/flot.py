"""Configuration creators for Flot models."""

import json
import logging
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _apply_flot_config_json(
    config: ModelConfig, config_json: Dict[str, Any], ckpt_path: str
):
    """Apply Flot configuration from config.json."""
    config.attn_config.head_num = config_json.get(
        "num_attention_heads", config.attn_config.head_num
    )
    kv_groups = config_json.get("kv_groups", config_json.get("num_key_value_heads", 0))
    config.attn_config.kv_head_num = (
        config.attn_config.head_num if kv_groups == 0 else kv_groups
    )
    if config_json.get("kv_channels", 0) != 0:
        config.attn_config.size_per_head = config_json.get("kv_channels")

    ffn_multiple_of = config_json.get("ffn_multiple_of", 0)
    if (ffn_multiple_of is not None) and ffn_multiple_of != 0:
        hidden_size = config.attn_config.head_num * config.attn_config.size_per_head
        ff_mult = 4 * 2 / 3
        ff_dim = int(ff_mult * hidden_size)
        ff_dim = ffn_multiple_of * ((ff_dim + ffn_multiple_of - 1) // ffn_multiple_of)
        config.inter_size = ff_dim
    else:
        config.inter_size = config_json.get("intermediate_size", 0)

    config.layernorm_eps = config_json.get("layer_norm_epsilon", config.layernorm_eps)
    config.num_layers = config_json.get("num_hidden_layers", config.num_layers)
    config.vocab_size = config_json.get("vocab_size", config.vocab_size)
    bos_token_id = config_json.get("bos_token_id", -1)
    eos_token_id = config_json.get("eos_token_id", 0)
    if isinstance(eos_token_id, list):
        config.special_tokens.stop_words_id_list = [[id] for id in eos_token_id]
    elif eos_token_id is not None:
        config.special_tokens.eos_token_id = eos_token_id

    if bos_token_id is not None:
        config.special_tokens.bos_token_id = bos_token_id

    config.attn_config.rope_config.style = 1
    config.attn_config.rope_config.scale = (
        config_json.get("max_position_embeddings", 4096) / 4096
    )
    config.attn_config.rope_config.base = config_json.get("rope_theta", 10000)
    config.hidden_size = config_json.get("hidden_size", config.hidden_size)

    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    if config_json.get("bidirectional_attention", False):
        config.attn_config.is_causal = False


@register_config_creator("flot")
def create_flot_config(ckpt_path: str) -> ModelConfig:
    """Create Flot model configuration."""
    config = ModelConfig()
    config.attn_config.head_num = 40
    config.attn_config.kv_head_num = 40
    config.attn_config.size_per_head = 128
    config.inter_size = 13824
    config.num_layers = 40
    config.max_seq_len = 8192
    config.vocab_size = 64000
    config.has_positional_encoding = False
    config.has_post_decoder_layernorm = True
    config.has_pre_decoder_layernorm = False
    config.activation_type = "gated-silu"
    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 1
    config.norm_type = "rmsnorm"
    config.layernorm_eps = 1e-5
    config.ckpt_path = ckpt_path
    config.special_tokens.eos_token_id = 2

    config_json = get_config_json(ckpt_path)
    if config_json:
        _apply_flot_config_json(config, config_json, ckpt_path)

    return config


@register_config_creator("flot_005")
def create_flot_005_config(ckpt_path: str) -> ModelConfig:
    """Create Flot_005 model configuration."""
    config = ModelConfig()
    config.attn_config.head_num = 40
    config.attn_config.kv_head_num = 4
    config.attn_config.size_per_head = 128
    config.inter_size = 16896
    config.num_layers = 40
    config.max_seq_len = 8192
    config.vocab_size = 64000
    config.has_positional_encoding = False
    config.has_post_decoder_layernorm = True
    config.has_pre_decoder_layernorm = False
    config.activation_type = "gated-silu"
    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 1
    config.norm_type = "rmsnorm"
    config.layernorm_eps = 1e-5
    config.ckpt_path = ckpt_path
    config.special_tokens.eos_token_id = 2

    config_json = get_config_json(ckpt_path)
    if config_json:
        _apply_flot_config_json(config, config_json, ckpt_path)

    return config
