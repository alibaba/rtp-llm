"""Configuration creators for StarCoder series models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("gpt_bigcode")
@register_config_creator("wizardcoder")
def create_starcoder_config(ckpt_path: str) -> ModelConfig:
    """Create StarCoder model configuration."""
    config_json = get_config_json(ckpt_path)

    if config_json:
        model_type = config_json.get("model_type", "")
        if model_type != "gpt_bigcode":
            raise BaseException(f"model type is not starcoder: {model_type}")

        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.head_num = config_json["n_head"]
        config.attn_config.size_per_head = (
            config_json["n_embd"] // config_json["n_head"]
        )
        config.num_layers = config_json["n_layer"]
        config.max_seq_len = config_json.get("n_positions", 8192)
        config.vocab_size = config_json["vocab_size"]
        config.attn_config.kv_head_num = 1
        config.layernorm_eps = config_json["layer_norm_epsilon"]
        config.inter_size = config_json["n_inner"]
        config.special_tokens.eos_token_id = config_json.get("eos_token_id", 0)
        config.special_tokens.bos_token_id = config_json.get("bos_token_id", -1)
        config.has_positional_encoding = True
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.config_dtype = config_json.get("torch_dtype", None)
    else:
        # Default config
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.head_num = 48
        config.attn_config.kv_head_num = 1
        config.attn_config.size_per_head = 128
        config.inter_size = 4 * 6144
        config.num_layers = 40
        config.max_seq_len = 8192
        config.vocab_size = 49152
        config.has_positional_encoding = True
        config.special_tokens.bos_token_id = 0
        config.special_tokens.eos_token_id = 0

    return config


@register_config_creator("starcoder2")
def create_starcoder2_config(ckpt_path: str) -> ModelConfig:
    """Create StarCoder2 model configuration."""
    config_json = get_config_json(ckpt_path)

    if config_json:
        model_type = config_json.get("model_type", "")
        if model_type != "starcoder2":
            raise BaseException(f"model type is not starcoder2: {model_type}")

        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json["num_key_value_heads"]
        config.attn_config.size_per_head = (
            config_json["hidden_size"] // config_json["num_attention_heads"]
        )
        config.num_layers = config_json["num_hidden_layers"]
        config.max_seq_len = config_json.get("max_position_embeddings", 8192)
        config.vocab_size = config_json["vocab_size"]
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.layernorm_eps = config_json["layer_norm_epsilon"]
        config.inter_size = config_json["intermediate_size"]
        config.special_tokens.eos_token_id = config_json.get("eos_token_id", 0)
        config.special_tokens.bos_token_id = config_json.get("bos_token_id", -1)
        config.activation_type = config_json["activation_function"]
        config.attn_config.rope_config.base = int(
            config_json.get("rope_theta", 1000000)
        )
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.config_dtype = config_json.get("torch_dtype", None)
    else:
        # Default config
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.head_num = 36
        config.attn_config.kv_head_num = 4
        config.attn_config.size_per_head = 128
        config.inter_size = 4 * 4608
        config.num_layers = 32
        config.max_seq_len = 16384
        config.vocab_size = 49152
        config.special_tokens.bos_token_id = 0
        config.special_tokens.eos_token_id = 0
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1

    return config
