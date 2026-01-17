"""Configuration creators for Bloom models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _apply_bloom_huggingface_config(config: ModelConfig, config_json: Dict[str, Any]):
    """Apply Bloom configuration from HuggingFace config.json."""
    model_type = config_json.get("model_type", "")
    if model_type != "bloom":
        raise BaseException(f"model type is not bloom: {model_type}")

    config.attn_config.head_num = config_json.get(
        "num_attention_heads", config_json.get("n_head")
    )
    config.attn_config.kv_head_num = config.attn_config.head_num
    config.hidden_size = config_json.get("n_embed", config_json.get("hidden_size"))
    config.attn_config.size_per_head = config.hidden_size // config.attn_config.head_num
    config.num_layers = config_json["n_layer"]
    config.max_seq_len = config_json.get("seq_length", 2048)
    config.vocab_size = config_json["vocab_size"]
    config.layernorm_eps = config_json["layer_norm_epsilon"]
    config.inter_size = config.hidden_size * 4
    config.special_tokens.eos_token_id = config_json.get("eos_token_id", 0)
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    config.config_dtype = config_json.get("torch_dtype", None)


@register_config_creator("bloom")
def create_bloom_config(ckpt_path: str) -> ModelConfig:
    """Create Bloom model configuration."""
    config_json = get_config_json(ckpt_path)

    if config_json:
        config = ModelConfig()
        _apply_bloom_huggingface_config(config, config_json)
    else:
        # Default config
        config = ModelConfig()
        config.attn_config.head_num = 32
        config.attn_config.kv_head_num = 32
        config.attn_config.size_per_head = 128
        config.inter_size = 4 * 32 * 128
        config.num_layers = 30
        config.max_seq_len = 2048
        config.vocab_size = 250880

    config.ckpt_path = ckpt_path
    config.layernorm_eps = 1e-5
    config.layernorm_type = "pre_layernorm"
    config.activation_type = "gelu"
    config.has_positional_encoding = False
    config.has_pre_decoder_layernorm = True
    config.has_post_decoder_layernorm = True
    config.use_attention_linear_bias = True

    return config
