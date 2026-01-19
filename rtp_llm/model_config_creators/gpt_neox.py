"""Configuration creators for GPT-NeoX models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _apply_gpt_neox_huggingface_config(
    config: ModelConfig, config_json: Dict[str, Any]
):
    """Apply GPT-NeoX configuration from HuggingFace config.json."""
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.kv_head_num = config.head_num
    config.attn_config.size_per_head = (
        config_json["hidden_size"] // config_json["num_attention_heads"]
    )
    config.num_layers = config_json["num_hidden_layers"]
    config.vocab_size = config_json["vocab_size"]
    config.layernorm_eps = config_json["layer_norm_eps"]
    config.inter_size = config_json["intermediate_size"]
    config.special_tokens.bos_token_id = config_json.get("bos_token_id", -1)
    config.special_tokens.eos_token_id = config_json.get("eos_token_id", 0)
    config.attn_config.rope_config.dim = int(
        config.attn_config.size_per_head * config_json.get("rotary_pct", 1.0)
    )
    config.attn_config.rope_config.style = 1
    if config_json.get("rope_scaling", None):
        config.attn_config.rope_config.style = 3
        config.attn_config.rope_config.scale = config_json["rope_scaling"]["factor"]
        config.org_embedding_max_pos = config_json.get("max_position_embeddings", 2048)
    config.has_pre_decoder_layernorm = False
    config.has_post_decoder_layernorm = True
    config.norm_type = "layernorm"
    config.use_norm_input_residual = True
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    config.config_dtype = config_json.get("torch_dtype", None)


@register_config_creator("gpt_neox")
def create_gpt_neox_config(ckpt_path: str) -> ModelConfig:
    """Create GPT-NeoX model configuration."""
    config_json = get_config_json(ckpt_path)

    if config_json:
        config = ModelConfig()
        config.head_num = 40
        config.size_per_head = 128
        config.layer_num = 40
        config.max_seq_len = 4096
        config.vocab_size = 250752
        _apply_gpt_neox_huggingface_config(config, config_json)
        config.ckpt_path = ckpt_path
    else:
        # Default config
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.head_num = 40
        config.attn_config.kv_head_num = 40
        config.attn_config.size_per_head = 128
        config.num_layers = 40
        config.max_seq_len = 4096
        config.vocab_size = 250752
        config.inter_size = 20480
        config.special_tokens.eos_token_id = 2
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "layernorm"
        config.use_norm_input_residual = True

    return config


@register_config_creator("gpt_neox_13b")
def create_gpt_neox_13b_config(ckpt_path: str) -> ModelConfig:
    """Create GPT-NeoX 13B model configuration."""
    config_json = get_config_json(ckpt_path)

    if config_json:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        _apply_gpt_neox_huggingface_config(config, config_json)
    else:
        # Default config
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.head_num = 40
        config.attn_config.kv_head_num = 40
        config.attn_config.size_per_head = 128
        config.num_layers = 40
        config.max_seq_len = 4096
        config.vocab_size = 250752
        config.inter_size = 20480

    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 1
    config.has_pre_decoder_layernorm = False
    config.has_post_decoder_layernorm = True
    config.norm_type = "rmsnorm"
    config.special_tokens.eos_token_id = 2

    return config
