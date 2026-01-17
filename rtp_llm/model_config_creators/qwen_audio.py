"""Configuration creators for Qwen Audio models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.qwen import create_qwen_v2_config
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("qwen_v2_audio")
def create_qwen_v2_audio_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen V2 Audio model configuration."""
    config = create_qwen_v2_config(ckpt_path)

    # Apply audio-specific config
    config_json = get_config_json(ckpt_path)
    if config_json:
        sep_token = config_json.get("audio_token_index")
        text_config = config_json.get("text_config", config_json)

        # Override with text_config if available
        if "text_config" in config_json:
            config.inter_size = text_config.get("intermediate_size", 11008)
            config.attn_config.head_num = text_config.get("num_attention_heads", 32)
            config.attn_config.kv_head_num = text_config.get(
                "num_key_value_heads", config.attn_config.head_num
            )
            config.attn_config.size_per_head = (
                text_config.get("hidden_size", 4096) // config.attn_config.head_num
            )
            config.num_layers = text_config.get("num_hidden_layers", 32)
            config.attn_config.rope_config.base = int(
                text_config.get("rope_theta", config.attn_config.rope_config.base)
            )
            config.vocab_size = text_config["vocab_size"]
            config.attn_config.rope_config.dim = config.attn_config.size_per_head
            config.layernorm_eps = text_config.get("rms_norm_eps", 1e-06)
            config.tie_word_embeddings = text_config.get("tie_word_embeddings", False)

        if sep_token is not None:
            config.mm_model_config.mm_sep_tokens = [[sep_token]]

    return config
