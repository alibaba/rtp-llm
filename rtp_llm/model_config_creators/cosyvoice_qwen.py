"""Configuration creators for CosyVoice Qwen models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.qwen import create_qwen_v2_config
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("cosyvoice_qwen")
def create_cosyvoice_qwen_config(ckpt_path: str) -> ModelConfig:
    """Create CosyVoice Qwen model configuration."""
    config = create_qwen_v2_config(ckpt_path)

    # Apply CosyVoice-specific config
    config_json = get_config_json(ckpt_path)
    if config_json:
        config.input_vocab_size = config_json.get(
            "input_vocab_size", config.vocab_size + 151938
        )

    config.mm_model_config.mm_sep_tokens = [[-200]]  # TODO: for SFT support
    return config
