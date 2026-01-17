"""Configuration creators for ChatGLM Vision models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.chatglm import create_chatglm_v4_config
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("chatglm4v")
def create_chatglm_v4_vision_config(ckpt_path: str) -> ModelConfig:
    """Create ChatGLM V4 Vision model configuration."""
    config = create_chatglm_v4_config(ckpt_path)

    # Apply vision-specific config
    config_json = get_config_json(ckpt_path)
    if config_json:
        vit_config = config_json.get("vision_config", {})
        config.mm_related_params.config.update(vit_config)
        # use initial hidden size for linear_proj and conv layer in eva2clip
        config.mm_related_params.config["use_vision_hidden_size"] = False
        config.mm_related_params.config["boi_token_id"] = config_json.get(
            "boi_token_id", 0
        )
        config.mm_related_params.config["eoi_token_id"] = config_json.get(
            "eoi_token_id", 0
        )
        config.mm_model_config.mm_sep_tokens = [
            [config_json.get("boi_token_id", 0), config_json.get("eoi_token_id", 0)]
        ]
        config.include_sep_tokens = True
        config.mm_position_ids_style = 1

    return config
