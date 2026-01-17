"""Configuration creators for VideoLogics models."""

import logging
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.llama import create_llama_config
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("video_logics")
def create_video_logics_config(ckpt_path: str) -> ModelConfig:
    """Create VideoLogics model configuration.

    VideoLogics inherits from Llama, so we start with Llama config
    and add video-specific multimodal configuration.
    """
    config = create_llama_config(ckpt_path)

    # Add video-specific multimodal config
    config_json = get_config_json(ckpt_path)
    if config_json is None:
        raise ValueError("config.ckpt_path has no config json")

    config.mm_related_params.config["vision_process_path"] = os.path.join(
        config.ckpt_path, config_json["mm_vision_tower"]
    )
    config.mm_related_params.config["mm_hidden_size"] = config_json["mm_hidden_size"]
    config.mm_related_params.config["hidden_size"] = config_json["hidden_size"]
    config.mm_related_params.config["mm_projector_type"] = config_json[
        "mm_projector_type"
    ]

    return config
