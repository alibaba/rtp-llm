"""Configuration creators for SGPT Bloom models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.bloom import create_bloom_config
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


@register_config_creator("sgpt_bloom")
@register_config_creator("sgpt_bloom_vector")
def create_sgpt_bloom_config(ckpt_path: str) -> ModelConfig:
    """Create SGPT Bloom model configuration.

    SGPTBloom and SGPTBloomVector inherit from Bloom and use the same
    configuration as Bloom models.
    """
    return create_bloom_config(ckpt_path)
