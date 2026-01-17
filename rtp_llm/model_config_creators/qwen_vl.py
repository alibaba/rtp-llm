"""Configuration creators for Qwen VL models."""

import json
import logging
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.qwen import create_qwen_config
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _load_qwen_vl_vit_param(config: ModelConfig, ckpt_path: str):
    """Load vision transformer parameters for Qwen VL."""
    config_json = get_config_json(ckpt_path)
    if not config_json:
        return

    vit_config = config_json["visual"]
    if config.mm_related_params.config is None:
        config.mm_related_params.config = {}
    config.mm_related_params.config.update(vit_config)
    if config.mm_related_params.special_token_ids is None:
        config.mm_related_params.special_token_ids = {}
    config.mm_related_params.special_token_ids.update(
        {
            "image_start_id": vit_config["image_start_id"],
            "image_end_id": vit_config["image_start_id"] + 1,
            "image_pad_id": vit_config["image_start_id"] + 2,
        }
    )
    if config.mm_related_params.special_tokens is None:
        config.mm_related_params.special_tokens = {}
    config.mm_related_params.special_tokens.update({"default_mm_token": "<img/>"})
    # Note: eval_param_count and eval_model_size are model-specific methods
    config.mm_model_config.mm_sep_tokens = [
        [vit_config["image_start_id"], vit_config["image_start_id"] + 1]
    ]


def create_qwen_vl_common_config(config: ModelConfig, ckpt_path: str) -> ModelConfig:
    from rtp_llm.model_config_creators.qwen import (
        _apply_qwen_hf_config_from_file,
        create_qwen_base_config,
    )

    config = create_qwen_base_config(config, ckpt_path)
    _apply_qwen_hf_config_from_file(config, ckpt_path)
    _load_qwen_vl_vit_param(config, ckpt_path)
    return config


@register_config_creator("qwen_vl")
def create_qwen_vl_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen VL model configuration."""
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = 0
    config.attn_config.size_per_head = 0
    config.num_layers = 0
    config.max_seq_len = 1024
    config.vocab_size = 0

    config = create_qwen_vl_common_config(config, ckpt_path)

    return config
