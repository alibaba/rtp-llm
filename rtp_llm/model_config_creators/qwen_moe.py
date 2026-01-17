"""Configuration creators for Qwen MoE models."""

import json
import logging
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.qwen import create_qwen_v2_config
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _load_qwen2_moe_config(ckpt_path: str, config: ModelConfig):
    """Load MoE configuration for Qwen2 MoE."""
    config_json = require_config_json(ckpt_path)

    config.moe_k = config_json["num_experts_per_tok"]
    config.expert_num = config_json["num_experts"]
    # Set inter_size and moe_inter_size for hybrid MoE
    config.moe_inter_size = config_json["moe_intermediate_size"]
    config.inter_size = config_json.get("shared_expert_intermediate_size", 0)
    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
    config.has_moe_norm = config_json.get("norm_topk_prob", False)
    # step for moe layer
    config.moe_style = 2
    moe_step = config_json["decoder_sparse_step"]

    # todo
    # qwen2 moe is supposed to have different inter size for moe and normal layers
    # so there should be two config for ffnlayer
    if moe_step != 1:
        raise Exception("Partial moe weights for qwen2 is not implemented yet!")
    config.moe_layer_index = [
        i for i in range(moe_step - 1, config.num_layers, moe_step)
    ]


@register_config_creator("qwen2_moe")
@register_config_creator(
    "qwen_2_moe"
)  # Also register as "qwen_2_moe" to match model registration
def create_qwen2_moe_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen2 MoE model configuration."""
    config = create_qwen_v2_config(ckpt_path)
    _load_qwen2_moe_config(ckpt_path, config)
    return config


@register_config_creator("qwen_3_moe")
@register_config_creator("qwen_3_moe_eagle3")
@register_config_creator("qwen3_coder_moe")
def create_qwen3_moe_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen3 MoE model configuration."""
    config = create_qwen_v2_config(ckpt_path)
    config.qk_norm = True
    config.moe_style = 1
    # Note: Qwen3Moe.load_moe_config would be called separately if needed
    return config


@register_config_creator("qwen3_vl_moe")
def create_qwen3_vl_moe_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen3 VL MoE model configuration."""
    from rtp_llm.model_config_creators.qwen2_vl import create_qwen2_vl_config

    config = create_qwen2_vl_config(ckpt_path)
    _load_qwen2_moe_config(ckpt_path, config)
    config.use_qk_norm = True
    return config
