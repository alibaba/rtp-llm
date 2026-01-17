"""Configuration creators for MiniCPMV models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.qwen import _apply_qwen_hf_config
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _init_minicpmv_vit_params(config: ModelConfig, config_json: Dict[str, Any]):
    """Initialize MiniCPMV vision transformer parameters."""
    if config.mm_related_params.config is None:
        config.mm_related_params.config = {}

    # Extract vision config
    vision_config = config_json.get("vision_config", {})
    config.mm_related_params.config.update(vision_config)

    # Additional vision parameters
    if "image_size" in config_json:
        config.mm_related_params.config["image_size"] = config_json["image_size"]
    if "patch_size" in config_json:
        config.mm_related_params.config["patch_size"] = config_json["patch_size"]


@register_config_creator("minicpmv")
def create_minicpmv_config(ckpt_path: str) -> ModelConfig:
    """Create MiniCPMV model configuration."""
    config_json = require_config_json(ckpt_path)

    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = 0
    config.attn_config.kv_head_num = 0
    config.attn_config.size_per_head = 0
    config.num_layers = 0
    config.inter_size = 0
    config.vocab_size = 0
    config.max_seq_len = 8192
    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 1
    config.activation_type = "SiGLU"
    config.has_pre_decoder_layernorm = False
    config.has_post_decoder_layernorm = True
    config.norm_type = "rmsnorm"

    # Apply QwenV2 config
    _apply_qwen_hf_config(config, config_json)
    _init_minicpmv_vit_params(config, config_json)

    return config


@register_config_creator("minicpmv_embedding")
def create_minicpmv_embedding_config(ckpt_path: str) -> ModelConfig:
    """Create MiniCPMV Embedding model configuration."""
    import math

    from rtp_llm.model_config_creators.llama import _apply_llama_huggingface_config

    config_json = require_config_json(ckpt_path)

    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = 0
    config.attn_config.size_per_head = 0
    config.num_layers = 0
    config.max_seq_len = 0
    config.vocab_size = 0
    config.activation_type = "SiGLU"
    config.norm_type = "rmsnorm"
    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 1
    config.has_post_decoder_layernorm = True

    # Apply Llama config
    _apply_llama_huggingface_config(config, config_json)
    config.input_embedding_scalar = config_json.get("scale_emb", 1)
    config.residual_scalar = config_json.get("scale_depth", 1.4) / math.sqrt(
        config.num_layers
    )

    _init_minicpmv_vit_params(config, config_json)

    return config
