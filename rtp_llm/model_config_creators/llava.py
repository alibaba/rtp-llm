"""Configuration creators for LLaVA models."""

import json
import logging
import os
from typing import Any, Dict

from transformers import CLIPVisionConfig

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.llama import create_llama_config
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _apply_llava_huggingface_config(config: ModelConfig, config_json: Dict[str, Any]):
    """Apply LLaVA configuration from HuggingFace config.json."""
    if "text_config" in config_json:
        text_config = config_json["text_config"]
        # Apply Llama config from text_config
        from rtp_llm.model_config_creators.llama import _apply_llama_huggingface_config

        _apply_llama_huggingface_config(config, text_config)

        vision_config = config_json["vision_config"]

        config.mm_related_params.config["vision_config"] = CLIPVisionConfig(
            vision_config
        )
    else:
        # Apply Llama config from root config
        from rtp_llm.model_config_creators.llama import _apply_llama_huggingface_config

        _apply_llama_huggingface_config(config, config_json)

        # Apply multimodal parameters
        mm_related_params_list = [
            ("mm_use_im_patch_token", False),
            ("mm_use_im_start_end", False),
            ("image_aspect_ratio", None),
            ("tune_mm_mlp_adapter", False),
            ("image_grid_pinpoints", []),
            ("mm_projector_type", "linear"),
            ("mm_patch_merge_type", "flat"),
            ("hidden_size", 0),
            ("mm_vision_select_layer", None),
            ("mm_vision_select_feature", "patch"),
            ("unfreeze_mm_vision_tower", False),
            ("mm_tunable_parts", ""),
            ("add_faster_video", False),
            ("mm_newline_position", "grid"),
            ("mm_spatial_pool_mode", "bilinear"),
        ]

        for param_name, default_value in mm_related_params_list:
            config.mm_related_params.config[param_name] = config_json.get(
                param_name, default_value
            )

        config.mm_related_params.config["mm_hidden_size"] = config_json.get(
            "mm_hidden_size", config.hidden_size
        )


@register_config_creator("llava")
def create_llava_config(ckpt_path: str) -> ModelConfig:
    """Create LLaVA model configuration."""
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.activation_type = "SiGLU"
    config.norm_type = "rmsnorm"
    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 1
    config.has_post_decoder_layernorm = True

    config_path = os.path.join(ckpt_path, "config.json")
    with open(config_path) as reader:
        content = reader.read()
        content = content.replace("LlavaForCausalLM", "LLaVAForCausalLM")
        config_json = json.loads(content)

    _apply_llava_huggingface_config(config, config_json)

    return config
