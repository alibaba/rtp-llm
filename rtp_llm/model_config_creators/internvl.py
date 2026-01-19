"""Configuration creators for InternVL models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.llama import _apply_llama_huggingface_config
from rtp_llm.model_config_creators.qwen import (
    _apply_qwen_hf_config,
    qwen2_from_config_json,
)
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _init_internvl_vit_params(config: ModelConfig, config_json: Dict[str, Any]):
    """Initialize InternVL vision transformer parameters."""
    config.mm_related_params.config = config_json["vision_config"]
    config.mm_related_params.config["select_layer"] = config_json["select_layer"]
    config.mm_related_params.config["llm_hidden_size"] = config_json["llm_config"][
        "hidden_size"
    ]
    config.mm_related_params.config["downsample_ratio"] = config_json[
        "downsample_ratio"
    ]
    config.mm_related_params.config["ps_version"] = config_json["ps_version"]


@register_config_creator("internvl")
def create_internvl_config(ckpt_path: str) -> ModelConfig:
    """Create InternVL model configuration."""
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 1
    config.has_pre_decoder_layernorm = False
    config_json = require_config_json(ckpt_path)
    llm_config = config_json["llm_config"]
    architecture = llm_config["architectures"][0]

    if architecture == "Qwen2ForCausalLM":
        qwen2_from_config_json(config, llm_config)
    elif architecture in ["InternLM2ForCausalLM", "LlamaForCausalLM"]:
        # Apply Llama config
        _apply_llama_huggingface_config(config, llm_config)
    else:
        raise Exception(f"unknown language model architecture: {architecture}")

    _init_internvl_vit_params(config, config_json)
    config.special_tokens.stop_words_str_list = ["<|im_end|>"]
    config.mm_related_params.special_tokens.update({"default_mm_token": "<image>"})

    assert (
        config.attn_config.head_num > 0
        and config.attn_config.kv_head_num > 0
        and config.attn_config.size_per_head > 0
        and config.num_layers > 0
        and config.inter_size > 0
    ), "error config"

    return config
