"""Configuration creators for Flot VL models."""

import json
import logging
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.flot import (
    _apply_flot_config_json,
    create_flot_005_config,
)
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _rewrite_rotary(config: ModelConfig, ckpt_path: str):
    """Rewrite rotary embedding config for Tbstars models."""
    config_json = get_config_json(ckpt_path)
    if not config_json:
        return

    config.attn_config.rope_config.style = 1
    config.attn_config.rope_config.base = config_json.get("rope_theta", 10000)
    if config_json.get("rope_scaling", None):
        config.attn_config.rope_config.scale = config_json["rope_scaling"]["factor"]
        config.attn_config.rope_config.max_pos = config_json.get(
            "max_position_embeddings", 4096
        )
        rope_type = config_json["rope_scaling"]["type"]
        if rope_type == "dynamic":
            config.attn_config.rope_config.style = 3
    else:
        # different from flot
        config.attn_config.rope_config.scale = 1


@register_config_creator("turing_005_vl")
def create_flot_005_vl_config(ckpt_path: str) -> ModelConfig:
    """Create Flot_005_VL model configuration."""
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        raise Exception("flot_005_vl parameter from unknown source")

    with open(config_path) as reader:
        content = reader.read()
        config_json = json.loads(content)

    if "text_config" in config_json:
        text_config = config_json["text_config"]
        config = ModelConfig(
            head_num=40,
            head_num_kv=4,
            size_per_head=128,
            inter_size=16896,
            layer_num=40,
            max_seq_len=8192,
            vocab_size=64000,
            has_positional_encoding=False,
            has_post_decoder_layernorm=True,
            has_pre_decoder_layernorm=False,
            activation_type="gated-silu",
            rotary_embedding_dim=128,
            rotary_embedding_style=1,
            norm_type="rmsnorm",
            layernorm_eps=1e-5,
            ckpt_path=ckpt_path,
        )
        config.special_tokens.eos_token_id = 2
        _apply_flot_config_json(config, text_config, ckpt_path)

        # Multimodal config
        config.mm_related_params.config["mm_use_im_patch_token"] = False
        config.mm_related_params.config["mm_use_im_start_end"] = False
        config.mm_related_params.config["tune_mm_mlp_adapter"] = False
        config.mm_related_params.config["image_aspect_ratio"] = config_json[
            "projector_config"
        ]["crop_image_process"]
        config.mm_related_params.config["mm_projector_type"] = config_json[
            "projector_config"
        ]["projector_type"]
        config.mm_related_params.config["hidden_size"] = config_json["text_config"][
            "hidden_size"
        ]
        config.mm_related_params.config["mm_vision_select_layer"] = config_json.get(
            "vision_select_layer", -2
        )
        config.mm_related_params.config["mm_vision_select_feature"] = config_json.get(
            "vision_select_feature", "patch"
        )
        config.mm_related_params.config["perceiver_num_queries"] = config_json[
            "projector_config"
        ]["num_queries"]
        config.mm_related_params.config["adaptive_crop_num"] = config_json[
            "projector_config"
        ].get("adaptive_crop_num", 0)
        config.mm_related_params.config["perceiver_num_heads"] = config_json[
            "projector_config"
        ]["perceiver_num_heads"]
        config.mm_related_params.config["vision_config"] = config_json["vision_config"]
        config.mm_related_params.config["image_size"] = config_json["vision_config"][
            "vision_cfg"
        ]["image_size"]
    else:
        config = create_flot_005_config(ckpt_path)
    config.mm_related_params.config["mm_hidden_size"] = config_json["mm_hidden_size"]
    config.mm_related_params.special_token_ids.update(
        {"ignore_token_index": -100, "image_token_index": -200}
    )
    config.mm_related_params.special_tokens.update(
        {
            "default_mm_token": "<image>",
            "default_im_start_token": "<im_start>",
            "default_im_end_token": "<im_end>",
        }
    )
    config.mm_model_config.mm_sep_tokens = [[-200]]  # image_token_index

    return config


@register_config_creator("tbstars_vl_002")
def create_tbstars_vl_002_config(ckpt_path: str) -> ModelConfig:
    """Create Tbstars_VL_002 model configuration."""
    config = create_flot_005_vl_config(ckpt_path)
    _rewrite_rotary(config, ckpt_path)
    return config


@register_config_creator("tbstars_vl_004")
def create_tbstars_vl_004_config(ckpt_path: str) -> ModelConfig:
    """Create Tbstars_VL_004 model configuration."""
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        raise Exception("tbstars_vl 004 parameter from unknown source")

    with open(config_path) as reader:
        content = reader.read()
        config_json = json.loads(content)

    config = create_flot_005_config(ckpt_path)
    config.mm_related_params.config["image_aspect_ratio"] = config_json[
        "image_aspect_ratio"
    ]
    config.mm_related_params.config["vision_model_type"] = config_json[
        "vision_model_type"
    ]
    config.mm_related_params.config["vision_select_feature"] = config_json[
        "vision_select_feature"
    ]
    config.mm_related_params.config["vision_select_layer"] = config_json[
        "vision_select_layer"
    ]
    config.mm_related_params.config["vision_config"] = config_json["vision_config"]
    config.mm_related_params.config["projector_config"] = config_json[
        "projector_config"
    ]

    # Load preprocessor_config.json
    processor_config = os.path.join(ckpt_path, "preprocessor_config.json")
    if os.path.exists(processor_config):
        with open(processor_config) as reader:
            content = reader.read()
            config_json = json.loads(content)
            config.mm_related_params.config["preprocessor_config"] = config_json
    else:
        raise Exception(
            "tbstars_vl_004 must have preprocessor_config.json in ckpt path"
        )

    _rewrite_rotary(config, ckpt_path)
    # Note: mm_sep_tokens would be set from tokenizer, not in config creator
    return config


@register_config_creator("tbstars_vl_008o")
def create_tbstars_vl_008o_config(ckpt_path: str) -> ModelConfig:
    """Create Tbstars_VL_008o model configuration."""
    config = create_tbstars_vl_004_config(ckpt_path)
    config_json = get_config_json(ckpt_path)
    if config_json:
        config.mm_related_params.config["vision_projector_config"] = config_json.get(
            "vision_projector_config"
        )
    return config


@register_config_creator("biencoder_vl_tbstars")
def create_biencoder_vl_tbstars_config(ckpt_path: str) -> ModelConfig:
    """Create BiEncoderVLTbstars model configuration."""
    config = create_tbstars_vl_008o_config(ckpt_path)
    config_json = get_config_json(ckpt_path)
    if config_json and config_json.get("bidirectional_attention", False):
        config.attn_config.is_causal = False
    return config
