"""Configuration creators for Qwen2 VL models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.qwen import create_qwen_v2_config
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def qwen2_vl_eval_vit_param_count(mm_related_params):
    vit_config = mm_related_params.config
    embed_dim = vit_config.get("embed_dim", 1280)
    hidden_size = vit_config.get("hidden_size", 3584)
    vit_size = (
        vit_config.get("temporal_patch_size", 2)
        * vit_config.get("spatial_patch_size", 14) ** 2
        * vit_config.get("in_chans", 3)
        * embed_dim
    )
    patch_merger_size = embed_dim * vit_config.get("spatial_merge_size", 2) ** 2
    vit_size += patch_merger_size**2 + patch_merger_size * hidden_size + embed_dim
    mlp_hidden_dim = embed_dim * vit_config.get("mlp_ratio", 4)
    vit_size += vit_config.get("depth", 32) * (
        embed_dim * 2 + embed_dim * mlp_hidden_dim + embed_dim * embed_dim * 4
    )

    return vit_size


def qwen2_vl_eval_vit_model_size(mm_related_params):
    data_width = 4
    return qwen2_vl_eval_vit_param_count(mm_related_params) * data_width


def qwen2_vl_eval_vit_param_count(mm_related_params):
    vit_config = mm_related_params.config
    embed_dim = vit_config.get("embed_dim", 1280)
    hidden_size = vit_config.get("hidden_size", 3584)
    vit_size = (
        vit_config.get("temporal_patch_size", 2)
        * vit_config.get("spatial_patch_size", 14) ** 2
        * vit_config.get("in_chans", 3)
        * embed_dim
    )
    patch_merger_size = embed_dim * vit_config.get("spatial_merge_size", 2) ** 2
    vit_size += patch_merger_size**2 + patch_merger_size * hidden_size + embed_dim
    mlp_hidden_dim = embed_dim * vit_config.get("mlp_ratio", 4)
    vit_size += vit_config.get("depth", 32) * (
        embed_dim * 2 + embed_dim * mlp_hidden_dim + embed_dim * embed_dim * 4
    )

    return vit_size


def _load_qwen2_vl_vit_param(config: ModelConfig, config_json: Dict[str, Any]):
    """Load vision transformer parameters for Qwen2 VL."""
    config.mm_related_params.config = config_json["vision_config"]
    if config.mm_related_params.special_tokens is None:
        from rtp_llm.config.model_config import SpecialTokens

        config.mm_related_params.special_tokens = SpecialTokens()
    config.mm_related_params.special_tokens.update({"default_mm_token": "<img/>"})
    config.mm_related_params.eval_param_count = qwen2_vl_eval_vit_param_count
    config.mm_related_params.eval_model_size = qwen2_vl_eval_vit_model_size
    config.mm_model_config.mm_sep_tokens = [
        [config_json["vision_start_token_id"], config_json["vision_end_token_id"]]
    ]


def _apply_qwen2_vl_hf_config(config: ModelConfig, config_json: Dict[str, Any]):
    """Apply Qwen2 VL configuration from HuggingFace config.json."""
    config.vocab_size = config_json["vocab_size"]
    config.max_seq_len = 10240
    config.activation_type = "SiGLU"
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.kv_head_num = config_json["num_key_value_heads"]
    config.hidden_size = config_json["hidden_size"]
    config.attn_config.size_per_head = (
        int(config_json.get("head_dim"))
        if "head_dim" in config_json
        else config_json["hidden_size"] // config.attn_config.head_num
    )
    config.num_layers = config_json["num_hidden_layers"]
    config.inter_size = config_json["intermediate_size"]
    config.norm_type = "rmsnorm"
    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
    config.attn_config.rope_config.base = int(config_json.get("rope_theta", 10000))
    config.attn_config.rope_config.dim = config.attn_config.size_per_head
    config.attn_config.rope_config.style = 1
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    config.config_dtype = config_json.get("torch_dtype", None)


@register_config_creator("qwen2_vl")
def create_qwen2_vl_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen2 VL model configuration."""
    config_json = require_config_json(ckpt_path)

    config = ModelConfig()
    config.ckpt_path = ckpt_path

    _apply_qwen2_vl_hf_config(config, config_json)
    _load_qwen2_vl_vit_param(config, config_json)
    config.mm_related_params.config["ckpt_path"] = ckpt_path

    return config


@register_config_creator("qwen2_5_vl")
def create_qwen2_5_vl_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen2.5 VL model configuration."""
    # Qwen2.5 VL uses the same base config as Qwen2 VL
    config = create_qwen2_vl_config(ckpt_path)
    return config
