"""Configuration creators for DeepSeek V2 models."""

import json
import logging
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.registry import register_config_creator
from rtp_llm.utils.model_weight import yarn_get_mscale

logger = logging.getLogger(__name__)


def _apply_deepseek_v2_hf_config(config: ModelConfig, config_json: Dict[str, Any]):
    """Apply DeepSeek V2 configuration from HuggingFace config.json."""
    config.inter_size = config_json["intermediate_size"]
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.kv_head_num = config_json.get(
        "num_key_value_heads", config.attn_config.head_num
    )
    config.num_layers = config_json["num_hidden_layers"]
    config.attn_config.rope_config.base = int(
        config_json.get("rope_theta", config.attn_config.rope_config.base)
    )
    config.vocab_size = config_json["vocab_size"]
    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    config.hidden_size = config_json["hidden_size"]

    # MLA config
    config.attn_config.use_mla = True
    q_lora_rank = config_json.get("q_lora_rank")
    config.attn_config.q_lora_rank = int(q_lora_rank) if q_lora_rank is not None else 0
    kv_lora_rank = config_json.get("kv_lora_rank")
    config.attn_config.kv_lora_rank = (
        int(kv_lora_rank) if kv_lora_rank is not None else 0
    )
    config.attn_config.nope_head_dim = config_json["qk_nope_head_dim"]
    config.attn_config.rope_head_dim = config_json["qk_rope_head_dim"]
    config.attn_config.v_head_dim = config_json["v_head_dim"]
    config.attn_config.size_per_head = (
        config.attn_config.nope_head_dim + config.attn_config.rope_head_dim
    )
    config.attn_config.rope_config.dim = config.attn_config.rope_head_dim
    from rtp_llm.ops import MlaOpsType

    # yarn rotary config
    if config.mla_ops_type != MlaOpsType.MHA:
        config.attn_config.rope_config.style = 0
    else:
        config.attn_config.rope_config.style = 5
    rope_scaling = config_json.get("rope_scaling")
    config.attn_config.rope_config.scale = rope_scaling["factor"]
    config.attn_config.rope_config.factor1 = float(rope_scaling.get("beta_slow", 1))
    config.attn_config.rope_config.factor2 = float(rope_scaling.get("beta_fast", 32))
    config.attn_config.rope_config.max_pos = rope_scaling[
        "original_max_position_embeddings"
    ]

    scaling_factor = rope_scaling["factor"]
    mscale = rope_scaling["mscale"]
    mscale_all_dim = rope_scaling["mscale_all_dim"]
    config.deepseek_rope_mscale = mscale
    config.deepseek_mscale_all_dim = mscale_all_dim
    config.attn_config.rope_config.mscale = yarn_get_mscale(
        scaling_factor, mscale
    ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
    config.attn_config.rope_config.offset = config.attn_config.nope_head_dim

    # softmax scale config
    softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    config.attn_config.softmax_extra_scale = softmax_mscale * softmax_mscale

    # MOE config
    if "scoring_func" in config_json:
        scoring_func = config_json["scoring_func"]
        if scoring_func == "softmax":
            config.scoring_func = 0
        elif scoring_func == "sigmoid":
            config.scoring_func = 1
        else:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")

    config.routed_scaling_factor = config_json["routed_scaling_factor"]
    config.moe_k = config_json["num_experts_per_tok"]
    config.expert_num = config_json["n_routed_experts"]
    moe_intermediate_size = config_json["moe_intermediate_size"]
    config.moe_n_group = config_json.get("n_group", 1)
    config.moe_topk_group = config_json.get("topk_group", 1)

    n_shared_experts = config_json["n_shared_experts"]
    config.inter_size = n_shared_experts * moe_intermediate_size

    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
    config.has_moe_norm = config_json.get("norm_topk_prob", False)
    config.moe_style = 2  # shared + expert

    moe_step = config_json["moe_layer_freq"]
    first_k_dense_replace = config_json["first_k_dense_replace"]
    config.moe_layer_index = [
        i
        for i in range(config.num_layers)
        if i >= first_k_dense_replace and i % moe_step == 0
    ]
    config.config_dtype = config_json.get("torch_dtype", None)


@register_config_creator("deepseek2")
@register_config_creator("deepseek3")
@register_config_creator("kimi_k2")
@register_config_creator("deepseek_v31")
def create_deepseek_v2_config(ckpt_path: str) -> ModelConfig:
    """Create DeepSeek V2 model configuration."""
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.head_num = 0
    config.attn_config.kv_head_num = 0
    config.attn_config.size_per_head = 0
    config.num_layers = 0
    config.inter_size = 0
    config.vocab_size = 102400
    config.max_seq_len = 8192
    config.norm_type = "rmsnorm"
    config.has_post_decoder_layernorm = True
    config.activation_type = "SiGLU"

    config_json = get_config_json(ckpt_path)
    if config_json:
        _apply_deepseek_v2_hf_config(config, config_json)

    return config


@register_config_creator("deepseek-v3-mtp")
def create_deepseek_v3_mtp_config(ckpt_path: str) -> ModelConfig:
    """Create DeepSeek V3 MTP model configuration."""
    config = create_deepseek_v2_config(ckpt_path)
    config.moe_layer_index = [i for i in range(config.num_layers)]
    config.reverse_e_h_norm = True
    config.is_mtp = True
    return config
