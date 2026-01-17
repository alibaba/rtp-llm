"""Configuration creators for Llama and related models."""

import json
import logging
import math
import os
from typing import Any, Dict, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    """Compute intermediate size for Llama models."""
    return multiple_of * (
        (int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of
    )


def get_mscale(scale: float):
    """Get mscale for Yarn rope scaling."""
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def _apply_llama_huggingface_config(config: ModelConfig, config_json: Dict[str, Any]):
    """Apply Llama configuration from HuggingFace config.json."""
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.kv_head_num = config_json.get(
        "num_key_value_heads", config.attn_config.head_num
    )
    config.hidden_size = config_json["hidden_size"]
    config.attn_config.size_per_head = (
        config_json["hidden_size"] // config_json["num_attention_heads"]
    )
    config.attn_config.size_per_head = config_json.get(
        "head_dim", config.attn_config.size_per_head
    )
    config.num_layers = config_json["num_hidden_layers"]
    config.max_seq_len = config_json.get("max_sequence_length", 2048)
    config.vocab_size = config_json["vocab_size"]
    config.layernorm_eps = config_json.get(
        "rms_norm_eps", config_json.get("layer_norm_eps", 1e-05)
    )
    config.inter_size = config_json["intermediate_size"]
    config.attn_config.rope_config.base = int(config_json.get("rope_theta", 10000))
    config.attn_config.rope_config.dim = config.attn_config.size_per_head
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

    rope_scaling = config_json.get("rope_scaling")
    if rope_scaling is not None:
        rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        if rope_type == "linear":
            config.attn_config.rope_config.scale = rope_scaling["factor"]
            config.attn_config.rope_config.max_pos = config_json.get(
                "max_position_embeddings", 2048
            )
        elif rope_type == "dynamic":
            config.attn_config.rope_config.style = 3
        elif rope_type == "yarn":
            config.attn_config.rope_config.style = 5
            config.attn_config.rope_config.scale = rope_scaling["factor"]
            config.attn_config.rope_config.factor1 = rope_scaling.get("beta_slow", 1)
            config.attn_config.rope_config.factor2 = rope_scaling.get("beta_fast", 32)
            config.attn_config.rope_config.max_pos = rope_scaling[
                "original_max_position_embeddings"
            ]
            config.attn_config.rope_config.mscale = get_mscale(
                config.attn_config.rope_config.scale
            )
        elif rope_type == "llama3":
            config.attn_config.rope_config.style = 6
            config.attn_config.rope_config.scale = rope_scaling["factor"]
            config.attn_config.rope_config.factor1 = rope_scaling["low_freq_factor"]
            config.attn_config.rope_config.factor2 = rope_scaling["high_freq_factor"]
            config.attn_config.rope_config.max_pos = rope_scaling[
                "original_max_position_embeddings"
            ]
        else:
            raise Exception(f"unsupport rope_scaling {rope_scaling}")

    eos_token_id = config_json.get("eos_token_id", 0)
    if isinstance(eos_token_id, list):
        config.special_tokens.eos_token_id = eos_token_id[0]
    else:
        config.special_tokens.eos_token_id = eos_token_id
    config.attn_config.use_logn_attn = config_json.get("use_logn_attn", False)
    config.config_dtype = config_json.get("torch_dtype", None)


def _apply_llama_params_config(config: ModelConfig, params_json: Dict[str, Any]):
    """Apply Llama configuration from params.json (llama-int8 format)."""
    config.attn_config.head_num = params_json["n_heads"]
    config.attn_config.kv_head_num = params_json.get(
        "n_kv_heads", config.attn_config.head_num
    )
    config.attn_config.size_per_head = params_json["dim"] // params_json["n_heads"]
    config.num_layers = params_json["n_layers"]
    config.max_seq_len = 2048
    config.vocab_size = 32000
    config.layernorm_eps = params_json["norm_eps"]
    config.inter_size = compute_intermediate_size(
        params_json["dim"],
        params_json.get("ffn_dim_multiplier", 1),
        params_json["multiple_of"],
    )
    config.special_tokens.bos_token_id = 1
    config.special_tokens.eos_token_id = 2
    config.attn_config.rope_config.dim = config.attn_config.size_per_head
    config.tie_word_embeddings = params_json.get("tie_word_embeddings", False)
    config.config_dtype = params_json.get("torch_dtype", None)


@register_config_creator("llama")
def create_llama_config(ckpt_path: str) -> ModelConfig:
    """Create Llama model configuration."""
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 1

    # Try config.json first (HuggingFace format)
    config_path = os.path.join(ckpt_path, "config.json")
    param_path = os.path.join(ckpt_path, "params.json")

    if os.path.exists(config_path):
        with open(config_path) as reader:
            content = reader.read()
            content = content.replace("LlamaForCausalLM", "LLaMAForCausalLM")
            config_json = json.loads(content)
        _apply_llama_huggingface_config(config, config_json)
    elif os.path.exists(param_path):
        logger.info("llama not find config.json, use default config")
        with open(param_path) as reader:
            param_json = json.load(reader)
        _apply_llama_params_config(config, param_json)
    else:
        raise Exception("llama parameter from unknown source")

    return config


@register_config_creator("baichuan")
def create_baichuan_config(ckpt_path: str) -> ModelConfig:
    """Create Baichuan model configuration."""
    config = create_llama_config(ckpt_path)
    if config.num_layers == 40:  # 13B
        config.attn_config.rope_config.style = 0
        config.attn_config.rope_config.dim = 0
        config.use_attention_linear_bias = True
    config.special_tokens.bos_token_id = -1
    return config


@register_config_creator("baichuan2")
def create_baichuan2_config(ckpt_path: str) -> ModelConfig:
    """Create Baichuan2 model configuration."""
    config = create_baichuan_config(ckpt_path)
    config.normalize_lm_head_weight = True
    return config


@register_config_creator("gemma")
def create_gemma_config(ckpt_path: str) -> ModelConfig:
    """Create Gemma model configuration."""
    config = create_llama_config(ckpt_path)
    # Gemma uses the same base config as Llama
    config.has_post_decoder_layernorm = True
    config.input_embedding_scalar = config.hidden_size**0.5
    config.attn_config.rope_config.dim = config.attn_config.size_per_head
    config.activation_type = "gated-gelu"
    return config


@register_config_creator("cohere")
def create_cohere_config(ckpt_path: str) -> ModelConfig:
    """Create Cohere model configuration."""
    config = create_llama_config(ckpt_path)
    config.attn_config.rope_config.style = 0
    config.norm_type = "layernorm"
    config.qk_norm = True
    return config
