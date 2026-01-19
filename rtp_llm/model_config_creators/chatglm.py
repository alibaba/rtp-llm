"""Configuration creators for ChatGLM series models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import get_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _apply_chatglm_huggingface_config(config: ModelConfig, config_json: Dict[str, Any]):
    """Apply ChatGLM configuration from HuggingFace config.json."""
    config.attn_config.head_num = 32
    config.attn_config.size_per_head = 128
    config.num_layers = 32
    config.max_seq_len = 8192
    config.vocab_size = 65024
    config.attn_config.head_num = config_json["num_attention_heads"]
    if config_json.get("multi_query_attention", False):
        config.attn_config.kv_head_num = config_json["multi_query_group_num"]
    else:
        config.attn_config.kv_head_num = config.attn_config.head_num
    config.attn_config.size_per_head = (
        config_json["hidden_size"] // config_json["num_attention_heads"]
    )
    config.num_layers = config_json["num_layers"]
    config.max_seq_len = config_json.get("seq_length", 8192)
    config.vocab_size = config_json["padded_vocab_size"]
    config.layernorm_eps = config_json["layernorm_epsilon"]
    config.inter_size = config_json["ffn_hidden_size"]
    config.add_bias_linear = config_json["add_bias_linear"]
    config.has_post_decoder_layernorm = config_json["post_layer_norm"]
    if "pre_seq_len" in config_json:
        config.pre_seq_len = config_json["pre_seq_len"]
    if "prefix_projection" in config_json:
        config.prefix_projection = config_json["prefix_projection"]
    config.src_quantization_bit = config_json.get("quantization_bit", 0)
    config.attn_config.rope_config.dim = config.attn_config.size_per_head
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    config.special_tokens.pad_token_id = config_json.get("pad_token_id", 0)
    config.attn_config.rope_config.scale = config_json.get("rope_ratio", 1)
    config.special_tokens.eos_token_id = config_json.get("eos_token_id", 2)
    config.config_dtype = config_json.get("torch_dtype", None)


def _apply_chatglm_base_config(config: ModelConfig):
    """Apply base ChatGLM configuration."""
    config.use_attention_linear_bias = False
    config.activation_type = "SiGLU"
    config.norm_type = "rmsnorm"
    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 2


@register_config_creator("chatglm2")
@register_config_creator("chat_glm_2")
def create_chatglm_v2_config(ckpt_path: str) -> ModelConfig:
    """Create ChatGLM V2 model configuration."""
    config_json = get_config_json(ckpt_path)

    if config_json is not None:
        config = ModelConfig()
        _apply_chatglm_huggingface_config(config, config_json)
    else:
        # Default config
        config = ModelConfig()
        config.attn_config.head_num = 32
        config.attn_config.kv_head_num = 2
        config.attn_config.size_per_head = 128
        config.num_layers = 32
        config.max_seq_len = 8192
        config.vocab_size = 65024
        config.layernorm_eps = 1e-5
        config.inter_size = 13696
        config.add_bias_linear = False
        config.has_post_decoder_layernorm = False

    config.ckpt_path = ckpt_path
    _apply_chatglm_base_config(config)
    return config


@register_config_creator("chatglm3")
@register_config_creator("chat_glm_3")
def create_chatglm_v3_config(ckpt_path: str) -> ModelConfig:
    """Create ChatGLM V3 model configuration."""
    config = create_chatglm_v2_config(ckpt_path)

    # ChatGLM V3 specific: update rope base
    config_json = get_config_json(ckpt_path)
    if config_json:
        config.attn_config.rope_config.base = config_json.get(
            "rope_theta", 10000
        ) * int(config_json.get("rope_ratio", 1))

    return config


@register_config_creator("chatglm4")
def create_chatglm_v4_config(ckpt_path: str) -> ModelConfig:
    """Create ChatGLM V4 model configuration."""
    config = create_chatglm_v3_config(ckpt_path)

    # ChatGLM V4 specific: handle eos_token_id as list
    config_json = get_config_json(ckpt_path)
    if config_json and isinstance(config_json.get("eos_token_id"), list):
        config.special_tokens.eos_token_id = config_json["eos_token_id"][0]
        config.special_tokens.stop_words_id_list = [
            [x] for x in config_json["eos_token_id"]
        ]

    return config
