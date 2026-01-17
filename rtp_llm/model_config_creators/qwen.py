"""Configuration creators for Qwen series models."""

import json
import logging
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def hidden_to_inter(hidden_size):
    """Compute intermediate size for Qwen models."""
    ffn_m = 256
    return int((int(4 * 2 / 3 * hidden_size) * 2 + ffn_m - 1) // ffn_m * ffn_m / 2)


def _apply_qwen_hf_config_from_file(config: ModelConfig, ckpt_path: str):
    """Apply Qwen configuration from HuggingFace config.json file.

    This function reads config.json from ckpt_path and applies the configuration.
    This is equivalent to QWen._from_hf method.
    """
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path) as reader:
        content = reader.read()
        config_json = json.loads(content)
    _apply_qwen_hf_config(config, config_json)


def _apply_qwen_hf_config(config: ModelConfig, config_json: Dict[str, Any]):
    """Apply Qwen configuration from HuggingFace config.json."""
    config.attn_config.head_num = config_json.get(
        "n_head", config_json.get("num_attention_heads", config.attn_config.head_num)
    )
    config.attn_config.kv_head_num = config.attn_config.head_num
    config.attn_config.size_per_head = config_json.get(
        "kv_channels", config.attn_config.size_per_head
    )
    config.hidden_size = config_json.get("hidden_size", config.hidden_size)
    config.inter_size = int(
        config_json.get(
            "intermediate_size",
            config_json.get(
                "ffn_hidden_size",
                hidden_to_inter(
                    config.attn_config.head_num * config.attn_config.size_per_head
                )
                * 2,
            ),
        )
        / 2
    )
    config.layernorm_eps = config_json.get("layer_norm_epsilon", config.layernorm_eps)
    config.num_layers = config_json.get(
        "num_hidden_layers", config_json.get("n_layer", config.num_layers)
    )
    config.vocab_size = config_json.get(
        "vocab_size", config_json.get("padded_vocab_size", config.vocab_size)
    )
    config.attn_config.rope_config.base = config_json.get("rotary_emb_base", 10000)
    config.attn_config.rope_config.dim = config.attn_config.size_per_head
    config.special_tokens.eos_token_id = config_json.get(
        "eos_token_id", config.special_tokens.eos_token_id
    )
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

    if config_json.get("use_dynamic_ntk"):
        config.attn_config.rope_config.style = 4
    config.attn_config.rope_config.max_pos = config_json.get("seq_length", 8192)
    config.attn_config.use_logn_attn = config_json.get("use_logn_attn")


@register_config_creator("qwen")
def create_qwen_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen model configuration."""
    config = ModelConfig()
    config.vocab_size = 152064
    config.max_seq_len = 8192
    config = create_qwen_base_config(config, ckpt_path)

    assert (
        config.attn_config.head_num > 0
        and config.attn_config.kv_head_num > 0
        and config.attn_config.size_per_head > 0
        and config.num_layers > 0
    ), "error config"
    return config


@register_config_creator("qwenbase")
def create_qwen_base_config(config: ModelConfig, ckpt_path: str) -> ModelConfig:
    config.ckpt_path = ckpt_path
    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 1
    config.has_pre_decoder_layernorm = False
    config.layernorm_eps = 1e-5
    config.special_tokens.bos_token_id = -1
    config.special_tokens.eos_token_id = 151643
    # <|im_start|> and <|im_end|>
    config.special_tokens.stop_words_id_list = [[151645], [151644]]
    _apply_qwen_hf_config_from_file(config, ckpt_path)
    return config

    return config


@register_config_creator("qwen_v2")
def create_qwen_v2_config(ckpt_path: str) -> ModelConfig:
    """Create QwenV2 model configuration."""
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.vocab_size = 152064
    config.max_seq_len = 8192
    config.attn_config.rope_config.dim = 128
    config.attn_config.rope_config.style = 1
    config.has_pre_decoder_layernorm = False
    config.special_tokens.bos_token_id = -1
    config.special_tokens.eos_token_id = 151643
    # <|im_start|> and <|im_end|>
    config.special_tokens.stop_words_id_list = [[151645], [151644]]

    # Load and apply QwenV2 specific config
    config_json = require_config_json(ckpt_path)
    config.inter_size = config_json["intermediate_size"]
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.kv_head_num = config_json.get(
        "num_key_value_heads", config.attn_config.head_num
    )
    config.attn_config.size_per_head = (
        int(config_json.get("head_dim"))
        if "head_dim" in config_json
        else config_json["hidden_size"] // config.attn_config.head_num
    )
    if config_json.get("hidden_size") is not None:
        config.hidden_size = config_json["hidden_size"]
    config.num_layers = config_json["num_hidden_layers"]
    config.attn_config.rope_config.base = int(
        config_json.get("rope_theta", config.attn_config.rope_config.base)
    )
    config.vocab_size = config_json["vocab_size"]
    config.attn_config.rope_config.dim = config.attn_config.size_per_head
    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
    config.config_dtype = config_json.get("torch_dtype", None)

    assert (
        config.attn_config.head_num > 0
        and config.attn_config.kv_head_num > 0
        and config.attn_config.size_per_head > 0
        and config.num_layers > 0
        and config.inter_size > 0
    ), f"error config config.attn_config.head_num={config.attn_config.head_num} config.attn_config.kv_head_num={config.attn_config.kv_head_num} config.attn_config.size_per_head={config.attn_config.size_per_head} config.num_layers={config.num_layers} config.inter_size={config.inter_size}"
    return config


@register_config_creator("qwen_v2_moe")
def create_qwen_v2_moe_config(ckpt_path: str) -> ModelConfig:
    config = create_qwen_v2_config(ckpt_path)
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        raise Exception("qwen2 moe should have config.json")
    with open(config_path) as reader:
        content = reader.read()
        config_json = json.loads(content)
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
        raise Exception("Paritial moe weights for qwen2 is not implemented yet!")
    config.moe_layer_index = [
        i for i in range(moe_step - 1, config.num_layers, moe_step)
    ]
    return config


@register_config_creator("qwen_v3_moe")
def create_qwen_v3_moe_config(ckpt_path: str) -> ModelConfig:
    config = create_qwen_v2_config(ckpt_path)
    config.qk_norm = True
    config.moe_style = 1
    return config


@register_config_creator("qwen_v2_embedding")
def create_qwen_v2_embedding_config(ckpt_path: str) -> ModelConfig:
    config = create_qwen_v2_config(ckpt_path)
    config.attn_config.is_causal = False
    return config


@register_config_creator("qwen_v2_mtp")
def create_qwen_v2_mtp_config(ckpt_path: str) -> ModelConfig:
    config = create_qwen_v2_config(ckpt_path)
    config.moe_layer_index = [i for i in range(config.num_layers)]
    config.is_mtp = True
    return config


@register_config_creator("qwen_v3_moe_eagle3")
def create_qwen_v3_moe_eagle3_config(ckpt_path: str) -> ModelConfig:
    config = create_qwen_v2_config(ckpt_path)
    return config


@register_config_creator("qwen_7b")
def create_qwen_7b_config(ckpt_path: str) -> ModelConfig:
    config = ModelConfig()
    config.attn_config.head_num = 32
    config.attn_config.kv_head_num = 32
    config.attn_config.size_per_head = 128
    config.num_layers = 32
    config.inter_size = hidden_to_inter(4096)  # 11008
    config.vocab_size = 151936
    config.max_seq_len = 8192
    config = create_qwen_base_config(config, ckpt_path)

    return config


def create_qwen_13b_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen13B model configuration."""
    config = ModelConfig()
    config.attn_config.head_num = 40
    config.attn_config.kv_head_num = 40
    config.attn_config.size_per_head = 128
    config.num_layers = 40
    config.inter_size = hidden_to_inter(5120)  # 13696
    config.vocab_size = 152064
    config.max_seq_len = 8192
    config = create_qwen_base_config(config, ckpt_path)
    return config


def create_qwen_1b8_config(ckpt_path: str) -> ModelConfig:
    """Create Qwen1B8 model configuration."""
    config = ModelConfig()
    config.attn_config.head_num = 16
    config.attn_config.kv_head_num = 16
    config.attn_config.size_per_head = 128
    config.num_layers = 24
    config.inter_size = hidden_to_inter(2048)  # 5504
    config.vocab_size = 151936
    config.max_seq_len = 2048
    config = create_qwen_base_config(config, ckpt_path)
    return config


@register_config_creator("qwen_3")
@register_config_creator("qwen_3_tool")
def create_qwen_v3_config(ckpt_path: str) -> ModelConfig:
    """Create QwenV3 model configuration."""
    config = create_qwen_v2_config(ckpt_path)
    config.qk_norm = True
    return config
