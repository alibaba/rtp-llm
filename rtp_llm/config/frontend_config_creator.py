"""Frontend config creator for creating minimal ModelConfig without model class dependency.

This module provides a simplified config creator that creates ModelConfig instances
for frontend use cases (OpenaiEndpoint and EmbeddingEndpoint) without requiring
model class instantiation or dependency on rtp_llm.models.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, get_task_type_from_ckpt_path
from rtp_llm.config.py_config_modules import (
    EmbeddingConfig,
    GenerateEnvConfig,
    LoraConfig,
    ProfilingDebugLoggingConfig,
    QuantizationConfig,
    RenderConfig,
)
from rtp_llm.ops import TaskType
from rtp_llm.utils.util import get_config_from_path


def _get_max_seq_len_from_config_json(
    config_json: Dict[str, Any], model_type: str
) -> Optional[int]:
    """Extract max_seq_len from config.json based on model type.

    Different models use different field names:
    - BERT/RoBERTa: "max_position_embeddings"
    - Most models: "max_sequence_length"
    - Some models: "seq_length" (e.g., BLOOM)
    - Some models: "max_position_embeddings" (e.g., StarCoder2, Qwen3Next, TBStars2_5)
    - Internal models: "max_sequence_length" (e.g., MixtBStars) or "max_position_embeddings" (e.g., TBStars2_5)
    - FLOT: hardcoded 8192, but may be overridden by config.json

    Args:
        config_json: Parsed config.json content
        model_type: Model type string

    Returns:
        max_seq_len value if found, None otherwise
    """
    # Try different field names based on model type
    if model_type in ["bert", "roberta", "megatron_bert", "jina_bert"]:
        # BERT family uses max_position_embeddings
        return config_json.get("max_position_embeddings")
    elif model_type in ["starcoder2", "qwen3_next", "tbstars2_5"]:
        # These models use max_position_embeddings
        return config_json.get("max_position_embeddings")
    elif model_type in ["bloom"]:
        # BLOOM uses seq_length
        return config_json.get("seq_length")
    elif model_type in ["mixtbstars", "mixtbstars-mtp"]:
        # MixtBStars uses max_sequence_length (default 2048 in model class)
        return config_json.get("max_sequence_length")
    elif model_type in ["flot", "flot_vl", "flot_005"]:
        # FLOT hardcodes 8192, but check config.json first
        # FLOT may have max_position_embeddings in text_config or root
        if "text_config" in config_json:
            text_config = config_json["text_config"]
            return text_config.get("max_position_embeddings")
        return config_json.get("max_position_embeddings")
    else:
        # Most models use max_sequence_length, fallback to max_position_embeddings
        return config_json.get("max_sequence_length") or config_json.get(
            "max_position_embeddings"
        )


def _apply_model_specific_config_from_json(
    config: ModelConfig, config_json: Dict[str, Any], model_type: str
) -> None:
    """Apply model-specific configuration from config.json.

    This function extracts and sets various configuration parameters that differ
    between models, including architecture parameters, activation types, normalization
    types, RoPE configuration, special tokens, and MOE settings.

    Args:
        config: ModelConfig instance to populate
        config_json: Parsed config.json content
        model_type: Model type string
    """
    # Basic architecture parameters (common across most models)
    if "vocab_size" in config_json:
        config.vocab_size = config_json["vocab_size"]
    if "hidden_size" in config_json:
        config.hidden_size = config_json["hidden_size"]
    if "num_hidden_layers" in config_json:
        config.num_layers = config_json["num_hidden_layers"]
    if "intermediate_size" in config_json:
        config.inter_size = config_json["intermediate_size"]

    # Attention configuration
    if "num_attention_heads" in config_json:
        config.attn_config.head_num = config_json["num_attention_heads"]
    if "num_key_value_heads" in config_json:
        config.attn_config.kv_head_num = config_json["num_key_value_heads"]
    elif "num_attention_heads" in config_json:
        # Default kv_head_num to head_num if not specified
        config.attn_config.kv_head_num = config_json["num_attention_heads"]

    # Calculate size_per_head if not directly specified
    if "head_dim" in config_json:
        config.attn_config.size_per_head = config_json["head_dim"]
    elif "hidden_size" in config_json and config.attn_config.head_num > 0:
        config.attn_config.size_per_head = (
            config_json["hidden_size"] // config.attn_config.head_num
        )

    # Activation type (varies by model)
    # Set default first (same as model classes do before reading config.json)
    if model_type in ["bert", "roberta", "megatron_bert", "jina_bert", "vision_bert"]:
        # BERT family default is "gelu" (same as Bert._create_config)
        config.activation_type = "gelu"

    if "hidden_act" in config_json:
        hidden_act = config_json["hidden_act"].lower()
        # Map common activation types
        activation_map = {
            "gelu": "gelu",
            "gelu_new": "gelu",
            "relu": "relu",
            "silu": "SiGLU",
            "swiglu": "SiGLU",
            "gated-silu": "gated-silu",
            "gated_silu": "gated-silu",
        }
        config.activation_type = activation_map.get(hidden_act, hidden_act)

    # Normalization type (varies by model)
    # BERT family models
    if model_type in ["bert", "roberta", "megatron_bert", "jina_bert", "vision_bert"]:
        config.norm_type = "layernorm"
        config.layernorm_type = "post_layernorm"
        config.has_pre_decoder_layernorm = True
        config.has_post_decoder_layernorm = False
        config.attn_config.is_causal = False
    # LLM models with RMSNorm
    elif model_type in [
        "llama",
        "baichuan",
        "qwen",
        "qwen_v2",
        "qwen_v3",
        "deepseek_v2",
        "deepseek2-vl",
        "qwen_vl",
        "qwen_vl_1b8",
        "qwen2_vl",
        "qwen3_vl_moe",
        "minicpmv",
        "minicpmv_embedding",
        "chat_glm_v2",
        "chat_glm_v3",
        "chat_glm_v4",
        "internvl",
        "llava",
    ]:
        config.norm_type = "rmsnorm"
        config.layernorm_type = "pre_layernorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.attn_config.is_causal = True
    # Internal models (FLOT, MixtBStars, TBStars)
    elif model_type in [
        "flot",
        "flot_vl",
        "mixtbstars",
        "mixtbstars-mtp",
        "tbstars2_5",
        "tbstars_vl",
        "tbstars_vl_002",
        "tbstars_vl_004",
        "tbstars_vl_008o",
        "biencoder_vl_tbstars",
    ]:
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        # Check bidirectional_attention for causal setting
        bidirectional = config_json.get("bidirectional_attention", False)
        if isinstance(bidirectional, bool):
            config.attn_config.is_causal = not bidirectional
        elif isinstance(bidirectional, dict):
            # Some models have bidirectional_attention as a dict
            config.attn_config.is_causal = not bidirectional.get("enabled", False)
        else:
            config.attn_config.is_causal = True  # Default to causal

    # LayerNorm epsilon
    if "layer_norm_eps" in config_json:
        config.layernorm_eps = config_json["layer_norm_eps"]
    elif "rms_norm_eps" in config_json:
        config.layernorm_eps = config_json["rms_norm_eps"]

    # RoPE configuration
    if "rope_theta" in config_json:
        config.attn_config.rope_config.base = int(config_json["rope_theta"])

    # Set RoPE dim based on model type or config
    # Models with RoPE
    if model_type in [
        "llama",
        "baichuan",
        "qwen",
        "qwen_v2",
        "qwen_v3",
        "qwen_vl",
        "qwen_vl_1b8",
        "qwen2_vl",
        "qwen3_vl_moe",
        "flot",
        "flot_vl",
        "mixtbstars",
        "mixtbstars-mtp",
        "tbstars2_5",
        "tbstars_vl",
        "tbstars_vl_002",
        "tbstars_vl_004",
        "tbstars_vl_008o",
        "deepseek_v2",
        "deepseek2-vl",
        "minicpmv",
        "minicpmv_embedding",
        "chat_glm_v2",
        "chat_glm_v3",
        "chat_glm_v4",
        "internvl",
        "llava",
    ]:
        if config.attn_config.size_per_head > 0:
            config.attn_config.rope_config.dim = config.attn_config.size_per_head
        # Default RoPE style is 1, but may be overridden by rope_scaling
        if config.attn_config.rope_config.style == 0:
            config.attn_config.rope_config.style = 1
    # BERT family (no RoPE)
    elif model_type in ["bert", "roberta", "megatron_bert", "jina_bert", "vision_bert"]:
        config.attn_config.rope_config.dim = 0
        config.attn_config.rope_config.style = 0
        # Roberta uses position_ids_style = 1 (same as Roberta.from_huggingface)
        # This affects truncate_length calculation in CommonInputGenerator
        if model_type == "roberta":
            config.position_ids_style = 1

    # Multimodal models with special position_ids_style
    # Qwen2VL uses mm_model_config.mm_position_ids_style = 2 (same as QWen2_VL._from_hf)
    if model_type in ["qwen2_vl", "qwen2vl"]:
        config.mm_model_config.mm_position_ids_style = 2
    # ChatGlmV4Vision uses mm_position_ids_style = 1 (same as ChatGlmV4Vision._create_config)
    elif model_type in ["chatglm4v", "chat_glm_v4_vision"]:
        config.mm_position_ids_style = 1

    # RoPE scaling (for models with extended context)
    rope_scaling = config_json.get("rope_scaling")
    if rope_scaling is not None:
        rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
        if rope_type == "linear":
            config.attn_config.rope_config.scale = rope_scaling.get("factor", 1.0)
            config.attn_config.rope_config.max_pos = config_json.get(
                "max_position_embeddings", 2048
            )
        elif rope_type == "dynamic":
            config.attn_config.rope_config.style = 3
        elif rope_type == "yarn":
            config.attn_config.rope_config.style = 5
            config.attn_config.rope_config.scale = rope_scaling.get("factor", 1.0)
            config.attn_config.rope_config.factor1 = rope_scaling.get("beta_slow", 1)
            config.attn_config.rope_config.factor2 = rope_scaling.get("beta_fast", 32)
            config.attn_config.rope_config.max_pos = rope_scaling.get(
                "original_max_position_embeddings", 2048
            )
        elif rope_type == "llama3":
            config.attn_config.rope_config.style = 6
            config.attn_config.rope_config.scale = rope_scaling.get("factor", 1.0)
            config.attn_config.rope_config.factor1 = rope_scaling.get(
                "low_freq_factor", 1.0
            )
            config.attn_config.rope_config.factor2 = rope_scaling.get(
                "high_freq_factor", 1.0
            )
            config.attn_config.rope_config.max_pos = rope_scaling.get(
                "original_max_position_embeddings", 2048
            )

    # Special tokens
    eos_token_id = config_json.get("eos_token_id")
    if eos_token_id is not None:
        if isinstance(eos_token_id, list):
            config.special_tokens.eos_token_id = eos_token_id[0]
            config.special_tokens.stop_words_id_list = [[id] for id in eos_token_id]
        else:
            config.special_tokens.eos_token_id = eos_token_id

    if "bos_token_id" in config_json:
        bos_token_id = config_json["bos_token_id"]
        if bos_token_id is not None:
            config.special_tokens.bos_token_id = bos_token_id

    if "pad_token_id" in config_json:
        config.special_tokens.pad_token_id = config_json["pad_token_id"]
    elif model_type in ["chat_glm_2", "chatglm2"]:
        # ChatGlmV2 default pad_token_id is 0 (same as ChatGlmV2.from_huggingface)
        config.special_tokens.pad_token_id = 0

    # Stop words (some models define them in config)
    if "stop_words_id_list" in config_json:
        config.special_tokens.stop_words_id_list = config_json["stop_words_id_list"]

    # Type vocab size (for BERT-like models)
    if "type_vocab_size" in config_json:
        config.type_vocab_size = config_json["type_vocab_size"]
    elif model_type in ["bert", "roberta", "megatron_bert", "jina_bert", "vision_bert"]:
        # BERT family default is 0 (same as Bert.from_huggingface)
        config.type_vocab_size = 0

    # Embedding configuration
    if "tie_word_embeddings" in config_json:
        config.tie_word_embeddings = config_json["tie_word_embeddings"]

    # Positional encoding
    if model_type in ["bert", "roberta", "megatron_bert", "jina_bert", "vision_bert"]:
        config.has_positional_encoding = True
    elif model_type in [
        "flot",
        "flot_vl",
        "qwen_v2",
        "qwen_v3",
        "qwen_vl",
        "qwen_vl_1b8",
        "qwen2_vl",
        "qwen3_vl_moe",
        "llama",
        "deepseek_v2",
        "deepseek2-vl",
        "minicpmv",
        "minicpmv_embedding",
        "chat_glm_v2",
        "chat_glm_v3",
        "chat_glm_v4",
        "internvl",
        "llava",
        "mixtbstars",
        "mixtbstars-mtp",
        "tbstars2_5",
        "tbstars_vl",
        "tbstars_vl_002",
        "tbstars_vl_004",
        "tbstars_vl_008o",
        "biencoder_vl_tbstars",
    ]:
        config.has_positional_encoding = False

    # MOE configuration (for MoE models)
    if (
        "num_local_experts" in config_json
        or "num_routed_experts" in config_json
        or "n_routed_experts" in config_json
    ):
        config.expert_num = (
            config_json.get("num_local_experts")
            or config_json.get("num_routed_experts")
            or config_json.get("n_routed_experts", 0)
        )

    if "num_experts_per_tok" in config_json:
        config.moe_k = config_json["num_experts_per_tok"]

    if "moe_intermediate_size" in config_json:
        config.moe_inter_size = config_json["moe_intermediate_size"]

    if "scoring_func" in config_json:
        scoring_func = config_json["scoring_func"]
        if scoring_func == "softmax":
            config.scoring_func = 0
        elif scoring_func == "sigmoid":
            config.scoring_func = 1

    if "routed_scaling_factor" in config_json:
        config.routed_scaling_factor = config_json["routed_scaling_factor"]

    if "n_group" in config_json:
        config.moe_n_group = config_json["n_group"]

    if "topk_group" in config_json:
        config.moe_topk_group = config_json["topk_group"]

    if "norm_topk_prob" in config_json:
        config.has_moe_norm = config_json["norm_topk_prob"]

    # MOE layer indices
    if "moe_layer_freq" in config_json and "first_k_dense_replace" in config_json:
        moe_step = config_json["moe_layer_freq"]
        first_k_dense_replace = config_json["first_k_dense_replace"]
        config.moe_layer_index = [
            i
            for i in range(config.num_layers)
            if i >= first_k_dense_replace and i % moe_step == 0
        ]

    # Template type (for chat models)
    if "template_type" in config_json:
        config.template_type = config_json["template_type"]

    # Model name
    if "model_type" in config_json:
        config.model_name = config_json["model_type"]

    # Config dtype (torch_dtype from config.json, used for weight loading)
    # This is set by Bert.from_huggingface but may not be needed for frontend
    # Only set if present in config.json
    if "torch_dtype" in config_json:
        config.config_dtype = config_json["torch_dtype"]


def create_frontend_model_config(
    model_args: ModelArgs,
    lora_config: LoraConfig,
    kv_cache_config: KVCacheConfig,
    profiling_debug_logging_config: ProfilingDebugLoggingConfig,
    generate_env_config: Optional[GenerateEnvConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    quantization_config: Optional[QuantizationConfig] = None,
    render_config: Optional[RenderConfig] = None,
) -> ModelConfig:
    """Create minimal ModelConfig for frontend use cases.

    This function creates a ModelConfig instance suitable for OpenaiEndpoint and
    EmbeddingEndpoint without requiring model class instantiation or ModelFactory dependency.
    It reads configuration from config.json and model_args, applying model-specific
    configuration logic based on model_type.

    Args:
        model_args: ModelArgs containing model configuration
        lora_config: LoraConfig containing LoRA configuration
        kv_cache_config: KVCacheConfig for model config building
        profiling_debug_logging_config: ProfilingDebugLoggingConfig for model config building
        generate_env_config: Optional GenerateEnvConfig for generation settings
        embedding_config: Optional EmbeddingConfig for embedding settings
        quantization_config: Optional QuantizationConfig for quantization settings
        render_config: Optional RenderConfig for renderer factory settings

    Returns:
        ModelConfig instance suitable for frontend endpoints
    """
    # Create ModelConfig instance without dependency on ModelFactory
    model_config = ModelConfig()
    model_config.ckpt_path = model_args.ckpt_path
    model_config.tokenizer_path = model_args.tokenizer_path
    model_config.extra_data_path = model_args.extra_data_path
    model_config.local_extra_data_path = model_args.local_extra_data_path
    model_config.model_type = model_args.model_type
    model_config.phy2log_path = model_args.phy2log_path

    if model_args.mla_ops_type:
        model_config.mla_ops_type = model_args.mla_ops_type

    # Get task_type first
    model_config.task_type = get_task_type_from_ckpt_path(
        model_args.task_type,
        model_args.ckpt_path,
        embedding_config,
    )

    # Try to read from config.json and apply model-specific settings
    # This mimics what model classes do in _create_config
    config_json = get_config_from_path(model_args.ckpt_path)
    if config_json:
        # Get max_seq_len using model-specific logic (same as model classes)
        max_seq_len_from_json = _get_max_seq_len_from_config_json(
            config_json, model_args.model_type
        )
        if max_seq_len_from_json:
            # Set max_seq_len from config.json (like model classes do)
            model_config.max_seq_len = max_seq_len_from_json
        else:
            # If not found in config.json, don't set it yet (will use default later)
            pass

        # Apply all model-specific configuration
        _apply_model_specific_config_from_json(
            model_config, config_json, model_args.model_type
        )
    else:
        logging.warning(f"Could not read config.json from {model_args.ckpt_path}")

    # Apply model_args overrides (same logic as build_model_config)
    # This matches the behavior: if model_args.max_seq_len is set, it overrides
    if model_args.max_seq_len:
        model_config.max_seq_len = model_args.max_seq_len

    # Set default only if max_seq_len is still not set (same as build_model_config)
    # Use model-specific defaults to match model class behavior
    if not model_config.max_seq_len:
        if model_args.model_type in [
            "bert",
            "roberta",
            "megatron_bert",
            "jina_bert",
            "vision_bert",
        ]:
            # BERT family default is 512 (same as Bert.from_huggingface)
            model_config.max_seq_len = 512
        else:
            # Other models default to 8192
            model_config.max_seq_len = 8192

    logging.info(
        f"max_seq_len: {model_config.max_seq_len} (task_type: {model_config.task_type})"
    )

    # Set quantization from quantization_config
    if quantization_config is not None:
        model_config.quantization = quantization_config.get_quantization()

    # Initialize precision configuration
    model_config.init_precision_config(
        kv_cache_config=kv_cache_config, act_type=model_args.act_type
    )
    model_config.attn_config.tokens_per_block = kv_cache_config.seq_size_per_block

    model_config.use_kvcache = model_config.task_type == TaskType.LANGUAGE_MODEL
    logging.info(
        f"model task type: {model_config.task_type}, use_kvcache: {model_config.use_kvcache}"
    )

    # Set lora_infos from lora_config
    if lora_config.lora_info:
        lora_infos = json.loads(lora_config.lora_info)
        model_config.lora_infos = lora_infos if lora_infos else {}

    # Set model_name (default to model_type)
    if not hasattr(model_config, "model_name") or not model_config.model_name:
        model_config.model_name = model_args.model_type

    # Set renderer configuration fields
    model_config.generate_env_config = (
        generate_env_config if generate_env_config is not None else GenerateEnvConfig()
    )
    model_config.render_config = (
        render_config if render_config is not None else RenderConfig()
    )

    return model_config
