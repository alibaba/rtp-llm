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
    EmbeddingEndpoint without requiring model class instantiation. It reads basic
    configuration from config.json and model_args, but does not require the full
    model architecture configuration that would normally be provided by model classes.

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
    # Create a minimal ModelConfig instance
    model_config = ModelConfig()
    model_config.ckpt_path = model_args.ckpt_path
    model_config.tokenizer_path = model_args.tokenizer_path
    model_config.extra_data_path = model_args.extra_data_path
    model_config.local_extra_data_path = model_args.local_extra_data_path
    model_config.model_type = model_args.model_type
    model_config.phy2log_path = model_args.phy2log_path

    if model_args.mla_ops_type:
        model_config.mla_ops_type = model_args.mla_ops_type

    # Set max_seq_len from model_args or config.json
    if model_args.max_seq_len:
        model_config.max_seq_len = model_args.max_seq_len
    else:
        # Try to read from config.json
        # Different models use different field names:
        # - Most models: "max_sequence_length"
        # - BERT/RoBERTa: "max_position_embeddings"
        config_json = get_config_from_path(model_args.ckpt_path)
        if config_json:
            # Try max_sequence_length first, then max_position_embeddings (for BERT/RoBERTa)
            max_seq_len = config_json.get("max_sequence_length") or config_json.get(
                "max_position_embeddings"
            )
            model_config.max_seq_len = max_seq_len if max_seq_len else 8192
        else:
            model_config.max_seq_len = 8192
    logging.info(f"max_seq_len: {model_config.max_seq_len}")

    # Get task_type from checkpoint path
    model_config.task_type = get_task_type_from_ckpt_path(
        model_args.task_type,
        model_config.ckpt_path,
        embedding_config,
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
    model_config.model_name = model_args.model_type

    # Set renderer configuration fields
    model_config.generate_env_config = (
        generate_env_config if generate_env_config is not None else GenerateEnvConfig()
    )
    model_config.render_config = (
        render_config if render_config is not None else RenderConfig()
    )

    # Try to read template_type from config.json if available
    config_json = get_config_from_path(model_args.ckpt_path)
    if config_json:
        # Some models have template_type in config.json
        model_config.template_type = config_json.get("template_type", None)
    else:
        model_config.template_type = None

    # Read basic model architecture info from config.json if available
    # This is optional and only for compatibility, not required for frontend
    if config_json:
        # Set some basic fields that might be needed
        if "vocab_size" in config_json:
            model_config.vocab_size = config_json["vocab_size"]
        if "hidden_size" in config_json:
            model_config.hidden_size = config_json["hidden_size"]
        if "num_hidden_layers" in config_json:
            model_config.num_layers = config_json["num_hidden_layers"]

    return model_config
