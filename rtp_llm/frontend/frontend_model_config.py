import json
import logging
from typing import Any, Optional

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.config.py_config_modules import (
    EmbeddingConfig,
    GenerateEnvConfig,
    LoraConfig,
    QuantizationConfig,
    RenderConfig,
)
from rtp_llm.ops import ProfilingDebugLoggingConfig
from rtp_llm.utils.util import get_config_from_path


def _first_int(config_json: dict[str, Any], keys: list[str]) -> Optional[int]:
    for key in keys:
        value = config_json.get(key)
        if isinstance(value, int):
            return value
    return None


def _apply_checkpoint_defaults(model_config: ModelConfig, ckpt_path: str) -> None:
    config_json = get_config_from_path(ckpt_path)
    if not config_json:
        return

    text_config = config_json.get("text_config")
    if not isinstance(text_config, dict):
        text_config = config_json

    model_config.config_dtype = text_config.get(
        "torch_dtype", config_json.get("torch_dtype", None)
    )
    max_seq_len = _first_int(
        text_config,
        [
            "max_sequence_length",
            "max_position_embeddings",
            "seq_length",
            "n_positions",
        ],
    )
    if max_seq_len is not None:
        model_config.max_seq_len = max_seq_len


def create_frontend_model_config(
    model_args: ModelArgs,
    lora_config: LoraConfig,
    kv_cache_config: KVCacheConfig,
    profiling_debug_logging_config: ProfilingDebugLoggingConfig,
    generate_env_config: Optional[GenerateEnvConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    quantization_config: Optional[QuantizationConfig] = None,
    render_config: Optional[Any] = None,
) -> ModelConfig:
    """Create the ModelConfig fields consumed by standalone frontend startup."""
    model_config = ModelConfig()
    _apply_checkpoint_defaults(model_config, model_args.ckpt_path)

    build_model_config(
        model_config=model_config,
        model_args=model_args,
        kv_cache_config=kv_cache_config,
        profiling_debug_logging_config=profiling_debug_logging_config,
        embedding_config=embedding_config,
        quantization_config=quantization_config,
    )

    if lora_config.lora_info:
        lora_infos = json.loads(lora_config.lora_info)
        model_config.lora_infos = lora_infos if lora_infos else {}

    model_config.model_name = model_args.model_type
    model_config.generate_env_config = (
        generate_env_config if generate_env_config is not None else GenerateEnvConfig()
    )
    model_config.render_config = (
        render_config if render_config is not None else RenderConfig()
    )

    logging.info("frontend model_config: %s", model_config.to_string())
    return model_config
