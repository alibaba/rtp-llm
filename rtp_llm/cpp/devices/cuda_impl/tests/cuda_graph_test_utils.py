"""
Shared utilities for CUDA Graph tests.
This module provides common model building and KV cache initialization functionality.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import KVCache, init_device
from rtp_llm.tools.api.hf_model_helper import get_model_info_from_hf


@dataclass
class ModelBuildConfig:
    """Configuration for building a model."""

    model_path: str
    max_seq_len: int = 4096
    tokens_per_block: int = 64
    max_total_tokens: int = 4096
    hack_layer_num: int = 1
    device_reserve_memory_bytes: int = -536870912
    act_type: Optional[str] = None
    device: str = "cuda:0"


@dataclass
class ModelBuildResult:
    """Result of building a model."""

    model: GptModelBase
    model_config: object
    compute_dtype: torch.dtype
    kv_cache: Optional[KVCache] = None
    layer_num: int = 0
    block_nums: int = 0
    kv_head_num: int = 0
    size_per_head: int = 0
    tokens_per_block: int = 0


class CudaGraphTestModelBuilder:
    """
    Shared model builder for CUDA Graph tests.
    Encapsulates common model building logic used by both prefill and decode tests.
    """

    def __init__(self, config: ModelBuildConfig):
        self.config = config
        self.py_env_configs: Optional[PyEnvConfigs] = None
        self.gpt_model = None

    def _set_configs(self) -> None:
        """Set configuration structures instead of environment variables."""
        self.py_env_configs = PyEnvConfigs()

        # Get model info from HuggingFace
        model_path, model_type = get_model_info_from_hf(self.config.model_path, None)
        self.py_env_configs.model_args.model_type = model_type
        self.py_env_configs.model_args.ckpt_path = model_path
        self.py_env_configs.model_args.max_seq_len = self.config.max_seq_len
        self.py_env_configs.kv_cache_config.seq_size_per_block = (
            self.config.tokens_per_block
        )
        self.py_env_configs.profiling_debug_logging_config.hack_layer_num = (
            self.config.hack_layer_num
        )

        if self.config.act_type:
            self.py_env_configs.model_args.act_type = self.config.act_type

        if not self.py_env_configs.model_args.tokenizer_path:
            self.py_env_configs.model_args.tokenizer_path = model_path

        # Set ModelSpecificConfig.load_python_model = True
        self.py_env_configs.model_specific_config.load_python_model = True
        self.py_env_configs.device_resource_config.not_use_default_stream = True

        # Set DeviceResourceConfig.device_reserve_memory_bytes
        if self.py_env_configs.device_resource_config.device_reserve_memory_bytes == 0:
            self.py_env_configs.device_resource_config.device_reserve_memory_bytes = (
                self.config.device_reserve_memory_bytes
            )

    def build_model(self, init_kv_cache: bool = False) -> ModelBuildResult:
        """
        Build model using ModelFactory.

        Args:
            init_kv_cache: Whether to initialize KV cache (needed for decode tests)

        Returns:
            ModelBuildResult containing the model and related configurations
        """
        self._set_configs()

        # Create EngineConfig from py_env_configs
        engine_config = EngineConfig.create(self.py_env_configs)

        # Create model configs
        model_config = ModelFactory.create_model_config(
            model_args=self.py_env_configs.model_args,
            lora_config=self.py_env_configs.lora_config,
            kv_cache_config=engine_config.kv_cache_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            generate_env_config=self.py_env_configs.generate_env_config,
            embedding_config=self.py_env_configs.embedding_config,
            quantization_config=self.py_env_configs.quantization_config,
            render_config=self.py_env_configs.render_config,
        )

        # Update engine_config based on model_config
        ModelFactory.update_engine_config_from_model_config(
            engine_config=engine_config,
            model_config=model_config,
        )

        # Create model using ModelFactory
        self.gpt_model = ModelFactory._create_model(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=None,
            merge_lora=False,
        )

        # Load the model
        self.gpt_model.load()
        compute_dtype = self.gpt_model.weight.dtype
        model = self.gpt_model.py_model
        py_model_config = model.config

        # Init device with new API
        init_device(
            parallelism_config=engine_config.parallelism_config,
            model_config=model_config,
            eplb_config=model_config.eplb_config,
            fmha_config=engine_config.fmha_config,
            device_resource_config=engine_config.device_resource_config,
            moe_config=engine_config.moe_config,
            sp_config=engine_config.sp_config,
            misc_config=engine_config.misc_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            hw_kernel_config=engine_config.hw_kernel_config,
            concurrency_config=engine_config.concurrency_config,
            ffn_disaggregate_config=engine_config.parallelism_config.ffn_disaggregate_config,
            runtime_config=engine_config.runtime_config,
        )

        result = ModelBuildResult(
            model=model,
            model_config=py_model_config,
            compute_dtype=compute_dtype,
        )

        # Initialize KV cache if requested
        if init_kv_cache:
            self._init_kv_cache(result, py_model_config)
            model.kv_cache = result.kv_cache

        return result

    def _init_kv_cache(self, result: ModelBuildResult, model_config) -> None:
        """Initialize KV cache, similar to auto_model.py"""
        result.kv_cache = KVCache()
        result.layer_num = model_config.num_layers
        result.kv_head_num = model_config.attn_config.kv_head_num
        result.size_per_head = model_config.attn_config.size_per_head
        result.tokens_per_block = model_config.attn_config.tokens_per_block

        result.block_nums = math.ceil(
            self.config.max_total_tokens / result.tokens_per_block
        )
        # since block_id start from 1, so we should add 1 in the corner case
        result.block_nums += 1

        kv_shape = [
            result.layer_num,
            result.block_nums,
            2,
            result.kv_head_num,
            result.tokens_per_block,
            result.size_per_head,
        ]

        kv_cache_total = torch.zeros(
            kv_shape, dtype=result.compute_dtype, device=self.config.device
        )
        k_cache_base = kv_cache_total
        v_cache_base = torch.empty(
            result.layer_num,
            0,
            result.kv_head_num,
            result.tokens_per_block,
            result.size_per_head,
            device=self.config.device,
        )

        result.kv_cache.k_cache_base = k_cache_base
        result.kv_cache.v_cache_base = v_cache_base
