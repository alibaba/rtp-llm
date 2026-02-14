import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

import rtp_llm.models
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_device,
    get_typemeta,
    init_device,
)
from rtp_llm.tools.api.hf_model_helper import get_model_info_from_hf
from rtp_llm.utils.model_weight import W


class AutoModel:
    def __init__(
        self,
        model_path_or_name: str,
        revision: Optional[str] = None,
        max_total_tokens: int = 4096,
        tokens_per_block: int = 64,
        stop_id_list: list[int] = [],
        **kwargs,
    ):
        # set configs instead of environment variables
        self._set_configs()

        model_path, model_type = get_model_info_from_hf(model_path_or_name, revision)
        self.py_env_configs.model_args.model_type = model_type
        self.py_env_configs.model_args.ckpt_path = model_path
        self.py_env_configs.model_args.max_seq_len = max_total_tokens
        self.py_env_configs.kv_cache_config.seq_size_per_block = tokens_per_block
        if not self.py_env_configs.model_args.tokenizer_path:
            self.py_env_configs.model_args.tokenizer_path = model_path

        # Create EngineConfig from py_env_configs
        engine_config = EngineConfig.create(self.py_env_configs, nccl_comm_config=None)

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
        self.compute_dtype = self.gpt_model.weight.dtype
        self.model = self.gpt_model.py_model
        self.model_config = self.gpt_model.model_config

        # init device - use engine_config's configs and model_config's eplb_config
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
            model_specific_config=engine_config.model_specific_config,
        )
        self.device = "cuda"

        # init kv cache and bind it to py model
        self.tokens_per_block = self.model_config.attn_config.tokens_per_block
        self.block_nums = math.ceil(max_total_tokens / self.tokens_per_block)
        # since block_id start from 1, so we should add 1 in the corner case
        self.block_nums += 1
        logging.info(f"total block nums: {self.block_nums}")
        self._init_kv_cache()

        self.model.kv_cache = self.kv_cache

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.ckpt_path)

        # add eos_token_id to stop_id_list
        self.stop_id_list = stop_id_list.copy() if stop_id_list else []
        if (
            self.tokenizer.eos_token_id is not None
            and self.tokenizer.eos_token_id not in self.stop_id_list
        ):
            self.stop_id_list.append(self.tokenizer.eos_token_id)

        # convert hidden_states(after final layernorm) to logits
        self.lm_head_weight = self.model.weight.get_global_weight(W.lm_head)

    @classmethod
    def from_pretrained(cls, model_path_or_name: str, **kwargs) -> "AutoModel":
        return cls(model_path_or_name, **kwargs)

    def _set_configs(self):
        """Set configuration structures instead of environment variables."""
        # Create PyEnvConfigs to hold all configurations
        self.py_env_configs = PyEnvConfigs()

        # Set ModelSpecificConfig.load_python_model = True (equivalent to LOAD_PYTHON_MODEL=1)
        self.py_env_configs.model_specific_config.load_python_model = True

        # Set DeviceResourceConfig.device_reserve_memory_bytes (equivalent to DEVICE_RESERVE_MEMORY_BYTES)
        # Default: 2GB = 2 * 1024 * 1024 * 1024 bytes
        if self.py_env_configs.device_resource_config.device_reserve_memory_bytes == 0:
            self.py_env_configs.device_resource_config.device_reserve_memory_bytes = (
                2 * 1024 * 1024 * 1024
            )

    def _init_kv_cache(self):
        self.kv_cache = KVCache()
        self.layer_num = self.model_config.num_layers
        self.kv_head_num = self.model_config.attn_config.kv_head_num
        self.size_per_head = self.model_config.attn_config.size_per_head
        self.tokens_per_block = self.model_config.attn_config.tokens_per_block
        kv_shape = [
            self.layer_num,
            self.block_nums,
            2,
            self.kv_head_num,
            self.tokens_per_block,
            self.size_per_head,
        ]

        kv_cache_total = torch.zeros(
            kv_shape, dtype=self.compute_dtype, device=self.device
        )
        kv_cache_base = kv_cache_total
        self.kv_cache.kv_cache_base = kv_cache_base

    def _prepare_prefill_attention_inputs(self, input_length: int) -> PyAttentionInputs:
        need_block_nums = self._check_block_nums(input_length)
        attention_inputs = PyAttentionInputs()
        attention_inputs.input_lengths = torch.tensor([input_length], dtype=torch.int32)
        attention_inputs.sequence_lengths = torch.tensor([], dtype=torch.int32)
        attention_inputs.cu_seqlens = torch.tensor(
            [0, input_length], dtype=torch.int32, device=self.device
        )
        attention_inputs.prefix_lengths = torch.tensor([0], dtype=torch.int32)
        attention_inputs.padding_offset = torch.tensor(
            [0 for _ in range(input_length)], dtype=torch.int32, device=self.device
        )
        attention_inputs.kv_cache_block_id_device = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]],
            dtype=torch.int32,
            device=self.device,
        )
        attention_inputs.kv_cache_block_id_host = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]], dtype=torch.int32
        )
        attention_inputs.dtype = get_typemeta(self.kv_cache.kv_cache_base)
        attention_inputs.is_prefill = True
        return attention_inputs

    def _check_block_nums(self, sequence_length: int) -> int:
        need_block_nums = math.ceil(sequence_length / self.tokens_per_block)
        # plus one for zero case
        assert need_block_nums + 1 <= self.block_nums, "sequence_length is too long"
        return need_block_nums

    def _prepare_decode_attention_inputs(
        self, attention_inputs: PyAttentionInputs, sequence_length: int
    ) -> PyAttentionInputs:
        need_block_nums = self._check_block_nums(sequence_length)
        attention_inputs.is_prefill = False
        attention_inputs.padding_offset = torch.tensor(
            [0], dtype=torch.int32, device=self.device
        )
        attention_inputs.input_lengths = torch.tensor([1], dtype=torch.int32)
        attention_inputs.prefix_lengths = torch.tensor([], dtype=torch.int32)

        # sequence_lengths is index, so minus 1
        attention_inputs.sequence_lengths = torch.tensor(
            [sequence_length - 1], dtype=torch.int32
        ).pin_memory()
        attention_inputs.kv_cache_block_id_device = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]],
            dtype=torch.int32,
            device=self.device,
        )
        attention_inputs.kv_cache_block_id_host = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]], dtype=torch.int32
        )
        attention_inputs.dtype = get_typemeta(self.kv_cache.kv_cache_base)
        return attention_inputs

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 10,
        sampling_params: dict = None,
    ) -> list[int]:
        output_ids = []

        # prefill
        input_length = len(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=self.device)
        attention_inputs = self._prepare_prefill_attention_inputs(input_length)
        model_inputs = PyModelInputs(
            input_ids=input_ids,
            attention_inputs=attention_inputs,
        )
        model_outputs = self.model.forward(model_inputs)
        next_token_id = self._sample_next_token(model_outputs, sampling_params)
        next_token_id_cpu = next_token_id.cpu().item()
        # check if the next token is a stop token
        if next_token_id_cpu in self.stop_id_list:
            return output_ids
        output_ids.append(next_token_id_cpu)

        # decode
        gen_tokens = 1
        while gen_tokens < max_new_tokens:
            attention_inputs = self._prepare_decode_attention_inputs(
                attention_inputs, input_length + gen_tokens
            )
            model_inputs = PyModelInputs(
                input_ids=next_token_id,
                attention_inputs=attention_inputs,
            )
            model_outputs = self.model.forward(model_inputs)
            next_token_id = self._sample_next_token(model_outputs, sampling_params)
            next_token_id_cpu = next_token_id.cpu().item()
            gen_tokens += 1
            # check if the next token is a stop token
            if next_token_id_cpu in self.stop_id_list:
                break
            output_ids.append(next_token_id_cpu)

        return output_ids

    def _sample_next_token(
        self, model_outputs: PyModelOutputs, sampling_params: dict = None
    ) -> torch.Tensor:
        hidden_states = model_outputs.hidden_states[-1:, :]
        logits = torch.matmul(hidden_states, self.lm_head_weight.t())
        next_token_id = torch.argmax(logits, dim=-1)
        return next_token_id
