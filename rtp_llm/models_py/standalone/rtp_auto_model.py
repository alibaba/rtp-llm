import os
import sys
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

rtp_opensouce_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(rtp_opensouce_path))

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
    init_device,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.distribute.worker_info import g_parallel_info


class AutoModel:
    def __init__(
        self,
        model_path_or_name: str,
        **kwargs,
    ):
        # get some initial parameters
        revision: Optional[str] = kwargs.get("revision", None)
        max_total_tokens: int = kwargs.get("max_total_tokens", 4096)
        tokens_per_block: int = kwargs.get("tokens_per_block", 64)
        stop_id_list: list[int] = kwargs.get("stop_id_list", [])

        # set configs instead of environment variables
        self._set_configs()
        
        # Handle HuggingFace model path - get model_type if needed
        from rtp_llm.model_factory_register import ModelDict
        from rtp_llm.tools.api.hf_model_helper import HfStyleModelInfo
        
        # Check if it's a HuggingFace model path
        if "/" in model_path_or_name and not os.path.exists(model_path_or_name):
            # Try to get model_type from HuggingFace repo name first (fast, no download)
            model_type = ModelDict.get_ft_model_type_by_hf_repo(model_path_or_name)
            if model_type:
                self.py_env_configs.model_args.model_type = model_type
                logging.info(f"Inferred model_type '{model_type}' from HuggingFace repo: {model_path_or_name}")
            else:
                # Try to get model_type from config.json only (faster than downloading full model)
                try:
                    hf_info = HfStyleModelInfo(model_path_or_name, revision)
                    if hf_info.ft_model_type:
                        self.py_env_configs.model_args.model_type = hf_info.ft_model_type
                        logging.info(f"Inferred model_type '{hf_info.ft_model_type}' from HuggingFace config")
                    # Get local path (will download if needed, but model_type is already set)
                    from rtp_llm.tools.api.hf_model_helper import get_model_info_from_hf
                    local_path, _ = get_model_info_from_hf(model_path_or_name, revision)
                    model_path_or_name = local_path
                except Exception as e:
                    logging.warning(f"Failed to get model_type from HuggingFace: {e}")
                    if not self.py_env_configs.model_args.model_type:
                        raise ValueError(f"Could not determine model_type for {model_path_or_name}. Please provide model_type explicitly.")
        
        # Set checkpoint path in model_args
        self.py_env_configs.model_args.ckpt_path = model_path_or_name
        if not self.py_env_configs.model_args.tokenizer_path:
            self.py_env_configs.model_args.tokenizer_path = model_path_or_name

        # Create EngineConfig from py_env_configs (gang_info can be None for standalone)
        engine_config = EngineConfig.create(self.py_env_configs, gang_info=None)
        
        # Create model configs
        model_config, _ = ModelFactory.create_model_configs(
            engine_config=engine_config,
            model_args=self.py_env_configs.model_args,
            lora_config=self.py_env_configs.lora_config,
            generate_env_config=self.py_env_configs.generate_env_config,
            embedding_config=self.py_env_configs.embedding_config,
            quantization_config=self.py_env_configs.quantization_config,
            render_config=self.py_env_configs.render_config,
        )
        
        # Create model using ModelFactory
        self.gpt_model = ModelFactory._create_model(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=None,
            merge_lora=False,
        )
        
        # Load the model
        self.gpt_model.load(parallel_info=g_parallel_info)
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
        )
        self.device = 'cuda:0'

        # init kv cache and bind it to py model
        self.block_nums = (
            max_total_tokens + tokens_per_block - 1
        ) // tokens_per_block + 1
        self.tokens_per_block = tokens_per_block
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
            self.py_env_configs.device_resource_config.device_reserve_memory_bytes = 2 * 1024 * 1024 * 1024

    def _init_kv_cache(self):
        self.kv_cache = KVCache()
        self.layer_num = self.model_config.num_layers
        # seq_size_per_block is stored in attn_config.rope_config or model_config
        # For now, use tokens_per_block directly
        self.kv_head_num = self.model_config.attn_config.kv_head_num
        self.size_per_head = self.model_config.attn_config.size_per_head
        kv_shape = [
            self.layer_num * 2,
            self.block_nums,
            self.kv_head_num,
            self.tokens_per_block,
            self.size_per_head,
        ]
        kv_cache_total = torch.zeros(kv_shape, dtype=self.compute_dtype, device=self.device)
        k_cache_base = kv_cache_total[: self.layer_num, :, :, :, :]
        v_cache_base = kv_cache_total[self.layer_num :, :, :, :, :]
        self.kv_cache.k_cache_base = k_cache_base
        self.kv_cache.v_cache_base = v_cache_base


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
        attention_inputs.dtype = get_typemeta(self.kv_cache.k_cache_base)
        attention_inputs.kv_block_offset = self.layer_num * self.block_nums
        attention_inputs.is_prefill = True
        return attention_inputs

    def _check_block_nums(self, sequence_length: int) -> int:
        need_block_nums = (
            sequence_length + self.tokens_per_block - 1
        ) // self.tokens_per_block + 1
        assert need_block_nums <= self.block_nums, "sequence_length is too long"
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
        attention_inputs.sequence_lengths = torch.tensor(
            [sequence_length], dtype=torch.int32
        ).pin_memory()
        attention_inputs.kv_cache_block_id_device = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]],
            dtype=torch.int32,
            device=self.device,
        )
        attention_inputs.kv_cache_block_id_host = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]], dtype=torch.int32
        )
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
