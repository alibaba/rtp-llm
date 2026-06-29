import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops.compute_ops import (
    CacheGroupType,
    KVCache,
    KVCacheRegionName,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
    init_exec_ctx,
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
        kernel_tokens_per_block: int = 0,
        stop_id_list: list[int] = [],
        hack_layer_num: int = 0,
        fp8_kv_cache: bool = False,
        init_tokenizer_and_lm_head: bool = True,
        **kwargs,
    ):
        # set configs instead of environment variables
        self._set_configs()
        self._set_standalone_parallelism()

        model_path, model_type = get_model_info_from_hf(model_path_or_name, revision)
        self.py_env_configs.model_args.model_type = model_type
        self.py_env_configs.model_args.ckpt_path = model_path
        self.py_env_configs.model_args.max_seq_len = max_total_tokens
        self.py_env_configs.kv_cache_config.seq_size_per_block = tokens_per_block
        self.py_env_configs.kv_cache_config.kernel_seq_size_per_block = (
            kernel_tokens_per_block
        )
        self.py_env_configs.kv_cache_config.fp8_kv_cache = fp8_kv_cache
        self.py_env_configs.profiling_debug_logging_config.hack_layer_num = (
            hack_layer_num
        )
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

        pc = engine_config.parallelism_config
        init_exec_ctx(
            device_id=pc.world_rank % pc.local_world_size,
            trace_memory=engine_config.profiling_debug_logging_config.trace_memory,
            enable_comm_overlap=engine_config.device_resource_config.enable_comm_overlap,
            mla_ops_type=int(model_config.mla_ops_type),
        )
        self.device = "cuda"

        # init kv cache and bind it to py model
        self.tokens_per_block = self.model_config.attn_config.tokens_per_block
        self.block_nums = math.ceil(max_total_tokens / self.tokens_per_block)
        # since block_id start from 1, so we should add 1 in the corner case
        self.block_nums += 1
        logging.info(f"total block nums: {self.block_nums}")
        self._init_kv_cache()

        self._initialize_py_model()

        if not init_tokenizer_and_lm_head:
            self.tokenizer = None
            self.stop_id_list = stop_id_list.copy() if stop_id_list else []
            self.lm_head_weight = None
            return

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

    def _set_standalone_parallelism(self):
        pc = self.py_env_configs.parallelism_config
        pc.world_size = 1
        pc.world_rank = 0
        pc.local_world_size = 1
        pc.local_rank = 0
        pc.tp_size = 1
        pc.tp_rank = 0
        pc.dp_size = 1
        pc.dp_rank = 0
        pc.ep_size = 1
        pc.ep_rank = 0
        pc.ffn_tp_size = 1
        pc.ffn_tp_rank = 0

    def _init_kv_cache(self):
        self.kv_cache = KVCache()
        self.layer_num = self.model_config.num_layers
        self.kv_head_num = self.model_config.attn_config.kv_head_num
        self.size_per_head = self.model_config.attn_config.size_per_head
        self.tokens_per_block = self.model_config.attn_config.tokens_per_block

        self.kv_cache.seq_size_per_block = self.tokens_per_block
        self.kv_cache.kernel_seq_size_per_block = (
            self.model_config.attn_config.kernel_tokens_per_block
        )
        self.kv_cache.num_kv_heads = self.kv_head_num
        self.kv_cache.head_dim = self.size_per_head
        self.kv_cache.use_mla = self.model_config.attn_config.use_mla
        self.kv_cache.kv_lora_rank = self.model_config.attn_config.kv_lora_rank
        self.kv_cache.rope_head_dim = self.model_config.attn_config.rope_head_dim
        if getattr(self.model_config.attn_config, "layer_compress_ratios", None):
            self._init_dsv4_kv_cache()
            return

        # Explicitly mark every layer as full-attention.
        self.kv_cache.layer_group_types = [
            CacheGroupType.FULL for _ in range(self.layer_num)
        ]

        per_layer_shape = [
            self.block_nums,
            2,
            self.kv_head_num,
            self.tokens_per_block,
            self.size_per_head,
        ]
        self.kv_cache.kv_cache_base_by_layer = [
            torch.zeros(per_layer_shape, dtype=self.compute_dtype, device=self.device)
            for _ in range(self.layer_num)
        ]

    def _init_dsv4_kv_cache(self):
        region_order = [
            KVCacheRegionName.CSA_KV,
            KVCacheRegionName.HCA_KV,
            KVCacheRegionName.INDEXER_KV,
            KVCacheRegionName.INDEXER_STATE,
            KVCacheRegionName.CSA_STATE,
            KVCacheRegionName.HCA_STATE,
            KVCacheRegionName.SWA_KV,
        ]
        region_count = int(KVCacheRegionName.SWA_KV) + 1
        ratios = list(self.model_config.attn_config.layer_compress_ratios)
        kernel_tokens = self.kv_cache.kernel_seq_size_per_block
        physical_tokens = self.kv_cache.seq_size_per_block
        kernel_blocks_per_kv_block = max(1, physical_tokens // kernel_tokens)
        head_dim = self.model_config.attn_config.size_per_head
        indexer_dim = self.model_config.attn_config.indexer_head_dim
        gen_num = int(getattr(self.model_config, "gen_num_per_cycle", 0) or 0)

        def align_up(value: int, align: int) -> int:
            return ((value + align - 1) // align) * align

        def align_dsv4_fp8_kv_bytes(value: int) -> int:
            return align_up(value, 576)

        def state_ring(compress_ratio: int, overlap: int) -> int:
            raw = (1 + overlap) * compress_ratio + max(gen_num, 0)
            return (raw + 1) & ~1

        csa_kv_stride = align_dsv4_fp8_kv_bytes((kernel_tokens // 4) * 584)
        hca_kv_stride = align_dsv4_fp8_kv_bytes((kernel_tokens // 128) * 584)
        indexer_kv_stride = (kernel_tokens // 4) * 132
        indexer_state_stride = state_ring(4, 1) * (4 * indexer_dim)
        csa_state_stride = state_ring(4, 1) * (4 * head_dim)
        hca_state_stride = state_ring(128, 0) * (2 * head_dim)
        swa_kv_stride = state_ring(128, 0) * 584
        region_specs = {
            int(KVCacheRegionName.CSA_KV): (torch.uint8, csa_kv_stride, True),
            int(KVCacheRegionName.HCA_KV): (torch.uint8, hca_kv_stride, True),
            int(KVCacheRegionName.INDEXER_KV): (
                torch.uint8,
                indexer_kv_stride,
                True,
            ),
            int(KVCacheRegionName.INDEXER_STATE): (
                torch.float32,
                indexer_state_stride,
                False,
            ),
            int(KVCacheRegionName.CSA_STATE): (
                torch.float32,
                csa_state_stride,
                False,
            ),
            int(KVCacheRegionName.HCA_STATE): (
                torch.float32,
                hca_state_stride,
                False,
            ),
            int(KVCacheRegionName.SWA_KV): (torch.uint8, swa_kv_stride, False),
        }

        self.kv_cache.group_region_names = region_order
        self.kv_cache.group_seq_size_per_block = [physical_tokens for _ in region_order]
        self.kv_cache.layer_region_to_group_id = [
            [-1 for _ in range(region_count)] for _ in range(self.layer_num)
        ]
        self.kv_cache.layer_group_types = [
            CacheGroupType.SWA for _ in range(self.layer_num)
        ]

        empty = torch.empty(0, device=self.device, dtype=torch.uint8)
        by_region = []
        for layer_id in range(self.layer_num):
            layer_regions = [empty for _ in range(region_count)]
            layer_ratio = int(ratios[layer_id]) if layer_id < len(ratios) else 0
            owned_regions = [KVCacheRegionName.SWA_KV]
            if layer_ratio == 4:
                owned_regions.extend(
                    [
                        KVCacheRegionName.CSA_KV,
                        KVCacheRegionName.INDEXER_KV,
                        KVCacheRegionName.INDEXER_STATE,
                        KVCacheRegionName.CSA_STATE,
                    ]
                )
            elif layer_ratio == 128:
                owned_regions.extend(
                    [KVCacheRegionName.HCA_KV, KVCacheRegionName.HCA_STATE]
                )

            for region in owned_regions:
                region_id = int(region)
                group_id = region_order.index(region)
                dtype, stride, is_full = region_specs[region_id]
                block_count = self.block_nums * (
                    kernel_blocks_per_kv_block if is_full else 1
                )
                layer_regions[region_id] = torch.zeros(
                    (block_count, stride), dtype=dtype, device=self.device
                )
                self.kv_cache.layer_region_to_group_id[layer_id][region_id] = group_id
            by_region.append(layer_regions)

        self.kv_cache.kv_cache_base_by_layer_region = by_region
        self.kv_cache.kv_cache_base_by_layer_region_flat = [
            by_region[layer][region]
            for layer in range(self.layer_num)
            for region in range(region_count)
        ]
        self.kv_cache.kv_cache_base_by_layer = [
            by_region[layer][int(KVCacheRegionName.SWA_KV)]
            for layer in range(self.layer_num)
        ]

    def _initialize_py_model(self):
        class _InitResources:
            pass

        init_resources = _InitResources()
        init_resources.kv_cache = self.kv_cache
        init_resources.is_speculative = False
        init_resources.is_decode_role = False
        init_resources.max_context_batch_size = 1
        if hasattr(self.model, "initialize"):
            self.model.initialize(init_resources)
        else:
            self.model.kv_cache = self.kv_cache

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
        attention_inputs.kv_cache_kernel_block_id_device = (
            attention_inputs.kv_cache_block_id_device
        )
        attention_inputs.kv_cache_block_id_host = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]], dtype=torch.int32
        )
        attention_inputs.kv_cache_kernel_block_id_host = (
            attention_inputs.kv_cache_block_id_host
        )
        attention_inputs.dtype = get_typemeta(self.kv_cache.kv_cache_base_by_layer[0])
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
        attention_inputs.kv_cache_kernel_block_id_device = (
            attention_inputs.kv_cache_block_id_device
        )
        attention_inputs.kv_cache_block_id_host = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]], dtype=torch.int32
        )
        attention_inputs.kv_cache_kernel_block_id_host = (
            attention_inputs.kv_cache_block_id_host
        )
        attention_inputs.dtype = get_typemeta(self.kv_cache.kv_cache_base_by_layer[0])
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
        logits = torch.matmul(
            hidden_states.to(self.lm_head_weight.dtype), self.lm_head_weight.t()
        ).to(torch.float32)
        next_token_id = torch.argmax(logits, dim=-1)
        return next_token_id
