import json
import logging
import math
import os
import typing

# make sure so init
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import torch

from rtp_llm.config.py_config_modules import (
    PyEnvConfigs,
    StaticConfig,
    get_env_bool,
    get_env_int,
    get_env_str,
)

from rtp_llm.config.task_type import TaskType
from rtp_llm.distribute.gang_info import GangInfo, get_gang_info
from rtp_llm.distribute.worker_info import (
    WORKER_INFO_PORT_NUM,
    ParallelInfo,
    g_master_info,
    g_parallel_info,
    g_worker_info,
)
from rtp_llm.ops import (
    ArpcConfig,
    BatchDecodeSchedulerConfig,
    CacheStoreConfig,
    ConcurrencyConfig,
    DeviceResourceConfig,
    EplbMode,
    FIFOSchedulerConfig,
    FMHAConfig,
    FfnDisAggregateConfig,
    GptInitParameter,
    HWKernelConfig,
    KVCacheConfig,
    MiscellaneousConfig,
    MlaOpsType,
    ModelConfig,
    ModelSpecificConfig,
    MMModelConfig,
    MoeConfig,
    ParallelismConfig,
    PDSepConfig,
    ProfilingDebugLoggingConfig,
    QuantAlgo,
    RoleType,
    RuntimeConfig,
    SchedulerConfig,
    SpecialTokens,
    SpeculativeExecutionConfig,
    EPLBConfig,
)
from rtp_llm.config.model_config import ModelConfig as PyModelConfig
from rtp_llm.utils.gemm_utils.cutlass_config import load_cutlass_gemm_config
from rtp_llm.utils.util import closest_power_of_2
from rtp_llm.utils.weight_type import WEIGHT_TYPE

def get_pad_size(size: int, align_size: int):
    return (align_size - (size % align_size)) % align_size


class VitParameters:
    # config includes origin vit config in ckpt/config.json
    config: Dict[str, Any] = {}
    special_token_ids: Dict[str, Any] = {}
    special_tokens: Dict[str, Any] = {}
    vit_weights: Any = None


class TemplateType(Enum):
    chat = "chat"
    vqa = "vqa"
    base = "image"


class ConfigMode(Enum):
    SimpleMode = 1
    ComplexMode = 2


class GptInitModelParameters:
    __slots__ = {
        "lora_infos",
        "multi_task_prompt",
        "template_type",
        "vit_run_batch",
        "phy2log",
        "py_env_configs",
        "th_nccl_port",
        "py_model_config",
        # C++ config objects (note: model_config is handled via py_model_config)
        "mm_model_config",
        "parallelism_config",
        "runtime_config",
        "eplb_config",
        "pd_sep_config",
        "concurrency_config",
        "fmha_config",
        "kv_cache_config",
        "profiling_debug_logging_config",
        "hw_kernel_config",
        "device_resource_config",
        "moe_config",
        "model_specific_config",
        "sp_config",
        "cache_store_config",
        "misc_config",
        "arpc_config",
        "ffn_disaggregate_config",
    }

    def __init__(self):
        self.py_model_config = PyModelConfig()
        
        self.template_type = TemplateType.chat
        self.phy2log: List[List[int]] = []

        # For cpp, we use `self`, `py_env_configs` for python.
        # There are some common envs in cpp and python, so they will
        # share some configs together.
        self.th_nccl_port = g_master_info.th_nccl_port  # Not in C++ ParallelismConfig, keep as self for compatibility
        
        # Create all C++ config objects directly
        # Note: model_config is handled via py_model_config (PyModelConfig inherits from CppModelConfig)
        self.mm_model_config = MMModelConfig()
        self.parallelism_config = ParallelismConfig(
            tp_size=g_parallel_info.tp_size,
            ep_size=g_parallel_info.ep_size,
            dp_size=g_parallel_info.dp_size,
            world_size=g_parallel_info.world_size,
            world_rank=g_parallel_info.world_rank,
            local_world_size=g_parallel_info.local_world_size,
            pp_size=g_parallel_info.pp_size,
            ffn_sp_size=g_parallel_info.ffn_sp_size,
        )
        # Set port and IP related fields
        self.parallelism_config.nccl_ip = g_master_info.ip
        self.parallelism_config.tp_nccl_port = g_master_info.tp_nccl_port
        self.parallelism_config.dp_tp_nccl_port = g_master_info.dp_tp_nccl_port
        self.parallelism_config.ffn_tp_nccl_port = g_master_info.ffn_tp_nccl_port
        self.parallelism_config.model_rpc_port = g_worker_info.rpc_server_port
        self.parallelism_config.http_port = g_worker_info.http_port
        
        self.runtime_config = RuntimeConfig()
        self.eplb_config = EPLBConfig()
        self.pd_sep_config = PDSepConfig()
        self.concurrency_config = ConcurrencyConfig()
        self.fmha_config = FMHAConfig()
        self.kv_cache_config = KVCacheConfig()
        self.profiling_debug_logging_config = ProfilingDebugLoggingConfig()
        self.hw_kernel_config = HWKernelConfig()
        self.device_resource_config = DeviceResourceConfig()
        self.moe_config = MoeConfig()
        self.model_specific_config = ModelSpecificConfig()
        self.sp_config = SpeculativeExecutionConfig()
        self.cache_store_config = CacheStoreConfig()
        self.misc_config = MiscellaneousConfig()
        self.arpc_config = ArpcConfig()
        self.ffn_disaggregate_config = FfnDisAggregateConfig()

        # Call update_from_env_for_test() on all config objects
        self.parallelism_config.update_from_env_for_test()
        self.runtime_config.update_from_env_for_test()
        self.eplb_config.update_from_env_for_test()
        self.concurrency_config.update_from_env_for_test()
        self.fmha_config.update_from_env_for_test()
        self.kv_cache_config.update_from_env_for_test()
        self.profiling_debug_logging_config.update_from_env_for_test()
        self.hw_kernel_config.update_from_env_for_test()
        self.device_resource_config.update_from_env_for_test()
        self.moe_config.update_from_env_for_test()
        self.model_specific_config.update_from_env_for_test()
        self.sp_config.update_from_env_for_test()
        self.cache_store_config.update_from_env_for_test()
        self.misc_config.update_from_env_for_test()

        # Special handling for scheduler config - need to set use_gather_batch_scheduler
        # C++ update_from_env_for_test() doesn't set this, so set it manually
        self.runtime_config.use_gather_batch_scheduler = get_env_bool("USE_GATHER_BATCH_SCHEDULER", False)
        if (
            self.runtime_config.use_gather_batch_scheduler
            and self.runtime_config.use_batch_decode_scheduler
        ):
            raise ValueError(
                "use_gather_batch_scheduler and use_batch_decode_scheduler cannot be true at the same time"
            )

        # BatchDecodeSchedulerConfig - C++ doesn't set batch_decode_scheduler_warmup_type
        self.runtime_config.batch_decode_scheduler_warmup_type = get_env_int(
            "BATCH_DECODE_SCHEDULER_WARMUP_TYPE", 0
        )
    
        self.py_env_configs = PyEnvConfigs()
        self.py_env_configs.update_from_env()
        self.py_env_configs.parallelism_distributed_config = self.parallelism_config
        StaticConfig.parallelism_distributed_config = self.parallelism_config

        self.runtime_config.vit_separation = self.py_env_configs.vit_config.vit_separation
        logging.info(f"vit_separation: {self.runtime_config.vit_separation}")
        self.pd_sep_config.role_type = (
            RoleType.VIT
            if self.runtime_config.vit_separation == 1
            else self.py_env_configs.role_config.role_type
        )
    
    @property
    def gpt_init_params(self) -> GptInitParameter:
        """
        Build GptInitParameter from individual config objects for C++ compatibility.
        This is used by C++ code that still expects GptInitParameter.
        Note: py_model_config (PyModelConfig) inherits from CppModelConfig, so it can be used directly.
        """
        params = GptInitParameter()
        # py_model_config inherits from CppModelConfig, so it can be used directly as model_config_
        params.model_config_ = self.py_model_config
        params.mm_model_config_ = self.mm_model_config
        params.parallelism_config = self.parallelism_config
        params.runtime_config = self.runtime_config
        params.eplb_config = self.eplb_config
        params.pd_sep_config = self.pd_sep_config
        params.concurrency_config = self.concurrency_config
        params.fmha_config = self.fmha_config
        params.kv_cache_config = self.kv_cache_config
        params.profiling_debug_logging_config = self.profiling_debug_logging_config
        params.hw_kernel_config = self.hw_kernel_config
        params.device_resource_config = self.device_resource_config
        params.moe_config = self.moe_config
        params.model_specific_config = self.model_specific_config
        params.sp_config = self.sp_config
        params.cache_store_config = self.cache_store_config
        params.misc_config = self.misc_config
        params.arpc_config = self.arpc_config
        params.ffn_disaggregate_config = self.ffn_disaggregate_config
        return params

    def update_worker_addrs(self):
        worker_addrs = []
        worker_grpc_addrs = []
        for member in get_gang_info().members:
            logging.info(
                f"member world rank: {member.world_rank}, member local rank: {member.local_rank}, local rank: {self.parallelism_config.local_rank}, "
                f"tp_size: {self.parallelism_config.tp_size}, dp_size: {self.parallelism_config.dp_size}, dp_rank: {self.parallelism_config.dp_rank}, use_all_gather: {self.parallelism_config.use_all_gather}"
            )
            if int((member.world_rank / self.parallelism_config.tp_size) % self.parallelism_config.dp_size) == self.parallelism_config.dp_rank:
                worker_addrs.append(
                    f"{member.ip}:{member.cache_store_listen_port}:{member.cache_store_rdma_listen_port}"
                )
                worker_grpc_addrs.append(f"{member.ip}:{member.rpc_server_port}")
                logging.info(
                    f"append member for pd sep "
                    f"{member.ip}:{member.rpc_server_port}, {member.cache_store_listen_port}, "
                    f"{member.cache_store_rdma_listen_port} to local rank {self.parallelism_config.local_rank}, world rank {member.world_rank}"
                )
        self.runtime_config.worker_grpc_addrs = worker_grpc_addrs
        self.runtime_config.worker_addrs = worker_addrs

    def update_gpt_init_params_from_env(
        self, parallel_info: ParallelInfo = g_parallel_info
    ):



    def update_task_prompt_tokens_id(self, tokenizer):
        if self.multi_task_prompt:
            for info in self.multi_task_prompt:
                task_id: str = str(info["task_id"])
                prompt: str = info["prompt"]
                tokens_id = tokenizer.encode(prompt)
                self.kv_cache_config.insertMultiTaskPromptTokens(task_id, tokens_id)

    def update_tokenizer_special_tokens(self, tokenizer):
        self.py_model_config.special_tokens.stop_words_id_list += tokenizer.stop_words_id_list
        self.py_model_config.special_tokens.stop_words_str_list += tokenizer.stop_words_str_list
        self.py_model_config.special_tokens.eos_token_id = tokenizer.eos_token_id

    def update_task_prompt_config(self):
        prompt_file_path = self.kv_cache_config.multi_task_prompt
        if prompt_file_path == "":
            self.multi_task_prompt = None
        else:
            with open(prompt_file_path, "r") as reader:
                multi_task_prompt = json.loads(reader.read(), strict=False)
                self.multi_task_prompt = multi_task_prompt
                return

        prompt_str = self.kv_cache_config.multi_task_prompt_str
        if prompt_str == "":
            self.multi_task_prompt = None
        else:
            self.multi_task_prompt = json.loads(prompt_str, strict=False)
            return

    def update_common(
        self,
        ckpt_path: str,
        lora_infos: Optional[Dict[str, str]],
        ptuning_path: Optional[str],
        tokenizer_path: str,
        quantization: str,
        data_type: str,
        kv_cache_type: str,
        max_seq_len: int,
        seq_size_per_block: int,
        gen_num_per_circle: int,
        parallel_info: ParallelInfo = g_parallel_info,
        config_mode: ConfigMode = ConfigMode.ComplexMode,
        gang_info: Optional[GangInfo] = None,
    ):

        # Initialize precision configuration (quant_algo, data_type, kv_cache_data_type)
        # This method sets all precision-related fields directly on py_model_config
        self.py_model_config.init_precision_config(
            ckpt_path, quantization, data_type, kv_cache_type, self.py_model_config.config_dtype
        )
        self.parallelism_config.tp_size = parallel_info.tp_size
        self.parallelism_config.tp_rank = parallel_info.tp_rank
        self.parallelism_config.ep_size = parallel_info.ep_size
        self.parallelism_config.ep_rank = parallel_info.ep_rank
        self.parallelism_config.dp_size = parallel_info.dp_size
        self.parallelism_config.dp_rank = parallel_info.dp_rank
        self.parallelism_config.ffn_tp_rank = parallel_info.ffn_tp_rank
        self.parallelism_config.ffn_tp_size = parallel_info.ffn_tp_size
        self.parallelism_config.enable_sp = parallel_info.ffn_sp_size > 1
        self.parallelism_config.local_rank = parallel_info.local_rank
        self.parallelism_config.use_all_gather = (
            bool(int(os.environ.get("USE_ALL_GATHER", 0)))
            and self.moe_config.use_deepep_low_latency == False
        )
        logging.info(f"use_all_gather: {self.parallelism_config.use_all_gather}")

        self.eplb_config.eplb_update_time = self.py_env_configs.py_eplb_config.eplb_update_time
        self.eplb_config.eplb_mode = EplbMode.__members__[
            self.py_env_configs.py_eplb_config.eplb_mode
        ]
        self.eplb_config.enable_eplb = self.eplb_config.eplb_mode != EplbMode.NONE

        self.eplb_config.phy_exp_num = (
            self.py_env_configs.py_eplb_config.redundant_expert + self.py_model_config.expert_num_
        )
        logging.info(f"phy_exp_num: {self.eplb_config.phy_exp_num}")

        if gang_info is not None:
            self.num_nodes = gang_info.num_nodes
        else:
            try:
                self.num_nodes = get_gang_info().num_nodes
            except:
                self.num_nodes = 1
        self.lora_infos = lora_infos
    
        self.py_model_config.ckpt_path = ckpt_path
        self.py_model_config.ckpt_path_ = ckpt_path

        self.py_model_config.tokenizer_path_ = tokenizer_path

        self.runtime_config.gen_num_per_circle = gen_num_per_circle
        self.py_model_config.ptuning_path = ptuning_path
        if max_seq_len != 0:
            self.py_model_config.max_seq_len_ = max_seq_len
        if self.py_model_config.max_seq_len_ < 1:
            # frontend not load ckpt config max_seq_len, use default 8192 or env
            self.py_model_config.max_seq_len_ = 8192
        logging.info(f"max_seq_len: {self.py_model_config.max_seq_len_}")

        # Update task_type and use_kvcache in py_model_config
        self.py_model_config.update_task_type_use_kvcache()
        # py_model_config is used directly, no need to sync

        if StaticConfig.ffn_disaggregate_config.enable_ffn_disaggregate:
            # 暂时先限制tp=1, 更多支持在python版本实现
            assert (
                g_parallel_info.tp_size == 1 and g_parallel_info.world_size > 1
            ), "enable_ffn_disaggregate must be used in dp = 1 world_size > 1"
            attention_dp_size = g_parallel_info.world_size - 1
            attention_tp_size = 1
            ffn_tp_size = 1
            assert (
                attention_tp_size == ffn_tp_size
            ), "attention_tp_size must be equal to ffn_tp_size"
            self.ffn_disaggregate_config.enable_ffn_disaggregate = True
            self.ffn_disaggregate_config.attention_tp_size = (
                attention_tp_size
            )
            self.ffn_disaggregate_config.attention_dp_size = (
                attention_dp_size
            )
            self.ffn_disaggregate_config.ffn_tp_size = ffn_tp_size
            # TODO: remove it, ffn dp is stupid
            self.ffn_disaggregate_config.ffn_dp_size = 1
            self.ffn_disaggregate_config.is_ffn_rank = (
                g_parallel_info.world_rank >= attention_tp_size * attention_dp_size
            )

        logging.info(f"config_mode = {config_mode}")
        if config_mode == ConfigMode.SimpleMode:
            return

        self.update_worker_addrs()
        self.py_model_config.update_inter_padding_size(
            self.parallelism_config.tp_size, 
            self.parallelism_config.ep_size, 
            self.parallelism_config.dp_size, 
            self.hw_kernel_config
        )
        self.update_task_prompt_config()

        load_cutlass_gemm_config(self.py_model_config.quant_algo_)

        hack_layer_num = self.profiling_debug_logging_config.hack_layer_num
        if hack_layer_num:
            logging.info(f"hack layernum to {hack_layer_num}")
            self.py_model_config.num_layers_ = hack_layer_num

        self.py_model_config.seq_size_per_block_ = closest_power_of_2(
            int(max(seq_size_per_block, self.py_model_config.max_seq_len_ // 128))
        )  # must be 2^n
        if self.py_env_configs.py_kv_cache_config.seq_size_per_block != -1:
            self.py_model_config.seq_size_per_block_ = int(
                self.py_env_configs.py_kv_cache_config.seq_size_per_block
            )
        # py_model_config.seq_size_per_block is a Python property that mirrors seq_size_per_block_
        self.py_model_config.seq_size_per_block = self.py_model_config.seq_size_per_block_

        logging.info(f"seq_size_per_block: {self.py_model_config.seq_size_per_block_}")
        self.runtime_config.max_generate_batch_size = (
            self.py_env_configs.concurrency_config.concurrency_limit
        )

        logging.info(f"max_generate_batch_size: {self.runtime_config.max_generate_batch_size}")
        # max_context_batch_size, enable_partial_fallback, enable_fast_gen are already set in runtime_config from env
        # No need to copy from fifo_scheduler_config as they are now merged into runtime_config
        logging.info(f"max_context_batch_size: {self.runtime_config.max_context_batch_size}")
        self.runtime_config.reserve_runtime_mem_mb = (
            self.py_env_configs.py_device_resource_config.reserver_runtime_mem_mb
        )
        logging.info(f"reserve_runtime_mem_mb: {self.runtime_config.reserve_runtime_mem_mb}")
        self.runtime_config.kv_cache_mem_mb = self.py_env_configs.py_kv_cache_config.kv_cache_mem_mb
        logging.info(f"kv_cache_mem_mb: {self.runtime_config.kv_cache_mem_mb}")
        self.runtime_config.block_nums = self.py_env_configs.py_kv_cache_config.test_block_num
        logging.info(f"block_nums: {self.runtime_config.block_nums}")
        logging.info(f"enable_partial_fallback: {self.runtime_config.enable_partial_fallback}")
        logging.info(f"enable_fast_gen: {self.runtime_config.enable_fast_gen}")
        self.runtime_config.warm_up = bool(self.py_env_configs.engine_config.warm_up)
        logging.info(f"warm_up: {self.runtime_config.warm_up}")
        self.runtime_config.warm_up_with_loss = bool(
            self.py_env_configs.engine_config.warm_up_with_loss
        )
        logging.info(f"warm_up_with_loss: {self.runtime_config.warm_up_with_loss}")

        # fast_gen_max_context_len uses fast_gen_context_budget from runtime_config
        if self.runtime_config.fast_gen_context_budget == -1:
            self.runtime_config.fast_gen_max_context_len = 1024
        else:
            self.runtime_config.fast_gen_max_context_len = self.runtime_config.fast_gen_context_budget
        logging.info(f"fast_gen_max_context_len: {self.runtime_config.fast_gen_max_context_len}")

        self.runtime_config.max_batch_tokens_size = int(
            os.environ.get(
                "MAX_BATCH_TOKENS_SIZE", self.runtime_config.max_context_batch_size * self.py_model_config.max_seq_len_
            )
        )
        logging.info(f"max_batch_tokens_size: {self.runtime_config.max_batch_tokens_size}")


        # Copy the entire config
        self.pd_sep_config = self.py_env_configs.pd_separation_config
        self.pd_sep_config.cache_store_listen_port = g_worker_info.cache_store_listen_port
        self.pd_sep_config.cache_store_connect_port = g_worker_info.cache_store_connect_port
        self.pd_sep_config.cache_store_rdma_listen_port = g_worker_info.cache_store_rdma_listen_port
        self.pd_sep_config.cache_store_rdma_connect_port = g_worker_info.cache_store_rdma_connect_port
        self.pd_sep_config.remote_rpc_server_port = g_worker_info.remote_rpc_server_port
        self.pd_sep_config.worker_port_offset = WORKER_INFO_PORT_NUM
        
        # Override with values from other sources
        if self.pd_sep_config.role_type in [RoleType.PREFILL, RoleType.DECODE]:
            self.pd_sep_config.cache_store_rdma_mode = (
                self.cache_store_config.cache_store_rdma_mode
            )

        self.runtime_config.scheduler_reserve_resource_ratio = int(
            os.environ.get("SCHEDUlER_RESERVE_RESOURCE_RATIO", 5)
        )
        logging.info(
            f"scheduler_reserve_resource_ratio: {self.runtime_config.scheduler_reserve_resource_ratio}"
        )
        self.runtime_config.reuse_cache = self.py_env_configs.py_kv_cache_config.reuse_cache
        logging.info(f"reuse_cache: {self.runtime_config.reuse_cache}")
        self.runtime_config.pre_allocate_op_mem = bool(int(os.environ.get("PRE_ALLOCATE_OP_MEM", 1)))
        logging.info(f"pre_allocate_op_mem: {self.runtime_config.pre_allocate_op_mem}")

        # use environment variables to update stop_words_str and stop_words_id
        self.py_model_config.update_stop_words_from_env(self.py_env_configs.generate_env_config)

        model_override_args = json.loads(
            StaticConfig.model_config.json_model_override_args
        )
        if model_override_args:
            # Apply rope_scaling override via py_model_config
            # py_model_config is the source of truth, no need to pass target_model_config
            self.py_model_config.apply_rope_scaling_override(
                model_override_args, 
                self.py_model_config
            )

