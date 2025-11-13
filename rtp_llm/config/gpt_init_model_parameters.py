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
    get_env_optional_bool,
    get_env_str,
)
from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    Fp8PerChannelCompressedQuantConfig,
    Fp8PerTensorCompressedQuantConfig,
    QuantizationConfig,
    init_quant_config,
)
from rtp_llm.config.task_type import TaskType, check_task_type
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
    GptInitParameter,
    HWKernelConfig,
    KVCacheConfig,
    MiscellaneousConfig,
    MlaOpsType,
    ModelSpecificConfig,
    MoeConfig,
    ParallelismDistributedConfig,
    ProfilingDebugLoggingConfig,
    QuantAlgo,
    RoleType,
    SamplerConfig,
    SchedulerConfig,
    ServiceDiscoveryConfig,
    SpecialTokens,
    SpeculativeExecutionConfig,
)
from rtp_llm.utils.gemm_utils.cutlass_config import load_cutlass_gemm_config
from rtp_llm.utils.util import closest_power_of_2
from rtp_llm.utils.weight_type import WEIGHT_TYPE

updated_params: Set[str] = set()


def get_pad_size(size: int, align_size: int):
    return (align_size - (size % align_size)) % align_size


class DataClassBase:
    @classmethod
    def from_dict(cls, kvs: Dict[str, Any]):
        n_kvs = {k: v for k, v in kvs.items() if k in {f.name for f in fields(cls)}}

        # 兼容老的sparse config使用的key 没有加layer
        for k, v in kvs.items():
            if k in ["head_num", "inter_size"] and isinstance(v, list):
                n_kvs.update({"layer_" + k: v})

        data_class = cls(**n_kvs)
        return data_class


mc_sim_7b_63 = [
    [0],
    [0, 0],
    [1],
    [0, 1],
    [2],
    [0, 0, 0],
    [1, 0],
    [0, 2],
    [3],
    [0, 3],
    [4],
    [0, 4],
    [2, 0],
    [0, 5],
    [0, 0, 1],
    [5],
    [0, 6],
    [6],
    [0, 7],
    [0, 1, 0],
    [1, 1],
    [7],
    [0, 8],
    [0, 0, 2],
    [3, 0],
    [0, 9],
    [8],
    [9],
    [1, 0, 0],
    [0, 2, 0],
    [1, 2],
    [0, 0, 3],
    [4, 0],
    [2, 1],
    [0, 0, 4],
    [0, 0, 5],
    [0, 0, 0, 0],
    [0, 1, 1],
    [0, 0, 6],
    [0, 3, 0],
    [5, 0],
    [1, 3],
    [0, 0, 7],
    [0, 0, 8],
    [0, 0, 9],
    [6, 0],
    [0, 4, 0],
    [1, 4],
    [7, 0],
    [0, 1, 2],
    [2, 0, 0],
    [3, 1],
    [2, 2],
    [8, 0],
    [0, 5, 0],
    [1, 5],
    [1, 0, 1],
    [0, 2, 1],
    [9, 0],
    [0, 6, 0],
    [0, 0, 0, 1],
    [1, 6],
    [0, 7, 0],
]


@dataclass
class SparseConfig(DataClassBase):
    layer_num: int = 0
    layer_head_num: List[int] = field(default_factory=lambda: [])
    layer_inter_size: List[int] = field(default_factory=lambda: [])

    def check(self) -> bool:
        if self.layer_num == 0:
            logging.info("sparse config layer_num must not be empty")
            return False
        if len(self.layer_head_num) != self.layer_num:
            logging.info(
                f"sparse config layer_num and head_num must match, layer_num: {self.layer_num}, head_num: {self.layer_head_num}"
            )
            return False
        if len(self.layer_inter_size) != self.layer_num:
            logging.info(
                f"sparse config layer_num and inter_size must match, layer_num: {self.layer_num}, inter_size: {self.layer_inter_size}"
            )
            return False
        return True


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
        "gpt_init_params",
        "_model_related_types",
        "has_lm_head_bias",
        "src_quantization_bit",
        "ptuning_path",
        "tp_split_emb_and_lm_head",
        "mm_related_params",
        "lora_infos",
        "multi_task_prompt",
        "normalize_lm_head_weight",
        "ref_module",
        "ref_dict",
        "tie_word_embeddings",
        "task_type",
        "add_special_tokens",
        "template_type",
        "build_position_ids",
        "vit_run_batch",
        "phy2log",
        "is_mtp",
        "num_nodes",
        "quant_config",
        "py_env_configs",
        "config_dtype",
        "th_nccl_port",
        "model_type",
    }

    # copy from rtp_llm/ops/libth_transformer.pyi for python intelligence
    activation_type: str
    add_bias_linear: bool
    block_nums: int
    cache_store_connect_port: int
    cache_store_listen_port: int
    cache_store_rdma_connect_port: int
    cache_store_rdma_listen_port: int
    cache_store_rdma_mode: bool
    ckpt_path: str
    cross_attn_input_len: int
    data_type: str
    decode_entrance: bool
    config_dtype: str
    decode_polling_kv_cache_step_ms: int
    decode_retry_timeout_ms: int
    decode_retry_times: int
    decode_retry_interval: int
    deepseek_mscale_all_dim: float
    deepseek_rope_mscale: float
    dp_rank: int
    dp_size: int
    dp_tp_nccl_port: int
    th_nccl_port: int
    embedding_size: int
    enable_eplb: bool
    enable_fast_gen: bool
    enable_partial_fallback: bool
    enable_sp: bool
    enable_speculative_decoding: bool
    ep_rank: int
    ep_size: int
    eplb_mode: EplbMode
    eplb_update_time: int
    expert_num: int
    fast_gen_max_context_len: int
    ffn_tp_nccl_port: int
    ffn_tp_rank: int
    ffn_tp_size: int
    gen_num_per_circle: int
    has_lm_head: bool
    has_moe_norm: bool
    has_positional_encoding: bool
    has_post_decoder_layernorm: bool
    has_pre_decoder_layernorm: bool
    head_num: int
    head_num_kv: int
    hidden_size: int
    http_port: int
    include_sep_tokens: bool
    input_embedding_scalar: float
    input_vocab_size: int
    inter_padding_size: int
    inter_size: int
    is_causal: bool
    is_multimodal: bool
    is_sparse_head: bool
    kv_cache_data_type: str
    kv_cache_mem_mb: int
    kv_lora_rank: int
    layer_head_num: list[int]
    layer_head_num_kv: list[int]
    layer_inter_padding_size: list[int]
    layer_inter_size: list[int]
    layer_num: int
    layernorm_eps: float
    layernorm_type: str
    load_cache_timeout_ms: int
    local_rank: int
    logit_scale: float
    max_context_batch_size: int
    max_generate_batch_size: int
    max_rpc_timeout_ms: int
    max_seq_len: int
    mla_ops_type: MlaOpsType
    mm_position_ids_style: int
    mm_sep_tokens: list[list[int]]
    model_name: str
    model_rpc_port: int
    moe_inter_padding_size: int
    moe_k: int
    moe_layer_index: list[int]
    moe_n_group: int
    moe_normalize_expert_scale: bool
    moe_style: int
    moe_topk_group: int
    routed_scaling_factor: float
    mrope_section: list[int]
    nccl_ip: str
    nope_head_dim: int
    norm_type: str
    num_layers: int
    num_valid_layer: int
    org_embedding_max_pos: int
    phy_exp_num: int
    position_id_len_factor: int
    position_ids_style: int
    pre_allocate_op_mem: bool
    pre_seq_len: int
    prefill_max_wait_timeout_ms: int
    prefill_retry_timeout_ms: int
    prefill_retry_times: int
    prefix_projection: bool
    py_eplb: typing.Any
    q_lora_rank: int
    q_scaling: float
    qk_norm: bool
    quant_algo: QuantAlgo
    rdma_connect_retry_times: int
    remote_rpc_server_port: int
    reserve_runtime_mem_mb: int
    residual_scalar: float
    reuse_cache: bool
    reverse_e_h_norm: bool
    rope_head_dim: int
    rotary_embedding_base: float
    rotary_embedding_dim: int
    rotary_embedding_mscale: float
    rotary_embedding_offset: int
    rotary_embedding_scale: float
    rotary_embedding_style: int
    rotary_factor1: float
    rotary_factor2: float
    partial_rotary_factor: float
    rotary_embedding_extrapolation_factor: float
    scoring_func: int
    seq_size_per_block: int
    size_per_head: int
    softmax_extra_scale: float
    special_tokens: SpecialTokens
    tokenizer_path: str
    tp_nccl_port: int
    tp_rank: int
    tp_size: int
    type_vocab_size: int
    use_all_gather: bool
    use_attention_linear_bias: bool

    use_cross_attn: bool
    use_fp32_to_compute_logit: bool
    use_kvcache: bool
    use_logn_attn: bool
    use_mla: bool
    use_norm_attn_out_residual: bool
    use_norm_input_residual: bool
    using_hf_sampling: bool
    v_head_dim: int
    vit_separation: int
    vocab_size: int
    warm_up: bool
    warm_up_with_loss: bool
    worker_addrs: list[str]
    worker_grpc_addrs: list[str]
    worker_port_offset: int
    world_size: int
    role_type: RoleType
    quant_config: Optional[QuantizationConfig]

    batch_decode_scheduler_config: BatchDecodeSchedulerConfig
    cache_store_config: CacheStoreConfig
    concurrency_config: ConcurrencyConfig
    device_resource_config: DeviceResourceConfig
    fifo_scheduler_config: FIFOSchedulerConfig
    fmha_config: FMHAConfig
    hw_kernel_config: HWKernelConfig
    kv_cache_config: KVCacheConfig
    misc_config: MiscellaneousConfig
    arpc_config: ArpcConfig
    model_specific_config: ModelSpecificConfig
    moe_config: MoeConfig
    parallelism_distributed_config: ParallelismDistributedConfig
    profiling_debug_logging_config: ProfilingDebugLoggingConfig
    sampler_config: SamplerConfig
    scheduler_config: SchedulerConfig
    service_discovery_config: ServiceDiscoveryConfig
    speculative_decoding_config: SpeculativeExecutionConfig
    py_env_configs: PyEnvConfigs

    def __init__(
        self,
        head_num: int,
        size_per_head: int,
        layer_num: int,
        max_seq_len: int,
        vocab_size: int,
        **kwargs: Any,
    ):
        hidden_size = head_num * size_per_head
        self.gpt_init_params = GptInitParameter(
            head_num, size_per_head, layer_num, max_seq_len, vocab_size, hidden_size
        )
        self._model_related_types: Dict[str, str] = {
            "layernorm_type": "setLayerNormType",
            "norm_type": "setNormType",
            "activation_type": "setActivationType",
            "data_type": "setDataType",
            "kv_cache_data_type": "setKvCacheDataType",
        }
        self.has_lm_head_bias = False
        self.normalize_lm_head_weight = False
        self.src_quantization_bit = 0
        self.tp_split_emb_and_lm_head = True

        self.ptuning_path = None
        self.multi_task_prompt = None
        self.pre_seq_len = 0
        self.prefix_projection = False
        self.mm_related_params: VitParameters = VitParameters()
        self.ref_module: Optional[torch.nn.Module] = None
        self.ref_dict: Dict[str, torch.Tensor] = {}
        self.task_type = TaskType.LANGUAGE_MODEL

        self.tie_word_embeddings = False
        self.nccl_ip = g_master_info.ip
        self.tp_nccl_port = g_master_info.tp_nccl_port
        self.dp_tp_nccl_port = g_master_info.dp_tp_nccl_port
        self.th_nccl_port = g_master_info.th_nccl_port
        self.ffn_tp_nccl_port = g_master_info.ffn_tp_nccl_port
        self.model_rpc_port = g_worker_info.rpc_server_port
        self.http_port = g_worker_info.http_port
        self.cache_store_listen_port = g_worker_info.cache_store_listen_port
        self.cache_store_connect_port = g_worker_info.cache_store_connect_port
        self.cache_store_rdma_listen_port = g_worker_info.cache_store_rdma_listen_port
        self.cache_store_rdma_connect_port = g_worker_info.cache_store_rdma_connect_port
        self.remote_rpc_server_port = g_worker_info.remote_rpc_server_port
        self.worker_port_offset = WORKER_INFO_PORT_NUM

        self.add_special_tokens = True
        self.template_type = TemplateType.chat
        self.build_position_ids = False
        self.routed_scaling_factor = 1.0
        self.vit_run_batch = False

        self.is_multimodal = False
        self.model_name = ""

        self.world_size = g_parallel_info.world_size
        self.phy2log: List[List[int]] = []

        self.is_mtp = False
        self.qk_norm = False
        self.quant_config = None
        self.config_dtype = None

        # For cpp, we use `gpt_init_params`, `py_env_configs` for python.
        # There are some common envs in cpp and python, so they will
        # share some configs together.
        self.update_gpt_init_params_from_env()
        self.py_env_configs = PyEnvConfigs()
        self.py_env_configs.update_from_env()
        self.py_env_configs.parallelism_distributed_config = (
            self.gpt_init_params.parallelism_distributed_config
        )
        StaticConfig.parallelism_distributed_config = (
            self.gpt_init_params.parallelism_distributed_config
        )

        self.vit_separation = self.py_env_configs.vit_config.vit_separation
        logging.info(f"vit_separation: {self.vit_separation}")
        self.role_type = (
            RoleType.VIT
            if self.vit_separation == 1
            else self.py_env_configs.role_config.role_type
        )

        for k, v in kwargs.items():
            setattr(self, k, v)

    # read and write directly through GptInitModelParameters.k
    def __getattr__(self, k: str):
        return getattr(self.gpt_init_params, k)

    def __setattr__(self, k: str, v: Any):
        updated_params.add(k)
        if k in self.__slots__:
            object.__setattr__(self, k, v)
        elif v is not None:
            self.gpt_init_params.__setattr__(k, v)
            if k in self._model_related_types:
                getattr(self.gpt_init_params, self._model_related_types[k])()

    def update(self, update_params: Dict[str, Any]):
        for k, v in update_params.items():
            setattr(self, k, v)
        return self

    def update_worker_addrs(self):
        worker_addrs = []
        worker_grpc_addrs = []
        for member in get_gang_info().members:
            logging.info(
                f"member world rank: {member.world_rank}, member local rank: {member.local_rank}, local rank: {self.local_rank}, "
                f"tp_size: {self.tp_size}, dp_size: {self.dp_size}, dp_rank: {self.dp_rank}, use_all_gather: {self.use_all_gather}"
            )
            if int((member.world_rank / self.tp_size) % self.dp_size) == self.dp_rank:
                worker_addrs.append(
                    f"{member.ip}:{member.cache_store_listen_port}:{member.cache_store_rdma_listen_port}"
                )
                worker_grpc_addrs.append(f"{member.ip}:{member.rpc_server_port}")
                logging.info(
                    f"append member for pd sep "
                    f"{member.ip}:{member.rpc_server_port}, {member.cache_store_listen_port}, "
                    f"{member.cache_store_rdma_listen_port} to local rank {self.local_rank}, world rank {member.world_rank}"
                )
        self.worker_grpc_addrs = worker_grpc_addrs
        self.worker_addrs = worker_addrs

    def update_gpt_init_params_from_env(
        self, parallel_info: ParallelInfo = g_parallel_info
    ):

        # ParallelismDistributedConfig
        # USE_ALL_GATHER: Enable all-gather communication for pure TP (ep_size == tp_size).
        # When enabled, DeepEP should not be used. Default is False.
        # Calculate use_all_gather: (USE_ALL_GATHER env is True) and (ep_size == tp_size)
        use_all_gather_env = get_env_bool("USE_ALL_GATHER", True)
        use_all_gather = use_all_gather_env and (
            parallel_info.ep_size == parallel_info.tp_size
        )
        self.gpt_init_params.parallelism_distributed_config = (
            ParallelismDistributedConfig(
                tp_size=parallel_info.tp_size,
                ep_size=parallel_info.ep_size,
                dp_size=parallel_info.dp_size,
                world_size=parallel_info.world_size,
                world_rank=parallel_info.world_rank,
                local_world_size=parallel_info.local_world_size,
                pp_size=parallel_info.pp_size,
                ffn_sp_size=parallel_info.ffn_sp_size,
                use_all_gather=use_all_gather,
            )
        )

        # CacheStoreConfig
        self.gpt_init_params.cache_store_config = CacheStoreConfig(
            cache_store_rdma_mode=get_env_bool("CACHE_STORE_RDMA_MODE", False),
            wrr_available_ratio=get_env_int("WRR_AVAILABLE_RATIO", 80),
            rank_factor=get_env_int("RANK_FACTOR", 0),
            thread_count=get_env_int("CACHE_STORE_THREAD_COUNT", 16),
            rdma_connect_timeout_ms=get_env_int(
                "CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS", 250
            ),
            rdma_qp_count_per_connection=get_env_int(
                "CACHE_STORE_RDMA_QP_COUNT_PER_CONNECTION", 2
            ),
            messager_worker_thread_count=get_env_int(
                "MESSAGER_WORKER_THREAD_COUNT", 32
            ),
            messager_io_thread_count=get_env_int("MESSAGER_IO_THREAD_COUNT", 4),
        )

        # ConcurrencyConfig
        self.gpt_init_params.concurrency_config = ConcurrencyConfig(
            concurrency_with_block=get_env_bool("CONCURRENCY_WITH_BLOCK", False),
            concurrency_limit=get_env_int("CONCURRENCY_LIMIT", 32),
        )

        # FMHAConfig
        self.gpt_init_params.fmha_config = FMHAConfig(
            enable_fmha=get_env_bool("ENABLE_FMHA", True),
            enable_trt_fmha=get_env_bool("ENABLE_TRT_FMHA", True),
            enable_paged_trt_fmha=get_env_bool("ENABLE_PAGED_TRT_FMHA", True),
            enable_open_source_fmha=get_env_bool("ENABLE_OPENSOURCE_FMHA", True),
            enable_paged_open_source_fmha=get_env_bool(
                "ENABLE_PAGED_OPEN_SOURCE_FMHA", True
            ),
            enable_trtv1_fmha=get_env_bool("ENABLE_TRTV1_FMHA", True),
            fmha_perf_instrument=get_env_bool("FMHA_PERF_INSTRUMENT", False),
            fmha_show_params=get_env_bool("FMHA_SHOW_PARAMS", False),
            disable_flash_infer=get_env_bool("DISABLE_FLASH_INFER", False),
            enable_xqa=get_env_bool("ENABLE_XQA", True),
        )

        # KVCacheConfig
        self.gpt_init_params.kv_cache_config = KVCacheConfig(
            reuse_cache=get_env_bool("REUSE_CACHE", False),
            multi_task_prompt=get_env_str("MULTI_TASK_PROMPT"),
            multi_task_prompt_str=get_env_str("MULTI_TASK_PROMPT_STR"),
            enable_3fs=get_env_bool("ENABLE_3FS", False),
            match_timeout_ms=get_env_int("THREEFS_MATCH_TIMEOUT_MS", 1000),
            rpc_get_cache_timeout_ms=get_env_int(
                "THREEFS_RPC_GET_CACHE_TIMEOUT_MS", 3000
            ),
            rpc_put_cache_timeout_ms=get_env_int(
                "THREEFS_RPC_PUT_CACHE_TIMEOUT_MS", 3000
            ),
            threefs_read_timeout_ms=get_env_int("THREEFS_READ_TIMEOUT_MS", 1000),
            threefs_write_timeout_ms=get_env_int("THREEFS_WRITE_TIMEOUT_MS", 2000),
            max_block_size_per_item=get_env_int("MAX_BLOCK_SIZE_PER_ITEM", 16),
            threefs_read_iov_size=get_env_int("THREEFS_READ_IOV_SIZE", 1 << 32),
            threefs_write_iov_size=get_env_int("THREEFS_WRITE_IOV_SIZE", 1 << 32),
            memory_block_cache_size_mb=get_env_int("MEMORY_BLOCK_CACHE_SIZE_MB", 0),
            memory_block_cache_sync_timeout_ms=get_env_int(
                "MEMORY_BLOCK_CACHE_SYNC_TIMEOUT_MS", 10000
            ),
        )

        enable_detail_log = get_env_bool("ENABLE_DETAIL_LOG", False)
        logging.info(f"enable_detail_log = {enable_detail_log}")

        # ProfilingDebugLoggingConfig
        self.gpt_init_params.profiling_debug_logging_config = (
            ProfilingDebugLoggingConfig(
                trace_memory=get_env_bool("RTP_LLM_TRACE_MEMORY", False),
                trace_malloc_stack=get_env_bool("RTP_LLM_TRACE_MALLOC_STACK", False),
                enable_device_perf=get_env_bool("ENABLE_DEVICE_PERF", False),
                ft_core_dump_on_exception=get_env_bool(
                    "FT_CORE_DUMP_ON_EXCEPTION", False
                ),
                ft_alog_conf_path=get_env_str("FT_ALOG_CONF_PATH"),
                log_level=get_env_str("LOG_LEVEL", "INFO"),
                gen_timeline_sync=get_env_bool("GEN_TIMELINE_SYNC", False),
                torch_cuda_profiler_dir=get_env_str("TORCH_CUDA_PROFILER_DIR", ""),
                log_path=get_env_str("log_path", "logs"),
                log_file_backup_count=get_env_int("LOG_FILE_BACKUP_COUNT", 16),
                nccl_debug_file=get_env_str("NCCL_DEBUG_FILE", ""),
                debug_load_server=get_env_bool("DEBUG_LOAD_SERVER", False),
                hack_layer_num=get_env_int("HACK_LAYER_NUM", 0),
                debug_start_fake_process=get_env_bool(
                    "DEBUG_START_FAKE_PROCESS", False
                ),
                dg_print_reg_reuse=get_env_bool("DG_PRINT_REG_REUSE", False),
                qwen_agent_debug=get_env_bool("QWEN_AGENT_DEBUG", False),
                disable_dpc_random=get_env_bool("DISABLE_DPC_RANDOM", False),
                enable_detail_log=get_env_bool("ENABLE_DETAIL_LOG", False),
                check_nan=get_env_bool("CHECK_NAN", False),
            )
        )
        # HWKernelConfig
        self.gpt_init_params.hw_kernel_config = HWKernelConfig(
            deep_gemm_num_sm=get_env_int("DEEP_GEMM_NUM_SM"),
            arm_gemm_use_kai=get_env_bool("ARM_GEMM_USE_KAI"),
            enable_stable_scatter_add=get_env_bool("ENABLE_STABLE_SCATTER_ADD", False),
            enable_multi_block_mode=get_env_bool("ENABLE_MULTI_BLOCK_MODE", True),
            rocm_hipblaslt_config=get_env_str(
                "ROCM_HIPBLASLT_CONFIG", "gemm_config.csv"
            ),
            use_swizzleA=(get_env_bool("USE_SWIZZLEA", False)),
            ft_disable_custom_ar=get_env_bool("FT_DISABLE_CUSTOM_AR", True),
            enable_cuda_graph=get_env_bool("ENABLE_CUDA_GRAPH", False),
            enable_cuda_graph_debug_mode=get_env_bool(
                "ENABLE_CUDA_GRAPH_DEBUG_MODE", False
            ),
            use_aiter_pa=get_env_bool("USE_AITER_PA", True),
            use_asm_pa=get_env_bool("USE_ASM_PA", True),
            enable_native_cuda_graph=get_env_bool("ENABLE_NATIVE_CUDA_GRAPH", False),
            num_native_cuda_graph=get_env_int("NUM_NATIVE_CUDA_GRAPH", 200),
        )

        # DeviceResourceConfig
        self.gpt_init_params.device_resource_config = DeviceResourceConfig(
            device_reserve_memory_bytes=get_env_int(
                "DEVICE_RESERVE_MEMORY_BYTES", -1073741824
            ),
            host_reserve_memory_bytes=get_env_int(
                "HOST_RESERVE_MEMORY_BYTES", 4 * 1024 * 1024 * 1024
            ),
            overlap_math_sm_count=get_env_int("OVERLAP_MATH_SM_COUNT", 0),
            overlap_comm_type=get_env_int("OVERLAP_COMM_TYPE", 0),
            m_split=get_env_int("M_SPLIT", 0),
            enable_comm_overlap=get_env_bool("ENABLE_COMM_OVERLAP", True),
            enable_layer_micro_batch=get_env_int("ENABLE_LAYER_MICRO_BATCH", 0),
            not_use_default_stream=get_env_bool("NOT_USE_DEFAULT_STREAM", False),
        )

        # MoeConfig
        use_deepep_moe_env = get_env_optional_bool("USE_DEEPEP_MOE")
        use_deepep_internode_env = get_env_optional_bool("USE_DEEPEP_INTERNODE")
        use_deepep_low_latency_env = get_env_optional_bool("USE_DEEPEP_LOW_LATENCY")

        self.gpt_init_params.moe_config = MoeConfig(
            use_deepep_moe=(
                use_deepep_moe_env if use_deepep_moe_env is not None else False
            ),
            use_deepep_internode=(
                use_deepep_internode_env
                if use_deepep_internode_env is not None
                else False
            ),
            use_deepep_low_latency=(
                use_deepep_low_latency_env
                if use_deepep_low_latency_env is not None
                else True
            ),
            use_deepep_p2p_low_latency=get_env_bool(
                "USE_DEEPEP_P2P_LOW_LATENCY", False
            ),
            fake_balance_expert=get_env_bool("FAKE_BALANCE_EXPERT", False),
            eplb_control_step=get_env_int("EPLB_CONTROL_STEP", 100),
            eplb_test_mode=get_env_bool("EPLB_TEST_MODE", False),
            hack_moe_expert=get_env_bool("HACK_MOE_EXPERT", False),
            eplb_balance_layer_per_step=get_env_int("EPLB_BALANCE_LAYER_PER_STEP", 1),
            deep_ep_num_sm=get_env_int("DEEP_EP_NUM_SM", 0),
            max_moe_normal_masked_token_num=get_env_int(
                "RTP_LLM_MAX_MOE_NORMAL_MASKED_TOKEN_NUM", 1024
            ),
        )

        # ModelSpecificConfig
        self.gpt_init_params.model_specific_config = ModelSpecificConfig(
            max_lora_model_size=get_env_int("MAX_LORA_MODEL_SIZE"),
            load_python_model=get_env_bool("LOAD_PYTHON_MODEL", False),
        )

        # ServiceDiscoveryConfig
        self.gpt_init_params.service_discovery_config = ServiceDiscoveryConfig(
            use_local=get_env_bool("USE_LOCAL"),
            remote_rpc_server_ip=get_env_str("REMOTE_RPC_SERVER_IP"),
            decode_cm2_config=get_env_str("RTP_LLM_DECODE_CM2_CONFIG"),
            remote_vit_server_ip=get_env_str("REMOTE_VIT_SERVER_IP"),
            multimodal_part_cm2_config=get_env_str(
                "RTP_LLM_MULTIMODAL_PART_CM2_CONFIG"
            ),
            # TODO(yinzhi): fix it
            # remote_backend_ip=get_env_str("REMOTE_BACKEND_IP"),
            # backend_cm2_config=get_env_str("RTP_LLM_BACKEND_CM2_CONFIG"),
        )

        # SchedulerConfig
        self.gpt_init_params.scheduler_config = SchedulerConfig(
            use_batch_decode_scheduler=get_env_bool("USE_BATCH_DECODE_SCHEDULER"),
            use_gather_batch_scheduler=get_env_bool("USE_GATHER_BATCH_SCHEDULER"),
        )
        if (
            self.gpt_init_params.scheduler_config.use_gather_batch_scheduler
            and self.gpt_init_params.scheduler_config.use_batch_decode_scheduler
        ):
            raise ValueError(
                "use_gather_batch_scheduler and use_batch_decode_scheduler cannot be true at the same time"
            )

        # BatchDecodeSchedulerConfig
        self.gpt_init_params.batch_decode_scheduler_config = BatchDecodeSchedulerConfig(
            batch_decode_scheduler_batch_size=get_env_int(
                "BATCH_DECODE_SCHEDULER_BATCH_SIZE", 1
            ),
            batch_decode_scheduler_warmup_type=get_env_int(
                "BATCH_DECODE_SCHEDULER_WARMUP_TYPE", 0
            ),
        )

        # FIFOSchedulerConfig
        self.gpt_init_params.fifo_scheduler_config = FIFOSchedulerConfig(
            max_context_batch_size=get_env_int("MAX_CONTEXT_BATCH_SIZE", 1),
            scheduler_reserve_resource_ratio=get_env_int(
                "SCHEDULER_RESERVE_RESOURCE_RATIO", 5
            ),
            enable_fast_gen=get_env_bool("ENABLE_FAST_GEN", False),
            enable_partial_fallback=get_env_bool("ENABLE_PARTIAL_FALLBACK", False),
            fast_gen_context_budget=get_env_int("FAST_GEN_MAX_CONTEXT_LEN", 0),
        )

        # SamplerConfig
        self.gpt_init_params.sampler_config = SamplerConfig(
            max_batch_size=get_env_int("SAMPLER_MAX_BATCH_SIZE", 0),
            enable_flashinfer_sample_kernel=get_env_bool(
                "ENABLE_FLASHINFER_SAMPLE_KERNEL", True
            ),
        )

        # SpeculativeExecutionConfig
        self.gpt_init_params.sp_config = SpeculativeExecutionConfig(
            sp_model_type=get_env_str("SP_MODEL_TYPE", ""),
            sp_type=get_env_str("SP_TYPE", ""),
            sp_min_token_match=get_env_int("SP_MIN_TOKEN_MATCH", 2),
            sp_max_token_match=get_env_int("SP_MAX_TOKEN_MATCH", 2),
            tree_decode_config=get_env_str("TREE_DECODE_CONFIG", ""),
            gen_num_per_cycle=get_env_int("GEN_NUM_PER_CIRCLE", 1),
            force_stream_sample=get_env_bool("FORCE_STREAM_SAMPLE", False),
            force_score_context_attention=get_env_bool(
                "FORCE_SCORE_CONTEXT_ATTENTION", True
            ),
        )

        # MiscellaneousConfig
        self.gpt_init_params.misc_config = MiscellaneousConfig(
            disable_pdl=get_env_bool("DISABLE_PDL", True),
            aux_string=get_env_str("AUX_STRING", ""),
        )

        # ArpcConfig
        self.gpt_init_params.arpc_config = ArpcConfig(
            threadNum=get_env_int("ARPC_THREAD_NUM", 10),
            queueNum=get_env_int("ARPC_QUEUE_NUM", 50),
            ioThreadNum=get_env_int("ARPC_IO_THREAD_NUM", 2),
        )

        # PD Seperation
        self.decode_entrance = get_env_bool("DECODE_ENTRANCE", False)

    def update_config_with_sparse_config(self, ckpt_path: str):
        sparse_config_file = None
        sparse_config = None
        if os.path.exists(os.path.join(ckpt_path, "config.json")):
            sparse_config_file = os.path.join(ckpt_path, "config.json")
        if self.py_env_configs.sparse_config.sparse_config_file:
            sparse_config_file = self.py_env_configs.sparse_config.sparse_config_file

        if sparse_config_file is not None:
            logging.info(f"read sparse config from: {sparse_config_file}")
            with open(sparse_config_file, "r") as reader:
                sparse_config_json = json.loads(reader.read())
                sparse_config = SparseConfig.from_dict(sparse_config_json)

        if sparse_config and sparse_config.check():
            self.layer_num = sparse_config.layer_num
            self.layer_head_num = sparse_config.layer_head_num
            self.layer_head_num_kv = sparse_config.layer_head_num
            self.layer_inter_size = sparse_config.layer_inter_size
            self.is_sparse_head = True

    def update_inter_padding_size(self, tp_size: int, ep_size: int, dp_size: int):
        if tp_size * dp_size != ep_size:
            raise ValueError(
                f"tp_size:{tp_size} * dp_size:{dp_size} != ep_size:{ep_size}"
            )
        # new tp_size just only for moe
        if self.quant_algo.isGroupwise():
            align_size = tp_size * self.quant_algo.getGroupSize()
            moe_align_size = self.quant_algo.getGroupSize()
        else:
            align_size = tp_size * 64
            moe_align_size = 64
            if self.quant_algo.isFp8PTPC():
                moe_align_size = 128
        if self.layer_inter_size:
            layer_inter_padding_size = []
            for idx in range(len(self.layer_inter_size)):
                inter_size = self.layer_inter_size[idx]
                layer_inter_padding_size.append(
                    inter_size
                    + (
                        get_pad_size(inter_size, align_size)
                        if (
                            self.quant_algo.isQuant()
                            or self.gpt_init_params.hw_kernel_config.use_swizzleA
                        )
                        else 0
                    )
                )
            self.layer_inter_padding_size = layer_inter_padding_size
        self.inter_padding_size = self.inter_size + (
            get_pad_size(self.inter_size, align_size)
            if (
                self.quant_algo.isQuant()
                or self.gpt_init_params.hw_kernel_config.use_swizzleA
            )
            else 0
        )
        if self.head_num_kv <= 0:
            self.head_num_kv = self.head_num
        if self.inter_padding_size <= 0:
            self.inter_padding_size = self.inter_size

        if self.moe_inter_padding_size <= 0:
            self.moe_inter_padding_size = self.inter_size
        if self.moe_inter_padding_size > 0:
            moe_align_size = moe_align_size if self.quant_algo.isQuant() else 8
            self.moe_inter_padding_size = self.moe_inter_padding_size + (
                get_pad_size(self.moe_inter_padding_size, moe_align_size)
            )

        logging.info(
            f"update_inter_padding_size: {self.inter_padding_size}, moe_inter_padding_size: {self.moe_inter_padding_size}, layer_inter_size: {self.layer_inter_size}"
        )

    def update_task_prompt_tokens_id(self, tokenizer):
        if self.multi_task_prompt:
            for info in self.multi_task_prompt:
                task_id: str = str(info["task_id"])
                prompt: str = info["prompt"]
                tokens_id = tokenizer.encode(prompt)
                self.insertMultiTaskPromptTokens(task_id, tokens_id)

    def update_tokenizer_special_tokens(self, tokenizer):
        self.special_tokens.stop_words_id_list += tokenizer.stop_words_id_list
        self.special_tokens.stop_words_str_list += tokenizer.stop_words_str_list
        self.special_tokens.eos_token_id = tokenizer.eos_token_id

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

    def update_task_type_use_kvcache(self):
        self.task_type = check_task_type(self.ckpt_path)
        self.setTaskType(self.task_type.value)
        self.use_kvcache = self.task_type == TaskType.LANGUAGE_MODEL
        logging.info(
            f"model task type: {self.task_type}, use_kvcache: {self.use_kvcache}"
        )

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
        ref_module: Optional[torch.nn.Module] = None,
        ref_dict: Dict[str, torch.Tensor] = {},
        parallel_info: ParallelInfo = g_parallel_info,
        config_mode: ConfigMode = ConfigMode.ComplexMode,
        gang_info: Optional[GangInfo] = None,
    ):

        self._init_precision_config(ckpt_path, quantization, data_type, kv_cache_type)

        self.tp_size = parallel_info.tp_size
        self.tp_rank = parallel_info.tp_rank
        self.ep_size = parallel_info.ep_size
        self.ep_rank = parallel_info.ep_rank
        self.dp_size = parallel_info.dp_size
        self.dp_rank = parallel_info.dp_rank
        self.ffn_tp_rank = parallel_info.ffn_tp_rank
        self.ffn_tp_size = parallel_info.ffn_tp_size
        self.enable_sp = parallel_info.ffn_sp_size > 1
        self.local_rank = parallel_info.local_rank
        self.use_all_gather = (
            # default enable since it has better performance in most cases
            self.gpt_init_params.parallelism_distributed_config.use_all_gather
            and self.gpt_init_params.ep_size == self.gpt_init_params.tp_size
        )
        logging.info(f"use_all_gather: {self.use_all_gather}")

        self.eplb_update_time = self.py_env_configs.py_eplb_config.eplb_update_time
        self.eplb_mode = EplbMode.__members__[
            self.py_env_configs.py_eplb_config.eplb_mode
        ]
        self.enable_eplb = self.eplb_mode != EplbMode.NONE

        self.phy_exp_num = (
            self.py_env_configs.py_eplb_config.redundant_expert + self.expert_num
        )
        logging.info(f"phy_exp_num: {self.phy_exp_num}")

        if gang_info is not None:
            self.num_nodes = gang_info.num_nodes
        else:
            try:
                self.num_nodes = get_gang_info().num_nodes
            except:
                self.num_nodes = 1

        self.ckpt_path = ckpt_path
        self.lora_infos = lora_infos
        self.tokenizer_path = tokenizer_path

        self.gen_num_per_circle = gen_num_per_circle
        self.ptuning_path = ptuning_path
        self.ref_module = ref_module
        self.ref_dict = ref_dict
        if max_seq_len != 0:
            self.max_seq_len = max_seq_len
        if self.max_seq_len < 1:
            # frontend not load ckpt config max_seq_len, use default 8192 or env
            self.max_seq_len = 8192
        logging.info(f"max_seq_len: {self.max_seq_len}")

        self.model_type = StaticConfig.model_config.model_type
        self.update_task_type_use_kvcache()

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
            self.gpt_init_params.ffn_disaggregate_config.enable_ffn_disaggregate = True
            self.gpt_init_params.ffn_disaggregate_config.attention_tp_size = (
                attention_tp_size
            )
            self.gpt_init_params.ffn_disaggregate_config.attention_dp_size = (
                attention_dp_size
            )
            self.gpt_init_params.ffn_disaggregate_config.ffn_tp_size = ffn_tp_size
            # TODO: remove it, ffn dp is stupid
            self.gpt_init_params.ffn_disaggregate_config.ffn_dp_size = 1
            self.gpt_init_params.ffn_disaggregate_config.is_ffn_rank = (
                g_parallel_info.world_rank >= attention_tp_size * attention_dp_size
            )

        logging.info(f"config_mode = {config_mode}")
        if config_mode == ConfigMode.SimpleMode:
            return

        self.update_worker_addrs()
        self.update_config_with_sparse_config(ckpt_path)
        self.update_inter_padding_size(self.tp_size, self.ep_size, self.dp_size)
        self.update_task_prompt_config()

        load_cutlass_gemm_config(self.quant_algo)

        hack_layer_num = self.profiling_debug_logging_config.hack_layer_num
        if hack_layer_num:
            logging.info(f"hack layernum to {hack_layer_num}")
            self.layer_num = hack_layer_num

        self.seq_size_per_block = closest_power_of_2(
            int(max(seq_size_per_block, self.max_seq_len // 128))
        )  # must be 2^n
        if self.py_env_configs.py_kv_cache_config.seq_size_per_block != -1:
            self.seq_size_per_block = int(
                self.py_env_configs.py_kv_cache_config.seq_size_per_block
            )

        logging.info(f"seq_size_per_block: {self.seq_size_per_block}")
        self.max_generate_batch_size = (
            self.py_env_configs.concurrency_config.concurrency_limit
        )

        logging.info(f"max_generate_batch_size: {self.max_generate_batch_size}")
        self.max_context_batch_size = self.fifo_scheduler_config.max_context_batch_size
        logging.info(f"max_context_batch_size: {self.max_context_batch_size}")
        self.reserve_runtime_mem_mb = (
            self.py_env_configs.py_device_resource_config.reserver_runtime_mem_mb
        )
        logging.info(f"reserve_runtime_mem_mb: {self.reserve_runtime_mem_mb}")
        self.kv_cache_mem_mb = self.py_env_configs.py_kv_cache_config.kv_cache_mem_mb
        logging.info(f"kv_cache_mem_mb: {self.kv_cache_mem_mb}")
        self.block_nums = self.py_env_configs.py_kv_cache_config.test_block_num
        logging.info(f"block_nums: {self.block_nums}")
        self.enable_partial_fallback = (
            self.fifo_scheduler_config.enable_partial_fallback
        )
        logging.info(f"enable_partial_fallback: {self.enable_partial_fallback}")
        self.enable_fast_gen = self.fifo_scheduler_config.enable_fast_gen
        logging.info(f"enable_fast_gen: {self.enable_fast_gen}")
        self.warm_up = bool(self.py_env_configs.engine_config.warm_up)
        logging.info(f"warm_up: {self.warm_up}")
        self.warm_up_with_loss = bool(
            self.py_env_configs.engine_config.warm_up_with_loss
        )
        logging.info(f"warm_up_with_loss: {self.warm_up_with_loss}")

        self.fast_gen_max_context_len = (
            1024
            if self.fifo_scheduler_config.fast_gen_context_budget == -1
            else self.fifo_scheduler_config.fast_gen_context_budget
        )
        logging.info(f"fast_gen_max_context_len: {self.fast_gen_max_context_len}")

        self.max_rpc_timeout_ms = int(os.environ.get("MAX_RPC_TIMEOUT_MS", 0))
        logging.info(f"max_rpc_timeout_ms: {self.max_rpc_timeout_ms}")

        self.max_batch_tokens_size = int(
            os.environ.get(
                "MAX_BATCH_TOKENS_SIZE", self.max_context_batch_size * self.max_seq_len
            )
        )
        logging.info(f"max_batch_tokens_size: {self.max_batch_tokens_size}")

        if self.role_type in [RoleType.PREFILL]:
            self.prefill_retry_times = (
                self.py_env_configs.pd_separation_config.prefill_retry_times
            )
            logging.info(f"prefill_retry_times: {self.prefill_retry_times}")
            self.prefill_retry_timeout_ms = (
                self.py_env_configs.pd_separation_config.prefill_retry_timeout_ms
            )
            logging.info(f"prefill_retry_timeout_ms: {self.prefill_retry_timeout_ms}")
            self.prefill_max_wait_timeout_ms = (
                self.py_env_configs.pd_separation_config.prefill_max_wait_timeout_ms
            )
            logging.info(
                f"prefill_max_wait_timeout_ms: {self.prefill_max_wait_timeout_ms}"
            )

        if self.role_type in [RoleType.PREFILL, RoleType.DECODE]:
            self.cache_store_rdma_mode = (
                self.gpt_init_params.cache_store_config.cache_store_rdma_mode
            )
            logging.info(f"cache_store_rdma_mode: {self.cache_store_rdma_mode}")

            self.load_cache_timeout_ms = (
                self.py_env_configs.pd_separation_config.load_cache_timeout_ms
            )
            logging.info(f"load_cache_timeout_ms: {self.load_cache_timeout_ms}")

            self.decode_retry_times = (
                self.py_env_configs.pd_separation_config.decode_retry_times
            )
            logging.info(f"decode_retry_times: {self.decode_retry_times}")
            self.decode_retry_timeout_ms = (
                self.py_env_configs.pd_separation_config.decode_retry_timeout_ms
            )
            logging.info(f"decode_retry_timeout_ms: {self.decode_retry_timeout_ms}")
            self.decode_retry_interval_ms = (
                self.py_env_configs.pd_separation_config.decode_retry_interval_ms
            )
            logging.info(f"decode_retry_interval_ms: {self.decode_retry_interval_ms}")

            self.rdma_connect_retry_times = (
                self.py_env_configs.pd_separation_config.rdma_connect_retry_times
            )
            logging.info(f"rdma_connect_retry_times: {self.rdma_connect_retry_times}")

            self.decode_polling_kv_cache_step_ms = (
                self.py_env_configs.pd_separation_config.decode_polling_kv_cache_step_ms
            )
            logging.info(
                f"decode_polling_kv_cache_step_ms: {self.decode_polling_kv_cache_step_ms}"
            )

            self.decode_polling_call_prefill_ms = int(
                os.environ.get("DECODE_POLLING_CALL_PREFILL_MS", 30)
            )
            logging.info(
                f"decode_polling_call_prefill_ms: {self.decode_polling_call_prefill_ms}"
            )

            self.decode_entrance = bool(
                self.py_env_configs.pd_separation_config.decode_entrance
            )
            logging.info(f"decode_entrance: {self.decode_entrance}")

        self.reuse_cache = self.py_env_configs.py_kv_cache_config.reuse_cache
        logging.info(f"reuse_cache: {self.reuse_cache}")
        self.pre_allocate_op_mem = bool(int(os.environ.get("PRE_ALLOCATE_OP_MEM", 1)))
        logging.info(f"pre_allocate_op_mem: {self.pre_allocate_op_mem}")
        logging.info(f"tp_split_emb_and_lm_head: {self.tp_split_emb_and_lm_head}")

        # use environment variables to update stop_words_str and stop_words_id
        env_stop_words_str = self.py_env_configs.generate_env_config.stop_words_str
        env_stop_words_id = self.py_env_configs.generate_env_config.stop_words_list
        env_stop_words_str_list = (
            json.loads(env_stop_words_str) if env_stop_words_str else []
        )
        env_stop_words_id_list = (
            json.loads(env_stop_words_id) if env_stop_words_id else []
        )
        env_force_stop = self.py_env_configs.generate_env_config.force_stop_words
        if env_force_stop:
            self.special_tokens.stop_words_str_list = env_stop_words_str_list
            self.special_tokens.stop_words_id_list = env_stop_words_id_list
        else:
            self.special_tokens.stop_words_str_list = (
                self.special_tokens.stop_words_str_list + env_stop_words_str_list
            )
            self.special_tokens.stop_words_id_list = (
                self.special_tokens.stop_words_id_list + env_stop_words_id_list
            )

        logging.info(
            f"use stop_words_str_list [{self.special_tokens.stop_words_str_list }],"
            f" stop_words_id_list [{self.special_tokens.stop_words_id_list}]"
        )

        model_override_args = json.loads(
            StaticConfig.model_config.json_model_override_args
        )
        if model_override_args:
            if "rope_scaling" in model_override_args:
                # be consistent with RopeStyle
                rope_type = {
                    "no": 0,
                    "base": 1,
                    "glm2": 2,
                    "dynamicntk": 3,
                    "qwendynamicntk": 4,
                    "yarn": 5,
                    "llama3": 6,
                    "mrope": 7,
                }
                rope_override_args = model_override_args["rope_scaling"]
                assert (
                    "type" in rope_override_args
                    and rope_override_args["type"] in rope_type
                )
                self.rotary_embedding_style = rope_type[rope_override_args["type"]]
                if rope_override_args["type"] == "yarn":
                    assert (
                        "factor" in rope_override_args
                        and "original_max_position_embeddings" in rope_override_args
                    )
                    self.rotary_embedding_scale = rope_override_args["factor"]
                    self.org_embedding_max_pos = rope_override_args[
                        "original_max_position_embeddings"
                    ]
                    self.rotary_factor1 = rope_override_args.get("beta_slow", 1.0)
                    self.rotary_factor2 = rope_override_args.get("beta_fast", 1.0)
                    mscale = rope_override_args.get("mscale", 1.0)
                    self.rotary_embedding_mscale = float(
                        (
                            1.0
                            if self.rotary_embedding_scale <= 1
                            else 0.1 * math.log(self.rotary_embedding_scale) + 1.0
                        )
                        * mscale
                    )
                    self.rotary_embedding_extrapolation_factor = rope_override_args.get(
                        "extrapolation_factor", 1.0
                    )

                logging.info(
                    f"rotary_embedding_style: {self.rotary_embedding_style}, "
                    f"rotary_embedding_scale: {self.rotary_embedding_scale}, "
                    f"org_embedding_max_pos: {self.org_embedding_max_pos}, "
                    f"rotary_factor1: {self.rotary_factor1}, "
                    f"rotary_factor2: {self.rotary_factor2}, "
                    f"rotary_embedding_mscale: {self.rotary_embedding_mscale}, "
                    f"rotary_embedding_extrapolation_factor: {self.rotary_embedding_extrapolation_factor}"
                )

    def _init_precision_config(
        self,
        ckpt_path: str,
        quantization: str,
        data_type_str: Optional[str],
        kv_cache_dtype_str: Optional[str],
    ):
        quant_config = self._load_quant_config_from_ckpt(ckpt_path)
        if not quant_config:
            if quantization:
                quant_config = init_quant_config(quantization)
                logging.info(f"need_load_quant by {quant_config.get_method()}")
        if quant_config:
            self.quant_algo.setQuantAlgo(
                quant_config.get_algo().lower(),
                quant_config.bits,
                quant_config.group_size(),
            )

        # Verify the data_type
        data_type, kv_cache_data_type = self._get_and_verify_dtype(
            quant_config, data_type_str, kv_cache_dtype_str
        )

        self.quant_config = quant_config
        self.data_type = data_type.to_str()
        self.kv_cache_data_type = kv_cache_data_type.to_str()
        logging.info(
            f"quant_config: {self.quant_config}, data_type:{self.data_type}, kv_cache_data_type: {self.kv_cache_data_type}"
        )

    @staticmethod
    def _load_quant_config_from_ckpt(ckpt_path: str) -> Optional[QuantizationConfig]:
        quant_config_path = os.path.join(ckpt_path, "smoothquant.ini")
        if os.path.exists(quant_config_path):
            return QuantizationConfig.from_config(
                {
                    "bits": 0,
                    "method": "smooth_quant",
                    "group_size": 0,
                    "is_quanted": True,
                }
            )

        per_tensor_config_path = os.path.join(ckpt_path, "pertensorquant.ini")

        if os.path.exists(per_tensor_config_path):
            return QuantizationConfig.from_config(
                {
                    "bits": 0,
                    "method": "pertensor_quant",
                    "group_size": 0,
                    "is_quanted": True,
                }
            )

        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return None

        config_json = json.load(open(config_path))
        quant_config = None
        quant_method = None
        if config_json.get("quantization_config", None):
            quant_config = config_json["quantization_config"]
            quant_method = quant_config["quant_method"].lower()

        if config_json.get("quantization", None):
            quant_config = config_json["quantization"]
            quant_method = quant_config["quant_algo"].lower()
        if quant_config is None:
            return None

        group_size = quant_config["group_size"] if "group_size" in quant_config else 0
        bits = quant_config["bits"] if "bits" in quant_config else 0
        if quant_method == "fp8":
            bits = 8
            if "weight_block_size" in quant_config:
                weight_block = quant_config.get("weight_block_size")
                assert isinstance(weight_block, list) and all(
                    element == weight_block[0] for element in weight_block
                ), f"weight_block_size: {weight_block} must be same"
                group_size = weight_block[0]
                quant_method = Fp8BlockWiseQuantConfig.get_method()
        if quant_method == "compressed-tensors":
            config_groups = quant_config["config_groups"]
            weights_config = config_groups["group_0"]["weights"]
            activation_config = config_groups["group_0"]["input_activations"]
            bits = weights_config["num_bits"]
            if (
                weights_config["type"] == "float"
                and bits == 8
                and weights_config["strategy"] == "channel"
            ):
                quant_method = Fp8PerChannelCompressedQuantConfig.get_method()
            elif (
                weights_config["type"] == "float"
                and bits == 8
                and weights_config["strategy"] == "tensor"
            ):
                quant_method = Fp8PerTensorCompressedQuantConfig.get_method()
                return Fp8PerTensorCompressedQuantConfig.from_config(
                    {
                        "bits": bits,
                        "method": quant_method,
                        "group_size": group_size,
                        "is_quanted": True,
                        "dynamic": activation_config["dynamic"],
                        "act_scale_suffix": ".input_scale",
                        "weight_scale_suffix": ".weight_scale",
                    }
                )

        return QuantizationConfig.from_config(
            {
                "bits": bits,
                "method": quant_method,
                "group_size": group_size,
                "is_quanted": True,
            }
        )

    def _get_and_verify_dtype(
        self, quant_config: QuantizationConfig, data_type_str, kv_cache_dtype_str
    ):
        data_type: WEIGHT_TYPE = None
        config_dtype = (
            WEIGHT_TYPE.from_str(self.config_dtype) if self.config_dtype else None
        )
        if data_type_str:
            data_type = WEIGHT_TYPE.from_str(data_type_str)
            logging.info(f"set data_type by args: {data_type}")

        if not data_type or data_type == WEIGHT_TYPE.AUTO:
            data_type = config_dtype if config_dtype else WEIGHT_TYPE.FP16
            logging.info(
                f"data_type is not set or it's auto,we will use config_dtype:{config_dtype} or {WEIGHT_TYPE.FP16}"
            )
        if quant_config and isinstance(quant_config, Fp8BlockWiseQuantConfig):
            data_type = WEIGHT_TYPE.BF16  # now fp8_block_wise only support bf16
            logging.info(f"now fp8_block_wise only support bf16")
        elif quant_config and quant_config.get_method().lower() in [
            "smooth_quant",
            "omni_quant",
        ]:
            data_type = WEIGHT_TYPE.FP16

        if config_dtype and data_type != config_dtype:
            if data_type == WEIGHT_TYPE.FP32:
                # Upcasting to float32 is allowed.
                logging.info("Upcasting %s to %s.", config_dtype, data_type)
            elif config_dtype == WEIGHT_TYPE.FP32:
                # Downcasting from float32 to float16 or bfloat16 is allowed.
                logging.info("Downcasting %s to %s.", config_dtype, data_type)
            else:
                # Casting between float16 and bfloat16 is allowed with a warning.
                logging.warning("Casting %s to %s.", config_dtype, data_type)

        kv_cache_data_type: Optional[WEIGHT_TYPE] = (
            WEIGHT_TYPE.from_str(kv_cache_dtype_str)
            if kv_cache_dtype_str
            else data_type
        )
        if quant_config and quant_config.get_method().lower() == "fp8":
            kv_cache_data_type = WEIGHT_TYPE.FP8

        if kv_cache_data_type == WEIGHT_TYPE.AUTO:
            kv_cache_data_type: WEIGHT_TYPE = data_type

        if quant_config:
            quant_config.verify_compute_dtype_and_kv_cache_dtype(
                data_type.to_torch_dtype(), kv_cache_data_type.to_torch_dtype()
            )
        return (data_type, kv_cache_data_type)

    def get_params_dict(self):
        res: Dict[str, Any] = {}
        for name in updated_params:
            res[name] = eval("self." + name)
        return res

    def eval_model_size(self):
        model_size = self.eval_model_weight_size()
        kv_cache_mem_size = self._eval_kv_cache_mem_size()
        runtime_buffer = self._eval_runtime_buffer_mem_size()
        total_size = model_size + kv_cache_mem_size + runtime_buffer
        logging.info(
            f"total_size(Bytes): {total_size}, model_size:{model_size}, kv_cache_mem_size:{kv_cache_mem_size}, runtime_buffer:{runtime_buffer}"
        )
        return total_size

    def eval_model_weight_size(self):
        layer_param_bytes = 2
        if self.quant_algo.getWeightBits() == 8:
            layer_param_bytes = 1
        elif self.quant_algo.getWeightBits() == 4:
            layer_param_bytes = 0.54

        model_size = (
            self.word_emb_param_count * 2
            + self.layer_weight_param_count * layer_param_bytes
            + self.gpt_init_params.hidden_size * layer_param_bytes
            + self.word_emb_param_count * 2
        )  # maybe some model donot have lm_head
        return model_size

    def _eval_kv_cache_mem_size(self):
        if self.task_type != TaskType.LANGUAGE_MODEL:
            return 0
        kv_cache_bytes = (
            1
            if self.kv_cache_data_type
            in [WEIGHT_TYPE.FP8.to_str(), WEIGHT_TYPE.INT8.to_str()]
            else 2
        )
        kv_cache_size = (
            2
            * self.layer_num
            * self.head_num_kv
            * self.size_per_head
            * kv_cache_bytes
            * self.max_seq_len
        )
        return kv_cache_size

    def _eval_runtime_buffer_mem_size(self):
        input_buffer = self.max_seq_len * self.gpt_init_params.hidden_size
        qkv_gemm_buffer_size = (
            self.max_seq_len
            * (self.head_num_kv * 2 + self.head_num_kv)
            * self.size_per_head
        )
        attn_buffer_size = self.max_seq_len * self.gpt_init_params.hidden_size
        ffn_export_num = self.expert_num if self.gpt_init_params.moe_k else 1
        ffn_w_count = 1 if self.activation_type == "gelu" else 2
        ffn_buffer = (
            self.max_seq_len * self.gpt_init_params.hidden_size
            + ffn_w_count * self.max_seq_len * self.inter_size
        ) * ffn_export_num
        return input_buffer + qkv_gemm_buffer_size + attn_buffer_size + ffn_buffer

    @property
    def model_param_count(self):
        return (
            self.word_emb_param_count * 2
            + self.layer_weight_param_count
            + self.gpt_init_params.hidden_size
        )

    @property
    def word_emb_param_count(self):
        return self.vocab_size * self.gpt_init_params.hidden_size

    @property
    def layer_weight_param_count(self):
        hidden_size = self.gpt_init_params.hidden_size

        layer_weight_param_count = 0
        # qkv
        if self.layer_head_num and isinstance(self.layer_head_num, list):
            for head_num in self.layer_head_num:
                layer_weight_param_count = (
                    layer_weight_param_count
                    + head_num * self.size_per_head * hidden_size * 3
                )
        elif self.head_num_kv != self.head_num:
            layer_weight_param_count = (
                layer_weight_param_count
                + self.layer_num * hidden_size * hidden_size
                + self.layer_num * (self.head_num_kv * self.size_per_head) * 2
            )
        else:
            layer_weight_param_count = (
                layer_weight_param_count
                + self.layer_num * hidden_size * hidden_size * 3
            )

        # attn_o_w
        if self.layer_head_num and isinstance(self.layer_head_num, list):
            for head_num in self.layer_head_num:
                layer_weight_param_count = (
                    layer_weight_param_count
                    + head_num * self.size_per_head * hidden_size
                )
        else:
            layer_weight_param_count = (
                layer_weight_param_count + self.layer_num * hidden_size * hidden_size
            )

        # ffn w1, w2, w3
        ffn_export_num = self.expert_num if self.expert_num > 0 else 1
        ffn_w_count = 2 if self.activation_type == "gelu" else 3
        if self.layer_inter_size and isinstance(self.layer_inter_size, list):
            for layer_inter_size in self.layer_inter_size:
                if self.moe_style == 1:
                    layer_weight_param_count = (
                        layer_weight_param_count
                        + layer_inter_size * hidden_size * ffn_w_count * ffn_export_num
                    )
                else:
                    layer_weight_param_count = (
                        layer_weight_param_count
                        + layer_inter_size * hidden_size * ffn_w_count
                    )
                    if self.moe_style == 2:
                        layer_weight_param_count = (
                            layer_weight_param_count
                            + self.moe_inter_padding_size
                            * hidden_size
                            * ffn_w_count
                            * ffn_export_num
                        )

        else:
            if self.moe_style == 1:
                layer_weight_param_count = (
                    layer_weight_param_count
                    + self.layer_num
                    * self.inter_size
                    * hidden_size
                    * ffn_w_count
                    * ffn_export_num
                )
            else:
                layer_weight_param_count = (
                    layer_weight_param_count
                    + self.layer_num * self.inter_size * hidden_size * ffn_w_count
                )
                if self.moe_style == 2:
                    layer_weight_param_count = (
                        layer_weight_param_count
                        + len(self.moe_layer_index)
                        * self.moe_inter_padding_size
                        * hidden_size
                        * ffn_w_count
                        * ffn_export_num
                    )

        if ffn_export_num > 1:
            layer_weight_param_count = (
                layer_weight_param_count
                + len(self.moe_layer_index) * hidden_size * ffn_export_num
            )
        # other small tensor
        layer_weight_param_count = (
            layer_weight_param_count + self.layer_num * hidden_size * 11
        )
        return layer_weight_param_count
