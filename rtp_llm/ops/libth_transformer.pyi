from __future__ import annotations

import typing

import torch

__all__ = [
    "ArpcConfig",
    "AttentionCommonInputs",
    "BatchDecodeSchedulerConfig",
    "CacheStoreConfig",
    "ConcurrencyConfig",
    "DeviceExporter",
    "DeviceResourceConfig",
    "DeviceType",
    "EngineScheduleInfo",
    "EngineTaskInfo",
    "EplbConfig",
    "EplbMode",
    "FIFOSchedulerConfig",
    "FMHAConfig",
    "FlashInferOp",
    "FfnDisAggregateConfig",
    "GptInitParameter",
    "HWKernelConfig",
    "Host",
    "KVCacheConfig",
    "KVCacheInfo",
    "WorkerStatusInfo",
    "MiscellaneousConfig",
    "MlaOpsType",
    "ModelSpecificConfig",
    "MoeConfig",
    "MultimodalInput",
    "ParallelismDistributedConfig",
    "ProfilingDebugLoggingConfig",
    "QuantAlgo",
    "RoleSpecialTokens",
    "RoleType",
    "RtpEmbeddingOp",
    "RtpLLMOp",
    "SamplerConfig",
    "SchedulerConfig",
    "ServiceDiscoveryConfig",
    "SpecialTokens",
    "SpeculativeExecutionConfig",
    "get_block_cache_keys",
    "get_device",
]

class ArpcConfig:
    ioThreadNum: int
    queueNum: int
    threadNum: int

    def __init__(
        self, threadNum: int = 10, queueNum: int = 50, ioThreadNum: int = 2
    ) -> None: ...
    def to_string(self) -> str: ...

class AttentionCommonInputs:
    def __init__(self) -> None: ...

class BatchDecodeSchedulerConfig:
    batch_decode_scheduler_batch_size: int
    batch_decode_scheduler_warmup_type: int
    def __init__(
        self,
        batch_decode_scheduler_batch_size: int = 1,
        batch_decode_scheduler_warmup_type: int = 0,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class CacheStoreConfig:
    cache_store_rdma_mode: bool
    rank_factor: int
    wrr_available_ratio: int
    thread_count: int
    rdma_connect_timeout_ms: int
    rdma_qp_count_per_connection: int
    rdma_io_thread_count: int
    rdma_worker_thread_count: int
    messager_io_thread_count: int
    messager_worker_thread_count: int

    def __init__(
        self,
        cache_store_rdma_mode: bool = False,
        wrr_available_ratio: int = 80,
        rank_factor: int = 0,
        thread_count: int = 16,
        rdma_connect_timeout_ms: int = 250,
        rdma_qp_count_per_connection: int = 2,
        rdma_io_thread_count: int = 1,
        rdma_worker_thread_count: int = 2,
        messager_io_thread_count: int = 2,
        messager_worker_thread_count: int = 16,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class ConcurrencyConfig:
    concurrency_limit: int
    concurrency_with_block: bool

    def __init__(
        self, concurrency_with_block: bool = False, concurrency_limit: int = 32
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class DeviceExporter:
    def get_device_id(self) -> int: ...
    def get_device_type(self) -> DeviceType: ...
    def pack_int8_tensor_to_packed_int4(self, weight: torch.Tensor) -> torch.Tensor: ...
    def preprocess_gemm_weight_by_key(
        self, key: str, weight: torch.Tensor, user_arm_gemm_use_kai: bool
    ) -> torch.Tensor: ...
    def preprocess_weight_scale(
        self, weight: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor: ...
    def preprocess_weights_for_mixed_gemm(
        self, weight: torch.Tensor, quant_type: typing.Any, arch: str
    ) -> torch.Tensor: ...
    def symmetric_quantize_last_axis_of_batched_matrix(
        self, weight: torch.Tensor, quant_type: typing.Any, arch: str
    ) -> list[torch.Tensor]: ...

class DeviceResourceConfig:
    device_reserve_memory_bytes: int
    enable_comm_overlap: bool
    enable_layer_micro_batch: int
    host_reserve_memory_bytes: int
    m_split: int
    not_use_default_stream: bool
    overlap_comm_type: int
    overlap_math_sm_count: int

    def __init__(
        self,
        device_reserve_memory_bytes: int = 0,
        host_reserve_memory_bytes: int = 4294967296,
        overlap_math_sm_count: int = 0,
        overlap_comm_type: int = 0,
        m_split: int = 0,
        enable_comm_overlap: bool = True,
        enable_layer_micro_batch: int = 0,
        not_use_default_stream: bool = False,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class DeviceType:
    """
    Members:

      Cpu

      Cuda

      Yitian

      ArmCpu

      ROCm

      Ppu
    """

    ArmCpu: typing.ClassVar[DeviceType]  # value = <DeviceType.ArmCpu: 3>
    Cpu: typing.ClassVar[DeviceType]  # value = <DeviceType.Cpu: 0>
    Cuda: typing.ClassVar[DeviceType]  # value = <DeviceType.Cuda: 1>
    Ppu: typing.ClassVar[DeviceType]  # value = <DeviceType.Ppu: 5>
    ROCm: typing.ClassVar[DeviceType]  # value = <DeviceType.ROCm: 4>
    Yitian: typing.ClassVar[DeviceType]  # value = <DeviceType.Yitian: 2>
    __members__: typing.ClassVar[
        dict[str, DeviceType]
    ]  # value = {'Cpu': <DeviceType.Cpu: 0>, 'Cuda': <DeviceType.Cuda: 1>, 'Yitian': <DeviceType.Yitian: 2>, 'ArmCpu': <DeviceType.ArmCpu: 3>, 'ROCm': <DeviceType.ROCm: 4>, 'Ppu': <DeviceType.Ppu: 5>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class EngineScheduleInfo:
    finished_task_info_list: list[EngineTaskInfo]
    last_schedule_delta: int
    running_task_info_list: list[EngineTaskInfo]
    def __init__(self) -> None: ...

class EngineTaskInfo:
    end_time_ms: int
    inter_request_id: int
    input_length: int
    iterate_count: int
    prefix_length: int
    request_id: int
    waiting_time_ms: int
    def __init__(self) -> None: ...

class EplbConfig:
    __hash__: typing.ClassVar[None] = None
    mode: EplbMode
    update_time: int
    def __eq__(self, arg0: EplbConfig) -> bool: ...
    def __init__(self) -> None: ...
    def __ne__(self, arg0: EplbConfig) -> bool: ...
    def __str__(self) -> str: ...

class EplbMode:
    """
    Members:

      NONE

      STATS

      EPLB

      ALL
    """

    ALL: typing.ClassVar[EplbMode]  # value = <EplbMode.ALL: 3>
    EPLB: typing.ClassVar[EplbMode]  # value = <EplbMode.EPLB: 2>
    NONE: typing.ClassVar[EplbMode]  # value = <EplbMode.NONE: 0>
    STATS: typing.ClassVar[EplbMode]  # value = <EplbMode.STATS: 1>
    __members__: typing.ClassVar[
        dict[str, EplbMode]
    ]  # value = {'NONE': <EplbMode.NONE: 0>, 'STATS': <EplbMode.STATS: 1>, 'EPLB': <EplbMode.EPLB: 2>, 'ALL': <EplbMode.ALL: 3>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class FIFOSchedulerConfig:
    enable_fast_gen: bool
    enable_partial_fallback: bool
    fast_gen_context_budget: int
    max_context_batch_size: int
    scheduler_reserve_resource_ratio: int

    def __init__(
        self,
        max_context_batch_size: int = 1,
        scheduler_reserve_resource_ratio: int = 5,
        enable_fast_gen: bool = False,
        enable_partial_fallback: bool = False,
        fast_gen_context_budget: int = -1,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class FMHAConfig:
    disable_flash_infer: bool
    enable_fmha: bool
    enable_open_source_fmha: bool
    enable_paged_open_source_fmha: bool
    enable_paged_trt_fmha: bool
    enable_trt_fmha: bool
    enable_trtv1_fmha: bool
    enable_xqa: bool
    fmha_perf_instrument: bool
    fmha_show_params: bool

    def __init__(
        self,
        enable_fmha: bool = True,
        enable_trt_fmha: bool = True,
        enable_paged_trt_fmha: bool = True,
        enable_open_source_fmha: bool = True,
        enable_paged_open_source_fmha: bool = True,
        enable_trtv1_fmha: bool = True,
        fmha_perf_instrument: bool = False,
        fmha_show_params: bool = False,
        disable_flash_infer: bool = False,
        enable_xqa: bool = True,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class FlashInferOp:
    def __init__(self, gpt_init_parameter: GptInitParameter) -> None: ...
    def forward(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_params: typing.Any,
    ) -> None: ...

class FfnDisAggregateConfig:
    attention_dp_size: int
    attention_tp_size: int
    enable_ffn_disaggregate: bool
    ffn_dp_size: int
    ffn_tp_size: int
    is_ffn_rank: bool

    def __init__(
        self,
        enable_ffn_disaggregate: bool = False,
        attention_tp_size: int = 1,
        attention_dp_size: int = 1,
        ffn_tp_size: int = 1,
        ffn_dp_size: int = 1,
        is_ffn_rank: bool = False,
    ) -> None: ...
    def is_ffn_service(self) -> bool: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class FMHAType:
    """
    Members:

      NONE

      PAGED_TRT_V2

      TRT_V2

      PAGED_OPEN_SOURCE

      OPEN_SOURCE

      TRT_V1

      FLASH_INFER

      XQA

      AITER_PREFILL

      AITER_DECODE
    """

    FLASH_INFER: typing.ClassVar[FMHAType]  # value = <FMHAType.FLASH_INFER: 6>
    NONE: typing.ClassVar[FMHAType]  # value = <FMHAType.NONE: 0>
    OPEN_SOURCE: typing.ClassVar[FMHAType]  # value = <FMHAType.OPEN_SOURCE: 4>
    PAGED_OPEN_SOURCE: typing.ClassVar[
        FMHAType
    ]  # value = <FMHAType.PAGED_OPEN_SOURCE: 3>
    PAGED_TRT_V2: typing.ClassVar[FMHAType]  # value = <FMHAType.PAGED_TRT_V2: 1>
    TRT_V1: typing.ClassVar[FMHAType]  # value = <FMHAType.TRT_V1: 5>
    TRT_V2: typing.ClassVar[FMHAType]  # value = <FMHAType.TRT_V2: 2>
    XQA: typing.ClassVar[FMHAType]  # value = <FMHAType.XQA: 7>
    AITER_PREFILL: typing.ClassVar[FMHAType]  # value = <FMHAType.AITER_PREFILL: 8
    AITER_DECODE: typing.ClassVar[FMHAType]  # value = <FMHAType.AITER_DECODE: 9>

    __members__: typing.ClassVar[
        dict[str, FMHAType]
    ]  # value = {'NONE': <FMHAType.NONE: 0>, 'PAGED_TRT_V2': <FMHAType.PAGED_TRT_V2: 1>, 'TRT_V2': <FMHAType.TRT_V2: 2>, 'PAGED_OPEN_SOURCE': <FMHAType.PAGED_OPEN_SOURCE: 3>, 'OPEN_SOURCE': <FMHAType.OPEN_SOURCE: 4>, 'TRT_V1': <FMHAType.TRT_V1: 5>, 'FLASH_INFER': <FMHAType.FLASH_INFER: 6>, 'XQA': <FMHAType.XQA: 7>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class GptInitParameter:
    activation_type: str
    add_bias_linear: bool
    batch_decode_scheduler_config: BatchDecodeSchedulerConfig
    block_nums: int
    cache_store_config: CacheStoreConfig
    cache_store_connect_port: int
    cache_store_listen_port: int
    cache_store_rdma_connect_port: int
    cache_store_rdma_listen_port: int
    cache_store_rdma_mode: bool
    ckpt_path: str
    concurrency_config: ConcurrencyConfig
    cross_attn_input_len: int
    data_type: str
    decode_polling_kv_cache_step_ms: int
    decode_retry_times: int
    decode_retry_timeout_ms: int
    decode_retry_interval_ms: int
    deepseek_mscale_all_dim: float
    deepseek_rope_mscale: float
    device_resource_config: DeviceResourceConfig
    dp_rank: int
    dp_size: int
    dp_tp_nccl_port: int
    embedding_size: int
    enable_3fs: bool
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
    ffn_disaggregate_config: FfnDisAggregateConfig
    ffn_tp_nccl_port: int
    ffn_tp_rank: int
    ffn_tp_size: int
    fifo_scheduler_config: FIFOSchedulerConfig
    fmha_config: FMHAConfig
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
    hw_kernel_config: HWKernelConfig
    include_sep_tokens: bool
    input_embedding_scalar: float
    input_vocab_size: int
    inter_padding_size: int
    inter_size: int
    is_causal: bool
    is_multimodal: bool
    is_sparse_head: bool
    kv_cache_config: KVCacheConfig
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
    misc_config: MiscellaneousConfig
    mla_ops_type: MlaOpsType
    mm_position_ids_style: int
    mm_sep_tokens: list[list[int]]
    model_name: str
    model_rpc_port: int
    model_specific_config: ModelSpecificConfig
    moe_config: MoeConfig
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
    parallelism_distributed_config: ParallelismDistributedConfig
    phy_exp_num: int
    position_id_len_factor: int
    position_ids_style: int
    pre_allocate_op_mem: bool
    pre_seq_len: int
    prefill_max_wait_timeout_ms: int
    prefill_retry_timeout_ms: int
    prefill_retry_times: int
    prefix_projection: bool
    profiling_debug_logging_config: ProfilingDebugLoggingConfig
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
    role_type: RoleType
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
    sampler_config: SamplerConfig
    scheduler_config: SchedulerConfig
    scheduler_reserve_resource_ratio: int
    scoring_func: int
    seq_size_per_block: int
    service_discovery_config: ServiceDiscoveryConfig
    size_per_head: int
    softmax_extra_scale: float
    sp_config: SpeculativeExecutionConfig
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

    def __init__(
        self,
        head_num: int,
        size_per_head: int,
        num_layers: int,
        max_seq_len: int,
        vocab_size: int,
        hidden_size: int,
    ) -> None: ...
    def insertMultiTaskPromptTokens(
        self, task_id: str, tokens_id: list[int]
    ) -> None: ...
    def isGatedActivation(self) -> bool: ...
    def isKvCacheQuant(self) -> bool: ...
    def setActivationType(self) -> None: ...
    def setKvCacheDataType(self) -> None: ...
    def setLayerNormType(self) -> None: ...
    def setNormType(self) -> None: ...
    def setTaskType(self, task: str) -> None: ...
    def showDebugInfo(self) -> None: ...

class HWKernelConfig:
    arm_gemm_use_kai: bool
    deep_gemm_num_sm: int
    enable_multi_block_mode: bool
    enable_stable_scatter_add: bool
    ft_disable_custom_ar: bool
    rocm_hipblaslt_config: str
    use_swizzleA: bool
    enable_cuda_graph: bool
    enable_cuda_graph_debug_mode: bool
    use_aiter_pa: bool
    use_asm_pa: bool
    enable_native_cuda_graph: bool
    num_native_cuda_graph: int
    def __init__(
        self,
        deep_gemm_num_sm: int = -1,
        arm_gemm_use_kai: bool = False,
        enable_stable_scatter_add: bool = False,
        enable_multi_block_mode: bool = True,
        ft_disable_custom_ar: bool = True,
        rocm_hipblaslt_config: str = "gemm_config.csv",
        use_swizzleA: bool = False,
        enable_cuda_graph: bool = False,
        enable_cuda_graph_debug_mode: bool = False,
        use_aiter_pa: bool = True,
        use_asm_pa: bool = True,
        enable_native_cuda_graph: bool = False,
        num_native_cuda_graph: int = 200,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class Host:
    http_port: int
    ip: str
    rpc_port: int
    def __init__(self, ip: str, http_port: int, rpc_port: int) -> None: ...
    def __repr__(self) -> str: ...

class KVCacheConfig:
    multi_task_prompt: str
    multi_task_prompt_str: str
    reuse_cache: bool
    enable_3fs: bool
    match_timeout_ms: int
    rpc_get_cache_timeout_ms: int
    rpc_put_cache_timeout_ms: int
    threefs_read_timeout_ms: int
    threefs_write_timeout_ms: int
    max_block_size_per_item: int
    threefs_read_iov_size: int
    threefs_write_iov_size: int

    def __init__(
        self,
        reuse_cache: bool = False,
        multi_task_prompt: str = "",
        multi_task_prompt_str: str = "",
        enable_3fs: bool = False,
        match_timeout_ms: int = 1000,
        rpc_get_cache_timeout_ms: int = 2000,
        rpc_put_cache_timeout_ms: int = 2000,
        threefs_read_timeout_ms: int = 1000,
        threefs_write_timeout_ms: int = 2000,
        max_block_size_per_item: int = 16,
        threefs_read_iov_size: int = 1 << 32,
        threefs_write_iov_size: int = 1 << 32,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class KVCacheInfo:
    available_kv_cache: int
    block_size: int
    cached_keys: list[int]
    total_kv_cache: int
    version: int
    def __init__(self) -> None: ...

class WorkerStatusInfo:
    role: RoleType
    engine_schedule_info: EngineScheduleInfo
    status_version: int
    alive: bool
    dp_size: int
    tp_size: int
    dp_rank: int
    precision: str
    def __init__(self) -> None: ...

class MiscellaneousConfig:
    disable_pdl: bool
    aux_string: str

    def __init__(
        self,
        disable_pdl: bool = True,
        aux_string: str = "",
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class MlaOpsType:
    """
    Members:

      AUTO

      MHA

      FLASH_INFER

      FLASH_MLA
    """

    AUTO: typing.ClassVar[MlaOpsType]  # value = <MlaOpsType.AUTO: 0>
    FLASH_INFER: typing.ClassVar[MlaOpsType]  # value = <MlaOpsType.FLASH_INFER: 2>
    FLASH_MLA: typing.ClassVar[MlaOpsType]  # value = <MlaOpsType.FLASH_MLA: 3>
    MHA: typing.ClassVar[MlaOpsType]  # value = <MlaOpsType.MHA: 1>
    __members__: typing.ClassVar[
        dict[str, MlaOpsType]
    ]  # value = {'AUTO': <MlaOpsType.AUTO: 0>, 'MHA': <MlaOpsType.MHA: 1>, 'FLASH_INFER': <MlaOpsType.FLASH_INFER: 2>, 'FLASH_MLA': <MlaOpsType.FLASH_MLA: 3>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ModelSpecificConfig:
    load_python_model: bool
    max_lora_model_size: int

    def __init__(
        self, max_lora_model_size: int = -1, load_python_model: bool = False
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class MoeConfig:
    deep_ep_num_sm: int
    eplb_balance_layer_per_step: int
    eplb_control_step: int
    eplb_test_mode: bool
    fake_balance_expert: bool
    hack_moe_expert: bool
    use_deepep_internode: bool
    use_deepep_low_latency: bool
    use_deepep_moe: bool
    use_deepep_p2p_low_latency: bool
    max_moe_normal_masked_token_num: int

    def __init__(
        self,
        use_deepep_moe: bool = False,
        use_deepep_internode: bool = False,
        use_deepep_low_latency: bool = True,
        use_deepep_p2p_low_latency: bool = False,
        fake_balance_expert: bool = False,
        eplb_control_step: int = 100,
        eplb_test_mode: bool = False,
        hack_moe_expert: bool = False,
        eplb_balance_layer_per_step: int = 1,
        deep_ep_num_sm: int = 0,
        max_moe_normal_masked_token_num: int = 1024,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class MultimodalInput:
    mm_type: int
    tensor: torch.Tensor
    url: str
    def __init__(self, url: str, tensor: torch.Tensor, mm_type: int) -> None: ...

class ParallelismDistributedConfig:
    dp_size: int
    ep_size: int
    ffn_sp_size: int
    local_world_size: int
    pp_size: int
    tp_size: int
    world_rank: int
    world_size: int

    def __init__(
        self,
        tp_size: int = 1,
        ep_size: int = 1,
        dp_size: int = 1,
        pp_size: int = 1,
        world_size: int = 1,
        world_rank: int = 0,
        local_world_size: int = 1,
        ffn_sp_size: int = 1,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class ProfilingDebugLoggingConfig:
    trace_memory: bool
    trace_malloc_stack: bool
    enable_device_perf: bool
    ft_core_dump_on_exception: bool
    ft_alog_conf_path: str
    log_level: str
    gen_timeline_sync: bool
    torch_cuda_profiler_dir: str
    log_path: str
    log_file_backup_count: int
    nccl_debug_file: str
    debug_load_server: bool
    hack_layer_num: int
    debug_start_fake_process: int
    dg_print_reg_reuse: bool
    qwen_agent_debug: bool
    disable_dpc_random: bool
    enable_detail_log: bool

    def __init__(
        self,
        trace_memory: bool = False,
        trace_malloc_stack: bool = False,
        enable_device_perf: bool = False,
        ft_core_dump_on_exception: bool = False,
        ft_alog_conf_path: str = "",
        log_level: str = "INFO",
        gen_timeline_sync: bool = False,
        torch_cuda_profiler_dir: str = "",
        log_path: str = "logs",
        log_file_backup_count: int = 16,
        nccl_debug_file: str = "",
        debug_load_server: bool = False,
        hack_layer_num: int = 0,
        debug_start_fake_process: bool = False,
        dg_print_reg_reuse: bool = False,
        qwen_agent_debug: bool = False,
        disable_dpc_random: bool = False,
        enable_detail_log: bool = False,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class QuantAlgo:
    def __getstate__(self) -> tuple: ...
    def __init__(self) -> None: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def getActivationBits(self) -> int: ...
    def getGroupSize(self) -> int: ...
    def getWeightBits(self) -> int: ...
    def isAwq(self) -> bool: ...
    def isFp8(self) -> bool: ...
    def isGptq(self) -> bool: ...
    def isGroupwise(self) -> bool: ...
    def isOmniQuant(self) -> bool: ...
    def isPerTensorQuant(self) -> bool: ...
    def isQuant(self) -> bool: ...
    def isSmoothQuant(self) -> bool: ...
    def isWeightOnlyPerCol(self) -> bool: ...
    def setQuantAlgo(self, quant_method: str, bits: int, group_size: int) -> None: ...

class RoleSpecialTokens:
    eos_token_ids: list[int]
    token_ids: list[int]
    def __init__(self) -> None: ...

class RoleType:
    """
    Members:

      PDFUSION

      PREFILL

      DECODE

      VIT
    """

    DECODE: typing.ClassVar[RoleType]  # value = <RoleType.DECODE: 2>
    PDFUSION: typing.ClassVar[RoleType]  # value = <RoleType.PDFUSION: 0>
    PREFILL: typing.ClassVar[RoleType]  # value = <RoleType.PREFILL: 1>
    VIT: typing.ClassVar[RoleType]  # value = <RoleType.VIT: 3>
    __members__: typing.ClassVar[
        dict[str, RoleType]
    ]  # value = {'PDFUSION': <RoleType.PDFUSION: 0>, 'PREFILL': <RoleType.PREFILL: 1>, 'DECODE': <RoleType.DECODE: 2>, 'VIT': <RoleType.VIT: 3>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @typing.overload
    def __str__(self) -> str: ...
    @typing.overload
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class RtpEmbeddingOp:
    def __init__(self) -> None: ...
    def decode(
        self,
        token_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        request_id: int,
        multimodal_inputs: list[MultimodalInput],
    ) -> typing.Any: ...
    def init(self, model: typing.Any, mm_process_engine: typing.Any) -> None: ...
    def stop(self) -> None: ...

class RtpLLMOp:
    def __init__(self) -> None: ...
    def add_lora(
        self, adapter_name: str, lora_a_weights: typing.Any, lora_b_weights: typing.Any
    ) -> None: ...
    def get_engine_schedule_info(self) -> EngineScheduleInfo: ...
    def get_worker_status_info(self, arg0: int) -> WorkerStatusInfo: ...
    def get_cache_status_info(self, arg0: int) -> KVCacheInfo: ...
    def init(
        self,
        model: typing.Any,
        mm_process_engine: typing.Any,
        propose_model: typing.Any,
        token_processor: typing.Any,
    ) -> None: ...
    def remove_lora(self, adapter_name: str) -> None: ...
    def start_http_server(
        self,
        model_weights_loader: typing.Any,
        lora_infos: typing.Any,
        gang_info: typing.Any,
        tokenizer: typing.Any,
        render: typing.Any,
    ) -> None: ...
    def stop(self) -> None: ...
    def update_eplb_config(self, config: EplbConfig) -> bool: ...
    def update_scheduler_info(self, arg0: str) -> None: ...

class SamplerConfig:
    enable_flashinfer_sample_kernel: bool
    max_batch_size: int

    def __init__(
        self, max_batch_size: int = 0, enable_flashinfer_sample_kernel: bool = True
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class SchedulerConfig:
    use_batch_decode_scheduler: bool
    def __init__(self, use_batch_decode_scheduler: bool = False) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class ServiceDiscoveryConfig:
    decode_cm2_config: str
    multimodal_part_cm2_config: str
    remote_rpc_server_ip: str
    remote_vit_server_ip: str
    use_local: bool

    def __init__(
        self,
        use_local: bool = False,
        remote_rpc_server_ip: str = "",
        decode_cm2_config: str = "",
        remote_vit_server_ip: str = "",
        multimodal_part_cm2_config: str = "",
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

class SpecialTokens:
    assistant: RoleSpecialTokens
    bos_token_id: int
    decoder_start_token_id: int
    eos_token_id: int
    pad_token_id: int
    stop_words_id_list: list[list[int]]
    stop_words_str_list: list[str]
    system: RoleSpecialTokens
    user: RoleSpecialTokens
    def __init__(self) -> None: ...

class SpeculativeExecutionConfig:
    gen_num_per_cycle: int
    sp_max_token_match: int
    sp_min_token_match: int
    sp_model_type: str
    sp_type: str
    tree_decode_config: str

    def __init__(
        self,
        sp_model_type: str = "",
        sp_type: str = "",
        sp_min_token_match: int = 2,
        sp_max_token_match: int = 2,
        tree_decode_config: str = "",
        gen_num_per_cycle: int = 1,
        force_stream_sample: bool = False,
        force_score_context_attention: bool = True,
    ) -> None: ...
    def to_string(self) -> str: ...
    def update_from_env(self) -> None: ...

def get_block_cache_keys(token_ids_list: list[list[int]]) -> list[int]: ...
def get_device() -> DeviceExporter: ...

class PyModelInitResources:
    k_cache_base: Optional[torch.Tensor]
    v_cache_base: Optional[torch.Tensor]

class KVCache:
    k_cache_base: torch.Tensor
    v_cache_base: torch.Tensor
    k_scale_base: torch.Tensor
    v_scale_base: torch.Tensor
    def get_layer_cache(self, idx: int) -> KVCache: ...

class PyCacheStoreInputs:
    def __init__(self) -> None: ...

class PyAttentionInputs:
    is_prefill: bool
    prefix_lengths: torch.Tensor
    sequence_lengths: torch.Tensor
    input_lengths: torch.Tensor
    kv_cache_block_id_host: torch.Tensor
    kv_cache_block_id_device: torch.Tensor
    dtype: torch.dtype
    kv_block_offset: int = 0
    cu_seqlens: torch.Tensor
    padding_offset: torch.Tensor
    def get_prefill_flash_infer_attn(self) -> typing.Any:
        pass

    def get_decode_flash_infer_attn(self) -> typing.Any:
        pass

class BertEmbeddingInputs:
    combo_position_ids: torch.Tensor
    position_encoding: torch.Tensor
    combo_tokens_type_ids: torch.Tensor
    token_type_embedding: torch.Tensor
    input_embedding_scalar: float
    def __init__(self) -> None: ...

class PyModelInputs:
    input_ids: torch.Tensor
    attention_inputs: PyAttentionInputs
    bert_embedding_inputs: BertEmbeddingInputs

class ParamsBase:
    def fill_params(
        self,
        sequence_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
        kv_cache_block_id_host: torch.Tensor,
        batch_size: int,
        seq_size_per_block: int,
    ) -> None: ...

class PyModelOutputs:
    hidden_states: torch.Tensor
    params_ptr: ParamsBase
    def __init__(self) -> None: ...
    def __init__(self, hidden_states: torch.Tensor, params_ptr: ParamsBase) -> None: ...
    def __init__(self, hidden_states: torch.Tensor) -> None: ...
    def __init__(self, params_ptr: ParamsBase) -> None: ...
