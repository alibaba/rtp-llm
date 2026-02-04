from __future__ import annotations

import typing

__all__: list[str] = ['ActivationType', 'ArpcConfig', 'AttentionConfigs', 'BatchDecodeSchedulerConfig', 'CacheStoreConfig', 'ConcurrencyConfig', 'DataType', 'DeviceResourceConfig', 'EPLBConfig', 'EplbMode', 'FIFOSchedulerConfig', 'FMHAConfig', 'FMHAType', 'FfnDisAggregateConfig', 'GrpcConfig', 'HWKernelConfig', 'HybridAttentionConfig', 'HybridAttentionType', 'KVCacheConfig', 'KvCacheDataType', 'LayerNormType', 'LinearAttentionConfig', 'MMModelConfig', 'MiscellaneousConfig', 'MlaOpsType', 'ModelConfig', 'ModelSpecificConfig', 'MoeConfig', 'NormType', 'PDSepConfig', 'ParallelismConfig', 'ProfilingDebugLoggingConfig', 'QuantAlgo', 'QuantMethod', 'RoleSpecialTokens', 'RoleType', 'RopeConfig', 'RopeStyle', 'RuntimeConfig', 'SpecialTokens', 'SpeculativeExecutionConfig', 'SpeculativeType', 'TaskType', 'VitConfig', 'VitSeparation', 'get_block_cache_keys']
class ActivationType:
    """
    Members:

      Gelu

      Relu

      Silu

      Swiglu

      Geglu

      Identity

      GeluNoneApproximate

      GeGluNoneApproximate

      Sigmoid

      InvalidType
    """
    GeGluNoneApproximate: typing.ClassVar[ActivationType]  # value = <ActivationType.GeGluNoneApproximate: 7>
    Geglu: typing.ClassVar[ActivationType]  # value = <ActivationType.Geglu: 4>
    Gelu: typing.ClassVar[ActivationType]  # value = <ActivationType.Gelu: 0>
    GeluNoneApproximate: typing.ClassVar[ActivationType]  # value = <ActivationType.GeluNoneApproximate: 6>
    Identity: typing.ClassVar[ActivationType]  # value = <ActivationType.Identity: 5>
    InvalidType: typing.ClassVar[ActivationType]  # value = <ActivationType.InvalidType: 9>
    Relu: typing.ClassVar[ActivationType]  # value = <ActivationType.Relu: 1>
    Sigmoid: typing.ClassVar[ActivationType]  # value = <ActivationType.Sigmoid: 8>
    Silu: typing.ClassVar[ActivationType]  # value = <ActivationType.Silu: 2>
    Swiglu: typing.ClassVar[ActivationType]  # value = <ActivationType.Swiglu: 3>
    __members__: typing.ClassVar[dict[str, ActivationType]]  # value = {'Gelu': <ActivationType.Gelu: 0>, 'Relu': <ActivationType.Relu: 1>, 'Silu': <ActivationType.Silu: 2>, 'Swiglu': <ActivationType.Swiglu: 3>, 'Geglu': <ActivationType.Geglu: 4>, 'Identity': <ActivationType.Identity: 5>, 'GeluNoneApproximate': <ActivationType.GeluNoneApproximate: 6>, 'GeGluNoneApproximate': <ActivationType.GeGluNoneApproximate: 7>, 'Sigmoid': <ActivationType.Sigmoid: 8>, 'InvalidType': <ActivationType.InvalidType: 9>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ArpcConfig:
    ioThreadNum: int
    queueNum: int
    threadNum: int
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class AttentionConfigs:
    dtype: torch.dtype
    fuse_qkv_add_bias: bool
    head_num: int
    is_causal: bool
    kv_cache_dtype: KvCacheDataType
    kv_head_num: int
    kv_lora_rank: int
    nope_head_dim: int
    q_lora_rank: int
    q_scaling: float
    rope_config: RopeConfig
    rope_head_dim: int
    size_per_head: int
    skip_append_kv_cache: bool
    softmax_extra_scale: float
    tokens_per_block: int
    use_logn_attn: bool
    use_mla: bool
    v_head_dim: int
    def __init__(self) -> None:
        ...
class BatchDecodeSchedulerConfig:
    batch_decode_scheduler_batch_size: int
    batch_decode_scheduler_warmup_type: int
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class CacheStoreConfig:
    cache_store_rdma_mode: bool
    messager_io_thread_count: int
    messager_worker_thread_count: int
    rank_factor: int
    rdma_connect_timeout_ms: int
    rdma_io_thread_count: int
    rdma_qp_count_per_connection: int
    rdma_worker_thread_count: int
    thread_count: int
    wrr_available_ratio: int
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class ConcurrencyConfig:
    concurrency_limit: int
    concurrency_with_block: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class DataType:
    """
    Members:

      TYPE_INVALID

      TYPE_BOOL

      TYPE_UINT8

      TYPE_UINT16

      TYPE_UINT32

      TYPE_UINT64

      TYPE_INT8

      TYPE_INT16

      TYPE_INT32

      TYPE_INT64

      TYPE_FP16

      TYPE_FP32

      TYPE_FP64

      TYPE_BYTES

      TYPE_BF16

      TYPE_FP8_E4M3

      TYPE_STR

      TYPE_VOID

      TYPE_QINT8

      TYPE_INT4X2

      TYPE_QINT4X2

      TYPE_QFP8_E4M3
    """
    TYPE_BF16: typing.ClassVar[DataType]  # value = <DataType.TYPE_BF16: 14>
    TYPE_BOOL: typing.ClassVar[DataType]  # value = <DataType.TYPE_BOOL: 1>
    TYPE_BYTES: typing.ClassVar[DataType]  # value = <DataType.TYPE_BYTES: 13>
    TYPE_FP16: typing.ClassVar[DataType]  # value = <DataType.TYPE_FP16: 10>
    TYPE_FP32: typing.ClassVar[DataType]  # value = <DataType.TYPE_FP32: 11>
    TYPE_FP64: typing.ClassVar[DataType]  # value = <DataType.TYPE_FP64: 12>
    TYPE_FP8_E4M3: typing.ClassVar[DataType]  # value = <DataType.TYPE_FP8_E4M3: 15>
    TYPE_INT16: typing.ClassVar[DataType]  # value = <DataType.TYPE_INT16: 7>
    TYPE_INT32: typing.ClassVar[DataType]  # value = <DataType.TYPE_INT32: 8>
    TYPE_INT4X2: typing.ClassVar[DataType]  # value = <DataType.TYPE_INT4X2: 19>
    TYPE_INT64: typing.ClassVar[DataType]  # value = <DataType.TYPE_INT64: 9>
    TYPE_INT8: typing.ClassVar[DataType]  # value = <DataType.TYPE_INT8: 6>
    TYPE_INVALID: typing.ClassVar[DataType]  # value = <DataType.TYPE_INVALID: 0>
    TYPE_QFP8_E4M3: typing.ClassVar[DataType]  # value = <DataType.TYPE_QFP8_E4M3: 21>
    TYPE_QINT4X2: typing.ClassVar[DataType]  # value = <DataType.TYPE_QINT4X2: 20>
    TYPE_QINT8: typing.ClassVar[DataType]  # value = <DataType.TYPE_QINT8: 18>
    TYPE_STR: typing.ClassVar[DataType]  # value = <DataType.TYPE_STR: 16>
    TYPE_UINT16: typing.ClassVar[DataType]  # value = <DataType.TYPE_UINT16: 3>
    TYPE_UINT32: typing.ClassVar[DataType]  # value = <DataType.TYPE_UINT32: 4>
    TYPE_UINT64: typing.ClassVar[DataType]  # value = <DataType.TYPE_UINT64: 5>
    TYPE_UINT8: typing.ClassVar[DataType]  # value = <DataType.TYPE_UINT8: 2>
    TYPE_VOID: typing.ClassVar[DataType]  # value = <DataType.TYPE_VOID: 17>
    __members__: typing.ClassVar[dict[str, DataType]]  # value = {'TYPE_INVALID': <DataType.TYPE_INVALID: 0>, 'TYPE_BOOL': <DataType.TYPE_BOOL: 1>, 'TYPE_UINT8': <DataType.TYPE_UINT8: 2>, 'TYPE_UINT16': <DataType.TYPE_UINT16: 3>, 'TYPE_UINT32': <DataType.TYPE_UINT32: 4>, 'TYPE_UINT64': <DataType.TYPE_UINT64: 5>, 'TYPE_INT8': <DataType.TYPE_INT8: 6>, 'TYPE_INT16': <DataType.TYPE_INT16: 7>, 'TYPE_INT32': <DataType.TYPE_INT32: 8>, 'TYPE_INT64': <DataType.TYPE_INT64: 9>, 'TYPE_FP16': <DataType.TYPE_FP16: 10>, 'TYPE_FP32': <DataType.TYPE_FP32: 11>, 'TYPE_FP64': <DataType.TYPE_FP64: 12>, 'TYPE_BYTES': <DataType.TYPE_BYTES: 13>, 'TYPE_BF16': <DataType.TYPE_BF16: 14>, 'TYPE_FP8_E4M3': <DataType.TYPE_FP8_E4M3: 15>, 'TYPE_STR': <DataType.TYPE_STR: 16>, 'TYPE_VOID': <DataType.TYPE_VOID: 17>, 'TYPE_QINT8': <DataType.TYPE_QINT8: 18>, 'TYPE_INT4X2': <DataType.TYPE_INT4X2: 19>, 'TYPE_QINT4X2': <DataType.TYPE_QINT4X2: 20>, 'TYPE_QFP8_E4M3': <DataType.TYPE_QFP8_E4M3: 21>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DeviceResourceConfig:
    device_reserve_memory_bytes: int
    enable_comm_overlap: bool
    enable_layer_micro_batch: int
    host_reserve_memory_bytes: int
    m_split: int
    overlap_comm_type: int
    overlap_math_sm_count: int
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class EPLBConfig:
    balance_method: str
    eplb_balance_layer_per_step: int
    eplb_control_step: int
    eplb_force_repack: int
    eplb_stats_window_size: int
    eplb_test_mode: bool
    eplb_update_time: int
    redundant_expert: int
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def enable_eplb(self) -> bool:
        """
        Get enable_eplb status
        """
    def phy_exp_num(self, expert_num: int) -> int:
        """
        Get physical expert number
        """
    @property
    def eplb_mode(self) -> EplbMode:
        ...
    @eplb_mode.setter
    def eplb_mode(self, arg1: typing.Any) -> None:
        ...
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
    max_batch_tokens_size: int
    max_context_batch_size: int
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class FMHAConfig:
    absorb_opt_len: int
    disable_flash_infer: bool
    enable_fmha: bool
    enable_open_source_fmha: bool
    enable_paged_open_source_fmha: bool
    enable_paged_trt_fmha: bool
    enable_trt_fmha: bool
    enable_trtv1_fmha: bool
    enable_xqa: bool
    use_aiter_pa: bool
    use_asm_pa: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class FMHAType:
    """
    Members:

      FLASH_INFER

      NONE

      OPEN_SOURCE

      PAGED_OPEN_SOURCE

      PAGED_TRT_V2

      TRT_V1

      TRT_V2

      XQA

      AITER_PREFILL

      AITER_ASM_PREFILL

      AITER_DECODE

      AITER_ASM_DECODE

      PY_FLASHINFER_PREFILL

      PY_FLASHINFER_DECODE
    """
    AITER_ASM_DECODE: typing.ClassVar[FMHAType]  # value = <FMHAType.AITER_ASM_DECODE: 11>
    AITER_ASM_PREFILL: typing.ClassVar[FMHAType]  # value = <FMHAType.AITER_ASM_PREFILL: 9>
    AITER_DECODE: typing.ClassVar[FMHAType]  # value = <FMHAType.AITER_DECODE: 10>
    AITER_PREFILL: typing.ClassVar[FMHAType]  # value = <FMHAType.AITER_PREFILL: 8>
    FLASH_INFER: typing.ClassVar[FMHAType]  # value = <FMHAType.FLASH_INFER: 0>
    NONE: typing.ClassVar[FMHAType]  # value = <FMHAType.NONE: 1>
    OPEN_SOURCE: typing.ClassVar[FMHAType]  # value = <FMHAType.OPEN_SOURCE: 2>
    PAGED_OPEN_SOURCE: typing.ClassVar[FMHAType]  # value = <FMHAType.PAGED_OPEN_SOURCE: 3>
    PAGED_TRT_V2: typing.ClassVar[FMHAType]  # value = <FMHAType.PAGED_TRT_V2: 4>
    PY_FLASHINFER_DECODE: typing.ClassVar[FMHAType]  # value = <FMHAType.PY_FLASHINFER_DECODE: 13>
    PY_FLASHINFER_PREFILL: typing.ClassVar[FMHAType]  # value = <FMHAType.PY_FLASHINFER_PREFILL: 12>
    TRT_V1: typing.ClassVar[FMHAType]  # value = <FMHAType.TRT_V1: 5>
    TRT_V2: typing.ClassVar[FMHAType]  # value = <FMHAType.TRT_V2: 6>
    XQA: typing.ClassVar[FMHAType]  # value = <FMHAType.XQA: 7>
    __members__: typing.ClassVar[dict[str, FMHAType]]  # value = {'FLASH_INFER': <FMHAType.FLASH_INFER: 0>, 'NONE': <FMHAType.NONE: 1>, 'OPEN_SOURCE': <FMHAType.OPEN_SOURCE: 2>, 'PAGED_OPEN_SOURCE': <FMHAType.PAGED_OPEN_SOURCE: 3>, 'PAGED_TRT_V2': <FMHAType.PAGED_TRT_V2: 4>, 'TRT_V1': <FMHAType.TRT_V1: 5>, 'TRT_V2': <FMHAType.TRT_V2: 6>, 'XQA': <FMHAType.XQA: 7>, 'AITER_PREFILL': <FMHAType.AITER_PREFILL: 8>, 'AITER_ASM_PREFILL': <FMHAType.AITER_ASM_PREFILL: 9>, 'AITER_DECODE': <FMHAType.AITER_DECODE: 10>, 'AITER_ASM_DECODE': <FMHAType.AITER_ASM_DECODE: 11>, 'PY_FLASHINFER_PREFILL': <FMHAType.PY_FLASHINFER_PREFILL: 12>, 'PY_FLASHINFER_DECODE': <FMHAType.PY_FLASHINFER_DECODE: 13>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class FfnDisAggregateConfig:
    attention_dp_size: int
    attention_tp_size: int
    enable_ffn_disaggregate: bool
    ffn_dp_size: int
    ffn_tp_size: int
    is_ffn_rank: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def is_ffn_service(self) -> bool:
        ...
    def to_string(self) -> str:
        ...
class GrpcConfig:
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, json_str: str) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def from_json(self, arg0: str) -> None:
        """
        Initialize from JSON string
        """
    def get_client_config(self) -> dict[str, int]:
        ...
    def get_server_config(self) -> dict[str, int]:
        ...
    def to_string(self) -> str:
        ...
class HWKernelConfig:
    arm_gemm_use_kai: bool
    decode_capture_batch_sizes: list[int]
    deep_gemm_num_sm: int
    disable_dpc_random: bool
    enable_cuda_graph: bool
    enable_cuda_graph_debug_mode: bool
    enable_multi_block_mode: bool
    enable_native_cuda_graph: bool
    enable_stable_scatter_add: bool
    ft_disable_custom_ar: bool
    num_native_cuda_graph: int
    prefill_capture_seq_lens: list[int]
    rocm_disable_custom_ag: bool
    rocm_hipblaslt_config: str
    use_swizzleA: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class HybridAttentionConfig:
    enable_hybrid_attention: bool
    hybrid_attention_types: list[HybridAttentionType]
    def __init__(self, enable_hybrid_attention: bool = False, hybrid_attention_types: list[HybridAttentionType] = []) -> None:
        ...
    def to_string(self) -> str:
        ...
class HybridAttentionType:
    """
    Members:

      NONE

      LINEAR

      SLIDING_WINDOW
    """
    LINEAR: typing.ClassVar[HybridAttentionType]  # value = <HybridAttentionType.LINEAR: 1>
    NONE: typing.ClassVar[HybridAttentionType]  # value = <HybridAttentionType.NONE: 0>
    SLIDING_WINDOW: typing.ClassVar[HybridAttentionType]  # value = <HybridAttentionType.SLIDING_WINDOW: 2>
    __members__: typing.ClassVar[dict[str, HybridAttentionType]]  # value = {'NONE': <HybridAttentionType.NONE: 0>, 'LINEAR': <HybridAttentionType.LINEAR: 1>, 'SLIDING_WINDOW': <HybridAttentionType.SLIDING_WINDOW: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class KVCacheConfig:
    fp8_kv_cache: int
    int8_kv_cache: int
    kv_cache_mem_mb: int
    max_block_size_per_item: int
    memory_cache_size_mb: int
    memory_cache_sync_timeout_ms: int
    multi_task_prompt: str
    multi_task_prompt_str: str
    multi_task_prompt_tokens: dict[str, list[int]]
    reserve_block_ratio: int
    reuse_cache: bool
    enable_remote_cache: bool
    enable_device_cache: bool
    sync_wait_write: bool
    seq_size_per_block: int
    test_block_num: int
    use_block_cache: int
    enable_memory_cache: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def insertMultiTaskPromptTokens(self, arg0: str, arg1: list[int]) -> None:
        ...
    def to_string(self) -> str:
        ...
class KvCacheDataType:
    """
    Members:

      BASE

      INT8

      FP8
    """
    BASE: typing.ClassVar[KvCacheDataType]  # value = <KvCacheDataType.BASE: 0>
    FP8: typing.ClassVar[KvCacheDataType]  # value = <KvCacheDataType.FP8: 2>
    INT8: typing.ClassVar[KvCacheDataType]  # value = <KvCacheDataType.INT8: 1>
    __members__: typing.ClassVar[dict[str, KvCacheDataType]]  # value = {'BASE': <KvCacheDataType.BASE: 0>, 'INT8': <KvCacheDataType.INT8: 1>, 'FP8': <KvCacheDataType.FP8: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LayerNormType:
    """
    Members:

      pre_layernorm

      post_layernorm

      invalid_type
    """
    __members__: typing.ClassVar[dict[str, LayerNormType]]  # value = {'pre_layernorm': <LayerNormType.pre_layernorm: 0>, 'post_layernorm': <LayerNormType.post_layernorm: 1>, 'invalid_type': <LayerNormType.invalid_type: 2>}
    invalid_type: typing.ClassVar[LayerNormType]  # value = <LayerNormType.invalid_type: 2>
    post_layernorm: typing.ClassVar[LayerNormType]  # value = <LayerNormType.post_layernorm: 1>
    pre_layernorm: typing.ClassVar[LayerNormType]  # value = <LayerNormType.pre_layernorm: 0>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LinearAttentionConfig:
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    def __init__(self, linear_conv_kernel_dim: int = 0, linear_key_head_dim: int = 0, linear_num_key_heads: int = 0, linear_num_value_heads: int = 0, linear_value_head_dim: int = 0) -> None:
        ...
    def to_string(self) -> str:
        ...
class MMModelConfig:
    include_sep_tokens: bool
    is_multimodal: bool
    mm_position_ids_style: int
    mm_sep_tokens: list[list[int]]
    def __init__(self) -> None:
        ...
class MiscellaneousConfig:
    aux_string: str
    disable_pdl: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
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
    def value(self) -> int:
        ...
class ModelConfig:
    add_bias_linear: bool
    attn_config: AttentionConfigs
    ckpt_path: str
    deepseek_mscale_all_dim: float
    deepseek_rope_mscale: float
    embedding_size: int
    eplb_config: EPLBConfig
    expert_num: int
    extra_data_path: str
    has_lm_head: bool
    has_moe_norm: bool
    has_positional_encoding: bool
    has_post_decoder_layernorm: bool
    has_pre_decoder_layernorm: bool
    hidden_size: int
    hybrid_attention_config: HybridAttentionConfig
    input_embedding_scalar: float
    input_vocab_size: int
    layernorm_eps: float
    linear_attention_config: LinearAttentionConfig
    local_extra_data_path: str
    logit_scale: float
    lora_infos: dict[str, str]
    max_seq_len: int
    mm_model_config: MMModelConfig
    model_type: str
    moe_k: int
    moe_layer_index: list[int]
    moe_n_group: int
    moe_normalize_expert_scale: bool
    moe_style: int
    moe_topk_group: int
    num_layers: int
    partial_rotary_factor: float
    position_ids_style: int
    pre_seq_len: int
    prefix_projection: bool
    ptuning_path: str
    qk_norm: bool
    quant_algo: QuantAlgo
    residual_scalar: float
    reverse_e_h_norm: bool
    routed_scaling_factor: float
    scoring_func: int
    special_tokens: SpecialTokens
    tokenizer_path: str
    type_vocab_size: int
    use_attention_linear_bias: bool
    use_fp32_to_compute_logit: bool
    use_kvcache: bool
    use_norm_attn_out_residual: bool
    use_norm_input_residual: bool
    vocab_size: int
    def __init__(self) -> None:
        ...
    def getAttentionConfigs(self, arg0: int) -> AttentionConfigs:
        ...
    def isGatedActivation(self) -> bool:
        ...
    def isKvCacheQuant(self) -> bool:
        ...
    def to_string(self) -> str:
        ...
    @property
    def activation_type(self) -> ActivationType:
        ...
    @activation_type.setter
    def activation_type(self, arg1: typing.Any) -> None:
        ...
    @property
    def data_type(self) -> DataType:
        ...
    @data_type.setter
    def data_type(self, arg1: str) -> None:
        ...
    @property
    def layernorm_type(self) -> LayerNormType:
        ...
    @layernorm_type.setter
    def layernorm_type(self, arg1: str) -> None:
        ...
    @property
    def mla_ops_type(self) -> MlaOpsType:
        ...
    @mla_ops_type.setter
    def mla_ops_type(self, arg1: str) -> None:
        ...
    @property
    def norm_type(self) -> NormType:
        ...
    @norm_type.setter
    def norm_type(self, arg1: str) -> None:
        ...
    @property
    def task_type(self) -> TaskType:
        ...
    @task_type.setter
    def task_type(self, arg1: typing.Any) -> None:
        ...
class ModelSpecificConfig:
    load_python_model: bool
    max_lora_model_size: int
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class MoeConfig:
    deep_ep_num_sm: int
    fake_balance_expert: bool
    hack_moe_expert: bool
    max_moe_normal_masked_token_num: int
    use_all_gather: bool
    use_deepep_internode: bool
    use_deepep_low_latency: bool
    use_deepep_moe: bool
    use_deepep_p2p_low_latency: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class NormType:
    """
    Members:

      layernorm

      rmsnorm

      alphanorm

      add_bias

      invalid_type
    """
    __members__: typing.ClassVar[dict[str, NormType]]  # value = {'layernorm': <NormType.layernorm: 0>, 'rmsnorm': <NormType.rmsnorm: 1>, 'alphanorm': <NormType.alphanorm: 2>, 'add_bias': <NormType.add_bias: 3>, 'invalid_type': <NormType.invalid_type: 4>}
    add_bias: typing.ClassVar[NormType]  # value = <NormType.add_bias: 3>
    alphanorm: typing.ClassVar[NormType]  # value = <NormType.alphanorm: 2>
    invalid_type: typing.ClassVar[NormType]  # value = <NormType.invalid_type: 4>
    layernorm: typing.ClassVar[NormType]  # value = <NormType.layernorm: 0>
    rmsnorm: typing.ClassVar[NormType]  # value = <NormType.rmsnorm: 1>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PDSepConfig:
    cache_store_connect_port: int
    cache_store_listen_port: int
    cache_store_rdma_connect_port: int
    cache_store_rdma_listen_port: int
    cache_store_rdma_mode: bool
    decode_entrance: bool
    decode_polling_call_prefill_ms: int
    decode_polling_kv_cache_step_ms: int
    decode_retry_interval_ms: int
    decode_retry_timeout_ms: int
    decode_retry_times: int
    load_cache_timeout_ms: int
    max_rpc_timeout_ms: int
    prefill_max_wait_timeout_ms: int
    prefill_retry_timeout_ms: int
    prefill_retry_times: int
    rdma_connect_retry_times: int
    remote_rpc_server_port: int
    role_type: RoleType
    worker_port_offset: int
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class ParallelismConfig:
    dp_rank: int
    dp_size: int
    dp_tp_nccl_port: int
    embedding_rpc_server_port: int
    enable_sp: bool
    ep_rank: int
    ep_size: int
    ffn_disaggregate_config: FfnDisAggregateConfig
    ffn_sp_size: int
    ffn_tp_nccl_port: int
    ffn_tp_rank: int
    ffn_tp_size: int
    http_port: int
    local_rank: int
    local_world_size: int
    model_rpc_port: int
    nccl_ip: str
    pp_size: int
    th_nccl_port: int
    tp_nccl_port: int
    tp_rank: int
    tp_size: int
    world_rank: int
    world_size: int
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class ProfilingDebugLoggingConfig:
    check_nan: bool
    debug_load_server: bool
    debug_start_fake_process: bool
    enable_detail_log: bool
    enable_device_perf: bool
    enable_torch_alloc_profile: bool
    ft_alog_conf_path: str
    ft_core_dump_on_exception: bool
    gen_timeline_sync: bool
    hack_layer_num: int
    log_file_backup_count: int
    torch_cuda_profiler_dir: str
    trace_malloc_stack: bool
    trace_memory: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class QuantAlgo:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, method: QuantMethod, bits: int, group_size: int) -> None:
        ...
    def getActivationBits(self) -> int:
        ...
    def getGroupSize(self) -> int:
        ...
    def getQuantMethod(self) -> QuantMethod:
        ...
    def getWeightBits(self) -> int:
        ...
    def isAwq(self) -> bool:
        ...
    def isFp8(self) -> bool:
        ...
    def isFp8PTPC(self) -> bool:
        ...
    def isGptq(self) -> bool:
        ...
    def isGroupwise(self) -> bool:
        ...
    def isOmniQuant(self) -> bool:
        ...
    def isPerTensorQuant(self) -> bool:
        ...
    def isQuant(self) -> bool:
        ...
    def isSmoothQuant(self) -> bool:
        ...
    def isWeightOnlyPerCol(self) -> bool:
        ...
    def setQuantAlgo(self, arg0: str, arg1: int, arg2: int) -> None:
        ...
class QuantMethod:
    """
    Members:

      None

      WeightOnlyPerCol

      GptQ

      Awq

      SmoothQuant

      OmniQuant

      PerTensorQuant

      FP8Quant

      FP8PTPC
    """
    Awq: typing.ClassVar[QuantMethod]  # value = <QuantMethod.Awq: 3>
    FP8PTPC: typing.ClassVar[QuantMethod]  # value = <QuantMethod.FP8PTPC: 8>
    FP8Quant: typing.ClassVar[QuantMethod]  # value = <QuantMethod.FP8Quant: 7>
    GptQ: typing.ClassVar[QuantMethod]  # value = <QuantMethod.GptQ: 2>
    None: typing.ClassVar[QuantMethod]  # value = <QuantMethod.None: 0>
    OmniQuant: typing.ClassVar[QuantMethod]  # value = <QuantMethod.OmniQuant: 5>
    PerTensorQuant: typing.ClassVar[QuantMethod]  # value = <QuantMethod.PerTensorQuant: 6>
    SmoothQuant: typing.ClassVar[QuantMethod]  # value = <QuantMethod.SmoothQuant: 4>
    WeightOnlyPerCol: typing.ClassVar[QuantMethod]  # value = <QuantMethod.WeightOnlyPerCol: 1>
    __members__: typing.ClassVar[dict[str, QuantMethod]]  # value = {'None': <QuantMethod.None: 0>, 'WeightOnlyPerCol': <QuantMethod.WeightOnlyPerCol: 1>, 'GptQ': <QuantMethod.GptQ: 2>, 'Awq': <QuantMethod.Awq: 3>, 'SmoothQuant': <QuantMethod.SmoothQuant: 4>, 'OmniQuant': <QuantMethod.OmniQuant: 5>, 'PerTensorQuant': <QuantMethod.PerTensorQuant: 6>, 'FP8Quant': <QuantMethod.FP8Quant: 7>, 'FP8PTPC': <QuantMethod.FP8PTPC: 8>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
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

      FRONTEND
    """
    DECODE: typing.ClassVar[RoleType]  # value = <RoleType.DECODE: 2>
    FRONTEND: typing.ClassVar[RoleType]  # value = <RoleType.FRONTEND: 4>
    PDFUSION: typing.ClassVar[RoleType]  # value = <RoleType.PDFUSION: 0>
    PREFILL: typing.ClassVar[RoleType]  # value = <RoleType.PREFILL: 1>
    VIT: typing.ClassVar[RoleType]  # value = <RoleType.VIT: 3>
    __members__: typing.ClassVar[dict[str, RoleType]]  # value = {'PDFUSION': <RoleType.PDFUSION: 0>, 'PREFILL': <RoleType.PREFILL: 1>, 'DECODE': <RoleType.DECODE: 2>, 'VIT': <RoleType.VIT: 3>, 'FRONTEND': <RoleType.FRONTEND: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int:
        ...
class RopeConfig:
    dim: int
    extrapolation_factor: float
    factor1: float
    factor2: float
    index_factor: int
    max_pos: int
    mrope_dim1: int
    mrope_dim2: int
    mrope_dim3: int
    mscale: float
    offset: int
    scale: float
    def __init__(self) -> None:
        ...
    @property
    def base(self) -> int:
        ...
    @base.setter
    def base(self, arg1: typing.Any) -> None:
        ...
    @property
    def style(self) -> RopeStyle:
        ...
    @style.setter
    def style(self, arg1: typing.Any) -> None:
        ...
class RopeStyle:
    """
    Members:

      No

      Base

      Glm2

      DynamicNTK

      QwenDynamicNTK

      Yarn

      Llama3

      Mrope
    """
    Base: typing.ClassVar[RopeStyle]  # value = <RopeStyle.Base: 1>
    DynamicNTK: typing.ClassVar[RopeStyle]  # value = <RopeStyle.DynamicNTK: 3>
    Glm2: typing.ClassVar[RopeStyle]  # value = <RopeStyle.Glm2: 2>
    Llama3: typing.ClassVar[RopeStyle]  # value = <RopeStyle.Llama3: 6>
    Mrope: typing.ClassVar[RopeStyle]  # value = <RopeStyle.Mrope: 7>
    No: typing.ClassVar[RopeStyle]  # value = <RopeStyle.No: 0>
    QwenDynamicNTK: typing.ClassVar[RopeStyle]  # value = <RopeStyle.QwenDynamicNTK: 4>
    Yarn: typing.ClassVar[RopeStyle]  # value = <RopeStyle.Yarn: 5>
    __members__: typing.ClassVar[dict[str, RopeStyle]]  # value = {'No': <RopeStyle.No: 0>, 'Base': <RopeStyle.Base: 1>, 'Glm2': <RopeStyle.Glm2: 2>, 'DynamicNTK': <RopeStyle.DynamicNTK: 3>, 'QwenDynamicNTK': <RopeStyle.QwenDynamicNTK: 4>, 'Yarn': <RopeStyle.Yarn: 5>, 'Llama3': <RopeStyle.Llama3: 6>, 'Mrope': <RopeStyle.Mrope: 7>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RuntimeConfig:
    acext_gemm_config_dir: str
    max_block_size_per_item: int
    max_generate_batch_size: int
    model_name: str
    pre_allocate_op_mem: bool
    reserve_runtime_mem_mb: int
    specify_gpu_arch: str
    use_batch_decode_scheduler: bool
    use_gather_batch_scheduler: bool
    warm_up: bool
    warm_up_with_loss: bool
    worker_addrs: list[str]
    worker_grpc_addrs: list[str]
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
    @property
    def batch_decode_scheduler_config(self) -> BatchDecodeSchedulerConfig:
        ...
    @property
    def fifo_scheduler_config(self) -> FIFOSchedulerConfig:
        ...
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
    checkpoint_path: str
    force_score_context_attention: bool
    force_stream_sample: bool
    gen_num_per_cycle: int
    model_type: str
    quantization: str
    sp_max_token_match: int
    sp_min_token_match: int
    tree_decode_config: str
    use_new_sp_engine: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
    @property
    def type(self) -> SpeculativeType:
        ...
    @type.setter
    def type(self, arg1: typing.Any) -> None:
        ...
class SpeculativeType:
    """
    Members:

      NONE

      VANILLA

      MTP

      EAGLE3

      EAGLE

      DETERMINISTIC
    """
    DETERMINISTIC: typing.ClassVar[SpeculativeType]  # value = <SpeculativeType.DETERMINISTIC: 5>
    EAGLE: typing.ClassVar[SpeculativeType]  # value = <SpeculativeType.EAGLE: 4>
    EAGLE3: typing.ClassVar[SpeculativeType]  # value = <SpeculativeType.EAGLE3: 3>
    MTP: typing.ClassVar[SpeculativeType]  # value = <SpeculativeType.MTP: 2>
    NONE: typing.ClassVar[SpeculativeType]  # value = <SpeculativeType.NONE: 0>
    VANILLA: typing.ClassVar[SpeculativeType]  # value = <SpeculativeType.VANILLA: 1>
    __members__: typing.ClassVar[dict[str, SpeculativeType]]  # value = {'NONE': <SpeculativeType.NONE: 0>, 'VANILLA': <SpeculativeType.VANILLA: 1>, 'MTP': <SpeculativeType.MTP: 2>, 'EAGLE3': <SpeculativeType.EAGLE3: 3>, 'EAGLE': <SpeculativeType.EAGLE: 4>, 'DETERMINISTIC': <SpeculativeType.DETERMINISTIC: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TaskType:
    """
    Members:

      DENSE_EMBEDDING

      ALL_EMBEDDING

      SPARSE_EMBEDDING

      COLBERT_EMBEDDING

      LANGUAGE_MODEL

      SEQ_CLASSIFICATION

      RERANKER

      LINEAR_SOFTMAX

      BGE_M3
    """
    ALL_EMBEDDING: typing.ClassVar[TaskType]  # value = <TaskType.ALL_EMBEDDING: 1>
    BGE_M3: typing.ClassVar[TaskType]  # value = <TaskType.BGE_M3: 8>
    COLBERT_EMBEDDING: typing.ClassVar[TaskType]  # value = <TaskType.COLBERT_EMBEDDING: 3>
    DENSE_EMBEDDING: typing.ClassVar[TaskType]  # value = <TaskType.DENSE_EMBEDDING: 0>
    LANGUAGE_MODEL: typing.ClassVar[TaskType]  # value = <TaskType.LANGUAGE_MODEL: 4>
    LINEAR_SOFTMAX: typing.ClassVar[TaskType]  # value = <TaskType.LINEAR_SOFTMAX: 7>
    RERANKER: typing.ClassVar[TaskType]  # value = <TaskType.RERANKER: 6>
    SEQ_CLASSIFICATION: typing.ClassVar[TaskType]  # value = <TaskType.SEQ_CLASSIFICATION: 5>
    SPARSE_EMBEDDING: typing.ClassVar[TaskType]  # value = <TaskType.SPARSE_EMBEDDING: 2>
    __members__: typing.ClassVar[dict[str, TaskType]]  # value = {'DENSE_EMBEDDING': <TaskType.DENSE_EMBEDDING: 0>, 'ALL_EMBEDDING': <TaskType.ALL_EMBEDDING: 1>, 'SPARSE_EMBEDDING': <TaskType.SPARSE_EMBEDDING: 2>, 'COLBERT_EMBEDDING': <TaskType.COLBERT_EMBEDDING: 3>, 'LANGUAGE_MODEL': <TaskType.LANGUAGE_MODEL: 4>, 'SEQ_CLASSIFICATION': <TaskType.SEQ_CLASSIFICATION: 5>, 'RERANKER': <TaskType.RERANKER: 6>, 'LINEAR_SOFTMAX': <TaskType.LINEAR_SOFTMAX: 7>, 'BGE_M3': <TaskType.BGE_M3: 8>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class VitConfig:
    vit_separation: VitSeparation
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def to_string(self) -> str:
        ...
class VitSeparation:
    """
    Members:

      VIT_SEPARATION_LOCAL

      VIT_SEPARATION_ROLE

      VIT_SEPARATION_REMOTE
    """
    VIT_SEPARATION_LOCAL: typing.ClassVar[VitSeparation]  # value = <VitSeparation.VIT_SEPARATION_LOCAL: 0>
    VIT_SEPARATION_REMOTE: typing.ClassVar[VitSeparation]  # value = <VitSeparation.VIT_SEPARATION_REMOTE: 2>
    VIT_SEPARATION_ROLE: typing.ClassVar[VitSeparation]  # value = <VitSeparation.VIT_SEPARATION_ROLE: 1>
    __members__: typing.ClassVar[dict[str, VitSeparation]]  # value = {'VIT_SEPARATION_LOCAL': <VitSeparation.VIT_SEPARATION_LOCAL: 0>, 'VIT_SEPARATION_ROLE': <VitSeparation.VIT_SEPARATION_ROLE: 1>, 'VIT_SEPARATION_REMOTE': <VitSeparation.VIT_SEPARATION_REMOTE: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def get_block_cache_keys(token_ids_list: list[list[int]]) -> list[int]:
    ...
