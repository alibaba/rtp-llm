from __future__ import annotations
import torch
import typing
__all__ = ['DeviceExporter', 'DeviceType', 'EmbeddingHandlerOp', 'EngineScheduleInfo', 'EngineTaskInfo', 'GptInitParameter', 'LoadBalanceInfo', 'MultimodalInput', 'QuantAlgo', 'RoleSpecialTokens', 'RtpEmbeddingOp', 'RtpLLMOp', 'SpecialTokens', 'create_linear_softmax_handler', 'get_device']
class DeviceExporter:
    def get_device_id(self) -> int:
        ...
    def get_device_type(self) -> DeviceType:
        ...
    def pack_int8_tensor_to_packed_int4(self, arg0: torch.Tensor) -> torch.Tensor:
        ...
    def preprocess_gemm_weight_by_key(self, arg0: str, arg1: torch.Tensor) -> torch.Tensor:
        ...
    def preprocess_weights_for_mixed_gemm(self, arg0: torch.Tensor, arg1: typing.Any) -> torch.Tensor:
        ...
    def symmetric_quantize_last_axis_of_batched_matrix(self, arg0: torch.Tensor, arg1: typing.Any) -> list[torch.Tensor]:
        ...
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
    __members__: typing.ClassVar[dict[str, DeviceType]]  # value = {'Cpu': <DeviceType.Cpu: 0>, 'Cuda': <DeviceType.Cuda: 1>, 'Yitian': <DeviceType.Yitian: 2>, 'ArmCpu': <DeviceType.ArmCpu: 3>, 'ROCm': <DeviceType.ROCm: 4>, 'Ppu': <DeviceType.Ppu: 5>}
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
class EmbeddingHandlerOp:
    def __init__(self) -> None:
        ...
    def forward(self, arg0: torch.Tensor, arg1: torch.Tensor) -> torch.Tensor:
        ...
    def load_tensor(self, arg0: dict[str, torch.Tensor]) -> None:
        ...
class EngineScheduleInfo:
    finished_task_info_list: list[EngineTaskInfo]
    last_schedule_delta: int
    running_task_info_list: list[EngineTaskInfo]
    def __init__(self) -> None:
        ...
class EngineTaskInfo:
    input_length: int
    prefix_length: int
    request_id: int
    def __init__(self) -> None:
        ...
class GptInitParameter:
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
    decode_polling_kv_cache_step_ms: int
    decode_retry_timeout_ms: int
    decode_retry_times: int
    decode_use_async_load_cache: bool
    enable_fast_gen: bool
    enable_partial_fallback: bool
    ep_rank: int
    ep_size: int
    expert_num: int
    fast_gen_max_context_len: int
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
    load_balance_policy_name: str
    load_cache_timeout_ms: int
    local_rank: int
    logit_scale: float
    max_context_batch_size: int
    max_generate_batch_size: int
    max_rpc_timeout_ms: int
    max_seq_len: int
    mm_position_ids_style: int
    mm_sep_tokens: list[list[int]]
    model_rpc_port: int
    moe_inter_padding_size: int
    moe_k: int
    moe_layer_index: list[int]
    moe_normalize_expert_scale: bool
    moe_style: int
    mrope_section: list[int]
    nccl_ip: str
    nccl_port: int
    nope_head_dim: int
    norm_type: str
    num_layers: int
    num_valid_layer: int
    org_embedding_max_pos: int
    pd_sep_enable_fallback: bool
    pd_separation: bool
    position_id_len_factor: int
    position_ids_style: int
    pre_allocate_op_mem: bool
    pre_seq_len: int
    prefill_retry_timeout_ms: int
    prefill_retry_times: int
    prefix_projection: bool
    q_lora_rank: int
    q_scaling: float
    qk_norm: bool
    quant_algo: QuantAlgo
    rdma_connect_retry_times: int
    remote_rpc_server_port: int
    reserve_runtime_mem_mb: int
    residual_scalar: float
    reuse_cache: bool
    rope_head_dim: int
    rotary_embedding_base: float
    rotary_embedding_dim: int
    rotary_embedding_mscale: float
    rotary_embedding_offset: int
    rotary_embedding_scale: float
    rotary_embedding_style: int
    rotary_factor1: float
    rotary_factor2: float
    scheduler_reserve_resource_ratio: int
    seq_size_per_block: int
    size_per_head: int
    softmax_extra_scale: float
    special_tokens: SpecialTokens
    tokenizer_path: str
    tp_rank: int
    tp_size: int
    type_vocab_size: int
    use_attention_linear_bias: bool
    use_cache_store: bool
    use_cross_attn: bool
    use_expert_attention: bool
    use_fp32_to_compute_logit: bool
    use_kvcache: bool
    use_logn_attn: bool
    use_medusa: bool
    use_mla: bool
    use_norm_attn_out_residual: bool
    use_norm_input_residual: bool
    using_hf_sampling: bool
    v_head_dim: int
    vocab_size: int
    warm_up: bool
    warm_up_with_loss: bool
    worker_port_offset: int
    worker_addrs: list[str]
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int, arg4: int, arg5: int) -> None:
        ...
    def insertMultiTaskPromptTokens(self, arg0: str, arg1: list[int]) -> None:
        ...
    def isGatedActivation(self) -> bool:
        ...
    def isKvCacheQuant(self) -> bool:
        ...
    def setActivationType(self) -> None:
        ...
    def setKvCacheDataType(self) -> None:
        ...
    def setLayerNormType(self) -> None:
        ...
    def setNormType(self) -> None:
        ...
    def setTaskType(self, arg0: str) -> None:
        ...
class LoadBalanceInfo:
    available_kv_cache: int
    iterate_count: int
    step_latency_us: int
    step_per_minute: int
    total_kv_cache: int
    def __init__(self) -> None:
        ...
class MultimodalInput:
    mm_type: int
    tensor: torch.Tensor
    url: str
    def __init__(self, arg0: str, arg1: torch.Tensor, arg2: int) -> None:
        ...
class QuantAlgo:
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def getGroupSize(self) -> int:
        ...
    def getWeightBits(self) -> int:
        ...
    def getActivationBits(self) -> int:
        ...
    def isAwq(self) -> bool:
        ...
    def isFp8(self) -> bool:
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
class RoleSpecialTokens:
    eos_token_ids: list[int]
    token_ids: list[int]
    def __init__(self) -> None:
        ...
class RtpEmbeddingOp:
    def __init__(self) -> None:
        ...
    def decode(self, arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: int, arg4: list[MultimodalInput]) -> typing.Any:
        ...
    def init(self, arg0: typing.Any, arg1: typing.Any) -> None:
        ...
    def stop(self) -> None:
        ...
class RtpLLMOp:
    def __init__(self) -> None:
        ...
    def add_lora(self, arg0: str, arg1: typing.Any, arg2: typing.Any) -> None:
        ...
    def get_engine_schedule_info(self) -> EngineScheduleInfo:
        ...
    def get_load_balance_info(self) -> LoadBalanceInfo:
        ...
    def init(self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any, arg3: typing.Any) -> None:
        ...
    def ready(self) -> bool:
        ...
    def remove_lora(self, arg0: str) -> None:
        ...
    def start_http_server(self, arg0: typing.Any, arg1: typing.Any) -> None:
        ...
    def stop(self) -> None:
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
    def __init__(self) -> None:
        ...
def create_linear_softmax_handler(arg0: GptInitParameter) -> EmbeddingHandlerOp:
    ...
def get_device() -> DeviceExporter:
    ...
