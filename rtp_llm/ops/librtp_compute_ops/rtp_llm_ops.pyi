"""
rtp llm custom ops
"""
from __future__ import annotations
import librtp_compute_ops
import libth_transformer_config
import torch
import typing
__all__: list[str] = ['FlashInferAttnParams', 'FlashInferDecodeOp', 'FlashInferMlaAttnParams', 'FlashInferPrefillOp', 'FusedMoEOp', 'FusedRopeKVCacheDecodeOp', 'FusedRopeKVCachePrefillOpQKVOut', 'FusedRopeKVCachePrefillOpQOut', 'GroupTopKOp', 'KVBlockArray', 'SelectTopkOp', 'TRTAttn', 'TRTAttnOp', 'TRTPagedAttnOp', 'XQAAttnOp', 'XQAParams', 'allocate_shared_buffer', 'cuda_graph_copy_large2small', 'cuda_graph_copy_small2large', 'cutlass_moe_mm', 'debug_kernel', 'dispose_communicator', 'embedding', 'embedding_bert', 'fill_mla_params', 'fused_add_layernorm', 'fused_add_rmsnorm', 'fused_qk_rmsnorm', 'get_cutlass_batched_moe_mm_data', 'get_cutlass_moe_mm_without_permute_info', 'init_communicator', 'layernorm', 'mla_k_merge', 'moe_post_reorder', 'moe_pre_reorder', 'moe_topk_softmax', 'open_ipc_handle', 'per_tensor_quant_fp8', 'per_token_group_quant_fp8', 'per_token_group_quant_fp8_v2', 'per_token_group_quant_int8', 'per_token_quant_fp8', 'register_buffer_to_communicator', 'reuse_kv_cache_indexed_batched', 'rmsnorm', 'silu_and_mul', 'trt_fp8_quantize_128', 'trt_fp8_quantize_128_inplace', 'userbuffers_recv', 'userbuffers_ring_all_gather', 'userbuffers_send', 'write_cache_store']
class FlashInferAttnParams(librtp_compute_ops.ParamsBase):
    def __init__(self) -> None:
        ...
class FlashInferDecodeOp:
    def __init__(self, attn_configs: libth_transformer_config.AttentionConfigs) -> None:
        ...
    def forward(self, q: torch.Tensor, kv_cache: librtp_compute_ops.KVCache | None, params: FlashInferAttnParams) -> torch.Tensor:
        ...
    def prepare(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> librtp_compute_ops.ParamsBase:
        ...
    def support(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> bool:
        ...
class FlashInferMlaAttnParams(librtp_compute_ops.ParamsBase):
    def __init__(self) -> None:
        ...
    def fill_params(self, prefix_lengths: torch.Tensor, sequence_lengths: torch.Tensor, input_lengths: torch.Tensor, kv_cache_block_id_host: torch.Tensor, seq_size_per_block: int) -> None:
        """
        Fill parameters for CUDA graph execution
        """
    @property
    def batch_indice_d(self) -> torch.Tensor:
        """
        Batch indices on DEVICE
        """
    @property
    def batch_indice_h(self) -> torch.Tensor:
        """
        Batch indices on HOST
        """
    @property
    def batch_reuse_info_vec_d(self) -> torch.Tensor:
        """
        Batch reuse info vector on DEVICE
        """
    @property
    def batch_reuse_info_vec_h(self) -> torch.Tensor:
        """
        Batch reuse info vector on HOST
        """
    @property
    def decode_page_indptr_d(self) -> torch.Tensor:
        """
        Decode page indptr on DEVICE
        """
    @property
    def decode_page_indptr_h(self) -> torch.Tensor:
        """
        Decode page indptr on HOST
        """
    @property
    def kvlen_d(self) -> torch.Tensor:
        """
        KV length on DEVICE
        """
    @property
    def kvlen_h(self) -> torch.Tensor:
        """
        KV length on HOST
        """
    @property
    def page_indice_d(self) -> torch.Tensor:
        """
        Page indices on DEVICE
        """
    @property
    def page_indice_h(self) -> torch.Tensor:
        """
        Page indices on HOST
        """
    @property
    def paged_kv_last_page_len_d(self) -> torch.Tensor:
        """
        Paged KV last page length on DEVICE
        """
    @property
    def paged_kv_last_page_len_h(self) -> torch.Tensor:
        """
        Paged KV last page length on HOST
        """
    @property
    def positions_d(self) -> torch.Tensor:
        """
        Positions on DEVICE
        """
    @property
    def positions_h(self) -> torch.Tensor:
        """
        Positions on HOST
        """
    @property
    def prefill_ragged_kv_len_indptr_d(self) -> torch.Tensor:
        """
        Prefill page indptr on DEVICE
        """
    @property
    def prefill_ragged_kv_len_indptr_h(self) -> torch.Tensor:
        """
        Prefill page indptr on HOST
        """
    @property
    def qo_indptr_d(self) -> torch.Tensor:
        """
        Query/output indptr on DEVICE
        """
    @property
    def qo_indptr_h(self) -> torch.Tensor:
        """
        Query/output indptr on HOST
        """
    @property
    def reuse_cache_page_indice_d(self) -> torch.Tensor:
        """
        Reuse cache page indices on DEVICE
        """
    @property
    def reuse_cache_page_indice_h(self) -> torch.Tensor:
        """
        Reuse cache page indices on HOST
        """
class FlashInferPrefillOp:
    def __init__(self, attn_configs: libth_transformer_config.AttentionConfigs) -> None:
        ...
    def forward(self, q: torch.Tensor, kv_cache: librtp_compute_ops.KVCache | None, params: FlashInferAttnParams) -> torch.Tensor:
        ...
    def prepare(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> librtp_compute_ops.ParamsBase:
        ...
    def support(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> bool:
        ...
class FusedMoEOp:
    def __init__(self, model_config: libth_transformer_config.ModelConfig, parallelism_config: libth_transformer_config.ParallelismConfig) -> None:
        ...
    def forward(self, hidden_states: torch.Tensor, up_proj: torch.Tensor, down_proj: torch.Tensor, expert_scales: torch.Tensor, expert_ids: torch.Tensor, outputs: torch.Tensor) -> None:
        ...
class FusedRopeKVCacheDecodeOp:
    def __init__(self, attn_configs: libth_transformer_config.AttentionConfigs) -> None:
        ...
    def forward(self, qkv: torch.Tensor, kv_cache: librtp_compute_ops.KVCache | None, params: TRTAttn) -> torch.Tensor:
        ...
    def prepare(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> TRTAttn:
        ...
class FusedRopeKVCachePrefillOpQKVOut:
    def __init__(self, attn_configs: libth_transformer_config.AttentionConfigs) -> None:
        ...
    def forward(self, qkv: torch.Tensor, kv_cache: librtp_compute_ops.KVCache | None, params: TRTAttn) -> torch.Tensor:
        ...
    def prepare(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> TRTAttn:
        ...
class FusedRopeKVCachePrefillOpQOut:
    def __init__(self, attn_configs: libth_transformer_config.AttentionConfigs) -> None:
        ...
    def forward(self, qkv: torch.Tensor, kv_cache: librtp_compute_ops.KVCache | None, params: TRTAttn) -> torch.Tensor:
        ...
    def prepare(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> TRTAttn:
        ...
class GroupTopKOp:
    def __init__(self) -> None:
        ...
    def forward(self, topk_values: torch.Tensor, topk_indices: torch.Tensor, scores: torch.Tensor, scores_with_bias: torch.Tensor, n_group: int, topk_group: int, topk: int, renormalize: bool, routed_scaling_factor: float) -> None:
        ...
class KVBlockArray:
    def __cpp_ptr__(self) -> int:
        """
        Get C++ object pointer address
        """
    def __init__(self) -> None:
        ...
class SelectTopkOp:
    def __init__(self, model_config: libth_transformer_config.ModelConfig, fake_balance_expert: bool, dp_rank: int) -> None:
        ...
    def forward(self, router_logits: torch.Tensor, expert_ids: torch.Tensor, expert_scales: torch.Tensor) -> None:
        ...
class TRTAttn(librtp_compute_ops.ParamsBase):
    kv_cache_offset: torch.Tensor
    def __cpp_ptr__(self) -> int:
        """
        Get C++ object pointer address
        """
    def __init__(self) -> None:
        ...
class TRTAttnOp:
    def __init__(self, attn_configs: libth_transformer_config.AttentionConfigs) -> None:
        ...
    def forward(self, input: torch.Tensor, kv_cache: librtp_compute_ops.KVCache | None, params: TRTAttn) -> torch.Tensor:
        ...
    def prepare(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> librtp_compute_ops.ParamsBase:
        ...
    def support(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> bool:
        ...
class TRTPagedAttnOp:
    def __init__(self, attn_configs: libth_transformer_config.AttentionConfigs) -> None:
        ...
    def forward(self, input: torch.Tensor, kv_cache: librtp_compute_ops.KVCache | None, params: TRTAttn) -> torch.Tensor:
        ...
    def prepare(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> librtp_compute_ops.ParamsBase:
        ...
    def support(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> bool:
        ...
class XQAAttnOp:
    def __init__(self, attn_configs: libth_transformer_config.AttentionConfigs) -> None:
        ...
    def forward(self, input: torch.Tensor, kv_cache: librtp_compute_ops.KVCache | None, params: XQAParams) -> torch.Tensor:
        ...
    def prepare(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> librtp_compute_ops.ParamsBase:
        ...
    def support(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> bool:
        ...
class XQAParams(librtp_compute_ops.ParamsBase):
    kv_cache_offset: torch.Tensor
    def __cpp_ptr__(self) -> int:
        """
        Get C++ object pointer address
        """
    def __init__(self) -> None:
        ...
def allocate_shared_buffer(size: int) -> tuple[int, torch.Tensor]:
    """
    Allocate shared CUDA buffer with IPC handle for inter-process communication
    """
def cuda_graph_copy_large2small(input_tensor: torch.Tensor, output_tensor: torch.Tensor, batch_size: torch.Tensor, max_batch_size: int, max_seq_len: int, input_lengths: torch.Tensor, hidden_size: int, cu_seq_len: torch.Tensor) -> None:
    """
    CUDA Graph copy kernel: Large to Small tensor copy
    """
def cuda_graph_copy_small2large(input_tensor: torch.Tensor, output_tensor: torch.Tensor, batch_size: torch.Tensor, max_batch_size: int, max_seq_len: int, input_lengths: torch.Tensor, hidden_size: int, cu_seq_len: torch.Tensor) -> None:
    """
    CUDA Graph copy kernel: Small to Large tensor copy
    """
def cutlass_moe_mm(out_tensors: torch.Tensor, a_tensors: torch.Tensor, b_tensors: torch.Tensor, a_scales: torch.Tensor, b_scales: torch.Tensor, expert_offsets: torch.Tensor, problem_sizes: torch.Tensor, a_strides: torch.Tensor, b_strides: torch.Tensor, c_strides: torch.Tensor, per_act_token: bool, per_out_ch: bool, profile: bool = False, m_tile: int = 0, n_tile: int = 0, k_tile: int = 0, cluster_m: int = 0, cluster_n: int = 0, cluster_k: int = 0, swap_ab: bool = False) -> None:
    ...
def debug_kernel(data: torch.Tensor, start_row: int, start_col: int, m: int, n: int, row_len: int, info_id: int) -> None:
    """
    Debug kernel to print 2D data blocks from GPU tensor
    """
def dispose_communicator(comm_ptr: int) -> None:
    """
    Dispose UbCommunicator with python address and release resources
    """
def embedding(output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor) -> None:
    """
    Embedding lookup kernel
    """
def embedding_bert(output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, combo_position_ids: torch.Tensor, position_encoding: torch.Tensor, combo_tokens_type_ids: torch.Tensor, token_type_embedding: torch.Tensor, input_embedding_scalar: float = 1.0) -> None:
    """
    EmbeddingBert lookup kernel
    """
def fill_mla_params(t_prefill_lengths: torch.Tensor, t_sequence_lengths: torch.Tensor, t_input_lengths: torch.Tensor, t_kv_cache_block_id_host: torch.Tensor, seq_size_per_block: int) -> FlashInferMlaAttnParams:
    ...
def fused_add_layernorm(input: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor, weight: torch.Tensor, beta: torch.Tensor, eps: float) -> None:
    """
    Fused Add LayerNorm kernel
    """
def fused_add_rmsnorm(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, cuda_stream: int = 0) -> None:
    """
    Fused Add RMSNorm kernel
    """
def fused_qk_rmsnorm(IO: torch.Tensor, q_gamma: torch.Tensor, k_gamma: torch.Tensor, layernorm_eps: float, q_group_num: int, k_group_num: int, m: int, n: int, norm_size: int) -> None:
    """
    Fused QK RMSNorm kernel
    """
def get_cutlass_batched_moe_mm_data(expert_offsets: torch.Tensor, problem_sizes1: torch.Tensor, problem_sizes2: torch.Tensor, expert_num_tokens: torch.Tensor, num_local_experts: int, padded_m: int, n: int, k: int, problem_1_swap_ab: bool, problem_2_swap_ab: bool) -> None:
    ...
def get_cutlass_moe_mm_without_permute_info(topk_ids: torch.Tensor, expert_offsets: torch.Tensor, problem_sizes1: torch.Tensor, problem_sizes2: torch.Tensor, num_experts: int, n: int, k: int, problem_1_swap_ab: bool, problem_2_swap_ab: bool, blockscale_offsets: torch.Tensor | None = None) -> None:
    ...
def init_communicator(local_rank: int, world_size: int) -> int:
    """
    Initialize UbCommunicator with IPC pointers from remote processes
    """
def layernorm(output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, beta: torch.Tensor, eps: float) -> None:
    """
    LayerNorm kernel
    """
def mla_k_merge(k_out: torch.Tensor, k_nope: torch.Tensor, k_pe: torch.Tensor) -> None:
    """
    Fused kernel to merge k_nope and k_pe efficiently
    """
def moe_post_reorder(permuted_hidden_states: torch.Tensor, topk_weights: torch.Tensor, inv_permuted_idx: torch.Tensor, expert_first_token_offset: torch.Tensor | None = None, topk: int, hidden_states: torch.Tensor) -> None:
    """
    moe ep unpermute kernel
    """
def moe_pre_reorder(input: torch.Tensor, topk_ids: torch.Tensor, token_expert_indices: torch.Tensor, expert_map: torch.Tensor | None = None, n_expert: int, n_local_expert: int, topk: int, align_block_size: int | None = None, permuted_input: torch.Tensor, expert_first_token_offset: torch.Tensor, inv_permuted_idx: torch.Tensor, permuted_idx: torch.Tensor) -> None:
    """
    moe ep permute kernel
    """
def moe_topk_softmax(topk_weights: torch.Tensor, topk_indices: torch.Tensor, token_expert_indices: torch.Tensor, gating_output: torch.Tensor) -> None:
    """
    MoE Topk Softmax kernel
    """
def open_ipc_handle(mem_handle: torch.Tensor) -> int:
    """
    Open IPC memory handle to access shared buffer from another process
    """
def per_tensor_quant_fp8(input: torch.Tensor, output_q: torch.Tensor, output_s: torch.Tensor, is_static: bool) -> None:
    ...
def per_token_group_quant_fp8(input: torch.Tensor, output_q: torch.Tensor, output_s: torch.Tensor, group_size: int, eps: float, fp8_min: float, fp8_max: float, scale_ue8m0: bool) -> None:
    """
    Fp8 Gemm Per Token Group
    """
def per_token_group_quant_fp8_v2(input: torch.Tensor, output_q: torch.Tensor, output_s: torch.Tensor, group_size: int, eps: float, fp8_min: float, fp8_max: float, scale_ue8m0: bool, fuse_silu_and_mul: bool, masked_m: torch.Tensor | None) -> None:
    """
    Fp8 Gemm Per Token Group
    """
def per_token_group_quant_int8(input: torch.Tensor, output_q: torch.Tensor, output_s: torch.Tensor, group_size: int, eps: float, int8_min: float, int8_max: float, scale_ue8m0: bool) -> None:
    """
    Int8 Gemm Per Token Group
    """
def per_token_quant_fp8(input: torch.Tensor, output_q: torch.Tensor, output_s: torch.Tensor) -> None:
    ...
def register_buffer_to_communicator(comm_ptr: int, buffer_ptrs: list[int]) -> int:
    """
    Register buffers to communicator for inter-process communication
    """
def reuse_kv_cache_indexed_batched(final_compressed_kv: torch.Tensor, final_k_pe: torch.Tensor, compressed_kv: torch.Tensor, k_pe: torch.Tensor, kv_cache_base: torch.Tensor, reuse_cache_page_indice: torch.Tensor, batch_reuse_info_vec: torch.Tensor, qo_indptr: torch.Tensor, tokens_per_block: int) -> None:
    """
    Reuse KV cache indexed batched kernel
    """
def rmsnorm(output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float, cuda_stream: int = 0) -> None:
    """
    RMSNorm kernel
    """
def silu_and_mul(output: torch.Tensor, input: torch.Tensor, cuda_stream: int = 0) -> None:
    """
    SiLU and Multiply kernel
    """
def trt_fp8_quantize_128(input: torch.Tensor, col_major_scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize BF16 weight matrix to FP8 format using 128-element block processing
    """
def trt_fp8_quantize_128_inplace(input: torch.Tensor, output_q: torch.Tensor, output_s: torch.Tensor, col_major_scale: bool = False) -> None:
    """
    Quantize BF16 weight matrix to FP8 format using 128-element block processing (in-place version)
    """
def userbuffers_recv(tensor: torch.Tensor, handler: int, srcoffset: int, dstoffset: int, comm_ptr: int, peer: int, stream: int) -> None:
    """
    Receive data via user buffers from peer
    """
def userbuffers_ring_all_gather(all_gather_tensor: torch.Tensor, tensor: torch.Tensor, handler: int, rank_offsets: list[int], comm_ptr: int, send_stream_ids: list[int], recv_stream: int) -> torch.Tensor:
    """
    Ring all-gather operation via user buffers
    """
def userbuffers_send(tensor: torch.Tensor, handler: int, srcoffset: int, dstoffset: int, bytes: int, comm_ptr: int, peer: int, stream: int) -> None:
    """
    Send data via user buffers to peer
    """
def write_cache_store(input_lengths: torch.Tensor, prefix_lengths: torch.Tensor, kv_cache_block_id_host: torch.Tensor, cache_store_member: librtp_compute_ops.PyCacheStoreInputs | None, kv_cache: librtp_compute_ops.KVCache | None) -> None:
    """
    WriteCacheStoreOp kernel
    """
