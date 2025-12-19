"""
rtp llm custom ops
"""
from __future__ import annotations
import librtp_compute_ops
import libth_transformer_config
import torch
import typing
__all__: list[str] = ['FlashInferAttnParams', 'FlashInferDecodeOp', 'FlashInferPrefillOp', 'FusedMoEOp', 'FusedRopeKVCacheDecodeOp', 'FusedRopeKVCachePrefillOp', 'GroupTopKOp', 'KVBlockArray', 'SelectTopkOp', 'TRTAttn', 'cuda_graph_copy_large2small', 'cuda_graph_copy_small2large', 'embedding', 'embedding_bert', 'fused_add_layernorm', 'fused_add_rmsnorm', 'fused_qk_rmsnorm', 'layernorm', 'mla_k_merge', 'moe_topk_softmax', 'per_token_group_quant_fp8', 'per_token_group_quant_fp8_v2', 'per_token_group_quant_int8', 'reuse_kv_cache_indexed_batched', 'rmsnorm', 'silu_and_mul', 'trt_fp8_quantize_128', 'trt_fp8_quantize_128_inplace', 'write_cache_store']
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
    def forward(self, qkv: torch.Tensor, fmha_type: libth_transformer_config.FMHAType, kv_cache: librtp_compute_ops.KVCache | None, params: TRTAttn) -> torch.Tensor:
        ...
    def prepare(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> TRTAttn:
        ...
class FusedRopeKVCachePrefillOp:
    def __init__(self, attn_configs: libth_transformer_config.AttentionConfigs) -> None:
        ...
    def forward(self, qkv: torch.Tensor, fmha_type: libth_transformer_config.FMHAType, kv_cache: librtp_compute_ops.KVCache | None, params: TRTAttn) -> torch.Tensor:
        ...
    def prepare(self, attn_inputs: librtp_compute_ops.PyAttentionInputs) -> TRTAttn:
        ...
class GroupTopKOp:
    def __init__(self) -> None:
        ...
    def forward(self, topk_values: torch.Tensor, topk_indices: torch.Tensor, scores: torch.Tensor, scores_with_bias: torch.Tensor, n_group: int, topk_group: int, topk: int, renormalize: bool, routed_scaling_factor: float) -> None:
        ...
class KVBlockArray:
    def __init__(self) -> None:
        ...
class SelectTopkOp:
    def __init__(self, model_config: libth_transformer_config.ModelConfig, fake_balance_expert: bool, dp_rank: int) -> None:
        ...
    def forward(self, router_logits: torch.Tensor, expert_ids: torch.Tensor, expert_scales: torch.Tensor) -> None:
        ...
class TRTAttn(librtp_compute_ops.ParamsBase):
    def __init__(self) -> None:
        ...
def cuda_graph_copy_large2small(input_tensor: torch.Tensor, output_tensor: torch.Tensor, batch_size: torch.Tensor, max_batch_size: int, max_seq_len: int, input_lengths: torch.Tensor, hidden_size: int, cu_seq_len: torch.Tensor) -> None:
    """
    CUDA Graph copy kernel: Large to Small tensor copy
    """
def cuda_graph_copy_small2large(input_tensor: torch.Tensor, output_tensor: torch.Tensor, batch_size: torch.Tensor, max_batch_size: int, max_seq_len: int, input_lengths: torch.Tensor, hidden_size: int, cu_seq_len: torch.Tensor) -> None:
    """
    CUDA Graph copy kernel: Small to Large tensor copy
    """
def embedding(output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor) -> None:
    """
    Embedding lookup kernel
    """
def embedding_bert(output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, combo_position_ids: torch.Tensor, position_encoding: torch.Tensor, combo_tokens_type_ids: torch.Tensor, token_type_embedding: torch.Tensor, input_embedding_scalar: float = 1.0) -> None:
    """
    EmbeddingBert lookup kernel
    """
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
def layernorm(output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, beta: torch.Tensor, eps: float) -> None:
    """
    LayerNorm kernel
    """
def mla_k_merge(k_out: torch.Tensor, k_nope: torch.Tensor, k_pe: torch.Tensor) -> None:
    """
    Fused kernel to merge k_nope and k_pe efficiently
    """
def moe_topk_softmax(topk_weights: torch.Tensor, topk_indices: torch.Tensor, token_expert_indices: torch.Tensor, gating_output: torch.Tensor) -> None:
    """
    MoE Topk Softmax kernel
    """
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
def write_cache_store(input_lengths: torch.Tensor, prefix_lengths: torch.Tensor, kv_cache_block_id_host: torch.Tensor, cache_store_member: librtp_compute_ops.PyCacheStoreInputs | None, kv_cache: librtp_compute_ops.KVCache | None) -> None:
    """
    WriteCacheStoreOp kernel
    """
