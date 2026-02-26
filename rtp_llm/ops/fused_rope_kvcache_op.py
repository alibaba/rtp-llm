from dataclasses import dataclass
from typing import Optional

import rtp_kernel
import torch
from librtp_compute_ops import KVCache, PyAttentionInputs
from libth_transformer_config import AttentionConfigs, FMHAType, KvCacheDataType

from .rope_cache import get_rope_cache


@dataclass
class TRTAttn:
    kv_cache_offset: torch.Tensor
    kv_cache_offset_h: Optional[torch.Tensor]
    padding_offset: Optional[torch.Tensor]
    cu_seqlens: torch.Tensor
    cu_kv_seqlens: torch.Tensor
    input_lengths: torch.Tensor
    prefix_lengths: torch.Tensor
    sequence_lengths: torch.Tensor
    # cu_mask_rows: torch.Tensor
    max_seq_len: int
    max_prefix_length: int
    context_total_kv_length: int
    decode_plan: bool
    attn_type: torch.dtype

class FusedRopeKVCachePrefillOpQKVOut:
    def __init__(self, attn_configs: AttentionConfigs, max_seq_len: int) -> None:
        self.attn_configs = attn_configs
        self.max_seq_len = max_seq_len

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: KVCache | None,
        params: TRTAttn,
    ) -> torch.Tensor:
        rope_config = self.attn_configs.rope_config
        rope_cache = get_rope_cache(rope_config, self.max_seq_len)

        return rtp_kernel.prefill_fused_rope_kvcache.PrefillFusedRopeKVCache(
            qkv,
            params.cu_seqlens,
            params.cu_seqlens.size(0) - 1,
            params.max_seq_len,
            self.attn_configs.head_num,
            self.attn_configs.kv_head_num,
            self.attn_configs.size_per_head,
            tokens_per_block=self.attn_configs.tokens_per_block,
            store_q_no_transpose=False,
            store_q=False,
            store_kv=False,
            store_qkv=True,
            store_qkv_fp8=False, # tmp not use qkv fp8 buffer
            store_cache=kv_cache is not None,
            use_paged_fmha=False,
            kv_cache=None if kv_cache is None else kv_cache.kv_cache_base,
            kv_cache_scale=None if kv_cache is None else kv_cache.kv_scale_base,
            kv_cache_offset=params.kv_cache_offset,
            kv_cache_offset_h=params.kv_cache_offset_h,
            rope_cache=rope_cache,
            padding_offset=params.padding_offset,
            use_logn_attn=self.attn_configs.use_logn_attn,
            rope_style=rope_config.style,
            rope_dim=rope_config.dim,
            rope_base=rope_config.base,
            rope_scale=rope_config.scale,
            rope_beta_slow=rope_config.factor1,
            rope_beta_fast=rope_config.factor2,
            rope_original_max_position_embeddings=rope_config.max_pos,
            rope_extrapolation_factor=rope_config.extrapolation_factor,
            rope_mscale=rope_config.mscale,
            rope_offset=rope_config.offset,
            rope_index_factor=rope_config.index_factor,
            rope_mrope_dim1=rope_config.mrope_dim1,
            rope_mrope_dim2=rope_config.mrope_dim2,
            rope_mrope_dim3=rope_config.mrope_dim3,
            prefix_prompt_lengths=params.prefix_lengths,
            max_prefix_length=params.max_prefix_length,
            count_length=True,
        )

    def prepare(self, attn_inputs: PyAttentionInputs) -> TRTAttn:
        kv_cache_offset = (
            rtp_kernel.convert_offset_to_block_array.ConvertOffsetToBlockArray(
                attn_inputs.kv_cache_block_id_device
            )
        )
        kv_cache_offset_h = None
        return TRTAttn(
            kv_cache_offset,
            kv_cache_offset_h,
            attn_inputs.padding_offset,
            attn_inputs.cu_seqlens,
            attn_inputs.cu_kv_seqlens,
            attn_inputs.input_lengths,
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths.max(),
            attn_inputs.prefix_lengths.max(),
            attn_inputs.context_total_kv_length,
            False,
            attn_inputs.dtype,
        )

class FusedRopeKVCachePrefillOpQOut:
    def __init__(self, attn_configs: AttentionConfigs, max_seq_len: int) -> None:
        self.attn_configs = attn_configs
        self.max_seq_len = max_seq_len

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: KVCache | None,
        params: TRTAttn,
    ) -> torch.Tensor:
        use_paged_fmha = kv_cache is not None and params.max_prefix_length > 0

        rope_config = self.attn_configs.rope_config
        rope_cache = get_rope_cache(rope_config, self.max_seq_len)

        return rtp_kernel.prefill_fused_rope_kvcache.PrefillFusedRopeKVCache(
            qkv,
            params.cu_seqlens,
            params.cu_seqlens.size(0) - 1,
            params.max_seq_len,
            self.attn_configs.head_num,
            self.attn_configs.kv_head_num,
            self.attn_configs.size_per_head,
            tokens_per_block=self.attn_configs.tokens_per_block,
            store_q_no_transpose=True,
            store_q=False,
            store_kv=False,
            store_qkv=False,
            store_qkv_fp8=False, # tmp not use qkv fp8 buffer
            store_cache=kv_cache is not None,
            use_paged_fmha=use_paged_fmha,
            kv_cache=None if kv_cache is None else kv_cache.kv_cache_base,
            kv_cache_scale=None if kv_cache is None else kv_cache.kv_scale_base,
            kv_cache_offset=params.kv_cache_offset,
            kv_cache_offset_h=params.kv_cache_offset_h,
            rope_cache=rope_cache,
            padding_offset=params.padding_offset,
            use_logn_attn=self.attn_configs.use_logn_attn,
            rope_style=rope_config.style,
            rope_dim=rope_config.dim,
            rope_base=rope_config.base,
            rope_scale=rope_config.scale,
            rope_beta_slow=rope_config.factor1,
            rope_beta_fast=rope_config.factor2,
            rope_original_max_position_embeddings=rope_config.max_pos,
            rope_extrapolation_factor=rope_config.extrapolation_factor,
            rope_mscale=rope_config.mscale,
            rope_offset=rope_config.offset,
            rope_index_factor=rope_config.index_factor,
            rope_mrope_dim1=rope_config.mrope_dim1,
            rope_mrope_dim2=rope_config.mrope_dim2,
            rope_mrope_dim3=rope_config.mrope_dim3,
            prefix_prompt_lengths=params.prefix_lengths,
            max_prefix_length=params.max_prefix_length,
            count_length=True,
        )

    def prepare(self, attn_inputs: PyAttentionInputs) -> TRTAttn:
        kv_cache_offset = (
            rtp_kernel.convert_offset_to_block_array.ConvertOffsetToBlockArray(
                attn_inputs.kv_cache_block_id_device
            )
        )
        kv_cache_offset_h = None
        return TRTAttn(
            kv_cache_offset,
            kv_cache_offset_h,
            attn_inputs.padding_offset,
            attn_inputs.cu_seqlens,
            attn_inputs.cu_kv_seqlens,
            attn_inputs.input_lengths,
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths.max(),
            attn_inputs.prefix_lengths.max(),
            attn_inputs.context_total_kv_length,
            False,
            attn_inputs.dtype,
        )


class FusedRopeKVCacheDecodeOp:
    def __init__(self, attn_configs: AttentionConfigs, max_seq_len: int) -> None:
        self.attn_configs = attn_configs
        self.max_seq_len = max_seq_len

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: KVCache,
        params: TRTAttn,
    ) -> torch.Tensor:
        rope_config = self.attn_configs.rope_config
        rope_cache = get_rope_cache(rope_config, self.max_seq_len)
        return rtp_kernel.decode_fused_rope_kvcache.DecodeFusedRopeKVCache(
            qkv,
            params.sequence_lengths,
            params.sequence_lengths.size(0),
            self.attn_configs.head_num,
            self.attn_configs.kv_head_num,
            self.attn_configs.size_per_head,
            kv_cache.kv_cache_base,
            params.kv_cache_offset,
            tokens_per_block=self.attn_configs.tokens_per_block,
            store_kv=False,
            kv_cache_scale=kv_cache.kv_scale_base,
            kv_cache_offset_h=params.kv_cache_offset_h,
            rope_cache=rope_cache,
            use_logn_attn=self.attn_configs.use_logn_attn,
            rope_style=rope_config.style,
            rope_dim=rope_config.dim,
            rope_base=rope_config.base,
            rope_scale=rope_config.scale,
            rope_beta_slow=rope_config.factor1,
            rope_beta_fast=rope_config.factor2,
            rope_original_max_position_embeddings=rope_config.max_pos,
            rope_extrapolation_factor=rope_config.extrapolation_factor,
            rope_mscale=rope_config.mscale,
            rope_offset=rope_config.offset,
            rope_index_factor=rope_config.index_factor,
            rope_mrope_dim1=rope_config.mrope_dim1,
            rope_mrope_dim2=rope_config.mrope_dim2,
            rope_mrope_dim3=rope_config.mrope_dim3,
        )

    def prepare(self, attn_inputs: PyAttentionInputs) -> TRTAttn:
        kv_cache_offset = (
            rtp_kernel.convert_offset_to_block_array.ConvertOffsetToBlockArray(
                attn_inputs.kv_cache_block_id_device
            )
        )
        kv_cache_offset_h = None
        return TRTAttn(
            kv_cache_offset,
            kv_cache_offset_h,
            attn_inputs.padding_offset,
            attn_inputs.cu_seqlens,
            attn_inputs.cu_kv_seqlens,
            attn_inputs.input_lengths,
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths.max(),
            attn_inputs.prefix_lengths.max(),
            attn_inputs.context_total_kv_length,
            True,
            attn_inputs.dtype,
        )
