from dataclasses import dataclass
from typing import Optional

import torch

from rtp_kernel.fused_rope_kvcache import (
    convert_offset_to_block_array,
    decode_fused_rope_kvcache,
    prefill_fused_rope_kvcache,
)

from librtp_compute_ops import KVCache, PyAttentionInputs, get_scalar_type
from libth_transformer_config import (
    AttentionConfigs,
    check_rope_cache,
    get_rope_cache_once,
)


@dataclass
class FusedRopeAttnParams:
    kv_cache_offset: Optional[torch.Tensor]
    kv_cache_offset_h: Optional[torch.Tensor]
    padding_offset: Optional[torch.Tensor]
    cp_position_ids: Optional[torch.Tensor]
    cu_seqlens: torch.Tensor
    cu_kv_seqlens: torch.Tensor
    input_lengths: torch.Tensor
    prefix_lengths: torch.Tensor
    sequence_lengths: torch.Tensor
    max_seq_len: int
    max_prefix_length: int
    context_total_kv_length: int
    decode_plan: bool
    attn_type: torch.dtype


class FusedRopeKVCachePrefillOpBase:
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs

    def prepare(self, attn_inputs: PyAttentionInputs) -> FusedRopeAttnParams:
        if (
            attn_inputs.kv_cache_block_id_host is not None
            and attn_inputs.kv_cache_block_id_host.numel() > 0
        ):
            kv_cache_offset = convert_offset_to_block_array(
                attn_inputs.kv_cache_block_id_device
            )
        else:
            kv_cache_offset = None
        kv_cache_offset_h = None

        cp_position_ids = None
        if attn_inputs.context_parallel_info is not None:
            cp_position_ids = attn_inputs.context_parallel_info.prefill_shuffle_indices

        return FusedRopeAttnParams(
            kv_cache_offset,
            kv_cache_offset_h,
            attn_inputs.padding_offset,
            cp_position_ids,
            attn_inputs.cu_seqlens,
            attn_inputs.cu_kv_seqlens,
            attn_inputs.input_lengths,
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths.max().item(),
            attn_inputs.prefix_lengths.max().item(),
            attn_inputs.context_total_kv_length,
            False,
            get_scalar_type(attn_inputs.dtype),
        )

    def _forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        params: FusedRopeAttnParams,
        store_q_no_transpose: bool,
        store_q: bool,
        store_kv: bool,
        store_qkv: bool,
        store_qkv_fp8: bool,
        use_paged_fmha: bool,
    ) -> torch.Tensor:
        store_cache = kv_cache is not None

        rope_config = self.attn_configs.rope_config
        rope_cache = get_rope_cache_once(rope_config, self.attn_configs.max_seq_len)

        return prefill_fused_rope_kvcache(
            qkv,
            params.cu_seqlens,
            params.cu_seqlens.size(0) - 1,
            params.max_seq_len,
            self.attn_configs.head_num,
            self.attn_configs.kv_head_num,
            self.attn_configs.size_per_head,
            tokens_per_block=self.attn_configs.tokens_per_block,
            store_q_no_transpose=store_q_no_transpose,
            store_q=store_q,
            store_kv=store_kv,
            store_qkv=store_qkv,
            store_qkv_fp8=store_qkv_fp8,
            store_cache=store_cache,
            use_paged_fmha=use_paged_fmha,
            kv_cache=None if kv_cache is None else kv_cache.kv_cache_base,
            kv_cache_scale=None if kv_cache is None else kv_cache.kv_scale_base,
            kv_cache_offset=params.kv_cache_offset,
            kv_cache_offset_h=params.kv_cache_offset_h,
            rope_cache=(
                rope_cache.data if check_rope_cache(rope_config, rope_cache) else None
            ),
            padding_offset=params.padding_offset,
            cp_position_ids=params.cp_position_ids,
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
            count_length=params.max_prefix_length > 0,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        params: FusedRopeAttnParams,
    ) -> torch.Tensor:
        raise NotImplementedError()


class FusedRopeKVCachePrefillOpQKVOut(FusedRopeKVCachePrefillOpBase):
    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        params: FusedRopeAttnParams,
    ) -> torch.Tensor:
        return self._forward(
            qkv, kv_cache, params, False, False, False, True, False, False
        )


class FusedRopeKVCachePrefillOpQOut(FusedRopeKVCachePrefillOpBase):
    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        params: FusedRopeAttnParams,
    ) -> torch.Tensor:
        use_paged_fmha = kv_cache is not None and params.max_prefix_length > 0

        return self._forward(
            qkv, kv_cache, params, True, False, False, False, False, use_paged_fmha
        )


class FusedRopeKVCacheDecodeOp:
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: KVCache,
        params: FusedRopeAttnParams,
    ) -> torch.Tensor:
        rope_config = self.attn_configs.rope_config
        rope_cache = get_rope_cache_once(rope_config, self.attn_configs.max_seq_len)
        assert params.kv_cache_offset is not None
        return decode_fused_rope_kvcache(
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
            rope_cache=(
                rope_cache.data if check_rope_cache(rope_config, rope_cache) else None
            ),
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

    def prepare(self, attn_inputs: PyAttentionInputs) -> FusedRopeAttnParams:
        assert (
            attn_inputs.kv_cache_block_id_host is not None
            and attn_inputs.kv_cache_block_id_host.numel() > 0
        )
        kv_cache_offset = convert_offset_to_block_array(
            attn_inputs.kv_cache_block_id_device
        )
        kv_cache_offset_h = None
        return FusedRopeAttnParams(
            kv_cache_offset,
            kv_cache_offset_h,
            attn_inputs.padding_offset,
            attn_inputs.position_ids,
            attn_inputs.cu_seqlens,
            attn_inputs.cu_kv_seqlens,
            attn_inputs.input_lengths,
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            0,
            0,
            attn_inputs.context_total_kv_length,
            True,
            get_scalar_type(attn_inputs.dtype),
        )
