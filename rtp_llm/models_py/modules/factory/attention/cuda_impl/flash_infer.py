import logging
from typing import Any, Optional

import flashinfer
import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAPrefillImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAType, KvCacheDataType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQKVOut,
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
)
from rtp_llm.utils.util import to_torch_dtype


class PyFlashinferPrefillAttnOp(object):
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device="cuda",
        )
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        # TODO: maybe use v_head_dim
        self.head_dim_vo = attn_configs.size_per_head
        self.prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
        )

    def prepare(self, attn_inputs: PyAttentionInputs) -> ParamsBase:
        cu_seqlen_without_padding = attn_inputs.cu_seqlens[
            : attn_inputs.input_lengths.size(0) + 1
        ]
        # Infer dtype from input if available, otherwise use default
        self.prefill_wrapper.plan(
            cu_seqlen_without_padding,
            cu_seqlen_without_padding,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.head_dim_vo,
            causal=True,
            q_data_type=attn_inputs.dtype,
        )
        return ParamsBase()

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, qkv: torch.Tensor, kv_cache: Optional[KVCache], params: ParamsBase
    ) -> torch.Tensor:
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim_qk * self.local_head_num,
                self.head_dim_qk * self.local_kv_head_num,
                self.head_dim_vo * self.local_kv_head_num,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        k = k.reshape(k.shape[0], self.local_kv_head_num, self.head_dim_qk)
        v = v.reshape(v.shape[0], self.local_kv_head_num, self.head_dim_vo)
        return self.prefill_wrapper.run(q, k, v)


class FlashInferPrefillImpl(FMHAPrefillImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            PyFlashinferPrefillAttnOp(attn_configs),
            FusedRopeKVCachePrefillOpQKVOut(attn_configs),
            attn_inputs,
        )
        self.support_ = self.support_ and (not attn_configs.use_mla)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True


class PyFlashinferDecodeAttnOp(object):
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device="cuda",
        )
        self.attn_configs = attn_configs
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.seq_size_per_block = attn_configs.tokens_per_block
        self.decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            kv_layout="HND",
        )
        # Use default dtype, will be inferred from input tensor in prepare/forward
        # Convert kv_cache_dtype to torch dtype
        if attn_configs.kv_cache_dtype == KvCacheDataType.INT8:
            self.kv_datatype = torch.int8
        elif attn_configs.kv_cache_dtype == KvCacheDataType.FP8:
            self.kv_datatype = torch.float8_e4m3fn
        else:  # BASE
            self.kv_datatype = torch.float16

    def prepare(self, attn_inputs: PyAttentionInputs) -> ParamsBase:
        # from rtp_llm.models_py.utils.debug import set_trace_on_tty
        # set_trace_on_tty()
        # Infer dtype from input if available, otherwise use default
        flashinfer_decode_params = fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.seq_size_per_block,
        )
        self.decode_wrapper.plan(
            flashinfer_decode_params.decode_page_indptr,
            flashinfer_decode_params.page_indice,
            flashinfer_decode_params.paged_kv_last_page_len,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.seq_size_per_block,
            q_data_type=attn_inputs.dtype,
            kv_data_type=(
                attn_inputs.dtype
                if attn_configs.kv_cache_dtype == KvCacheDataType.BASE
                else self.kv_datatype
            ),
        )
        return ParamsBase()

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[KVCache], params: ParamsBase
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        return self.decode_wrapper.run(q, kv_cache.k_cache_base)


class FlashInferDecodeImpl(FMHADecodeImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            PyFlashinferDecodeAttnOp(attn_configs),
            FusedRopeKVCacheDecodeOp(attn_configs),
            attn_inputs,
        )
        self.support_ = self.support_ and (not attn_configs.use_mla)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True
