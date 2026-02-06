import math
from typing import Optional

import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    FMHAType,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQKVOut,
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
    get_scalar_type,
)


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
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            "NHD",
            backend="auto",
        )

    def prepare(self, attn_inputs: PyAttentionInputs) -> ParamsBase:
        cu_seqlen_without_padding = attn_inputs.cu_seqlens[
            : attn_inputs.input_lengths.size(0) + 1
        ]
        self.prefill_wrapper.plan(
            cu_seqlen_without_padding,
            cu_seqlen_without_padding,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.head_dim_vo,
            causal=True,
            q_data_type=get_scalar_type(attn_inputs.dtype),
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


class PyFlashinferPrefillImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        super().__init__(
            PyFlashinferPrefillAttnOp(attn_configs),
            FusedRopeKVCachePrefillOpQKVOut(attn_configs),
            attn_inputs,
        )

    def support(self):
        return True

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASHINFER_PREFILL


from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


def determine_use_tensor_core_from_configs(attn_configs: AttentionConfigs) -> bool:
    """Determine whether to use tensor cores based on attention configs."""
    # Use tensor cores for larger head dimensions and when kv_head_num matches requirements
    return attn_configs.head_num // attn_configs.kv_head_num >= 4


class PyFlashinferDecodeAttnOp(object):
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        # Get dtype from attn_configs (ScalarType is automatically converted to torch.dtype by pybind11)
        self.g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device="cuda",
        )
        # attn_configs already has head_num and kv_head_num divided by tp_size
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.seq_size_per_block = attn_configs.tokens_per_block
        self.use_tensor_core = determine_use_tensor_core_from_configs(attn_configs)
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            use_tensor_cores=self.use_tensor_core,
        )
        self.kv_cache_dtype = attn_configs.kv_cache_dtype

    def prepare(self, attn_inputs: PyAttentionInputs):
        # from rtp_llm.models_py.utils.debug import set_trace_on_tty
        # set_trace_on_tty()
        # Convert kv_cache_dtype to torch dtype
        if self.kv_cache_dtype == KvCacheDataType.INT8:
            kv_datatype = torch.int8
        elif self.kv_cache_dtype == KvCacheDataType.FP8:
            kv_datatype = torch.float8_e4m3fn
        else:  # BASE
            kv_datatype = get_scalar_type(attn_inputs.dtype)
        flashinfer_decode_params = fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.seq_size_per_block,
        )
        # Get torch.dtype from attention configs
        self.decode_wrapper.plan(
            flashinfer_decode_params.decode_page_indptr_d,
            flashinfer_decode_params.page_indice_d,
            flashinfer_decode_params.paged_kv_last_page_len_d,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.seq_size_per_block,
            q_data_type=get_scalar_type(attn_inputs.dtype),
            kv_data_type=kv_datatype,
        )
        return flashinfer_decode_params

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[KVCache], params: ParamsBase
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        return self.decode_wrapper.run(q, kv_cache.kv_cache_base)


class PyFlashinferDecodeImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        super().__init__(
            PyFlashinferDecodeAttnOp(attn_configs),
            FusedRopeKVCacheDecodeOp(attn_configs),
            attn_inputs,
        )
        self.support_ = attn_configs.use_mla == False

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASHINFER_DECODE

    def support(self):
        return self.support_
