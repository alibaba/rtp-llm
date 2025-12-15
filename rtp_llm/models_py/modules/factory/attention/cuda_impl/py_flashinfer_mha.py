import logging
from typing import Optional

import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.fmha import (
    FMHADecodeImplBase,
    FMHAPrefillImplBase,
    FMHAType,
)
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOp,
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
)
from rtp_llm.utils.util import to_torch_dtype


class PyFlashinferPrefillAttnOp(object):
    def __init__(self, config: GptInitModelParameters) -> None:
        self.g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device="cuda",
        )
        self.local_head_num = config.head_num // config.tp_size
        self.local_kv_head_num = config.head_num_kv // config.tp_size
        self.head_dim_qk = config.size_per_head
        # TODO: maybe use v_head_dim
        self.head_dim_vo = config.size_per_head
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            "NHD",
            backend="auto",
        )
        self.datatype = to_torch_dtype(config.data_type)

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
            q_data_type=self.datatype,
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


class PyFlashinferPrefillImpl(FMHAPrefillImplBase):
    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            PyFlashinferPrefillAttnOp(config),
            FusedRopeKVCachePrefillOp(config.gpt_init_params),
            attn_inputs,
        )

    def support(self):
        return True

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASH_INFER_PREFILL


from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


class PyFlashinferDecodeAttnOp(object):
    def __init__(self, config: GptInitModelParameters) -> None:
        self.g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device="cuda",
        )
        self.local_head_num = config.head_num // config.tp_size
        self.local_kv_head_num = config.head_num_kv // config.tp_size
        self.head_dim_qk = config.size_per_head
        self.head_dim_vo = config.size_per_head
        self.seq_size_per_block = config.seq_size_per_block
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
        )
        self.datatype = to_torch_dtype(config.data_type)

    def prepare(self, attn_inputs: PyAttentionInputs):
        # from rtp_llm.models_py.utils.debug import set_trace_on_tty
        # set_trace_on_tty()
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
            q_data_type=self.datatype,
            kv_data_type=self.datatype,
        )
        return flashinfer_decode_params

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[KVCache], params
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        return self.decode_wrapper.run(q, kv_cache.k_cache_base)


class PyFlashinferDecodeImpl(FMHADecodeImplBase):
    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            PyFlashinferDecodeAttnOp(config),
            FusedRopeKVCacheDecodeOp(config.gpt_init_params),
            attn_inputs,
        )
        self.support_ = config.use_mla == False

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASH_INFER_DECODE

    def support(self):
        return self.support_
