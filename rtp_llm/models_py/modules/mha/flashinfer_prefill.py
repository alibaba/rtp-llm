import logging
from typing import Optional

import torch
import flashinfer

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.mha.base import (
    FMHAPrefillImplBase,
    FMHAType,
    PREFILL_MHA_IMPS,
)
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
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
        self.prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
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


try:
    from rtp_llm.ops.compute_ops import FusedRopeKVCachePrefillOpQKVOut
    class FlashInferPrefillImpl(FMHAPrefillImplBase):

        def __init__(
            self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
        ) -> None:
            super().__init__(
                PyFlashinferPrefillAttnOp(config.gpt_init_params),
                FusedRopeKVCachePrefillOpQKVOut(config.gpt_init_params),
                attn_inputs,
            )
            self.support_ = self.support_ and (config.use_mla == False)

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.FLASH_INFER

        def support_cuda_graph(self) -> bool:
            return True

    PREFILL_MHA_IMPS.append(FlashInferPrefillImpl)

except ImportError:
    logging.info("FlashInferPrefillImpl not available, skipped.")
