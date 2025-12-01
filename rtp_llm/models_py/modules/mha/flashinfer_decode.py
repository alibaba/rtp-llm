import logging

from typing import Optional

import torch
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.ops import FMHAType
import flashinfer
from rtp_llm.ops.compute_ops import PyAttentionInputs, ParamsBase, KVCache, rtp_llm_ops
from rtp_llm.utils.util import to_torch_dtype
from rtp_llm.models_py.modules.mha.base import (
    FMHADecodeImplBase,
    DECODE_MHA_IMPS,
)

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
        self.decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
        )
        self.datatype = to_torch_dtype(config.data_type)

    def prepare(self, attn_inputs: PyAttentionInputs) -> ParamsBase:
        # from rtp_llm.models_py.utils.debug import set_trace_on_tty
        # set_trace_on_tty()
        flashinfer_decode_params = rtp_llm_ops.fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.seq_size_per_block,
        )
        self.decode_wrapper.plan(
            flashinfer_decode_params.page_indptr,
            flashinfer_decode_params.page_indice,
            flashinfer_decode_params.paged_kv_last_page_len,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.seq_size_per_block,
            q_data_type=self.datatype,
            kv_data_type=self.datatype,
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

try:
    from rtp_llm.ops.compute_ops import FusedRopeKVCacheDecodeOp

    class FlashInferDecodeImpl(FMHADecodeImplBase):

        def __init__(
            self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
        ) -> None:
            super().__init__(
                PyFlashinferDecodeAttnOp(config.gpt_init_params),
                FusedRopeKVCacheDecodeOp(config.gpt_init_params),
                attn_inputs,
            )
            self.support_ = self.support_ and (config.use_mla == False)

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.FLASH_INFER

        def support_cuda_graph(self) -> bool:
            return True

    DECODE_MHA_IMPS.append(FlashInferDecodeImpl)

except ImportError:
    logging.info("FlashInferDecodeOp not available, skipped.")
