import logging
import os
from typing import Any, List, Optional

import aiter
import torch
from aiter import dtypes
from librtp_compute_ops.rtp_llm_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOp,
)

try:
    from librtp_compute_ops.rtp_llm_ops import AiterAttnPyParams
except ImportError:
    AiterAttnPyParams = None

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.ops import FMHAType, KVCache, ParamsBase, PyAttentionInputs


class FMHAPrefillImplBase(FMHAImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        attn_inputs: PyAttentionInputs,
        config: GptInitModelParameters,
    ) -> None:
        super().__init__(
            fmha_impl, FusedRopeKVCachePrefillOp(config.gpt_init_params), attn_inputs
        )


class FMHADecodeImplBase(FMHAImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        attn_inputs: PyAttentionInputs,
        config: GptInitModelParameters,
    ) -> None:
        super().__init__(
            fmha_impl, FusedRopeKVCacheDecodeOp(config.gpt_init_params), attn_inputs
        )


PREFILL_MHA_IMPS: List[type[FMHAPrefillImplBase]] = []
DECODE_MHA_IMPS: List[type[FMHADecodeImplBase]] = []

try:

    class AiterPrefillImpl(FMHAPrefillImplBase):
        def __init__(
            self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
        ) -> None:
            super().__init__(AiterPrefillAttnOp(config), attn_inputs, config)

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.AITER_PREFILL

    PREFILL_MHA_IMPS.append(AiterPrefillImpl)
except ImportError:
    logging.info("AiterPrefillImpl not available, skipped.")


class AiterPrefillAttnOp:
    def __init__(self, config: GptInitModelParameters):
        self.head_num = config.head_num
        self.head_dim = config.hidden_size // config.head_num
        self.head_num_kv = config.head_num_kv
        self.kv_cache_data_type = config.kv_cache_data_type

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        if AiterAttnPyParams is None:
            raise ImportError(
                "AiterAttnPyParams is not available from librtp_compute_ops.rtp_llm_ops"
            )
        from librtp_compute_ops.rtp_llm_ops import AiterParamsCreator

        fmha_params = AiterAttnPyParams(attn_inputs.input_lengths, True)
        params_creator = AiterParamsCreator()
        self.fmha_params = params_creator.create_prefill_params(fmha_params)
        return self.fmha_params

    def advanced_qkv_split(self, qkv, head_num, head_num_kv, size_per_head):
        token_num = qkv.shape[0]
        qkv_reshaped = qkv.reshape(token_num, head_num + 2 * head_num_kv, size_per_head)
        q = qkv_reshaped[:, :head_num, :]
        k = qkv_reshaped[:, head_num : head_num + head_num_kv, :]
        v = qkv_reshaped[:, head_num + head_num_kv : head_num + 2 * head_num_kv, :]
        return q, k, v

    def forward(self, qkv, kv_cache, fmha_params):
        cu_seqlens_q = fmha_params.cu_seqlens_q_.to(qkv.device)
        cu_seqlens_k = fmha_params.cu_seqlens_k_.to(qkv.device)
        max_seqlen_q = fmha_params.max_seqlen_q_
        max_seqlen_k = fmha_params.max_seqlen_k_

        q_tensor, k_tensor, v_tensor = self.advanced_qkv_split(
            qkv, self.head_num, self.head_num_kv, self.head_dim
        )
        res = aiter.flash_attn_varlen_func(
            q_tensor,  # Query张量: (total_q, nheads, headdim_q) - 批次中所有query token的总数
            k_tensor,  # Key张量: (total_k, nheads_k, headdim_q) - 批次中所有key token的总数
            v_tensor,  # Value张量: (total_k, nheads_k, headdim_v) - 批次中所有value token的总数
            cu_seqlens_q,  # Query累积序列长度: (batch_size + 1,) dtype=int32 - 用于索引q张量
            cu_seqlens_k,  # Key累积序列长度: (batch_size + 1,) dtype=int32 - 用于索引k/v张量
            max_seqlen_q,  # 批次中最大query序列长度
            max_seqlen_k,  # 批次中最大key序列长度
            dropout_p=0.0,  # Dropout概率 - 评估时应设为0.0
            causal=True,  # 因果注意力掩码 - 用于自回归建模，每个位置只能关注自己和之前的位置
        )
        token_num = res.shape[0]
        final_result = res.reshape(token_num, self.head_num * self.head_dim)
        return final_result


try:

    class AiterDecodeImpl(FMHADecodeImplBase):
        def __init__(
            self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
        ) -> None:
            super().__init__(AiterDecodeAttnOp(config), attn_inputs, config)

    DECODE_MHA_IMPS.append(AiterDecodeImpl)
except ImportError:
    logging.info("PagedAttnDecodeOp not available, skipped.")


class AiterDecodeAttnOp:
    def __init__(self, config: GptInitModelParameters):
        self.head_num = config.head_num
        self.head_dim = config.hidden_size // config.head_num
        self.head_num_kv = config.head_num_kv
        self.kv_cache_data_type = config.kv_cache_data_type
        self.use_asm_pa = config.hw_kernel_config.use_asm_pa
        self.enable_cuda_graph = (
            config.gpt_init_params.hw_kernel_config.enable_cuda_graph
        )

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        if AiterAttnPyParams is None:
            raise ImportError(
                "AiterAttnPyParams is not available from librtp_compute_ops.rtp_llm_ops"
            )
        from librtp_compute_ops.rtp_llm_ops import AiterParamsCreator

        fmha_params = AiterAttnPyParams(
            attn_inputs.input_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.kv_cache_block_id_host,
            attn_inputs.kv_cache_block_id_device,
            self.enable_cuda_graph,
        )

        params_creator = AiterParamsCreator()
        self.fmha_params = params_creator.create_decode_params(fmha_params)
        return self.fmha_params

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params
    ) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens_
        key_cache = kv_cache.k_cache_base
        value_cache = kv_cache.v_cache_base

        block_tables_id_device = fmha_params.kv_cache_block_id_device_
        max_num_blocks = block_tables_id_device.shape[1]
        # for now not support fp8
        if self.use_asm_pa:
            output = aiter.pa_fwd_asm(
                query,  # [num_seqs, num_heads, head_size]
                key_cache,  # [num_blocks, num_kv_heads, block_size, head_size/x, x]
                value_cache,  # [num_blocks, num_kv_heads, block_size, head_size/x, x]
                block_tables_id_device,
                seq_lens,
                max_num_blocks,
            )
        else:
            max_seq_len = fmha_params.max_seq_len_
            scale = 1.0 / (self.head_dim**0.5)
            alibi_slopes = None
            k_scale = (
                kv_cache.k_scale_base
                if kv_cache and kv_cache.k_scale_base is not None
                else torch.tensor(1.0, device=query.device, dtype=query.dtype)
            )
            v_scale = (
                kv_cache.v_scale_base
                if kv_cache and kv_cache.v_scale_base is not None
                else torch.tensor(1.0, device=query.device, dtype=query.dtype)
            )
            num_kv_heads = self.head_num_kv
            num_seqs, num_heads, head_size = query.shape
            block_size = value_cache.shape[2]
            _PARTITION_SIZE_ROCM = 256

            # init output
            output = torch.empty_like(query)

            max_num_partitions = (
                max_seq_len + _PARTITION_SIZE_ROCM - 1
            ) // _PARTITION_SIZE_ROCM
            assert _PARTITION_SIZE_ROCM % block_size == 0
            # init tmp_output
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )

            # init exp_sums
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            fp8_out_scale = None
            cpa_fp8_out = False
            # init max_logits
            max_logits = torch.ones_like(exp_sums)

            kv_cache_dtype = "auto"
            # key_cache_reshaped = key_cache.permute(0, 1, 3, 2)
            # value_cache_reshaped = value_cache.permute(0, 1, 3, 2)

            aiter.paged_attention_rocm(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                float(scale),
                block_tables_id_device,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,  # kv_cache_dtype
                k_scale,
                v_scale,
                fp8_out_scale if cpa_fp8_out else None,
                _PARTITION_SIZE_ROCM,
            )

        output_reshaped = output.view(output.shape[0], -1)
        return output_reshaped
