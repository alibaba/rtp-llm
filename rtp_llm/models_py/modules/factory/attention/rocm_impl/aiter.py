import logging
from typing import Any, Optional

import aiter
import torch
from aiter import dtypes

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAImplBase,
    FMHAPrefillImplBase,
)
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOp,
    KVCache,
    ParamsBase,
    PyAttentionInputs,
)


# Pure Python implementation of FMHAParams
class FMHAParams(ParamsBase):
    """Python implementation of FMHAParams for Aiter attention operations."""

    def __init__(
        self,
        attn_inputs: PyAttentionInputs,
        is_prefill: bool = True,
    ):
        super().__init__()

        # Prefill mode
        if is_prefill:
            input_lengths = attn_inputs.input_lengths
            prefix_lengths = (
                attn_inputs.prefix_lengths
                if hasattr(attn_inputs, "prefix_lengths")
                else None
            )

            self.max_seq_len = input_lengths.max().item()
            batch_size = input_lengths.size(0)

            # Create cu_seqlens_q for query (based on input_lengths only)
            self.cu_seqlens_q = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=input_lengths.device
            )
            self.cu_seqlens_q[1:] = torch.cumsum(input_lengths, 0)

            kv_lengths = torch.zeros_like(input_lengths)
            # Create cu_seqlens_k for key/value (includes prefix_lengths)
            if prefix_lengths is not None and prefix_lengths.numel() > 0:
                kv_lengths = input_lengths + prefix_lengths
                self.cu_seqlens_k = torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=input_lengths.device
                )
                self.cu_seqlens_k[1:] = torch.cumsum(kv_lengths, 0)
                # Calculate max sequence length including prefix
                max_prefix_length = (
                    prefix_lengths.max().item() if prefix_lengths.numel() > 0 else 0
                )
                self.max_seqlen_k = self.max_seq_len + max_prefix_length
            else:
                self.cu_seqlens_k = self.cu_seqlens_q.clone()
                self.max_seqlen_k = self.max_seq_len

            self.max_seqlen_q = self.max_seq_len
            self.seq_lens = None
            self.kv_cache_block_id_device = getattr(
                attn_inputs, "kv_cache_block_id_device", None
            )
            self.prefix_lengths = prefix_lengths
            self.token_q_num = input_lengths.sum().item()
            self.token_kv_num = kv_lengths.sum().item()
        # Decode mode
        else:
            input_lengths = attn_inputs.input_lengths
            sequence_lengths = getattr(attn_inputs, "sequence_lengths", None)
            kv_cache_block_id_device = getattr(
                attn_inputs, "kv_cache_block_id_device", None
            )
            enable_cuda_graph = getattr(attn_inputs, "enable_cuda_graph", False)

            self.sequence_lengths = sequence_lengths
            self.kv_cache_block_id_device = kv_cache_block_id_device

            if enable_cuda_graph:
                self.max_seq_len = 8192
            else:
                self.max_seq_len = input_lengths.max().item() + 1

            self.max_seqlen_k = self.max_seq_len
            self.max_seqlen_q = 0
            self.cu_seqlens_q = None
            self.cu_seqlens_k = None

            # Create seq_lens on CUDA
            if sequence_lengths is not None:
                self.seq_lens = (sequence_lengths + 1).to(torch.device("cuda"))
            else:
                self.seq_lens = None

    def fillParams(self, sequence_lengths, input_lengths, kv_cache_block_id_host):
        self.sequence_lengths = sequence_lengths
        self.input_lengths = input_lengths
        self.kv_cache_block_id_host = kv_cache_block_id_host
        if self.seq_lens is not None and self.sequence_lengths is not None:
            self.seq_lens.copy_((self.sequence_lengths + 1).to(torch.device("cuda")))
            self.max_seq_len = 8192

    def check_recycle(self) -> bool:
        """Check whether the params can be recycled automatically."""
        return True


class AiterPrefillAttnOp:
    def __init__(self, config: GptInitModelParameters):
        self.head_num = config.head_num // config.tp_size
        self.head_dim = config.size_per_head
        self.head_num_kv = config.head_num_kv // config.tp_size
        self.kv_cache_data_type = config.kv_cache_data_type

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
        )
        return fmha_params

    def reshape_qkv(self, qkv):
        q_contiguous = qkv[0].permute(1, 0, 2).contiguous()
        k_contiguous = qkv[1].permute(1, 0, 2).contiguous()
        v_contiguous = qkv[2].permute(1, 0, 2).contiguous()
        return q_contiguous, k_contiguous, v_contiguous

    def forward(self, qkv, kv_cache, fmha_params):

        q_tensor, k_tensor, v_tensor = self.reshape_qkv(qkv)

        q_tensor = q_tensor[: fmha_params.token_q_num]
        k_tensor = k_tensor[: fmha_params.token_kv_num]
        v_tensor = v_tensor[: fmha_params.token_kv_num]

        cu_seqlens_q = fmha_params.cu_seqlens_q.to(q_tensor.device)
        cu_seqlens_k = fmha_params.cu_seqlens_k.to(k_tensor.device)
        max_seqlen_q = fmha_params.max_seqlen_q
        max_seqlen_k = fmha_params.max_seqlen_k

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
        token_num = fmha_params.token_q_num
        final_result = res.reshape(token_num, self.head_num * self.head_dim)
        return final_result


class AiterDecodeAttnOp:
    def __init__(self, config: GptInitModelParameters):
        self.head_num = config.head_num // config.tp_size
        self.head_dim = config.size_per_head
        self.head_num_kv = config.head_num_kv // config.tp_size
        self.kv_cache_data_type = config.kv_cache_data_type
        self.use_asm_pa = config.hw_kernel_config.use_asm_pa
        self.enable_cuda_graph = (
            config.gpt_init_params.hw_kernel_config.enable_cuda_graph
        )

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Create decode parameters using pure Python implementation
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=False,
        )
        return fmha_params

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params
    ) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens
        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)
        block_tables_id_device = fmha_params.kv_cache_block_id_device
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
            max_seq_len = fmha_params.max_seq_len
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


class AiterPrefillImpl(FMHAPrefillImplBase):
    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            AiterPrefillAttnOp(config),
            FusedRopeKVCachePrefillOp(config.gpt_init_params),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.AITER_PREFILL


class AiterDecodeImpl(FMHADecodeImplBase):
    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            AiterDecodeAttnOp(config),
            FusedRopeKVCacheDecodeOp(config.gpt_init_params),
            attn_inputs,
        )
