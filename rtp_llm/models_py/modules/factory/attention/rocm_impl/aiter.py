import logging
from typing import Any, List, Optional

import aiter
import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAImplBase,
    FMHAPrefillImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOpAsm,
    FusedRopeKVCacheDecodeOpNonAsm,
    FusedRopeKVCachePrefillOpAsm,
    FusedRopeKVCachePrefillOpNonAsm,
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
        enable_cuda_graph: bool = True,
    ):
        super().__init__()
        self.enable_cuda_graph = enable_cuda_graph

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

            self.sequence_lengths = sequence_lengths
            self.kv_cache_block_id_device = kv_cache_block_id_device

            if self.enable_cuda_graph:
                self.max_seq_len = 8192
            else:
                self.max_seq_len = input_lengths.max().item() + 1

            self.max_seqlen_k = self.max_seq_len
            self.max_seqlen_q = 0
            self.cu_seqlens_q = None
            self.cu_seqlens_k = None

            self.attn_inputs = attn_inputs

            # For backward compatibility, also store seq_lens
            sequence_lengths_plus_1_d = getattr(
                attn_inputs, "sequence_lengths_plus_1_d", None
            )
            if (
                sequence_lengths_plus_1_d is not None
                and sequence_lengths_plus_1_d.numel() > 0
            ):
                self.seq_lens = sequence_lengths_plus_1_d
            elif sequence_lengths is not None:
                self.seq_lens = (sequence_lengths + 1).to(torch.device("cuda"))
            else:
                self.seq_lens = None

    def check_recycle(self) -> bool:
        """Check whether the params can be recycled automatically."""
        return True


class AiterPrefillAttnOp:
    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self.attn_inputs = attn_inputs
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
        )
        return fmha_params

    def advanced_qkv_split(self, qkv, head_num, head_num_kv, size_per_head):
        token_num = qkv.shape[0]
        qkv_reshaped = qkv.reshape(token_num, head_num + 2 * head_num_kv, size_per_head)
        q = qkv_reshaped[:, :head_num, :]
        k = qkv_reshaped[:, head_num : head_num + head_num_kv, :]
        v = qkv_reshaped[:, head_num + head_num_kv : head_num + 2 * head_num_kv, :]
        return q, k, v

    def reshape_qkv(self, qkv):
        q_contiguous = qkv[0].permute(1, 0, 2).contiguous()
        k_contiguous = qkv[1].permute(1, 0, 2).contiguous()
        v_contiguous = qkv[2].permute(1, 0, 2).contiguous()
        return q_contiguous, k_contiguous, v_contiguous

    def forward(self, qkv, kv_cache, fmha_params):
        has_prefix = (
            fmha_params.prefix_lengths is not None
            and fmha_params.prefix_lengths.numel() > 0
            and fmha_params.prefix_lengths.max().item() > 0
        )
        if has_prefix:
            q_tensor, k_tensor, v_tensor = self.reshape_qkv(qkv)
            q_tensor = q_tensor[: fmha_params.token_q_num]
            k_tensor = k_tensor[: fmha_params.token_kv_num]
            v_tensor = v_tensor[: fmha_params.token_kv_num]
        else:
            q_tensor, k_tensor, v_tensor = self.advanced_qkv_split(
                qkv[0],
                self.head_num,
                self.head_num_kv,
                self.head_dim,
            )
        cu_seqlens_q = fmha_params.cu_seqlens_q.to(q_tensor.device)
        cu_seqlens_k = fmha_params.cu_seqlens_k.to(k_tensor.device)
        max_seqlen_q = fmha_params.max_seqlen_q
        max_seqlen_k = fmha_params.max_seqlen_k

        if (
            q_tensor.dtype == torch.float8_e4m3fnuz
            and k_tensor.dtype == torch.float8_e4m3fnuz
            and v_tensor.dtype == torch.float8_e4m3fnuz
        ):
            res = aiter.flash_attn_varlen_fp8_pertensor_func(
                q_tensor,
                k_tensor,
                v_tensor,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=True,
            )
        else:
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


class AiterDecodeAttnOpBase:
    """Base class for Aiter decode attention operations."""

    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.enable_cuda_graph = True
        self.attn_inputs = None
        # Pre-allocated output buffer for HIP graph mode
        self.output_buffer = None
        self.output_buffer_size = 0

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self.attn_inputs = attn_inputs
        # Create decode parameters using pure Python implementation
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=False,
            enable_cuda_graph=self.enable_cuda_graph,
        )
        return fmha_params

    def get_or_create_output_buffer(self, query: torch.Tensor) -> torch.Tensor:
        """Get or create a pre-allocated output buffer for HIP graph mode."""
        required_size = query.numel()
        if self.output_buffer is None or self.output_buffer_size < required_size:
            # Allocate a larger buffer to handle different batch sizes
            self.output_buffer = torch.empty_like(query)
            self.output_buffer_size = required_size
        # Return a view of the buffer with the correct shape
        return self.output_buffer[: query.shape[0]]


class AiterDecodeAttnOpAsm(AiterDecodeAttnOpBase):
    """Aiter decode attention operation using ASM paged attention."""

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params
    ) -> torch.Tensor:
        # For HIP graph mode, we must access tensors through fmha_params.attn_inputs
        # because these tensors are updated in-place by the graph runner during replay.
        # The graph captured the kernel reading from these specific tensor addresses.
        attn_inputs = (
            self.attn_inputs
            if self.attn_inputs is not None
            else fmha_params.attn_inputs
        )
        seq_lens = attn_inputs.sequence_lengths_plus_1_d
        block_tables_id_device = attn_inputs.kv_cache_block_id_device

        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)
        max_num_blocks = block_tables_id_device.shape[1]
        K_QScale = None
        V_QScale = None
        if (
            key_cache.dtype == torch.float8_e4m3fnuz
            and value_cache.dtype == torch.float8_e4m3fnuz
        ):
            K_QScale = kv_cache.kv_scale_base.select(1, 0)
            V_QScale = kv_cache.kv_scale_base.select(1, 1)
        # Use pre-allocated buffer for HIP graph mode to ensure consistent memory addresses
        out_ = self.get_or_create_output_buffer(query)
        output = aiter.pa_fwd_asm(
            query,  # [num_seqs, num_heads, head_size]
            key_cache,  # [num_blocks, num_kv_heads, block_size, head_size/x, x]
            value_cache,  # [num_blocks, num_kv_heads, block_size, head_size/x, x]
            block_tables_id_device,
            seq_lens,
            max_num_blocks,
            1,
            K_QScale,
            V_QScale,
            out_,
            None,
            0,
        )
        output_reshaped = output.view(output.shape[0], -1)
        return output_reshaped


class AiterDecodeAttnOpNonAsm(AiterDecodeAttnOpBase):
    """Aiter decode attention operation using non-ASM paged attention."""

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params
    ) -> torch.Tensor:
        # For HIP graph mode, we must access tensors through fmha_params.attn_inputs
        # because these tensors are updated in-place by the graph runner during replay.
        # The graph captured the kernel reading from these specific tensor addresses.
        attn_inputs = (
            self.attn_inputs
            if self.attn_inputs is not None
            else fmha_params.attn_inputs
        )
        seq_lens = attn_inputs.sequence_lengths_plus_1_d
        block_tables_id_device = attn_inputs.kv_cache_block_id_device

        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)

        key_scale = kv_cache.kv_scale_base.select(1, 0)
        value_scale = kv_cache.kv_scale_base.select(1, 0)

        max_seq_len = fmha_params.max_seq_len
        scale = 1.0 / (self.head_dim**0.5)
        alibi_slopes = None
        k_scale = (
            key_scale
            if kv_cache and key_scale is not None
            else torch.tensor(1.0, device=query.device, dtype=query.dtype)
        )
        v_scale = (
            value_scale
            if kv_cache and value_scale is not None
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


class AiterPrefillImplAsm(FMHAPrefillImplBase):
    """Aiter prefill attention implementation using ASM."""

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            AiterPrefillAttnOp(attn_configs),
            FusedRopeKVCachePrefillOpAsm(attn_configs),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.AITER_ASM_PREFILL

    def support_cuda_graph(self) -> bool:
        return True


class AiterPrefillImplNonAsm(FMHAPrefillImplBase):
    """Aiter prefill attention implementation using non-ASM."""

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            AiterPrefillAttnOp(attn_configs),
            FusedRopeKVCachePrefillOpNonAsm(attn_configs),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.AITER_PREFILL

    def support_cuda_graph(self) -> bool:
        return True


class AiterDecodeImplAsm(FMHADecodeImplBase):
    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            AiterDecodeAttnOpAsm(attn_configs),
            FusedRopeKVCacheDecodeOpAsm(attn_configs),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.AITER_ASM_DECODE

    def support_cuda_graph(self) -> bool:
        return True


class AiterDecodeImplNonAsm(FMHADecodeImplBase):
    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            AiterDecodeAttnOpNonAsm(attn_configs),
            FusedRopeKVCacheDecodeOpNonAsm(attn_configs),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.AITER_DECODE

    def support_cuda_graph(self) -> bool:
        return True
