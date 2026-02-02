import logging
from typing import Any, List, Optional

import aiter
import torch
import triton.language as tl

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

# Triton dtype mappings
TORCH_TO_TL_DTYPE = {
    torch.float8_e4m3fnuz: tl.float8e4b8,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
}


def _check_pa_decode_gluon_aot_available():
    try:
        from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import pa_decode_gluon_aot
        return True
    except ImportError:
        return False

def _get_pa_decode_gluon_aot():
    from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import pa_decode_gluon_aot
    return pa_decode_gluon_aot


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
    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
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


class AiterPrefillAttnOpTriton:
    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.context_partition_size = 256
        self.enable_cuda_graph = True

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        if not _check_pa_decode_gluon_aot_available():
            return False
        has_prefix = (
            attn_inputs.prefix_lengths is not None
            and attn_inputs.prefix_lengths.numel() > 0
            and attn_inputs.prefix_lengths.max().item() > 0
        )
        return has_prefix

    def prepare(self, attn_inputs: PyAttentionInputs):
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
            enable_cuda_graph=self.enable_cuda_graph,
        )
        return fmha_params

    def forward(self, qkv, kv_cache, fmha_params) -> torch.Tensor:
        block_tables_id_device = fmha_params.kv_cache_block_id_device
        # block_tables_id_device.shape: [batch_size, max_blocks]
        num_seqs = block_tables_id_device.shape[0] if block_tables_id_device is not None else 1

        query = qkv[0]
        total_tokens = query.shape[0]
        query_length = total_tokens // num_seqs
        max_seq_len = fmha_params.max_seqlen_k

        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)

        x = 16 // key_cache.element_size()
        kv_sizes = key_cache.shape
        key_cache = key_cache.view(kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x)
        value_cache = value_cache.view(kv_sizes[0], kv_sizes[1], kv_sizes[2] // x, kv_sizes[3], x)

        key_scale = None
        value_scale = None
        if kv_cache.kv_scale_base is not None:
            key_scale = kv_cache.kv_scale_base.select(1, 0)
            value_scale = kv_cache.kv_scale_base.select(1, 1)
            if key_scale.numel() > 1:
                key_scale = key_scale.unsqueeze(-1)
                value_scale = value_scale.unsqueeze(-1)

        num_query_heads = self.head_num
        head_size = self.head_dim
        num_kv_heads = self.head_num_kv
        query_group_size = num_query_heads // num_kv_heads

        cu_seqlens_k = fmha_params.cu_seqlens_k.to(query.device)
        seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

        query_dtype = query.dtype
        if query_dtype in TORCH_TO_TL_DTYPE:
            compute_type = TORCH_TO_TL_DTYPE[query_dtype]
        else:
            compute_type = tl.bfloat16

        if query_dtype == torch.float8_e4m3fnuz or query_dtype == torch.float8_e4m3fn:
            output_dtype = torch.bfloat16
        else:
            output_dtype = query_dtype

        softmax_scale = 1.0 / (head_size ** 0.5)

        max_context_partition_num = (
            max_seq_len + self.context_partition_size - 1
        ) // self.context_partition_size

        equivalent_query_group_size = query_length * query_group_size

        output = torch.empty(
            (num_seqs * query_length, num_query_heads, head_size),
            dtype=output_dtype,
            device=query.device,
        )

        # Gluon tensors need separate allocation for query_length > 1
        gluon_size = num_kv_heads * query_length * query_group_size
        output_gluon = torch.empty(
            (num_seqs, gluon_size, head_size),
            dtype=output_dtype,
            device=query.device,
        )
        query_gluon = torch.empty(
            (num_seqs, gluon_size, head_size),
            dtype=query.dtype,
            device=query.device,
        )

        query_scale_gluon = torch.tensor([1.0], dtype=torch.float32, device=query.device)

        exp_sums = torch.zeros(
            (num_seqs, num_kv_heads, max_context_partition_num, equivalent_query_group_size),
            dtype=torch.float32,
            device=query.device,
        )

        max_logits = torch.full(
            (num_seqs, num_kv_heads, max_context_partition_num, equivalent_query_group_size),
            -float("inf"),
            dtype=torch.float32,
            device=query.device,
        )

        temporary_output = torch.zeros(
            (num_seqs, num_kv_heads, max_context_partition_num, equivalent_query_group_size, head_size),
            dtype=output_dtype,
            device=query.device,
        )

        context_lengths = seq_lens.to(dtype=torch.int32, device=query.device)
        block_tables = block_tables_id_device.to(dtype=torch.int32, device=query.device)

        pa_decode_gluon_aot = _get_pa_decode_gluon_aot()
        pa_decode_gluon_aot(
            output=output,
            output_gluon=output_gluon,
            query=query,
            query_gluon=query_gluon,
            query_scale_gluon=query_scale_gluon,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lengths=context_lengths,
            block_tables=block_tables,
            softmax_scale=softmax_scale,
            query_length=query_length,
            max_context_length=max_seq_len,
            context_partition_size=self.context_partition_size,
            compute_type=compute_type,
            query_scale=None,
            key_scale=key_scale,
            value_scale=value_scale,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
            alibi_slopes=None,
            sinks=None,
        )

        total_tokens = query.shape[0]
        output_reshaped = output.view(total_tokens, -1)
        return output_reshaped


class AiterDecodeAttnOpBase:
    """Base class for Aiter decode attention operations."""

    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.enable_cuda_graph = True

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Create decode parameters using pure Python implementation
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=False,
            enable_cuda_graph=self.enable_cuda_graph,
        )
        return fmha_params


class AiterDecodeAttnOpAsm(AiterDecodeAttnOpBase):
    """Aiter decode attention operation using ASM paged attention."""

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params
    ) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens
        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)
        block_tables_id_device = fmha_params.kv_cache_block_id_device
        max_num_blocks = block_tables_id_device.shape[1]
        K_QScale = None
        V_QScale = None
        if (
            key_cache.dtype == torch.float8_e4m3fnuz
            and value_cache.dtype == torch.float8_e4m3fnuz
        ):
            K_QScale = kv_cache.kv_scale_base.select(1, 0)
            V_QScale = kv_cache.kv_scale_base.select(1, 1)
        out_ = torch.empty_like(query)
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
        seq_lens = fmha_params.seq_lens
        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)

        key_scale = kv_cache.kv_scale_base.select(1, 0)
        value_scale = kv_cache.kv_scale_base.select(1, 0)

        block_tables_id_device = fmha_params.kv_cache_block_id_device

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


class AiterDecodeAttnOpTriton(AiterDecodeAttnOpBase):

    def __init__(self, attn_configs: AttentionConfigs):
        super().__init__(attn_configs)
        self.context_partition_size = 256

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return _check_pa_decode_gluon_aot_available()

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params
    ) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens
        block_tables_id_device = fmha_params.kv_cache_block_id_device
        max_seq_len = fmha_params.max_seq_len

        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)

        x = 16 // key_cache.element_size()
        kv_sizes = key_cache.shape
        # key_cache: [num_blocks, num_kv_heads, kv_block_size, head_size]
        #         -> [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
        key_cache = key_cache.view(kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x)
        # value_cache: [num_blocks, num_kv_heads, kv_block_size, head_size]
        #           -> [num_blocks, num_kv_heads, kv_block_size // x, head_size, x] (transposed layout)
        value_cache = value_cache.view(kv_sizes[0], kv_sizes[1], kv_sizes[2] // x, kv_sizes[3], x)

        key_scale, value_scale = None, None
        if kv_cache.kv_scale_base is not None:
            key_scale = kv_cache.kv_scale_base.select(1, 0)
            value_scale = kv_cache.kv_scale_base.select(1, 1)
            if key_scale.numel() > 1:
                key_scale = key_scale.unsqueeze(-1)
                value_scale = value_scale.unsqueeze(-1)

        num_seqs, num_query_heads, head_size = query.shape
        num_kv_heads = self.head_num_kv
        query_group_size = num_query_heads // num_kv_heads
        query_length = 1
        query_dtype = query.dtype
        compute_type = TORCH_TO_TL_DTYPE.get(query_dtype, tl.bfloat16)
        output_dtype = torch.bfloat16 if query_dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn) else query_dtype
        softmax_scale = 1.0 / (head_size ** 0.5)

        max_context_partition_num = (
            max_seq_len + self.context_partition_size - 1
        ) // self.context_partition_size

        equivalent_query_group_size = query_length * query_group_size

        output = torch.empty(
            (num_seqs * query_length, num_query_heads, head_size),
            dtype=output_dtype,
            device=query.device,
        )

        output_gluon = output.view(
            num_seqs, num_kv_heads * query_length * query_group_size, head_size
        )
        query_gluon = query.view(
            num_seqs, num_kv_heads * query_length * query_group_size, head_size
        )

        query_scale = None
        query_scale_gluon = torch.tensor([1.0], dtype=torch.float32, device=query.device)

        exp_sums = torch.zeros(
            (num_seqs, num_kv_heads, max_context_partition_num, equivalent_query_group_size),
            dtype=torch.float32,
            device=query.device,
        )

        max_logits = torch.full(
            (num_seqs, num_kv_heads, max_context_partition_num, equivalent_query_group_size),
            -float("inf"),
            dtype=torch.float32,
            device=query.device,
        )

        temporary_output = torch.zeros(
            (num_seqs, num_kv_heads, max_context_partition_num, equivalent_query_group_size, head_size),
            dtype=output_dtype,
            device=query.device,
        )

        context_lengths = seq_lens.to(torch.int32)
        block_tables = block_tables_id_device.to(torch.int32)

        pa_decode_gluon_aot = _get_pa_decode_gluon_aot()
        pa_decode_gluon_aot(
            output=output,
            output_gluon=output_gluon,
            query=query,
            query_gluon=query_gluon,
            query_scale_gluon=query_scale_gluon,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lengths=context_lengths,
            block_tables=block_tables,
            softmax_scale=softmax_scale,
            query_length=query_length,
            max_context_length=max_seq_len,
            context_partition_size=self.context_partition_size,
            compute_type=compute_type,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
            alibi_slopes=None,
            sinks=None,
        )

        output_reshaped = output.view(num_seqs, -1)
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


class AiterPrefillImplTriton(FMHAPrefillImplBase):
    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            AiterPrefillAttnOpTriton(attn_configs),
            FusedRopeKVCachePrefillOpAsm(attn_configs),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.AITER_TRITON_PREFILL


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


class AiterDecodeImplTriton(FMHADecodeImplBase):
    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            AiterDecodeAttnOpTriton(attn_configs),
            FusedRopeKVCacheDecodeOpAsm(attn_configs),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.AITER_TRITON_DECODE
