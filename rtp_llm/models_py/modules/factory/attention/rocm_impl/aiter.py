import logging
from typing import Any, List, Optional

import aiter
import torch
from aiter_meta.csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import (
    pa_decode_gluon_aot,
)

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOpAsm,
    FusedRopeKVCacheDecodeOpNonAsm,
    FusedRopeKVCachePrefillOpAsm,
    FusedRopeKVCachePrefillOpNonAsm,
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    paged_attention_atrex,
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
                # No prefix, kv_lengths equals input_lengths
                kv_lengths = input_lengths
                self.cu_seqlens_k = self.cu_seqlens_q.clone()
                self.max_seqlen_k = self.max_seq_len

            self.max_seqlen_q = self.max_seq_len
            self.seq_lens = None
            self.kv_cache_block_id_device = getattr(
                attn_inputs, "kv_cache_kernel_block_id_device", None
            )
            self.prefix_lengths = prefix_lengths
            self.token_q_num = input_lengths.sum().item()
            self.token_kv_num = kv_lengths.sum().item()
        # Decode mode
        else:
            input_lengths = attn_inputs.input_lengths
            sequence_lengths = getattr(attn_inputs, "sequence_lengths", None)
            kv_cache_block_id_device = getattr(
                attn_inputs, "kv_cache_kernel_block_id_device", None
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
        self.is_causal = attn_configs.is_causal

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self.fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
        )
        return self.fmha_params

    def reshape_qkv(self, qkv):
        """Reshape qkv tensor(s) to the format expected by flash attention.
        Returns:
            Tuple of (q, k, v) tensors, each with shape (total_tokens, num_heads, head_dim).
        """
        if isinstance(qkv, (tuple, list)) and len(qkv) == 3 and qkv[0].dim() == 3:

            # 3D case: (head_num, tokens, head_dim) - need to permute
            q_contiguous = qkv[0].permute(1, 0, 2).contiguous()
            k_contiguous = qkv[1].permute(1, 0, 2).contiguous()
            v_contiguous = qkv[2].permute(1, 0, 2).contiguous()

            # Apply slicing based on fmha_params
            q_contiguous = q_contiguous[: self.fmha_params.token_q_num]
            k_contiguous = k_contiguous[: self.fmha_params.token_kv_num]
            v_contiguous = v_contiguous[: self.fmha_params.token_kv_num]

            return q_contiguous, k_contiguous, v_contiguous

        if isinstance(qkv, (tuple, list)) and len(qkv) == 3 and qkv[0].dim() == 2:
            qkv = qkv[0]  # specific for fp8 attention

        tokens = qkv.size(0)
        q_size = self.head_num * self.head_dim
        kv_size = self.head_num_kv * self.head_dim
        # Split qkv into q, k, v
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        # Reshape to (tokens, num_heads, head_dim)
        q = q.view(tokens, self.head_num, self.head_dim)
        k = k.view(tokens, self.head_num_kv, self.head_dim)
        v = v.view(tokens, self.head_num_kv, self.head_dim)
        # Apply slicing based on fmha_params
        q = q[: self.fmha_params.token_q_num]
        k = k[: self.fmha_params.token_kv_num]
        v = v[: self.fmha_params.token_kv_num]
        return q.contiguous(), k.contiguous(), v.contiguous()

    def forward(self, qkv, kv_cache, fmha_params):
        q_tensor, k_tensor, v_tensor = self.reshape_qkv(qkv)

        cu_seqlens_q = fmha_params.cu_seqlens_q.to(q_tensor.device)
        cu_seqlens_k = fmha_params.cu_seqlens_k.to(k_tensor.device)
        max_seqlen_q = fmha_params.max_seqlen_q
        max_seqlen_k = fmha_params.max_seqlen_k

        _fp8 = aiter.dtypes.fp8
        if q_tensor.dtype == _fp8 and k_tensor.dtype == _fp8 and v_tensor.dtype == _fp8:
            res = aiter.flash_attn_varlen_fp8_pertensor_func(
                q_tensor,
                k_tensor,
                v_tensor,
                None,
                None,
                None,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=self.is_causal,
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
                causal=self.is_causal,  # 因果注意力掩码 - 用于自回归建模，每个位置只能关注自己和之前的位置
            )
        token_num = fmha_params.token_q_num
        final_result = res.reshape(token_num, self.head_num * self.head_dim)
        return final_result


class AiterPrefillAttnOpPaged:
    """Paged prefill attention"""

    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
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
        )
        return fmha_params

    def forward(self, qkv, kv_cache, fmha_params) -> torch.Tensor:
        q_tensor = qkv[0][: fmha_params.token_q_num]
        device = q_tensor.device

        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)

        x = 16 // key_cache.element_size()
        kv_sizes = key_cache.shape
        key_cache = key_cache.view(
            kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x
        )
        value_cache = value_cache.view(
            kv_sizes[0], kv_sizes[1], kv_sizes[2] // x, kv_sizes[3], x
        )

        cu_seqlens_q = fmha_params.cu_seqlens_q.to(device)
        batch_size = cu_seqlens_q.shape[0] - 1

        cu_seqlens_k = fmha_params.cu_seqlens_k.to(device)
        seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)

        block_table = fmha_params.kv_cache_block_id_device.to(
            dtype=torch.int32, device=device
        )

        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        kv_page_indices = torch.zeros(1, dtype=torch.int32, device=device)

        max_seqlen_q = fmha_params.max_seqlen_q
        max_seqlen_k = fmha_params.max_seqlen_k

        q_descale = None
        k_descale = None
        v_descale = None
        if key_cache.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            q_descale = torch.ones(1, dtype=torch.float32, device=device)
            k_descale = torch.ones(1, dtype=torch.float32, device=device)
            v_descale = torch.ones(1, dtype=torch.float32, device=device)

        res = aiter.mha_batch_prefill_func(
            q_tensor,
            key_cache,
            value_cache,
            cu_seqlens_q,
            kv_indptr,
            kv_page_indices,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
            block_table=block_table,
            seqlen_k=seqlen_k,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

        token_num = fmha_params.token_q_num
        return res.reshape(token_num, self.head_num * self.head_dim)


def _run_triton_paged_attention(
    query: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    kv_scale_base,
    num_seqs: int,
    query_length: int,
    seq_lens: torch.Tensor,
    block_tables_id_device: torch.Tensor,
    max_seq_len: int,
    num_kv_heads: int,
    context_partition_size: int,
) -> torch.Tensor:
    key_cache = paged_kv_cache.select(1, 0)
    value_cache = paged_kv_cache.select(1, 1)

    x = 16 // key_cache.element_size()
    kv_sizes = key_cache.shape
    key_cache = key_cache.view(
        kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x
    )
    value_cache = value_cache.view(
        kv_sizes[0], kv_sizes[1], kv_sizes[2] // x, kv_sizes[3], x
    )

    key_scale, value_scale = None, None
    if kv_scale_base is not None:
        key_scale = torch.ones(1, dtype=torch.float32, device=query.device)
        value_scale = torch.ones(1, dtype=torch.float32, device=query.device)

    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    query_group_size = num_query_heads // num_kv_heads

    query_dtype = query.dtype
    compute_type = (
        torch.bfloat16
        if query_dtype
        not in (
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
            torch.bfloat16,
            torch.float16,
        )
        else query_dtype
    )
    output_dtype = (
        torch.bfloat16
        if query_dtype
        in (
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
        )
        else query_dtype
    )

    softmax_scale = 1.0 / (head_size**0.5)
    max_context_partition_num = (
        max_seq_len + context_partition_size - 1
    ) // context_partition_size
    equivalent_query_group_size = query_length * query_group_size

    output = torch.empty(
        (num_seqs * query_length, num_query_heads, head_size),
        dtype=output_dtype,
        device=query.device,
    )
    exp_sums = torch.zeros(
        (
            num_seqs,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
        ),
        dtype=torch.float32,
        device=query.device,
    )
    max_logits = torch.full(
        (
            num_seqs,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
        ),
        -float("inf"),
        dtype=torch.float32,
        device=query.device,
    )
    temporary_output = torch.zeros(
        (
            num_seqs,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
            head_size,
        ),
        dtype=output_dtype,
        device=query.device,
    )

    context_lengths = seq_lens.to(dtype=torch.int32, device=query.device)
    block_tables = block_tables_id_device.to(dtype=torch.int32, device=query.device)

    query_scale = (
        torch.tensor([1.0], device=query.device, dtype=torch.float32)
        if query.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
        else None
    )

    pa_decode_gluon_aot(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lengths=context_lengths,
        block_tables=block_tables,
        softmax_scale=softmax_scale,
        query_length=query_length,
        max_context_partition_num=max_context_partition_num,
        context_partition_size=context_partition_size,
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
    return output


class AiterPrefillAttnOpTriton:
    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.context_partition_size = 256

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
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
        )
        return fmha_params

    def forward(self, qkv, kv_cache, fmha_params) -> torch.Tensor:
        block_tables_id_device = fmha_params.kv_cache_block_id_device
        num_seqs = (
            block_tables_id_device.shape[0] if block_tables_id_device is not None else 1
        )
        query = qkv[0]
        token_num = query.shape[0]
        device = query.device

        cu_seqlens_q = fmha_params.cu_seqlens_q.to(device)
        q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
        max_q_len = q_lens.max().item()
        real_token_num = cu_seqlens_q[-1].item()

        cu_seqlens_k = fmha_params.cu_seqlens_k.to(device)
        seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

        output = _run_triton_paged_attention(
            query,
            kv_cache.kv_cache_base,
            kv_cache.kv_scale_base,
            num_seqs,
            max_q_len,
            seq_lens,
            block_tables_id_device,
            fmha_params.max_seqlen_k,
            self.head_num_kv,
            self.context_partition_size,
        )

        if token_num != real_token_num:
            seq_ids = torch.arange(num_seqs, device=device).repeat_interleave(q_lens)
            within_seq_pos = (
                torch.arange(real_token_num, device=device) - cu_seqlens_q[seq_ids]
            )
            dst_indices = (
                seq_ids * max_q_len + (max_q_len - q_lens[seq_ids]) + within_seq_pos
            )
            output = output[dst_indices]

        return output.view(real_token_num, -1)


class AiterDecodeAttnOpBase:
    """Base class for Aiter decode attention operations."""

    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.tokens_per_block = attn_configs.kernel_tokens_per_block
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

    def reshape_kv_cache(self, paged_kv_cache):
        return common.reshape_paged_kv_cache(
            paged_kv_cache, self.head_num_kv, self.tokens_per_block, self.head_dim
        )


class AiterDecodeAttnOpAsm(AiterDecodeAttnOpBase):
    """Aiter decode attention operation using ASM paged attention."""

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[LayerKVCache], fmha_params
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
        self, query: torch.Tensor, kv_cache: Optional[LayerKVCache], fmha_params
    ) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens
        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)

        K_QScale = None
        V_QScale = None
        using_fp8_kvcache = False
        if (
            key_cache.dtype == torch.float8_e4m3fnuz
            and value_cache.dtype == torch.float8_e4m3fnuz
        ):
            K_QScale = kv_cache.kv_scale_base.select(1, 0)
            V_QScale = kv_cache.kv_scale_base.select(1, 1)
            using_fp8_kvcache = True

        block_tables_id_device = fmha_params.kv_cache_block_id_device

        max_seq_len = fmha_params.max_seq_len
        scale = 1.0 / (self.head_dim**0.5)
        alibi_slopes = None
        num_kv_heads = self.head_num_kv
        num_seqs, num_heads, head_size = query.shape
        block_size = value_cache.shape[2]
        if max_seq_len <= 16384 and (not using_fp8_kvcache):
            _PARTITION_SIZE_ROCM = 512
            max_num_partitions = (
                max_seq_len + _PARTITION_SIZE_ROCM - 1
            ) // _PARTITION_SIZE_ROCM
            x = 16 // key_cache.element_size()
            grp_size = num_heads // num_kv_heads
            kv_sizes = value_cache.shape
            # init output
            output = torch.empty_like(query).view((num_seqs, num_heads, head_size))
            exp_sums = torch.empty(
                size=(num_seqs, num_kv_heads, max_num_partitions, grp_size),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            # init tmp_output
            tmp_output = torch.empty(
                size=(num_seqs, num_kv_heads, max_num_partitions, grp_size, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            query = query.view((num_seqs, num_heads, head_size))
            key_cache = key_cache.view(
                (kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x)
            )
            value_cache = value_cache.view(
                (kv_sizes[0], kv_sizes[1], kv_sizes[3], kv_sizes[2])
            )
            paged_attention_atrex(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                seq_lens,
                block_tables_id_device,
                scale,
                max_seq_len,
                alibi_slopes,
            )
        else:
            _PARTITION_SIZE_ROCM = 256

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
            k_scale = (
                K_QScale
                if kv_cache and K_QScale is not None
                else torch.tensor(1.0, device=query.device, dtype=query.dtype)
            )
            v_scale = (
                V_QScale
                if kv_cache and V_QScale is not None
                else torch.tensor(1.0, device=query.device, dtype=query.dtype)
            )
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
        return True

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params
    ) -> torch.Tensor:
        num_seqs = query.shape[0]
        paged_kv_cache = self.reshape_kv_cache(kv_cache.kv_cache_base)
        output = _run_triton_paged_attention(
            query,
            paged_kv_cache,
            kv_cache.kv_scale_base,
            num_seqs,
            1,
            fmha_params.seq_lens,
            fmha_params.kv_cache_block_id_device,
            fmha_params.max_seq_len,
            self.head_num_kv,
            self.context_partition_size,
        )
        return output.view(num_seqs, -1)


class AiterPrefillImplAsm(FMHAImplBase):
    """Aiter prefill attention implementation using ASM."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterPrefillAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpAsm(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class AiterPrefillImplNonAsm(FMHAImplBase):
    """Aiter prefill attention implementation using non-ASM."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterPrefillAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpNonAsm(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class AiterPrefillImplPaged(FMHAImplBase):
    """Paged prefill impl: dispatches between CK batch-prefill and Triton PA at runtime.

    - seq_len <= 4: Triton PA (short query optimization)
    - Otherwise: CK batch-prefill (general paged prefill)
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.head_num_kv = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.tokens_per_block = attn_configs.tokens_per_block

        self.batch_prefill_impl = AiterPrefillAttnOpPaged(attn_configs)
        self.triton_prefill_impl = AiterPrefillAttnOpTriton(attn_configs)

        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpAsm(attn_configs)
        self.rope_kvcache_impl.use_paged_fmha = True

        self.attn_inputs = attn_inputs
        self.fmha_params = self.batch_prefill_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        has_prefix = (
            attn_inputs.prefix_lengths is not None
            and attn_inputs.prefix_lengths.numel() > 0
            and attn_inputs.prefix_lengths.max().item() > 0
        )
        return has_prefix

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        cu_seqlens_q = self.fmha_params.cu_seqlens_q
        batch_size = cu_seqlens_q.shape[0] - 1
        if batch_size > 0:
            max_q_len = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
            token_num = cu_seqlens_q[-1].item()
        else:
            max_q_len = 0
            token_num = 0
        use_triton = batch_size > 0 and 0 < max_q_len <= 4

        if self.need_rope_kv_cache:
            self.rope_kvcache_impl.pad_query = (
                use_triton and token_num != batch_size * max_q_len
            )
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        kv_cache.kv_cache_base = common.reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            self.head_num_kv,
            self.tokens_per_block,
            self.head_dim,
        )

        if use_triton:
            return self.triton_prefill_impl.forward(
                fmha_input, kv_cache, self.fmha_params
            )
        else:
            return self.batch_prefill_impl.forward(
                fmha_input, kv_cache, self.fmha_params
            )


class AiterDecodeImplAsm(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterDecodeAttnOpAsm(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOpAsm(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class AiterDecodeImplNonAsm(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterDecodeAttnOpNonAsm(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOpNonAsm(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class AiterDecodeImplTriton(FMHAImplBase):
    """Aiter decode attention implementation using Triton."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterDecodeAttnOpTriton(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOpAsm(attn_configs)

        self.attn_inputs = attn_inputs

        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
