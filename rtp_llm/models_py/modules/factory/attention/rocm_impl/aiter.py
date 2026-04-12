import logging
from typing import Any, List, Optional

import aiter
import torch
from aiter.ops.triton.attention.mha import (
    flash_attn_with_kvcache as triton_flash_attn_with_kvcache,
)
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
        self.tokens_per_block = attn_configs.kernel_tokens_per_block
        self.is_causal = attn_configs.is_causal

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self.fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
        )
        return self.fmha_params

    def _split_qkv_fp8(self, qkv_fp8):
        """Split FP8 QKV buffer into separate Q, K, V tensors."""
        token_num = qkv_fp8.shape[0]
        qkv_reshaped = qkv_fp8.reshape(
            token_num, self.head_num + 2 * self.head_num_kv, self.head_dim
        )
        query = qkv_reshaped[:, : self.head_num, :]
        key = qkv_reshaped[:, self.head_num : self.head_num + self.head_num_kv, :]
        value = qkv_reshaped[
            :,
            self.head_num + self.head_num_kv : self.head_num + 2 * self.head_num_kv,
            :,
        ]
        return query, key, value

    def _split_raw_qkv(self, qkv, token_q_num, token_kv_num):
        """Split a raw concatenated QKV tensor into separate Q, K, V.

        Used for encoder-only models (e.g. BERT) where kv_cache is None and QKV
        arrives as a single flat tensor from qkv_proj.
        """
        token_num = qkv.size(0)
        q_size = self.head_num * self.head_dim
        kv_size = self.head_num_kv * self.head_dim
        query, key, value = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        query = query.view(token_num, self.head_num, self.head_dim)[:token_q_num]
        key = key.view(token_num, self.head_num_kv, self.head_dim)[:token_kv_num]
        value = value.view(token_num, self.head_num_kv, self.head_dim)[:token_kv_num]
        return query.contiguous(), key.contiguous(), value.contiguous()

    def _forward_varlen(self, qkv, fmha_params):
        """Fallback path using flash_attn_varlen_func for models without KV cache.

        Handles raw QKV tensor (from qkv_proj) by splitting and reshaping, then
        dispatches to aiter.flash_attn_varlen_func.
        """
        if isinstance(qkv, (tuple, list)):
            qkv = qkv[0]
        query, key, value = self._split_raw_qkv(
            qkv, fmha_params.token_q_num, fmha_params.token_kv_num
        )
        cu_seqlens_q = fmha_params.cu_seqlens_q.to(query.device)
        cu_seqlens_k = fmha_params.cu_seqlens_k.to(query.device)
        res = aiter.flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            fmha_params.max_seqlen_q,
            fmha_params.max_seqlen_k,
            dropout_p=0.0,
            causal=self.is_causal,
        )
        return res.reshape(fmha_params.token_q_num, self.head_num * self.head_dim)

    def forward(self, qkv, kv_cache, fmha_params):
        q_tensor = qkv[0] if isinstance(qkv, (tuple, list)) else qkv

        # FP8 path: C++ returns full qkv_buf_fp8 (Q+K+V concatenated in FP8).
        # Split into Q/K/V and use flash_attn_varlen_fp8_pertensor_func.
        if q_tensor.dtype == torch.float8_e4m3fnuz:
            query, key, value = self._split_qkv_fp8(q_tensor)
            cu_seqlens_q = fmha_params.cu_seqlens_q.to(query.device)
            cu_seqlens_k = fmha_params.cu_seqlens_k.to(query.device)
            res = aiter.flash_attn_varlen_fp8_pertensor_func(
                query,
                key,
                value,
                None,
                None,
                None,
                cu_seqlens_q,
                cu_seqlens_k,
                fmha_params.max_seqlen_q,
                fmha_params.max_seqlen_k,
                causal=self.is_causal,
            )
            return res.reshape(fmha_params.token_q_num, self.head_num * self.head_dim)

        if kv_cache is None:
            return self._forward_varlen(qkv, fmha_params)

        # Unified paged path: reshape raw 2D buffer if needed, then select K/V and
        # view into 5D vectorized layout for mha_batch_prefill_func.
        kv_cache_base = common.reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            self.head_num_kv,
            self.tokens_per_block,
            self.head_dim,
        )
        # kv_cache_base: [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
        k_cache = kv_cache_base.select(1, 0)
        v_cache = kv_cache_base.select(1, 1)
        x = 16 // k_cache.element_size()
        kv_sizes = k_cache.shape
        # K: [num_blocks, num_kv_heads, hd//x, ps, x]
        k_cache = k_cache.view(
            kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x
        )
        # V: [num_blocks, num_kv_heads, ps//x, hd, x]
        v_cache = v_cache.view(
            kv_sizes[0], kv_sizes[1], kv_sizes[2] // x, kv_sizes[3], x
        )

        device = q_tensor.device
        block_table = fmha_params.kv_cache_block_id_device

        cu_seqlens_q = fmha_params.cu_seqlens_q.to(device)
        batch_size = cu_seqlens_q.shape[0] - 1

        cu_seqlens_k = fmha_params.cu_seqlens_k.to(device)
        seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)

        # Build CSR-format paging metadata (SGLANG_PAGE_TABLE_1D) from block_table.
        # The CK batch_prefill kernel is compiled with SGLANG mode, so kv_indptr and
        # kv_page_indices must be correctly populated — passing them as zeros causes
        # the kernel to compute seqlen_k=0 and return all-zero output.
        ps = self.tokens_per_block
        pages_per_seq = (seqlen_k + ps - 1) // ps  # [batch_size], int32
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        torch.cumsum(pages_per_seq, dim=0, out=kv_indptr[1:])
        total_pages = kv_indptr[-1].item()
        kv_page_indices = torch.zeros(total_pages, dtype=torch.int32, device=device)
        if block_table is not None:
            for b in range(batch_size):
                start = kv_indptr[b].item()
                end = kv_indptr[b + 1].item()
                kv_page_indices[start:end] = block_table[b, : end - start]

        kv_last_page_lens = (seqlen_k % ps).to(torch.int32)
        kv_last_page_lens[kv_last_page_lens == 0] = ps

        torch.cuda.synchronize()
        res = aiter.mha_batch_prefill_func(
            q_tensor,
            k_cache,
            v_cache,
            cu_seqlens_q,
            kv_indptr,
            kv_page_indices,
            fmha_params.max_seqlen_q,
            fmha_params.max_seqlen_k,
            causal=self.is_causal,
            kv_last_page_lens=kv_last_page_lens,
        )
        torch.cuda.synchronize()

        return res.reshape(fmha_params.token_q_num, self.head_num * self.head_dim)


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

        torch.cuda.synchronize()
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
        torch.cuda.synchronize()

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
        self.tokens_per_block = attn_configs.kernel_tokens_per_block
        self.is_causal = attn_configs.is_causal

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
        query = qkv[0] if isinstance(qkv, (tuple, list)) else qkv
        token_num = query.shape[0]
        device = query.device
        paged_kv_cache = common.reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            self.head_num_kv,
            self.tokens_per_block,
            self.head_dim,
        )
        key_cache = paged_kv_cache.select(1, 0).permute(0, 2, 1, 3).contiguous()
        value_cache = paged_kv_cache.select(1, 1).permute(0, 2, 1, 3).contiguous()

        cu_seqlens_q = fmha_params.cu_seqlens_q.to(device)
        q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
        max_q_len = q_lens.max().item()
        real_token_num = cu_seqlens_q[-1].item()

        cu_seqlens_k = fmha_params.cu_seqlens_k.to(device)
        seq_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)
        block_table = block_tables_id_device.to(dtype=torch.int32, device=device)

        seq_ids = torch.arange(num_seqs, device=device).repeat_interleave(q_lens)
        within_seq_pos = (
            torch.arange(real_token_num, device=device) - cu_seqlens_q[seq_ids]
        )
        dst_indices = (
            seq_ids * max_q_len + (max_q_len - q_lens[seq_ids]) + within_seq_pos
        )
        if token_num == real_token_num:
            padded_query = torch.zeros(
                (num_seqs, max_q_len, self.head_num, self.head_dim),
                dtype=query.dtype,
                device=device,
            )
            padded_query.view(num_seqs * max_q_len, self.head_num, self.head_dim)[
                dst_indices
            ] = query
        else:
            padded_query = query.view(num_seqs, max_q_len, self.head_num, self.head_dim)

        output = triton_flash_attn_with_kvcache(
            padded_query,
            key_cache,
            value_cache,
            cache_seqlens=seq_lens,
            causal=self.is_causal,
            block_table=block_table,
        )
        output = output.view(num_seqs * max_q_len, self.head_num, self.head_dim)[
            dst_indices
        ]

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

        # Materialize compact K/V views so paged_attention_rocm does not inherit
        # the packed [K, V] block stride from the shared cache buffer.
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
        if (
            key_cache.dtype == torch.float8_e4m3fnuz
            and value_cache.dtype == torch.float8_e4m3fnuz
        ):
            K_QScale = kv_cache.kv_scale_base.select(1, 0)
            V_QScale = kv_cache.kv_scale_base.select(1, 1)

        block_tables_id_device = fmha_params.kv_cache_block_id_device

        max_seq_len = fmha_params.max_seq_len
        scale = 1.0 / (self.head_dim**0.5)
        alibi_slopes = None
        num_kv_heads = self.head_num_kv
        num_seqs, num_heads, head_size = query.shape

        # V1 kernel now writes V via getVLocalIdx<BASE> → vectorized [heads, ps/x, D, x].
        # Reshape K/V into 5D layout for paged_attention_rocm (V-shuffle auto-detected).
        # After select(1, 0/1), shape is [num_blocks, num_kv_heads, tokens_per_block, head_dim].
        x = 16 // key_cache.element_size()
        num_blocks = key_cache.shape[0]
        block_size = self.tokens_per_block

        # K: vectorized layout [num_blocks, num_kv_heads, hd//x, block_size, x]
        key_cache = key_cache.view(
            num_blocks, num_kv_heads, self.head_dim // x, block_size, x
        )
        # V: vectorized layout [num_blocks, num_kv_heads, block_size//x, hd, x]
        # paged_attention_rocm auto-detects 5D as V-shuffle
        value_cache = value_cache.view(
            num_blocks, num_kv_heads, block_size // x, self.head_dim, x
        )

        _PARTITION_SIZE_ROCM = 256
        max_num_partitions = (
            max_seq_len + _PARTITION_SIZE_ROCM - 1
        ) // _PARTITION_SIZE_ROCM

        output = torch.empty_like(query).view((num_seqs, num_heads, head_size))
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
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
            kv_cache_dtype,
            k_scale,
            v_scale,
            None,  # fp8_out_scale
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
        self, query: torch.Tensor, kv_cache: Optional[LayerKVCache], fmha_params
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
        layer_idx: int,
    ) -> torch.Tensor:
        if kv_cache is None:
            return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)

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
        layer_idx: int,
    ) -> torch.Tensor:
        if kv_cache is None:
            return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)

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
        self.tokens_per_block = attn_configs.kernel_tokens_per_block

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
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
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
        layer_idx: int,
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
        layer_idx: int,
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
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
