import logging
import math
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
        graph_max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.enable_cuda_graph = enable_cuda_graph
        self.graph_max_seq_len = graph_max_seq_len

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

            if (
                self.enable_cuda_graph
                and self.graph_max_seq_len is not None
                and self.graph_max_seq_len > 0
            ):
                self.max_seq_len = self.graph_max_seq_len
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

    def fillParams(
        self,
        sequence_lengths,
        input_lengths,
        kv_cache_block_id_host=None,
        kv_cache_block_id_device=None,
    ):
        self.sequence_lengths = sequence_lengths
        self.input_lengths = input_lengths
        self.kv_cache_block_id_host = kv_cache_block_id_host
        if kv_cache_block_id_device is not None:
            self.kv_cache_block_id_device = kv_cache_block_id_device
        if self.seq_lens is not None and self.sequence_lengths is not None:
            self.seq_lens.copy_((self.sequence_lengths + 1).to(torch.device("cuda")))
            if (
                self.enable_cuda_graph
                and self.graph_max_seq_len is not None
                and self.graph_max_seq_len > 0
            ):
                self.max_seq_len = self.graph_max_seq_len
            else:
                self.max_seq_len = self.sequence_lengths.max().item() + 1
            self.max_seqlen_k = self.max_seq_len

    def check_recycle(self) -> bool:
        """Check whether the params can be recycled automatically."""
        return True


class AiterPrefillAttnOp:
    def __init__(self, attn_configs: AttentionConfigs, v1_kv_layout: bool = False):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.tokens_per_block = attn_configs.kernel_tokens_per_block
        self.is_causal = attn_configs.is_causal
        self.v1_kv_layout = v1_kv_layout

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self.fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
        )
        return self.fmha_params

    def _reshape_kv_cache_vectorized(self, kv_cache_base):
        """Reshape kv_cache_base into 5D VECTORIZED_LAYOUT for mha_batch_prefill.

        Returns (k_cache_5d, v_cache_5d):
            K: [num_blocks, num_kv_heads, head_dim/vs, page_size, vs]
            V: [num_blocks, num_kv_heads, page_size/vs, head_dim, vs]
        """
        block_num = kv_cache_base.shape[0]
        hk = self.head_num_kv
        ps = self.tokens_per_block
        hd = self.head_dim
        vs = 16 // kv_cache_base.element_size()
        expected_elems = 2 * hk * ps * hd

        flat = kv_cache_base[:, :expected_elems].reshape(block_num, 2, hk, ps * hd)

        # K: V1 kernel writes via getKLocalIdx<BASE> → vectorized [hd//vs, ps, vs].
        # This matches the target 5D shape directly via view.
        k_cache = flat[:, 0, :, :].view(block_num, hk, hd // vs, ps, vs)

        if self.v1_kv_layout:
            # V1 kernel writes V via non-template getVLocalIdx → linear [hd, ps].
            # Target layout for mha_batch_prefill: [ps//vs, hd, vs].
            # Permute [hd, ps] → [hd, ps//vs, vs] → [ps//vs, hd, vs].
            v_linear = flat[:, 1, :, :].view(block_num, hk, hd, ps)
            v_cache = (
                v_linear.reshape(block_num, hk, hd, ps // vs, vs)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
        else:
            # ASM kernel writes V via getVLocalIdx<BASE> → vectorized [ps//vs, hd, vs].
            v_cache = flat[:, 1, :, :].view(block_num, hk, ps // vs, hd, vs)

        return k_cache, v_cache

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

        # Unified path: always use mha_batch_prefill from paged KV cache
        k_cache, v_cache = self._reshape_kv_cache_vectorized(kv_cache.kv_cache_base)
        block_table = fmha_params.kv_cache_block_id_device
        cu_seqlens_q = fmha_params.cu_seqlens_q.to(q_tensor.device)

        # prefix_lengths: default to zeros when no prefix (unified logic)
        batch_size = cu_seqlens_q.shape[0] - 1
        if (
            fmha_params.prefix_lengths is not None
            and fmha_params.prefix_lengths.numel() > 0
        ):
            prefix_lengths_device = fmha_params.prefix_lengths.to(q_tensor.device)
        else:
            prefix_lengths_device = torch.zeros(
                batch_size, dtype=torch.int32, device=q_tensor.device
            )

        input_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqlen_k = (prefix_lengths_device + input_lengths).to(torch.int32)

        softmax_scale = 1.0 / math.sqrt(self.head_dim)
        kv_indptr = cu_seqlens_q
        kv_page_indices = torch.empty(0, dtype=torch.int32, device=q_tensor.device)

        res = aiter.mha_batch_prefill_func(
            q_tensor,
            k_cache,
            v_cache,
            cu_seqlens_q,
            kv_indptr,
            kv_page_indices,
            fmha_params.max_seqlen_q,
            fmha_params.max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=self.is_causal,
            window_size=(-1, 0),
            block_table=block_table,
            seqlen_k=seqlen_k,
        )
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
        self.max_seq_len = attn_configs.max_seq_len
        # Updated per-request in prepare(); keep default false to avoid stale state
        # before first input is prepared.
        self.enable_cuda_graph = False
        # Pre-allocated output tensor for CUDA graph capture/replay stability.
        # Allocated once in prepare() when enable_cuda_graph is True, then reused
        # in every forward() call so that graph replay always writes to the same
        # device address captured during graph recording.
        self._graph_output: Optional[torch.Tensor] = None

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        if not self.enable_cuda_graph:
            self._graph_output = None
        # Create decode parameters using pure Python implementation
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=False,
            enable_cuda_graph=self.enable_cuda_graph,
            graph_max_seq_len=self.max_seq_len,
        )
        fmha_params.max_seqlen_k = fmha_params.max_seq_len

        return fmha_params

    def _get_output(self, query: torch.Tensor) -> torch.Tensor:
        """Return a pre-allocated output tensor when running under CUDA graph,
        or a freshly allocated one otherwise.

        On the first call under graph mode the tensor is lazily allocated so
        that dtype/device are derived from the actual query tensor.  Subsequent
        calls reuse the same storage so that graph replay always writes to the
        address captured during recording."""
        if self.enable_cuda_graph:
            if self._graph_output is None or self._graph_output.shape != query.shape:
                self._graph_output = torch.empty_like(query)
            return self._graph_output
        return torch.empty_like(query)

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
        out_ = self._get_output(query)
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
        output = self._get_output(query).view((num_seqs, num_heads, head_size))
        if max_seq_len <= 16384 and (not using_fp8_kvcache):
            _PARTITION_SIZE_ROCM = 512
            max_num_partitions = (
                max_seq_len + _PARTITION_SIZE_ROCM - 1
            ) // _PARTITION_SIZE_ROCM
            x = 16 // key_cache.element_size()
            grp_size = num_heads // num_kv_heads
            kv_sizes = value_cache.shape
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
            # output already allocated above via _get_output(query); reuse it here.
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
        self.fmha_impl = AiterPrefillAttnOp(attn_configs, v1_kv_layout=True)
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


class AiterDecodeImplBase(FMHAImplBase):
    fmha_params: Any
    rope_params: Any

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        # Replay path must reuse capture-time FMHA params object to keep graph memory stable.
        self.fmha_params.fillParams(
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            attn_inputs.kv_cache_kernel_block_id_device,
        )
        if attn_inputs.kv_cache_kernel_block_id_device is not None:
            update_kv_cache_offset = getattr(
                self.rope_params, "update_kv_cache_offset", None
            )
            if not callable(update_kv_cache_offset):
                raise TypeError(
                    "AiterDecodeImplBase.prepare_cuda_graph expects rope_params to provide "
                    "update_kv_cache_offset(kv_cache_kernel_block_id_device)"
                )
            update_kv_cache_offset(attn_inputs.kv_cache_kernel_block_id_device)


class AiterDecodeImplAsm(AiterDecodeImplBase):
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


class AiterDecodeImplNonAsm(AiterDecodeImplBase):
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


class AiterDecodeImplTriton(AiterDecodeImplBase):
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
