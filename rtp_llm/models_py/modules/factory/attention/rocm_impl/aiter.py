import logging
import math
from typing import Any, List, Optional

import aiter
import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
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

try:
    from aiter.ops.mha import mha_batch_prefill_func as _aiter_mha_batch_prefill

    _HAS_MHA_BATCH_PREFILL = True
except ImportError:
    _HAS_MHA_BATCH_PREFILL = False


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
        self.tokens_per_block = attn_configs.tokens_per_block
        self.is_causal = attn_configs.is_causal

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
            K: [num_blocks, num_kv_heads, head_dim/k_vector_size, page_size, k_vector_size]
            V: [num_blocks, num_kv_heads, page_size/k_vector_size, head_dim, k_vector_size]
        """
        block_num = kv_cache_base.shape[0]
        hk = self.head_num_kv
        ps = self.tokens_per_block
        hd = self.head_dim
        expected_elems = 2 * hk * ps * hd
        cache = kv_cache_base[:, :expected_elems].reshape(block_num, 2, hk, ps, hd)
        k_cache_4d = cache[:, 0, :, :, :]  # [block_num, hk, ps, hd]
        v_cache_4d = cache[:, 1, :, :, :]  # [block_num, hk, ps, hd]

        k_vector_size = 16 // kv_cache_base.element_size()
        k_cache = k_cache_4d.reshape(
            block_num, hk, hd // k_vector_size, ps, k_vector_size
        )
        v_cache = v_cache_4d.reshape(
            block_num, hk, ps // k_vector_size, hd, k_vector_size
        )
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

    def forward(self, qkv, kv_cache, fmha_params):
        q_tensor = qkv[0]  # Always packed Q: [total_q, head_num, head_dim]

        # FP8 special path: needs linear KV from qkv_buf_fp8
        if q_tensor.dtype == torch.float8_e4m3fnuz:
            query, key, value = self._split_qkv_fp8(q_tensor)
            cu_seqlens_q = fmha_params.cu_seqlens_q.to(query.device)
            cu_seqlens_k = fmha_params.cu_seqlens_k.to(query.device)
            res = aiter.flash_attn_varlen_fp8_pertensor_func(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                fmha_params.max_seqlen_q,
                fmha_params.max_seqlen_k,
                causal=self.is_causal,
            )
            return res.reshape(fmha_params.token_q_num, self.head_num * self.head_dim)

        # Unified path: always use mha_batch_prefill from paged KV cache
        if not _HAS_MHA_BATCH_PREFILL:
            raise RuntimeError(
                "mha_batch_prefill_func is not available. "
                "Please install aiter >= 0.1.11 with mha_batch_prefill support."
            )

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

        res = _aiter_mha_batch_prefill(
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


class AiterDecodeAttnOpBase:
    """Base class for Aiter decode attention operations."""

    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.tokens_per_block = attn_configs.tokens_per_block
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
        # attention exposes kv_cache_base as opaque int8 bytes with shape:
        #   per-layer: [block_num, kv_block_stride_bytes]
        # Flashinfer expects a 4D/5D paged KV cache; our non-hybrid canonical layout is:
        #   per-layer: [block_num, 2, local_kv_head_num, tokens_per_block, head_dim]
        block_num = paged_kv_cache.shape[0]
        expected_elems_per_block = (
            2 * self.head_num_kv * self.tokens_per_block * self.head_dim
        )
        # Hybrid stride is max(full, linear). For full-attn layers, the actual used region is a prefix.
        # So we slice the prefix and reshape into the canonical 5D paged KV cache expected by flashinfer.
        if paged_kv_cache.shape[1] < expected_elems_per_block:
            raise ValueError(
                f"packed kv_cache_base has insufficient stride: "
                f"got stride={paged_kv_cache.shape[1]} elems, need={expected_elems_per_block} elems"
            )
        paged_kv_cache = paged_kv_cache[:, :expected_elems_per_block].reshape(
            block_num,
            2,
            self.head_num_kv,
            self.tokens_per_block,
            self.head_dim,
        )
        return paged_kv_cache


class AiterDecodeAttnOpAsm(AiterDecodeAttnOpBase):
    """Aiter decode attention operation using ASM paged attention."""

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params
    ) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens

        paged_kv_cache = self.reshape_kv_cache(kv_cache.kv_cache_base)
        key_cache = paged_kv_cache.select(1, 0)
        value_cache = paged_kv_cache.select(1, 1)
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
        paged_kv_cache = self.reshape_kv_cache(kv_cache.kv_cache_base)
        key_cache = paged_kv_cache.select(1, 0)
        value_cache = paged_kv_cache.select(1, 1)

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
        kv_cache: Optional[KVCache],
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
        kv_cache: Optional[KVCache],
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
        kv_cache: Optional[KVCache],
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
        kv_cache: Optional[KVCache],
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
