"""CUDA-specific indexer operations for DeepSeek-V3.2 DSA mechanism."""

from typing import Any, Optional, Tuple

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, barrier
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.ops.compute_ops import KVCache, rtp_llm_ops

# Try to import CUDA dependencies, but don't fail if running on CPU
try:
    import deep_gemm
except Exception as e:
    print(f"Warning: Failed to import deep_gemm (likely running on CPU): {e}")
    deep_gemm = None

try:
    import flashinfer.rope as rope
except Exception as e:
    print(f"Warning: Failed to import flashinfer.rope (likely running on CPU): {e}")
    rope = None


def _unpack_ue8m0_scale(sf_packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack UE8M0 scale format.

    Args:
        sf_packed: Packed scale tensor (..., num_scales), dtype=int32

    Returns:
        Unpacked scale tensor in float32
    """
    # Extract the lowest byte via bitwise ops to avoid view.
    sf_u8 = (sf_packed & 0xFF).to(torch.int32)  # extract lowest byte
    # Shift left to float32 exponent position (bits 23-30).
    sf_i32 = sf_u8 << 23
    # Reinterpret as float32.
    sf_fp32 = sf_i32.view(torch.float32)
    return sf_fp32


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Hadamard transform for activation rotation.

    Args:
        x: Input tensor in bfloat16

    Returns:
        Rotated activation tensor
    """
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."

    return hadamard_transform(x, scale=hidden_size**-0.5)


class IndexerOp(nn.Module):
    """
    Indexer operations for DeepSeek-V3.2 DSA mechanism.
    Provides low-level operations for quantization and TopK computation.
    """

    def __init__(
        self,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        rope_head_dim: int,
        cos_sin_cache: Optional[torch.Tensor] = None,
        blocksize: int = 64,
        block_size: int = 128,
        scale_fmt: str = "ue8m0",
        is_neox_style: bool = True,
    ):
        """
        Initialize IndexerOp.

        Args:
            index_n_heads: Number of indexer heads
            index_head_dim: Dimension of indexer heads
            index_topk: TopK value for sparse attention
            rope_head_dim: Dimension of RoPE embeddings
            cos_sin_cache: Precomputed cos/sin cache for RoPE (optional)
            blocksize: Page size (default: 64)
            block_size: Quantization block size (default: 128)
            scale_fmt: FP8 quantization format (default: "ue8m0")
        """
        super().__init__()
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.rope_head_dim = rope_head_dim
        self.cos_sin_cache = cos_sin_cache
        self.blocksize = blocksize
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.is_neox_style = is_neox_style
        self.kv_len = None  # set in quant_q_k_cp (total KV length) for CP topk methods
        self.total_kv_len = None
        self.q0_idx = None
        self.q1_idx = None
        self.q0_idx_global = None
        self.q1_idx_global = None
        self.kv_restore_unpad_indices = None
        self.total_global_ids = None
        self.total_local_ids = None
        self.cu_kv_seqlens_global = None
        self._cu_local_kv_seqlens = None
        self._total_local_kv = None
        self._kv_allgather_restore_indices = None
        self._local_indexer_slot_mapping = None

    def apply_rope_and_rotate_q_k(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE and Hadamard transform to query and key tensors.

        Args:
            q: Query tensor [num_tokens, index_n_heads, index_head_dim]
            k: Key tensor [num_tokens, index_head_dim]
            positions: Position IDs for RoPE

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        # Extract position embedding part (exclude rope_head_dim from the end)
        q_pe = q[:, :, : self.index_head_dim - self.rope_head_dim]
        k_pe = k[:, : self.index_head_dim - self.rope_head_dim]

        # Apply RoPE (same as vllm indexer rope)
        if self.cos_sin_cache is not None:
            rope._apply_rope_pos_ids_cos_sin_cache(
                q=q_pe,
                k=k_pe.unsqueeze(1),
                q_rope=q_pe,
                k_rope=k_pe.unsqueeze(1),
                cos_sin_cache=self.cos_sin_cache,
                pos_ids=positions,
                interleave=not self.is_neox_style,
            )

        # Apply Hadamard transform (activation rotation)
        query = _rotate_activation(q)
        key = _rotate_activation(k)

        return query, key

    def apply_rope_and_rotate_q_k_cp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE and Hadamard transform to query and key tensors.
        Split by q0_idx/q1_idx so only valid (unpadded) tokens get RoPE; write back in-place to q's PE part.

        Args:
            q: Query tensor [num_tokens, index_n_heads, index_head_dim]
            k: Key tensor [num_tokens, index_head_dim]
            positions: Position IDs for RoPE (local order, length = num_tokens)

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        # q_pe / k_pe are views of the PE part; we will modify them in-place via q_pe0/q_pe1
        pe_dim = self.index_head_dim - self.rope_head_dim
        q_pe = q[:, :, :pe_dim]
        k_pe = k[:, :pe_dim]

        if self.cos_sin_cache is not None and self.total_local_ids.size(0) > 0:
            q_pe_local = q_pe[self.total_local_ids]  # element wise
            k_pe_local = k_pe[self.total_local_ids]  # element wise
            k_rope = k_pe_local.unsqueeze(1)
            pos_ids_q0_global = positions[self.total_global_ids]  # element wise
            rope._apply_rope_pos_ids_cos_sin_cache(
                q=q_pe_local,
                k=k_rope,
                q_rope=q_pe_local,
                k_rope=k_rope,
                cos_sin_cache=self.cos_sin_cache,
                pos_ids=pos_ids_q0_global,
                interleave=not self.is_neox_style,
            )
            k_rope = k_rope.squeeze(1)
            k_pe[self.total_local_ids] = k_rope  # element wise
            q_pe[self.total_local_ids] = q_pe_local  # element wise

        # Apply Hadamard transform (activation rotation)
        query = _rotate_activation(q)
        key = _rotate_activation(k)

        return query, key

    def apply_rope_and_rotate_k(
        self,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RoPE and Hadamard transform to key tensor only.

        Args:
            k: Key tensor [num_tokens, index_head_dim]
            positions: Position IDs for RoPE

        Returns:
            Rotated key tensor
        """
        # Extract position embedding part (exclude rope_head_dim from the end)
        k_pe = k[:, : self.index_head_dim - self.rope_head_dim]

        # Apply RoPE (same as vllm indexer rope)
        if self.cos_sin_cache is not None:
            rope._apply_rope_pos_ids_cos_sin_cache(
                q=k_pe.unsqueeze(1),
                k=k_pe.unsqueeze(1),
                q_rope=k_pe.unsqueeze(1),
                k_rope=k_pe.unsqueeze(1),
                cos_sin_cache=self.cos_sin_cache,
                pos_ids=positions,
                interleave=not self.is_neox_style,
            )

        # Apply Hadamard transform (activation rotation)
        key = _rotate_activation(k)

        return key

    def quant_k_only(
        self,
        key: torch.Tensor,
        kv_cache: KVCache,
        slot_mapping: torch.Tensor,
    ) -> None:
        """
        Quantize and cache only the key tensor (fast path for decode).

        Args:
            key: Key tensor in BF16/FP16 [num_tokens, index_head_dim]
            kv_cache: KV cache object with kv_scale_base
            slot_mapping: Physical slot indices [num_tokens]
        """
        assert kv_cache is not None, "kv_cache is required"
        rtp_llm_ops.indexer_k_quant_and_cache(
            key,  # Original key in BF16/FP16 [num_tokens, index_head_dim]
            kv_cache.kv_scale_base,  # [num_blocks, block_size, cache_stride]
            slot_mapping,  # [num_tokens] physical slot indices
            self.block_size,  # quantization block size (128)
            self.scale_fmt,  # "ue8m0" for power-of-2 scaling
        )

    def quant_q_k(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache: KVCache,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize query and key tensors, and cache the key.

        Args:
            query: Query tensor [num_tokens, index_n_heads, index_head_dim]
            key: Key tensor [num_tokens, index_head_dim]
            kv_cache: KV cache object
            slot_mapping: Physical slot indices [num_tokens]

        Returns:
            Tuple of (q_fp8, q_scale) where:
            - q_fp8: Quantized query [num_tokens, index_n_heads, index_head_dim]
            - q_scale: Query scale [num_tokens, index_n_heads, 1]
        """
        # Quantize query
        query_flat = query.view(-1, self.index_head_dim)
        q_fp8, q_scale = sgl_per_token_group_quant_fp8(
            query_flat,
            group_size=self.block_size,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=(self.scale_fmt == "ue8m0"),
        )
        q_fp8 = q_fp8.view(-1, self.index_n_heads, self.index_head_dim)

        if self.scale_fmt == "ue8m0":
            q_scale = _unpack_ue8m0_scale(q_scale)
        q_scale = q_scale.view(-1, self.index_n_heads, 1)

        # Cache key
        assert kv_cache is not None, "kv_cache is required"
        rtp_llm_ops.indexer_k_quant_and_cache(
            key,  # Original key in BF16/FP16 [num_tokens, index_head_dim]
            kv_cache.kv_scale_base,  # [num_blocks, block_size, cache_stride]
            slot_mapping,  # [num_tokens] physical slot indices
            self.block_size,  # quantization block size (128)
            self.scale_fmt,  # "ue8m0" for power-of-2 scaling
        )

        return q_fp8, q_scale

    def quant_q_k_cp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache: KVCache,
        slot_mapping: torch.Tensor,
        attention_inputs: Any,
        kv_cache_sharded: bool = False,
        local_indexer_slot_mapping: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Context-parallel variant: write local K to indexer cache, quantize Q locally.

        When kv_cache_sharded=True (round-robin), local K is written directly to the
        sharded indexer cache using local_indexer_slot_mapping (no all_gather needed
        for cache write). The topk computation path gathers FP8 K from the sharded
        cache later via _get_topk_ragged_cp_roundrobin.

        When kv_cache_sharded=False (zigzag), all K is all-gathered, restored, and
        written to the full cache as before.

        Args:
            query: Local query tensor [local_tokens, index_n_heads, index_head_dim]
            key: Local key tensor [local_tokens, index_head_dim]
            kv_cache: KV cache object
            slot_mapping: Physical slot indices [total_tokens] for full context
            attention_inputs: Must have context_parallel_info with prefill_qkv_restore_indice
                and prefill_qkv_padding_mask
            kv_cache_sharded: If True, use direct local cache write (no all_gather for cache write).
            local_indexer_slot_mapping: [local_tokens] slot mapping with physical slots
                for owned tokens, -1 for non-owned. Required when kv_cache_sharded=True.

        Returns:
            Tuple of (q_fp8, q_scale) for local context only, shapes [local_tokens, ...].

        Side effect:
            Sets self.kv_len for CP topk methods.
        """
        self.kv_len = self.total_kv_len
        assert kv_cache is not None, "kv_cache is required"

        if kv_cache_sharded:
            assert local_indexer_slot_mapping is not None, (
                "local_indexer_slot_mapping is required when kv_cache_sharded=True"
            )
            rtp_llm_ops.indexer_k_quant_and_cache(
                key,
                kv_cache.kv_scale_base,
                local_indexer_slot_mapping,
                self.block_size,
                self.scale_fmt,
            )
        else:
            gathered_key = all_gather(key.contiguous(), group=Group.TP)
            gathered_key = gathered_key.reshape(-1, key.size(-1))
            restored_key = gathered_key[self.kv_restore_unpad_indices]

            rtp_llm_ops.indexer_k_quant_and_cache(
                restored_key,
                kv_cache.kv_scale_base,
                slot_mapping,
                self.block_size,
                self.scale_fmt,
            )

        query_flat = query.view(-1, self.index_head_dim)
        q_fp8, q_scale = sgl_per_token_group_quant_fp8(
            query_flat,
            group_size=self.block_size,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=(self.scale_fmt == "ue8m0"),
        )
        q_fp8 = q_fp8.view(-1, self.index_n_heads, self.index_head_dim)
        if self.scale_fmt == "ue8m0":
            q_scale = _unpack_ue8m0_scale(q_scale)
        q_scale = q_scale.view(-1, self.index_n_heads, 1)
        return q_fp8, q_scale

    def _get_topk_paged(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
    ) -> torch.Tensor:
        """
        Compute TopK indices for paged attention (decode phase).

        Args:
            q_fp8: Quantized query [num_tokens, index_n_heads, index_head_dim]
            weights: Weights tensor [num_tokens, index_n_heads, 1]
            kv_cache: KV cache object
            fmha_params: FMHA parameters with expanded_seq_lens, etc.
            attention_inputs: Attention inputs with decode_cu_seqlens_d, kv_cache_block_id_device

        Returns:
            TopK indices tensor
        """
        from rtp_llm.models_py.kernels.cuda.fast_topk import fast_topk_transform_fused

        weights = weights.view(-1, self.index_n_heads)
        kv_cache_fp8 = kv_cache.kv_scale_base

        num_heads_kv = 1
        head_dim_with_sf = (
            self.index_head_dim + self.index_head_dim // self.block_size * 4
        )
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], self.blocksize, num_heads_kv, head_dim_with_sf
        ).view(dtype=torch.uint8)

        max_seq_len = (
            attention_inputs.kv_cache_block_id_device.shape[1] * self.blocksize
        )

        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            fmha_params.kvlen_d,
            self.blocksize,
            deep_gemm.get_num_sms(),
        )

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8.unsqueeze(1),
            kv_cache_fp8.view(dtype=torch.uint8),
            weights,
            fmha_params.kvlen_d,
            attention_inputs.kv_cache_block_id_device,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )

        assert (
            fmha_params.expanded_seq_lens.device == logits.device
        ), "expanded_seq_lens must be on the same device as logits"
        assert (
            attention_inputs.decode_cu_seqlens_d.device == logits.device
        ), "cu_seqlens must be on the same device as logits"

        topk_result = fast_topk_transform_fused(
            score=logits,
            lengths=fmha_params.expanded_seq_lens,  # expanded_seq_lens
            cu_seqlens_q=attention_inputs.decode_cu_seqlens_d,  # bs + 1
            topk=self.index_topk,
            row_starts=None,
        )

        return topk_result

    def _get_topk_ragged(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
    ) -> torch.Tensor:
        """
        Compute TopK indices for ragged attention (prefill phase).
        This method handles gathering quantized keys from cache and computing TopK.

        Args:
            q_fp8: Quantized query [num_tokens, index_n_heads, index_head_dim]
            weights: Weights tensor [num_tokens, index_n_heads, 1]
            kv_cache: KV cache object
            fmha_params: FMHA parameters with ks, ke, expanded_seq_lens, topk_indices_offset
            attention_inputs: Attention inputs with kv_cache_block_id_device, cu_kv_seqlens

        Returns:
            TopK indices tensor
        """
        from rtp_llm.models_py.kernels.cuda.fast_topk import (
            fast_topk_transform_ragged_fused,
        )

        # Gather quantized key from cache for prefill
        num_tokens = q_fp8.shape[0]
        k_fp8 = torch.empty(
            (num_tokens, self.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device=q_fp8.device,
        )
        k_scale = torch.empty(
            (num_tokens, self.index_head_dim // self.block_size * 4),
            dtype=torch.uint8,
            device=q_fp8.device,
        )

        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            kv_cache.kv_scale_base,  # [num_blocks, block_size, cache_stride]
            k_fp8,  # output [num_tokens, index_head_dim]
            k_scale,  # output [num_tokens, scale_size]
            attention_inputs.kv_cache_block_id_device,  # [batch_size, num_blocks]
            attention_inputs.cu_kv_seqlens,
        )

        # Compute logits
        weights = weights.squeeze(-1)
        kv_fp8 = (k_fp8, k_scale.view(torch.float32))

        assert (
            fmha_params.ks is not None and fmha_params.ke is not None
        ), "ks/ke must be prepared in prefill"

        logits = deep_gemm.fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            weights,
            fmha_params.ks,
            fmha_params.ke,
            clean_logits=False,
        )

        assert (
            fmha_params.expanded_seq_lens.device == logits.device
        ), "expanded_seq_lens must be on the same device as logits"
        assert (
            fmha_params.topk_indices_offset.device == logits.device
        ), "topk_indices_offset must be on the same device as logits"
        assert (
            fmha_params.ks.device == logits.device
        ), "ks must be on the same device as logits"

        topk_result = fast_topk_transform_ragged_fused(
            score=logits,
            lengths=fmha_params.expanded_seq_lens,
            topk_indices_offset=fmha_params.topk_indices_offset,
            topk=self.index_topk,
            row_starts=fmha_params.ks,
        )

        return topk_result

    # ---- CP gather helpers ----

    def _gather_kv_fp8(
        self,
        total_kv_tokens: int,
        cache_tensor: torch.Tensor,
        block_table: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather full quantized K from a paged cache tensor.

        Args:
            total_kv_tokens: Total number of KV tokens to gather.
            cache_tensor: Paged cache [num_blocks, block_size, cache_stride].
            block_table: Block table [batch_size, max_blocks_per_req].
            device: Target device.

        Returns:
            (k_fp8, k_scale) tuple.
        """
        k_fp8 = torch.empty(
            (total_kv_tokens, self.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        k_scale = torch.empty(
            (total_kv_tokens, self.index_head_dim // self.block_size * 4),
            dtype=torch.uint8,
            device=device,
        )
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            cache_tensor,
            k_fp8,
            k_scale,
            block_table,
            self.cu_kv_seqlens_global,
        )
        return k_fp8, k_scale

    def _run_cp_logits_topk(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_fp8: Tuple[torch.Tensor, torch.Tensor],
        fmha_params: Any,
    ) -> Optional[torch.Tensor]:
        """Shared logits + topk computation for CP prefill (both zigzag and round-robin)."""
        from rtp_llm.models_py.kernels.cuda.fast_topk import (
            fast_topk_transform_ragged_fused,
        )

        if self.total_local_ids.size(0) == 0:
            return None

        weights_sq = weights.squeeze(-1)
        q0 = q_fp8[self.total_local_ids].contiguous()
        weights_sq0 = weights_sq[self.total_local_ids].contiguous()

        assert (
            fmha_params.ks is not None and fmha_params.ke is not None
        ), "ks/ke must be prepared in prefill"

        n_tokens_fmha = fmha_params.ks.shape[0]
        if self.total_global_ids.numel() > 0:
            max_idx = self.total_global_ids.max().item()
            if max_idx >= n_tokens_fmha:
                raise ValueError(
                    f"total_global_ids out of range for fmha_params.ks: "
                    f"max(total_global_ids)={max_idx}, fmha_params.ks.shape[0]={n_tokens_fmha}. "
                    "Check that attn_inputs used for fill_params use global "
                    "(prefill_actual_input_lengths_cpu) lengths."
                )

        ks = fmha_params.ks[self.total_global_ids]
        ke = fmha_params.ke[self.total_global_ids]
        lengths = fmha_params.expanded_seq_lens[self.total_global_ids]
        topk_off = fmha_params.topk_indices_offset[self.total_global_ids]

        logits = deep_gemm.fp8_mqa_logits(
            q0,
            kv_fp8,
            weights_sq0,
            ks,
            ke,
            clean_logits=False,
        )
        return fast_topk_transform_ragged_fused(
            score=logits,
            lengths=lengths,
            topk_indices_offset=topk_off,
            topk=self.index_topk,
            row_starts=ks,
        )

    def _get_topk_ragged_cp_zigzag(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
    ) -> Optional[torch.Tensor]:
        """Zigzag CP topk: read full K directly from the (redundant) real cache."""
        total_kv_tokens = self.kv_len
        assert total_kv_tokens is not None and total_kv_tokens > 0

        k_fp8, k_scale = self._gather_kv_fp8(
            total_kv_tokens,
            kv_cache.kv_scale_base,
            attention_inputs.kv_cache_block_id_device,
            q_fp8.device,
        )
        kv_fp8 = (k_fp8, k_scale.view(torch.float32))
        return self._run_cp_logits_topk(q_fp8, weights, kv_fp8, fmha_params)

    def _get_topk_ragged_cp_roundrobin(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        fmha_params: Any,
        kv_cache: Any = None,
        attention_inputs: Any = None,
    ) -> Optional[torch.Tensor]:
        """Round-robin CP topk: gather FP8 K from sharded cache, all_gather, restore to global order."""
        total_kv_tokens = self.kv_len
        assert total_kv_tokens is not None and total_kv_tokens > 0
        assert kv_cache is not None, "kv_cache is required for round-robin CP topk"
        assert attention_inputs is not None

        local_kv_len = self._total_local_kv
        k_fp8_local = torch.empty(
            (local_kv_len, self.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device=q_fp8.device,
        )
        k_scale_local = torch.empty(
            (local_kv_len, self.index_head_dim // self.block_size * 4),
            dtype=torch.uint8,
            device=q_fp8.device,
        )
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            kv_cache.kv_scale_base,
            k_fp8_local,
            k_scale_local,
            attention_inputs.kv_cache_block_id_device,
            self._cu_local_kv_seqlens,
        )

        k_fp8_all = all_gather(k_fp8_local.contiguous(), group=Group.TP)
        k_scale_all = all_gather(k_scale_local.contiguous(), group=Group.TP)

        k_fp8_all = k_fp8_all.reshape(-1, self.index_head_dim)
        k_scale_all = k_scale_all.reshape(-1, k_scale_local.shape[-1])
        k_fp8 = k_fp8_all[self._kv_allgather_restore_indices]
        k_scale = k_scale_all[self._kv_allgather_restore_indices]

        kv_fp8 = (k_fp8, k_scale.view(torch.float32))
        return self._run_cp_logits_topk(q_fp8, weights, kv_fp8, fmha_params)
