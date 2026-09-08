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


_IDX_POOL_3D_CACHE = {}


def _indexer_pool_3d(kv_cache, pool):
    """Normalize the engine OPAQUE_KV pool [blocks, block_bytes] to the legacy
    kv_scale_base geometry [blocks, tokens_per_block, entry_bytes] the indexer
    kernels expect. Cached per backing storage: the pool never moves and hot
    paths hit this 61x per step."""
    if pool.dim() != 2:
        return pool
    key = (pool.data_ptr(), pool.size(0), pool.size(1))
    v = _IDX_POOL_3D_CACHE.get(key)
    if v is None:
        spb = int(getattr(kv_cache, "indexer_seq_size_per_block", 0)) or 64
        v = pool.view(pool.size(0), spb, -1)
        _IDX_POOL_3D_CACHE[key] = v
    return v


def _indexer_write_dest(kv_cache, slot_mapping, indexer_slot_mapping):
    """Destination (pool, slots) for indexer-K writes.

    With the decoupled dsa_indexer_k pool the write must go through the
    companion's own slot mapping; after main-KV prefix offload the legacy scale
    region's blocks are freed, so silently falling back would write/read freed
    memory - the exact bug class the decoupling removes. Hence: declared pool
    without its slot mapping is a hard error, not a preference.
    """
    pool = getattr(kv_cache, "indexer_cache_base", None)
    if pool is not None and pool.numel() > 0:
        assert (
            indexer_slot_mapping is not None and indexer_slot_mapping.numel() > 0
        ), "independent indexer pool declared but indexer_slot_mapping is missing"
        return _indexer_pool_3d(kv_cache, pool), indexer_slot_mapping
    return kv_cache.kv_scale_base, slot_mapping


def _indexer_score_src(kv_cache, attention_inputs):
    """Source (pool, block_table) for indexer scoring, same contract as writes."""
    pool = getattr(kv_cache, "indexer_cache_base", None)
    if pool is not None and pool.numel() > 0:
        table = getattr(attention_inputs, "indexer_cache_kernel_block_id_device", None)
        assert (
            table is not None and table.numel() > 0
        ), "independent indexer pool declared but its kernel block table is missing"
        return _indexer_pool_3d(kv_cache, pool), table
    return kv_cache.kv_scale_base, attention_inputs.kv_cache_kernel_block_id_device


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
        full_rope_pos_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CP variant of apply_rope_and_rotate_q_k.

        Uses ``full_rope_pos_ids`` (length = num_tokens) so RoPE runs on the
        entire buffer in-place, eliminating the per-layer gather/scatter EW
        kernels.  Padding rows receive pos=0 and produce garbage, but they are
        never consumed downstream (everything selects by ``total_local_ids``).

        Args:
            q: Query tensor [num_tokens, index_n_heads, index_head_dim]
            k: Key tensor [num_tokens, index_head_dim]
            full_rope_pos_ids: Full-length position IDs [num_tokens], built
                in plan() via ``full[total_local_ids] = positions_d[total_global_ids]``.
                None when no valid tokens exist (n_q == 0).

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        q_pe = q[:, :, : self.index_head_dim - self.rope_head_dim]
        k_pe = k[:, : self.index_head_dim - self.rope_head_dim]

        if self.cos_sin_cache is not None and full_rope_pos_ids is not None:
            rope._apply_rope_pos_ids_cos_sin_cache(
                q=q_pe,
                k=k_pe.unsqueeze(1),
                q_rope=q_pe,
                k_rope=k_pe.unsqueeze(1),
                cos_sin_cache=self.cos_sin_cache,
                pos_ids=full_rope_pos_ids,
                interleave=not self.is_neox_style,
            )

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
        indexer_slot_mapping: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Quantize and cache only the key tensor (fast path for decode).

        Args:
            key: Key tensor in BF16/FP16 [num_tokens, index_head_dim]
            kv_cache: KV cache object with kv_scale_base
            slot_mapping: Physical slot indices [num_tokens]
            indexer_slot_mapping: Slots into the independent indexer pool, when declared
        """
        assert kv_cache is not None, "kv_cache is required"
        dest_pool, dest_slots = _indexer_write_dest(
            kv_cache, slot_mapping, indexer_slot_mapping
        )
        rtp_llm_ops.indexer_k_quant_and_cache(
            key,  # Original key in BF16/FP16 [num_tokens, index_head_dim]
            dest_pool,  # [num_blocks, block_size, cache_stride]
            dest_slots,  # [num_tokens] physical slot indices
            self.block_size,  # quantization block size (128)
            self.scale_fmt,  # "ue8m0" for power-of-2 scaling
        )

    def quant_q_k(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache: KVCache,
        slot_mapping: torch.Tensor,
        indexer_slot_mapping: Optional[torch.Tensor] = None,
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
        dest_pool, dest_slots = _indexer_write_dest(
            kv_cache, slot_mapping, indexer_slot_mapping
        )
        rtp_llm_ops.indexer_k_quant_and_cache(
            key,  # Original key in BF16/FP16 [num_tokens, index_head_dim]
            dest_pool,  # [num_blocks, block_size, cache_stride]
            dest_slots,  # [num_tokens] physical slot indices
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
        kv_restore_unpad_indices: torch.Tensor,
        indexer_slot_mapping: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Context-parallel variant: all-gather only K from all CP ranks, restore to logical
        order and write to KV cache; quantize Q locally (no all_gather of Q).

        Each rank keeps its local query and only quantizes it; all ranks all-gather
        key so that full K is written to cache once per rank, ensuring decode and
        indexer topk see the same full K.

        Args:
            query: Local query tensor [local_tokens, index_n_heads, index_head_dim]
            key: Local key tensor [local_tokens, index_head_dim]
            kv_cache: KV cache object
            slot_mapping: Physical slot indices [total_tokens] for full context
            kv_restore_unpad_indices: Index tensor mapping all-gathered key rows to logical order.

        Returns:
            Tuple of (q_fp8, q_scale) for local context only, shapes [local_tokens, ...].
        """
        assert kv_cache is not None, "kv_cache is required"
        gathered_key = all_gather(key.contiguous(), group=Group.TP)
        gathered_key = gathered_key.reshape(-1, key.size(-1))
        restored_key = gathered_key[kv_restore_unpad_indices]  # element wise

        dest_pool, dest_slots = _indexer_write_dest(
            kv_cache, slot_mapping, indexer_slot_mapping
        )
        rtp_llm_ops.indexer_k_quant_and_cache(
            restored_key,
            dest_pool,
            dest_slots,
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
            attention_inputs: Attention inputs with decode_cu_seqlens_device, kv_cache_kernel_block_id_device

        Returns:
            TopK indices tensor
        """
        from rtp_llm.models_py.kernels.cuda.fast_topk import fast_topk_transform_fused

        weights = weights.view(-1, self.index_n_heads)
        # Independent indexer pool when declared; legacy scale region otherwise.
        # After main-KV offload only the companion pool still holds every token's
        # indexer-K, so scoring must follow the same source as the writes.
        kv_cache_fp8, block_table = _indexer_score_src(kv_cache, attention_inputs)

        num_heads_kv = 1
        head_dim_with_sf = (
            self.index_head_dim + self.index_head_dim // self.block_size * 4
        )
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], self.blocksize, num_heads_kv, head_dim_with_sf
        ).view(dtype=torch.uint8)

        max_seq_len = block_table.shape[1] * self.blocksize

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
            block_table,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )

        assert (
            fmha_params.expanded_seq_lens.device == logits.device
        ), "expanded_seq_lens must be on the same device as logits"
        assert (
            attention_inputs.decode_cu_seqlens_device.device == logits.device
        ), "cu_seqlens must be on the same device as logits"

        topk_result = fast_topk_transform_fused(
            score=logits,
            lengths=fmha_params.expanded_seq_lens,  # expanded_seq_lens
            cu_seqlens_q=attention_inputs.decode_cu_seqlens_device,  # bs + 1
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
            attention_inputs: Attention inputs with kv_cache_kernel_block_id_device, cu_kv_seqlens

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

        src_pool, src_table = _indexer_score_src(kv_cache, attention_inputs)
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            src_pool,  # [num_blocks, block_size, cache_stride]
            k_fp8,  # output [num_tokens, index_head_dim]
            k_scale,  # output [num_tokens, scale_size]
            src_table,  # [batch_size, num_blocks]
            attention_inputs.cu_kv_seqlens_device,
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

    def _get_topk_ragged_cp(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        total_local_ids: torch.Tensor,
        cu_kv_seqlens_global: torch.Tensor,
        num_kv_tokens: int,
        precomputed_ks: torch.Tensor,
        precomputed_ke: torch.Tensor,
        precomputed_lengths: torch.Tensor,
        precomputed_topk_off: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute TopK indices for ragged attention (prefill phase) with context parallel
        chunking. Splits q by CP chunk indices, runs fp8_mqa_logits + topk per chunk,
        and returns (topk0, topk1) for the two CP chunks so the caller can use them directly.

        Full KV for logits: deep_gemm.fp8_mqa_logits requires full kv_fp8 for mathematical
        correctness. Each row i uses K[ks[i]:ke[i]] (ragged); we only chunk the q dimension,
        so we still pass the full gathered kv_fp8 and per-chunk ks/ke for each chunk.

        Args:
            q_fp8: Local quantized query for this CP rank [local_tokens, ...].
            weights: Weights tensor for this rank, shape [local_tokens, ...].
            kv_cache: KV cache object
            fmha_params: FMHA parameters (ks/ke/etc. on the full tensor; not used for indexing).
            attention_inputs: Attention inputs with kv_cache_kernel_block_id_device.
            total_local_ids: Rows of ``q_fp8`` / ``weights`` to participate in logits.
            cu_kv_seqlens_global: Cumulative KV lengths for the full (gathered) sequence.
            num_kv_tokens: Full KV token count (length of logical KV after restore).
            precomputed_ks: ``fmha_params.ks[total_global_ids]``, precomputed in plan().
            precomputed_ke: ``fmha_params.ke[total_global_ids]``, precomputed in plan().
            precomputed_lengths: ``fmha_params.expanded_seq_lens[total_global_ids]``, precomputed in plan().
            precomputed_topk_off: ``fmha_params.topk_indices_offset[total_global_ids]``, precomputed in plan().

        Returns:
            TopK indices for the CP chunks, shape [len(total_local_ids), index_topk].
        """
        from rtp_llm.models_py.kernels.cuda.fast_topk import (
            fast_topk_transform_ragged_fused,
        )

        total_kv_tokens = num_kv_tokens
        assert total_kv_tokens > 0, "num_kv_tokens must be positive"

        device = q_fp8.device
        weights_sq = weights.squeeze(-1)

        q0 = q_fp8[total_local_ids].contiguous()
        weights_sq0 = weights_sq[total_local_ids].contiguous()

        # Full KV from cache (KV not split).
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
        cp_src_pool, cp_src_table = _indexer_score_src(kv_cache, attention_inputs)
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            cp_src_pool,
            k_fp8,
            k_scale,
            cp_src_table,
            cu_kv_seqlens_global,
        )
        kv_fp8_full = (k_fp8, k_scale.view(torch.float32))

        def run_part_logits_topk(
            q_part: torch.Tensor,
            weights_part: torch.Tensor,
            ks: torch.Tensor,
            ke: torch.Tensor,
            lengths: torch.Tensor,
            topk_off: torch.Tensor,
        ) -> torch.Tensor:
            logits_p = deep_gemm.fp8_mqa_logits(
                q_part,
                kv_fp8_full,
                weights_part,
                ks,
                ke,
                clean_logits=False,
            )
            return fast_topk_transform_ragged_fused(
                score=logits_p,
                lengths=lengths,
                topk_indices_offset=topk_off,
                topk=self.index_topk,
                row_starts=ks,
            )

        if total_local_ids.size(0) > 0:
            topk = run_part_logits_topk(
                q0,
                weights_sq0,
                precomputed_ks,
                precomputed_ke,
                precomputed_lengths,
                precomputed_topk_off,
            )
        else:
            topk = None
        return topk
