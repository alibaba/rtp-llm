"""CUDA-specific indexer operations for DeepSeek-V3.2 DSA mechanism."""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, barrier
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4.chunk_env import dsv4_chunk_tokens_from_env
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


_PD_DEBUG_INDEXER_LOG_COUNTS: Dict[str, int] = {}
_persistent_topk_workspace: Dict[torch.device, torch.Tensor] = {}


def _get_topk_workspace(device: torch.device) -> torch.Tensor:
    ws = _persistent_topk_workspace.get(device)
    if ws is None:
        ws = torch.empty(1 << 20, dtype=torch.uint8, device=device)
        _persistent_topk_workspace[device] = ws
    return ws


def _fp8_mqa_logits_chunk_rows() -> int:
    """Rows per FP8 MQA logits chunk.

    Long-context CP prefill can otherwise materialize a [local_q, full_k]
    logits matrix. Keep this opt-in for the shared base indexer path; the GLM5
    smoke target enables it explicitly with the same env used by DSV4.
    """
    return dsv4_chunk_tokens_from_env(
        "DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS",
        default=0,
        min_value=0,
    )


def _halve_chunk_rows(chunk_rows: int) -> int:
    next_rows = max(1, chunk_rows // 2)
    if next_rows >= 1024:
        return max(1024, (next_rows // 1024) * 1024)
    if next_rows >= 128:
        return max(128, (next_rows // 128) * 128)
    return next_rows


def _pd_debug_enabled() -> bool:
    return os.environ.get("RTP_LLM_PD_DEBUG", "0") == "1"


def _pd_debug_take(tag: str, limit: int = 16) -> bool:
    key = f"{tag}:{os.getpid()}"
    count = _PD_DEBUG_INDEXER_LOG_COUNTS.get(key, 0)
    if count >= limit:
        return False
    _PD_DEBUG_INDEXER_LOG_COUNTS[key] = count + 1
    return True


def _cuda_graph_capturing() -> bool:
    try:
        return bool(
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
    except Exception:
        return False


def _rank_tag() -> str:
    return (
        f"rank={os.environ.get('RANK', os.environ.get('WORLD_RANK', '?'))} "
        f"local_rank={os.environ.get('LOCAL_RANK', '?')}"
    )


def _tensor_summary(t: Optional[torch.Tensor]) -> str:
    if t is None:
        return "None"
    try:
        if t.is_cuda and _cuda_graph_capturing():
            return (
                f"shape={tuple(t.shape)} device={t.device} dtype={t.dtype} " "capture=1"
            )
        if t.numel() == 0:
            return f"shape={tuple(t.shape)} numel=0"
        flat = t.detach().reshape(-1)
        if flat.numel() <= 16:
            tc = flat.cpu() if flat.is_cuda else flat
            return f"shape={tuple(t.shape)} values={tc.tolist()}"
        head = flat[:4]
        tail = flat[-4:]
        if head.is_cuda:
            head = head.cpu()
            tail = tail.cpu()
        return (
            f"shape={tuple(t.shape)} numel={flat.numel()} "
            f"head={head.tolist()} tail={tail.tolist()}"
        )
    except Exception as exc:
        return f"shape={tuple(t.shape)} summary_error={exc}"


def _physical_block_table(attention_inputs: Any) -> torch.Tensor:
    """Return the physical paged-cache block table.

    Indexer cache reads use ``kv_cache`` as physical pages
    ``[num_blocks, block_size, ...]``.  The kernel-granularity table can be
    token-level when ``kernel_seq_size_per_block == 1``; using it here reads
    unrelated cache pages and corrupts sparse MLA top-k.
    """
    physical = getattr(attention_inputs, "kv_cache_block_id_device", None)
    if isinstance(physical, torch.Tensor) and physical.numel() > 0:
        return physical
    return attention_inputs.kv_cache_kernel_block_id_device


def _prefill_physical_block_table(attention_inputs: Any) -> torch.Tensor:
    return _physical_block_table(attention_inputs)


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


def _indexer_cache_blocks(
    kv_cache_fp8: torch.Tensor, blocksize: int, head_dim_with_sf: int
) -> torch.Tensor:
    """Normalize indexer cache storage to physical pages.

    LayerKVCache can expose the same contiguous cache either as physical pages
    [num_blocks, blocksize, stride] or as kernel-block pages
    [num_blocks * blocksize, 1, stride] when kernel_seq_size_per_block == 1.
    Indexer kernels and DeepGEMM paged logits need physical pages.
    """
    return kv_cache_fp8.view(-1, blocksize, head_dim_with_sf)


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


def _try_fused_prefill_rope_hadamard_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: Optional[torch.Tensor],
    cos_sin_cache: Optional[torch.Tensor],
    rope_head_dim: int,
    is_neox_style: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Try to use the fused prefill kernel; return None if not applicable.

    Fast-path conditions:
      - head_dim == 128 (kernel hardcoded for 7-stage butterfly)
      - cos_sin_cache is not None and dtype is fp32 (flashinfer convention)
      - positions is not None
      - rope_head_dim is even and > 0
    Otherwise returns None and caller falls back to the unfused 4-op chain.

    Supports both NeOX (half-split) and interleaved (even/odd) RoPE styles.
    Yields ~2.35x at T=4096, up to 4.2x at T=16384 on DSV3.2 (B300, eager mode).
    """
    if cos_sin_cache is None or positions is None:
        return None
    if cos_sin_cache.dtype != torch.float32:
        return None
    if q.dim() != 3 or k.dim() != 2:
        return None
    if q.shape[-1] != 128 or k.shape[-1] != 128:
        return None
    if rope_head_dim == 0 or rope_head_dim % 2 != 0:
        return None

    from rtp_llm.models_py.triton_kernels.sparse_mla.fused_prefill_rope_hadamard import (
        fused_prefill_rope_hadamard_qk,
    )

    return fused_prefill_rope_hadamard_qk(
        q,
        k,
        positions,
        cos_sin_cache,
        rope_head_dim,
        is_neox_style=is_neox_style,
    )


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

    def _head_dim_with_sf(self) -> int:
        return self.index_head_dim + self.index_head_dim // self.block_size * 4

    def _kv_cache_blocks(self, kv_cache: KVCache) -> torch.Tensor:
        return _indexer_cache_blocks(
            kv_cache.kv_scale_base, self.blocksize, self._head_dim_with_sf()
        )

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
        # Fast path: fused Triton kernel + cuBLAS GEMM (2 launches instead of 4)
        # Empirical: 2.35x at T=4096, up to 4.2x at T=16384 on DSV3.2 (eager mode).
        fused = _try_fused_prefill_rope_hadamard_qk(
            q,
            k,
            positions,
            self.cos_sin_cache,
            self.rope_head_dim,
            self.is_neox_style,
        )
        if fused is not None:
            return fused

        # Fallback: unfused 4-op chain (rope_q + rope_k + had_q + had_k)
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
        # Fast path: fused Triton kernel + cuBLAS GEMM (2 launches instead of 4)
        # Skips when full_rope_pos_ids is None (CP edge case: n_q == 0).
        fused = _try_fused_prefill_rope_hadamard_qk(
            q,
            k,
            full_rope_pos_ids,
            self.cos_sin_cache,
            self.rope_head_dim,
            self.is_neox_style,
        )
        if fused is not None:
            return fused

        # Fallback: unfused 4-op chain
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
            self._kv_cache_blocks(kv_cache),  # [num_blocks, block_size, cache_stride]
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
            self._kv_cache_blocks(kv_cache),  # [num_blocks, block_size, cache_stride]
            slot_mapping,  # [num_tokens] physical slot indices
            self.block_size,  # quantization block size (128)
            self.scale_fmt,  # "ue8m0" for power-of-2 scaling
        )

        return q_fp8, q_scale

    def quant_q_only(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize query only (no key caching). Used by dual-stream path."""
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

    def fused_rope_quant_q(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused RoPE + Hadamard + FP8 quantization for Q.

        Replaces separate apply_rope + rotate_activation + quant_q_only
        with a single Triton kernel.

        Args:
            q: Pre-RoPE query [num_tokens, index_n_heads, index_head_dim]
            positions: Position IDs [num_tokens]

        Returns:
            (q_fp8, q_scale) same as quant_q_only output.
        """
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
            fused_q_rope_quant,
        )

        # RoPE is applied to first (index_head_dim - rope_head_dim) dims
        actual_rot_dim = self.index_head_dim - self.rope_head_dim
        return fused_q_rope_quant(
            q,
            positions,
            self.cos_sin_cache,
            self.index_n_heads,
            self.index_head_dim,
            actual_rot_dim,
            is_neox_style=self.is_neox_style,
        )

    def fused_rope_quant_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused QK: K(RoPE+Hadamard→bf16) + Q(RoPE+Hadamard+FP8) in one kernel.

        Args:
            q: Pre-RoPE query [num_tokens, index_n_heads, index_head_dim]
            k: Pre-RoPE key [num_tokens, index_head_dim] (after k_norm)
            positions: Position IDs [num_tokens]

        Returns:
            (q_fp8, q_scale, k_out):
                q_fp8: [num_tokens, index_n_heads, index_head_dim] float8_e4m3fn
                q_scale: [num_tokens, index_n_heads, 1] float32
                k_out: [num_tokens, index_head_dim] bf16
        """
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
            fused_qk_rope_quant,
        )

        actual_rot_dim = self.index_head_dim - self.rope_head_dim
        return fused_qk_rope_quant(
            q,
            k,
            positions,
            self.cos_sin_cache,
            self.index_n_heads,
            self.index_head_dim,
            actual_rot_dim,
            is_neox_style=self.is_neox_style,
        )

    def quant_q_k_cp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache: KVCache,
        slot_mapping: torch.Tensor,
        kv_restore_unpad_indices: torch.Tensor,
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

        rtp_llm_ops.indexer_k_quant_and_cache(
            restored_key,
            self._kv_cache_blocks(kv_cache),
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
            attention_inputs: Attention inputs with decode_cu_seqlens_d, kv_cache_kernel_block_id_device

        Returns:
            TopK indices tensor
        """
        weights = weights.view(-1, self.index_n_heads)
        is_target_verify = bool(getattr(attention_inputs, "is_target_verify", False))

        num_heads_kv = 1
        head_dim_with_sf = self._head_dim_with_sf()
        kv_cache_fp8 = (
            self._kv_cache_blocks(kv_cache)
            .view(-1, self.blocksize, num_heads_kv, head_dim_with_sf)
            .view(dtype=torch.uint8)
        )

        block_table = _physical_block_table(attention_inputs)
        cu_seqlens_q = attention_inputs.decode_cu_seqlens_d
        lengths = fmha_params.expanded_seq_lens

        if is_target_verify:
            # Target verify has multiple query tokens per request. Treat it as
            # flattened decode: each verify token gets its own context length and
            # the same block-table row as its parent request. This matches the
            # paged DSA path used by SGLang/vLLM and avoids the ragged
            # fp8_mqa_logits path under CUDA graph replay.
            num_tokens = q_fp8.shape[0]
            batch_size = block_table.shape[0]
            assert batch_size > 0
            assert (
                num_tokens % batch_size == 0
            ), f"target verify tokens {num_tokens} not divisible by batch {batch_size}"
            tokens_per_batch = num_tokens // batch_size
            block_table = block_table.repeat_interleave(
                tokens_per_batch, dim=0, output_size=num_tokens
            )
            lengths = fmha_params.expanded_seq_lens
            kvlen_2d = lengths.reshape(-1, 1)
            cu_seqlens_q = torch.arange(
                0, num_tokens + 1, 1, dtype=torch.int32, device=q_fp8.device
            )
        else:
            # deep_gemm 2.5.0 expects context_lens as [batch_size, next_n].
            # fmha_params.kvlen_d is 1D [B]; unsqueeze(1) -> [B, 1] for next_n=1.
            kvlen_2d = fmha_params.kvlen_d.unsqueeze(1)

        max_seq_len = block_table.shape[1] * self.blocksize
        schedule_metadata = getattr(fmha_params, "schedule_metadata", None)
        has_schedule_metadata = False
        if isinstance(schedule_metadata, torch.Tensor):
            try:
                has_schedule_metadata = schedule_metadata.numel() > 0
            except RuntimeError:
                has_schedule_metadata = False
        if not has_schedule_metadata:
            schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                kvlen_2d,
                self.blocksize,
                deep_gemm.get_num_sms(),
            )

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8.unsqueeze(1),
            kv_cache_fp8.view(dtype=torch.uint8),
            weights,
            kvlen_2d,
            block_table,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )

        assert (
            fmha_params.expanded_seq_lens.device == logits.device
        ), "expanded_seq_lens must be on the same device as logits"

        topk_result = logits.new_empty(
            (logits.shape[0], self.index_topk), dtype=torch.int32
        )
        rtp_llm_ops.dsv4_persistent_topk(
            logits,
            lengths,
            topk_result,
            _get_topk_workspace(logits.device),
            self.index_topk,
            max_seq_len,
        )

        if _pd_debug_enabled():
            if _pd_debug_take(f"paged:{self.index_topk}", 32):
                logging.info(
                    "[PD_DEBUG][INDEXER_TOPK_PAGED] %s is_target_verify=%s "
                    "q_fp8=%s weights=%s kv_cache_fp8=%s block_table=%s "
                    "kernel_block_table=%s kvlen_2d=%s lengths=%s "
                    "cu_seqlens_q=%s logits=%s topk=%s",
                    _rank_tag(),
                    is_target_verify,
                    _tensor_summary(q_fp8),
                    _tensor_summary(weights),
                    _tensor_summary(kv_cache_fp8),
                    _tensor_summary(block_table),
                    _tensor_summary(
                        getattr(
                            attention_inputs, "kv_cache_kernel_block_id_device", None
                        )
                    ),
                    _tensor_summary(kvlen_2d),
                    _tensor_summary(lengths),
                    _tensor_summary(cu_seqlens_q),
                    _tensor_summary(logits),
                    _tensor_summary(topk_result),
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
        assert not bool(getattr(attention_inputs, "is_target_verify", False)), (
            "target verify must use paged DSA topk; ragged fp8_mqa_logits is "
            "not CUDA-graph safe"
        )

        # Gather quantized key from cache for prefill.
        # total_kv_tokens = sum(input_lengths + prefix_lengths) across all
        # requests — this is the number of K rows that
        # cp_gather_indexer_k_quant_cache will write (governed by cu_kv_seqlens).
        # Using q_fp8.shape[0] (= sum(input_lengths)) is wrong when
        # prefix_lengths > 0 (e.g. target-verify in speculative decoding).
        total_kv_tokens = fmha_params.prefill_total_kv_tokens
        k_fp8 = torch.empty(
            (total_kv_tokens, self.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device=q_fp8.device,
        )
        k_scale = torch.empty(
            (total_kv_tokens, self.index_head_dim // self.block_size * 4),
            dtype=torch.uint8,
            device=q_fp8.device,
        )

        block_table = _prefill_physical_block_table(attention_inputs)
        if _pd_debug_enabled():
            if _pd_debug_take(f"ragged_block_table:{self.index_topk}", 16):
                logging.info(
                    "[PD_DEBUG][INDEXER_RAGGED_BLOCK_TABLE] %s "
                    "physical=%s kernel=%s kv_cache_blocks=%s cu_kv=%s",
                    _rank_tag(),
                    _tensor_summary(block_table),
                    _tensor_summary(
                        getattr(
                            attention_inputs, "kv_cache_kernel_block_id_device", None
                        )
                    ),
                    kv_cache.kv_scale_base.shape[0],
                    _tensor_summary(attention_inputs.cu_kv_seqlens),
                )

        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            self._kv_cache_blocks(kv_cache),  # [num_blocks, block_size, cache_stride]
            k_fp8,  # output [num_tokens, index_head_dim]
            k_scale,  # output [num_tokens, scale_size]
            block_table,  # [batch_size, physical_blocks]
            attention_inputs.cu_kv_seqlens,
        )

        # Compute logits
        weights = weights.squeeze(-1)
        kv_fp8 = (k_fp8, k_scale.view(torch.float32).squeeze(-1))

        assert (
            fmha_params.ks is not None and fmha_params.ke is not None
        ), "ks/ke must be prepared in prefill"

        num_rows = q_fp8.shape[0]
        topk_result = torch.empty(
            (num_rows, self.index_topk), dtype=torch.int32, device=q_fp8.device
        )
        if num_rows == 0:
            return topk_result
        chunk_rows_env = _fp8_mqa_logits_chunk_rows()
        chunk_rows = num_rows if chunk_rows_env <= 0 else min(chunk_rows_env, num_rows)
        row_start = 0
        while row_start < num_rows:
            row_end = min(row_start + chunk_rows, num_rows)
            try:
                logits = deep_gemm.fp8_mqa_logits(
                    q_fp8[row_start:row_end],
                    kv_fp8,
                    weights[row_start:row_end],
                    fmha_params.ks[row_start:row_end],
                    fmha_params.ke[row_start:row_end],
                    clean_logits=False,
                )
            except torch.OutOfMemoryError:
                next_chunk_rows = _halve_chunk_rows(chunk_rows)
                if next_chunk_rows == chunk_rows:
                    raise
                logging.warning(
                    "[INDEXER_FP8_MQA_CHUNK] OOM with chunk_rows=%s, "
                    "retrying with chunk_rows=%s for num_rows=%s total_kv_tokens=%s",
                    chunk_rows,
                    next_chunk_rows,
                    num_rows,
                    total_kv_tokens,
                )
                torch.cuda.empty_cache()
                chunk_rows = next_chunk_rows
                continue
            topk_part = logits.new_empty(
                (logits.shape[0], self.index_topk), dtype=torch.int32
            )
            rtp_llm_ops.dsv4_top_k_per_row_prefill(
                logits,
                fmha_params.ks[row_start:row_end],
                fmha_params.ke[row_start:row_end],
                topk_part,
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                self.index_topk,
            )
            topk_result[row_start:row_end] = torch.where(
                topk_part >= 0,
                topk_part
                + fmha_params.topk_indices_offset[row_start:row_end].unsqueeze(1),
                topk_part,
            )
            del logits, topk_part
            row_start = row_end

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
        block_table = _prefill_physical_block_table(attention_inputs)
        if _pd_debug_enabled():
            if _pd_debug_take(f"ragged_cp_block_table:{self.index_topk}", 16):
                logging.info(
                    "[PD_DEBUG][INDEXER_RAGGED_CP_BLOCK_TABLE] %s "
                    "physical=%s kernel=%s kv_cache_blocks=%s cu_kv_global=%s "
                    "total_local_ids=%s precomputed_ks=%s precomputed_ke=%s "
                    "precomputed_lengths=%s precomputed_topk_off=%s q0=%s "
                    "weights_sq0=%s",
                    _rank_tag(),
                    _tensor_summary(block_table),
                    _tensor_summary(
                        getattr(
                            attention_inputs, "kv_cache_kernel_block_id_device", None
                        )
                    ),
                    kv_cache.kv_scale_base.shape[0],
                    _tensor_summary(cu_kv_seqlens_global),
                    _tensor_summary(total_local_ids),
                    _tensor_summary(precomputed_ks),
                    _tensor_summary(precomputed_ke),
                    _tensor_summary(precomputed_lengths),
                    _tensor_summary(precomputed_topk_off),
                    _tensor_summary(q0),
                    _tensor_summary(weights_sq0),
                )

        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            self._kv_cache_blocks(kv_cache),
            k_fp8,
            k_scale,
            block_table,
            cu_kv_seqlens_global,
        )
        kv_fp8_full = (k_fp8, k_scale.view(torch.float32).squeeze(-1))

        def run_part_logits_topk(
            q_part: torch.Tensor,
            weights_part: torch.Tensor,
            ks: torch.Tensor,
            ke: torch.Tensor,
            topk_off: torch.Tensor,
        ) -> torch.Tensor:
            nr = q_part.shape[0]
            topk_result = torch.empty(
                (nr, self.index_topk), dtype=torch.int32, device=q_part.device
            )
            if nr == 0:
                return topk_result
            chunk_rows_env = _fp8_mqa_logits_chunk_rows()
            chunk_rows = nr if chunk_rows_env <= 0 else min(chunk_rows_env, nr)
            row_start = 0
            while row_start < nr:
                row_end = min(row_start + chunk_rows, nr)
                try:
                    logits_p = deep_gemm.fp8_mqa_logits(
                        q_part[row_start:row_end],
                        kv_fp8_full,
                        weights_part[row_start:row_end],
                        ks[row_start:row_end],
                        ke[row_start:row_end],
                        clean_logits=False,
                    )
                except torch.OutOfMemoryError:
                    next_chunk_rows = _halve_chunk_rows(chunk_rows)
                    if next_chunk_rows == chunk_rows:
                        raise
                    logging.warning(
                        "[INDEXER_FP8_MQA_CHUNK] OOM with chunk_rows=%s, "
                        "retrying with chunk_rows=%s for nr=%s total_kv_tokens=%s",
                        chunk_rows,
                        next_chunk_rows,
                        nr,
                        total_kv_tokens,
                    )
                    torch.cuda.empty_cache()
                    chunk_rows = next_chunk_rows
                    continue
                topk_part = logits_p.new_empty(
                    (logits_p.shape[0], self.index_topk), dtype=torch.int32
                )
                rtp_llm_ops.dsv4_top_k_per_row_prefill(
                    logits_p,
                    ks[row_start:row_end],
                    ke[row_start:row_end],
                    topk_part,
                    logits_p.shape[0],
                    logits_p.stride(0),
                    logits_p.stride(1),
                    self.index_topk,
                )
                topk_result[row_start:row_end] = torch.where(
                    topk_part >= 0,
                    topk_part + topk_off[row_start:row_end].unsqueeze(1),
                    topk_part,
                )
                del logits_p, topk_part
                row_start = row_end
            return topk_result

        if total_local_ids.size(0) > 0:
            topk = run_part_logits_topk(
                q0,
                weights_sq0,
                precomputed_ks,
                precomputed_ke,
                precomputed_topk_off,
            )
        else:
            topk = None
        return topk
