"""
Unified Sparse MLA implementation for both prefill and decode stages.
Uses flash_mla_sparse_fwd kernel with triton-based index conversion.
"""

from typing import Any, Dict, List, Optional

import torch
from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
    triton_convert_req_index_to_global_index,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType, KvCacheDataType
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops
from rtp_llm.utils.model_weight import W

from .rope_emb_new import NewMlaRotaryEmbeddingOp, NewMlaRotaryEmbeddingParams


# for bf16 prefill && decode
class SparseMlaOp(object):
    """Unified Sparse MLA FMHA operator for both prefill and decode stages."""

    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        softmax_extra_scale: float,
        top_k: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.token_per_block = page_size
        self.softmax_extra_scale = softmax_extra_scale
        self.scale = (self.qk_head_dim**-0.5) * softmax_extra_scale
        self.top_k = top_k

        # Batch-related indices will be computed in plan
        self.block_table = None
        self.mla_params = None

    def plan(
        self, mla_params: rtp_llm_ops.FlashInferMlaAttnParams, block_table: torch.Tensor
    ):
        self.block_table = block_table
        self.mla_params = mla_params

    def _convert_topk_indices_to_global(
        self, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert topk_indices from request-local indices to global cache indices.

        Args:
            topk_indices: [num_tokens, h_kv, topk] - local indices within each request
                         typically h_kv=1 for MLA

        Returns:
            global_indices: [num_tokens, h_kv, topk] - global cache indices
        """
        # Handle both 2D [num_tokens, topk] and 3D [num_tokens, h_kv, topk] input
        if topk_indices.dim() == 2:
            num_tokens, topk = topk_indices.shape
            h_kv = 1
            topk_indices_2d = topk_indices
        else:
            num_tokens, h_kv, topk = topk_indices.shape
            # Flatten to 2D for triton kernel: [num_tokens, topk]
            # All heads share the same indices, so we can just take the first head
            topk_indices_2d = topk_indices[:, 0, :]

        assert topk == self.top_k, f"topk {topk} not equal to top_k {self.top_k}"
        assert self.block_table is not None
        assert self.mla_params is not None

        global_indices_2d = triton_convert_req_index_to_global_index(
            req_id=self.mla_params.batch_indice_d,
            block_table=self.block_table,
            token_indices=topk_indices_2d,
            BLOCK_SIZE=self.token_per_block,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=min(128, topk),  # tile width, must divide topk
            HAS_PREFILL_WORKSPACE=False,
        )

        # Expand back to 3D: [num_tokens, h_kv, topk]
        global_indices_3d = global_indices_2d.unsqueeze(1).expand(
            num_tokens, h_kv, topk
        )

        return global_indices_3d

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        kv_scale: Optional[torch.Tensor] = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for sparse MLA attention (both prefill and decode).

        Args:
            q: Query tensor of shape [total_q_len, num_heads, qk_head_dim]
            kv: Key-Value tensor of shape [total_cache_len, kv_lora_rank + rope_head_dim]
                For prefill: this is the concatenated [compressed_kv, k_pe]
                For decode: this is read from kv_cache with global indices
            topk_indices: Sparse indices tensor of shape [total_q_len, num_heads, topk]
                These are request-local indices that need conversion to global cache indices

        Returns:
            Attention output of shape [total_q_len, num_heads, kv_lora_rank]
        """
        flatten_topk_indices = self._convert_topk_indices_to_global(topk_indices)
        out_batch, _, _ = flash_mla_sparse_fwd(
            q,
            kv,  # Full KV cache with global indices
            flatten_topk_indices,
            self.scale,
            d_v=self.kv_lora_rank,
        )
        return out_batch


class SparseMlaFp8DecodeParams(object):
    def __init__(
        self,
        tile_scheduler_metadata: Any,
        num_splits: Any,
    ):
        self.tile_scheduler_metadata = tile_scheduler_metadata
        self.num_splits = num_splits


class SparseMlaFp8Op(SparseMlaOp):
    """FP8 quantized Sparse MLA FMHA operator for both prefill and decode stages.

    This implementation follows vllm's _forward_fp8_kv_mixed_batch approach,
    which treats all tokens as one batch and uses flash_mla_with_kvcache kernel.
    """

    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        softmax_extra_scale: float,
        top_k: int,
    ):
        super().__init__(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
        )
        self._fp8_kernel_metadata = None

    def plan(
        self, mla_params: rtp_llm_ops.FlashInferMlaAttnParams, block_table: torch.Tensor
    ):
        self.block_table = block_table
        self.mla_params = mla_params

        # Note that get_mla_metadata not doing anything but return empty structure
        tile_scheduler_metadata, num_splits = get_mla_metadata(  # type: ignore
            cache_seqlens=None,
            num_q_tokens_per_head_k=mla_params.batch_indice_h.shape[0] * self.num_heads,
            topk=self.top_k,
            num_heads_q=self.num_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        self._fp8_kernel_metadata = SparseMlaFp8DecodeParams(
            tile_scheduler_metadata, num_splits
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        kv_scale: Optional[torch.Tensor] = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for FP8 quantized sparse MLA attention (mixed batch approach).

        This follows vllm's _forward_fp8_kv_mixed_batch logic:
        1. Convert per-request indices to global indices
        2. Call flash_mla_with_kvcache with FP8 support

        Args:
            q: Query tensor of shape [total_q_len, num_heads, qk_head_dim]
            kv: Key-Value cache tensor (FP8) of shape [num_blocks, block_size, kv_dim]
                where kv_dim = kv_lora_rank + rope_head_dim
            topk_indices: Sparse indices tensor of shape [total_q_len, h_kv, topk]
                These are request-local indices that need conversion to global cache indices
            kv_scale: FP8 scale tensor (optional, for future use)

        Returns:
            Attention output of shape [total_q_len, num_heads, kv_lora_rank]
        """
        # Convert per-request indices to global slots (decode) or workspace offsets (prefill)
        # Output shape: [total_q_len, h_kv, topk]
        global_topk_indices = self._convert_topk_indices_to_global(topk_indices)

        # Squeeze h_kv dimension if it's 1 (MLA typically uses h_kv=1)
        # Shape: [total_q_len, topk]
        if global_topk_indices.shape[1] == 1:
            global_topk_indices = global_topk_indices.squeeze(1)

        # Add batch dimension for kernel: (T, topk) -> (1, T, topk)
        # This treats all tokens as one batch (mixed batch approach)
        global_topk_indices = global_topk_indices.unsqueeze(0)

        # Add batch dimension to q: (T, H, D) -> (1, T, H, D)
        q_batched = q.unsqueeze(0)

        # Prepare KV cache for kernel
        # Convert to uint8 view and add head dimension if needed
        # Expected shape: (num_blocks, block_size, num_heads_k, head_dim)
        if kv.dtype == torch.float8_e4m3fn or kv.dtype == torch.float8_e5m2:
            kv_cache = kv.view(torch.uint8)
        else:
            kv_cache = kv

        # Add head dimension if not present (MLA uses single KV head)
        # Shape: (num_blocks, block_size, kv_dim) -> (num_blocks, block_size, 1, kv_dim)
        if kv_cache.ndim == 3:
            kv_cache = kv_cache.unsqueeze(-2)

        if layer_id == 0 and self._fp8_kernel_metadata is not None:
            metadata = self._fp8_kernel_metadata.tile_scheduler_metadata
            if metadata is not None:
                metadata.tile_scheduler_metadata = None
                metadata.num_splits = None

        # Call FlashMLA sparse decode kernel
        attn_out, _ = flash_mla_with_kvcache(
            q=q_batched,
            k_cache=kv_cache,
            block_table=self.block_table,
            head_dim_v=self.kv_lora_rank,
            cache_seqlens=None,
            tile_scheduler_metadata=self._fp8_kernel_metadata.tile_scheduler_metadata,  # type: ignore
            num_splits=self._fp8_kernel_metadata.num_splits,  # type: ignore
            is_fp8_kvcache=True,
            indices=global_topk_indices,
            softmax_scale=self.scale,
        )

        # Output is (1, T, H, D_v), squeeze back to (T, H, D_v)
        return attn_out.squeeze(0)


class SparseMlaImpl(MlaImplBase):
    """
    Unified Sparse MLA implementation for both prefill and decode stages.
    Uses the same operator (SparseMlaOp) for both stages with triton-based index conversion.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
    ) -> None:
        super().__init__(
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            weights=weights,
            cos_sin_cache=cos_sin_cache,
            fmha_config=fmha_config,
            use_trt_fmha=use_trt_fmha,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
        )
        self.seq_size_per_block = attn_configs.tokens_per_block
        self.num_heads = attn_configs.head_num
        self.kv_lora_rank = attn_configs.kv_lora_rank
        self.rope_head_dim = attn_configs.rope_head_dim
        self.nope_head_dim = attn_configs.nope_head_dim
        self.is_prefill = attn_inputs.is_prefill

        self.fmha_params = None
        self.rope_params = None

        self.fmha_impl = None
        self.rope_kvcache_impl = None
        self.write_cache_store_impl = None
        # Check support
        self.support_ = (
            attn_configs.is_sparse
            and attn_configs.use_mla
            and attn_configs.kv_cache_dtype
            in (KvCacheDataType.BASE, KvCacheDataType.FP8)
        )
        if not self.support_:
            return

        # Initialize unified FMHA operator for both prefill and decode
        if attn_configs.kv_cache_dtype == KvCacheDataType.BASE:
            fmha_impl_cls = SparseMlaOp
        elif attn_configs.kv_cache_dtype == KvCacheDataType.FP8:
            fmha_impl_cls = SparseMlaFp8Op
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {attn_configs.kv_cache_dtype}"
            )

        self.fmha_impl = fmha_impl_cls(
            attn_configs.head_num,
            attn_configs.kv_lora_rank,
            attn_configs.rope_head_dim,
            attn_configs.nope_head_dim,
            attn_configs.tokens_per_block,
            attn_configs.softmax_extra_scale,
            attn_configs.indexer_topk,
        )

        self.rope_kvcache_impl = NewMlaRotaryEmbeddingOp(
            head_size=attn_configs.nope_head_dim,
            cos_sin_cache=cos_sin_cache,
            kv_lora_rank=attn_configs.kv_lora_rank,
            rope_head_dim=attn_configs.rope_head_dim,
            token_per_block=attn_configs.tokens_per_block,
            is_neox_style=False,
            kv_cache_dtype=attn_configs.kv_cache_dtype,
        )

        # Setup cache store if needed (only for prefill)
        if self.is_prefill and attn_inputs.cache_store_inputs:
            self.write_cache_store_impl = WriteCacheStoreOp(
                attn_inputs.input_lengths,
                attn_inputs.prefix_lengths,
                attn_inputs.kv_cache_block_id_host,
                attn_inputs.cache_store_inputs,
            )

        # Create parameters
        self.create_params(attn_inputs)

        # Prepare if not using CUDA graph
        if not is_cuda_graph:
            self.prepare(attn_inputs)

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create FMHA parameters."""
        if self.support_:
            self.fmha_params = rtp_llm_ops.SparseMlaParams()
            self.rope_params = None

    @staticmethod
    def fmha_type() -> FMHAType:
        """Return FMHA type."""
        return FMHAType.SPARSE_FLASHMLA

    @staticmethod
    def is_sparse() -> bool:
        return True

    def support(self):
        return self.support_

    def support_cuda_graph(self) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        """Prepare stage: update parameters and plan for processing."""
        assert (
            self.fmha_params is not None
        ), "fmha_params should be initialized in __init__"

        # Fill parameters - one call fills all parameters (base and derived)
        self.fmha_params.fill_params(attn_inputs, self.seq_size_per_block)
        # Plan for processing
        self.fmha_impl.plan(self.fmha_params, attn_inputs.kv_cache_block_id_device)
        self.rope_params = NewMlaRotaryEmbeddingParams(self.fmha_params)

    def _apply_input_bmm(self, q: torch.Tensor, layer_id: int) -> torch.Tensor:
        """
        Apply input batch matrix multiplication to transform q_nope.

        Args:
            q: Query tensor of shape [*, num_heads, qk_head_dim]
            layer_id: Current layer ID

        Returns:
            Transformed query tensor with same shape as input
        """
        # Split q into nope and pe parts
        q_nope, q_pe = q.view(
            -1, self.num_heads, self.nope_head_dim + self.rope_head_dim
        ).split([self.nope_head_dim, self.rope_head_dim], dim=-1)

        q_transformed = torch.empty(
            q_nope.shape[0],
            self.num_heads,
            self.kv_lora_rank + self.rope_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        k_weight = self.weights[layer_id][W.mla_kc]
        out_nope = q_transformed[..., : self.kv_lora_rank].transpose(0, 1)
        torch.bmm(q_nope.transpose(0, 1), k_weight, out=out_nope)  # type: ignore
        q_transformed[..., self.kv_lora_rank :] = q_pe
        return q_transformed

    def _apply_output_bmm(
        self, attn_output: torch.Tensor, layer_id: int
    ) -> torch.Tensor:
        """
        Apply output batch matrix multiplication to get final output.

        Args:
            attn_output: Attention output of shape [*, num_heads, kv_lora_rank]
            layer_id: Current layer ID

        Returns:
            Final output tensor with shape [*, num_heads, nope_head_dim]
        """
        v_weight = self.weights[layer_id][W.mla_vc]
        output = torch.bmm(attn_output.transpose(0, 1), v_weight)
        output = output.transpose(0, 1)
        return output

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for sparse MLA attention (prefill or decode).

        Args:
            q: Query tensor
                - Prefill: [total_q_len, num_heads, qk_head_dim]
                - Decode: [batch_size, num_heads, qk_head_dim]
            compressed_kv: Compressed KV tensor
                - Prefill: [total_kv_len, kv_lora_rank]
                - Decode: [batch_size, kv_lora_rank] (not used)
            k_pe: Key position encoding
                - Prefill: [total_kv_len, rope_head_dim]
                - Decode: [batch_size, rope_head_dim] (not used)
            kv_cache: KV cache object
            layer_id: Current layer ID
            topk_indices: Sparse indices (request-local)
                - Prefill: [total_q_len, num_heads, topk]
                - Decode: [batch_size, num_heads, topk]

        Returns:
            Attention output
                - Prefill: [total_q_len, num_heads, nope_head_dim]
                - Decode: [batch_size, num_heads, nope_head_dim]
        """
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        assert topk_indices is not None, "topk_indices is required for sparse MLA"
        assert kv_cache is not None, "kv_cache is required for sparse MLA"
        assert self.fmha_impl is not None, "fmha_impl is not initialized"

        # Apply RoPE to q_pe and k_pe
        q_pe = q[:, :, self.nope_head_dim :]
        self.rope_kvcache_impl.forward(
            q_pe, k_pe, compressed_kv, self.rope_params, kv_cache
        )

        # Apply input BMM to transform query
        q_transformed = self._apply_input_bmm(q, layer_id)

        # Get full KV cache: [num_blocks * page_size, kv_lora_rank + rope_head_dim]
        # Reshape from [num_blocks, page_size, kv_dim] to [num_blocks * page_size, h_kv, kv_dim]
        kv_cache_flat = kv_cache.kv_cache_base.view(
            -1, 1, kv_cache.kv_cache_base.size(-1)
        )

        if self.is_prefill:
            # Prefill stage: write to cache and use input kv
            # Write to cache store if needed
            if (
                self.attn_inputs.cache_store_inputs
                and self.write_cache_store_impl is not None
            ):
                self.write_cache_store_impl(kv_cache)

        # Call unified Op with input kv (returns [total_q_len, num_heads, kv_lora_rank])
        attn_output = self.fmha_impl.forward(
            q_transformed, kv_cache_flat, topk_indices, layer_id=layer_id
        )

        # Apply output BMM to get final output
        output = self._apply_output_bmm(attn_output, layer_id)

        return output
