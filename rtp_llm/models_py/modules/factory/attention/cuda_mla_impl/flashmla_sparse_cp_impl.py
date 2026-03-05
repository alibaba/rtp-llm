"""
Unified Sparse MLA implementation for both prefill and decode stages.
Uses flash_mla_sparse_fwd kernel with triton-based index conversion.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

# Check CUDA version for flash_mla compatibility
_FLASH_MLA_AVAILABLE = False
try:
    if torch.version.cuda:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) >= (12, 9):
            from flash_mla import flash_mla_with_kvcache, get_mla_metadata

            _FLASH_MLA_AVAILABLE = True
except (ImportError, AttributeError, ValueError) as e:
    import logging

    logging.warning(f"flash_mla not available: {e}. Requires CUDA >= 12.9")

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_kv_indices,
    generate_q_indices,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops

from .flashmla_sparse_impl import (
    SparseMlaFp8DecodeParams,
    SparseMlaFp8Op,
    SparseMlaImpl,
)


class SparseMlaFp8CPOp(SparseMlaFp8Op):
    """
    Context Parallel prefill for Sparse MLA (FP8).

    All-gather KV, restore to logical order, write via the same kv_cache_write_op as
    non-CP (line 508 in flashmla_sparse_impl), then run attention in two parts (q0, q1)
    using self.block_table, self._fp8_kernel_metadata, self._convert_topk_indices_to_global.
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
        attn_inputs: Optional[PyAttentionInputs] = None,
        parallelism_config: Optional[ParallelismConfig] = None,
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

        self.attn_inputs = attn_inputs
        self.cp_info = attn_inputs.context_parallel_info
        assert (
            self.cp_info is not None
        ), "Context parallel info is required for SparseMlaFp8CPOp"

        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size
        self.device = torch.cuda.current_device()
        self.kv_restore_unpad_indices = None

        self.q0_idx = None
        self.q1_idx = None
        self.q0_idx_global = None
        self.q1_idx_global = None
        self.kv0_idx = None
        self.kv1_idx = None
        self.kv_cache_write_op = None
        self.write_cache_store_impl = None

    def plan(
        self, mla_params: rtp_llm_ops.FlashInferMlaAttnParams, block_table: torch.Tensor
    ) -> None:
        self.block_table = block_table
        self.mla_params = mla_params

        cp_info = self.cp_info
        # TODO： ensure kv_restore_indices is diff from mha
        padding_mask = cp_info.prefill_qkv_padding_mask
        kv_restore_indices = cp_info.prefill_qkv_restore_indice
        self.kv_restore_unpad_indices = kv_restore_indices[padding_mask == 1]

        chunk_lengths = cp_info.prefill_cp_chunk_lengths
        q0_idx, q1_idx = generate_q_indices(chunk_lengths)
        kv0_idx, kv1_idx = generate_kv_indices(
            chunk_lengths,
            self.prefill_cp_rank,
            self.prefill_cp_size,
        )

        self.kv0_idx = kv_restore_indices[kv0_idx]
        self.kv1_idx = kv_restore_indices[kv1_idx]
        self.q0_idx = torch.tensor(q0_idx, device=self.device, dtype=torch.long)
        self.q1_idx = torch.tensor(q1_idx, device=self.device, dtype=torch.long)

        # Zig-zag: restore_indices[global_pos] = source_flat_index → global_idx = inv_restore[cp_rank * local_tokens + local_idx]
        if hasattr(chunk_lengths, "cpu"):
            chunk_lengths_list = chunk_lengths.cpu().tolist()
        else:
            chunk_lengths_list = list(chunk_lengths)
        local_tokens = sum(chunk_lengths_list)
        restore = kv_restore_indices.to(device=self.device, dtype=torch.long)
        inv_restore = torch.empty(restore.size(0), device=self.device, dtype=torch.long)
        inv_restore[restore] = torch.arange(
            restore.size(0), device=self.device, dtype=torch.long
        )
        source_flat_0 = self.prefill_cp_rank * local_tokens + self.q0_idx
        source_flat_1 = self.prefill_cp_rank * local_tokens + self.q1_idx
        self.q0_idx_global = inv_restore[source_flat_0]
        self.q1_idx_global = inv_restore[source_flat_1]

        self.total_global_ids = torch.cat(
            [self.q0_idx_global, self.q1_idx_global], dim=0
        )

        # get_mla_metadata: num_q_tokens_per_head_k = num_q_tokens * num_heads_q // num_heads_k (for tile scheduling).
        # For q0 and q1 we need separate metadata since each part has different q token count.
        n_q0 = len(q0_idx)
        n_q1 = len(q1_idx)
        tile_sched_q0, num_splits_q0 = get_mla_metadata(  # type: ignore
            cache_seqlens=None,
            num_q_tokens_per_head_k=n_q0 * self.num_heads,
            topk=self.top_k,
            num_heads_q=self.num_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        tile_sched_q1, num_splits_q1 = get_mla_metadata(  # type: ignore
            cache_seqlens=None,
            num_q_tokens_per_head_k=n_q1 * self.num_heads,
            topk=self.top_k,
            num_heads_q=self.num_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        self._fp8_kernel_metadata_q0 = SparseMlaFp8DecodeParams(
            tile_sched_q0, num_splits_q0
        )
        self._fp8_kernel_metadata_q1 = SparseMlaFp8DecodeParams(
            tile_sched_q1, num_splits_q1
        )

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        topk0: torch.Tensor,
        topk1: torch.Tensor,
        batch_indice_d: torch.Tensor,
        kv_cache=None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """
        CP prefill forward: all-gather KV, restore, write to kv_cache, then two-part attention.

        Args:
            q: [total_q_len, num_heads, qk_head_dim], already RoPE-applied and input-BMM applied (q_transformed).
            compressed_kv: [total_kv_len, kv_lora_rank], local.
            k_pe: [total_kv_len, rope_head_dim], local.
            topk0: [len(q0_idx), topk] or [len(q0_idx), num_heads, topk], request-local for first CP chunk.
            topk1: [len(q1_idx), topk] or [len(q1_idx), num_heads, topk], request-local for second CP chunk.
            batch_indice_d: [total_q_len], int32, request id per token.
            kv_cache: KV cache to write restored KV into (same paged layout as non-CP).
            layer_id: layer id.

        Returns:
            attn_output: [total_q_len, num_heads, kv_lora_rank], same as non-CP SparseMlaOp.
        """

        self.kv_cache_write_op.forward(
            compressed_kv, k_pe, kv_cache, self.mla_params, self.total_global_ids
        )
        from rtp_llm.models_py.distributed.collective_torch import Group, barrier

        barrier(group=Group.TP)
        # TODO: write cache for each cp_rank
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Convert request-local topk0/topk1 to global indices for flash_mla_with_kvcache
        topk0 = self._convert_topk_indices_to_global(topk0)
        topk1 = self._convert_topk_indices_to_global(topk1)

        # Two-part attention (q0, q1) using self.block_table, self._fp8_kernel_metadata
        kv_cache_flat = kv_cache.kv_cache_base.view(
            -1, 1, kv_cache.kv_cache_base.size(-1)
        ).view(torch.uint8)
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)

        if layer_id == 0:
            for meta in (self._fp8_kernel_metadata_q0, self._fp8_kernel_metadata_q1):
                if meta is not None:
                    metadata = meta.tile_scheduler_metadata
                    if metadata is not None:
                        metadata.tile_scheduler_metadata = None
                        metadata.num_splits = None

        q0 = torch.index_select(q, 0, self.q0_idx).contiguous()
        q1 = torch.index_select(q, 0, self.q1_idx).contiguous()

        def run_part(
            q_part: torch.Tensor,
            global_topk: torch.Tensor,
            fp8_kernel_metadata: SparseMlaFp8DecodeParams,
        ) -> torch.Tensor:
            q_batched = q_part.unsqueeze(0)
            if global_topk.dim() == 3 and global_topk.shape[1] == 1:
                global_topk = global_topk.squeeze(1)
            indices_batched = global_topk.unsqueeze(0)
            part_out, _ = flash_mla_with_kvcache(
                q=q_batched,
                k_cache=kv_cache_flat,
                block_table=self.block_table,
                head_dim_v=self.kv_lora_rank,
                cache_seqlens=None,
                tile_scheduler_metadata=fp8_kernel_metadata.tile_scheduler_metadata,  # type: ignore
                num_splits=fp8_kernel_metadata.num_splits,  # type: ignore
                is_fp8_kvcache=True,
                indices=indices_batched,
                softmax_scale=self.scale,
            )
            return part_out.squeeze(0)

        out0 = run_part(q0, topk0, self._fp8_kernel_metadata_q0)
        out1 = run_part(q1, topk1, self._fp8_kernel_metadata_q1)

        total_q = q.size(0)
        out = torch.empty(
            total_q, out0.size(1), out0.size(2), dtype=out0.dtype, device=out0.device
        )
        out[self.q0_idx] = out0
        out[self.q1_idx] = out1
        return out


class SparseMlaCpImpl(SparseMlaImpl):
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
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.cp_info = attn_inputs.context_parallel_info
        attn_inputs.input_lengths = self.cp_info.prefill_actual_input_lengths_cpu
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
            parallelism_config=parallelism_config,
            fmha_impl=SparseMlaFp8CPOp,
        )
        self.fmha_impl.kv_cache_write_op = self.kv_cache_write_op
        self.fmha_impl.write_cache_store_impl = self.write_cache_store_impl

    @staticmethod
    def fmha_type() -> FMHAType:
        """Return FMHA type."""
        return FMHAType.CP_SPARSE_FLASHMLA

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create FMHA parameters and pack CP indices into cp_params."""
        self.fmha_params = rtp_llm_ops.SparseMlaParams()
        self.rope_params = self.fmha_params
        self.prepare(attn_inputs)
        # Pack CP indices from fmha_impl for use by indexer and others
        self.cp_params = SimpleNamespace(
            kv_restore_unpad_indices=self.fmha_impl.kv_restore_unpad_indices,
            q0_idx=self.fmha_impl.q0_idx,
            q1_idx=self.fmha_impl.q1_idx,
            q0_idx_global=self.fmha_impl.q0_idx_global,
            q1_idx_global=self.fmha_impl.q1_idx_global,
            total_global_ids=self.fmha_impl.total_global_ids,
        )

    @classmethod
    def support_prefill_cp(cls) -> bool:
        return True

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Tuple[torch.Tensor, torch.Tensor],
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
            topk_indices: (topk0, topk1) from indexer CP path, request-local for the two chunks.

        Returns:
            Attention output
                - Prefill: [total_q_len, num_heads, nope_head_dim]
                - Decode: [batch_size, num_heads, nope_head_dim]
        """
        assert self.rope_impl is not None and self.rope_params is not None
        assert kv_cache is not None, "kv_cache is required for sparse MLA"
        assert self.fmha_impl is not None, "fmha_impl is not initialized"

        # Apply RoPE to q_pe and k_pe (use cp_params when available, else fmha_impl)
        q_pe = q[:, :, self.nope_head_dim :]
        cp = getattr(self, "cp_params", None)
        if cp is not None:
            q_total_pos_ids = cp.total_global_ids
        else:
            q_total_pos_ids = self.fmha_impl.total_global_ids
        self.rope_impl.forward(q_pe, k_pe, self.rope_params, q_total_pos_ids)

        # Apply input BMM to transform query
        q_transformed = self._apply_input_bmm(q, layer_id)

        assert self.fmha_params is not None
        topk0, topk1 = topk_indices[0], topk_indices[1]
        attn_output = self.fmha_impl.forward(
            q_transformed,
            compressed_kv,
            k_pe,
            topk0,
            topk1,
            self.fmha_params.batch_indice_d,
            kv_cache,
            layer_id=layer_id,
        )

        # Apply output BMM to get final output
        output = self._apply_output_bmm(attn_output, layer_id)

        return output
