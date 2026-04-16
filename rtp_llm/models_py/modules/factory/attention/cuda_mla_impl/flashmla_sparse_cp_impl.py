"""
Unified Sparse MLA implementation for both prefill and decode stages.
Uses flash_mla_sparse_fwd kernel with triton-based index conversion.
"""

import copy
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

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, barrier
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_full_causal_kv_indices,
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

        self.attn_inputs = None
        self.cp_info = None

        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size
        self.device = torch.cuda.current_device()
        self.kv_restore_unpad_indices = None
        self.total_global_ids = None
        self.total_local_ids = None
        self.cu_kv_seqlens_global = None
        self.total_kv_len: int = 0
        self.kv_cache_write_op = None
        self.write_cache_store_impl = None
        self.precomputed_req_ids = None
        self.full_rope_pos_ids = None

    def plan(
        self,
        mla_params: rtp_llm_ops.FlashInferMlaAttnParams,
        block_table: torch.Tensor,
        attn_inputs: Optional[PyAttentionInputs] = None,
    ) -> None:
        self.block_table = block_table
        self.mla_params = mla_params

        self.attn_inputs = attn_inputs
        self.cp_info = attn_inputs.context_parallel_info
        assert (
            self.cp_info is not None
        ), "Context parallel info is required for SparseMlaFp8CPOp"

        cp_info = self.cp_info
        chunk_lengths = cp_info.prefill_cp_chunk_lengths

        # Zig-zag: restore_indices[global_pos] = source_flat_index → global_idx = inv_restore[cp_rank * local_tokens + local_idx]
        if isinstance(chunk_lengths, torch.Tensor):
            chunk_lengths_list = chunk_lengths.cpu().tolist()
        else:
            chunk_lengths_list = list(chunk_lengths)

        # These stay in Python (pure CPU list operations)
        q0_idx, q1_idx = generate_q_indices(chunk_lengths_list)

        local_tokens = sum(chunk_lengths_list)

        # Ensure CPU tensors for C++ processing
        padding_mask_cpu = cp_info.prefill_qkv_padding_mask
        if padding_mask_cpu.is_cuda:
            padding_mask_cpu = padding_mask_cpu.cpu()
        kv_restore_cpu = cp_info.prefill_qkv_restore_indice
        if kv_restore_cpu.is_cuda:
            kv_restore_cpu = kv_restore_cpu.cpu()

        # Single C++ call replaces ~26 GPU kernel launches
        mla_params.fill_cp_plan_params(
            padding_mask_cpu,
            kv_restore_cpu,
            q0_idx,
            q1_idx,
            self.prefill_cp_rank,
            local_tokens,
            cp_info.prefill_actual_input_lengths_cpu,
            self.attn_inputs.prefix_lengths,
        )

        self.kv_restore_unpad_indices = mla_params.cp_kv_restore_unpad_indices
        self.total_global_ids = mla_params.cp_total_global_ids
        self.total_local_ids = mla_params.cp_total_local_ids
        self.cu_kv_seqlens_global = mla_params.cp_cu_kv_seqlens_global
        self.total_kv_len = mla_params.cp_total_kv_len
        # get_mla_metadata stays in Python (external flash_mla library)
        n_q = self.total_global_ids.size(0)
        tile_sched_q0, num_splits_q0 = get_mla_metadata(  # type: ignore
            cache_seqlens=None,
            num_q_tokens_per_head_k=n_q * self.num_heads,
            topk=self.top_k,
            num_heads_q=self.num_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        self._fp8_kernel_metadata_q0 = SparseMlaFp8DecodeParams(
            tile_sched_q0, num_splits_q0
        )
        self.precomputed_req_ids = (
            mla_params.batch_indice_d[self.total_global_ids] if n_q > 0 else None
        )

        # Build full_rope_pos_ids so the attention-side RoPE path can run
        # in-place on the entire buffer, consistent with create_params().
        if n_q > 0:
            positions_d = mla_params.positions_d
            full_rope_pos_ids = torch.zeros(
                positions_d.size(0),
                dtype=positions_d.dtype,
                device=positions_d.device,
            )
            precomputed_positions = positions_d[self.total_global_ids]
            full_rope_pos_ids[self.total_local_ids] = precomputed_positions
            self.full_rope_pos_ids = full_rope_pos_ids
        else:
            self.full_rope_pos_ids = None

    def _convert_topk_indices_to_global(
        self, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """CP: topk 行与 total_local_ids 对齐，req_id 需用 total_global_ids 取 batch_indice_d，保证第 i 行对应 global token 的 request id。"""
        if topk_indices.dim() == 2:
            num_tokens, topk = topk_indices.shape
            h_kv = 1
            topk_indices_2d = topk_indices
        else:
            num_tokens, h_kv, topk = topk_indices.shape
            topk_indices_2d = topk_indices[:, 0, :]
        assert topk == self.top_k
        assert self.block_table is not None
        assert self.mla_params is not None
        assert self.precomputed_req_ids is not None, (
            "precomputed_req_ids must be set in plan() before _convert_topk_indices_to_global"
        )
        req_id = self.precomputed_req_ids
        from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
            triton_convert_req_index_to_global_index,
        )

        global_indices_2d = triton_convert_req_index_to_global_index(
            req_id=req_id,
            block_table=self.block_table,
            token_indices=topk_indices_2d,
            BLOCK_SIZE=self.token_per_block,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=min(128, topk),
            HAS_PREFILL_WORKSPACE=False,
        )
        global_indices_3d = global_indices_2d.unsqueeze(1).expand(
            num_tokens, h_kv, topk
        )
        return global_indices_3d

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        topk: Optional[torch.Tensor],
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
        # All-gather KV across CP ranks, restore to global order, then write full KV to cache
        gathered_ckv = all_gather(compressed_kv.contiguous(), group=Group.TP)
        gathered_ckv = gathered_ckv.reshape(-1, compressed_kv.size(-1))
        gathered_k_pe = all_gather(k_pe.contiguous(), group=Group.TP)
        gathered_k_pe = gathered_k_pe.reshape(-1, k_pe.size(-1))

        restored_ckv = gathered_ckv[self.kv_restore_unpad_indices]
        restored_k_pe = gathered_k_pe[self.kv_restore_unpad_indices]

        self.kv_cache_write_op.forward(
            restored_ckv, restored_k_pe, kv_cache, self.mla_params
        )

        # TODO: write cache for each cp_rank
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        if topk is None:
            return None

        assert (
            q is not None and q.size(0) > 0
        ), "q is required for sparse MLA CP (KV write runs after this); dim 0 (tokens) must be > 0"
        # Convert request-local topk0/topk1 to global indices for flash_mla_with_kvcache
        global_topk = self._convert_topk_indices_to_global(topk)
        # Two-part attention (q0, q1) using self.block_table, self._fp8_kernel_metadata
        kv_cache_flat = kv_cache.kv_cache_base.view(
            -1, 1, kv_cache.kv_cache_base.size(-1)
        ).view(torch.uint8)
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)

        if layer_id == 0:
            meta = self._fp8_kernel_metadata_q0
            if meta is not None:
                # meta.tile_scheduler_metadata is get_mla_metadata's scheduler object
                # (e.g. FlashMLASchedMeta); clear nested fields on that object, not on
                # SparseMlaFp8DecodeParams.
                sched = meta.tile_scheduler_metadata
                if sched is not None:
                    sched.tile_scheduler_metadata = None
                    sched.num_splits = None

        q0 = q[self.total_local_ids].contiguous()

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

        out0 = run_part(q0, global_topk, self._fp8_kernel_metadata_q0)

        total_q = q.size(0)
        out = torch.zeros(
            total_q, out0.size(1), out0.size(2), dtype=out0.dtype, device=out0.device
        )
        out[self.total_local_ids] = out0
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
        cp_info = attn_inputs.context_parallel_info
        # ContextParallelProcessor leaves per-chunk lengths on shared attn_inputs; sparse
        # fill_params / cache store need per-request actual lengths. Shallow-copy and set
        # input_lengths on the copy only (pattern 1.22: do not mutate shared attn_inputs).
        attn_inputs_for_init = copy.copy(attn_inputs)
        attn_inputs_for_init.input_lengths = cp_info.prefill_actual_input_lengths_cpu
        super().__init__(
            attn_configs=attn_configs,
            attn_inputs=attn_inputs_for_init,
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

    def prepare(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> None:
        """Parent prepare sees prefill_actual_input_lengths on a shallow copy; caller's attn_inputs unchanged."""
        cp_info = attn_inputs.context_parallel_info
        assert cp_info is not None
        attn_for_prepare = copy.copy(attn_inputs)
        attn_for_prepare.input_lengths = cp_info.prefill_actual_input_lengths_cpu
        super().prepare(attn_for_prepare, forbid_realloc=forbid_realloc)

    @staticmethod
    def fmha_type() -> FMHAType:
        """Return FMHA type."""
        return FMHAType.CP_SPARSE_FLASHMLA

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create FMHA parameters and pack CP indices into cp_params."""
        self.fmha_params = rtp_llm_ops.SparseMlaParams()
        self.rope_params = self.fmha_params
        self.prepare(attn_inputs)

        # plan() already precomputed full_rope_pos_ids and precomputed_req_ids
        # on self.fmha_impl. Only precompute the topk-related params here
        # (ks/ke/lengths/topk_off are only needed by the indexer path via cp_params).
        total_global_ids = self.fmha_impl.total_global_ids
        total_local_ids = self.fmha_impl.total_local_ids
        has_tokens = total_global_ids is not None and total_global_ids.size(0) > 0

        precomputed_ks = (
            self.fmha_params.ks[total_global_ids] if has_tokens else None
        )
        precomputed_ke = (
            self.fmha_params.ke[total_global_ids] if has_tokens else None
        )
        precomputed_lengths = (
            self.fmha_params.expanded_seq_lens[total_global_ids] if has_tokens else None
        )
        precomputed_topk_off = (
            self.fmha_params.topk_indices_offset[total_global_ids] if has_tokens else None
        )

        # Pack CP indices from fmha_impl for use by indexer and others
        self.cp_params = SimpleNamespace(
            kv_restore_unpad_indices=self.fmha_impl.kv_restore_unpad_indices,
            total_global_ids=total_global_ids,
            total_local_ids=total_local_ids,
            cu_kv_seqlens_global=self.fmha_impl.cu_kv_seqlens_global,
            total_kv_len=self.fmha_impl.total_kv_len,
            full_rope_pos_ids=self.fmha_impl.full_rope_pos_ids,
            precomputed_ks=precomputed_ks,
            precomputed_ke=precomputed_ke,
            precomputed_lengths=precomputed_lengths,
            precomputed_topk_off=precomputed_topk_off,
            precomputed_req_ids=self.fmha_impl.precomputed_req_ids,
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
        topk_indices: Optional[torch.Tensor],
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

        # Apply RoPE in-place to full q_pe/k_pe using full_rope_pos_ids.
        # Padding rows get pos=0 (garbage RoPE), but they are never consumed:
        #   - q goes through _apply_input_bmm then q[total_local_ids] in fmha_impl
        #   - k_pe goes through all_gather then kv_restore_unpad_indices
        q_pe = q[:, :, self.nope_head_dim :]

        if self.fmha_impl.full_rope_pos_ids is not None:
            self.rope_impl.forward(
                q_pe,
                k_pe,
                self.rope_params,
                precomputed_pos_ids=self.fmha_impl.full_rope_pos_ids,
            )

        # Apply input BMM to transform query
        q_transformed = self._apply_input_bmm(q, layer_id)

        assert self.fmha_params is not None
        attn_output = self.fmha_impl.forward(
            q_transformed,
            compressed_kv,
            k_pe,
            topk_indices,
            self.fmha_params.batch_indice_d,
            kv_cache,
            layer_id=layer_id,
        )

        # Apply output BMM to get final output
        if attn_output is None:
            return None
        output = self._apply_output_bmm(attn_output, layer_id)

        return output
