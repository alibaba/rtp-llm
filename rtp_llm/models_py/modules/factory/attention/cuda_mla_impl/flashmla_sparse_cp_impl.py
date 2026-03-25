"""
Unified Sparse MLA implementation for both prefill and decode stages.
Uses flash_mla_sparse_fwd kernel with triton-based index conversion.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

_FLASH_MLA_AVAILABLE = False
try:
    if torch.version.cuda:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) >= (12, 9):
            from flash_mla import (
                flash_mla_sparse_fwd,
                flash_mla_with_kvcache,
                get_mla_metadata,
            )

            _FLASH_MLA_AVAILABLE = True
except (ImportError, AttributeError, ValueError) as e:
    import logging

    logging.warning(f"flash_mla not available: {e}. Requires CUDA >= 12.9")

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, barrier
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_kv_indices,
    generate_q_indices,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.filter_topk_for_sharded_cache import (
    triton_filter_topk_for_sharded_cache,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.merge_states_kv_cascade import (
    merge_states_kv_cascade_torch_reference,
)
from rtp_llm.ops import (
    AttentionConfigs,
    CPProcessorType,
    FMHAConfig,
    FMHAType,
    ParallelismConfig,
    compute_ops,
)
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops

from .flashmla_sparse_impl import (
    SparseMlaFp8DecodeParams,
    SparseMlaFp8Op,
    SparseMlaImpl,
)

# FlashInfer cascade merge kernels (MergeState / MergeStates) are tuned for typical MHA
# head_dim (often <= 128/256). MLA uses kv_lora_rank (e.g. 512) as the last dim of v,
# which can exceed supported launch configs and fail with cudaErrorInvalidConfiguration.
_FLASHINFER_MERGE_MAX_HEAD_DIM = 256


class ZigZagSparseMlaFp8CPOp(SparseMlaFp8Op):
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
        self.kv_cache_sharded = False

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

        # Keep only indices where padding_mask is 1 (valid); drop padded positions (0)
        padding_mask_d = padding_mask.to(device=self.device)
        valid_mask_q0 = padding_mask_d[self.q0_idx_global] == 1
        valid_mask_q1 = padding_mask_d[self.q1_idx_global] == 1
        self.q0_idx_global = self.q0_idx_global[valid_mask_q0]
        self.q1_idx_global = self.q1_idx_global[valid_mask_q1]
        self.q0_idx = self.q0_idx[valid_mask_q0]
        self.q1_idx = self.q1_idx[valid_mask_q1]

        # Convert from padded to unpadded coordinate space.
        # inv_restore yields indices in the padded global space (0..padded_total-1),
        # but positions_d / ks / ke / batch_indice_d are sized by unpadded_total.
        pad_to_unpad = torch.cumsum(padding_mask_d, dim=0).long() - 1
        self.q0_idx_global = pad_to_unpad[self.q0_idx_global]
        self.q1_idx_global = pad_to_unpad[self.q1_idx_global]

        self.total_global_ids = torch.cat(
            [self.q0_idx_global, self.q1_idx_global], dim=0
        )
        self.total_local_ids = torch.cat([self.q0_idx, self.q1_idx], dim=0)

        # --- Bounds checks (moved from forward hot-path to avoid device-host sync) ---
        unpadded_total = int(padding_mask.sum().item())
        if self.total_local_ids.numel() > 0:
            max_lid = self.total_local_ids.max().item()
            if max_lid >= local_tokens:
                raise ValueError(
                    f"[plan] total_local_ids out of range: "
                    f"max(total_local_ids)={max_lid}, local_tokens={local_tokens}. "
                    "Check CP plan() local chunk vs actual input size."
                )
        if self.total_global_ids.numel() > 0:
            max_gid = self.total_global_ids.max().item()
            if max_gid >= unpadded_total:
                raise ValueError(
                    f"[plan] total_global_ids out of range: "
                    f"max(total_global_ids)={max_gid}, unpadded_total={unpadded_total}. "
                    "Check padded-to-unpadded coordinate conversion."
                )

        # attention_inputs.cu_kv_seqlens is based on local CP chunk lengths
        # (input_lengths is overwritten by ContextParallelProcessor), but the
        # gather kernel needs cumulative lengths covering the full (global) sequence.
        actual_input_lengths = cp_info.prefill_actual_input_lengths_cpu
        prefix_lengths = self.attn_inputs.prefix_lengths
        kv_lengths = actual_input_lengths.int() + prefix_lengths.int()
        cu_kv_seqlens_cpu = torch.zeros(kv_lengths.shape[0] + 1, dtype=torch.int32)
        cu_kv_seqlens_cpu[1:] = torch.cumsum(kv_lengths, dim=0)
        self.total_kv_len = int(cu_kv_seqlens_cpu[-1])
        self.cu_kv_seqlens_global = cu_kv_seqlens_cpu.to(self.device)

        # get_mla_metadata: num_q_tokens_per_head_k = num_q_tokens * num_heads_q // num_heads_k (for tile scheduling).
        # For q0 and q1 we need separate metadata since each part has different q token count (use filtered counts).
        n_q = self.total_global_ids.size(0)
        tile_sched_q0, num_splits_q0 = get_mla_metadata(
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
        # req_id[i] = request id for global token total_global_ids[i]
        req_id = self.mla_params.batch_indice_d[self.total_global_ids]
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
            restored_ckv,
            restored_k_pe,
            kv_cache,
            self.mla_params,
        )

        # TODO: write cache for each rank
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        if topk is None:
            return None

        # Convert request-local topk0/topk1 to global indices for flash_mla_with_kvcache
        global_topk = self._convert_topk_indices_to_global(topk)

        kv_cache_flat = kv_cache.kv_cache_base.view(
            -1, 1, kv_cache.kv_cache_base.size(-1)
        ).view(torch.uint8)
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)

        if layer_id == 0:
            meta = self._fp8_kernel_metadata_q0
            if meta is not None:
                metadata = meta.tile_scheduler_metadata
                if metadata is not None:
                    metadata.tile_scheduler_metadata = None
                    metadata.num_splits = None

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


# ---------------------------------------------------------------------------
# RoundRobin CP Op
# ---------------------------------------------------------------------------
class RoundRobinSparseMlaFp8CPOp(SparseMlaFp8Op):
    """Round-robin CP: token i → rank (i % cp_size), sharded cache + FP8 workspace for attention.

    Each rank writes only its owned tokens to the sharded kv_cache (via slot_mapping
    with -1 for non-owned positions). For attention there are two paths:

    Without prefix cache (has_prefix_cache=False):
      All-gather KV → build temporary FP8 workspace → local q attends to full workspace.
      This is the original path that avoids cross-rank communication during attention.

    With prefix cache (has_prefix_cache=True):
      1. All-gather q and topk so every rank has ALL q tokens.
      2. Filter the global topk indices to keep only positions this rank owns.
      3. Map those positions to sharded cache addresses.
      4. Run flash_mla_with_kvcache on the local sharded cache, getting (out, lse).
      5. All-gather (out, lse) across CP ranks and merge via flashinfer merge_state.
      This path naturally supports prefix cache since prefix blocks are already
      present in the sharded cache via cache match.
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
        assert self.cp_info is not None, "Context parallel info is required"

        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size
        self.kv_cache_sharded = parallelism_config.prefill_cp_config.kv_cache_sharded
        self.virtual_block_size = self.token_per_block * self.prefill_cp_size
        self.device = torch.cuda.current_device()

        self.kv_restore_unpad_indices = None
        # Round-robin has no q0/q1 split, but keep attrs for indexer compatibility
        self.q0_idx = None
        self.q1_idx = None
        self.q0_idx_global = None
        self.q1_idx_global = None
        self.kv_cache_write_op = None
        self.write_cache_store_impl = None
        self._ws_fp8 = None

    def plan(
        self, mla_params: rtp_llm_ops.FlashInferMlaAttnParams, block_table: torch.Tensor
    ) -> None:
        self.block_table = block_table
        self.mla_params = mla_params

        cp_info = self.cp_info
        padding_mask = cp_info.prefill_qkv_padding_mask
        kv_restore_indices = cp_info.prefill_qkv_restore_indice
        self.kv_restore_unpad_indices = kv_restore_indices[padding_mask == 1]

        chunk_lengths = cp_info.prefill_cp_chunk_lengths
        if hasattr(chunk_lengths, "cpu"):
            chunk_lengths_list = chunk_lengths.cpu().tolist()
        else:
            chunk_lengths_list = list(chunk_lengths)

        local_tokens = sum(chunk_lengths_list)
        C = self.prefill_cp_size

        _empty = torch.empty(0, device=self.device, dtype=torch.long)
        # Flat local indices 0..local_tokens-1 for this rank (same layout as all-gather
        # layout [rank0_chunk | rank1_chunk | ...]).
        total_local_ids_all = torch.arange(
            local_tokens, device=self.device, dtype=torch.long
        )

        # Map local flat positions → global unpadded token ids via restore inverse + padding,
        # matching ZigZagSparseMlaFp8CPOp.plan(). Manual j*cp_size+rank + clamp was wrong
        # when streams have CP padding (uneven seq_len % cp_size): padded slots must be
        # dropped and coordinates mapped to unpadded space for batch_indice_d / topk.
        restore = kv_restore_indices.to(device=self.device, dtype=torch.long)
        inv_restore = torch.empty(restore.size(0), device=self.device, dtype=torch.long)
        inv_restore[restore] = torch.arange(
            restore.size(0), device=self.device, dtype=torch.long
        )
        source_flat = self.prefill_cp_rank * local_tokens + total_local_ids_all
        total_global_ids_padded = inv_restore[source_flat]

        padding_mask_d = padding_mask.to(device=self.device)
        valid_mask = padding_mask_d[total_global_ids_padded] == 1
        total_global_ids_padded = total_global_ids_padded[valid_mask]
        self.total_local_ids = total_local_ids_all[valid_mask]

        pad_to_unpad = torch.cumsum(padding_mask_d, dim=0).long() - 1
        self.total_global_ids = pad_to_unpad[total_global_ids_padded]

        unpadded_total = int(padding_mask.sum().item())
        if self.total_local_ids.numel() > 0:
            max_lid = self.total_local_ids.max().item()
            if max_lid >= local_tokens:
                raise ValueError(
                    f"[RoundRobin plan] total_local_ids out of range: "
                    f"max(total_local_ids)={max_lid}, local_tokens={local_tokens}. "
                    "Check CP chunk lengths vs local tensor size."
                )
        if self.total_global_ids.numel() > 0:
            max_gid = self.total_global_ids.max().item()
            if max_gid >= unpadded_total:
                raise ValueError(
                    f"[RoundRobin plan] total_global_ids out of range: "
                    f"max(total_global_ids)={max_gid}, unpadded_total={unpadded_total}. "
                    "Check padded-to-unpadded coordinate conversion."
                )

        # No q0/q1 split for round-robin; set for indexer compatibility
        self.q0_idx = self.total_local_ids
        self.q1_idx = _empty
        self.q0_idx_global = self.total_global_ids
        self.q1_idx_global = _empty

        # Build global cu_kv_seqlens (includes prefix_lengths)
        actual_input_lengths = cp_info.prefill_actual_input_lengths_cpu
        prefix_lengths = self.attn_inputs.prefix_lengths
        kv_lengths = actual_input_lengths.int() + prefix_lengths.int()
        cu_kv_seqlens_cpu = torch.zeros(kv_lengths.shape[0] + 1, dtype=torch.int32)
        cu_kv_seqlens_cpu[1:] = torch.cumsum(kv_lengths, dim=0)
        self.cu_kv_seqlens_global = cu_kv_seqlens_cpu.to(self.device)

        # Determine whether prefix cache is active
        self.has_prefix_cache = prefix_lengths.sum().item() > 0

        # Workspace metadata for indexer (all-gathered new KV, no prefix).
        # The workspace stores new tokens only; it is used by indexer_k_quant_and_cache
        # to write the all-gathered new K into a contiguous paged tensor.
        new_total_kv = self.kv_restore_unpad_indices.size(0)
        self._ws_total_kv = new_total_kv
        page_size = self.token_per_block
        self._ws_total_pages = (new_total_kv + page_size - 1) // page_size
        self._ws_slot_mapping = torch.arange(
            new_total_kv, dtype=torch.int64, device=self.device
        )

        if not self.has_prefix_cache:
            # No prefix cache: workspace block_table built from full cu_kv_seqlens_global
            self._ws_block_table = common.build_contiguous_block_table(
                self.cu_kv_seqlens_global,
                page_size,
                self.device,
            )
            # Not needed for non-prefix path
            self._cu_local_kv_seqlens = None
            self._total_local_kv = None
            self._kv_allgather_restore_indices = None
            self._global_fp8_metadata = None
        else:
            # Workspace block_table for indexer: built from new-token-only cu_seqlens
            new_kv_lengths = actual_input_lengths.int()
            cu_new_kv_cpu = torch.zeros(new_kv_lengths.shape[0] + 1, dtype=torch.int32)
            cu_new_kv_cpu[1:] = torch.cumsum(new_kv_lengths, dim=0)
            cu_new_kv = cu_new_kv_cpu.to(self.device)
            self._ws_block_table = common.build_contiguous_block_table(
                cu_new_kv,
                page_size,
                self.device,
            )

            # Sharded cu_kv_seqlens: each rank's local token capacity per request.
            # All ranks have the same capacity per request (localBlockCount * block_size),
            # which ensures all_gather tensors have equal size across ranks.
            kv_lengths_list = kv_lengths.int().tolist()
            local_kv_counts = []
            vbs = self.virtual_block_size
            for kv_len in kv_lengths_list:
                n_vblocks = (kv_len + vbs - 1) // vbs
                local_capacity = n_vblocks * page_size
                local_kv_counts.append(local_capacity)
            self._local_kv_counts = local_kv_counts
            cu_local_kv_cpu = torch.zeros(len(local_kv_counts) + 1, dtype=torch.int32)
            cu_local_kv_cpu[1:] = torch.cumsum(
                torch.tensor(local_kv_counts, dtype=torch.int32), dim=0
            )
            self._cu_local_kv_seqlens = cu_local_kv_cpu.to(self.device)
            self._total_local_kv = int(cu_local_kv_cpu[-1].item())

            # Precompute restore indices: map all-gathered local K tokens to global order.
            restore_pieces: list[torch.Tensor] = []
            for s, kv_len in enumerate(kv_lengths_list):
                positions = torch.arange(kv_len, dtype=torch.long)
                ranks = (positions % vbs) % C
                local_idx = (positions // vbs) * page_size + (positions % vbs) // C
                cu_s = int(cu_local_kv_cpu[s].item())
                flat_idx = ranks * self._total_local_kv + cu_s + local_idx
                restore_pieces.append(flat_idx)
            if restore_pieces:
                self._kv_allgather_restore_indices = torch.cat(restore_pieces).to(
                    self.device
                )
            else:
                self._kv_allgather_restore_indices = torch.empty(
                    0, dtype=torch.long, device=self.device
                )

            # get_mla_metadata for the GLOBAL total_q (all ranks' q tokens after
            # all-gather + restore), because each rank runs attention for ALL q
            # tokens against its own KV shard.
            global_total_q = self.kv_restore_unpad_indices.size(0)
            if global_total_q > 0:
                tile_sched, num_splits = get_mla_metadata(
                    cache_seqlens=None,
                    num_q_tokens_per_head_k=global_total_q * self.num_heads,
                    topk=self.top_k,
                    num_heads_q=self.num_heads,
                    num_heads_k=1,
                    is_fp8_kvcache=True,
                )
                self._global_fp8_metadata = SparseMlaFp8DecodeParams(
                    tile_sched, num_splits
                )
            else:
                self._global_fp8_metadata = None

        # Local metadata for the indexer / non-prefix workspace path (valid q rows only)
        n_q = self.total_global_ids.size(0)
        if n_q > 0:
            tile_sched, num_splits = get_mla_metadata(
                cache_seqlens=None,
                num_q_tokens_per_head_k=n_q * self.num_heads,
                topk=self.top_k,
                num_heads_q=self.num_heads,
                num_heads_k=1,
                is_fp8_kvcache=True,
            )
            self._fp8_kernel_metadata_q0 = SparseMlaFp8DecodeParams(
                tile_sched, num_splits
            )
        else:
            self._fp8_kernel_metadata_q0 = None

    def _convert_topk_indices_to_workspace(
        self,
        topk_indices: torch.Tensor,
        workspace_block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Convert request-local topk indices to workspace-global indices (no-prefix path)."""
        if topk_indices.dim() == 2:
            num_tokens, topk = topk_indices.shape
            h_kv = 1
            topk_indices_2d = topk_indices
        else:
            num_tokens, h_kv, topk = topk_indices.shape
            topk_indices_2d = topk_indices[:, 0, :]
        assert topk == self.top_k
        assert self.mla_params is not None

        req_id = self.mla_params.batch_indice_d[self.total_global_ids]
        from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
            triton_convert_req_index_to_global_index,
        )

        global_indices_2d = triton_convert_req_index_to_global_index(
            req_id=req_id,
            block_table=workspace_block_table,
            token_indices=topk_indices_2d,
            BLOCK_SIZE=self.token_per_block,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=min(128, topk),
            HAS_PREFILL_WORKSPACE=False,
        )
        return global_indices_2d.unsqueeze(1).expand(num_tokens, h_kv, topk)

    def _filter_topk_to_sharded_cache(
        self,
        topk_indices: torch.Tensor,
        req_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Filter global topk indices to local rank and map to sharded cache addresses.

        Args:
            topk_indices: [num_tokens, topk] or [num_tokens, h_kv, topk],
                request-local token positions from the indexer.
            req_id: [num_tokens] int32, request id per q token. If None,
                uses batch_indice_d[total_global_ids] (for local-only path).

        Returns:
            sharded_indices: [num_tokens, topk] — indices_in_kvcache format for
                flash_mla_with_kvcache. Positions not owned by this rank are -1.
        """
        if topk_indices.dim() == 2:
            topk_indices_2d = topk_indices
        else:
            topk_indices_2d = topk_indices[:, 0, :]

        if req_id is None:
            assert self.mla_params is not None
            req_id = self.mla_params.batch_indice_d[self.total_global_ids]

        sharded_indices = triton_filter_topk_for_sharded_cache(
            req_id=req_id,
            block_table=self.block_table,
            token_indices=topk_indices_2d,
            cp_rank=self.prefill_cp_rank,
            cp_size=self.prefill_cp_size,
            block_size=self.token_per_block,
            BLOCK_N=min(128, self.top_k),
        )
        return sharded_indices

    def _merge_attention_across_ranks(
        self,
        local_out: torch.Tensor,
        local_lse: torch.Tensor,
    ) -> torch.Tensor:
        """All-gather (out, lse) from all CP ranks and merge via flashinfer merge_state.

        Each rank computes attention for ALL q tokens against its own KV shard.
        The all-gather collects these partial results, and merge_state combines
        them into the final output as if attention was computed over the full KV.

        Args:
            local_out: [total_q, num_heads, kv_lora_rank] — partial attention output
                       from this rank's KV shard, computed over ALL q tokens.
            local_lse: [total_q, num_heads] — partial log-sum-exp (float32).

        Returns:
            merged_out: [total_q, num_heads, kv_lora_rank] — merged attention output.
        """
        # all-gather across CP ranks: [cp_size * total_q, ...]
        all_out = all_gather(local_out.contiguous(), group=Group.TP)
        all_lse = all_gather(local_lse.contiguous(), group=Group.TP)

        n_q = local_out.shape[0]
        C = self.prefill_cp_size
        head_dim = local_out.shape[-1]

        # Reshape to [cp_size, total_q, ...]
        all_out = all_out.view(C, n_q, *local_out.shape[1:])
        all_lse = all_lse.view(C, n_q, *local_lse.shape[1:])

        if C == 1:
            return all_out[0]

        v = all_out.permute(1, 0, 2, 3).contiguous()
        s = all_lse.permute(1, 0, 2).contiguous()

        if head_dim > _FLASHINFER_MERGE_MAX_HEAD_DIM:
            try:
                from rtp_llm.models_py.triton_kernels.sparse_mla.merge_states_kv_cascade import (
                    triton_merge_states_kv_cascade,
                )

                return triton_merge_states_kv_cascade(v, s)
            except Exception:
                return merge_states_kv_cascade_torch_reference(v, s)

        from flashinfer.cascade import merge_states

        merged_out, _ = merge_states(v, s)
        return merged_out

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
        # Step 1: All-gather KV and restore to global order
        gathered_ckv = all_gather(compressed_kv.contiguous(), group=Group.TP)
        gathered_ckv = gathered_ckv.reshape(-1, compressed_kv.size(-1))
        gathered_k_pe = all_gather(k_pe.contiguous(), group=Group.TP)
        gathered_k_pe = gathered_k_pe.reshape(-1, k_pe.size(-1))

        restored_ckv = gathered_ckv[self.kv_restore_unpad_indices]
        restored_k_pe = gathered_k_pe[self.kv_restore_unpad_indices]

        # Step 2: Write sharded cache (slot_mapping has -1 for non-owned tokens)
        self.kv_cache_write_op.forward(
            restored_ckv,
            restored_k_pe,
            kv_cache,
            self.mla_params,
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        if topk is None:
            return None

        if self.has_prefix_cache:
            return self._forward_prefix_cache(q, topk, kv_cache, layer_id)
        else:
            return self._forward_workspace(
                q, restored_ckv, restored_k_pe, topk, kv_cache, layer_id
            )

    def _forward_workspace(
        self,
        q: torch.Tensor,
        restored_ckv: torch.Tensor,
        restored_k_pe: torch.Tensor,
        topk: Optional[torch.Tensor],
        kv_cache=None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """No-prefix path: build temporary FP8 workspace from all-gathered KV, attend locally."""
        # Step 3: Build temporary FP8 workspace from all-gathered KV
        kv_dim_bytes = kv_cache.kv_cache_base.size(-1)
        self._ws_fp8 = torch.empty(
            self._ws_total_pages,
            self.token_per_block,
            kv_dim_bytes,
            dtype=kv_cache.kv_cache_base.dtype,
            device=self.device,
        )
        scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        compute_ops.concat_and_cache_mla(
            restored_ckv,
            restored_k_pe,
            self._ws_fp8,
            self._ws_slot_mapping,
            self.kv_cache_write_op.kv_cache_type,
            scale,
        )

        # Step 4: Attention — valid local q rows attend to full global KV workspace
        # (same row alignment as indexer topk: one row per total_local_ids entry).
        workspace_topk = self._convert_topk_indices_to_workspace(
            topk,
            self._ws_block_table,
        )

        kv_cache_flat = self._ws_fp8.view(torch.uint8)
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)

        q_part = q[self.total_local_ids].contiguous()
        q_batched = q_part.unsqueeze(0)
        if workspace_topk.dim() == 3 and workspace_topk.shape[1] == 1:
            workspace_topk = workspace_topk.squeeze(1)
        indices_batched = workspace_topk.unsqueeze(0)

        if layer_id == 0 and self._fp8_kernel_metadata_q0 is not None:
            metadata = self._fp8_kernel_metadata_q0.tile_scheduler_metadata
            if metadata is not None:
                metadata.tile_scheduler_metadata = None
                metadata.num_splits = None

        attn_part, _ = flash_mla_with_kvcache(
            q=q_batched,
            k_cache=kv_cache_flat,
            block_table=self._ws_block_table,
            head_dim_v=self.kv_lora_rank,
            cache_seqlens=None,
            tile_scheduler_metadata=self._fp8_kernel_metadata_q0.tile_scheduler_metadata,
            num_splits=self._fp8_kernel_metadata_q0.num_splits,
            is_fp8_kvcache=True,
            indices=indices_batched,
            softmax_scale=self.scale,
        )
        attn_part = attn_part.squeeze(0)
        total_q = q.size(0)
        out = torch.zeros(
            total_q,
            attn_part.size(1),
            attn_part.size(2),
            dtype=attn_part.dtype,
            device=attn_part.device,
        )
        out[self.total_local_ids] = attn_part
        return out

    def _forward_prefix_cache(
        self,
        q: torch.Tensor,
        topk: Optional[torch.Tensor],
        kv_cache=None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """Prefix-cache path: all-gather q/topk, local attention on sharded cache, merge states."""
        local_tokens = q.shape[0]

        # topk has len(total_local_ids) rows which can differ across ranks when
        # seq_len % cp_size != 0 (padding tokens are filtered out by total_local_ids).
        # Pad back to local_tokens so all ranks contribute equal-sized tensors to all_gather.
        if topk.shape[0] < local_tokens:
            padded_topk = torch.full(
                (local_tokens, *topk.shape[1:]),
                fill_value=-1,
                dtype=topk.dtype,
                device=topk.device,
            )
            padded_topk[self.total_local_ids] = topk
            topk_for_gather = padded_topk
        else:
            topk_for_gather = topk

        # Step 3: All-gather q and topk so every rank has ALL q tokens.
        all_q = all_gather(q.contiguous(), group=Group.TP)
        all_q = all_q.reshape(-1, *q.shape[1:])
        all_topk = all_gather(topk_for_gather.contiguous(), group=Group.TP)
        all_topk = all_topk.reshape(-1, *topk_for_gather.shape[1:])

        # Restore to global unpadded token order (undo CP interleaving/padding).
        all_q = all_q[self.kv_restore_unpad_indices]
        all_topk = all_topk[self.kv_restore_unpad_indices]

        total_q = all_q.shape[0]

        # Step 4: Filter topk to this rank's KV shard.
        all_batch_indice = self.mla_params.batch_indice_d[:total_q]
        sharded_indices = self._filter_topk_to_sharded_cache(
            all_topk, req_id=all_batch_indice
        )

        # Step 5: Local attention — ALL q tokens attend to this rank's KV shard.
        kv_cache_flat = kv_cache.kv_cache_base.view(
            -1, 1, kv_cache.kv_cache_base.size(-1)
        ).view(torch.uint8)
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)

        meta = self._global_fp8_metadata
        if layer_id == 0 and meta is not None:
            ts = meta.tile_scheduler_metadata
            if ts is not None:
                ts.tile_scheduler_metadata = None
                ts.num_splits = None

        q_batched = all_q.unsqueeze(0)
        indices_batched = sharded_indices.unsqueeze(0)

        local_out, local_lse = flash_mla_with_kvcache(
            q=q_batched,
            k_cache=kv_cache_flat,
            block_table=self.block_table,
            head_dim_v=self.kv_lora_rank,
            cache_seqlens=None,
            tile_scheduler_metadata=meta.tile_scheduler_metadata if meta else None,
            num_splits=meta.num_splits if meta else None,
            is_fp8_kvcache=True,
            indices=indices_batched,
            softmax_scale=self.scale,
        )
        local_out = local_out.squeeze(0)
        local_lse = (
            local_lse.squeeze(0).float().transpose(0, 1)
        )  # [num_heads, total_q] -> [total_q, num_heads]
        # Step 6: Merge attention states across all CP ranks.
        merged_out = self._merge_attention_across_ranks(local_out, local_lse)

        # Step 7: Extract this rank's local q tokens from the merged global output,
        # and scatter into a local_tokens-sized buffer (consistent with _forward_workspace).
        local_result = merged_out[self.total_global_ids]
        out = torch.zeros(
            local_tokens,
            *local_result.shape[1:],
            dtype=local_result.dtype,
            device=local_result.device,
        )
        out[self.total_local_ids] = local_result
        return out


class SparseMlaCpImpl(SparseMlaImpl):
    """Sparse MLA CP implementation. Selects ZigZag or RoundRobin Op based on processor_type."""

    _OP_MAP = {
        CPProcessorType.ZIG_ZAG: ZigZagSparseMlaFp8CPOp,
        CPProcessorType.ROUND_ROBIN: RoundRobinSparseMlaFp8CPOp,
    }

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
        # Restore input_lengths to global (pre-CP-chunking) lengths.
        # handleInputs() overwrites input_lengths to local chunk sizes, but we
        # need global lengths for:
        #   - fill_params: computes positions 0..seq_len-1 and slot_mapping
        #     (kv_cache_sharded mode sets slot=-1 for non-owned tokens)
        #   - WriteCacheStoreOp (PD separation): C++ writeCacheStore handles
        #     sharding via cp_slot_mapper->localBlockCount(global_len)
        # Pad tokens (from round-robin when seq_len % cp_size != 0) are NOT
        # included here — prefill_actual_input_lengths_cpu is the original
        # unpadded length, and kv_restore_unpad_indices excludes pad tokens
        # from the all-gathered KV before cache write.
        attn_inputs.input_lengths = self.cp_info.prefill_actual_input_lengths_cpu
        self._cp_parallelism_config = parallelism_config

        processor_type = parallelism_config.prefill_cp_config.processor_type
        op_cls = self._OP_MAP.get(processor_type)
        assert op_cls is not None, (
            f"Unsupported CP processor_type: {processor_type}. "
            f"Must be one of {list(self._OP_MAP.keys())}"
        )

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
            fmha_impl=op_cls,
        )
        self.fmha_impl.kv_cache_write_op = self.kv_cache_write_op
        self.fmha_impl.write_cache_store_impl = self.write_cache_store_impl

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.CP_SPARSE_FLASHMLA

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create FMHA parameters and pack CP indices into cp_params."""
        self.fmha_params = rtp_llm_ops.SparseMlaParams()
        self.rope_params = self.fmha_params
        self.prepare(attn_inputs)

    def prepare(self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False):
        """Override prepare to pass CP rank/size/kv_cache_sharded to fill_params."""
        assert (
            self.fmha_params is not None
        ), "fmha_params should be initialized in __init__"

        pc = self._cp_parallelism_config
        self.fmha_params.fill_params(
            attn_inputs,
            self.seq_size_per_block,
            forbid_realloc,
            cp_rank=pc.tp_rank,
            cp_size=pc.tp_size,
            kv_cache_sharded=pc.prefill_cp_config.kv_cache_sharded,
        )
        self.fmha_impl.plan(self.fmha_params, attn_inputs.kv_cache_block_id_device)

        # Build shared workspace block_table & slot_mapping (per-request, not per-layer).
        # These are computed in plan() and depend on has_prefix_cache:
        # - No prefix: ws_block_table from cu_kv_seqlens_global (full KV = new tokens only)
        # - With prefix: ws_block_table from new-token-only cu_seqlens (for indexer workspace)
        impl = self.fmha_impl
        ws_block_table = None
        ws_slot_mapping = None
        cu_local_kv_seqlens = None
        total_local_kv = None
        kv_allgather_restore_indices = None
        if getattr(impl, "kv_cache_sharded", False):
            ws_slot_mapping = impl._ws_slot_mapping
            ws_block_table = impl._ws_block_table
            cu_local_kv_seqlens = impl._cu_local_kv_seqlens
            total_local_kv = impl._total_local_kv
            kv_allgather_restore_indices = impl._kv_allgather_restore_indices

        self.cp_params = SimpleNamespace(
            kv_restore_unpad_indices=impl.kv_restore_unpad_indices,
            q0_idx=impl.q0_idx,
            q1_idx=impl.q1_idx,
            q0_idx_global=impl.q0_idx_global,
            q1_idx_global=impl.q1_idx_global,
            total_global_ids=impl.total_global_ids,
            total_local_ids=impl.total_local_ids,
            cu_kv_seqlens_global=impl.cu_kv_seqlens_global,
            total_kv_len=getattr(impl, "total_kv_len", 0),
            kv_cache_sharded=impl.kv_cache_sharded,
            ws_block_table=ws_block_table,
            ws_slot_mapping=ws_slot_mapping,
            ws_total_pages=ws_total_pages,
            cu_local_kv_seqlens=cu_local_kv_seqlens,
            total_local_kv=total_local_kv,
            kv_allgather_restore_indices=kv_allgather_restore_indices,
            has_prefix_cache=getattr(impl, "has_prefix_cache", False),
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
        assert self.rope_impl is not None and self.rope_params is not None
        assert kv_cache is not None, "kv_cache is required for sparse MLA"
        assert self.fmha_impl is not None, "fmha_impl is not initialized"

        # Apply RoPE to q_pe and k_pe
        q_pe = q[:, :, self.nope_head_dim :]
        import flashinfer.rope as rope

        if self.fmha_impl.total_local_ids.size(0) > 0:
            q_pe_local = q_pe[self.fmha_impl.total_local_ids]
            k_pe_local = k_pe[self.fmha_impl.total_local_ids]
            k_rope = k_pe_local.unsqueeze(1)
            pos_ids_q0_global = self.rope_params.positions_d[
                self.fmha_impl.total_global_ids
            ]
            rope._apply_rope_pos_ids_cos_sin_cache(
                q=q_pe_local,
                k=k_rope,
                q_rope=q_pe_local,
                k_rope=k_rope,
                cos_sin_cache=self.rope_impl.cos_sin_cache,
                pos_ids=pos_ids_q0_global,
                interleave=not self.rope_impl.is_neox_style,
            )
            k_rope = k_rope.squeeze(1)
            k_pe[self.fmha_impl.total_local_ids] = k_rope
            q_pe[self.fmha_impl.total_local_ids] = q_pe_local

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

        if attn_output is None:
            return None
        return self._apply_output_bmm(attn_output, layer_id)
