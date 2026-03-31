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

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
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
        self._cu_local_kv_seqlens = None
        self._total_local_kv = None
        self._kv_allgather_restore_indices = None
        self._local_indexer_slot_mapping = None
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
      Supports two selectable implementations:
      - ag_q: all-gather q/topk, attend on local KV shard, then all-gather out/lse and merge.
      - ag_kv: read local sharded KV rows, all-gather KV rows, restore a global workspace,
        then reuse the no-prefix workspace attention path.
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

        self._scale_one = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        self.kv_restore_unpad_indices = None
        # Round-robin has no q0/q1 split, but keep attrs for indexer compatibility
        self.q0_idx = None
        self.q1_idx = None
        self.q0_idx_global = None
        self.q1_idx_global = None
        self.kv_cache_write_op = None
        self.write_cache_store_impl = None
        self._ws_fp8 = None

        self._local_req_ids = None
        self._global_req_ids = None
        self._local_kv_pack_dst_rows = None
        self._local_kv_pack_src_slots = None
        self._global_fp8_metadata = None
        self._fp8_kernel_metadata_q0 = None
        self._use_prefix_q_path = True

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
        self.total_kv_len = int(cu_kv_seqlens_cpu[-1])
        self.cu_kv_seqlens_global = cu_kv_seqlens_cpu.to(self.device)

        # Determine whether prefix cache is active
        self.has_prefix_cache = prefix_lengths.sum().item() > 0
        self._use_prefix_q_path = self._should_use_prefix_q_path()

        page_size = self.token_per_block

        # --- MLA workspace metadata ---
        # The workspace stores all-gathered KV in a contiguous paged tensor
        # for the workspace attention path used by both no-prefix and prefix-hit.
        ws_cu = self.cu_kv_seqlens_global.cpu().int()
        ws_lengths_t = ws_cu[1:] - ws_cu[:-1]
        ws_pages_per_req = (ws_lengths_t + page_size - 1) // page_size
        ws_aligned_sizes = ws_pages_per_req * page_size
        ws_cu_aligned = torch.zeros(ws_aligned_sizes.size(0) + 1, dtype=torch.int32)
        ws_cu_aligned[1:] = torch.cumsum(ws_aligned_sizes, dim=0)
        ws_total_aligned = int(ws_cu_aligned[-1].item())
        self._ws_total_pages = (ws_total_aligned + page_size - 1) // page_size

        total_ws_tokens = int(ws_lengths_t.sum().item())
        if total_ws_tokens > 0:
            offsets = ws_cu_aligned[:-1].long()
            lengths = ws_lengths_t.long()
            req_ids = torch.cat(
                [
                    torch.full((int(l),), i, dtype=torch.long)
                    for i, l in enumerate(lengths.tolist())
                ]
            )
            within_req = torch.cat(
                [torch.arange(int(l), dtype=torch.long) for l in lengths.tolist()]
            )
            self._ws_slot_mapping = (offsets[req_ids] + within_req).to(self.device)
        else:
            self._ws_slot_mapping = torch.empty(
                0, dtype=torch.int64, device=self.device
            )

        self._ws_block_table = common.build_contiguous_block_table(
            ws_cu_aligned.to(self.device),
            page_size,
            self.device,
        )

        # --- Sharded-cache metadata (always computed) ---
        # Used by the indexer and by the prefix-hit AG-KV path that packs local shard
        # rows, all-gathers them, and restores a global workspace KV cache.
        kv_lengths_t = kv_lengths.int()
        kv_lengths_list = kv_lengths_t.tolist()
        vbs = self.virtual_block_size
        n_vblocks = (kv_lengths_t + vbs - 1) // vbs
        local_kv_counts_t = n_vblocks * page_size
        self._local_kv_counts = local_kv_counts_t.tolist()
        cu_local_kv_cpu = torch.zeros(local_kv_counts_t.size(0) + 1, dtype=torch.int32)
        cu_local_kv_cpu[1:] = torch.cumsum(local_kv_counts_t, dim=0)
        self._cu_local_kv_seqlens = cu_local_kv_cpu.to(self.device)
        self._total_local_kv = int(cu_local_kv_cpu[-1].item())

        total_kv = sum(kv_lengths_list)
        if total_kv > 0:
            positions_flat = torch.cat(
                [torch.arange(kv_len, dtype=torch.long) for kv_len in kv_lengths_list]
            )
            req_ids = torch.cat(
                [
                    torch.full((kv_len,), s, dtype=torch.long)
                    for s, kv_len in enumerate(kv_lengths_list)
                ]
            )
            cu_offsets = cu_local_kv_cpu[req_ids].long()
            ranks = (positions_flat % vbs) % C
            local_idx = (positions_flat // vbs) * page_size + (
                positions_flat % vbs
            ) // C
            self._kv_allgather_restore_indices = (
                ranks * self._total_local_kv + cu_offsets + local_idx
            ).to(self.device)
        else:
            self._kv_allgather_restore_indices = torch.empty(
                0, dtype=torch.long, device=self.device
            )

        # --- Precompute local slot mapping for direct cache write ---
        # Both MLA and indexer use the same physical slot mapping (derived from
        # mla_params.slot_mapping), so we compute it once and share.
        local_slot_mapping = self._build_local_slot_mapping(
            mla_params.slot_mapping, local_tokens, C
        )
        self._local_mla_slot_mapping = local_slot_mapping
        self._local_indexer_slot_mapping = local_slot_mapping
        self._local_req_ids = self.mla_params.batch_indice_d[self.total_global_ids]
        if self.has_prefix_cache:
            self._global_req_ids = self.mla_params.batch_indice_d[
                : self.kv_restore_unpad_indices.size(0)
            ]
        else:
            self._global_req_ids = None

        # Pack actual owned rows from the paged sharded cache into a contiguous local
        # shard buffer so prefix-hit can reuse the same workspace attention path as
        # no-prefix after all_gather + restore.
        if total_kv > 0:
            owned_mask = ranks == self.prefill_cp_rank
            req_ids_d = req_ids.to(self.device)
            # Sharded RR cache stores one local page per virtual block, not per
            # global page. Map global token position to:
            #   page id   = global_pos // virtual_block_size
            #   page lane = (global_pos % virtual_block_size) // cp_size
            block_offsets_d = (positions_flat // vbs).to(self.device)
            token_offsets_d = ((positions_flat % vbs) // C).to(self.device)
            physical_slots = (
                self.block_table[req_ids_d, block_offsets_d].long() * page_size
                + token_offsets_d
            )
            self._local_kv_pack_dst_rows = (cu_offsets + local_idx)[owned_mask].to(
                self.device
            )
            self._local_kv_pack_src_slots = physical_slots[owned_mask].to(self.device)
        else:
            self._local_kv_pack_dst_rows = torch.empty(
                0, dtype=torch.long, device=self.device
            )
            self._local_kv_pack_src_slots = torch.empty(
                0, dtype=torch.long, device=self.device
            )

        if self.has_prefix_cache and self._use_prefix_q_path:
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

    def _build_local_slot_mapping(
        self,
        global_slot_mapping: torch.Tensor,
        local_tokens: int,
        cp_size: int,
    ) -> torch.Tensor:
        """Build a slot mapping for local tokens that maps only owned tokens to physical slots.

        Uses the already-computed total_local_ids (valid local indices) and
        total_global_ids (corresponding global unpadded indices) from plan().
        For owned tokens, the global slot_mapping already has the correct
        physical slot. For non-owned tokens, slot_mapping has -1.

        Args:
            global_slot_mapping: [total_unpadded_kv_tokens] physical slot indices
                (with -1 for non-owned in sharded mode).
            local_tokens: Number of local tokens on this rank.
            cp_size: Context parallel size.

        Returns:
            local_slot_mapping: [local_tokens] with physical slots for owned tokens, -1 otherwise.
        """
        if local_tokens == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        result = torch.full((local_tokens,), -1, dtype=torch.int64, device=self.device)
        if self.total_local_ids.numel() > 0:
            global_slot = global_slot_mapping.to(self.device)
            result[self.total_local_ids] = global_slot[self.total_global_ids]
        return result

    def _should_use_prefix_q_path(self) -> bool:
        """Return whether prefix-hit should use the q-path.

        Model-specific constants:
        - all_gather KV: 656 B/token
        - all_gather q/topk/out/lse: 104.25 KiB/token

        Therefore q-side communication is cheaper only when:
            global_q_len / total_kv_len <= 656 / (104.25 * 1024) ~= 0.00614
        """
        global_q_len = int(self.kv_restore_unpad_indices.size(0))
        total_kv_len = int(self.total_kv_len)
        if global_q_len <= 0 or total_kv_len <= 0:
            return True
        q_to_kv_ratio = global_q_len / total_kv_len
        return q_to_kv_ratio <= 0.00614

    def _alloc_workspace_kv_cache(self, kv_cache: KVCache) -> torch.Tensor:
        """Lazily allocate the temporary KV cache workspace, reused across layers within one forward pass."""
        kv_dim_bytes = kv_cache.kv_cache_base.size(-1)
        expected_shape = (self._ws_total_pages, self.token_per_block, kv_dim_bytes)
        if self._ws_fp8 is None or self._ws_fp8.shape != expected_shape:
            self._ws_fp8 = torch.empty(
                expected_shape,
                dtype=kv_cache.kv_cache_base.dtype,
                device=self.device,
            )
        return self._ws_fp8

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

        req_id = self._local_req_ids
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

    def _build_workspace_from_dense_kv(
        self,
        restored_ckv: torch.Tensor,
        restored_k_pe: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        workspace_cache = self._alloc_workspace_kv_cache(kv_cache)
        compute_ops.concat_and_cache_mla(
            restored_ckv,
            restored_k_pe,
            workspace_cache,
            self._ws_slot_mapping,
            self.kv_cache_write_op.kv_cache_type,
            self._scale_one,
        )
        return workspace_cache

    def _build_workspace_from_sharded_cache(self, kv_cache: KVCache) -> torch.Tensor:
        kv_dim_bytes = kv_cache.kv_cache_base.size(-1)
        local_rows = torch.zeros(
            self._total_local_kv,
            kv_dim_bytes,
            dtype=kv_cache.kv_cache_base.dtype,
            device=self.device,
        )
        if self._local_kv_pack_dst_rows.numel() > 0:
            kv_cache_rows = kv_cache.kv_cache_base.view(-1, kv_dim_bytes)
            local_rows[self._local_kv_pack_dst_rows] = kv_cache_rows[
                self._local_kv_pack_src_slots
            ]
        gathered_rows = all_gather(local_rows.contiguous(), group=Group.TP).reshape(
            -1, kv_dim_bytes
        )
        restored_rows = torch.index_select(
            gathered_rows, 0, self._kv_allgather_restore_indices
        )
        workspace_cache = self._alloc_workspace_kv_cache(kv_cache)
        workspace_rows = workspace_cache.view(-1, kv_dim_bytes)
        workspace_rows[self._ws_slot_mapping] = restored_rows
        return workspace_cache

    def _filter_topk_to_sharded_cache(
        self,
        topk_indices: torch.Tensor,
        req_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if topk_indices.dim() == 2:
            topk_indices_2d = topk_indices
        else:
            topk_indices_2d = topk_indices[:, 0, :]

        if req_id is None:
            req_id = self._local_req_ids

        return triton_filter_topk_for_sharded_cache(
            req_id=req_id,
            block_table=self.block_table,
            token_indices=topk_indices_2d,
            cp_rank=self.prefill_cp_rank,
            cp_size=self.prefill_cp_size,
            block_size=self.token_per_block,
            BLOCK_N=min(128, self.top_k),
        )

    def _merge_attention_across_ranks(
        self,
        local_out: torch.Tensor,
        local_lse: torch.Tensor,
    ) -> torch.Tensor:
        C = self.prefill_cp_size

        all_out = all_gather(local_out.contiguous(), group=Group.TP)
        all_lse = all_gather(local_lse.contiguous(), group=Group.TP)

        n_q = local_out.shape[0]
        head_dim = local_out.shape[-1]

        all_out = all_out.view(C, n_q, *local_out.shape[1:])
        all_lse = all_lse.view(C, n_q, *local_lse.shape[1:])

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
        # Step 1: Write local KV to sharded cache (owned tokens only, no all_gather needed)
        compute_ops.concat_and_cache_mla(
            compressed_kv,
            k_pe,
            kv_cache.kv_cache_base,
            self._local_mla_slot_mapping,
            self.kv_cache_write_op.kv_cache_type,
            self._scale_one,
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        no_valid_tokens = topk is None

        if self.has_prefix_cache:
            if no_valid_tokens:
                topk = torch.empty(0, self.top_k, dtype=torch.int32, device=q.device)
                self._forward_prefix_cache(q, topk, kv_cache, layer_id)
                return None
            return self._forward_prefix_cache(q, topk, kv_cache, layer_id)
        else:
            # Non-prefix: all ranks must participate in KV all_gather
            gathered_ckv = all_gather(
                compressed_kv.contiguous(), group=Group.TP
            ).reshape(-1, compressed_kv.size(-1))
            gathered_k_pe = all_gather(k_pe.contiguous(), group=Group.TP).reshape(
                -1, k_pe.size(-1)
            )

            if no_valid_tokens:
                return None

            restored_ckv = torch.index_select(
                gathered_ckv, 0, self.kv_restore_unpad_indices
            )
            restored_k_pe = torch.index_select(
                gathered_k_pe, 0, self.kv_restore_unpad_indices
            )
            return self._forward_workspace(
                q, restored_ckv, restored_k_pe, topk, kv_cache, layer_id
            )

    def _forward_workspace_from_cache(
        self,
        q: torch.Tensor,
        topk: torch.Tensor,
        workspace_cache: torch.Tensor,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """Run local-q attention against a prebuilt global workspace KV cache."""
        valid_local_tokens = self.total_local_ids.numel()
        if valid_local_tokens == 0:
            return torch.zeros(
                q.shape[0],
                self.num_heads,
                self.kv_lora_rank,
                dtype=q.dtype,
                device=q.device,
            )

        workspace_topk = self._convert_topk_indices_to_workspace(
            topk,
            self._ws_block_table,
        )

        kv_cache_flat = workspace_cache.view(torch.uint8)
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)

        q_part = torch.index_select(q, 0, self.total_local_ids).contiguous()
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

    def _forward_workspace(
        self,
        q: torch.Tensor,
        restored_ckv: torch.Tensor,
        restored_k_pe: torch.Tensor,
        topk: Optional[torch.Tensor],
        kv_cache=None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """No-prefix path: build global workspace from gathered dense KV, then attend."""
        workspace_cache = self._build_workspace_from_dense_kv(
            restored_ckv,
            restored_k_pe,
            kv_cache,
        )
        return self._forward_workspace_from_cache(
            q,
            topk,
            workspace_cache,
            layer_id,
        )

    def _forward_prefix_cache(
        self,
        q: torch.Tensor,
        topk: Optional[torch.Tensor],
        kv_cache=None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        if self._use_prefix_q_path:
            return self._forward_prefix_cache_ag_q(q, topk, kv_cache, layer_id)
        return self._forward_prefix_cache_ag_kv(q, topk, kv_cache, layer_id)

    def _forward_prefix_cache_ag_kv(
        self,
        q: torch.Tensor,
        topk: Optional[torch.Tensor],
        kv_cache=None,
        layer_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """Prefix-cache path: all-gather sharded KV into a workspace, then reuse workspace attention."""
        workspace_cache = self._build_workspace_from_sharded_cache(kv_cache)
        if topk.numel() == 0:
            return None
        return self._forward_workspace_from_cache(
            q,
            topk,
            workspace_cache,
            layer_id,
        )

    def _forward_prefix_cache_ag_q(
        self,
        q: torch.Tensor,
        topk: Optional[torch.Tensor],
        kv_cache=None,
        layer_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """Prefix-cache path: all-gather q/topk, attend on local shard, then merge states."""
        local_tokens = q.shape[0]
        valid_local_tokens = self.total_local_ids.numel()
        no_valid = topk.numel() == 0

        if not no_valid and topk.shape[0] not in (valid_local_tokens, local_tokens):
            raise ValueError(
                "[RoundRobin prefix] topk rows must be either valid local rows "
                "(aligned with total_local_ids) or fully padded local rows "
                "(aligned with q for collectives). "
                f"Got topk.shape[0]={topk.shape[0]}, "
                f"valid_local_tokens={valid_local_tokens}, local_tokens={local_tokens}."
            )

        if no_valid or topk.shape[0] < local_tokens:
            padded_topk = torch.full(
                (local_tokens, *topk.shape[1:]),
                -1,
                dtype=topk.dtype,
                device=self.device,
            )
            if not no_valid:
                padded_topk.index_copy_(0, self.total_local_ids, topk)
            topk_for_gather = padded_topk
        else:
            topk_for_gather = topk

        all_q = all_gather(q.contiguous(), group=Group.TP)
        all_q = all_q.reshape(-1, *q.shape[1:])
        all_topk = all_gather(topk_for_gather.contiguous(), group=Group.TP)
        all_topk = all_topk.reshape(-1, *topk_for_gather.shape[1:])

        all_q = torch.index_select(all_q, 0, self.kv_restore_unpad_indices)
        all_topk = torch.index_select(all_topk, 0, self.kv_restore_unpad_indices)

        all_batch_indice = self._global_req_ids
        sharded_indices = self._filter_topk_to_sharded_cache(
            all_topk, req_id=all_batch_indice
        )

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

        all_q = all_q.contiguous()
        q_batched = torch.as_strided(
            all_q,
            (1, *all_q.shape),
            (0, *all_q.stride()),
        )
        sharded_indices = sharded_indices.contiguous()
        indices_batched = torch.as_strided(
            sharded_indices,
            (1, *sharded_indices.shape),
            (0, *sharded_indices.stride()),
        )

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
        local_lse = local_lse.squeeze(0).float().transpose(0, 1)
        merged_out = self._merge_attention_across_ranks(local_out, local_lse)

        local_result = torch.index_select(merged_out, 0, self.total_global_ids)
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

        impl = self.fmha_impl

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
            has_prefix_cache=getattr(impl, "has_prefix_cache", False),
            # Sharded-cache metadata (always available for indexer topk)
            cu_local_kv_seqlens=getattr(impl, "_cu_local_kv_seqlens", None),
            total_local_kv=getattr(impl, "_total_local_kv", None),
            kv_allgather_restore_indices=getattr(
                impl, "_kv_allgather_restore_indices", None
            ),
            # Local slot mappings for direct cache write
            local_indexer_slot_mapping=getattr(
                impl, "_local_indexer_slot_mapping", None
            ),
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
