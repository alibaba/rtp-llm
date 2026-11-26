"""
CP (context-parallel) variant of sparse MLA.
Mirrors flashmla_sparse_impl.py but with all-gather + restore + zig-zag q split.
"""

import copy
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, List, NamedTuple, Optional

import torch

try:
    cuda_ver = torch.version.cuda or ""
    _major, _minor = (int(x) for x in (cuda_ver.split(".") + ["0", "0"])[:2])
    if (_major, _minor) >= (12, 9):
        from rtp_llm.models_py.utils.flash_mla_legacy import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
except (ImportError, AttributeError, ValueError) as _e:
    logging.warning(f"flash_mla not available: {_e}. Requires CUDA >= 12.9")

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.distributed.pynccl_cp import (
    all_gather,
    defer_padded_scatter_enabled,
)
from rtp_llm.models_py.distributed.pynccl_cp import enabled as cp_opt_enabled
from rtp_llm.models_py.distributed.pynccl_cp import warmup as warmup_pynccl
from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_cp_assembler as asm
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_q_indices,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.triton_kv_scatter import (
    prepare_kv_scatter_indices,
    triton_kv_scatter,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops

from rtp_llm.utils.model_weight import W
from .cp_topk_offsets_triton import offset_cp_topk
from .flashmla_sparse_impl import (
    SparseMlaFp8DecodeParams,
    SparseMlaFp8Op,
    SparseMlaImpl,
    _allocate_prefill_fused_kv,
    _as_uint8,
    _GatherWorkspace,
    _topk_2d,
)
from .glm53_prefill_workspace import Glm53PrefillWorkspace


class _RankMajorKv(NamedTuple):
    """Gathered storage and the logical-token to physical-row mapping.

    Storage is allocated on compute; joining the ready event orders auxiliary
    writes before attention reads. Compute-stream allocator ordering protects
    queued consumers even after the Python view is released. Valid mapped
    TopK entries never reference padding rows.
    """

    values: torch.Tensor
    restore_indices: torch.Tensor


def _rank_major_slot_mapping(
    logical_slots: torch.Tensor,
    restore_indices: torch.Tensor,
    padded_rows: int,
) -> torch.Tensor:
    """Move the write plan, not the per-layer KV payload, into gather order.

    CP's restore indices are unique rank-major row IDs for valid logical
    tokens. Unused rows and tokens owned by another rank must remain -1.
    Build once per eager plan; the writer keeps its original FP8 quantization.
    """
    if (
        logical_slots.dim() != 1
        or restore_indices.dim() != 1
        or logical_slots.numel() != restore_indices.numel()
        or logical_slots.dtype != torch.int64
        or restore_indices.dtype != torch.int64
        or logical_slots.device != restore_indices.device
        or padded_rows < logical_slots.numel()
    ):
        raise ValueError("invalid CP rank-major KV write plan")
    slots = torch.full(
        (padded_rows,), -1, dtype=torch.int64, device=logical_slots.device
    )
    slots.index_copy_(0, restore_indices, logical_slots)
    return slots


def _total_local_ids_are_identity(
    padding_mask_cpu: torch.Tensor,
    kv_restore_cpu: torch.Tensor,
    q0_idx: List[int],
    q1_idx: List[int],
    cp_rank: int,
    local_tokens: int,
) -> bool:
    ordered_ids = [int(idx) for idx in q0_idx] + [int(idx) for idx in q1_idx]
    if len(ordered_ids) != local_tokens:
        return False
    if any(idx != pos for pos, idx in enumerate(ordered_ids)):
        return False
    if local_tokens == 0:
        return True

    padding_mask_cpu = padding_mask_cpu.reshape(-1).to(dtype=torch.int32)
    kv_restore_cpu = kv_restore_cpu.reshape(-1).to(dtype=torch.long)
    padded_total = int(padding_mask_cpu.numel())
    if padded_total == 0:
        return False

    cpu_device = padding_mask_cpu.device
    source_flat = torch.tensor(ordered_ids, dtype=torch.long, device=cpu_device)
    source_flat += int(cp_rank) * int(local_tokens)
    if bool(torch.any((source_flat < 0) | (source_flat >= padded_total)).item()):
        return False

    valid_restore = (kv_restore_cpu >= 0) & (kv_restore_cpu < padded_total)
    inv_restore = torch.full((padded_total,), -1, dtype=torch.long, device=cpu_device)
    restore_positions = torch.arange(padded_total, dtype=torch.long, device=cpu_device)
    inv_restore[kv_restore_cpu[valid_restore]] = restore_positions[valid_restore]

    global_padded = inv_restore[source_flat]
    if bool(torch.any((global_padded < 0) | (global_padded >= padded_total)).item()):
        return False
    return bool(torch.all(padding_mask_cpu[global_padded] == 1).item())


def _copy_or_replace_graph_tensor(
    current: Optional[torch.Tensor],
    new_tensor: Optional[torch.Tensor],
    name: str,
    use_cuda_graph: bool,
) -> Optional[torch.Tensor]:
    if not use_cuda_graph or current is None:
        return new_tensor
    if new_tensor is None:
        raise RuntimeError(f"{name} became None during CUDA graph replay")
    if (
        tuple(current.shape) != tuple(new_tensor.shape)
        or current.dtype != new_tensor.dtype
        or current.device != new_tensor.device
    ):
        raise RuntimeError(
            f"{name} shape/dtype/device changed during CUDA graph replay: "
            f"captured={(tuple(current.shape), current.dtype, current.device)} "
            f"current={(tuple(new_tensor.shape), new_tensor.dtype, new_tensor.device)}"
        )
    current.copy_(new_tensor)
    return current


def _cp_sharded_slot_mapping(
    positions: torch.Tensor,
    block_table_local: torch.Tensor,
    batch_indice: torch.Tensor,
    tokens_per_block: int,
    cp_size: int,
    cp_rank: int,
    owner_tokens_per_block: Optional[int] = None,
) -> torch.Tensor:
    """Build local slot mapping for RR-sharded paged KV.

    Ownership is decided at the *owner* (physical KV-block) granularity, not
    the kernel block granularity. Each owner block contains
    ``bpk = owner_tokens_per_block / tokens_per_block`` consecutive kernel
    sub-blocks that all live on the same rank.

    Logical owner block ``g_owner`` is owned by rank ``g_owner % cp_size`` and
    appears at local compact owner column ``g_owner // cp_size``; that
    compact owner column maps to ``bpk`` consecutive kernel columns
    ``[col_owner*bpk, col_owner*bpk + bpk - 1]`` in ``block_table_local``
    (the kernel block table). Non-owned tokens get ``-1`` so writer kernels
    skip them.

    When ``owner_tokens_per_block`` is ``None`` or equals ``tokens_per_block``
    (i.e. bpk == 1, kernel and physical block sizes coincide), the formula
    degenerates to the legacy ``(g % cp_size, g // cp_size)`` ownership at
    kernel granularity.
    """
    if positions.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=positions.device)
    if int(block_table_local.shape[1]) == 0:
        return torch.full(
            (positions.numel(),), -1, dtype=torch.int64, device=positions.device
        )

    kernel_tpb = int(tokens_per_block)
    owner_tpb = (
        int(owner_tokens_per_block)
        if owner_tokens_per_block is not None
        else kernel_tpb
    )
    if owner_tpb <= 0 or kernel_tpb <= 0 or owner_tpb % kernel_tpb != 0:
        raise ValueError(
            f"owner_tokens_per_block ({owner_tpb}) must be a positive multiple "
            f"of tokens_per_block ({kernel_tpb})"
        )
    bpk = owner_tpb // kernel_tpb

    pos = positions.to(torch.int64)
    bid = batch_indice.to(torch.int64)
    bt = block_table_local.to(torch.int64)
    # Owner block decides which rank holds this token (matches C++ CPSlotMapper
    # which RR-shards at owner_tpb granularity).
    owner_block = pos // owner_tpb
    owned = (owner_block % int(cp_size)) == int(cp_rank)
    local_owner_block = owner_block // int(cp_size)
    # Kernel sub-block index inside the owner block, in [0, bpk).
    kernel_in_owner = (pos % owner_tpb) // kernel_tpb
    # Compact kernel column on this rank: bpk consecutive kernel ids per owner.
    local_kernel_col = local_owner_block * int(bpk) + kernel_in_owner
    in_capacity = local_kernel_col < int(bt.shape[1])
    valid = owned & in_capacity
    safe_local_col = torch.where(
        valid, local_kernel_col, torch.zeros_like(local_kernel_col)
    )
    block_id = bt[bid, safe_local_col]
    valid = valid & (block_id > 0)
    slot = block_id * kernel_tpb + (pos % kernel_tpb)
    return torch.where(valid, slot, torch.full_like(slot, -1)).contiguous()


def _safe_expand_cp_sharded_block_table(
    attn_inputs: PyAttentionInputs,
    tokens_per_block: int,
    cp_size: int,
    owner_tokens_per_block: Optional[int] = None,
) -> PyAttentionInputs:
    """Return an attention input copy whose host block table has full width.

    SparseMlaParams.fill_params builds generic page/slot metadata before the
    CP op can override sharded writes. That generic code indexes by global
    logical (kernel) block id, so a compact page-RR block table can go out of
    bounds. The expanded table is only a bounds-safe placeholder; real
    sharded reads and writes use the original compact block table.

    Page-RR ownership is decided at the *owner* (physical KV-block) granularity:
    ``bpk = owner_tokens_per_block / tokens_per_block`` consecutive kernel
    sub-blocks share one owner block, and all live on the same rank. So the
    expansion needs to map global kernel column ``k_global`` to the compact
    kernel column on its owner's local table, namely
    ``(k_global // bpk // cp_size) * bpk + (k_global % bpk)``.

    When ``owner_tokens_per_block`` is ``None`` or equals ``tokens_per_block``
    (bpk == 1), this degenerates to the legacy ``k_global // cp_size`` mapping.
    """
    compact_host = getattr(attn_inputs, "kv_cache_kernel_block_id", None)
    if not isinstance(compact_host, torch.Tensor) or compact_host.numel() == 0:
        compact_host = getattr(attn_inputs, "kv_cache_block_id", None)
    if not isinstance(compact_host, torch.Tensor) or compact_host.numel() == 0:
        return attn_inputs
    if int(compact_host.shape[1]) == 0:
        return attn_inputs

    input_lengths = attn_inputs.input_lengths
    prefix_lengths = attn_inputs.prefix_lengths
    if not isinstance(input_lengths, torch.Tensor) or not isinstance(
        prefix_lengths, torch.Tensor
    ):
        return attn_inputs
    total_lens = input_lengths.detach().cpu().to(torch.int64).reshape(-1)
    total_lens = total_lens + prefix_lengths.detach().cpu().to(torch.int64).reshape(-1)
    if total_lens.numel() == 0:
        return attn_inputs

    kernel_tpb = int(tokens_per_block)
    owner_tpb = (
        int(owner_tokens_per_block)
        if owner_tokens_per_block is not None
        else kernel_tpb
    )
    if owner_tpb <= 0 or kernel_tpb <= 0 or owner_tpb % kernel_tpb != 0:
        raise ValueError(
            f"owner_tokens_per_block ({owner_tpb}) must be a positive multiple "
            f"of tokens_per_block ({kernel_tpb})"
        )
    bpk = owner_tpb // kernel_tpb

    max_global_blocks = int(
        ((int(total_lens.max().item()) + kernel_tpb - 1) // kernel_tpb)
    )
    if max_global_blocks <= 0:
        return attn_inputs

    # Expand at kernel granularity: every global kernel column k maps to
    # compact owner column (k // bpk // cp_size), then to local kernel column
    # owner_col * bpk + (k % bpk). Clamp guards against ranks that hold no
    # owner block (compact width = 0 already early-returned).
    k_global = torch.arange(
        max_global_blocks, dtype=torch.long, device=compact_host.device
    )
    owner_global = k_global // int(bpk)
    owner_local = owner_global // int(cp_size)
    kernel_in_owner = k_global % int(bpk)
    local_cols = (owner_local * int(bpk) + kernel_in_owner).clamp_max(
        int(compact_host.shape[1]) - 1
    )
    safe_host = compact_host.index_select(1, local_cols).contiguous()

    expanded = copy.copy(attn_inputs)
    expanded.kv_cache_kernel_block_id = safe_host
    expanded.kv_cache_block_id = safe_host

    compact_device = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
    if not isinstance(compact_device, torch.Tensor) or compact_device.numel() == 0:
        compact_device = getattr(attn_inputs, "kv_cache_block_id_device", None)
    if isinstance(compact_device, torch.Tensor) and compact_device.numel() > 0:
        device_cols = local_cols.to(device=compact_device.device)
        safe_device = compact_device.index_select(1, device_cols).contiguous()
        expanded.kv_cache_kernel_block_id_device = safe_device
        expanded.kv_cache_block_id_device = safe_device
    return expanded


class SparseMlaFp8CPOp(SparseMlaFp8Op):
    """Context-parallel sparse MLA prefill: all-gather KV, restore to global order,
    write to paged cache, then run attention only on q tokens this rank owns."""

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
        use_cuda_graph: bool = False,
        deterministic_decode: bool = False,
    ):
        super().__init__(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
            use_cuda_graph=use_cuda_graph,
            deterministic_decode=deterministic_decode,
        )
        self.attn_inputs = None
        self.cp_info = None
        self.prefill_cp_rank = int(getattr(parallelism_config, "tp_rank", 0))
        self.prefill_cp_size = int(getattr(parallelism_config, "tp_size", 1))
        self.device = torch.cuda.current_device()
        # Default to kernel granularity (bpk == 1). SparseMlaCpImpl overrides
        # this with attn_configs.tokens_per_block right after construction so
        # page-RR sharding uses the same owner granularity as the C++
        # CPSlotMapper.
        self.kv_owner_tokens_per_block = int(page_size)
        cp_cfg = getattr(parallelism_config, "prefill_cp_config", None)
        self.kv_cache_sharded = bool(
            cp_cfg is not None
            and getattr(cp_cfg, "kv_cache_sharded", False)
            and self.prefill_cp_size > 1
        )

        # Filled per-forward in plan(); read by forward() and create_params()
        self.kv_restore_unpad_indices: Optional[torch.Tensor] = None
        self.total_global_ids: Optional[torch.Tensor] = None
        self.total_local_ids: Optional[torch.Tensor] = None
        self.scatter_reverse_indices = None
        self.precomputed_kv_offsets = None
        self.rank_major_slot_mapping = None
        self.total_local_ids_is_identity: bool = False
        self.cu_kv_seqlens_global: Optional[torch.Tensor] = None
        self.total_kv_len: int = 0
        self.precomputed_req_ids: Optional[torch.Tensor] = None
        self.full_rope_pos_ids: Optional[torch.Tensor] = None
        self.sharded_slot_mapping: Optional[torch.Tensor] = None
        self.sharded_local_kv_lens: Optional[torch.Tensor] = None
        self.sharded_workspace_starts: Optional[torch.Tensor] = None
        self.sharded_kv_restore_indices: Optional[torch.Tensor] = None
        self.sharded_total_local_kv_len: int = 0
        # Capture-stable buffers for the actual-len gather path used by
        # _gather_sharded_kv_cache. Mirror sharded_local_kv_lens /
        # sharded_workspace_starts / sharded_total_local_kv_len but at *actual
        # owned* granularity (cp_actual_owned_kv_lens) instead of padded. The
        # gather kernel walks the compact kernel block_table linearly, so
        # feeding it padded lengths reads past the end of the per-rank table
        # (page_idx in [local_owner_blocks*bpk, padded/kernel_tpb) is OOB).
        # Indexer's _get_topk_ragged_cp already uses this actual/padded split
        # via build_indexer_cp_chunk_plan; these fields bring the MLA dense
        # gather into the same shape.
        self.sharded_actual_local_kv_lens: Optional[torch.Tensor] = None
        self.sharded_actual_workspace_starts: Optional[torch.Tensor] = None
        self.sharded_actual_total_local_kv_len: int = 0
        self.indexer_cp_plan: Optional[asm.IndexerCPChunkPlan] = None
        self.indexer_cp_local_cu: Optional[torch.Tensor] = None
        self.indexer_copy_dst_idx: Optional[torch.Tensor] = None
        self.indexer_src_for_padded: Optional[torch.Tensor] = None
        self._fp8_kernel_metadata_q0: Optional[SparseMlaFp8DecodeParams] = None
        self._fp8_kernel_metadata_q0_key = None
        # Wired up by SparseMlaCpImpl post-construction
        self.kv_cache_write_op = None
        self.write_cache_store_impl = None

    def _refresh_fp8_kernel_metadata(self, n_q: int) -> None:
        key = (
            int(n_q) * int(self.num_heads),
            int(self.top_k),
            int(self.num_heads),
            1,
            True,
        )
        if self.use_cuda_graph and self._fp8_kernel_metadata_q0_key is not None:
            if self._fp8_kernel_metadata_q0_key != key:
                raise RuntimeError(
                    "CP sparse MLA CUDA graph replay changed scheduler shape: "
                    f"captured={self._fp8_kernel_metadata_q0_key}, current={key}"
                )
            return

        # A chunk tail can leave a CP rank with padding only. Such ranks still
        # exchange KV, but must not initialize an attention scheduler for zero Q.
        if n_q == 0:
            self._fp8_kernel_metadata_q0 = None
            self._fp8_kernel_metadata_q0_key = key
            return

        tile_sched_q0, num_splits_q0 = get_mla_metadata(  # type: ignore
            cache_seqlens=None,
            num_q_tokens_per_head_k=key[0],
            topk=self.top_k,
            num_heads_q=self.num_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        self._fp8_kernel_metadata_q0 = SparseMlaFp8DecodeParams(
            tile_sched_q0, num_splits_q0
        )
        self._fp8_kernel_metadata_q0_key = key

    def plan(
        self,
        mla_params: rtp_llm_ops.FlashInferMlaAttnParams,
        block_table: torch.Tensor,
        attn_inputs: Optional[PyAttentionInputs] = None,
    ) -> None:
        self.block_table = block_table
        self.mla_params = mla_params
        # fill_params has already uploaded this forward's lengths. Its host
        # mirror also avoids a prefix readback when the input lives on device.
        if (
            cp_opt_enabled()
            and not self.use_cuda_graph
            and attn_inputs.prefix_lengths.device.type != "cpu"
        ):
            actual = attn_inputs.input_lengths
            indptr = mla_params.prefill_ragged_kv_len_indptr_h
            if (
                actual.device.type == "cpu"
                and actual.dtype == torch.int32
                and actual.ndim == 1
                and actual.shape == attn_inputs.prefix_lengths.shape
                and indptr is not None
                and indptr.device.type == "cpu"
                and indptr.dtype == torch.int32
                and indptr.ndim == 1
                and indptr.numel() >= actual.numel() + 1
            ):
                ptr = indptr[: actual.numel() + 1]
                attn_inputs = copy.copy(attn_inputs)
                attn_inputs.prefix_lengths = ptr[1:] - ptr[:-1] - actual
        self.attn_inputs = attn_inputs
        self.cp_info = attn_inputs.context_parallel_info
        assert self.cp_info is not None, "context_parallel_info required for CP"

        chunk_lengths = self.cp_info.prefill_cp_chunk_lengths
        if isinstance(chunk_lengths, torch.Tensor):
            chunk_lengths_list = chunk_lengths.cpu().tolist()
        else:
            chunk_lengths_list = list(chunk_lengths)
        q0_idx, q1_idx = generate_q_indices(chunk_lengths_list)
        local_tokens = sum(chunk_lengths_list)

        # CPU tensors required by fill_cp_plan_params
        def host_indices(mirror, original):
            if (
                isinstance(mirror, torch.Tensor)
                and mirror.device.type == "cpu"
                and mirror.shape == original.shape
            ):
                return mirror
            return original.cpu()

        padding_mask_cpu = host_indices(
            self.cp_info.prefill_qkv_padding_mask_cpu,
            self.cp_info.prefill_qkv_padding_mask,
        )
        kv_restore_cpu = host_indices(
            self.cp_info.prefill_qkv_restore_indice_cpu,
            self.cp_info.prefill_qkv_restore_indice,
        )

        mla_params.fill_cp_plan_params(
            padding_mask_cpu,
            kv_restore_cpu,
            q0_idx,
            q1_idx,
            self.prefill_cp_rank,
            local_tokens,
            self.cp_info.prefill_actual_input_lengths_cpu,
            self.attn_inputs.prefix_lengths,
        )

        self.kv_restore_unpad_indices = mla_params.cp_kv_restore_unpad_indices
        self.total_global_ids = mla_params.cp_total_global_ids
        self.total_local_ids = mla_params.cp_total_local_ids
        self.total_local_ids_is_identity = _total_local_ids_are_identity(
            padding_mask_cpu,
            kv_restore_cpu,
            q0_idx,
            q1_idx,
            self.prefill_cp_rank,
            local_tokens,
        )
        self.cu_kv_seqlens_global = mla_params.cp_cu_kv_seqlens_global
        self.total_kv_len = mla_params.cp_total_kv_len

        if self.kv_cache_sharded:
            owner_tpb = int(self.kv_owner_tokens_per_block)
            new_slot_mapping = _cp_sharded_slot_mapping(
                mla_params.positions_d,
                block_table,
                mla_params.batch_indice_d,
                self.token_per_block,
                self.prefill_cp_size,
                self.prefill_cp_rank,
                owner_tokens_per_block=owner_tpb,
            )
            self.sharded_slot_mapping = _copy_or_replace_graph_tensor(
                self.sharded_slot_mapping,
                new_slot_mapping,
                "CP sparse MLA sharded_slot_mapping",
                self.use_cuda_graph,
            )
            per_req_total_kv_lens = (
                self.cu_kv_seqlens_global[1:].to(torch.int64)
                - self.cu_kv_seqlens_global[:-1].to(torch.int64)
            ).contiguous()
            # padded local KV lengths and the gather-restore index map must be
            # computed at OWNER granularity to match the C++ CPSlotMapper, not
            # at kernel granularity. With bpk > 1 the kernel-granularity
            # version would straddle physical block boundaries inconsistently
            # with what the C++ writer actually placed in each rank's cache.
            actual = self.cp_info.prefill_actual_input_lengths_cpu
            prefix = self.attn_inputs.prefix_lengths
            host_lengths = None
            if (
                not self.use_cuda_graph
                and actual.device.type == prefix.device.type == "cpu"
            ):
                host_lengths = actual.to(torch.int64) + prefix.to(torch.int64)
            cp_ctx = SimpleNamespace(
                cp_size=self.prefill_cp_size, cp_rank=self.prefill_cp_rank
            )
            self.indexer_cp_plan = asm.build_indexer_cp_chunk_plan(
                cp_ctx=cp_ctx,
                per_req_total_kv_lens=per_req_total_kv_lens,
                per_req_total_kv_lens_host=host_lengths,
                total_kv_len=self.total_kv_len,
                block_size=self.token_per_block,
                device=block_table.device,
                owner_block_size=owner_tpb,
            )
            if host_lengths is not None and cp_opt_enabled():
                from rtp_llm.models_py.triton_kernels.sparse_mla.indexer_early_history import (
                    make_history_plan,
                )

                self.indexer_cp_plan.early_history = make_history_plan(
                    prefix,
                    host_lengths,
                    self.prefill_cp_size,
                    owner_tpb,
                    block_table.device,
                )
            local_lens = self.indexer_cp_plan.per_req_local_kv_lens.to(torch.int32)
            workspace_starts = torch.zeros(
                int(local_lens.numel()), dtype=torch.int32, device=local_lens.device
            )
            if int(local_lens.numel()) > 1:
                workspace_starts[1:] = torch.cumsum(local_lens[:-1], dim=0).to(
                    torch.int32
                )
            restore_indices = self.indexer_cp_plan.restore_indices
            self.sharded_local_kv_lens = _copy_or_replace_graph_tensor(
                self.sharded_local_kv_lens,
                local_lens.contiguous(),
                "CP sparse MLA sharded_local_kv_lens",
                self.use_cuda_graph,
            )
            self.sharded_workspace_starts = _copy_or_replace_graph_tensor(
                self.sharded_workspace_starts,
                workspace_starts.contiguous(),
                "CP sparse MLA sharded_workspace_starts",
                self.use_cuda_graph,
            )
            self.sharded_kv_restore_indices = _copy_or_replace_graph_tensor(
                self.sharded_kv_restore_indices,
                restore_indices,
                "CP sparse MLA sharded_kv_restore_indices",
                self.use_cuda_graph,
            )
            self.sharded_total_local_kv_len = self.indexer_cp_plan.total_local_T
            # Actual owned local KV lens per request (= cp_actual_owned_kv_lens
            # at owner granularity). Used by _gather_sharded_kv_cache to drive
            # the gather kernel only across rows the compact kernel block_table
            # can actually resolve. Mirrors the indexer side's
            # per_req_actual_local_kv_lens in build_indexer_cp_chunk_plan.
            actual_local_lens = self.indexer_cp_plan.per_req_actual_local_kv_lens.to(
                torch.int32
            )
            actual_workspace_starts = torch.zeros(
                int(actual_local_lens.numel()),
                dtype=torch.int32,
                device=actual_local_lens.device,
            )
            if int(actual_local_lens.numel()) > 1:
                actual_workspace_starts[1:] = torch.cumsum(
                    actual_local_lens[:-1], dim=0
                ).to(torch.int32)
            self.sharded_actual_local_kv_lens = _copy_or_replace_graph_tensor(
                self.sharded_actual_local_kv_lens,
                actual_local_lens.contiguous(),
                "CP sparse MLA sharded_actual_local_kv_lens",
                self.use_cuda_graph,
            )
            self.sharded_actual_workspace_starts = _copy_or_replace_graph_tensor(
                self.sharded_actual_workspace_starts,
                actual_workspace_starts.contiguous(),
                "CP sparse MLA sharded_actual_workspace_starts",
                self.use_cuda_graph,
            )
            self.sharded_actual_total_local_kv_len = (
                self.indexer_cp_plan.total_actual_local_T
            )
            self.indexer_cp_local_cu = asm.build_actual_local_cu_kv_seqlens(
                self.indexer_cp_plan
            )
            # Precompute scatter dst_idx for copy_actual_to_padded (B>1, has padding)
            _plan = self.indexer_cp_plan
            _B = int(_plan.per_req_actual_local_kv_lens.numel())
            if (
                _plan.total_actual_local_T > 0
                and _B > 1
                and _plan.total_actual_local_T != _plan.total_local_T
            ):
                _dev = block_table.device
                _a_lens = _plan.per_req_actual_local_kv_lens
                _p_lens = _plan.per_req_local_kv_lens
                _req_ids = torch.repeat_interleave(
                    torch.arange(_B, device=_dev, dtype=torch.int64),
                    _a_lens,
                    output_size=_plan.total_actual_local_T,
                )
                _a_cu = torch.zeros(_B, dtype=torch.int64, device=_dev)
                _p_cu = torch.zeros(_B, dtype=torch.int64, device=_dev)
                _a_cu[1:] = torch.cumsum(_a_lens[:-1], dim=0)
                _p_cu[1:] = torch.cumsum(_p_lens[:-1], dim=0)
                _in_req_pos = torch.arange(
                    _plan.total_actual_local_T, device=_dev, dtype=torch.int64
                ) - _a_cu.index_select(0, _req_ids)
                self.indexer_copy_dst_idx = _copy_or_replace_graph_tensor(
                    self.indexer_copy_dst_idx,
                    _p_cu.index_select(0, _req_ids) + _in_req_pos,
                    "CP actual to padded indices",
                    self.use_cuda_graph,
                )
                # Inverse map: for each padded row, which actual row maps to it (-1 = zero)
                self.indexer_src_for_padded = _copy_or_replace_graph_tensor(
                    self.indexer_src_for_padded,
                    torch.full(
                        (_plan.total_local_T,), -1, dtype=torch.int64, device=_dev
                    ),
                    "CP padded to actual indices",
                    self.use_cuda_graph,
                )
                self.indexer_src_for_padded[self.indexer_copy_dst_idx] = torch.arange(
                    _plan.total_actual_local_T, device=_dev, dtype=torch.int64
                )
            else:
                self.indexer_copy_dst_idx = None
                self.indexer_src_for_padded = None
        else:
            self.sharded_slot_mapping = None
            self.sharded_local_kv_lens = None
            self.sharded_workspace_starts = None
            self.sharded_kv_restore_indices = None
            self.sharded_total_local_kv_len = 0
            self.sharded_actual_local_kv_lens = None
            self.sharded_actual_workspace_starts = None
            self.sharded_actual_total_local_kv_len = 0
            self.indexer_cp_plan = None
            self.indexer_cp_local_cu = None
            self.indexer_copy_dst_idx = None
            self.indexer_src_for_padded = None

        self.rank_major_slot_mapping = None
        if self.kv_cache_sharded and not self.use_cuda_graph and cp_opt_enabled():
            self.rank_major_slot_mapping = _rank_major_slot_mapping(
                self.sharded_slot_mapping,
                self.kv_restore_unpad_indices,
                local_tokens * self.prefill_cp_size,
            )
        reverse = (
            prepare_kv_scatter_indices(self.total_local_ids, local_tokens)
            if cp_opt_enabled()
            and self.total_local_ids.is_cuda
            and not self.total_local_ids_is_identity
            else None
        )
        self.scatter_reverse_indices = _copy_or_replace_graph_tensor(
            self.scatter_reverse_indices,
            reverse,
            "CP scatter indices",
            self.use_cuda_graph,
        )
        n_q = self.total_global_ids.size(0)
        self._refresh_fp8_kernel_metadata(n_q)
        new_req_ids = (
            mla_params.batch_indice_d[self.total_global_ids] if n_q > 0 else None
        )
        self.precomputed_req_ids = _copy_or_replace_graph_tensor(
            self.precomputed_req_ids,
            new_req_ids,
            "CP sparse MLA precomputed_req_ids",
            self.use_cuda_graph,
        )

        # Empty CP ranks still pass a padded buffer through RoPE. Keep its
        # address stable when replaying a CUDA graph, including zero-work ranks.
        positions_d = mla_params.positions_d
        full_rope_size = int(local_tokens)
        if self.use_cuda_graph and self.full_rope_pos_ids is not None:
            if (
                self.full_rope_pos_ids.size(0) != full_rope_size
                or self.full_rope_pos_ids.dtype != positions_d.dtype
                or self.full_rope_pos_ids.device != positions_d.device
            ):
                raise RuntimeError(
                    "CP sparse MLA full_rope_pos_ids shape/dtype/device changed "
                    "during CUDA graph replay"
                )
            full_rope = self.full_rope_pos_ids
            full_rope.zero_()
        else:
            full_rope = torch.zeros(
                full_rope_size, dtype=positions_d.dtype, device=positions_d.device
            )
        if n_q > 0:
            full_rope[self.total_local_ids] = positions_d[self.total_global_ids]
        self.full_rope_pos_ids = full_rope

        # Sharded prefill caches require gathering before attention.
        gather_enabled = (
            self.kv_cache_sharded
            and attn_inputs is not None
            and getattr(attn_inputs, "is_prefill", False)
            and (not self.use_cuda_graph or self.kv_cache_sharded)
        )
        self._gather = self._build_gather_workspace() if gather_enabled else None

        offsets = (
            self._gather.workspace_starts[self.precomputed_req_ids]
            if self._gather is not None and self.precomputed_req_ids is not None
            else None
        )
        self.precomputed_kv_offsets = _copy_or_replace_graph_tensor(
            self.precomputed_kv_offsets, offsets, "CP KV offsets", self.use_cuda_graph
        )

    def _build_gather_workspace(self) -> Optional[_GatherWorkspace]:
        """Record the gather layout from prefill_ragged_kv_len_indptr_d.

        CP's prepare() sets input_lengths = prefill_actual_input_lengths, so the
        indptr reflects per-request full KV length (same as non-CP impl)."""
        assert self.mla_params is not None and self.block_table is not None
        batch_size = int(self.block_table.shape[0])
        indptr = self.mla_params.prefill_ragged_kv_len_indptr_d[: batch_size + 1]
        host_indptr = self.mla_params.prefill_ragged_kv_len_indptr_h
        if (
            not self.use_cuda_graph
            and isinstance(host_indptr, torch.Tensor)
            and host_indptr.device.type == "cpu"
            and host_indptr.numel() >= batch_size + 1
        ):
            total_kv_len = int(host_indptr[batch_size])
        else:
            total_kv_len = int(indptr[batch_size].item())
        if total_kv_len == 0:
            return None
        return _GatherWorkspace(
            workspace_starts=indptr[:batch_size],
            seq_lens=indptr[1:] - indptr[:batch_size],
            total_kv_len=total_kv_len,
            batch_size=batch_size,
        )

    def _convert_topk_indices_to_global(
        self, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """CP: topk rows align with total_local_ids; req_id is precomputed_req_ids
        (= mla_params.batch_indice_d[total_global_ids]) so row i maps to the
        request id of the i-th GLOBAL q token. Returns [T, 1, topk]."""
        from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
            triton_convert_req_index_to_global_index,
        )

        assert self.block_table is not None and self.mla_params is not None
        assert self.precomputed_req_ids is not None
        topk_2d = _topk_2d(topk_indices)
        topk = topk_2d.shape[1]
        assert topk == self.top_k
        global_2d = triton_convert_req_index_to_global_index(
            req_id=self.precomputed_req_ids,
            block_table=self.block_table,
            token_indices=topk_2d,
            TOKENS_PER_BLOCK_FOR_BLOCK_TABLE=self.token_per_block,
            ENTRIES_PER_BLOCK=self.token_per_block,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=min(128, topk),
            HAS_PREFILL_WORKSPACE=False,
        )
        return global_2d.unsqueeze(1)

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        topk: Optional[torch.Tensor],
        batch_indice_d: torch.Tensor,
        kv_cache=None,
        layer_id: int = 0,
        scatter_out: Optional[torch.Tensor] = None,
        prepared_kv=None,
        original_q_len: Optional[int] = None,
        defer_output_scatter: bool = False,
    ) -> torch.Tensor:
        """CP prefill: all-gather → restore → write to kv_cache → attend on q tokens
        owned by this rank (q[total_local_ids]). Returns [total_q_len, H, kv_lora_rank]
        with non-owned positions zero (scattered later by total_local_ids)."""
        fused_kv = None
        if prepared_kv is None:
            if cp_opt_enabled() and compressed_kv.dtype == k_pe.dtype:
                packed = torch.cat((compressed_kv, k_pe), dim=-1)
                gathered = all_gather(packed, group=Group.TP, role="mla_ckv")
                gathered_ckv, gathered_k_pe = gathered.reshape(
                    -1, packed.size(-1)
                ).split((compressed_kv.size(-1), k_pe.size(-1)), dim=-1)
            else:
                gathered_ckv = all_gather(
                    compressed_kv.contiguous(), group=Group.TP, role="mla_ckv"
                ).reshape(-1, compressed_kv.size(-1))
                gathered_k_pe = all_gather(
                    k_pe.contiguous(), group=Group.TP, role="mla_kpe"
                ).reshape(-1, k_pe.size(-1))
            self.write_gathered_kv(gathered_ckv, gathered_k_pe, kv_cache)
        else:
            fused_kv, ready = prepared_kv
            compute = torch.cuda.current_stream(q.device)
            compute.wait_event(ready)
            if isinstance(fused_kv, _RankMajorKv):
                fused_kv.values.record_stream(compute)
            else:
                fused_kv.record_stream(compute)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        no_q_work = (
            topk is None
            or topk.numel() == 0
            or self.total_local_ids is None
            or self.total_local_ids.numel() == 0
        )
        if no_q_work:
            if (
                prepared_kv is None
                and self.kv_cache_sharded
                and self._gather is not None
            ):
                self._gather_sharded_kv_cache(kv_cache, None)
            if self.total_local_ids is not None and self.total_local_ids.numel() == 0:
                return q.new_zeros((q.size(0), q.size(1), self.kv_lora_rank))
            return None

        assert q is not None and q.size(0) > 0
        total_q_len = q.size(0) if original_q_len is None else original_q_len

        use_identity_q = (
            self.total_local_ids_is_identity
            and self.total_local_ids is not None
            and q.size(0) == self.total_local_ids.size(0)
        )
        if original_q_len is not None or use_identity_q:
            q0 = q if q.is_contiguous() else q.contiguous()
        else:
            q0 = q[self.total_local_ids].contiguous()
        if self._gather is not None:
            out0 = self._attend_gather(q0, kv_cache, topk, fused_kv)
        else:
            out0 = self._attend_with_kvcache(q0, kv_cache, topk, layer_id)

        if use_identity_q or defer_output_scatter:
            return out0
        reverse = self.scatter_reverse_indices
        if reverse is not None and reverse.numel() != total_q_len:
            # Direct callers can supply a larger, globally sized Q tensor.
            reverse = None
        return triton_kv_scatter(
            out0,
            self.total_local_ids,
            total_q_len,
            out=scatter_out,
            reverse_indices=reverse,
        )

    def write_gathered_kv(self, gathered_ckv, gathered_k_pe, kv_cache):
        slots = self.rank_major_slot_mapping
        if slots is None:
            slots = (
                self.sharded_slot_mapping
                if self.kv_cache_sharded
                else self.mla_params.slot_mapping
            )
            if (
                cp_opt_enabled()
                and self.kv_cache_write_op.kv_cache_type == "fp8_ds_mla"
            ):
                from rtp_llm.models_py.triton_kernels.sparse_mla.cp_cache_fusion import (
                    try_indexed_cache_write,
                )

                if try_indexed_cache_write(
                    gathered_ckv,
                    gathered_k_pe,
                    self.kv_restore_unpad_indices,
                    slots,
                    _as_uint8(kv_cache.kv_cache_base),
                ):
                    return
            gathered_ckv = gathered_ckv[self.kv_restore_unpad_indices]
            gathered_k_pe = gathered_k_pe[self.kv_restore_unpad_indices]
        self.kv_cache_write_op.forward(
            gathered_ckv,
            gathered_k_pe,
            kv_cache,
            self.mla_params,
            slot_mapping_override=slots,
        )

    def _allocate_fused_kv(self) -> torch.Tensor:
        """Allocate the layer-local BF16 gather destination."""
        ws = self._gather
        assert ws is not None and self.block_table is not None
        return _allocate_prefill_fused_kv(
            ws.total_kv_len,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.block_table.device,
        )

    def _gather_sharded_kv_cache(
        self,
        kv_cache,
        fused_kv: Optional[torch.Tensor],
        *,
        local=None,
        received=None,
        rank_major=False,
    ) -> Optional[_RankMajorKv]:
        """Mirror indexer's sharded gather flow (indexer_op.py:1017-1078):

          1) Drive the GPU gather kernel with *actual* owned lengths so
             ``page_idx = local_idx / kernel_tpb`` never walks past the rank's
             compact kernel block_table tail. Feeding it
             ``sharded_local_kv_lens`` (padded, owner-grain) reads
             ``block_table[bid, page_idx]`` past the end of valid columns —
             that OOB int32 then gets used as a block_id and the kernel
             happily reads K bytes from an unrelated physical block, polluting
             ``local_fused`` padding rows.
          2) Scatter the packed actual rows into a zero-initialized padded
             ``local_fused`` using precomputed per-request offsets. Padding rows
             stay clean zero, matching the indexer's
             ``copy_actual_indexer_k_to_padded`` contract.
          3) all_gather the padded local buffer.
          4) ``gathered[sharded_kv_restore_indices]`` restores logical KV.

        ``sharded_local_kv_lens`` / ``sharded_workspace_starts`` /
        ``sharded_kv_restore_indices`` (padded, owner-grain) are unchanged —
        they only drive the all_gather layout and restore index map, not the
        per-rank read kernel. The new ``sharded_actual_*`` fields are
        gather-only.
        """
        ws = self._gather
        assert ws is not None
        assert self.sharded_local_kv_lens is not None
        assert self.sharded_kv_restore_indices is not None
        assert self.sharded_actual_local_kv_lens is not None
        assert self.sharded_actual_workspace_starts is not None

        src = _as_uint8(kv_cache.kv_cache_base)
        if src.ndim == 4:
            src = src.squeeze(2)

        shape = (
            self.sharded_total_local_kv_len,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        local_fused = (
            local
            if local is not None
            else torch.empty(shape, dtype=torch.bfloat16, device=src.device)
        )
        padded_pack = cp_opt_enabled() and src.is_cuda and not self.use_cuda_graph
        if padded_pack:
            rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_padded(
                src,
                local_fused,
                self.block_table.to(torch.int32),
                self.sharded_actual_local_kv_lens,
                self.sharded_local_kv_lens,
                self.sharded_workspace_starts,
            )
        else:
            local_fused.zero_()
        if not padded_pack and self.sharded_actual_total_local_kv_len > 0:
            direct = (
                ws.batch_size == 1
                or self.sharded_total_local_kv_len
                == self.sharded_actual_total_local_kv_len
            )
            actual_fused = (
                local_fused[: self.sharded_actual_total_local_kv_len]
                if direct
                else local_fused.new_empty(
                    (self.sharded_actual_total_local_kv_len, local_fused.size(1))
                )
            )
            rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
                src,
                actual_fused,
                self.block_table.to(torch.int32),
                self.sharded_actual_local_kv_lens,
                self.sharded_actual_workspace_starts,
                ws.batch_size,
                self.sharded_actual_total_local_kv_len,
            )
            if not direct:
                local_fused.index_copy_(0, self.indexer_copy_dst_idx, actual_fused)

        if received is None:
            gathered = all_gather(local_fused, group=Group.TP, role="mla_history")
        else:
            from rtp_llm.models_py.distributed.pynccl_cp import all_gather_kv_overlap

            gathered = all_gather_kv_overlap(local_fused, Group.TP, output=received)
        gathered = gathered.reshape(-1, local_fused.size(-1))
        if rank_major:
            return _RankMajorKv(gathered, self.sharded_kv_restore_indices)
        if fused_kv is not None:
            torch.index_select(gathered, 0, self.sharded_kv_restore_indices, out=fused_kv)
        return None

    def _attend_with_kvcache(
        self,
        q0: torch.Tensor,
        kv_cache,
        topk: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """flash_mla_with_kvcache on FP8 paged cache (CP equivalent of non-CP baseline)."""
        if self.kv_cache_sharded:
            return self._attend_gather(q0, kv_cache, topk)
        kv_cache_flat = _as_uint8(
            kv_cache.kv_cache_base.view(-1, 1, kv_cache.kv_cache_base.size(-1))
        )
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)
        global_topk = self._convert_topk_indices_to_global(topk).squeeze(1).unsqueeze(0)
        meta = self._fp8_kernel_metadata_q0
        attn_out, _ = flash_mla_with_kvcache(
            q=q0.unsqueeze(0),
            k_cache=kv_cache_flat,
            block_table=self.block_table,
            head_dim_v=self.kv_lora_rank,
            cache_seqlens=None,
            tile_scheduler_metadata=meta.tile_scheduler_metadata,
            num_splits=meta.num_splits,
            is_fp8_kvcache=True,
            indices=global_topk,
            softmax_scale=self.scale,
        )
        return attn_out.squeeze(0)

    def _attend_gather(
        self,
        q0: torch.Tensor,
        kv_cache,
        topk: torch.Tensor,
        fused_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """gather + flash_mla_sparse_fwd. After CP all-gather/restore/write, the paged
        cache has the full per-request KV; the only CP-specific bit is using
        precomputed_req_ids (req id per global q token) for the offset lookup."""
        ws = self._gather
        assert ws is not None and self.precomputed_req_ids is not None
        restore = None
        if fused_kv is None and self.rank_major_slot_mapping is not None:
            fused_kv = self._gather_sharded_kv_cache(kv_cache, None, rank_major=True)
        if isinstance(fused_kv, _RankMajorKv):
            fused_kv, restore = fused_kv
        elif fused_kv is None and self.kv_cache_sharded:
            fused_kv = self._allocate_fused_kv()
            self._gather_sharded_kv_cache(kv_cache, fused_kv)
        elif fused_kv is None:
            fused_kv = self._allocate_fused_kv()
            src = _as_uint8(kv_cache.kv_cache_base)
            if src.ndim == 4:
                src = src.squeeze(2)
            rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
                src,
                fused_kv,
                self.block_table.to(torch.int32),
                ws.seq_lens,
                ws.workspace_starts,
                ws.batch_size,
                ws.total_kv_len,
            )
        global_indices = offset_cp_topk(
            _topk_2d(topk), self.precomputed_kv_offsets, restore, fused_kv.size(0)
        ).unsqueeze(1)
        out, _, _ = flash_mla_sparse_fwd(
            q0,
            fused_kv.unsqueeze(1),
            global_indices,
            self.scale,
            d_v=self.kv_lora_rank,
        )
        return out


class SparseMlaCpImpl(SparseMlaImpl):
    """Sparse MLA wrapper that selects SparseMlaFp8CPOp and packs CP indices."""

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
        cp_cfg = getattr(parallelism_config, "prefill_cp_config", None)
        self._prepared_kv = None
        self._cp_rank = int(getattr(parallelism_config, "tp_rank", 0))
        self._cp_size = int(getattr(parallelism_config, "tp_size", 1))
        # Build the optional direct-NCCL communicator before the first hot
        # gather (and, importantly, before any CUDA graph capture starts).
        # The communicator is process-group/device cached, so later layers are
        # a dictionary lookup only.
        if cp_opt_enabled() and self._cp_size > 1:
            warmup_pynccl(Group.TP)

        self._kv_cache_sharded = bool(
            cp_cfg is not None
            and getattr(cp_cfg, "kv_cache_sharded", False)
            and self._cp_size > 1
        )
        # Owner (physical) block size used by the C++ KVCacheAllocator /
        # CPSlotMapper to decide RR ownership. May be larger than the kernel
        # block size (bpk = owner_tpb / kernel_tpb >= 1). All page-RR sharding
        # logic must use this granularity to stay consistent with the C++
        # allocator; the kernel block size is only used for slot offset math.
        # Must be set BEFORE super().__init__() because the base class may
        # invoke self.prepare() during construction, which reads this attribute.
        self._kv_owner_tokens_per_block = int(
            getattr(
                attn_configs, "tokens_per_block", attn_configs.kernel_tokens_per_block
            )
        )
        # ContextParallelProcessor leaves per-chunk lengths on shared attn_inputs;
        # sparse fill_params / cache_store need per-request actual lengths. Use a
        # shallow copy here so we don't mutate the caller's attn_inputs.
        attn_inputs_for_init = copy.copy(attn_inputs)
        attn_inputs_for_init.input_lengths = (
            attn_inputs.context_parallel_info.prefill_actual_input_lengths_cpu
        )
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
        # Defensive: create_params (called from super().__init__()) already set
        # this before the first plan(). Re-assign here so any post-construction
        # code that swaps fmha_impl still sees the owner granularity.
        self.fmha_impl.kv_owner_tokens_per_block = self._kv_owner_tokens_per_block
        self._glm53_prefill_workspace: Optional[Glm53PrefillWorkspace] = None

    @property
    def overlap_kv_projections(self):
        return (
            self._glm53_prefill_workspace is not None
            and cp_opt_enabled()
            and self._kv_cache_sharded
            and self._cp_size > 1
            and self.attn_configs.q_lora_rank > 0
            and self.fmha_impl._gather is not None
            and not self.fmha_impl.use_cuda_graph
        )

    @contextmanager
    def prefetch_kv_projections(self, compressed_kv, k_pe, kv_cache):
        from rtp_llm.models_py.distributed.pynccl_cp import (
            all_gather_kv_overlap,
            warmup_kv_overlap,
        )

        op = self.fmha_impl
        compute = torch.cuda.current_stream(compressed_kv.device)
        stream = warmup_kv_overlap(Group.TP, compressed_kv.device)
        # Q rotary runs independently on compute. Give the collective its own K.
        key = k_pe
        if op.full_rope_pos_ids is not None:
            self.rope_impl.forward(
                key.unsqueeze(1)[:, :0, :], key, self.rope_params,
                precomputed_pos_ids=op.full_rope_pos_ids,
            )
        rows = compressed_kv.size(0)
        history_rows = op.sharded_total_local_kv_len
        capacity = max(rows, history_rows)
        width = compressed_kv.size(1) + key.size(1)
        local = compressed_kv.new_empty((capacity, width))
        received = local.new_empty((capacity * self._cp_size, width))
        packed = torch.cat((compressed_kv, key), dim=-1, out=local[:rows])
        rope_width = key.size(-1)
        stream.wait_stream(compute)
        try:
            with torch.cuda.stream(stream):
                packed.record_stream(stream)
                gathered = all_gather_kv_overlap(
                    packed, Group.TP, output=received[: rows * self._cp_size]
                )
                op.write_gathered_kv(
                    gathered[:, :-rope_width], gathered[:, -rope_width:], kv_cache
                )
                fused = op._gather_sharded_kv_cache(
                    kv_cache,
                    None,
                    local=local[:history_rows],
                    received=received[: history_rows * self._cp_size],
                    rank_major=True,
                )
                ready = stream.record_event()
            self._prepared_kv = fused, ready
            yield
        finally:
            compute.wait_stream(stream)
            self._prepared_kv = None

    def enable_glm53_prefill_workspace(self) -> None:
        """Enable GLM5.3's per-forward attention workspace manager."""
        self._glm53_prefill_workspace = Glm53PrefillWorkspace()

    def prepare(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> None:
        cp_info = attn_inputs.context_parallel_info
        assert cp_info is not None
        attn_for_prepare = copy.copy(attn_inputs)
        attn_for_prepare.input_lengths = cp_info.prefill_actual_input_lengths_cpu
        if self._kv_cache_sharded:
            safe_attn_for_fill = _safe_expand_cp_sharded_block_table(
                attn_for_prepare,
                self.seq_size_per_block,
                self._cp_size,
                owner_tokens_per_block=self._kv_owner_tokens_per_block,
            )
            self.fmha_params.fill_params(
                safe_attn_for_fill, self.seq_size_per_block, forbid_realloc
            )
            self._refresh_paged_mqa_schedule_metadata(
                safe_attn_for_fill, forbid_realloc
            )
            block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
            if not isinstance(block_table, torch.Tensor) or block_table.numel() == 0:
                block_table = attn_inputs.kv_cache_block_id_device
            self.fmha_impl.plan(
                self.fmha_params,
                block_table,
                attn_inputs=attn_for_prepare,
            )
        else:
            super().prepare(attn_for_prepare, forbid_realloc=forbid_realloc)
        self._refresh_cp_params(use_cuda_graph=forbid_realloc)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.CP_SPARSE_FLASHMLA

    @classmethod
    def support_prefill_cp(cls) -> bool:
        return True

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create fmha_params, run plan() via prepare(), then pack CP indices into
        cp_params for the indexer to consume. plan() already filled
        full_rope_pos_ids and precomputed_req_ids on fmha_impl."""
        self.fmha_params = rtp_llm_ops.SparseMlaParams()
        self.rope_params = self.fmha_params
        # Owner-grain RR contract: plan() reads self.fmha_impl.kv_owner_tokens_per_block
        # to build _cp_sharded_slot_mapping / cp_padded_local_kv_lens /
        # build_kv_allgather_restore_indices. SparseMlaFp8CPOp.__init__ defaults this
        # to kernel granularity; we must override before the first plan() so the
        # very first prefill sees the right bpk and the slot_mapping matches the
        # C++ CPSlotMapper (which shards at seq_size_per_block).
        self.fmha_impl.kv_owner_tokens_per_block = self._kv_owner_tokens_per_block
        self.prepare(attn_inputs)

    def _refresh_cp_params(self, use_cuda_graph: bool = False):
        if self.fmha_params is None:
            return

        gid = self.fmha_impl.total_global_ids
        has_tokens = gid is not None and gid.size(0) > 0

        def _pick(t):
            return t[gid] if has_tokens else None

        new_params = dict(
            kv_restore_unpad_indices=self.fmha_impl.kv_restore_unpad_indices,
            total_global_ids=gid,
            total_local_ids=self.fmha_impl.total_local_ids,
            cu_kv_seqlens_global=self.fmha_impl.cu_kv_seqlens_global,
            total_kv_len=self.fmha_impl.total_kv_len,
            full_rope_pos_ids=self.fmha_impl.full_rope_pos_ids,
            precomputed_ks=_pick(self.fmha_params.ks),
            precomputed_ke=_pick(self.fmha_params.ke),
            precomputed_lengths=_pick(self.fmha_params.expanded_seq_lens),
            precomputed_topk_off=_pick(self.fmha_params.topk_indices_offset),
            precomputed_req_ids=self.fmha_impl.precomputed_req_ids,
            kv_cache_sharded=self.fmha_impl.kv_cache_sharded,
            cp_rank=self._cp_rank,
            cp_size=self._cp_size,
            sharded_slot_mapping=self.fmha_impl.sharded_slot_mapping,
            indexer_cp_plan=self.fmha_impl.indexer_cp_plan,
            indexer_cp_local_cu=self.fmha_impl.indexer_cp_local_cu,
            indexer_copy_dst_idx=self.fmha_impl.indexer_copy_dst_idx,
            indexer_src_for_padded=self.fmha_impl.indexer_src_for_padded,
            total_local_ids_is_identity=self.fmha_impl.total_local_ids_is_identity,
        )

        if self.cp_params is None or not use_cuda_graph:
            self.cp_params = SimpleNamespace(**new_params)
            return

        tensor_fields = (
            "kv_restore_unpad_indices",
            "total_global_ids",
            "total_local_ids",
            "cu_kv_seqlens_global",
            "full_rope_pos_ids",
            "precomputed_ks",
            "precomputed_ke",
            "precomputed_lengths",
            "precomputed_topk_off",
            "precomputed_req_ids",
            "sharded_slot_mapping",
            "indexer_cp_local_cu",
        )
        for name in tensor_fields:
            setattr(
                self.cp_params,
                name,
                _copy_or_replace_graph_tensor(
                    getattr(self.cp_params, name, None),
                    new_params[name],
                    f"CP sparse MLA cp_params.{name}",
                    True,
                ),
            )
        self.cp_params.total_kv_len = new_params["total_kv_len"]
        self.cp_params.kv_cache_sharded = new_params["kv_cache_sharded"]
        self.cp_params.cp_rank = new_params["cp_rank"]
        self.cp_params.cp_size = new_params["cp_size"]
        self.cp_params.indexer_cp_plan = new_params["indexer_cp_plan"]
        self.cp_params.indexer_copy_dst_idx = new_params["indexer_copy_dst_idx"]
        self.cp_params.indexer_src_for_padded = new_params["indexer_src_for_padded"]
        self.cp_params.total_local_ids_is_identity = new_params[
            "total_local_ids_is_identity"
        ]

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        workspace = self._glm53_prefill_workspace
        try:
            return self._forward_impl(
                q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices
            )
        finally:
            # GLM's large MoE follows attention in every layer. Return the slot
            # here so MoE can reuse it; keeping it across layers would raise the
            # non-attention peak. This also covers OOM/kernel exception paths.
            if workspace is not None:
                workspace.release()

    def _forward_impl(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CP sparse MLA forward. q: [total_q_len, H, qk_head_dim]; topk_indices
        is request-local. Returns [total_q_len, H, nope_head_dim]."""
        assert kv_cache is not None

        # RoPE in-place on full q_pe / k_pe via full_rope_pos_ids. Padding rows
        # get pos=0 but never read: q is selected by total_local_ids; k_pe is
        # all-gathered then re-indexed by kv_restore_unpad_indices.
        q_pe = q[:, :, self.nope_head_dim :]
        if (
            self._prepared_kv is not None
            and self.fmha_impl.full_rope_pos_ids is not None
        ):
            self.rope_impl.forward_query(q_pe, self.fmha_impl.full_rope_pos_ids)
        elif self.fmha_impl.full_rope_pos_ids is not None:
            self.rope_impl.forward(
                q_pe,
                k_pe,
                self.rope_params,
                precomputed_pos_ids=self.fmha_impl.full_rope_pos_ids,
            )

        workspace = self._glm53_prefill_workspace
        q_transform_out = None
        scatter_out = None
        if workspace is not None:
            q_transform_out = workspace.q_transformed(
                q.size(0),
                self.num_heads,
                self.kv_lora_rank + self.rope_head_dim,
                dtype=q.dtype,
                device=q.device,
            )

        prepared_kv = self._prepared_kv
        original_q_len = None
        local_ids = self.fmha_impl.total_local_ids
        if (
            q_transform_out is not None
            and not self.fmha_impl.use_cuda_graph
            and not self.fmha_impl.total_local_ids_is_identity
            and local_ids is not None
            and local_ids.numel() > 0
            and topk_indices is not None
            and topk_indices.numel() > 0
        ):
            # RoPE above uses the original token positions. Select the narrower
            # Q (192 + 64), then project directly into the existing workspace
            # (512 + 64), avoiding a second full-size transformed Q allocation.
            original_q_len = q.size(0)
            q_selected = q[local_ids]
            q_transformed = self._apply_input_bmm(
                q_selected, layer_id, out=q_transform_out[: local_ids.numel()]
            )
            # Drop the 256-wide temporary before attention allocates KV/output.
            del q_selected
        else:
            q_transformed = self._apply_input_bmm(q, layer_id, out=q_transform_out)
        defer_scatter = (
            cp_opt_enabled()
            and original_q_len is not None
            and original_q_len >= 256
            and self.fmha_impl.scatter_reverse_indices is not None
            and self.fmha_impl.scatter_reverse_indices.numel() == original_q_len
            and (
                local_ids.numel() == original_q_len
                or (
                    defer_padded_scatter_enabled()
                    and q.dtype == torch.bfloat16
                    and local_ids.numel() >= 1024
                )
            )
            and self.weights[layer_id][W.mla_vc].size(-1) < self.kv_lora_rank
        )
        if workspace is not None and not defer_scatter:
            scatter_out = workspace.scatter_output(
                q.size(0), self.num_heads, self.kv_lora_rank
            )
        attn_output = self.fmha_impl.forward(
            q_transformed,
            compressed_kv,
            k_pe,
            topk_indices,
            self.fmha_params.batch_indice_d,
            kv_cache,
            layer_id=layer_id,
            scatter_out=scatter_out,
            defer_output_scatter=defer_scatter,
            **(
                {"original_q_len": original_q_len} if original_q_len is not None else {}
            ),
            **({"prepared_kv": prepared_kv} if prepared_kv is not None else {}),
        )
        if attn_output is None:
            return None
        output = self._apply_output_bmm(attn_output, layer_id)
        if defer_scatter:
            output = triton_kv_scatter(
                output,
                local_ids,
                original_q_len,
                out=workspace.scatter_output(
                    original_q_len, self.num_heads, output.size(-1)
                ),
                reverse_indices=self.fmha_impl.scatter_reverse_indices,
            )
        return output
