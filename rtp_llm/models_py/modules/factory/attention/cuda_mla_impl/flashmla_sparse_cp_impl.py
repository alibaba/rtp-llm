"""
CP (context-parallel) variant of sparse MLA.
Mirrors flashmla_sparse_impl.py but with all-gather + restore + zig-zag q split.
"""

import copy
import hashlib
import logging
import os
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch

try:
    cuda_ver = torch.version.cuda or ""
    _major, _minor = (int(x) for x in (cuda_ver.split(".") + ["0", "0"])[:2])
    if (_major, _minor) >= (12, 9):
        from flash_mla import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
except (ImportError, AttributeError, ValueError) as _e:
    logging.warning(f"flash_mla not available: {_e}. Requires CUDA >= 12.9")

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4.cp import (
    build_kv_allgather_restore_indices,
    cp_actual_owned_kv_lens,
    cp_padded_local_kv_lens,
)
from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_cp_assembler as asm
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_q_indices,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.triton_kv_scatter import (
    triton_kv_scatter,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops

from .flashmla_sparse_impl import (
    SparseMlaFp8DecodeParams,
    SparseMlaFp8Op,
    SparseMlaImpl,
    _as_uint8,
    _GatherWorkspace,
    _topk_2d,
)

_PD_DEBUG_PLAN_LOGGED: set[str] = set()


def _pd_debug_enabled() -> bool:
    return os.environ.get("RTP_LLM_PD_DEBUG", "0") == "1"


def _rank_tag() -> str:
    return (
        f"rank={os.environ.get('RANK', os.environ.get('WORLD_RANK', '?'))} "
        f"local_rank={os.environ.get('LOCAL_RANK', '?')}"
    )


def _cuda_graph_capturing() -> bool:
    try:
        return bool(
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
    except Exception:
        return False


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
        tc = t.detach()
        if tc.is_cuda:
            tc = tc.cpu()
        if tc.numel() <= 16:
            return f"shape={tuple(t.shape)} values={tc.tolist()}"
        return (
            f"shape={tuple(t.shape)} numel={tc.numel()} "
            f"min={tc.min().item()} max={tc.max().item()} "
            f"head={tc[:4].tolist()} tail={tc[-4:].tolist()}"
        )
    except Exception as exc:
        return f"shape={tuple(t.shape)} summary_error={exc}"


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
    compact_host = getattr(attn_inputs, "kv_cache_kernel_block_id_host", None)
    if not isinstance(compact_host, torch.Tensor) or compact_host.numel() == 0:
        compact_host = getattr(attn_inputs, "kv_cache_block_id_host", None)
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
    expanded.kv_cache_kernel_block_id_host = safe_host
    expanded.kv_cache_block_id_host = safe_host

    compact_device = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
    if not isinstance(compact_device, torch.Tensor) or compact_device.numel() == 0:
        compact_device = getattr(attn_inputs, "kv_cache_block_id_device", None)
    if isinstance(compact_device, torch.Tensor) and compact_device.numel() > 0:
        device_cols = local_cols.to(device=compact_device.device)
        safe_device = compact_device.index_select(1, device_cols).contiguous()
        expanded.kv_cache_kernel_block_id_device = safe_device
        expanded.kv_cache_block_id_device = safe_device
    return expanded


def _scatter_actual_to_padded(
    *,
    actual: torch.Tensor,
    padded: torch.Tensor,
    per_req_actual_local_kv_lens: torch.Tensor,
    per_req_padded_local_kv_lens: torch.Tensor,
) -> None:
    """Scatter packed actual-len rows into the per-request padded local layout.

    Mirrors ``_indexer_cp_assembler.copy_actual_indexer_k_to_padded`` (see
    rtp_llm/models_py/modules/dsv4/fp8/_indexer_cp_assembler.py:129-196) but
    takes a single tensor instead of a (k_quant, k_scale) pair: MLA's
    ``local_fused`` is one ``[T_padded, fused_dim]`` buffer that fuses kv_nope
    + rope into a single all_gather payload. ``dst_idx`` computation, the B==1
    fast path, and dtype handling are kept verbatim with the indexer helper so
    future refactors can swap in a shared scatter API without re-deriving the
    math.

    Caller responsibility: ``padded`` must be zero-initialized (not torch.empty)
    so that the padding rows participate in all_gather as clean zeros — exactly
    matching the indexer flow where ``local_k = torch.zeros(...)`` is used. The
    gather kernel only fills the actual region; padding rows must stay 0 to
    avoid polluting any restore_indices that land on the padded tail (multi-req
    batches, last-block-partial rows, or workspace reuse across layers).
    """
    total_actual = int(actual.shape[0])
    if total_actual == 0:
        return
    if int(per_req_actual_local_kv_lens.numel()) == 1:
        padded[:total_actual].copy_(actual)
        return

    device = padded.device
    actual_lens = per_req_actual_local_kv_lens.to(device=device, dtype=torch.int64)
    padded_lens = per_req_padded_local_kv_lens.to(device=device, dtype=torch.int64)
    B = int(actual_lens.shape[0])

    req_ids = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64),
        actual_lens,
        output_size=total_actual,
    )
    actual_cu = torch.zeros(B, dtype=torch.int64, device=device)
    padded_cu = torch.zeros(B, dtype=torch.int64, device=device)
    if B > 1:
        actual_cu[1:] = torch.cumsum(actual_lens[:-1], dim=0)
        padded_cu[1:] = torch.cumsum(padded_lens[:-1], dim=0)
    in_req_pos = torch.arange(
        total_actual, device=device, dtype=torch.int64
    ) - actual_cu.index_select(0, req_ids)
    dst_idx = padded_cu.index_select(0, req_ids) + in_req_pos
    padded.index_copy_(0, dst_idx, actual)


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
        if self._fp8_kernel_metadata_q0 is None:
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
            return

        if self.use_cuda_graph:
            if self._fp8_kernel_metadata_q0_key != key:
                raise RuntimeError(
                    "CP sparse MLA CUDA graph replay changed scheduler shape: "
                    f"captured={self._fp8_kernel_metadata_q0_key}, current={key}"
                )
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
        padding_mask_cpu = self.cp_info.prefill_qkv_padding_mask
        if padding_mask_cpu.is_cuda:
            padding_mask_cpu = padding_mask_cpu.cpu()
        kv_restore_cpu = self.cp_info.prefill_qkv_restore_indice
        if kv_restore_cpu.is_cuda:
            kv_restore_cpu = kv_restore_cpu.cpu()

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
            local_lens = cp_padded_local_kv_lens(
                per_req_total_kv_lens,
                self.prefill_cp_size,
                owner_tpb,
            ).to(torch.int32)
            workspace_starts = torch.zeros(
                int(local_lens.numel()), dtype=torch.int32, device=local_lens.device
            )
            if int(local_lens.numel()) > 1:
                workspace_starts[1:] = torch.cumsum(local_lens[:-1], dim=0).to(
                    torch.int32
                )
            restore_indices = build_kv_allgather_restore_indices(
                per_req_total_kv_lens,
                self.prefill_cp_size,
                owner_tpb,
                block_table.device,
            )
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
            self.sharded_total_local_kv_len = int(local_lens.sum().item())
            # Actual owned local KV lens per request (= cp_actual_owned_kv_lens
            # at owner granularity). Used by _gather_sharded_kv_cache to drive
            # the gather kernel only across rows the compact kernel block_table
            # can actually resolve. Mirrors the indexer side's
            # per_req_actual_local_kv_lens in build_indexer_cp_chunk_plan.
            actual_local_lens = cp_actual_owned_kv_lens(
                per_req_total_kv_lens,
                self.prefill_cp_size,
                owner_tpb,
                self.prefill_cp_rank,
            ).to(torch.int32)
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
            self.sharded_actual_total_local_kv_len = int(actual_local_lens.sum().item())
            cp_ctx = SimpleNamespace(
                cp_size=self.prefill_cp_size, cp_rank=self.prefill_cp_rank
            )
            self.indexer_cp_plan = asm.build_indexer_cp_chunk_plan(
                cp_ctx=cp_ctx,
                per_req_total_kv_lens=per_req_total_kv_lens,
                block_size=self.token_per_block,
                device=block_table.device,
                owner_block_size=owner_tpb,
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
                self.indexer_copy_dst_idx = (
                    _p_cu.index_select(0, _req_ids) + _in_req_pos
                )
                # Inverse map: for each padded row, which actual row maps to it (-1 = zero)
                self.indexer_src_for_padded = torch.full(
                    (_plan.total_local_T,), -1, dtype=torch.int64, device=_dev
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

        # Pack rope positions for the local padded q/k buffer; padding rows keep
        # pos=0 and are never selected by total_local_ids.
        if n_q > 0:
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
                    full_rope_size,
                    dtype=positions_d.dtype,
                    device=positions_d.device,
                )
            full_rope[self.total_local_ids] = positions_d[self.total_global_ids]
            self.full_rope_pos_ids = full_rope
        else:
            if self.use_cuda_graph and self.full_rope_pos_ids is not None:
                raise RuntimeError(
                    "CP sparse MLA full_rope_pos_ids became None during CUDA graph replay"
                )
            self.full_rope_pos_ids = None

        # Gather path: prefill-only, gated by USE_GATHER_PATH (mirrors non-CP).
        gather_enabled = (
            (os.environ.get("USE_GATHER_PATH", "0") == "1" or self.kv_cache_sharded)
            and attn_inputs is not None
            and getattr(attn_inputs, "is_prefill", False)
            and (not self.use_cuda_graph or self.kv_cache_sharded)
        )
        self._gather = self._build_gather_workspace() if gather_enabled else None

        if _pd_debug_enabled():
            log_key = f"{os.getpid()}:{self.prefill_cp_rank}"
            if log_key not in _PD_DEBUG_PLAN_LOGGED:
                _PD_DEBUG_PLAN_LOGGED.add(log_key)
                logging.info(
                    "[PD_DEBUG][CP_MLA_PLAN] %s cp_rank=%s cp_size=%s "
                    "chunk_lengths=%s actual_lengths=%s prefix_lengths=%s "
                    "local_tokens=%s kv_restore=%s total_global=%s total_local=%s "
                    "cu_kv=%s total_kv_len=%s local_ids_identity=%s "
                    "gather_enabled=%s",
                    _rank_tag(),
                    self.prefill_cp_rank,
                    self.prefill_cp_size,
                    chunk_lengths_list,
                    _tensor_summary(self.cp_info.prefill_actual_input_lengths_cpu),
                    _tensor_summary(self.attn_inputs.prefix_lengths),
                    local_tokens,
                    _tensor_summary(self.kv_restore_unpad_indices),
                    _tensor_summary(self.total_global_ids),
                    _tensor_summary(self.total_local_ids),
                    _tensor_summary(self.cu_kv_seqlens_global),
                    self.total_kv_len,
                    self.total_local_ids_is_identity,
                    self._gather is not None,
                )

    def _build_gather_workspace(self) -> Optional[_GatherWorkspace]:
        """Allocate the BF16 workspace from prefill_ragged_kv_len_indptr_d.

        CP's prepare() sets input_lengths = prefill_actual_input_lengths, so the
        indptr reflects per-request full KV length (same as non-CP impl)."""
        assert self.mla_params is not None and self.block_table is not None
        batch_size = int(self.block_table.shape[0])
        indptr = self.mla_params.prefill_ragged_kv_len_indptr_d[: batch_size + 1]
        total_kv_len = int(indptr[batch_size].item())
        if total_kv_len == 0:
            return None
        return _GatherWorkspace(
            fused_kv=torch.empty(
                (total_kv_len, self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=torch.bfloat16,
                device=self.block_table.device,
            ),
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
            # REBASE CONFLICT CONTEXT(e2e00e570): source branch used
            # `token_indices=topk_2d, BLOCK_SIZE=self.token_per_block`; new base
            # uses explicit `TOKENS_PER_BLOCK_FOR_BLOCK_TABLE` and
            # `ENTRIES_PER_BLOCK`. Keep that API while preserving the source
            # branch's normalized `topk_2d` rows.
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
    ) -> torch.Tensor:
        """CP prefill: all-gather → restore → write to kv_cache → attend on q tokens
        owned by this rank (q[total_local_ids]). Returns [total_q_len, H, kv_lora_rank]
        with non-owned positions zero (scattered later by total_local_ids)."""
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
            slot_mapping_override=(
                self.sharded_slot_mapping if self.kv_cache_sharded else None
            ),
        )
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
            if self.kv_cache_sharded and self._gather is not None:
                self._gather_sharded_kv_cache(kv_cache)
            return None

        assert q is not None and q.size(0) > 0

        use_identity_q = (
            self.total_local_ids_is_identity
            and self.total_local_ids is not None
            and q.size(0) == self.total_local_ids.size(0)
        )
        if use_identity_q:
            q0 = q if q.is_contiguous() else q.contiguous()
        else:
            q0 = q[self.total_local_ids].contiguous()
        if self._gather is not None:
            out0 = self._attend_gather(q0, kv_cache, topk)
        else:
            out0 = self._attend_with_kvcache(q0, kv_cache, topk, layer_id)

        if use_identity_q:
            return out0
        out = triton_kv_scatter(out0, self.total_local_ids, q.size(0))
        return out

    def _gather_sharded_kv_cache(self, kv_cache) -> None:
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
             ``local_fused`` via ``_scatter_actual_to_padded``. Padding rows
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

        # Padded buffer MUST be torch.zeros (not torch.empty): padding rows
        # participate in NCCL all_gather and must stay clean 0. Otherwise a
        # restore_indices value that lands on a padding row (multi-request
        # batches with differing per-req actual/padded, last-block-partial,
        # or workspace reuse across layers) propagates uninitialized memory
        # into ws.fused_kv → garbled attention output.
        local_fused = torch.zeros(
            (self.sharded_total_local_kv_len, ws.fused_kv.size(1)),
            dtype=ws.fused_kv.dtype,
            device=ws.fused_kv.device,
        )

        if self.sharded_actual_total_local_kv_len > 0:
            actual_fused = torch.empty(
                (self.sharded_actual_total_local_kv_len, ws.fused_kv.size(1)),
                dtype=ws.fused_kv.dtype,
                device=ws.fused_kv.device,
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
            _scatter_actual_to_padded(
                actual=actual_fused,
                padded=local_fused,
                per_req_actual_local_kv_lens=self.sharded_actual_local_kv_lens,
                per_req_padded_local_kv_lens=self.sharded_local_kv_lens,
            )

        gathered = all_gather(local_fused.contiguous(), group=Group.TP).reshape(
            -1, local_fused.size(-1)
        )
        ws.fused_kv.copy_(gathered[self.sharded_kv_restore_indices])

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
    ) -> torch.Tensor:
        """gather + flash_mla_sparse_fwd. After CP all-gather/restore/write, the paged
        cache has the full per-request KV; the only CP-specific bit is using
        precomputed_req_ids (req id per global q token) for the offset lookup."""
        ws = self._gather
        assert ws is not None and self.precomputed_req_ids is not None
        if self.kv_cache_sharded:
            self._gather_sharded_kv_cache(kv_cache)
        else:
            src = _as_uint8(kv_cache.kv_cache_base)
            if src.ndim == 4:
                src = src.squeeze(2)
            rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
                src,
                ws.fused_kv,
                self.block_table.to(torch.int32),
                ws.seq_lens,
                ws.workspace_starts,
                ws.batch_size,
                ws.total_kv_len,
            )
        offsets = ws.workspace_starts[self.precomputed_req_ids]
        topk_2d = _topk_2d(topk)
        # FIX: topk_2d contains -1 as padding (invalid KV position).
        # Adding offsets to -1 turns it into a large positive index that
        # points into another request's KV region in fused_kv, causing
        # cross-request KV pollution → gibberish / repeat output.
        # Preserve -1 so flash_mla_sparse_fwd skips these positions.
        padding_mask = topk_2d < 0
        raw_global = topk_2d + offsets.unsqueeze(1)
        global_indices = raw_global.masked_fill(padding_mask, -1).unsqueeze(1)
        out, _, _ = flash_mla_sparse_fwd(
            q0,
            ws.fused_kv.unsqueeze(1),
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
        self._cp_rank = int(getattr(parallelism_config, "tp_rank", 0))
        self._cp_size = int(getattr(parallelism_config, "tp_size", 1))
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
        # REBASE CONFLICT CONTEXT(e2e00e570): source branch introduced a longer
        # `create_params` implementation; new base has the same fields expressed
        # through the compact `_pick()` helper. Keep the compact base form.
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
        """CP sparse MLA forward. q: [total_q_len, H, qk_head_dim]; topk_indices
        is request-local. Returns [total_q_len, H, nope_head_dim]."""
        assert kv_cache is not None

        # RoPE in-place on full q_pe / k_pe via full_rope_pos_ids. Padding rows
        # get pos=0 but never read: q is selected by total_local_ids; k_pe is
        # all-gathered then re-indexed by kv_restore_unpad_indices.
        q_pe = q[:, :, self.nope_head_dim :]
        if self.fmha_impl.full_rope_pos_ids is not None:
            self.rope_impl.forward(
                q_pe,
                k_pe,
                self.rope_params,
                precomputed_pos_ids=self.fmha_impl.full_rope_pos_ids,
            )

        q_transformed = self._apply_input_bmm(q, layer_id)
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
