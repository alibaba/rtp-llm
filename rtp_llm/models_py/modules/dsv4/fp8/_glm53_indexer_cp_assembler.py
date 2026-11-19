"""CP-sharded indexer K assembler (Stage 5b-4).

Under ``kv_cache_sharded`` each CP rank's ``INDEXER_KV`` pool only holds
the RR-owned 1/cp_size of logical entries. The existing
``rtp_llm_ops.cp_gather_indexer_k_quant_cache`` C++ op reads from
``[block_table[r, :n_blocks]]`` into packed ``k_quant``/``k_scale``
buffers; with the rank's local block_table that yields **only owned**
positions in compact form. To restore the chunk's expected logical layout
this assembler does:

  1. Compute per-rank ``local_T = sum_r owned_token_count(T_r)`` for the
     chunk's request slice.
  2. Build a ``local`` cu_kv_seqlens / block_table view that points at
     each request's owned local blocks (== the rank's existing
     block_table sliced to ``ceil(T_r/(block_size*cp_size))`` entries).
  3. Call the existing C++ op with packed local buffers
     ``[local_T, …]``.
  4. NCCL all_gather → ``[cp_size * local_T, …]``.
  5. Pre-built ``restore_indices`` (from
     ``cp.build_kv_allgather_restore_indices``) reorder gathered rows
     into request-concatenated logical token order, written into the
     caller-provided ``[chunk_T, …]`` buffers.

Only one all_gather per chunk per layer; the C++ op is unchanged. The
assembler is pure CPU/PyTorch glue around the existing primitives so it
can be unit-tested independently.
"""

from __future__ import annotations

from typing import Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.distributed.glm53_collective_gemm import all_gather
from rtp_llm.models_py.modules.dsv4.cp import CPContext
from rtp_llm.models_py.modules.dsv4.forward_metadata import metadata_cache
from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_cp_assembler as shared
from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler import (
    IndexerCPChunkPlan,
    build_local_cu_kv_seqlens,
    copy_actual_indexer_k_to_padded,
)


def build_indexer_cp_chunk_plan(
    cp_ctx: CPContext,
    per_req_total_kv_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
    owner_block_size: Optional[int] = None,
    geometry_key: Optional[tuple] = None,
) -> IndexerCPChunkPlan:
    """Build per-chunk indexer assembler plan (CPU; no NCCL)."""
    if cp_ctx.cp_size <= 0:
        raise ValueError(f"cp_size must be > 0, got {cp_ctx.cp_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    owner_bs = int(owner_block_size or block_size)
    if owner_bs <= 0:
        raise ValueError(f"owner_block_size must be > 0, got {owner_bs}")
    cache = metadata_cache() if geometry_key is not None else None
    cache_key = (
        "indexer_cp_plan",
        id(cp_ctx),
        geometry_key,
        block_size,
        owner_bs,
        per_req_total_kv_lens.numel(),
        str(device),
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    plan = shared.build_indexer_cp_chunk_plan(
        cp_ctx, per_req_total_kv_lens, block_size, device, owner_block_size
    )
    if cache is not None:
        cache[cache_key] = plan
    return plan


def build_actual_local_cu_kv_seqlens(plan: IndexerCPChunkPlan) -> torch.Tensor:
    """Per-rank cu_kv_seqlens (int32) over actual owned local lengths.

    This is the length vector that should drive the paged-pool read kernel.
    The padded cu vector is only for NCCL shape/restore layout.
    """
    if getattr(plan, "actual_local_cu", None) is None:
        plan.actual_local_cu = shared.build_actual_local_cu_kv_seqlens(plan)
    return plan.actual_local_cu


def assemble_indexer_k(
    *,
    plan: IndexerCPChunkPlan,
    local_k_quant: torch.Tensor,  # [total_local_T, slot_q_bytes] uint8
    local_k_scale: torch.Tensor,  # [total_local_T, scale_bytes] uint8
    out_k_quant: torch.Tensor,  # [chunk_T, slot_q_bytes] uint8
    out_k_scale: torch.Tensor,  # [chunk_T, scale_bytes] uint8
) -> None:
    """all_gather owned local K + restore into out buffers.

    Caller is responsible for invoking ``rtp_llm_ops.cp_gather_indexer_k_quant_cache``
    with the rank-local ``cu_kv_seqlens`` (from ``build_local_cu_kv_seqlens``)
    and block_table to fill ``local_k_quant`` / ``local_k_scale`` first.
    """
    chunk_T = shared._validate_assemble_args(
        plan=plan,
        local_k_quant=local_k_quant,
        local_k_scale=local_k_scale,
        out_k_quant=out_k_quant,
        out_k_scale=out_k_scale,
    )
    if chunk_T == 0:
        return
    # Use the GLM prefill TP group for the two rank-major KV shards.
    gathered_q = all_gather(local_k_quant, group=Group.TP)
    gathered_s = all_gather(local_k_scale, group=Group.TP)
    shared.restore_indexer_k(plan, gathered_q, gathered_s, out_k_quant, out_k_scale)
