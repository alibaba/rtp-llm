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

from dataclasses import dataclass
from typing import Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    build_kv_allgather_restore_indices,
    cp_padded_local_kv_lens,
)


@dataclass
class IndexerCPChunkPlan:
    """Per-chunk metadata so indexer.py only computes restore once per chunk."""

    cp_ctx: CPContext
    # Per-request KV length (in indexer-pool entries) for the chunk's req slice.
    per_req_total_kv_lens: torch.Tensor  # [b_chunk] int64
    # Per-rank local lengths (== n_virtual_blocks * block_size).
    per_req_local_kv_lens: torch.Tensor  # [b_chunk] int64
    total_local_T: int
    # restore_indices into [cp_size * total_local_T] gathered rows.
    restore_indices: torch.Tensor  # [chunk_T] int64
    block_size: int


def build_indexer_cp_chunk_plan(
    cp_ctx: CPContext,
    per_req_total_kv_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> IndexerCPChunkPlan:
    """Build per-chunk indexer assembler plan (CPU; no NCCL)."""
    if cp_ctx.cp_size <= 0:
        raise ValueError(f"cp_size must be > 0, got {cp_ctx.cp_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    per_req = per_req_total_kv_lens.to(device=device, dtype=torch.int64).contiguous()
    local_lens = cp_padded_local_kv_lens(
        per_req, cp_ctx.cp_size, block_size
    ).contiguous()
    total_local = int(local_lens.sum().item())
    restore = build_kv_allgather_restore_indices(
        per_req, cp_ctx.cp_size, block_size, device
    )
    return IndexerCPChunkPlan(
        cp_ctx=cp_ctx,
        per_req_total_kv_lens=per_req,
        per_req_local_kv_lens=local_lens,
        total_local_T=total_local,
        restore_indices=restore,
        block_size=block_size,
    )


def build_local_cu_kv_seqlens(plan: IndexerCPChunkPlan) -> torch.Tensor:
    """Per-rank cu_kv_seqlens (int32) over local lengths for the C++ op."""
    cu = torch.zeros(
        plan.per_req_local_kv_lens.numel() + 1,
        dtype=torch.int32,
        device=plan.per_req_local_kv_lens.device,
    )
    cu[1:] = torch.cumsum(plan.per_req_local_kv_lens, dim=0).to(torch.int32)
    return cu


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
    if local_k_quant.shape[0] != plan.total_local_T:
        raise ValueError(
            f"local_k_quant rows {local_k_quant.shape[0]} != total_local_T "
            f"{plan.total_local_T}"
        )
    if local_k_scale.shape[0] != plan.total_local_T:
        raise ValueError(
            f"local_k_scale rows {local_k_scale.shape[0]} != total_local_T "
            f"{plan.total_local_T}"
        )
    chunk_T = int(plan.restore_indices.numel())
    if out_k_quant.shape[0] != chunk_T or out_k_scale.shape[0] != chunk_T:
        raise ValueError(
            f"out shapes [{out_k_quant.shape[0]}, {out_k_scale.shape[0]}] != "
            f"chunk_T {chunk_T}"
        )
    if chunk_T == 0:
        return
    # Use raw rank-major all_gather (NOT cp_all_gather_full_async, which
    # asserts T_local == cp_ctx.chunk_length — that's the prefill-token
    # space; here local_k_* lives in KV-pool-entry space sized by
    # plan.total_local_T = sum_b n_virtual_blocks * block_size). The
    # restore_indices are built against this rank-major layout.
    # INVARIANT: the nested compressor writer lands its indexer-pool writes
    # on the current stream (cp_wait_gather_full + writer _launch), so NCCL
    # stream-ordering alone suffices for visibility. No CPU-blocking sync —
    # at chunk granularity this would serialize the pipeline. See
    # _pool_reader.fill for the same pattern.
    gathered_q = all_gather(local_k_quant, group=Group.TP)
    gathered_s = all_gather(local_k_scale, group=Group.TP)
    out_k_quant.copy_(gathered_q[plan.restore_indices])
    out_k_scale.copy_(gathered_s[plan.restore_indices])
