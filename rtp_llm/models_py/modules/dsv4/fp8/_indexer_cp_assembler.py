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
from typing import Any, Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    build_kv_allgather_restore_indices,
    cp_actual_owned_kv_lens,
    cp_padded_local_kv_lens,
)


@dataclass
class IndexerCPChunkPlan:
    """Per-chunk metadata so indexer.py only computes restore once per chunk."""

    cp_ctx: CPContext
    # Per-request KV length (in indexer-pool entries) for the chunk's req slice.
    per_req_total_kv_lens: torch.Tensor  # [b_chunk] int64
    # Per-rank padded local lengths (== n_virtual_blocks * block_size).
    per_req_local_kv_lens: torch.Tensor  # [b_chunk] int64
    # Per-rank actual owned lengths for the current cp_rank. The last owned
    # block may be partial; callers must not read past this from the pool.
    per_req_actual_local_kv_lens: torch.Tensor  # [b_chunk] int64
    total_local_T: int
    total_actual_local_T: int
    # restore_indices into [cp_size * total_local_T] gathered rows.
    restore_indices: torch.Tensor  # [chunk_T] int64
    block_size: int
    owner_block_size: int


@dataclass
class IndexerKCPGatherHandle:
    plan: IndexerCPChunkPlan
    gathered_q: torch.Tensor
    gathered_s: torch.Tensor
    work_q: Any
    work_s: Any
    completion_event: torch.cuda.Event
    stream: torch.cuda.Stream
    out_k_quant: torch.Tensor
    out_k_scale: torch.Tensor
    done_event: Optional[torch.cuda.Event] = None
    work_waited: bool = False


def build_indexer_cp_chunk_plan(
    cp_ctx: CPContext,
    per_req_total_kv_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
    owner_block_size: Optional[int] = None,
) -> IndexerCPChunkPlan:
    """Build per-chunk indexer assembler plan (CPU; no NCCL)."""
    if cp_ctx.cp_size <= 0:
        raise ValueError(f"cp_size must be > 0, got {cp_ctx.cp_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    owner_bs = int(owner_block_size or block_size)
    if owner_bs <= 0:
        raise ValueError(f"owner_block_size must be > 0, got {owner_bs}")
    per_req = per_req_total_kv_lens.to(device=device, dtype=torch.int64).contiguous()
    local_lens = cp_padded_local_kv_lens(per_req, cp_ctx.cp_size, owner_bs).contiguous()
    actual_lens = cp_actual_owned_kv_lens(
        per_req, cp_ctx.cp_size, owner_bs, cp_ctx.cp_rank
    ).contiguous()
    # Host scalars size the rank-local all_gather buffers.  Keep these syncs in
    # plan construction; the per-chunk gather/restore path below should remain
    # device work plus the single NCCL Work.wait().
    total_local = int(local_lens.sum().item())
    total_actual = int(actual_lens.sum().item())
    restore = build_kv_allgather_restore_indices(
        per_req, cp_ctx.cp_size, owner_bs, device
    )
    return IndexerCPChunkPlan(
        cp_ctx=cp_ctx,
        per_req_total_kv_lens=per_req,
        per_req_local_kv_lens=local_lens,
        per_req_actual_local_kv_lens=actual_lens,
        total_local_T=total_local,
        total_actual_local_T=total_actual,
        restore_indices=restore,
        block_size=block_size,
        owner_block_size=owner_bs,
    )


def build_local_cu_kv_seqlens(plan: IndexerCPChunkPlan) -> torch.Tensor:
    """Per-rank cu_kv_seqlens (int32) over padded local lengths."""
    cu = torch.zeros(
        plan.per_req_local_kv_lens.numel() + 1,
        dtype=torch.int32,
        device=plan.per_req_local_kv_lens.device,
    )
    cu[1:] = torch.cumsum(plan.per_req_local_kv_lens, dim=0).to(torch.int32)
    return cu


def build_actual_local_cu_kv_seqlens(plan: IndexerCPChunkPlan) -> torch.Tensor:
    """Per-rank cu_kv_seqlens (int32) over actual owned local lengths.

    This is the length vector that should drive the paged-pool read kernel.
    The padded cu vector is only for NCCL shape/restore layout.
    """
    cu = torch.zeros(
        plan.per_req_actual_local_kv_lens.numel() + 1,
        dtype=torch.int32,
        device=plan.per_req_actual_local_kv_lens.device,
    )
    cu[1:] = torch.cumsum(plan.per_req_actual_local_kv_lens, dim=0).to(torch.int32)
    return cu


def copy_actual_indexer_k_to_padded(
    *,
    plan: IndexerCPChunkPlan,
    actual_k_quant: torch.Tensor,
    actual_k_scale: torch.Tensor,
    padded_k_quant: torch.Tensor,
    padded_k_scale: torch.Tensor,
) -> None:
    """Scatter compact actual rows into the padded per-rank local layout.

    ``cp_gather_indexer_k_quant_cache`` can only express one compact
    ``cu_seq_lens`` layout. For CP-sharded storage we must pass actual owned
    lengths to avoid reading partial-block garbage, but the downstream
    all_gather restore expects each request to start at the padded local base.
    This helper inserts the per-request gaps and leaves those tails zero.
    """
    if actual_k_quant.shape[0] != plan.total_actual_local_T:
        raise ValueError(
            f"actual_k_quant rows {actual_k_quant.shape[0]} != "
            f"total_actual_local_T {plan.total_actual_local_T}"
        )
    if actual_k_scale.shape[0] != plan.total_actual_local_T:
        raise ValueError(
            f"actual_k_scale rows {actual_k_scale.shape[0]} != "
            f"total_actual_local_T {plan.total_actual_local_T}"
        )
    if padded_k_quant.shape[0] != plan.total_local_T:
        raise ValueError(
            f"padded_k_quant rows {padded_k_quant.shape[0]} != "
            f"total_local_T {plan.total_local_T}"
        )
    if padded_k_scale.shape[0] != plan.total_local_T:
        raise ValueError(
            f"padded_k_scale rows {padded_k_scale.shape[0]} != "
            f"total_local_T {plan.total_local_T}"
        )
    total_actual = plan.total_actual_local_T
    if total_actual == 0:
        return

    device = padded_k_quant.device
    if int(plan.per_req_actual_local_kv_lens.numel()) == 1:
        padded_k_quant[:total_actual].copy_(actual_k_quant)
        padded_k_scale[:total_actual].copy_(actual_k_scale)
        return

    actual_lens = plan.per_req_actual_local_kv_lens.to(device=device, dtype=torch.int64)
    padded_lens = plan.per_req_local_kv_lens.to(device=device, dtype=torch.int64)
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
    padded_k_quant.view(torch.uint8).index_copy_(
        0, dst_idx, actual_k_quant.view(torch.uint8)
    )
    padded_k_scale.index_copy_(0, dst_idx, actual_k_scale)


def _pack_indexer_k_payload(
    local_k_quant: torch.Tensor,
    local_k_scale: torch.Tensor,
) -> "tuple[torch.Tensor, int, int, torch.dtype, torch.dtype]":
    if local_k_quant.dim() != 2 or local_k_scale.dim() != 2:
        raise ValueError(
            "indexer K pack expects 2D quant/scale buffers, got "
            f"{tuple(local_k_quant.shape)} and {tuple(local_k_scale.shape)}"
        )
    if local_k_quant.device != local_k_scale.device:
        raise ValueError("indexer K pack requires quant/scale on the same device")
    rows = int(local_k_quant.shape[0])
    if int(local_k_scale.shape[0]) != rows:
        raise ValueError(
            f"indexer K pack row mismatch: quant={rows}, "
            f"scale={int(local_k_scale.shape[0])}"
        )
    local_q_bytes = local_k_quant.contiguous().view(torch.uint8)
    local_s_bytes = local_k_scale.contiguous().view(torch.uint8)
    q_cols = int(local_q_bytes.shape[1])
    s_cols = int(local_s_bytes.shape[1])
    packed = torch.empty(
        (rows, q_cols + s_cols),
        dtype=torch.uint8,
        device=local_k_quant.device,
    )
    packed[:, :q_cols].copy_(local_q_bytes)
    packed[:, q_cols:].copy_(local_s_bytes)
    return packed, q_cols, s_cols, local_k_quant.dtype, local_k_scale.dtype


def _split_indexer_k_payload(
    gathered_packed: torch.Tensor,
    q_cols: int,
    s_cols: int,
    q_dtype: torch.dtype,
    s_dtype: torch.dtype,
) -> "tuple[torch.Tensor, torch.Tensor]":
    if gathered_packed.dim() != 2:
        raise ValueError(
            f"packed indexer K gather must be 2D, got {tuple(gathered_packed.shape)}"
        )
    expected_cols = int(q_cols) + int(s_cols)
    if int(gathered_packed.shape[1]) != expected_cols:
        raise ValueError(
            f"packed indexer K cols {int(gathered_packed.shape[1])} != "
            f"expected {expected_cols}"
        )
    if q_dtype != torch.uint8 or s_dtype != torch.uint8:
        raise ValueError(
            "packed indexer K currently expects uint8 quant/scale byte buffers, "
            f"got {q_dtype} and {s_dtype}"
        )
    return (
        gathered_packed[:, : int(q_cols)],
        gathered_packed[:, int(q_cols) : expected_cols],
    )


def _all_gather_indexer_k_packed_sync(
    local_packed: torch.Tensor,
    plan: IndexerCPChunkPlan,
) -> torch.Tensor:
    profile_name = "dsv4.cp.all_gather.indexer_k.packed"

    def torch_fallback() -> torch.Tensor:
        with record_function_range(f"{profile_name}.launch"):
            return all_gather(local_packed, group=Group.TP)

    if not (local_packed.is_cuda and torch.distributed.is_initialized()):
        return torch_fallback()

    from rtp_llm.models_py.distributed import collective_torch, pynccl_cp

    process_group = collective_torch._get_group(Group.TP)
    world_size = torch.distributed.get_world_size(process_group)
    if world_size != plan.cp_ctx.cp_size:
        raise RuntimeError(
            f"indexer K packed gather world_size({world_size}) "
            f"!= cp_size({plan.cp_ctx.cp_size})"
        )
    rows = world_size * int(local_packed.shape[0])
    cols = int(local_packed.shape[1])
    return pynccl_cp.cp_all_gather_sync(
        local_packed,
        lambda r, c, d: torch.empty((r, c), dtype=d, device=local_packed.device),
        torch_fallback,
        role="indexer_k_packed",
        rows=rows,
        cols=cols,
        process_group=process_group,
        stream=torch.cuda.current_stream(local_packed.device),
        profile_name=profile_name,
        symm_variable=True,
    )


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
    chunk_T = _validate_assemble_args(
        plan=plan,
        local_k_quant=local_k_quant,
        local_k_scale=local_k_scale,
        out_k_quant=out_k_quant,
        out_k_scale=out_k_scale,
    )
    if chunk_T == 0:
        return
    # Use raw rank-major all_gather (NOT cp_all_gather_full_async, which
    # asserts T_local == cp_ctx.chunk_length — that's the prefill-token
    # space; here local_k_* lives in KV-pool-entry space sized by
    # plan.total_local_T = sum_b n_virtual_blocks * block_size). The
    # restore_indices are built against this rank-major layout. Quant and scale
    # share identical row ownership, so pack them row-wise and issue one
    # all_gather instead of two small back-to-back NCCL collectives.
    # INVARIANT: the nested compressor writer lands its indexer-pool writes
    # on the current stream (cp_wait_gather_full + writer _launch), so NCCL
    # stream-ordering alone suffices for visibility. No CPU-blocking sync —
    # at chunk granularity this would serialize the pipeline. See
    # _pool_reader.fill for the same pattern.
    local_packed, q_cols, s_cols, q_dtype, s_dtype = _pack_indexer_k_payload(
        local_k_quant, local_k_scale
    )
    gathered_packed = _all_gather_indexer_k_packed_sync(local_packed, plan)
    gathered_q, gathered_s = _split_indexer_k_payload(
        gathered_packed, q_cols, s_cols, q_dtype, s_dtype
    )
    restore_indexer_k(plan, gathered_q, gathered_s, out_k_quant, out_k_scale)


def start_assemble_indexer_k_async(
    *,
    plan: IndexerCPChunkPlan,
    local_k_quant: torch.Tensor,
    local_k_scale: torch.Tensor,
    out_k_quant: torch.Tensor,
    out_k_scale: torch.Tensor,
    stream: torch.cuda.Stream,
) -> Optional[IndexerKCPGatherHandle]:
    chunk_T = _validate_assemble_args(
        plan=plan,
        local_k_quant=local_k_quant,
        local_k_scale=local_k_scale,
        out_k_quant=out_k_quant,
        out_k_scale=out_k_scale,
    )
    if chunk_T == 0:
        return None
    if (
        stream is None
        or not local_k_quant.is_cuda
        or not torch.distributed.is_initialized()
        or torch.cuda.is_current_stream_capturing()
    ):
        return None

    from rtp_llm.models_py.distributed import collective_torch, pynccl_cp

    process_group = collective_torch._get_group(Group.TP)
    world_size = torch.distributed.get_world_size(process_group)
    if world_size != plan.cp_ctx.cp_size:
        raise RuntimeError(
            f"indexer K gather world_size({world_size}) != cp_size({plan.cp_ctx.cp_size})"
        )

    device = local_k_quant.device
    current_stream = torch.cuda.current_stream(device)
    local_packed, q_cols, s_cols, q_dtype, s_dtype = _pack_indexer_k_payload(
        local_k_quant, local_k_scale
    )
    stream.wait_stream(current_stream)
    local_packed.record_stream(stream)
    # Quant+scale use the same rank-major row layout, so pack them into one
    # uint8 payload and issue a single all_gather. ``indexer_k_packed`` is not a
    # symmetric-memory role: under DSV4_CP_SYMM it falls back to ordinary pynccl,
    # avoiding extra persistent windows for this async-specialized path.
    def _empty(r, c, d):
        return torch.empty((r, c), dtype=d, device=device)

    gathered_packed, work_packed, completion_event = pynccl_cp.cp_all_gather(
        local_packed,
        _empty,
        role="indexer_k_packed",
        rows=world_size * int(local_packed.shape[0]),
        cols=int(local_packed.shape[1]),
        process_group=process_group,
        gather_stream=stream,
        profile_name="dsv4.cp.all_gather.indexer_k.packed",
        symm_variable=True,
    )
    gathered_q, gathered_s = _split_indexer_k_payload(
        gathered_packed, q_cols, s_cols, q_dtype, s_dtype
    )

    return IndexerKCPGatherHandle(
        plan=plan,
        gathered_q=gathered_q,
        gathered_s=gathered_s,
        work_q=work_packed,
        work_s=None,
        completion_event=completion_event,
        stream=stream,
        out_k_quant=out_k_quant,
        out_k_scale=out_k_scale,
    )


def prepare_assemble_indexer_k_async(
    handle: IndexerKCPGatherHandle,
    *,
    stream: torch.cuda.Stream,
) -> None:
    if handle.done_event is not None:
        return
    if stream is None:
        raise ValueError("prepare_assemble_indexer_k_async requires an explicit stream")
    current_stream = torch.cuda.current_stream(handle.out_k_quant.device)
    # The output buffers are caller-owned and may have just been initialized on
    # the current stream.  Preserve that ordering here instead of relying on
    # allocator stream semantics alone.
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        stream.wait_event(handle.completion_event)
        # Fence NCCL on the stream that will restore into out_k_* below.
        with record_function_range("dsv4.cp.all_gather.indexer_k.wait_host"):
            _wait_indexer_k_work_once(handle)
        handle.gathered_q.record_stream(stream)
        handle.gathered_s.record_stream(stream)
        handle.out_k_quant.record_stream(stream)
        handle.out_k_scale.record_stream(stream)
        try:
            restore_indexer_k(
                handle.plan,
                handle.gathered_q,
                handle.gathered_s,
                handle.out_k_quant,
                handle.out_k_scale,
            )
        finally:
            handle.done_event = torch.cuda.Event()
            handle.done_event.record(stream)


def wait_assemble_indexer_k_async(handle: IndexerKCPGatherHandle) -> None:
    current_stream = torch.cuda.current_stream(handle.out_k_quant.device)
    if handle.done_event is not None:
        current_stream.wait_event(handle.done_event)
    else:
        current_stream.wait_event(handle.completion_event)
    _wait_indexer_k_work_once(handle)


def discard_assemble_indexer_k_async(handle: IndexerKCPGatherHandle) -> None:
    wait_assemble_indexer_k_async(handle)


def _wait_indexer_k_work_once(handle: IndexerKCPGatherHandle) -> None:
    if not handle.work_waited:
        if handle.work_q is not None:  # None on the pynccl path (stream-ordered)
            handle.work_q.wait()
        if handle.work_s is not None:
            handle.work_s.wait()
        handle.work_waited = True


def restore_indexer_k(
    plan: IndexerCPChunkPlan,
    gathered_q: torch.Tensor,
    gathered_s: torch.Tensor,
    out_k_quant: torch.Tensor,
    out_k_scale: torch.Tensor,
) -> None:
    with record_function_range("dsv4.cp.all_gather.indexer_k.restore"):
        out_k_quant.copy_(gathered_q[plan.restore_indices])
        out_k_scale.copy_(gathered_s[plan.restore_indices])


def _validate_assemble_args(
    *,
    plan: IndexerCPChunkPlan,
    local_k_quant: torch.Tensor,
    local_k_scale: torch.Tensor,
    out_k_quant: torch.Tensor,
    out_k_scale: torch.Tensor,
) -> int:
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
    return chunk_T
