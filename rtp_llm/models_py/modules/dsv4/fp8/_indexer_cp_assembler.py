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

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    build_kv_allgather_restore_indices,
    cp_actual_owned_kv_lens,
    cp_padded_local_kv_lens,
)


def _actual_to_padded_index_opt_enabled() -> bool:
    return os.environ.get("RTP_LLM_CP_ACTUAL_TO_PADDED_INDEX_OPT", "1") != "0"


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


def build_actual_to_padded_indices(
    plan: IndexerCPChunkPlan,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build reusable actual->padded scatter indices from small CPU metadata.

    ``plan.per_req_*`` tensors are tiny request-length vectors. Building the
    expanded row map with CUDA ``repeat_interleave`` is disproportionately
    expensive on prefill batches because it can trigger host synchronization
    and first-use CUDA kernel overhead. Construct the row map on CPU, then copy
    the final compact index tensors to the target device once per plan.
    """
    total_actual = int(plan.total_actual_local_T)
    total_padded = int(plan.total_local_T)
    if not _actual_to_padded_index_opt_enabled():
        return None, None
    if total_actual == 0 or total_actual == total_padded:
        return None, None
    if int(plan.per_req_actual_local_kv_lens.numel()) == 1:
        return None, None

    target_device = device if device is not None else plan.per_req_local_kv_lens.device
    actual_lens = (
        plan.per_req_actual_local_kv_lens.detach()
        .to(device="cpu", dtype=torch.int64)
        .reshape(-1)
    )
    padded_lens = (
        plan.per_req_local_kv_lens.detach()
        .to(device="cpu", dtype=torch.int64)
        .reshape(-1)
    )

    dst_cpu = torch.empty(total_actual, dtype=torch.int64)
    out = 0
    padded_base = 0
    for actual_len_t, padded_len_t in zip(actual_lens, padded_lens):
        actual_len = int(actual_len_t.item())
        if actual_len > 0:
            dst_cpu[out : out + actual_len] = torch.arange(
                padded_base,
                padded_base + actual_len,
                dtype=torch.int64,
            )
            out += actual_len
        padded_base += int(padded_len_t.item())

    if out != total_actual:
        raise RuntimeError(
            f"built {out} actual->padded indices, expected {total_actual}"
        )

    src_cpu = torch.full((total_padded,), -1, dtype=torch.int64)
    src_cpu[dst_cpu] = torch.arange(total_actual, dtype=torch.int64)
    return (
        dst_cpu.to(device=target_device, non_blocking=True),
        src_cpu.to(device=target_device, non_blocking=True),
    )


def copy_actual_indexer_k_to_padded(
    *,
    plan: IndexerCPChunkPlan,
    actual_k_quant: torch.Tensor,
    actual_k_scale: torch.Tensor,
    padded_k_quant: torch.Tensor,
    padded_k_scale: torch.Tensor,
    dst_idx: Optional[torch.Tensor] = None,
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
    if dst_idx is not None:
        padded_k_quant.view(torch.uint8).index_copy_(
            0, dst_idx, actual_k_quant.view(torch.uint8)
        )
        padded_k_scale.index_copy_(0, dst_idx, actual_k_scale)
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
    # restore_indices are built against this rank-major layout.
    # INVARIANT: the nested compressor writer lands its indexer-pool writes
    # on the current stream (cp_wait_gather_full + writer _launch), so NCCL
    # stream-ordering alone suffices for visibility. No CPU-blocking sync —
    # at chunk granularity this would serialize the pipeline. See
    # _pool_reader.fill for the same pattern.
    with record_function_range("dsv4.cp.all_gather.indexer_k.quant.launch"):
        gathered_q = all_gather(local_k_quant, group=Group.TP)
    with record_function_range("dsv4.cp.all_gather.indexer_k.scale.launch"):
        gathered_s = all_gather(local_k_scale, group=Group.TP)
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

    from rtp_llm.models_py.distributed import collective_torch

    process_group = collective_torch._get_group(Group.TP)
    world_size = torch.distributed.get_world_size(process_group)
    if world_size != plan.cp_ctx.cp_size:
        raise RuntimeError(
            f"indexer K gather world_size({world_size}) != cp_size({plan.cp_ctx.cp_size})"
        )

    device = local_k_quant.device
    current_stream = torch.cuda.current_stream(device)
    stream.wait_stream(current_stream)
    local_k_quant.record_stream(stream)
    local_k_scale.record_stream(stream)
    with torch.cuda.stream(stream):
        gathered_q = torch.empty(
            (world_size * int(local_k_quant.shape[0]), local_k_quant.shape[1]),
            dtype=local_k_quant.dtype,
            device=device,
        )
        gathered_s = torch.empty(
            (world_size * int(local_k_scale.shape[0]), local_k_scale.shape[1]),
            dtype=local_k_scale.dtype,
            device=device,
        )
        with record_function_range("dsv4.cp.all_gather.indexer_k.quant.launch"):
            work_q = torch.distributed.all_gather_into_tensor(
                gathered_q,
                local_k_quant,
                group=process_group,
                async_op=True,
            )
        with record_function_range("dsv4.cp.all_gather.indexer_k.scale.launch"):
            try:
                work_s = torch.distributed.all_gather_into_tensor(
                    gathered_s,
                    local_k_scale,
                    group=process_group,
                    async_op=True,
                )
            except Exception:
                # quant gather already launched — drain it before propagating.
                work_q.wait()
                raise
        try:
            completion_event = torch.cuda.Event()
            completion_event.record(stream)
        except Exception:
            # Both NCCL gathers in flight but pending handle never returns;
            # drain both before propagating.
            work_q.wait()
            work_s.wait()
            raise

    return IndexerKCPGatherHandle(
        plan=plan,
        gathered_q=gathered_q,
        gathered_s=gathered_s,
        work_q=work_q,
        work_s=work_s,
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
        handle.work_q.wait()
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
