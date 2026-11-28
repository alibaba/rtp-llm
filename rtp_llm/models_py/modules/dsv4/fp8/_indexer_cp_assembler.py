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

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.cp import (
    CoalescedCPGatherHandle,
    CPContext,
    build_kv_allgather_restore_indices,
    cp_actual_owned_kv_len,
    cp_actual_owned_kv_lens,
    cp_padded_local_kv_len,
    cp_padded_local_kv_lens,
    discard_cp_all_gather_coalesced,
    prepare_cp_all_gather_coalesced,
    start_cp_all_gather_coalesced,
    wait_cp_all_gather_coalesced,
)

_ORIG_ALL_GATHER = all_gather
_CSA_KV_COALESCE_TLS = threading.local()


def _acquire_asm_buffer(
    device: torch.device,
    slot_key: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.uint8,
) -> torch.Tensor:
    acq = getattr(collective_torch, "acquire_grow_only_buffer", None)
    if (
        acq is not None
        and all_gather is _ORIG_ALL_GATHER
        and torch.distributed.is_initialized()
        and (device.type != "cuda" or not torch.cuda.is_current_stream_capturing())
    ):
        return acq(device, slot_key, shape, dtype=dtype)
    return torch.empty(shape, dtype=dtype, device=device)


@dataclass
class CSAKVCoalesceJob:
    """Thread-local carrier coalescing INDEXER_KV with CSA_KV and SWA_KV."""

    cmp_local_flat: Optional[torch.Tensor] = None
    swa_local_slices: Optional[torch.Tensor] = None
    gathered_cmp: Optional[torch.Tensor] = None
    gathered_swa: Optional[torch.Tensor] = None
    ready_event: Optional[Any] = None
    ready_stream: Optional[Any] = None
    producer_stream: Optional[Any] = None
    active_gather_used: bool = False
    consumed: bool = False


def get_active_csa_kv_coalesce_job() -> Optional[CSAKVCoalesceJob]:
    stack = getattr(_CSA_KV_COALESCE_TLS, "stack", None)
    if stack:
        return stack[-1]
    return None


@contextmanager
def csa_kv_coalesce_scope(
    job: Optional[CSAKVCoalesceJob],
) -> Iterator[Optional[CSAKVCoalesceJob]]:
    if job is None:
        yield None
        return
    stack = getattr(_CSA_KV_COALESCE_TLS, "stack", None)
    if stack is None:
        stack = []
        _CSA_KV_COALESCE_TLS.stack = stack
    stack.append(job)
    try:
        yield job
    finally:
        stack.pop()


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
    gathered_q: Optional[torch.Tensor]
    gathered_s: Optional[torch.Tensor]
    work_q: Any
    work_s: Any
    completion_event: torch.cuda.Event
    stream: torch.cuda.Stream
    producer_stream: torch.cuda.Stream
    local_k_quant: torch.Tensor
    local_k_scale: torch.Tensor
    out_k_quant: torch.Tensor
    out_k_scale: torch.Tensor
    done_event: Optional[torch.cuda.Event] = None
    work_waited: bool = False
    packed_q_cols: Optional[int] = None
    local_packed: Optional[torch.Tensor] = None
    coalesced_handle: Optional[CoalescedCPGatherHandle] = None
    coalesced_job: Optional[CSAKVCoalesceJob] = None
    cmp_item_idx: int = -1
    swa_item_idx: int = -1


def build_indexer_cp_chunk_plan(
    cp_ctx: CPContext,
    per_req_total_kv_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
    owner_block_size: Optional[int] = None,
    total_kv_len: Optional[int] = None,
) -> IndexerCPChunkPlan:
    """Build per-chunk indexer assembler plan (CPU; no NCCL)."""
    if cp_ctx.cp_size <= 0:
        raise ValueError(f"cp_size must be > 0, got {cp_ctx.cp_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    owner_bs = int(owner_block_size or block_size)
    if owner_bs <= 0:
        raise ValueError(f"owner_block_size must be > 0, got {owner_bs}")
    if per_req_total_kv_lens.device.type == "cpu":
        per_req_cpu = per_req_total_kv_lens.to(dtype=torch.int64).contiguous()
        non_blocking = device.type == "cuda"
        if int(per_req_cpu.numel()) == 1 and total_kv_len is not None:
            total_local = cp_padded_local_kv_len(
                int(total_kv_len), cp_ctx.cp_size, owner_bs
            )
            total_actual = cp_actual_owned_kv_len(
                int(total_kv_len),
                cp_ctx.cp_size,
                owner_bs,
                cp_ctx.cp_rank,
            )
            local_lens = torch.full((1,), total_local, dtype=torch.int64, device=device)
            actual_lens = torch.full(
                (1,), total_actual, dtype=torch.int64, device=device
            )
        else:
            local_lens_cpu = cp_padded_local_kv_lens(
                per_req_cpu, cp_ctx.cp_size, owner_bs
            ).contiguous()
            actual_lens_cpu = cp_actual_owned_kv_lens(
                per_req_cpu, cp_ctx.cp_size, owner_bs, cp_ctx.cp_rank
            ).contiguous()
            total_local = int(local_lens_cpu.sum().item())
            total_actual = int(actual_lens_cpu.sum().item())
            local_lens = local_lens_cpu.to(device=device, non_blocking=non_blocking)
            actual_lens = actual_lens_cpu.to(device=device, non_blocking=non_blocking)
        per_req = per_req_cpu.to(device=device, non_blocking=non_blocking)
    else:
        per_req = per_req_total_kv_lens.to(
            device=device, dtype=torch.int64
        ).contiguous()
        if int(per_req.numel()) == 1 and total_kv_len is not None:
            total_local = cp_padded_local_kv_len(
                int(total_kv_len), cp_ctx.cp_size, owner_bs
            )
            total_actual = cp_actual_owned_kv_len(
                int(total_kv_len),
                cp_ctx.cp_size,
                owner_bs,
                cp_ctx.cp_rank,
            )
            local_lens = torch.full((1,), total_local, dtype=torch.int64, device=device)
            actual_lens = torch.full(
                (1,), total_actual, dtype=torch.int64, device=device
            )
        else:
            local_lens = cp_padded_local_kv_lens(
                per_req, cp_ctx.cp_size, owner_bs
            ).contiguous()
            actual_lens = cp_actual_owned_kv_lens(
                per_req, cp_ctx.cp_size, owner_bs, cp_ctx.cp_rank
            ).contiguous()
            # Host scalars size the rank-local all_gather buffers. Consolidate the
            # multi-request fallback into one synchronization.
            total_local, total_actual = (
                int(v)
                for v in torch.stack([local_lens.sum(), actual_lens.sum()]).tolist()
            )
    restore = build_kv_allgather_restore_indices(
        per_req,
        cp_ctx.cp_size,
        owner_bs,
        device,
        total_kv_len=total_kv_len,
        total_local_kv=total_local,
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
    n = int(plan.per_req_local_kv_lens.numel())
    cu = torch.empty(
        n + 1,
        dtype=torch.int32,
        device=plan.per_req_local_kv_lens.device,
    )
    cu[0] = 0
    if n > 0:
        torch.cumsum(plan.per_req_local_kv_lens, dim=0, dtype=torch.int32, out=cu[1:])
    return cu


def build_actual_local_cu_kv_seqlens(plan: IndexerCPChunkPlan) -> torch.Tensor:
    """Per-rank cu_kv_seqlens (int32) over actual owned local lengths.

    This is the length vector that should drive the paged-pool read kernel.
    The padded cu vector is only for NCCL shape/restore layout.
    """
    n = int(plan.per_req_actual_local_kv_lens.numel())
    cu = torch.empty(
        n + 1,
        dtype=torch.int32,
        device=plan.per_req_actual_local_kv_lens.device,
    )
    cu[0] = 0
    if n > 0:
        torch.cumsum(
            plan.per_req_actual_local_kv_lens, dim=0, dtype=torch.int32, out=cu[1:]
        )
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
    actual_lens = plan.per_req_actual_local_kv_lens
    B = int(actual_lens.numel())
    if B == 1:
        padded_k_quant[:total_actual].copy_(actual_k_quant)
        padded_k_scale[:total_actual].copy_(actual_k_scale)
        return

    padded_lens = plan.per_req_local_kv_lens
    if (
        _TRITON_AVAILABLE
        and padded_k_quant.is_cuda
        and actual_k_quant.is_cuda
        and actual_k_scale.is_cuda
        and padded_k_scale.is_cuda
        and actual_k_quant.dim() == 2
        and actual_k_scale.dim() == 2
        and padded_k_quant.dim() == 2
        and padded_k_scale.dim() == 2
        and actual_k_quant.is_contiguous()
        and actual_k_scale.is_contiguous()
        and padded_k_quant.is_contiguous()
        and padded_k_scale.is_contiguous()
        and B <= 256
    ):
        aq_bytes = actual_k_quant.view(torch.uint8)
        as_bytes = actual_k_scale.view(torch.uint8)
        pq_bytes = padded_k_quant.view(torch.uint8)
        ps_bytes = padded_k_scale.view(torch.uint8)
        q_cols = int(aq_bytes.shape[1])
        s_cols = int(as_bytes.shape[1])
        if (
            int(pq_bytes.shape[1]) == q_cols
            and int(ps_bytes.shape[1]) == s_cols
            and q_cols > 0
            and s_cols > 0
        ):
            actual_lens_d = actual_lens.to(device=device)
            padded_lens_d = padded_lens.to(device=device)
            block_m = 16
            _copy_actual_indexer_k_to_padded_kernel[
                (triton.cdiv(total_actual, block_m),)
            ](
                aq_bytes,
                as_bytes,
                pq_bytes,
                ps_bytes,
                actual_lens_d,
                padded_lens_d,
                total_actual=total_actual,
                batch_size=B,
                q_cols=q_cols,
                s_cols=s_cols,
                aq_stride=int(aq_bytes.stride(0)),
                as_stride=int(as_bytes.stride(0)),
                pq_stride=int(pq_bytes.stride(0)),
                ps_stride=int(ps_bytes.stride(0)),
                BLOCK_M=block_m,
                BLOCK_K=_next_power_of_2(q_cols),
                BLOCK_S=_next_power_of_2(s_cols),
                BATCH_BLOCK=_next_power_of_2(B),
                num_warps=4,
            )
            return

    actual_lens_l = actual_lens.to(device=device, dtype=torch.int64)
    padded_lens_l = padded_lens.to(device=device, dtype=torch.int64)
    req_ids = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64),
        actual_lens_l,
        output_size=total_actual,
    )

    actual_cu = torch.zeros(B, dtype=torch.int64, device=device)
    padded_cu = torch.zeros(B, dtype=torch.int64, device=device)
    if B > 1:
        actual_cu[1:] = torch.cumsum(actual_lens_l[:-1], dim=0)
        padded_cu[1:] = torch.cumsum(padded_lens_l[:-1], dim=0)
    in_req_pos = torch.arange(
        total_actual, device=device, dtype=torch.int64
    ) - actual_cu.index_select(0, req_ids)
    dst_idx = padded_cu.index_select(0, req_ids) + in_req_pos
    padded_k_quant.view(torch.uint8).index_copy_(
        0, dst_idx, actual_k_quant.view(torch.uint8)
    )
    padded_k_scale.index_copy_(0, dst_idx, actual_k_scale)


if _TRITON_AVAILABLE:

    @triton.jit(
        do_not_specialize=[
            "total_actual",
            "batch_size",
            "q_cols",
            "s_cols",
            "aq_stride",
            "as_stride",
            "pq_stride",
            "ps_stride",
        ]
    )
    def _copy_actual_indexer_k_to_padded_kernel(
        actual_q_ptr,
        actual_s_ptr,
        padded_q_ptr,
        padded_s_ptr,
        actual_lens_ptr,
        padded_lens_ptr,
        total_actual,
        batch_size,
        q_cols,
        s_cols,
        aq_stride,
        as_stride,
        pq_stride,
        ps_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BATCH_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_ids < total_actual

        batch_offsets = tl.arange(0, BATCH_BLOCK)
        batch_mask = batch_offsets < batch_size
        actual_lens = tl.load(
            actual_lens_ptr + batch_offsets, mask=batch_mask, other=0
        ).to(tl.int64)
        padded_lens = tl.load(
            padded_lens_ptr + batch_offsets, mask=batch_mask, other=0
        ).to(tl.int64)
        actual_ends = tl.cumsum(actual_lens, axis=0)
        actual_starts = actual_ends - actual_lens
        padded_starts = tl.cumsum(padded_lens, axis=0) - padded_lens

        batch_idx = tl.sum(
            (row_ids[:, None] >= actual_ends[None, :]).to(tl.int32), axis=1
        )
        eq_mask = batch_idx[:, None] == batch_offsets[None, :]
        actual_start = tl.sum(tl.where(eq_mask, actual_starts[None, :], 0), axis=1)
        padded_start = tl.sum(tl.where(eq_mask, padded_starts[None, :], 0), axis=1)
        dst_rows = padded_start + (row_ids - actual_start)

        k_cols = tl.arange(0, BLOCK_K)
        k_mask = row_mask[:, None] & (k_cols[None, :] < q_cols)
        k_vals = tl.load(
            actual_q_ptr + row_ids[:, None] * aq_stride.to(tl.int64) + k_cols[None, :],
            mask=k_mask,
            other=0,
        )
        tl.store(
            padded_q_ptr + dst_rows[:, None] * pq_stride.to(tl.int64) + k_cols[None, :],
            k_vals,
            mask=k_mask,
        )

        s_idx = tl.arange(0, BLOCK_S)
        s_mask = row_mask[:, None] & (s_idx[None, :] < s_cols)
        s_vals = tl.load(
            actual_s_ptr + row_ids[:, None] * as_stride.to(tl.int64) + s_idx[None, :],
            mask=s_mask,
            other=0,
        )
        tl.store(
            padded_s_ptr + dst_rows[:, None] * ps_stride.to(tl.int64) + s_idx[None, :],
            s_vals,
            mask=s_mask,
        )

    @triton.jit(do_not_specialize=["n_rows", "q_stride", "s_stride", "packed_stride"])
    def _pack_indexer_k_kernel(
        local_q_ptr,
        local_s_ptr,
        packed_ptr,
        n_rows,
        q_stride,
        s_stride,
        packed_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row_ids = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_ids < n_rows
        k_cols = tl.arange(0, BLOCK_K)
        k_mask = row_mask[:, None] & (k_cols[None, :] < q_stride)
        k_vals = tl.load(
            local_q_ptr + row_ids[:, None] * q_stride.to(tl.int64) + k_cols[None, :],
            mask=k_mask,
            other=0,
        )
        tl.store(
            packed_ptr
            + row_ids[:, None] * packed_stride.to(tl.int64)
            + k_cols[None, :],
            k_vals,
            mask=k_mask,
        )

        s_cols = tl.arange(0, BLOCK_S)
        s_mask = row_mask[:, None] & (s_cols[None, :] < s_stride)
        s_vals = tl.load(
            local_s_ptr + row_ids[:, None] * s_stride.to(tl.int64) + s_cols[None, :],
            mask=s_mask,
            other=0,
        )
        tl.store(
            packed_ptr
            + row_ids[:, None] * packed_stride.to(tl.int64)
            + q_stride.to(tl.int64)
            + s_cols[None, :],
            s_vals,
            mask=s_mask,
        )

    @triton.jit(do_not_specialize=["n_rows", "packed_stride", "k_stride", "s_stride"])
    def _restore_packed_indexer_k_kernel(
        gathered_packed_ptr,
        out_k_ptr,
        out_s_ptr,
        indices_ptr,
        packed_stride,
        k_stride,
        s_stride,
        n_rows,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row_ids = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_ids < n_rows
        src_rows = tl.load(indices_ptr + row_ids, mask=row_mask, other=0).to(tl.int64)

        k_cols = tl.arange(0, BLOCK_K)
        k_mask = row_mask[:, None] & (k_cols[None, :] < k_stride)
        k_vals = tl.load(
            gathered_packed_ptr
            + src_rows[:, None] * packed_stride.to(tl.int64)
            + k_cols[None, :],
            mask=k_mask,
            other=0,
        )
        tl.store(
            out_k_ptr + row_ids[:, None] * k_stride.to(tl.int64) + k_cols[None, :],
            k_vals,
            mask=k_mask,
        )

        s_cols = tl.arange(0, BLOCK_S)
        s_mask = row_mask[:, None] & (s_cols[None, :] < s_stride)
        s_vals = tl.load(
            gathered_packed_ptr
            + src_rows[:, None] * packed_stride.to(tl.int64)
            + k_stride.to(tl.int64)
            + s_cols[None, :],
            mask=s_mask,
            other=0,
        )
        tl.store(
            out_s_ptr + row_ids[:, None] * s_stride.to(tl.int64) + s_cols[None, :],
            s_vals,
            mask=s_mask,
        )

    @triton.jit(do_not_specialize=["n_rows", "k_stride", "s_stride"])
    def _restore_separate_indexer_k_kernel(
        gathered_k_ptr,
        gathered_s_ptr,
        out_k_ptr,
        out_s_ptr,
        indices_ptr,
        k_stride,
        s_stride,
        n_rows,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row_ids = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_ids < n_rows
        src_rows = tl.load(indices_ptr + row_ids, mask=row_mask, other=0).to(tl.int64)

        k_cols = tl.arange(0, BLOCK_K)
        k_mask = row_mask[:, None] & (k_cols[None, :] < k_stride)
        k_vals = tl.load(
            gathered_k_ptr
            + src_rows[:, None] * k_stride.to(tl.int64)
            + k_cols[None, :],
            mask=k_mask,
            other=0,
        )
        tl.store(
            out_k_ptr + row_ids[:, None] * k_stride.to(tl.int64) + k_cols[None, :],
            k_vals,
            mask=k_mask,
        )

        s_cols = tl.arange(0, BLOCK_S)
        s_mask = row_mask[:, None] & (s_cols[None, :] < s_stride)
        s_vals = tl.load(
            gathered_s_ptr
            + src_rows[:, None] * s_stride.to(tl.int64)
            + s_cols[None, :],
            mask=s_mask,
            other=0,
        )
        tl.store(
            out_s_ptr + row_ids[:, None] * s_stride.to(tl.int64) + s_cols[None, :],
            s_vals,
            mask=s_mask,
        )


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pack_indexer_k_payload(
    local_k_quant: torch.Tensor,
    local_k_scale: torch.Tensor,
    *,
    slot_key: Optional[str] = None,
) -> tuple[torch.Tensor, int, int]:
    """Pack one-byte FP8 K and scale rows into one contiguous uint8 buffer."""
    if local_k_quant.dim() != 2 or local_k_scale.dim() != 2:
        raise ValueError(
            "indexer K pack expects 2D quant/scale buffers, got "
            f"{tuple(local_k_quant.shape)} and {tuple(local_k_scale.shape)}"
        )
    if local_k_quant.device != local_k_scale.device:
        raise ValueError("indexer K pack requires quant/scale on the same device")
    if local_k_quant.shape[0] != local_k_scale.shape[0]:
        raise ValueError(
            "indexer K pack row mismatch: "
            f"quant={local_k_quant.shape[0]}, scale={local_k_scale.shape[0]}"
        )
    if local_k_quant.element_size() != 1 or local_k_scale.element_size() != 1:
        raise ValueError(
            "indexer K pack requires one-byte quant/scale elements, got "
            f"{local_k_quant.dtype} and {local_k_scale.dtype}"
        )

    local_q_bytes = local_k_quant.contiguous().view(torch.uint8)
    local_s_bytes = local_k_scale.contiguous().view(torch.uint8)
    rows = int(local_q_bytes.shape[0])
    q_cols = int(local_q_bytes.shape[1])
    s_cols = int(local_s_bytes.shape[1])
    if slot_key is not None:
        packed = _acquire_asm_buffer(
            local_k_quant.device,
            slot_key,
            (rows, q_cols + s_cols),
            dtype=torch.uint8,
        )
    else:
        packed = torch.empty(
            (rows, q_cols + s_cols),
            dtype=torch.uint8,
            device=local_k_quant.device,
        )
    if rows > 0:
        if _TRITON_AVAILABLE and packed.is_cuda:
            block_m = 16
            _pack_indexer_k_kernel[(triton.cdiv(rows, block_m),)](
                local_q_bytes,
                local_s_bytes,
                packed,
                rows,
                q_cols,
                s_cols,
                q_cols + s_cols,
                BLOCK_M=block_m,
                BLOCK_K=_next_power_of_2(q_cols),
                BLOCK_S=_next_power_of_2(s_cols),
                num_warps=4,
            )
        else:
            packed[:, :q_cols].copy_(local_q_bytes)
            packed[:, q_cols:].copy_(local_s_bytes)
    return packed, q_cols, s_cols


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
    # Pack [quant | scale] into one [local_T, q_cols + s_cols] uint8 payload so
    # each assemble issues a single rank-major NCCL AllGather instead of two.
    with record_function_range("dsv4.cp.all_gather.indexer_k.packed.pack"):
        local_packed, q_cols, s_cols = _pack_indexer_k_payload(
            local_k_quant,
            local_k_scale,
            slot_key="indexer_asm.sync.local_packed",
        )
    all_gather_into = getattr(collective_torch, "all_gather_into", None)
    if (
        all_gather is _ORIG_ALL_GATHER
        and all_gather_into is not None
        and torch.distributed.is_initialized()
        and (
            local_packed.device.type != "cuda"
            or not torch.cuda.is_current_stream_capturing()
        )
    ):
        pg = collective_torch._get_group(Group.TP)
        world_size = torch.distributed.get_world_size(pg)
        gathered_packed = _acquire_asm_buffer(
            local_packed.device,
            "indexer_asm.sync.gathered_packed",
            (world_size * int(local_packed.shape[0]), q_cols + s_cols),
            dtype=torch.uint8,
        )
        with record_function_range("dsv4.cp.all_gather.indexer_k.packed.launch"):
            all_gather_into(gathered_packed, local_packed, group=Group.TP)
    else:
        with record_function_range("dsv4.cp.all_gather.indexer_k.packed.launch"):
            gathered_packed = all_gather(local_packed, group=Group.TP)
    _restore_indexer_k_packed(
        plan, gathered_packed, q_cols, s_cols, out_k_quant, out_k_scale
    )


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

    process_group = collective_torch._get_group(Group.TP)
    world_size = torch.distributed.get_world_size(process_group)
    if world_size != plan.cp_ctx.cp_size:
        raise RuntimeError(
            f"indexer K gather world_size({world_size}) != cp_size({plan.cp_ctx.cp_size})"
        )

    device = local_k_quant.device
    current_stream = torch.cuda.current_stream(device)
    active_job = get_active_csa_kv_coalesce_job()
    if (
        active_job is not None
        and not active_job.active_gather_used
        and all_gather is _ORIG_ALL_GATHER
        and (
            active_job.cmp_local_flat is not None
            or active_job.swa_local_slices is not None
        )
    ):
        local_packed, q_cols, s_cols = _pack_indexer_k_payload(
            local_k_quant,
            local_k_scale,
            slot_key="indexer_asm.coalesced.local_packed",
        )
        tensors = [local_packed]
        cmp_item_idx = -1
        swa_item_idx = -1
        if active_job.cmp_local_flat is not None:
            cmp_item_idx = len(tensors)
            tensors.append(active_job.cmp_local_flat)
        if active_job.swa_local_slices is not None:
            swa_item_idx = len(tensors)
            tensors.append(active_job.swa_local_slices)
        coalesced_handle = start_cp_all_gather_coalesced(
            tensors,
            group=Group.TP,
            async_op=True,
            stream=stream,
            allow_col2d=False,
            slot_key="cp.coalesced.g2",
            profile_name="dsv4.cp.all_gather.indexer_k.coalesced",
        )
        active_job.active_gather_used = True
        assert coalesced_handle.completion_event is not None
        return IndexerKCPGatherHandle(
            plan=plan,
            gathered_q=None,
            gathered_s=None,
            work_q=None,
            work_s=None,
            completion_event=coalesced_handle.completion_event,
            stream=stream,
            producer_stream=current_stream,
            local_k_quant=local_k_quant,
            local_k_scale=local_k_scale,
            out_k_quant=out_k_quant,
            out_k_scale=out_k_scale,
            packed_q_cols=q_cols,
            local_packed=local_packed,
            coalesced_handle=coalesced_handle,
            coalesced_job=active_job,
            cmp_item_idx=cmp_item_idx,
            swa_item_idx=swa_item_idx,
        )

    local_packed, q_cols, s_cols = _pack_indexer_k_payload(
        local_k_quant,
        local_k_scale,
        slot_key="indexer_asm.async.local_packed",
    )
    gathered_packed = _acquire_asm_buffer(
        device,
        "indexer_asm.async.gathered_packed",
        (world_size * int(local_packed.shape[0]), q_cols + s_cols),
        dtype=torch.uint8,
    )
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        with record_function_range("dsv4.cp.all_gather.indexer_k.packed.launch"):
            work_q = torch.distributed.all_gather_into_tensor(
                gathered_packed,
                local_packed,
                group=process_group,
                async_op=True,
            )
        try:
            completion_event = torch.cuda.Event()
            completion_event.record(stream)
        except Exception:
            work_q.wait()
            current_stream.wait_stream(stream)
            raise

    return IndexerKCPGatherHandle(
        plan=plan,
        gathered_q=gathered_packed,
        gathered_s=None,
        work_q=work_q,
        work_s=None,
        completion_event=completion_event,
        stream=stream,
        producer_stream=current_stream,
        local_k_quant=local_k_quant,
        local_k_scale=local_k_scale,
        out_k_quant=out_k_quant,
        out_k_scale=out_k_scale,
        packed_q_cols=q_cols,
        local_packed=local_packed,
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
    if handle.coalesced_handle is not None:
        outputs = prepare_cp_all_gather_coalesced(
            handle.coalesced_handle, stream=stream
        )
        handle.work_waited = True
        handle.gathered_q = outputs[0]
        if handle.coalesced_job is not None:
            if handle.cmp_item_idx >= 0:
                handle.coalesced_job.gathered_cmp = outputs[handle.cmp_item_idx]
            if handle.swa_item_idx >= 0:
                handle.coalesced_job.gathered_swa = outputs[handle.swa_item_idx]
            handle.coalesced_job.ready_stream = stream
        current_stream = torch.cuda.current_stream(handle.out_k_quant.device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            try:
                q_cols = (
                    handle.packed_q_cols
                    if handle.packed_q_cols is not None
                    else int(handle.out_k_quant.shape[1])
                )
                s_cols = int(handle.out_k_scale.shape[1])
                _restore_indexer_k_packed(
                    handle.plan,
                    handle.gathered_q,
                    q_cols,
                    s_cols,
                    handle.out_k_quant,
                    handle.out_k_scale,
                )
            finally:
                try:
                    handle.done_event = torch.cuda.Event()
                    handle.done_event.record(stream)
                    if handle.coalesced_job is not None:
                        handle.coalesced_job.ready_event = handle.done_event
                except Exception:
                    stream.synchronize()
                    raise
        return
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
        try:
            assert handle.gathered_q is not None
            if handle.gathered_s is None:
                q_cols = (
                    handle.packed_q_cols
                    if handle.packed_q_cols is not None
                    else int(handle.out_k_quant.shape[1])
                )
                s_cols = int(handle.out_k_scale.shape[1])
                _restore_indexer_k_packed(
                    handle.plan,
                    handle.gathered_q,
                    q_cols,
                    s_cols,
                    handle.out_k_quant,
                    handle.out_k_scale,
                )
            else:
                restore_indexer_k(
                    handle.plan,
                    handle.gathered_q,
                    handle.gathered_s,
                    handle.out_k_quant,
                    handle.out_k_scale,
                )
        finally:
            try:
                handle.done_event = torch.cuda.Event()
                handle.done_event.record(stream)
            except Exception:
                stream.synchronize()
                raise


def wait_assemble_indexer_k_async(handle: IndexerKCPGatherHandle) -> None:
    if handle.coalesced_handle is not None:
        if handle.done_event is not None:
            current_stream = torch.cuda.current_stream(handle.out_k_quant.device)
            current_stream.wait_event(handle.done_event)
            if handle.producer_stream != current_stream:
                handle.producer_stream.wait_event(handle.done_event)
            handle.stream.wait_event(handle.done_event)
            return
        wait_cp_all_gather_coalesced(handle.coalesced_handle)
        return
    current_stream = torch.cuda.current_stream(handle.out_k_quant.device)
    terminal_event = handle.done_event or handle.completion_event
    current_stream.wait_event(terminal_event)
    if handle.producer_stream != current_stream:
        handle.producer_stream.wait_event(terminal_event)
    if handle.done_event is not None:
        handle.stream.wait_event(handle.done_event)
    _wait_indexer_k_work_once(handle)


def discard_assemble_indexer_k_async(handle: IndexerKCPGatherHandle) -> None:
    if handle.coalesced_handle is not None:
        discard_cp_all_gather_coalesced(handle.coalesced_handle)
        if handle.done_event is not None:
            current_stream = torch.cuda.current_stream(handle.out_k_quant.device)
            current_stream.wait_event(handle.done_event)
            if handle.producer_stream != current_stream:
                handle.producer_stream.wait_event(handle.done_event)
            handle.stream.wait_event(handle.done_event)
        return
    wait_assemble_indexer_k_async(handle)


def _wait_indexer_k_work_once(handle: IndexerKCPGatherHandle) -> None:
    if not handle.work_waited:
        if handle.work_q is not None:
            handle.work_q.wait()
        if handle.work_s is not None:
            handle.work_s.wait()
        handle.work_waited = True


def _restore_indexer_k_packed(
    plan: IndexerCPChunkPlan,
    gathered_packed: torch.Tensor,
    q_cols: int,
    s_cols: int,
    out_k_quant: torch.Tensor,
    out_k_scale: torch.Tensor,
) -> None:
    with record_function_range("dsv4.cp.all_gather.indexer_k.restore"):
        chunk_T = int(plan.restore_indices.numel())
        if chunk_T == 0:
            return
        idx = plan.restore_indices.to(device=gathered_packed.device)
        if (
            _TRITON_AVAILABLE
            and gathered_packed.is_cuda
            and out_k_quant.dim() == 2
            and out_k_scale.dim() == 2
            and out_k_quant.is_contiguous()
            and out_k_scale.is_contiguous()
            and gathered_packed.is_contiguous()
            and int(out_k_quant.shape[1]) == q_cols
            and int(out_k_scale.shape[1]) == s_cols
        ):
            block_m = 16
            _restore_packed_indexer_k_kernel[(triton.cdiv(chunk_T, block_m),)](
                gathered_packed,
                out_k_quant.view(torch.uint8),
                out_k_scale.view(torch.uint8),
                idx,
                q_cols + s_cols,
                q_cols,
                s_cols,
                chunk_T,
                BLOCK_M=block_m,
                BLOCK_K=_next_power_of_2(q_cols),
                BLOCK_S=_next_power_of_2(s_cols),
                num_warps=4,
            )
            return
        out_k_quant.view(torch.uint8).copy_(gathered_packed[idx, :q_cols])
        out_k_scale.view(torch.uint8).copy_(
            gathered_packed[idx, q_cols : q_cols + s_cols]
        )


def restore_indexer_k(
    plan: IndexerCPChunkPlan,
    gathered_q: torch.Tensor,
    gathered_s: torch.Tensor,
    out_k_quant: torch.Tensor,
    out_k_scale: torch.Tensor,
) -> None:
    with record_function_range("dsv4.cp.all_gather.indexer_k.restore"):
        chunk_T = int(plan.restore_indices.numel())
        if chunk_T == 0:
            return
        idx = plan.restore_indices.to(device=gathered_q.device)
        if (
            _TRITON_AVAILABLE
            and gathered_q.is_cuda
            and out_k_quant.dim() == 2
            and out_k_scale.dim() == 2
            and gathered_q.is_contiguous()
            and gathered_s.is_contiguous()
            and out_k_quant.is_contiguous()
            and out_k_scale.is_contiguous()
        ):
            q_cols = int(out_k_quant.shape[1])
            s_cols = int(out_k_scale.shape[1])
            block_m = 16
            _restore_separate_indexer_k_kernel[(triton.cdiv(chunk_T, block_m),)](
                gathered_q.view(torch.uint8),
                gathered_s.view(torch.uint8),
                out_k_quant.view(torch.uint8),
                out_k_scale.view(torch.uint8),
                idx,
                q_cols,
                s_cols,
                chunk_T,
                BLOCK_M=block_m,
                BLOCK_K=_next_power_of_2(q_cols),
                BLOCK_S=_next_power_of_2(s_cols),
                num_warps=4,
            )
            return
        out_k_quant.copy_(gathered_q[idx])
        out_k_scale.copy_(gathered_s[idx])


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
