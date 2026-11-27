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

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    build_kv_allgather_restore_indices,
    cp_actual_owned_kv_len,
    cp_actual_owned_kv_lens,
    cp_padded_local_kv_len,
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
    per_req = per_req_total_kv_lens.to(device=device, dtype=torch.int64).contiguous()
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
        actual_lens = torch.full((1,), total_actual, dtype=torch.int64, device=device)
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
            int(v) for v in torch.stack([local_lens.sum(), actual_lens.sum()]).tolist()
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


if _TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["n_rows", "q_stride", "s_stride", "packed_stride"])
    def _pack_indexer_k_kernel(
        local_q_ptr,
        local_s_ptr,
        packed_ptr,
        n_rows,
        q_stride,
        s_stride,
        packed_stride,
        BLOCK_K: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row_id = tl.program_id(0).to(tl.int64)
        if row_id >= n_rows:
            return
        k_cols = tl.arange(0, BLOCK_K)
        k_mask = k_cols < q_stride
        k_vals = tl.load(local_q_ptr + row_id * q_stride + k_cols, mask=k_mask)
        tl.store(packed_ptr + row_id * packed_stride + k_cols, k_vals, mask=k_mask)

        s_cols = tl.arange(0, BLOCK_S)
        s_mask = s_cols < s_stride
        s_vals = tl.load(local_s_ptr + row_id * s_stride + s_cols, mask=s_mask)
        tl.store(
            packed_ptr + row_id * packed_stride + q_stride + s_cols,
            s_vals,
            mask=s_mask,
        )

    @triton.jit(
        do_not_specialize=["n_rows", "packed_stride", "k_stride", "s_stride"]
    )
    def _restore_packed_indexer_k_kernel(
        gathered_packed_ptr,
        out_k_ptr,
        out_s_ptr,
        indices_ptr,
        packed_stride,
        k_stride,
        s_stride,
        n_rows,
        BLOCK_K: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row_id = tl.program_id(0).to(tl.int64)
        if row_id >= n_rows:
            return
        src_row = tl.load(indices_ptr + row_id).to(tl.int64)
        k_cols = tl.arange(0, BLOCK_K)
        k_mask = k_cols < k_stride
        k_vals = tl.load(
            gathered_packed_ptr + src_row * packed_stride + k_cols, mask=k_mask
        )
        tl.store(out_k_ptr + row_id * k_stride + k_cols, k_vals, mask=k_mask)

        s_cols = tl.arange(0, BLOCK_S)
        s_mask = s_cols < s_stride
        s_vals = tl.load(
            gathered_packed_ptr + src_row * packed_stride + k_stride + s_cols,
            mask=s_mask,
        )
        tl.store(out_s_ptr + row_id * s_stride + s_cols, s_vals, mask=s_mask)

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
        BLOCK_K: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row_id = tl.program_id(0).to(tl.int64)
        if row_id >= n_rows:
            return
        src_row = tl.load(indices_ptr + row_id).to(tl.int64)
        k_cols = tl.arange(0, BLOCK_K)
        k_mask = k_cols < k_stride
        k_vals = tl.load(gathered_k_ptr + src_row * k_stride + k_cols, mask=k_mask)
        tl.store(out_k_ptr + row_id * k_stride + k_cols, k_vals, mask=k_mask)

        s_cols = tl.arange(0, BLOCK_S)
        s_mask = s_cols < s_stride
        s_vals = tl.load(gathered_s_ptr + src_row * s_stride + s_cols, mask=s_mask)
        tl.store(out_s_ptr + row_id * s_stride + s_cols, s_vals, mask=s_mask)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pack_indexer_k_payload(
    local_k_quant: torch.Tensor,
    local_k_scale: torch.Tensor,
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
    packed = torch.empty(
        (rows, q_cols + s_cols),
        dtype=torch.uint8,
        device=local_k_quant.device,
    )
    if rows > 0:
        if _TRITON_AVAILABLE and packed.is_cuda:
            _pack_indexer_k_kernel[(rows,)](
                local_q_bytes,
                local_s_bytes,
                packed,
                rows,
                q_cols,
                s_cols,
                q_cols + s_cols,
                BLOCK_K=_next_power_of_2(q_cols),
                BLOCK_S=_next_power_of_2(s_cols),
                num_warps=2 if q_cols <= 128 else 4,
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
            local_k_quant, local_k_scale
        )
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
    with torch.cuda.stream(stream):
        local_packed, q_cols, s_cols = _pack_indexer_k_payload(
            local_k_quant, local_k_scale
        )
        gathered_packed = torch.empty(
            (world_size * int(local_packed.shape[0]), q_cols + s_cols),
            dtype=torch.uint8,
            device=device,
        )
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
    current_stream = torch.cuda.current_stream(handle.out_k_quant.device)
    terminal_event = handle.done_event or handle.completion_event
    current_stream.wait_event(terminal_event)
    if handle.producer_stream != current_stream:
        handle.producer_stream.wait_event(terminal_event)
    if handle.done_event is not None:
        handle.stream.wait_event(handle.done_event)
    _wait_indexer_k_work_once(handle)


def discard_assemble_indexer_k_async(handle: IndexerKCPGatherHandle) -> None:
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
            _restore_packed_indexer_k_kernel[(chunk_T,)](
                gathered_packed,
                out_k_quant.view(torch.uint8),
                out_k_scale.view(torch.uint8),
                idx,
                q_cols + s_cols,
                q_cols,
                s_cols,
                chunk_T,
                BLOCK_K=_next_power_of_2(q_cols),
                BLOCK_S=_next_power_of_2(s_cols),
                num_warps=2 if q_cols <= 128 else 4,
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
            _restore_separate_indexer_k_kernel[(chunk_T,)](
                gathered_q.view(torch.uint8),
                gathered_s.view(torch.uint8),
                out_k_quant.view(torch.uint8),
                out_k_scale.view(torch.uint8),
                idx,
                q_cols,
                s_cols,
                chunk_T,
                BLOCK_K=_next_power_of_2(q_cols),
                BLOCK_S=_next_power_of_2(s_cols),
                num_warps=2 if q_cols <= 128 else 4,
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
