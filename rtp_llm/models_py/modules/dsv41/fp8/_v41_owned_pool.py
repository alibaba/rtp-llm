"""Owned-row joint FP4 transport, preserving the codec's byte/float contract.

All collective sizes come from common host lengths. Scratch is allocated once
and reused in stream order, including when an outer caller uses a side stream.
Only the request outputs survive the next group; no packet aliases escape.
The 256 MiB bound covers send/receive, slots and inverse row metadata. All BF16
keys and index outputs are separately reported required model outputs; their
lifetime is independent of transport grouping and they never alias scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_fp4_triton as codec

_BYTES = 356
_INDEX_BYTES = 68
_INVERSE_BYTES = 16
_LIMIT = 256 * 1024**2


@dataclass(frozen=True)
class OwnedGroup:
    first: int
    stop: int
    rows: int
    local_rows: int


@dataclass(frozen=True)
class OwnedPlan:
    groups: tuple[OwnedGroup, ...]
    max_local_rows: int
    max_output_rows: int
    temporary_bytes: int
    index_output_bytes: int


def owned_count(count, cp_size, owner_rows, rank):
    cycles, tail = divmod(count, cp_size * owner_rows)
    return cycles * owner_rows + min(owner_rows, max(0, tail - rank * owner_rows))


def plan_groups(counts, *, cp_size, owner_rows, limit=_LIMIT):
    """Bound persistent transport scratch independently of final outputs.

    The local/output caps are independent, so taking their maxima across groups
    also obeys the bound. Padding is only at the end of each rank's packet;
    each request's 256-row scorer padding exists only in the final outputs.
    """
    if (
        cp_size not in (1, 2, 4)
        or type(owner_rows) is not int
        or owner_rows <= 0
        or not 1 <= len(counts) <= 128
        or any(type(count) is not int or count < 0 for count in counts)
    ):
        return None
    # Validate before hashing: bool/float counts compare equal to integers.
    build = (
        _plan_groups_cached
        if type(cp_size) is int and type(limit) is int
        else _plan_groups_cached.__wrapped__
    )
    return build(tuple(counts), cp_size, owner_rows, limit)


@lru_cache(maxsize=128, typed=True)
def _plan_groups_cached(counts, cp_size, owner_rows, limit):
    """Retain immutable host plans only; no request tensors or device state."""
    scratch_per_local = _BYTES * (cp_size + 1 if cp_size > 1 else 1) + 16
    # Independent caps also bound the simultaneous maxima of different groups.
    local_cap = limit // (scratch_per_local + _INVERSE_BYTES * cp_size)
    output_cap = local_cap * cp_size
    groups, first, rows = [], 0, 0
    owned = [0] * cp_size
    for request, count in enumerate(counts):
        padded = triton.cdiv(count, 256) * 256
        contribution = [
            owned_count(count, cp_size, owner_rows, rank) for rank in range(cp_size)
        ]
        if max(contribution) > local_cap or padded > output_cap:
            return None
        if (
            rows + padded > output_cap
            or max(a + b for a, b in zip(owned, contribution)) > local_cap
        ):
            groups.append(OwnedGroup(first, request, rows, max(owned)))
            first, rows, owned = request, 0, [0] * cp_size
        rows += padded
        owned = [a + b for a, b in zip(owned, contribution)]
    groups.append(OwnedGroup(first, len(counts), rows, max(owned)))
    local_max = max(group.local_rows for group in groups)
    output_max = max(group.rows for group in groups)
    return OwnedPlan(
        tuple(groups),
        local_max,
        output_max,
        scratch_per_local * local_max + _INVERSE_BYTES * output_max,
        _INDEX_BYTES * sum(group.rows for group in groups),
    )


@triton.jit
def _owned(counts, U: tl.constexpr, CP: tl.constexpr, rank):
    return counts // (U * CP) * U + tl.minimum(
        U, tl.maximum(0, counts % (U * CP) - rank * U)
    )


@triton.jit(
    do_not_specialize=[
        "ROWS",
        "CANONICAL",
        "FIRST",
        "STOP",
        "ES",
        "MC",
        "MS",
        "IC",
        "IS",
    ]
)
def _owned_pool_metadata(
    ends,
    mt,
    it,
    slots,
    inverse,
    offsets,
    ROWS,
    CANONICAL,
    FIRST,
    STOP,
    ES,
    MC,
    MS,
    IC,
    IS,
    MTPB: tl.constexpr,
    ITPB: tl.constexpr,
    MEB: tl.constexpr,
    IEB: tl.constexpr,
    OWNER: tl.constexpr,
    RATIO: tl.constexpr,
    RANK: tl.constexpr,
    CP_SIZE: tl.constexpr,
    DIRECT: tl.constexpr,
):
    from_slot: tl.constexpr = OWNER // RATIO
    ids = tl.arange(0, 128)
    counts = tl.load(ends + (FIRST + ids) * ES, FIRST + ids < STOP, other=0) // RATIO
    base = tl.program_id(0) * 128
    row = base + ids
    if base < ROWS:
        sizes = _owned(counts, from_slot, CP_SIZE, RANK)
        prefix = tl.cumsum(sizes)
        lower = tl.full((128,), 0, tl.int32)
        upper = tl.full((128,), STOP - FIRST, tl.int32)
        for _ in tl.static_range(8):
            active = lower < upper
            middle = (lower + upper) // 2
            cutoff = tl.gather(prefix, tl.minimum(middle, 127), 0)
            lower = tl.where(active & (row >= cutoff), middle + 1, lower)
            upper = tl.where(active & (row < cutoff), middle, upper)
        request = tl.minimum(lower, 127)
        local = row - tl.gather(prefix - sizes, request, 0)
        valid = (row < ROWS) & (FIRST + lower < STOP)
        logical = (local // from_slot * CP_SIZE + RANK) * from_slot + local % from_slot
        pos = (logical + 1) * RATIO - 1
        req = (FIRST + request).to(tl.int64)
        # Ownership is logical and independent of main/index physical pages.
        owner_block = pos // OWNER
        mcol = owner_block // CP_SIZE * (OWNER // MTPB) + pos % OWNER // MTPB
        icol = owner_block // CP_SIZE * (OWNER // ITPB) + pos % OWNER // ITPB
        mb = tl.load(mt + req * MS + mcol, valid & (mcol < MC), other=0).to(tl.int64)
        ib = tl.load(it + req * IS + icol, valid & (icol < IC), other=0).to(tl.int64)
        mi = tl.where(valid & (mb > 0), mb * MEB + pos % MTPB // RATIO, -1)
        ii = tl.where(valid & (ib > 0), ib * IEB + pos % ITPB // RATIO, -1)
        tl.store(slots + row, mi, row < ROWS)
        tl.store(slots + ROWS + row, ii, row < ROWS)

    if base < CANONICAL:
        # Each request has 256-row scorer padding, so this 128-row tile belongs
        # to one request for every ragged geometry, including empty requests.
        padded = tl.cdiv(counts, 256) * 256
        inverse_prefix = tl.cumsum(padded)
        inverse_request = tl.sum((inverse_prefix <= base).to(tl.int32), 0)
        start = tl.sum(tl.where(ids < inverse_request, padded, 0), 0)
        count = tl.sum(tl.where(ids == inverse_request, counts, 0), 0)
        inverse_local = row - start
        inverse_owner = inverse_local // from_slot % CP_SIZE
        inverse_source = (
            inverse_local // (from_slot * CP_SIZE) * from_slot
            + inverse_local % from_slot
        )
        for rank in tl.static_range(CP_SIZE):
            owned_sizes = _owned(counts, from_slot, CP_SIZE, rank)
            before = tl.sum(tl.where(ids < inverse_request, owned_sizes, 0), 0)
            inverse_source += tl.where(inverse_owner == rank, before, 0)
        inverse_valid = (row < CANONICAL) & (inverse_local < count)
        encoded = tl.where(
            inverse_valid, (inverse_source.to(tl.int64) << 2) | inverse_owner, -1
        )
        destination = (
            tl.load(
                offsets + FIRST + inverse_request,
                FIRST + inverse_request < STOP,
                other=0,
            )
            + inverse_local
            if DIRECT
            else row
        )
        tl.store(inverse + row, encoded, row < CANONICAL)
        tl.store(inverse + CANONICAL + row, destination, row < CANONICAL)


@triton.jit(do_not_specialize=["ROWS", "LOCAL"])
def _owned_pool_restore(
    packet,
    inverse,
    out,
    oq,
    os,
    ROWS,
    LOCAL,
    DIRECT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    encoded = tl.load(inverse + row)
    valid = encoded >= 0
    address = tl.maximum(encoded, 0)
    rank = address & 3
    source = address >> 2
    base = packet + rank * LOCAL * 356
    columns = tl.arange(0, 64)
    q = tl.load(base + source * 64 + columns, valid, other=0)
    s = tl.load(
        (base + LOCAL * 64).to(tl.pointer_type(tl.int32)) + source, valid, other=0
    )
    tl.store(oq + row * 64 + columns, q.to(tl.int8, bitcast=True))
    tl.store(os + row, s)
    if valid:
        destination = tl.load(inverse + ROWS + row) if DIRECT else row
        # Keep all FP4 decode arithmetic in its original codec. The base shift
        # selects a gathered source row and the caller's final destination.
        raw = base + LOCAL * 68 + (source - row) * 288
        stride = tl.full((), 288, tl.int64)
        codec._fp4_global_dequant_kernel(
            raw,
            raw,
            out + (destination - row) * 512,
            ROWS,
            512,
            16,
            288,
            1,
            stride,
            ROWS,
            tl.bfloat16,
            True,
        )
    elif not DIRECT:
        channels = tl.arange(0, 512)
        tl.store(out + row * 512 + channels, 0)


def _metadata_args(
    ends,
    mt,
    it,
    slots,
    inverse,
    offsets,
    rows,
    canonical_rows,
    first,
    stop,
    layout,
    direct,
):
    ratio, owner, mtpb, itpb, rank, _, _, meb, ieb, cp = layout
    return (
        ends,
        mt,
        it,
        slots,
        inverse,
        offsets,
        rows,
        canonical_rows,
        first,
        stop,
        ends.stride(0),
        mt.shape[1],
        mt.stride(0),
        it.shape[1],
        it.stride(0),
        mtpb,
        itpb,
        meb,
        ieb,
        owner,
        ratio,
        rank,
        cp,
        direct,
    )


def warmup(layout, device, gather_fn):
    """Compile all pointers at their minimum supported alignment; retain handles only."""
    _, owner, _, _, _, mdtype, idtype, meb, ieb, cp = layout
    if cp not in (1, 2, 4):
        return None
    ends = torch.full((33,), 3 * owner + 1, dtype=torch.int64, device=device)[1:]
    mt = torch.zeros(33, dtype=mdtype, device=device)[1:].view(32, 1)
    it = torch.zeros(33, dtype=idtype, device=device)[1:].view(32, 1)
    slots = torch.empty(2 * 256 + 1, dtype=torch.int64, device=device)[1:]
    inverse = torch.empty(2 * 256 + 1, dtype=torch.int64, device=device)[1:]
    offsets = torch.zeros(33, dtype=torch.int64, device=device)[1:]
    main = torch.zeros(meb * 288 + 1, dtype=torch.uint8, device=device)[1:].view(
        1, meb, 288
    )
    index = torch.zeros(ieb * 68 + 4, dtype=torch.uint8, device=device)[4:].view(
        1, ieb, 68
    )
    q = torch.empty(256 * 64 + 1, dtype=torch.int8, device=device)[1:].view(256, 64)
    s = torch.empty(257, dtype=torch.int32, device=device)[1:]
    raw = torch.empty(256 * 288 + 1, dtype=torch.uint8, device=device)[1:].view(
        256, 288
    )
    meta_args = _metadata_args(
        ends, mt, it, slots, inverse, offsets, 256, 256, 0, 32, layout, False
    )
    gather_args = (
        slots,
        main,
        index,
        q,
        s,
        raw,
        256,
        main.stride(0),
        index.stride(0),
        meb,
        ieb,
    )
    compiled = [
        _owned_pool_metadata.warmup(*meta_args, grid=(1,), num_warps=4),
        gather_fn.warmup(*gather_args, grid=(1,), num_warps=4),
    ]
    compiled[0][(2, 1, 1)](*meta_args)
    compiled[1][(256, 1, 1)](*gather_args)
    # Compile/launch a tiny independent, fully initialized decode fixture.
    packet = torch.zeros(cp * 256 * _BYTES, dtype=torch.uint8, device=device)
    out = torch.empty(256 * 512 + 1, dtype=torch.bfloat16, device=device)[1:].view(
        256, 512
    )
    direct_meta = None
    for direct in (False, True):
        if direct:
            args = _metadata_args(
                ends, mt, it, slots, inverse, offsets, 256, 256, 0, 32, layout, True
            )
            direct_meta = _owned_pool_metadata.warmup(*args, grid=(1,), num_warps=4)
            direct_meta[(2, 1, 1)](*args)
        args = (
            packet,
            inverse,
            out,
            q,
            s,
            256,
            256,
            direct,
        )
        kernel = _owned_pool_restore.warmup(*args, grid=(1,), num_warps=4)
        kernel[(256, 1, 1)](*args)
        compiled.append(kernel)
    compiled.append(direct_meta)
    return tuple(compiled)


def _collect(gathered, send, cp_size, cp_rank):
    if cp_size == 1:
        return
    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.distributed.collective_torch import Group

    group = collective_torch._get_group(Group.TP)
    if (
        torch.distributed.get_world_size(group) != cp_size
        or torch.distributed.get_rank(group) != cp_rank
    ):
        raise RuntimeError("Owned pool CP size/rank differs from its process group")
    work = torch.distributed.all_gather_into_tensor(
        gathered, send, group=group, async_op=True
    )
    # NCCL's internal stream must join the current stream before scratch reuse
    # or the outer caller's completion event. This is a GPU dependency.
    work.wait()


def _validate_local(main, index, ends, mt, it, batch, layout):
    _, _, _, _, _, mdtype, idtype, meb, ieb, _ = layout
    device = ends.device
    if ends.dtype != torch.int64 or ends.shape != (batch,) or ends.stride(0) <= 0:
        raise ValueError("Owned pool requires the prepared sequence-end tensor")
    for table, dtype in ((mt, mdtype), (it, idtype)):
        if (
            table.device != device
            or table.dtype != dtype
            or table.ndim != 2
            or table.shape[0] != batch
            or table.stride(1) != 1
        ):
            raise ValueError("Owned pool block table has incompatible storage")
    if index.data_ptr() % 4 or index.stride(0) % 4:
        raise ValueError("Owned pool index scale words must be 4-byte aligned")
    for pool, entries, width in ((main, meb, 288), (index, ieb, 68)):
        if (
            pool.device != device
            or pool.dtype != torch.uint8
            or pool.ndim != 3
            or pool.shape[1:] != (entries, width)
            or pool.stride(1) != width
            or pool.stride(2) != 1
            or pool.stride(0) < entries * width
        ):
            raise ValueError("Owned pool cache has incompatible native storage")


def try_gather(
    main,
    index,
    ends,
    seq_ends,
    mt,
    it,
    layout,
    kernels,
    *,
    workspace=None,
    swa=None,
    output_stream=None,
):
    from . import _v41_joint_pool as joint
    from ._v41_prefill_indexer import PrefillIndexerKeys

    ratio, owner, _, _, rank, _, _, _, _, cp = layout
    counts = tuple(end // ratio for end in ends)
    plan = plan_groups(counts, cp_size=cp, owner_rows=owner // ratio)
    if plan is None:
        return None
    if kernels is None:
        raise RuntimeError("Owned pool kernels were not prepared by startup warmup")
    _validate_local(main, index, seq_ends, mt, it, len(ends), layout)
    if workspace is None and swa is not None:
        from ._v41_prefill_kv_workspace import prepare_global_workspace

        workspace = prepare_global_workspace(swa, counts, output_stream=output_stream)
    if workspace is not None:
        joint._validate_workspace(workspace, counts, seq_ends.device)
        global_views = joint._split_rows(
            workspace.output,
            joint._row_partitions(
                counts, workspace.global_starts, workspace.output.shape[0]
            ),
        )
    device = seq_ends.device
    send_buffer = torch.empty(
        plan.max_local_rows * _BYTES, dtype=torch.uint8, device=device
    )
    receive_buffer = (
        torch.empty(cp * plan.max_local_rows * _BYTES, dtype=torch.uint8, device=device)
        if cp > 1
        else send_buffer
    )
    slots = torch.empty(2 * plan.max_local_rows, dtype=torch.int64, device=device)
    inverse = torch.empty(2 * plan.max_output_rows, dtype=torch.int64, device=device)
    result = []
    for group in plan.groups:
        first, stop, rows, local = group.first, group.stop, group.rows, group.local_rows
        send = send_buffer[: local * _BYTES]
        receive = receive_buffer[: cp * local * _BYTES]
        if local:
            q, s, raw = joint._views(send, local)
            with record_function_range("dsv41.prefill.pool.owned.pack"):
                kernels[4 if workspace is not None else 0][
                    (triton.cdiv(max(local, rows), 128), 1, 1)
                ](
                    *_metadata_args(
                        seq_ends,
                        mt,
                        it,
                        slots,
                        inverse,
                        seq_ends if workspace is None else workspace.device_offsets,
                        local,
                        rows,
                        first,
                        stop,
                        layout,
                        workspace is not None,
                    )
                )
                kernels[1][(local, 1, 1)](
                    slots,
                    main,
                    index,
                    q,
                    s,
                    raw,
                    local,
                    main.stride(0),
                    index.stride(0),
                    main.shape[1],
                    index.shape[1],
                )
            with record_function_range("dsv41.prefill.pool.owned.all_gather"):
                _collect(receive, send, cp, rank)
        out = (
            torch.empty((rows, 512), dtype=torch.bfloat16, device=device)
            if workspace is None
            else workspace.output
        )
        oq = torch.empty((rows, 64), dtype=torch.int8, device=device)
        os = torch.empty(rows, dtype=torch.int32, device=device)
        if rows:
            with record_function_range("dsv41.prefill.pool.owned.restore"):
                kernels[3 if workspace is not None else 2][(rows, 1, 1)](
                    receive,
                    inverse,
                    out,
                    oq,
                    os,
                    rows,
                    local,
                    workspace is not None,
                )
        starts, cursor = [], 0
        for count in counts[first:stop]:
            starts.append(cursor)
            cursor += triton.cdiv(count, 256) * 256
        partitions = joint._row_partitions(counts[first:stop], starts, rows)
        gs = (
            joint._split_rows(out, partitions)
            if workspace is None
            else global_views[first:stop]
        )
        result.extend(
            (g, PrefillIndexerKeys(q, s))
            for g, q, s in zip(
                gs, joint._split_rows(oq, partitions), joint._split_rows(os, partitions)
            )
        )
    return result if workspace is None else workspace.publish(result)
