"""Typed packet assembly for V4.1 compact CP; persistent installation is separate."""

from dataclasses import dataclass

import torch

from ._v41_compact_cp_plan import HEADER_BYTES, RECORD_BYTES, TAIL_RECORD_BYTES


@dataclass(frozen=True)
class PacketMetadata:
    header: torch.Tensor
    positions: torch.Tensor
    tail_identity: torch.Tensor | None


@dataclass(frozen=True)
class ValidationStack:
    """Pre-stacked per-sender expectations for one producer's four plans.

    Stacking once at admission replaces the per-sender compare/reduce kernel
    chain in unpack with one batched comparison per field.
    """

    header: torch.Tensor  # (4, 16) int64
    positions: torch.Tensor  # (4, rows) int64
    tail_identity: torch.Tensor | None  # (4, 2, 2) int64, ratio 2 only
    tail_state_exempt: torch.Tensor | None  # (4,) bool, True for the tail owner


def prepare_metadata(plan, device):
    tail = None
    if plan.ratio == 2:
        tail = torch.tensor(
            (
                [[plan.start + 4094, 1], [plan.start + 4095, 1]]
                if plan.tail_count
                else [[-1, 0], [-1, 0]]
            ),
            dtype=torch.int64,
            device=device,
        )
    return PacketMetadata(
        torch.tensor(plan.header, dtype=torch.int64, device=device),
        torch.tensor(plan.boundary_positions, dtype=torch.int64, device=device),
        tail,
    )


def prepare_metadata_batch(plans, device):
    """Create forward metadata with one host-to-device copy for all producers."""
    values, offsets = [], []
    for plan in plans:
        begin = len(values)
        values.extend(plan.header)
        values.extend(plan.boundary_positions)
        if plan.ratio == 2:
            values.extend(
                (plan.start + 4094, 1, plan.start + 4095, 1)
                if plan.tail_count
                else (-1, 0, -1, 0)
            )
        offsets.append(begin)
    storage = torch.tensor(values, dtype=torch.int64, device=device)
    result = []
    for plan, begin in zip(plans, offsets):
        positions = begin + 16
        end = positions + plan.rows
        result.append(
            PacketMetadata(
                storage[begin:positions],
                storage[positions:end],
                storage[end : end + 4].view(2, 2) if plan.ratio == 2 else None,
            )
        )
    return tuple(result)


def prepare_validation_stacks(plan_groups, device):
    """Stack each producer's four sender expectations with one H2D copy.

    ``plan_groups`` is one tuple of four rank-ordered plans per producer, in
    the same order the runtime admits producers. The returned header/position/
    tail tensors are views into a single storage, mirroring
    prepare_metadata_batch lifetimes.
    """
    values, slices = [], []
    for plans in plan_groups:
        if len(plans) != 4 or any(p.rows != plans[0].rows for p in plans):
            raise ValueError("A validation stack requires four same-row plans")
        rows = plans[0].rows
        header_begin = len(values)
        for plan in plans:
            values.extend(plan.header)
        positions_begin = len(values)
        for plan in plans:
            values.extend(plan.boundary_positions)
        tail_begin = None
        if plans[0].ratio == 2:
            tail_begin = len(values)
            for plan in plans:
                values.extend(
                    (plan.start + 4094, 1, plan.start + 4095, 1)
                    if plan.tail_count
                    else (-1, 0, -1, 0)
                )
        slices.append((header_begin, positions_begin, tail_begin, rows))
    storage = torch.tensor(values, dtype=torch.int64, device=device)
    result = []
    for plans, (header_begin, positions_begin, tail_begin, rows) in zip(
        plan_groups, slices
    ):
        if tail_begin is not None:
            tail_identity = storage[tail_begin : tail_begin + 16].view(4, 2, 2)
            tail_state_exempt = torch.tensor(
                [bool(plan.tail_count) for plan in plans],
                dtype=torch.bool,
                device=device,
            )
        else:
            tail_identity = tail_state_exempt = None
        result.append(
            ValidationStack(
                storage[header_begin : header_begin + 64].view(4, 16),
                storage[positions_begin : positions_begin + 4 * rows].view(4, rows),
                tail_identity,
                tail_state_exempt,
            )
        )
    return tuple(result)


def pack(
    plan, metadata, main, index_payload, index_scales, state_tail=None, *, out=None
):
    """Construct the fixed wire ABI without changing quantization bytes."""
    device = main.device
    if (
        main.shape != (plan.rows, 288)
        or main.dtype != torch.uint8
        or index_payload.shape != (plan.rows, 64)
        or index_payload.dtype != torch.int8
        or index_scales.shape != (plan.rows,)
        or index_scales.dtype != torch.int32
        or any(
            t.device != device
            for t in (index_payload, index_scales, metadata.header, metadata.positions)
        )
        or metadata.header.shape != (16,)
        or metadata.header.dtype != torch.int64
        or metadata.positions.shape != (plan.rows,)
        or metadata.positions.dtype != torch.int64
    ):
        raise ValueError("Invalid compact CP packet operands")
    if plan.ratio == 2 and (
        metadata.tail_identity is None
        or metadata.tail_identity.shape != (2, 2)
        or metadata.tail_identity.dtype != torch.int64
        or metadata.tail_identity.device != device
    ):
        raise ValueError("Invalid compact CP tail identity")
    if plan.tail_count:
        if (
            state_tail is None
            or state_tail.shape != (2, 1024)
            or state_tail.dtype != torch.float32
            or state_tail.device != device
        ):
            raise ValueError("Compact CP owner must supply two FP32 state rows")
    elif state_tail is not None:
        raise ValueError("Only the CSA tail owner may supply state rows")
    if out is None:
        out = torch.empty(plan.send_bytes, dtype=torch.uint8, device=device)
    if (
        out.shape != (plan.send_bytes,)
        or out.dtype != torch.uint8
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError("Invalid compact CP packet storage")
    out[:HEADER_BYTES].copy_(metadata.header.contiguous().view(torch.uint8))
    end = HEADER_BYTES + plan.rows * RECORD_BYTES
    records = out[HEADER_BYTES:end].view(plan.rows, RECORD_BYTES)
    records[:, :8].copy_(
        metadata.positions.contiguous().view(torch.uint8).view(plan.rows, 8)
    )
    records[:, 8:296].copy_(main)
    records[:, 296:360].copy_(index_payload.contiguous().view(torch.uint8))
    records[:, 360:364].copy_(
        index_scales.contiguous().view(torch.uint8).view(plan.rows, 4)
    )
    if plan.ratio == 2:
        tail = out[end:].view(2, TAIL_RECORD_BYTES)
        tail[:, :16].copy_(metadata.tail_identity.contiguous().view(torch.uint8))
        if plan.tail_count:
            tail[:, 16:].copy_(state_tail.contiguous().view(torch.uint8))
        else:
            tail[:, 16:].zero_()
    return out


def _stack_from_metadata(plans, metadata, device):
    """Build a ValidationStack from per-sender metadata (compatibility path)."""
    ratio2 = plans[0].ratio == 2
    return ValidationStack(
        torch.stack([expected.header for expected in metadata]),
        torch.stack([expected.positions for expected in metadata]),
        (
            torch.stack([expected.tail_identity for expected in metadata])
            if ratio2
            else None
        ),
        (
            torch.tensor(
                [bool(plan.tail_count) for plan in plans],
                dtype=torch.bool,
                device=device,
            )
            if ratio2
            else None
        ),
    )


def unpack(plans, metadata, received, validation=None):
    """Validate every sender before returning any rows for persistent scatter.

    Validation is batched across all four senders: one comparison per field
    plus a single trailing assertion, instead of a per-sender compare/reduce
    chain. ``validation`` is the admission-time stack from
    prepare_validation_stacks; when omitted it is rebuilt from ``metadata``
    (compatibility path for direct callers).

    Device assertions execute on the caller's stream before subsequent cache
    writes. A failed CUDA assertion poisons the worker; callers must not retry.
    """
    if len(plans) != 4 or len(metadata) != 4:
        raise ValueError("Compact CP requires exactly four sender descriptors")
    size = plans[0].send_bytes
    if (
        received.shape != (4, size)
        or received.dtype != torch.uint8
        or not received.is_contiguous()
    ):
        raise ValueError("Invalid compact CP receive storage")
    if any(
        p.rank != rank or p.send_bytes != size or p.header[:12] != plans[0].header[:12]
        for rank, p in enumerate(plans)
    ):
        raise ValueError("Compact CP sender schedule disagrees")
    rows = plans[0].rows
    end = HEADER_BYTES + rows * RECORD_BYTES
    records = received[:, HEADER_BYTES:end].reshape(4, rows, RECORD_BYTES)
    if validation is None:
        validation = _stack_from_metadata(plans, metadata, received.device)
    if (
        not isinstance(validation, ValidationStack)
        or validation.header.shape != (4, 16)
        or validation.header.dtype != torch.int64
        or validation.positions.shape != (4, rows)
        or validation.positions.dtype != torch.int64
        or validation.header.device != received.device
        or validation.positions.device != received.device
    ):
        raise ValueError("Invalid compact CP validation stack")
    valid = (received[:, :HEADER_BYTES].view(torch.int64) == validation.header).all()
    # The record stride (364 B) is not int64-aligned, so compare positions in
    # byte space: the contiguous stacked expectation views to (4, rows, 8)
    # natively, matching the little-endian bytes pack() wrote, with no copy.
    position_bytes = validation.positions.view(torch.uint8).view(4, rows, 8)
    valid &= (records[:, :, :8] == position_bytes).all()
    if plans[0].ratio == 2:
        if (
            validation.tail_identity is None
            or validation.tail_state_exempt is None
            or validation.tail_identity.shape != (4, 2, 2)
            or validation.tail_identity.dtype != torch.int64
            or validation.tail_state_exempt.shape != (4,)
            or validation.tail_state_exempt.dtype != torch.bool
        ):
            raise ValueError("Invalid compact CP tail validation stack")
        tails = received[:, end:].view(4, 2, TAIL_RECORD_BYTES)
        valid &= (tails[:, :, :16].view(torch.int64) == validation.tail_identity).all()
        state_zero = (tails[:, :, 16:] == 0).all(dim=(1, 2))
        valid &= (state_zero | validation.tail_state_exempt).all()
    if received.is_cuda:
        torch._assert_async(valid, "Invalid compact CP identity or row order")
    elif not bool(valid):
        raise ValueError("Invalid compact CP identity or row order")
    main = records[:, :, 8:296].reshape(4 * rows, 288).contiguous()
    index = records[:, :, 296:364].reshape(4 * rows, 68).contiguous()
    state = None
    if plans[0].ratio == 2:
        state = (
            received[0, end:]
            .view(2, TAIL_RECORD_BYTES)[:, 16:]
            .contiguous()
            .view(torch.float32)
        )
    return main, index, state
