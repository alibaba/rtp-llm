"""Typed packet assembly for V4.1 compact CP; persistent installation is separate."""

from dataclasses import dataclass

import torch

from ._v41_compact_cp_plan import HEADER_BYTES, RECORD_BYTES, TAIL_RECORD_BYTES


@dataclass(frozen=True)
class PacketMetadata:
    header: torch.Tensor
    positions: torch.Tensor
    tail_identity: torch.Tensor | None


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


def unpack(plans, metadata, received):
    """Validate every sender before returning any rows for persistent scatter.

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
    for rank, (plan, expected) in enumerate(zip(plans, metadata)):
        header = received[rank, :HEADER_BYTES].view(torch.int64)
        positions = records[rank, :, :8].contiguous().view(torch.int64).flatten()
        valid = (header == expected.header).all() & (
            positions == expected.positions
        ).all()
        if plan.ratio == 2:
            tail = received[rank, end:].view(2, TAIL_RECORD_BYTES)
            identity = tail[:, :16].contiguous().view(torch.int64)
            valid &= (identity == expected.tail_identity).all()
            if not plan.tail_count:
                valid &= (tail[:, 16:] == 0).all()
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
