"""Integer-only fusion for CP prefill planning (RTP_GLM5_PREFILL_OPTIMIZE=1).

Lengths are nonnegative and request IDs/physical page tables follow the CP
planner contract. No tensor values are copied to the host except the total
restore length needed to allocate its dynamically sized output.
"""

from typing import Optional

import torch
import triton as tr
import triton.language as tl


def _supported(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in (torch.int32, torch.int64) and x.is_contiguous()


@tr.jit
def _floor_div(x, divisor: tl.constexpr):
    return x // divisor - (x % divisor < 0).to(x.dtype)


@tr.jit(do_not_specialize=["N", "COLS", "PAGE_STRIDE", "PAGE_ROWS"])
def _slot_mapping(
    Positions,
    Requests,
    Pages,
    Out,
    N,
    COLS,
    PAGE_STRIDE,
    PAGE_ROWS,
    TPB: tl.constexpr,
    OWNER_TPB: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Positions + i, i < N, other=0).to(tl.int64)
    req = tl.load(Requests + i, i < N, other=0).to(tl.int64)
    if COLS == 0:
        slot = tl.full((BLOCK,), -1, tl.int64)
    else:
        owner = _floor_div(pos, OWNER_TPB)
        local_owner = _floor_div(owner, CP)
        owner_offset = pos - owner * OWNER_TPB
        col = local_owner * (OWNER_TPB // TPB) + owner_offset // TPB
        valid = (owner - local_owner * CP == RANK) & (col < COLS)
        # Match Torch's negative-index convention on valid advanced indices.
        col = tl.where(valid, col, 0)
        col = tl.where(col < 0, col + COLS, col)
        req = tl.where(req < 0, req + PAGE_ROWS, req)
        page = tl.load(Pages + req * PAGE_STRIDE + col, i < N, other=0).to(tl.int64)
        slot = page * TPB + pos - _floor_div(pos, TPB) * TPB
        slot = tl.where(valid & (page > 0), slot, -1)
    tl.store(Out + i, slot, i < N)


def try_slot_mapping(
    positions: torch.Tensor,
    pages: torch.Tensor,
    requests: torch.Tensor,
    tokens_per_block: int,
    owner_tokens_per_block: int,
    cp_size: int,
    cp_rank: int,
) -> Optional[torch.Tensor]:
    if not (
        _supported(positions)
        and _supported(requests)
        and pages.is_cuda
        and pages.dtype in (torch.int32, torch.int64)
        and positions.ndim == requests.ndim == 1
        and pages.ndim == 2
        and pages.stride(1) == 1
        and positions.shape == requests.shape
        and positions.device == pages.device == requests.device
        and tokens_per_block > 0
        and owner_tokens_per_block > 0
        and owner_tokens_per_block % tokens_per_block == 0
        and cp_size > 0
        and 0 <= cp_rank < cp_size
    ):
        return None
    out = torch.empty(positions.shape, dtype=torch.int64, device=positions.device)
    if out.numel():
        with torch.cuda.device(positions.device):
            _slot_mapping[(tr.cdiv(out.numel(), 256),)](
                positions,
                requests,
                pages,
                out,
                out.numel(),
                pages.shape[1],
                pages.stride(0),
                pages.shape[0],
                tokens_per_block,
                owner_tokens_per_block,
                cp_size,
                cp_rank,
                256,
            )
    return out


@tr.jit(do_not_specialize=["N"])
def _local_lengths(
    Lengths,
    Out,
    N,
    CP: tl.constexpr,
    TPB: tl.constexpr,
    RANK: tl.constexpr,
    ACTUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(Lengths + i, i < N, other=0).to(tl.int64)
    if ACTUAL:
        length = tl.maximum(length, 0)
        cycles = length // (TPB * CP)
        tail = length % (TPB * CP)
        result = cycles * TPB + tl.minimum(tl.maximum(tail - RANK * TPB, 0), TPB)
    else:
        result = _floor_div(length + TPB * CP - 1, TPB * CP) * TPB
    tl.store(Out + i, result, i < N)


def try_local_lengths(
    lengths: torch.Tensor,
    cp_size: int,
    block_size: int,
    cp_rank: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if not (
        _supported(lengths)
        and lengths.ndim == 1
        and cp_size > 0
        and block_size > 0
        and (cp_rank is None or 0 <= cp_rank < cp_size)
    ):
        return None
    out = torch.empty(lengths.shape, dtype=torch.int64, device=lengths.device)
    if out.numel():
        with torch.cuda.device(lengths.device):
            _local_lengths[(tr.cdiv(out.numel(), 256),)](
                lengths,
                out,
                out.numel(),
                cp_size,
                block_size,
                0 if cp_rank is None else cp_rank,
                cp_rank is not None,
                256,
            )
    return out


@tr.jit(do_not_specialize=["TOTAL", "BATCH"])
def _restore_indices(
    Lengths,
    Out,
    TOTAL,
    BATCH,
    CP: tl.constexpr,
    TPB: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.arange(0, NB)
    lens = tl.load(Lengths + b, b < BATCH, other=0).to(tl.int64)
    ends = tl.cumsum(lens)
    local = (lens + TPB * CP - 1) // (TPB * CP) * TPB
    local_starts = tl.cumsum(local) - local
    total_local = tl.sum(local, 0)
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    belongs = (row[:, None] >= (ends - lens)[None, :]) & (row[:, None] < ends[None, :])
    start = tl.sum(tl.where(belongs, (ends - lens)[None, :], 0), 1)
    local_start = tl.sum(tl.where(belongs, local_starts[None, :], 0), 1)
    pos = row - start
    block = pos // TPB
    result = (block % CP) * total_local + local_start + (block // CP) * TPB + pos % TPB
    tl.store(Out + row, result, row < TOTAL)


def try_restore_indices(
    lengths: torch.Tensor,
    cp_size: int,
    block_size: int,
    device: torch.device,
    known_total: Optional[int] = None,
) -> Optional[torch.Tensor]:
    # Bound the broadcast prefix lookup; larger batches retain the Torch path.
    if not (
        _supported(lengths)
        and lengths.ndim == 1
        and lengths.numel() <= 32
        and lengths.device == torch.device(device)
        and cp_size > 0
        and block_size > 0
    ):
        return None
    # The CP planner already computes this exact sum from host lengths.
    # Reusing it avoids a device reduction and synchronous scalar readback.
    total = (
        known_total
        if known_total is not None
        else (int(lengths.sum().item()) if lengths.numel() else 0)
    )
    if not isinstance(total, int) or total < 0:
        raise ValueError("known_total must be a nonnegative integer")
    out = torch.empty(total, dtype=torch.int64, device=lengths.device)
    if total:
        with torch.cuda.device(lengths.device):
            _restore_indices[(tr.cdiv(total, 128),)](
                lengths,
                out,
                total,
                lengths.numel(),
                cp_size,
                block_size,
                tr.next_power_of_2(lengths.numel()),
                128,
            )
    return out
