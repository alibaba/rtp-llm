"""Lossless FP32 pair-state snapshots and deferred commits."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["stride", "pages"])
def _transfer(
    pool, slots, rows, stride, pages, ENTRIES: tl.constexpr, WRITE: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    cols = tl.arange(0, 1024)
    tl.device_assert(
        (slot >= -1) & (slot < pages * ENTRIES), "V4.1 invalid pair-state slot"
    )
    active = (slot >= 0) & (slot < pages * ENTRIES)
    address = (slot // ENTRIES) * stride.to(tl.int64) + (slot % ENTRIES) * 1024 + cols
    if WRITE:
        tl.store(pool + address, tl.load(rows + row * 1024 + cols), active)
    else:
        tl.store(rows + row * 1024 + cols, tl.load(pool + address, active, other=0))


def transfer(state, slots, rows, *, write):
    _transfer[(slots.numel(),)](
        state.raw,
        slots,
        rows,
        state.raw.stride(0),
        state.raw.shape[0],
        ENTRIES=state.entries,
        WRITE=write,
        num_warps=4,
    )
