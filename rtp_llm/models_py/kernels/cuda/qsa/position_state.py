"""Fixed-shape masked stores for the model's int32 MRoPE coordinate ring."""

import triton
import triton.language as tl


@triton.jit
def _store(
    pool, slots, values, count: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row, row < count, -1)
    for axis in tl.static_range(3):
        value = tl.load(values + row * stride + axis, row < count, 0)
        tl.store(pool + slot * 3 + axis, value, (row < count) & (slot >= 0))


def store_positions(pool, slots, values):
    _store[(triton.cdiv(slots.numel(), 128),)](
        pool,
        slots,
        values,
        slots.numel(),
        values.stride(0),
        BLOCK=128,
    )
