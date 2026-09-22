"""Unpack rank-local padded scale rows directly into the FP8 GEMM layout."""

import triton
import triton.language as tl

from .cached_launch import CachedLaunch


@triton.jit
def _repack_ag_scale_wire(
    WIRE,
    OUT,
    M: tl.constexpr,
    LOCAL_PAD: tl.constexpr,
    GROUPS: tl.constexpr,
    RANKS: tl.constexpr,
    GLOBAL_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    group, row = offset // GLOBAL_PAD, offset % GLOBAL_PAD
    rank, local_row = row // M, row % M
    source = (rank * GROUPS + group) * LOCAL_PAD + local_row
    value = tl.load(WIRE + source, (group < GROUPS) & (row < RANKS * M), other=0)
    # All output words, including global tail padding, are written exactly once.
    tl.store(OUT + offset, value, offset < GROUPS * GLOBAL_PAD)


_launch_repack = CachedLaunch(_repack_ag_scale_wire)


def repack_ag_scale_wire(wire, local_rows, world_size):
    """Repack the contiguous CUDA int32 wire allocated by _all_gather_quantized.

    QuantizedActivation validates the local layout before the collective.
    """
    groups = wire.shape[0] // world_size
    global_pad = (local_rows * world_size + 3) // 4 * 4
    out = wire.new_empty((groups, global_pad))
    if out.numel():
        _launch_repack(
            ((out.numel() + 255) // 256, 1, 1),
            (wire, out),
            (local_rows, wire.shape[1], groups, world_size, global_pad, 256),
        )
    return out
