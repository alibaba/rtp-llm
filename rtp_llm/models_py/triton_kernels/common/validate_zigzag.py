"""Check shared input permutations on device; never read a CUDA scalar on host."""

import torch
import triton
import triton.language as tl


@triton.jit
def _validate_mapping(restore, mask, expected, padding, flags, N: tl.constexpr,
                      RS: tl.constexpr, MS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offsets < N
    actual = tl.load(restore + offsets * RS, live, other=0)
    valid = tl.load(mask + offsets * MS, live, other=0)
    reference = tl.load(expected + offsets, live, other=0)
    reference_valid = tl.load(padding + offsets, live, other=0)
    mismatch = live & ((actual != reference) | (valid != reference_valid))
    tl.store(flags + tl.program_id(0), tl.sum(mismatch.to(tl.int32), 0) == 0)


def validate_zigzag(restore, mask, expected, padding, flags):
    _validate_mapping[(flags.numel(),)](restore, mask, expected, padding, flags,
                                      restore.numel(), restore.stride(0), mask.stride(0), 4096)
    torch._assert_async(flags.all(), "shared restore/padding differs from natural routing plan")
