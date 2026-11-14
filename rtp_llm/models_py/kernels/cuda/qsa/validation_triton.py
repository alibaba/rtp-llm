"""Combine independently produced error flags without host scalar reads."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_checks(Inputs, Result, SHIFTS: tl.constexpr):
    verdict = tl.full((), 0, tl.int64)
    for i in tl.static_range(len(SHIFTS)):
        verdict = verdict | (tl.load(Inputs[i]).to(tl.int64) << SHIFTS[i])
    tl.store(Result, verdict)


def pack_checks(flags, shifts):
    result = torch.empty((), dtype=torch.int64, device=flags[0].device)
    _pack_checks[(1,)](tuple(flags), result, tuple(shifts), num_warps=4)
    return result
