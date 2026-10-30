"""Single-group MoE routing for small FP32 gate-projection batches."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _k31_decode_topk(
    logits, bias, weights, ids, renormalize: tl.constexpr, scaling_factor: tl.constexpr
):
    row = tl.program_id(0)
    expert = tl.arange(0, 1024)
    value = tl.load(logits + row * 896 + expert, expert < 896, other=0)
    correction = tl.load(bias + expert, expert < 896, other=0)
    score = tl.div_rn(1.0, 1.0 + libdevice.exp(-value)) + correction
    score = tl.where(score == 0.0, 0.0, score)

    # Sort by biased score, then prefer the smaller expert ID on ties.
    bits = score.to(tl.uint32, bitcast=True)
    sign = tl.full((), 0x80000000, tl.uint32)
    sortable = tl.where((bits & sign) != 0, ~bits, bits ^ sign)
    key = (sortable.to(tl.uint64) << 32) | (896 - expert).to(tl.uint32).to(tl.uint64)
    finite = (tl.abs(value) < float("inf")) & (tl.abs(correction) < float("inf"))
    selected = tl.topk(tl.where((expert < 896) & finite, key, 0), 16)

    # The existing CUDA router fills vacant slots with expert zero and uses
    # that expert's unbiased score if it is finite.
    index = tl.where(selected != 0, 896 - selected.to(tl.uint32).to(tl.int32), 0)
    value = tl.load(logits + row * 896 + index)
    correction = tl.load(bias + index)
    finite = (tl.abs(value) < float("inf")) & (tl.abs(correction) < float("inf"))
    unbiased = tl.where(finite, tl.div_rn(1.0, 1.0 + libdevice.exp(-value)), 0.0)
    scale = scaling_factor
    if renormalize:
        scale = tl.div_rn(scaling_factor, tl.sum(unbiased, axis=0) + 1.0e-20)
    slot = tl.arange(0, 16)
    tl.store(weights + row * 16 + slot, unbiased * scale)
    tl.store(ids + row * 16 + slot, index)


def k31_decode_topk(logits, bias, weights, ids, renormalize, scale):
    """Write top-16 FP32 weights and int32 IDs for contiguous FP32 [rows, 896]."""
    _k31_decode_topk[(logits.shape[0],)](
        logits, bias, weights, ids, renormalize, scale, num_warps=4
    )
