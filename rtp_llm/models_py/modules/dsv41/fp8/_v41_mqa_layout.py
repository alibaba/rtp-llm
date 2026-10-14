"""Normalize latest DeepGEMM compressed BF16 logits for V4.1 selectors."""

import torch
import triton
import triton.language as tl


@triton.jit
def _normalize(
    scores,
    starts,
    ends,
    output,
    width: tl.constexpr,
    source_width: tl.constexpr,
    source_stride: tl.constexpr,
    relative: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(starts + row)
    end = tl.load(ends + row)
    source_col = col if relative else col - begin
    valid = (
        (col < width)
        & (source_col >= 0)
        & (source_col < source_width)
        & (source_col < end - begin)
    )
    value = tl.load(
        scores + row * source_stride + source_col, valid, other=-float("inf")
    ).to(tl.float32)
    tl.store(output + row * width + col, value, col < width)


def normalize_mqa_logits(scores, starts, ends, width, *, relative):
    output = torch.empty(
        (scores.shape[0], width), dtype=torch.float32, device=scores.device
    )
    if output.numel():
        _normalize[(scores.shape[0], triton.cdiv(width, 512))](
            scores,
            starts,
            ends,
            output,
            width,
            scores.shape[1],
            scores.stride(0),
            relative,
            512,
        )
    return output
