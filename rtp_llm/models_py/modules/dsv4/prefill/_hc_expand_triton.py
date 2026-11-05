"""Materialize four independent mHC lanes from the Prefill embedding.

R52 port of b394128ed4 onto the dsv4 layout. Byte-exact data movement: the
Triton kernel reproduces ``source.unsqueeze(1).repeat(1, 4, 1)`` exactly
(upstream test asserts torch.equal). Gated behind DSV41_MHC_TRITON_FUSION.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_hc_kernel(
    source,
    output,
    tokens: tl.constexpr,
    width: tl.constexpr,
):
    rows = (tl.program_id(0) * 4 + tl.arange(0, 4)).to(tl.int64)
    cols = tl.program_id(1) * 512 + tl.arange(0, 512)
    valid = (rows[:, None] < tokens) & (cols[None, :] < width)
    values = tl.load(source + rows[:, None] * width + cols[None, :], valid, other=0)
    for lane in tl.static_range(4):
        tl.store(
            output + rows[:, None] * (4 * width) + lane * width + cols[None, :],
            values,
            valid,
        )


def expand_hc(source: torch.Tensor) -> torch.Tensor:
    tokens, width = source.shape
    output = torch.empty((tokens, 4, width), device=source.device, dtype=source.dtype)
    _expand_hc_kernel[(triton.cdiv(tokens, 4), triton.cdiv(width, 512))](
        source, output, tokens, width, num_warps=4
    )
    return output
