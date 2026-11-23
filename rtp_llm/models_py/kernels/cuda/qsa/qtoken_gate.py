"""Apply the Qwen4 BF16 output gate to a paged QSA result."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ELEMENTS"])
def _qtoken_output_gate(
    Output,
    Gate,
    LiveTokens,
    ELEMENTS,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    bounds = offsets < ELEMENTS
    live = bounds
    if LiveTokens is not None:
        live = live & (offsets // WIDTH < tl.load(LiveTokens))

    values = tl.load(Output + offsets, live, other=0).to(tl.float32)
    logits = tl.load(Gate + offsets, live, other=0).to(tl.float32)
    # Attention has already rounded its context to BF16. Keep the sigmoid
    # and multiplication in FP32, then round the gated result on store.
    weights = tl.sigmoid(logits)
    tl.store(Output + offsets, values * weights, bounds)


def apply_qtoken_gate(
    output: torch.Tensor, gate: torch.Tensor, *, live_tokens: torch.Tensor | None = None
) -> torch.Tensor:
    """Gate contiguous BF16 context in place, zeroing optional padded rows.

    Gate logits have the same element count and memory order as the context.
    ``live_tokens`` is an optional device scalar containing the live row count.
    """
    tokens = output.size(0)
    if tokens == 0:
        return output
    elements = output.numel()
    block = 512
    _qtoken_output_gate[(triton.cdiv(elements, block),)](
        output, gate, live_tokens, elements, elements // tokens, block, num_warps=4
    )
    return output
