"""FP8 vocabulary lookup without materializing token or tile index tensors."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _vocab_rows(
    Tokens,
    Weight,
    Scale,
    Mask,
    Output,
    N,
    V: tl.constexpr,
    D: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // D
    col = offsets % D
    token = tl.load(Tokens + row * TOKEN_STRIDE, row < N, other=-1).to(tl.int64)
    valid = (row < N) & (token >= 0) & (token < V)
    if HAS_MASK:
        keep = tl.load(Mask + row * MASK_STRIDE, row < N, other=0) != 0
        valid = valid & keep
    # Match the original gather's two BF16 conversions and BF16 product.
    raw = tl.load(
        Weight.to(tl.pointer_type(tl.uint8)) + token * D + col, valid, other=0
    )
    value = raw.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
    scale = tl.load(Scale).to(tl.bfloat16)
    value = (value.to(tl.float32) * scale.to(tl.float32)).to(tl.bfloat16)
    # A masked text row is zero even when the vocabulary scale is nonfinite.
    if HAS_MASK:
        value = tl.where(keep, value, 0)
    tl.store(Output + offsets, value, offsets < N * D)


def vocab_rows(tokens, weight, scale, text_mask=None):
    """Return the TP-local hidden columns in their original order."""
    n, width = tokens.numel(), weight.shape[1]
    output = torch.empty((n, width), device=weight.device, dtype=torch.bfloat16)
    if n:
        has_mask = text_mask is not None and text_mask.numel() != 0
        _vocab_rows[(triton.cdiv(n * width, 1024),)](
            tokens,
            weight,
            scale,
            text_mask if has_mask else tokens,
            output,
            N=n,
            V=weight.shape[0],
            D=width,
            TOKEN_STRIDE=tokens.stride(0),
            MASK_STRIDE=text_mask.stride(0) if has_mask else 0,
            HAS_MASK=has_mask,
            BLOCK=1024,
            num_warps=4,
        )
    return output


@triton.jit(do_not_specialize=["N"])
def _gather_hyper_rows(
    Source,
    Out,
    N,
    LOCAL_D: tl.constexpr,
    TP: tl.constexpr,
    BRANCHES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = TP * LOCAL_D
    row = offset // (BRANCHES * width)
    column = offset % width
    rank = column // LOCAL_D
    source = (rank * N + row) * LOCAL_D + column % LOCAL_D
    value = tl.load(Source + source, row < N, other=0)
    tl.store(Out + offset, value, row < N)


def gather_hyper_rows(gathered, rows, local_width, tp_size, branches):
    """Combine TP reorder and HC repetition without increasing communication."""
    if not (
        gathered.is_cuda
        and gathered.dtype == torch.bfloat16
        and gathered.is_contiguous()
        and min(local_width, tp_size, branches) > 0
        and rows >= 0
        and gathered.numel() == rows * local_width * tp_size
    ):
        raise ValueError("invalid gathered FP8 vocabulary output layout")
    output = torch.empty(
        (rows, branches * tp_size * local_width),
        device=gathered.device,
        dtype=gathered.dtype,
    )
    if output.numel():
        _gather_hyper_rows[(triton.cdiv(output.numel(), 1024),)](
            gathered,
            output,
            N=rows,
            LOCAL_D=local_width,
            TP=tp_size,
            BRANCHES=branches,
            BLOCK=1024,
            num_warps=4,
        )
    return output
