"""Masked vocabulary-shard lookup, preserving source FP8 hidden-slice scales."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _vocab_rows(
    Tokens,
    Weight,
    Scales,
    Mask,
    Output,
    N,
    START: tl.constexpr,
    LOCAL_V: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    FP8: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // D, offsets % D
    token = tl.load(Tokens + row * TOKEN_STRIDE, row < N, other=-1).to(tl.int64)
    owned = (row < N) & (token >= START) & (token < START + LOCAL_V) & (token < V)
    if HAS_MASK:
        owned = owned & (tl.load(Mask + row * MASK_STRIDE, row < N, other=0) != 0)
    address = (token - START) * D + col
    if FP8:
        raw = tl.load(Weight.to(tl.pointer_type(tl.uint8)) + address, owned, other=0)
        value = raw.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        scale = tl.load(Scales + col // (D // 2)).to(tl.bfloat16)
        value = (value.to(tl.float32) * scale.to(tl.float32)).to(tl.bfloat16)
    else:
        value = tl.load(Weight + address, owned, other=0)
    # Non-owned tokens must contribute zero, including with nonfinite scales.
    value = tl.where(owned, value, 0)
    tl.store(Output + offsets, value, offsets < N * D)


def vocab_rows(tokens, weight, start, vocab_size, scales=None, text_mask=None):
    rows, width = tokens.numel(), weight.shape[1]
    output = torch.empty((rows, width), device=weight.device, dtype=torch.bfloat16)
    if rows:
        has_mask = text_mask is not None and text_mask.numel() != 0
        _vocab_rows[(triton.cdiv(rows * width, 1024),)](
            tokens,
            weight,
            scales if scales is not None else tokens,
            text_mask if has_mask else tokens,
            output,
            rows,
            START=start,
            LOCAL_V=weight.shape[0],
            V=vocab_size,
            D=width,
            FP8=weight.dtype == torch.float8_e4m3fn,
            TOKEN_STRIDE=tokens.stride(0),
            MASK_STRIDE=text_mask.stride(0) if has_mask else 0,
            HAS_MASK=has_mask,
            BLOCK=1024,
            num_warps=4,
        )
    return output
