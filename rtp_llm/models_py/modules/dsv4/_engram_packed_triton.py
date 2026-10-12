# SPDX-License-Identifier: Apache-2.0
"""Split CPU Engram packed rows into DeepGEMM MXFP8 inputs.

Each token contains ``heads`` adjacent ``[dim FP8 bytes, dim/32 scale bytes]``
records. This is a byte permutation, not a BF16 dequantization/requantization.
In particular, scale bytes 0/255 and FP8 NaNs are preserved; numerical
equivalence to the BF16 round trip must be checked at the WKV/model boundary.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["tokens", "packed_stride", "scale_stride"])
def _unpack_packed_rows_kernel(
    packed,
    values,
    scales,
    tokens,
    packed_stride,
    scale_stride,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # A virtual K padded to 128 lets each four-scale word have one writer,
    # including a partial final word. Four-row padding is written in this same
    # launch, so TMA never reads uninitialized scale bytes beyond the last row.
    K: tl.constexpr = HEADS * DIM
    GROUPS: tl.constexpr = K // 32
    PACKED_K: tl.constexpr = triton.cdiv(K, 128) * 128
    RECORD: tl.constexpr = DIM + DIM // 32
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token, column = offset // PACKED_K, offset % PACKED_K
    source = token * packed_stride + column // DIM * RECORD + column % DIM
    valid = (token < tokens) & (column < K)
    data = tl.load(packed + source, valid, other=0)
    # Store through a byte pointer: no FP8 conversion or NaN canonicalization.
    value_bytes = values.to(tl.pointer_type(tl.uint8))
    tl.store(value_bytes + token * K + column, data, valid)

    word = tl.program_id(0).to(tl.int64) * (BLOCK // 128) + tl.arange(0, BLOCK // 128)
    scale_token, scale_column = word // (PACKED_K // 128), word % (PACKED_K // 128)
    byte = tl.arange(0, 4)
    group = scale_column[:, None] * 4 + byte[None, :]
    scale_source = (
        scale_token[:, None] * packed_stride
        + group // (DIM // 32) * RECORD
        + DIM
        + group % (DIM // 32)
    )
    exponent = tl.load(
        packed + scale_source,
        (scale_token < tokens)[:, None] & (group < GROUPS),
        other=0,
    ).to(tl.uint32)
    bits = tl.sum(exponent << (byte[None, :] * 8), 1).to(tl.uint32)
    # DeepGEMM expects [ceil(groups/4), align4(T)] int32 storage. Byte j of a
    # word is group 4*c+j (little endian), not FlashInfer's F8_128x4 swizzle.
    tl.store(
        scales + scale_column * scale_stride + scale_token,
        bits.to(tl.int32, bitcast=True),
        scale_token < scale_stride,
    )


def is_supported(packed, heads: int, dim: int) -> bool:
    """Metadata-only gate; unsupported inputs never allocate or launch."""
    return (
        type(heads) is int
        and type(dim) is int
        and heads > 0
        and dim > 0
        and dim % 32 == 0
        and isinstance(packed, torch.Tensor)
        and packed.is_cuda
        and packed.layout == torch.strided
        and packed.dtype == torch.uint8
        and packed.ndim == 2
        and packed.shape[1] == heads * (dim + dim // 32)
        and packed.stride(1) == 1
        and packed.stride(0) >= packed.shape[1]
        and not packed.is_neg()
        and not packed.is_conj()
    )


def unpack_packed_rows(packed, heads: int, dim: int):
    """Return contiguous E4M3 values and packed column-major UE8M0, or None.

    The owning CPU lookup runtime warms this kernel before publishing readiness.
    Token count and row strides do not specialize the kernel, so subsequent
    request sizes (including short tails) use the same compiled variant.
    """
    if not is_supported(packed, heads, dim):
        return None
    tokens = packed.shape[0]
    width = heads * dim
    aligned_tokens = triton.cdiv(tokens, 4) * 4
    scale_columns = triton.cdiv(width // 32, 4)
    values = torch.empty(
        (tokens, width), dtype=torch.float8_e4m3fn, device=packed.device
    )
    storage = torch.empty(
        (scale_columns, aligned_tokens), dtype=torch.int32, device=packed.device
    )
    scales = storage.as_strided((tokens, scale_columns), (1, aligned_tokens))
    if tokens:
        block = 4096
        grid = (triton.cdiv(aligned_tokens * scale_columns * 128, block),)
        _unpack_packed_rows_kernel[grid](
            packed,
            values,
            scales,
            tokens,
            packed.stride(0),
            aligned_tokens,
            HEADS=heads,
            DIM=dim,
            BLOCK=block,
            num_warps=4,
        )
    return values, scales
