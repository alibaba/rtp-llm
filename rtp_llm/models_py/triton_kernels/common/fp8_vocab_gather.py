"""Transport unchanged FP8 vocabulary bytes; dequantize in the HC consumer.

Keep RTP's existing column shards and every shard's original quantization
scale. vLLM's byte communication motivates this boundary, not a weight
requantization or a change to vocabulary ownership.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _vocab_raw_rows(
    Tokens,
    Weight,
    Output,
    N: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    TS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // D, offset % D
    token = tl.load(Tokens + row * TS, row < N, other=-1).to(tl.int64)
    value = tl.load(
        Weight.to(tl.pointer_type(tl.uint8)) + token * D + col,
        (row < N) & (token >= 0) & (token < V),
        other=0,
    )
    tl.store(Output + offset, value, offset < N * D)


@triton.jit
def _vocab_gather_dequant_hyper(
    Source,
    Scales,
    Mask,
    Output,
    N: tl.constexpr,
    LOCAL_D: tl.constexpr,
    TP: tl.constexpr,
    BRANCHES: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = TP * LOCAL_D
    row = offset // (BRANCHES * width)
    col = offset % width
    rank = col // LOCAL_D
    source = (rank * N + row) * LOCAL_D + col % LOCAL_D
    raw = tl.load(Source + source, row < N, other=0)
    value = raw.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
    scale = tl.load(Scales + rank).to(tl.bfloat16)
    value = (value.to(tl.float32) * scale.to(tl.float32)).to(tl.bfloat16)
    if HAS_MASK:
        keep = tl.load(Mask + row * MS, row < N, other=0) != 0
        value = tl.where(keep, value, 0)
    tl.store(Output + offset, value, row < N)


def raw_rows(tokens, weight):
    if not (
        tokens.is_cuda
        and weight.is_cuda
        and tokens.device == weight.device
        and tokens.ndim == 1
        and tokens.dtype in (torch.int32, torch.int64)
        and weight.ndim == 2
        and weight.dtype == torch.float8_e4m3fn
        and weight.is_contiguous()
        and min(weight.shape) > 0
    ):
        raise ValueError("invalid FP8 vocabulary byte lookup")
    rows, width = tokens.numel(), weight.shape[1]
    output = torch.empty((rows, width), dtype=torch.uint8, device=weight.device)
    if rows:
        _vocab_raw_rows[(triton.cdiv(rows * width, 1024),)](
            tokens,
            weight,
            output,
            rows,
            weight.shape[0],
            width,
            tokens.stride(0),
            1024,
            num_warps=4,
        )
    return output


def dequant_hyper(gathered, scales, rows, local_width, tp_size, branches, mask=None):
    if not (
        gathered.is_cuda
        and gathered.dtype == torch.uint8
        and gathered.is_contiguous()
        and rows >= 0
        and min(local_width, tp_size, branches) > 0
        and gathered.numel() == rows * local_width * tp_size
        and scales.device == gathered.device
        and scales.dtype == torch.float32
        and scales.is_contiguous()
        and scales.numel() == tp_size
    ):
        raise ValueError("invalid gathered vocabulary bytes or column scales")
    has_mask = mask is not None and mask.numel() != 0
    if has_mask and (mask.device != gathered.device or mask.shape != (rows,)):
        raise ValueError("vocabulary text mask must match packed token ids")
    output = torch.empty(
        (rows, branches * tp_size * local_width),
        dtype=torch.bfloat16,
        device=gathered.device,
    )
    if rows:
        _vocab_gather_dequant_hyper[(triton.cdiv(output.numel(), 1024),)](
            gathered,
            scales,
            mask if has_mask else gathered,
            output,
            rows,
            local_width,
            tp_size,
            branches,
            has_mask,
            mask.stride(0) if has_mask else 0,
            1024,
            num_warps=4,
        )
    return output
