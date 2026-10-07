"""Fixed-row M3.1 Prefill routers with FP32 accumulation.

FP32 gates use three BF16 components reconstructing the raw weight exactly.
BF16 gates keep their BF16 input, weight and output without expansion. Both
paths fix the reduction order across row counts; neither claims vendor GEMM
bit equivalence.
"""

import torch
import triton
import triton.language as tl


def expand_fp32_router_weight(weight: torch.Tensor):
    """Initialization-only expansion; never call during forward or capture."""
    if weight.dtype != torch.float32 or weight.shape != (128, 6144):
        raise ValueError("M3.1 Prefill router requires FP32 [128,6144] gate")
    if weight.stride() not in ((6144, 1), (1, 128)):
        raise ValueError("M3.1 Prefill router requires dense row/column-major gate")
    if not torch.isfinite(weight).all().item():
        raise ValueError("M3.1 Prefill router requires finite FP32 gate")
    high = weight.to(torch.bfloat16)
    residual = weight - high.float()
    middle = residual.to(torch.bfloat16)
    low = (residual - middle.float()).to(torch.bfloat16)
    if not torch.equal((high.float() + middle.float()) + low.float(), weight):
        raise ValueError("Three BF16 parts do not reconstruct the FP32 gate exactly")
    return high, middle, low


@triton.jit
def _prefill_bf16_router(
    X, W, OUTPUT, ROWS: tl.constexpr, WS0: tl.constexpr, WS1: tl.constexpr
):
    rows = (tl.program_id(0) * 16 + tl.arange(0, 16)).to(tl.int64)
    experts = tl.program_id(1) * 64 + tl.arange(0, 64)
    inner = tl.arange(0, 128)
    acc = tl.zeros((16, 64), tl.float32)
    # Fixed K reduction, even when a request moves within a different M tile.
    for step in range(48):
        columns = step * 128 + inner
        x = tl.load(
            X + rows[:, None] * 6144 + columns[None, :],
            rows[:, None] < ROWS,
            other=0,
        )
        w = tl.load(W + columns[:, None] * WS1 + experts[None, :] * WS0)
        acc = tl.dot(x, w, acc)
    # Store to BF16, preserving the configured gate's output precision.
    tl.store(
        OUTPUT + rows[:, None] * 128 + experts[None, :],
        acc,
        rows[:, None] < ROWS,
    )


def minimax_m31_prefill_bf16_router_logits(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """BF16 gate with shape-invariant row math and no partial workspace."""
    if (
        x.ndim != 2
        or x.shape[1] != 6144
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or weight.shape != (128, 6144)
        or weight.dtype != torch.bfloat16
        or weight.device != x.device
        or weight.stride() not in ((6144, 1), (1, 128))
    ):
        raise ValueError(
            "M3.1 Prefill BF16 router requires contiguous CUDA [M,6144] rows "
            "and dense CUDA BF16 [128,6144] gate"
        )
    rows = x.shape[0]
    output = torch.empty((rows, 128), device=x.device, dtype=torch.bfloat16)
    if rows:
        _prefill_bf16_router[(triton.cdiv(rows, 16), 2)](
            x, weight, output, rows, *weight.stride(), num_warps=4, num_stages=2
        )
    return output


@triton.jit
def _prefill_router_partials(
    X,
    HIGH,
    MIDDLE,
    LOW,
    PARTIAL,
    ROWS: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
):
    # Widen before multiplying by the hidden stride: row 349525 already
    # crosses INT32_MAX within its 6144-element input row.
    rows = (tl.program_id(0) * 16 + tl.arange(0, 16)).to(tl.int64)
    experts = tl.program_id(1) * 64 + tl.arange(0, 64)
    inner = tl.arange(0, 64)
    split = tl.program_id(2).to(tl.int64)
    high_acc = tl.zeros((16, 64), tl.float32)
    middle_acc = tl.zeros((16, 64), tl.float32)
    low_acc = tl.zeros((16, 64), tl.float32)
    for step in range(12):
        columns = (step * 8 + split) * 64 + inner
        x = tl.load(
            X + rows[:, None] * 6144 + columns[None, :], rows[:, None] < ROWS, other=0
        )
        offset = columns[:, None] * WS1 + experts[None, :] * WS0
        high_acc = tl.dot(x, tl.load(HIGH + offset), high_acc)
        middle_acc = tl.dot(x, tl.load(MIDDLE + offset), middle_acc)
        low_acc = tl.dot(x, tl.load(LOW + offset), low_acc)
    tl.store(
        PARTIAL + split * ROWS * 128 + rows[:, None] * 128 + experts[None, :],
        (high_acc + middle_acc) + low_acc,
        rows[:, None] < ROWS,
    )


@triton.jit
def _prefill_router_reduce(PARTIAL, OUTPUT, SIZE: tl.constexpr):
    elements = (tl.program_id(0) * 256 + tl.arange(0, 256)).to(tl.int64)
    splits = tl.arange(0, 8).to(tl.int64)
    values = tl.load(
        PARTIAL + splits[:, None] * SIZE + elements[None, :],
        elements[None, :] < SIZE,
        other=0,
    )
    tl.store(OUTPUT + elements, tl.sum(values, axis=0), elements < SIZE)


def minimax_m31_prefill_router_logits(x: torch.Tensor, parts):
    """Ordinary Prefill only; temporary split workspace is 4096*rows bytes.

    Parts are prepared/owned by the model, not cached by tensor address here.
    No full FP32 activation materialization and no mutable cross-request workspace.
    """
    if (
        x.ndim != 2
        or x.shape[1] != 6144
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or len(parts) != 3
    ):
        raise ValueError("M3.1 Prefill router requires contiguous CUDA BF16 [M,6144]")
    first = parts[0]
    if first.stride() not in ((6144, 1), (1, 128)) or any(
        w.shape != (128, 6144)
        or w.dtype != torch.bfloat16
        or w.device != x.device
        or w.stride() != first.stride()
        for w in parts
    ):
        raise ValueError(
            "M3.1 Prefill router requires three same-layout CUDA BF16 gate parts"
        )
    rows = x.shape[0]
    output = torch.empty((rows, 128), device=x.device, dtype=torch.float32)
    if rows == 0:
        return output
    partial = torch.empty((8, rows, 128), device=x.device, dtype=torch.float32)
    _prefill_router_partials[(triton.cdiv(rows, 16), 2, 8)](
        x, *parts, partial, rows, *first.stride(), num_warps=4, num_stages=2
    )
    _prefill_router_reduce[(triton.cdiv(rows * 128, 256),)](
        partial, output, rows * 128, num_warps=4
    )
    return output
