"""Prepare the native RTP gate and shared-expert scale layout for MegaMoE."""

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_expert_sigmoid(
    logits, gates, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Identical FP32 expression to _SigmoidGateScaleAdd_kernel, no BF16 cast.
    value = tl.load(logits + row * STRIDE, row < N, 0).to(tl.float32)
    tl.store(gates + row, tl.sigmoid(value), row < N)


def shared_expert_sigmoid(logits):
    if logits.ndim != 2 or logits.shape[1] != 1 or not logits.is_cuda:
        raise ValueError("Expected CUDA gate logits [tokens, 1]")
    gates = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.float32)
    if gates.numel():
        _shared_expert_sigmoid[(triton.cdiv(gates.numel(), 256),)](
            logits, gates, gates.numel(), logits.stride(0), 256
        )
    return gates


@triton.jit
def _stage_shared_scales(
    source,
    destination,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PAD_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx % ROWS, idx // ROWS
    local = row % PAD_M
    original = (local // 128) * 128 + (local % 4) * 32 + (local % 128) // 4
    token = row // PAD_M * BLOCK_M + original
    valid = (idx < ROWS * COLS) & (original < BLOCK_M) & (token < N)
    value = tl.load(source + token * S0 + col * S1, valid, 0)
    tl.store(destination + row * D0 + col * D1, value, idx < ROWS * COLS)


def stage_shared_scales(destination, source, block_m):
    if source.dtype != torch.int32 or destination.dtype != torch.int32:
        raise ValueError("MegaMoE shared scales must be packed UE8M0 int32")
    if destination.shape[1] != source.shape[1]:
        raise ValueError("Shared scale width mismatch")
    rows, cols = destination.shape
    if triton.cdiv(source.shape[0], block_m) * triton.cdiv(block_m, 128) * 128 > rows:
        raise ValueError("Shared scale buffer does not cover the local token batch")
    _stage_shared_scales[(triton.cdiv(rows * cols, 256),)](
        source,
        destination,
        source.shape[0],
        rows,
        cols,
        *source.stride(),
        *destination.stride(),
        block_m,
        triton.cdiv(block_m, 128) * 128,
        256
    )
