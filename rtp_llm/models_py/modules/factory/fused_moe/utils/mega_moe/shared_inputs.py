"""Prepare the native RTP shared-expert gate for MegaMoE."""

import torch
import triton
import triton.language as tl

_SHARED_GATE_CACHE: dict[str, torch.Tensor] = {}


@triton.jit
def _shared_expert_sigmoid(
    logits, gates, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Identical FP32 expression to _SigmoidGateScaleAdd_kernel, no BF16 cast.
    value = tl.load(logits + row * STRIDE, row < N, 0).to(tl.float32)
    tl.store(gates + row, tl.sigmoid(value), row < N)


def ensure_shared_gate_capacity(device, capacity: int) -> torch.Tensor:
    """Preallocate a CUDA-graph-safe FP32 gate buffer of at least ``capacity``."""
    capacity = max(int(capacity), 1)
    device = torch.device(device)
    key = str(device)
    cached = _SHARED_GATE_CACHE.get(key)
    if (
        cached is None
        or cached.numel() < capacity
        or cached.device != device
        or cached.dtype != torch.float32
    ):
        cached = torch.empty((capacity,), device=device, dtype=torch.float32)
        _SHARED_GATE_CACHE[key] = cached
    return cached


def shared_expert_sigmoid(logits):
    if logits.ndim != 2 or logits.shape[1] != 1 or not logits.is_cuda:
        raise ValueError("Expected CUDA gate logits [tokens, 1]")
    tokens = logits.shape[0]
    cached = _SHARED_GATE_CACHE.get(str(logits.device))
    if cached is None or cached.numel() < max(tokens, 1):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "shared_expert_sigmoid has no static gate buffer large enough "
                f"for tokens={tokens} during CUDA graph capture"
            )
        cached = ensure_shared_gate_capacity(logits.device, max(tokens, 1))
    gates = cached[:tokens]
    if tokens:
        _shared_expert_sigmoid[(triton.cdiv(tokens, 256),)](
            logits, gates, tokens, logits.stride(0), 256
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
        256,
    )


@triton.jit
def _expand_packed_activation_scales(
    source,
    destination,
    N: tl.constexpr,
    COLS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx % N, idx // N
    bits = tl.load(source + row * S0 + (col // 4) * S1, idx < N * COLS, 0).to(tl.uint32)
    exponent = (bits >> ((col % 4) * 8)) & 255
    tl.store(
        destination + row * D0 + col * D1,
        exponent * 0x01010101,
        idx < N * COLS,
    )


def expand_packed_activation_scales(destination, source):
    """Write group-128 UE8M0 scales as four identical group-32 scales.

    The source is already packed by the activation quantizer. Replicating its
    exponent bytes preserves the quantization exactly, without the device-to-host
    checks in the general floating-point unpack/repack path. Destination rows
    beyond the local token count remain untouched, including for an empty batch.
    """
    if source.dtype != torch.int32 or destination.dtype != torch.int32:
        raise ValueError("Expected packed int32 UE8M0 activation scales")
    if source.ndim != 2 or destination.ndim != 2:
        raise ValueError("Expected per-token scale matrices")
    n = source.shape[0]
    if destination.shape[0] < n or destination.shape[1] != source.shape[1] * 4:
        raise ValueError("Activation scale geometry mismatch")
    if source.device != destination.device or not source.is_cuda:
        raise ValueError("Expected activation scales on the same CUDA device")
    if n:
        cols = destination.shape[1]
        _expand_packed_activation_scales[(triton.cdiv(n * cols, 256),)](
            source,
            destination,
            n,
            cols,
            *source.stride(),
            *destination.stride(),
            256,
        )
