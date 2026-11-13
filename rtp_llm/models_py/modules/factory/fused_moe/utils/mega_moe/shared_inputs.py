"""Prepare native RTP gates for MegaMoE (qwen35_vl_omega c80e569647a1)."""

import torch
import triton
import triton.language as tl

_SHARED_GATE_CACHE: dict[str, torch.Tensor] = {}


@triton.jit(do_not_specialize=["N"])
def _shared_expert_sigmoid(logits, gates, N, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
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
