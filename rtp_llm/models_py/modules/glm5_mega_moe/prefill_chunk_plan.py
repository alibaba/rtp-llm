"""Forward-local EP call-count contract for eager NVFP4 prefill.

The caller must agree on ``chunks`` across the actual MegaMoE group before
entering any routed layer. This module does not select execution phases or
perform collectives, and must not be used to pad attention or KV state.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefillChunkPlan:
    capacity: int
    chunks: int

    def __post_init__(self):
        if self.capacity < 1 or self.chunks < 1:
            raise ValueError("prefill chunk capacity and count must be positive")

    def validate_rows(self, rows: int) -> None:
        if rows < 0 or rows > self.capacity * self.chunks:
            raise ValueError("local rows exceed agreed prefill chunk plan")


def local_chunk_count(rows: int, capacity: int) -> int:
    if rows < 0 or capacity < 1:
        raise ValueError("invalid prefill row count or chunk capacity")
    return max(1, (rows + capacity - 1) // capacity)


def run_prefill_chunks(
    hidden,
    weights,
    indices,
    forward_fn,
    plan: PrefillChunkPlan,
    *,
    forward_into_fn=None
):
    """Run exactly the agreed count; preserve real rows before buffer reuse.

    ``forward_fn`` may return a view of a reusable output buffer. Even a rank
    with only one real chunk must copy that result before its dummy calls.
    Dummy routes are valid distinct IDs with zero weights, not masked -1 IDs.
    They execute routed MoE only and never escape into real hidden states.
    An optional native callback writes real chunks into the final allocation;
    dummy calls still use the reusable scratch through ``forward_fn``.
    """
    rows = hidden.shape[0]
    plan.validate_rows(rows)
    if (
        hidden.ndim != 2
        or weights.ndim != 2
        or indices.shape != weights.shape
        or weights.shape[0] != rows
        or weights.shape[1] < 1
    ):
        raise ValueError("invalid prefill routed-MoE input shapes")
    if plan.chunks == 1 and rows:
        return forward_fn(hidden, weights, indices)

    output = (
        torch.empty_like(hidden)
        if forward_into_fn is None
        else torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    )
    dummy = None
    for chunk in range(plan.chunks):
        start = chunk * plan.capacity
        end = min(start + plan.capacity, rows)
        if start < rows:
            if forward_into_fn is None:
                output[start:end].copy_(
                    forward_fn(
                        hidden[start:end], weights[start:end], indices[start:end]
                    )
                )
            else:
                forward_into_fn(
                    hidden[start:end],
                    weights[start:end],
                    indices[start:end],
                    output[start:end],
                )
        else:
            if dummy is None:
                dummy = (
                    hidden.new_zeros((1, hidden.shape[1])),
                    weights.new_zeros((1, weights.shape[1])),
                    torch.arange(
                        weights.shape[1], dtype=indices.dtype, device=indices.device
                    ).unsqueeze(0),
                )
            forward_fn(*dummy)
    return output
