"""Batch-shape-stable execution for row-independent linear layers."""

from typing import Callable

import torch


def fixed_m_linear(
    linear: Callable[[torch.Tensor], torch.Tensor],
    input: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    """Run ``linear`` with one fixed GEMM M dimension.

    BF16 GEMM dispatchers may use different K-reduction configurations for
    different M values.  That is normally harmless, but a MoE router can turn
    a one-ULP logit difference into a different expert assignment.  Full
    chunks and the zero-padded tail are therefore all evaluated with exactly
    ``chunk_rows`` rows.
    """
    if input.dim() != 2:
        raise ValueError(f"fixed_m_linear expects a 2D input, got {input.dim()}D")
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if input.size(0) == 0:
        return linear(input)

    num_rows, hidden_size = input.shape
    output_chunks = []
    for begin in range(0, num_rows, chunk_rows):
        end = min(begin + chunk_rows, num_rows)
        chunk = input[begin:end]
        valid_rows = end - begin
        if valid_rows != chunk_rows:
            padded = torch.zeros(
                (chunk_rows, hidden_size), dtype=input.dtype, device=input.device
            )
            padded[:valid_rows].copy_(chunk)
            chunk = padded
        output_chunks.append(linear(chunk)[:valid_rows])
    return torch.cat(output_chunks, dim=0)
