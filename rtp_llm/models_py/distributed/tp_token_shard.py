"""Route replicated TP tokens once, retaining equal collective shapes."""

from typing import Callable, NamedTuple

import torch


class RoutedTokenShard(NamedTuple):
    hidden: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    rows: int


def slice_routed_tokens(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> RoutedTokenShard:
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("invalid TP rank/size")
    tokens = hidden.shape[0]
    if weights.shape[0] != tokens or indices.shape != weights.shape:
        raise ValueError("routed input and routing metadata must have the same rows")
    rows = max(1, (tokens + tp_size - 1) // tp_size)
    begin = min(tp_rank * rows, tokens)
    size = min(rows, tokens - begin)
    values = [x.narrow(0, begin, size) for x in (hidden, weights, indices)]
    if size != rows:
        # All ranks must execute the same number of MegaMoE chunks, including
        # T < TP and a tail exactly at the chunk-capacity boundary. Zero-weight
        # padding routes to a valid expert and is discarded after the gather.
        values = [
            torch.cat((x, x.new_zeros((rows - size, *x.shape[1:]))), dim=0)
            for x in values
        ]
    return RoutedTokenShard(*values, rows)


def gather_routed_tokens(
    local_output: torch.Tensor,
    original_tokens: int,
    gather: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """The caller supplies the TP collective; shared experts stay outside it."""
    return gather(local_output)[:original_tokens]
