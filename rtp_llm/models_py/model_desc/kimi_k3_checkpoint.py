"""Pure-Python checkpoint layout helpers for packed KDA Prefill."""

from __future__ import annotations

from typing import Optional

import torch


def join_packed_conv_outputs(
    outputs: list[torch.Tensor], output_target: Optional[torch.Tensor]
) -> torch.Tensor:
    if len(outputs) == 1:
        return outputs[0]
    if output_target is not None:
        return output_target
    return torch.cat(outputs, dim=0)


def packed_checkpoint_layout(
    sequence_lengths: list[int], checkpoint_interval: int
) -> tuple[list[int], list[int]]:
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if any(length <= 0 for length in sequence_lengths):
        raise ValueError(
            f"checkpoint sequences must be non-empty, got {sequence_lengths}"
        )
    counts = [
        (length + checkpoint_interval - 1) // checkpoint_interval
        for length in sequence_lengths
    ]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return counts, offsets


__all__ = ["join_packed_conv_outputs", "packed_checkpoint_layout"]
