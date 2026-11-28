"""Expert token counts accumulated into the current step's output."""

from __future__ import annotations

from typing import Optional

import torch


def record_expert_load(
    topk_ids: torch.Tensor,
    expert_token_counts: torch.Tensor,
) -> None:
    """Add destination-visible assignments, including padding, to step counts."""
    from rtp_llm.ops import compute_ops

    compute_ops.record_expert_load(topk_ids, expert_token_counts)


class ExpertLoadContext:
    """A layer's step count tensor, indexed by local expert ID."""

    def __init__(self, expert_token_counts: torch.Tensor):
        self.expert_token_counts = expert_token_counts

    def record_ids(self, ids: torch.Tensor) -> None:
        record_expert_load(ids, self.expert_token_counts)


def expert_load_for_layer(
    expert_token_counts: Optional[torch.Tensor], layer_id: int
) -> Optional[ExpertLoadContext]:
    if expert_token_counts is None:
        return None
    return ExpertLoadContext(expert_token_counts[layer_id])
