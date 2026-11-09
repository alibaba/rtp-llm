"""Torch oracle for QSA selection, with native top-k tie ordering.

Only the ranking primitive is shared with the production fallback. Token ids
are constructed by repeat/interleave; no production selection helper is called.
"""

import torch


def select_qsa_paged_tokens(
    block_logits: torch.Tensor,
    token_lengths: torch.Tensor,
    *,
    compress_ratio: int,
    token_budget: int,
    validate_lengths: bool = True,
) -> torch.Tensor:
    complete = token_lengths // compress_ratio
    if validate_lengths and bool(
        ((token_lengths < 0) | (complete > block_logits.shape[1])).any()
    ):
        raise ValueError("visible token lengths exceed the score table")

    columns = torch.arange(block_logits.shape[1], device=block_logits.device)
    scores = block_logits.masked_fill(columns[None] >= complete[:, None], -torch.inf)
    count = min(token_budget // compress_ratio, block_logits.shape[1])
    values, blocks = scores.topk(count, dim=1)
    # Keep native tie/nonfinite ordering before replacing invalid selections.
    selected = blocks.repeat_interleave(compress_ratio, dim=1) * compress_ratio
    selected += (
        torch.arange(count * compress_ratio, device=scores.device) % compress_ratio
    )
    selected.masked_fill_(
        ~values.isfinite().repeat_interleave(compress_ratio, dim=1), -1
    )

    padding = blocks.new_full((len(block_logits), token_budget - selected.shape[1]), -1)
    tail = complete[:, None] * compress_ratio + torch.arange(
        compress_ratio - 1, device=scores.device
    )
    tail.masked_fill_(tail >= token_lengths[:, None], -1)
    return torch.cat((selected, padding, tail), dim=1).to(torch.int32)
