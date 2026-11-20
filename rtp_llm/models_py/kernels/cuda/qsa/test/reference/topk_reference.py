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


def assert_qsa_selection_close(actual, expected, scores, lengths):
    if not (
        scores.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(scores.device) == (9, 0)
        and scores.dtype == torch.float32
        and scores.numel() > 0
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        return
    # Native RTP TopK has unspecified order and equal-score winners.
    # Keep the value multiset, unique blocks, grouping and exact tail contract.
    torch.testing.assert_close(actual[:, 2048:], expected[:, 2048:], atol=0, rtol=0)
    blocks = actual[:, :2048:4]
    valid = blocks >= 0
    indices = blocks // 4
    complete = lengths[:, None].to(torch.int64) // 4
    in_bounds = (indices < scores.shape[1]) & (indices < complete)
    assert bool(torch.all(~valid | ((blocks % 4 == 0) & in_bounds)))
    grouped = torch.where(
        valid[:, :, None],
        blocks[:, :, None] + torch.arange(4, device=scores.device, dtype=blocks.dtype),
        -1,
    ).reshape(-1, 2048)
    torch.testing.assert_close(actual[:, :2048], grouped, atol=0, rtol=0)
    ordered = indices.masked_fill(~valid, scores.shape[1]).sort(-1).values
    unique = (ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] == scores.shape[1])
    assert bool(torch.all(unique))

    def selected_values(tokens):
        block = tokens[:, :2048:4]
        values = scores.gather(1, (block // 4).clamp(0, scores.shape[1] - 1).long())
        assert bool(torch.all((block < 0) | (values > -torch.inf)))
        return values.masked_fill(block < 0, -torch.inf).sort(-1).values

    torch.testing.assert_close(
        selected_values(actual), selected_values(expected), atol=0, rtol=0
    )
