"""QSA selected-token plan assembly; no cache mutation."""

from __future__ import annotations

import torch


def select_qsa_paged_tokens(
    block_logits: torch.Tensor,
    token_lengths: torch.Tensor,
    *,
    compress_ratio: int,
    token_budget: int,
    validate_lengths: bool = True,
    scores_are_masked: bool = False,
) -> torch.Tensor:
    """Expand scored block top-k plus the unscored partial tail to token ids.

    ``block_logits`` has one row per query and one column per completed compressed
    block. ``token_lengths`` is the corresponding visible main-KV token count.
    The fixed output layout matches :meth:`Qwen4ExpQSAIndexer.forward_causal`:
    ``token_budget`` scored-block token slots followed by ``ratio - 1`` tail
    slots, with ``-1`` padding.

    ``scores_are_masked`` is an internal producer contract: all columns beyond
    ``token_lengths // compress_ratio`` must already contain negative infinity.
    It is used only with the paged-score kernel; generic inputs keep masking.
    """
    if block_logits.dim() != 2 or not block_logits.is_floating_point():
        raise ValueError("block_logits must be a floating-point [rows, blocks] tensor")
    if token_lengths.dim() != 1 or int(token_lengths.numel()) != int(
        block_logits.shape[0]
    ):
        raise ValueError("token_lengths must have one value per logits row")
    if (
        token_lengths.dtype != torch.int32
        or token_lengths.device != block_logits.device
    ):
        raise ValueError("token_lengths must be int32 on the logits device")
    if compress_ratio <= 0 or token_budget <= 0 or token_budget % compress_ratio:
        raise ValueError(
            "token_budget must be positive and divisible by compress_ratio"
        )
    if validate_lengths:
        complete_blocks = token_lengths // compress_ratio
        invalid_lengths = (token_lengths < 0) | (
            complete_blocks > int(block_logits.shape[1])
        )
        if bool(torch.any(invalid_lengths).item()):
            if bool(torch.any(token_lengths < 0).item()):
                raise ValueError("token_lengths must be non-negative")
            raise ValueError(
                "block_logits do not cover every completed block in token_lengths"
            )

    from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_topk_triton import (
        try_select_qsa_tokens,
    )

    result = try_select_qsa_tokens(
        block_logits,
        token_lengths,
        compress_ratio=compress_ratio,
        token_budget=token_budget,
        scores_are_masked=scores_are_masked,
    )
    if result is not None:
        return result

    if not validate_lengths:
        complete_blocks = token_lengths // compress_ratio
    rows = int(block_logits.shape[0])
    block_topk = token_budget // compress_ratio
    output = torch.full(
        (rows, token_budget + compress_ratio - 1),
        -1,
        dtype=torch.int32,
        device=block_logits.device,
    )
    selected_block_count = min(block_topk, int(block_logits.shape[1]))
    if selected_block_count:
        block_ids = torch.arange(
            int(block_logits.shape[1]), device=block_logits.device
        ).unsqueeze(0)
        masked_logits = torch.where(
            block_ids < complete_blocks.unsqueeze(1),
            block_logits,
            torch.full_like(block_logits, float("-inf")),
        )
        values, blocks = torch.topk(masked_logits, selected_block_count, dim=-1)
        valid = torch.isfinite(values)
        block_tokens = blocks.unsqueeze(-1) * compress_ratio + torch.arange(
            compress_ratio, device=block_logits.device
        )
        block_tokens = torch.where(valid.unsqueeze(-1), block_tokens, -1)
        output[:, : selected_block_count * compress_ratio] = block_tokens.flatten(1).to(
            torch.int32
        )

    tail_width = compress_ratio - 1
    if tail_width:
        tail_count = token_lengths % compress_ratio
        tail_offsets = torch.arange(tail_width, device=block_logits.device)
        tail = complete_blocks.unsqueeze(1) * compress_ratio + tail_offsets
        tail = torch.where(
            tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1), tail, -1
        )
        output[:, token_budget:] = tail.to(torch.int32)
    return output
