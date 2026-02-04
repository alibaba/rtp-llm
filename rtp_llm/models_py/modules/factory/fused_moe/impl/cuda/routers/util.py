from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather


def calc_tp_slice(token_num: int, tp_size: int, tp_rank: int) -> Tuple[int, int, int]:
    """Compute the token slice handled by the current TP rank.

    Conventions:
    - token_num is the global token count (inputs are identical/replicated on each TP rank)
    - we approximately evenly split the token dimension by tp_size, and each tp_rank
      is responsible for dispatching its own slice

    Returns:
        (slice_begin, slice_size, tp_token_size)
    """
    # Check the validity of the input parameters
    if tp_size <= 0:
        raise ValueError(f"tp_size must be > 0, but got {tp_size}")
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), but got {tp_rank}")
    if token_num < 0:
        raise ValueError(f"token_num must be >= 0, but got {token_num}")
    # Calculate the token size handled by the current TP rank
    tp_token_size = (token_num + tp_size - 1) // tp_size
    slice_begin = min(tp_token_size * tp_rank, token_num)
    slice_size = min(token_num - slice_begin, tp_token_size)
    return slice_begin, slice_size, tp_token_size


def prepare_tp_slice(
    a1: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
    a1_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Slice dispatch tokens (and their topk_idx/topk_weights) by TP rank.

    Background: the token dimension is replicated across TP ranks, so we split tokens
    across TP ranks and let each rank dispatch a different subset to avoid redundant
    communication / compute.

    Args:
        a1: [num_tokens, hidden]
        a1_scale: Optional quant scale for a1. If provided, it will be sliced the same
            way as a1 along the token dimension. Shape is expected to be
            [num_tokens, ...].
        topk_ids: [num_tokens, topk] (will be converted to int64 here)
        topk_weights: [num_tokens, topk]
        tp_size/tp_rank: from the parallelism config

    Returns:
        (tp_a1, tp_topk_ids, tp_topk_weights, tp_a1_scale)
    """
    # Convert topk_ids to int64
    topk_ids = topk_ids.to(torch.int64)
    # Calculate the token slice handled by the current TP rank
    token_num = a1.size(0)
    slice_begin, slice_size, _ = calc_tp_slice(
        token_num=token_num, tp_size=tp_size, tp_rank=tp_rank
    )
    # Slice the dispatch tokens (and their topk_idx/topk_weights) by TP rank
    tp_a1 = torch.narrow(a1, 0, slice_begin, slice_size)
    tp_a1_scale = (
        torch.narrow(a1_scale, 0, slice_begin, slice_size)
        if a1_scale is not None
        else None
    )
    tp_topk_ids = torch.narrow(topk_ids, 0, slice_begin, slice_size)
    tp_topk_weights = torch.narrow(topk_weights, 0, slice_begin, slice_size)
    return tp_a1, tp_topk_ids, tp_topk_weights, tp_a1_scale


def finalize_tp_gather(
    combined_x: torch.Tensor,
    *,
    tp_size: int,
    extra_finalize_args: Optional[Dict[str, Any]],
) -> torch.Tensor:
    """All-gather across TP ranks after combine to restore the original token order.

    Conventions:
    - combined_x is the local output slice on the current TP rank, shape [slice_size, hidden]
    - extra_finalize_args must contain original_num_tokens (the global token count before combine)
    """
    # Check the validity of the input parameters
    assert combined_x.dim() == 2, "combined_x must be a 2D tensor"
    if tp_size <= 1:
        return combined_x
    assert extra_finalize_args is not None, "extra_finalize_args is None"
    assert (
        "original_num_tokens" in extra_finalize_args
    ), "extra_finalize_args must contain 'original_num_tokens'"
    # Calculate the token size handled by the current TP rank
    original_num_tokens = int(extra_finalize_args["original_num_tokens"])
    tp_token_size = (original_num_tokens + tp_size - 1) // tp_size
    # If the token size is less than the TP token size, pad the combined_x
    if combined_x.size(0) < tp_token_size:
        padding = torch.empty(
            size=(tp_token_size - combined_x.size(0), combined_x.size(1)),
            device=combined_x.device,
            dtype=combined_x.dtype,
        )
        combined_x = torch.cat([combined_x, padding], dim=0)
    # All-gather the combined_x across TP ranks
    gathered_output = all_gather(combined_x, group=Group.TP).reshape(
        tp_size * tp_token_size, -1
    )
    return gathered_output[:original_num_tokens, :]
