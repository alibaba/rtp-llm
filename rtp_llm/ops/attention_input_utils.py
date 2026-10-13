from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from librtp_compute_ops import PyAttentionInputs


def select_prefill_position_ids(
    attn_inputs: "PyAttentionInputs",
) -> Optional[torch.Tensor]:
    """Select positions for a CP prefill attention invocation.

    The non-prefill guard protects future callers if CP decode support is added.
    """
    position_ids = attn_inputs.combo_position_ids
    if attn_inputs.context_parallel_info is not None:
        if not attn_inputs.is_prefill:
            raise ValueError("context-parallel position ids require a pure-prefill batch")
        shuffle_indices = attn_inputs.context_parallel_info.prefill_shuffle_indices
        if position_ids is None or position_ids.numel() == 0:
            # Prefix reuse requires absolute positions. The C++ CP processor
            # materializes those positions once; reaching this fallback with a
            # non-zero prefix means a caller bypassed that contract.
            prefix_lengths = attn_inputs.prefix_lengths
            if (
                prefix_lengths is not None
                and prefix_lengths.numel() > 0
                and torch.any(prefix_lengths != 0).item()
            ):
                raise ValueError("CP prefix reuse requires explicit position ids")
            return shuffle_indices
        token_count = shuffle_indices.numel()
        if token_count == 0 or position_ids.numel() % token_count != 0:
            raise ValueError("CP position ids must align with the local token count")
        position_axis_count = position_ids.numel() // token_count
        if position_axis_count not in (1, 3):
            raise ValueError("CP position ids must contain one text axis or three mRoPE axes")
    return position_ids
