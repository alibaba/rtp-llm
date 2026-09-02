# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The host metadata contract is adapted from vLLM's attention-state merge.
"""AOT CUDA segmented state merge for natural-log FlashMLA outputs."""

import torch


def merge_attention_states_segmented_in_place(
    output: torch.Tensor,
    output_lse: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    partial_q_indptr: torch.Tensor,
    destination_starts: torch.Tensor,
) -> None:
    """Merge one packed partial segment per owner into canonical states.

    This is a private executor primitive. ``partial_q_indptr`` must be monotonic,
    start at zero, and cover every partial row exactly. Destination ranges must
    be in bounds, ordered, and disjoint. State buffers must not alias each other
    or the metadata backing storage; metadata views may share non-overlapping
    ranges of one allocation. The forward workspace owns these invariants.
    """

    from rtp_llm.ops.compute_ops import rtp_llm_ops

    rtp_llm_ops._flashmla_merge_attention_states_segmented_in_place(
        output,
        output_lse,
        partial_output,
        partial_lse,
        partial_q_indptr,
        destination_starts,
    )
