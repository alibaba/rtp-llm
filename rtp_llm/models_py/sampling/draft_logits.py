# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True, slots=True)
class DraftLogitsCache:
    values: torch.Tensor
    row_mapping: torch.Tensor | None = None
    # Probabilities are converted to pre-temperature logits on each kernel read.
    values_are_probs: bool = False


@triton.jit
def draft_logits_row(req_state_idx, row_mapping_ptr):
    """Row of a request in this rank's draft logits cache, or -1 if no row.

    ``row_mapping_ptr`` is None for a replicated cache, which is indexed by request
    state. Otherwise it is `DraftLogitsCache.row_mapping`.
    """
    row = req_state_idx
    if row_mapping_ptr is not None:
        row = tl.load(
            row_mapping_ptr + req_state_idx, mask=req_state_idx >= 0, other=-1
        )
    return row.to(tl.int64)
