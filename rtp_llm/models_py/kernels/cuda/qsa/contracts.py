"""Validated tensor operands shared by QSA producers and readers."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class PagedInputs:
    batch: int
    query_len: int
    tokens: int
    selected_indices: torch.Tensor
    kv_lens: torch.Tensor
    block_table: torch.Tensor
    page_size: int
    validated_cache_blocks: Optional[int] = None
