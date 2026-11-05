"""Validated tensor operands shared by QSA producers and readers."""

import os
from dataclasses import dataclass
from typing import Optional

import torch


def qsa_strict_validation_enabled() -> bool:
    """Retain synchronous validation for explicit diagnostics."""
    return os.environ.get("RTP_LLM_QSA_STRICT_VALIDATION", "") not in (
        "",
        "0",
        "false",
        "False",
    )


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
