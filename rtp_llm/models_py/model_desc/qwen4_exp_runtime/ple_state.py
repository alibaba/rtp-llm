from dataclasses import dataclass
from typing import Dict, Optional

import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs


@dataclass
class _PLETargetLayerStage:
    layer_idx: int
    page_size: int
    prefixes: torch.Tensor
    state_pool: torch.Tensor
    ctx_pool: torch.Tensor
    state_inputs: PyAttentionInputs
    ctx_inputs: PyAttentionInputs
    initial_state: torch.Tensor
    candidate_state_inputs: torch.Tensor
    initial_context: torch.Tensor
    candidate_ids: torch.Tensor


@dataclass
class _PLEPreparedWrite:
    state_pool: torch.Tensor
    state_blocks: torch.Tensor
    state_values: torch.Tensor
    original_state: torch.Tensor
    ctx_pool: torch.Tensor
    ctx_blocks: torch.Tensor
    ctx_values: torch.Tensor
    original_context: torch.Tensor


@dataclass
class _PLETargetTransaction:
    batch_size: int
    query_len: int
    prefixes: torch.Tensor
    expected_layers: frozenset[int]
    stages: Dict[int, _PLETargetLayerStage]
    prepared_writes: Optional[list[_PLEPreparedWrite]] = None
    commit_started: bool = False
    tentative_committed: bool = False
    graph_key: Optional[int] = None


@dataclass(frozen=True)
class PLELookupResult:
    """Optional gathered rows; absence requests the regular embedding lookup."""

    embeddings: Optional[torch.Tensor] = None
