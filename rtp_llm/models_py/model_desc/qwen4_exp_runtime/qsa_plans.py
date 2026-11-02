"""Validated metadata plans retain the cache tensors for their invocation."""

from dataclasses import dataclass

import torch

from .qsa_step_plan import Qwen4ExpPrefillStepPlan


@dataclass(frozen=True)
class IncrementalPrefillPlan:
    lengths: list[int]
    batch_size: int
    head_dim: int
    head_num: int
    ratio: int
    token_budget: int
    prefixes: torch.Tensor
    cu_seqlens: torch.Tensor
    logical_positions: torch.Tensor
    current_cos: torch.Tensor
    current_sin: torch.Tensor
    visible_lengths: torch.Tensor
    compressed_lengths: torch.Tensor
    max_ctx_len: int
    kv_tokens_per_block: int
    kv_entries_per_block: int
    state_tokens_per_block: int
    kv_pool: torch.Tensor
    state_pool: torch.Tensor
    kv_table: torch.Tensor
    state_table: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    # The frozen per-step metadata this layer's plan was assembled from.
    # Selection/FMHA consumers read its per-step constants instead of
    # rebuilding them; the fallbacks keep direct constructions working.
    step: Qwen4ExpPrefillStepPlan | None = None


@dataclass(frozen=True)
class TargetVerifyPlan:
    batch_size: int
    query_len: int
    prefixes: torch.Tensor
    cu_seqlens: torch.Tensor
    visible_lengths: torch.Tensor
    compressed_lengths: torch.Tensor
    max_ctx_len: int
    kv_tokens_per_block: int
    kv_entries_per_block: int
    state_tokens_per_block: int
    kv_pool: torch.Tensor
    state_pool: torch.Tensor
    kv_table: torch.Tensor
    state_table: torch.Tensor
    indexer_id: int
    query_positions: torch.Tensor | None = None
    block_starts: torch.Tensor | None = None
