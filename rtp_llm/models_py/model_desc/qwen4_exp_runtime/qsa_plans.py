"""Validated metadata plans retain the cache tensors for their invocation."""

from dataclasses import dataclass

import torch


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
