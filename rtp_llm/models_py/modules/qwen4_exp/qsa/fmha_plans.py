"""Validated paged FMHA reader inputs, materialized before main-cache mutation."""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class PrefillRequest:
    start: int
    end: int
    kv_lens: torch.Tensor
    selected_indices: torch.Tensor


@dataclass(frozen=True)
class IncrementalPrefillInputs:
    batch: int
    tokens: int
    block_table: torch.Tensor
    page_size: int
    requests: list[PrefillRequest]
    selected_indices: torch.Tensor
    prefixes_device: torch.Tensor
