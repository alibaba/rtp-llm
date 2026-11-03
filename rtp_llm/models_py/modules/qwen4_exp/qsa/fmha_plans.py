"""Validated paged FMHA reader inputs, materialized before main-cache mutation."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import Optional, overload

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
    prefix: Optional[int] = None


class _PrefillRequests(Sequence[PrefillRequest]):
    """Materialize request-local views only when a fallback consumes them."""

    def __init__(
        self,
        lengths: Sequence[int],
        prefixes: Sequence[int],
        visible: torch.Tensor,
        selected: torch.Tensor,
    ) -> None:
        self.lengths = tuple(lengths)
        self.prefixes = tuple(prefixes)
        self.offsets = (0, *accumulate(self.lengths))
        self.visible = visible
        self.selected = selected
        self._requests: list[Optional[PrefillRequest]] = [None] * len(self.lengths)

    def __len__(self) -> int:
        return len(self.lengths)

    @overload
    def __getitem__(self, index: int) -> PrefillRequest: ...

    @overload
    def __getitem__(self, index: slice) -> list[PrefillRequest]: ...

    def __getitem__(self, index: int | slice) -> PrefillRequest | list[PrefillRequest]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        request = self._requests[index]
        if request is None:
            start, end = self.offsets[index : index + 2]
            request = PrefillRequest(
                start,
                end,
                self.visible[None, start:end],
                self.selected[None, start:end].contiguous(),
                self.prefixes[index],
            )
            self._requests[index] = request
        return request


@dataclass(frozen=True)
class IncrementalPrefillInputs:
    batch: int
    tokens: int
    block_table: torch.Tensor
    page_size: int
    requests: Sequence[PrefillRequest]
    selected_indices: torch.Tensor
    prefixes_device: torch.Tensor
    cu_seqlens: Optional[torch.Tensor] = None

    @property
    def lengths_host(self) -> tuple[int, ...]:
        if isinstance(self.requests, _PrefillRequests):
            return self.requests.lengths
        return tuple(req.end - req.start for req in self.requests)

    @property
    def prefixes_host(self) -> tuple[Optional[int], ...]:
        if isinstance(self.requests, _PrefillRequests):
            return self.requests.prefixes
        return tuple(req.prefix for req in self.requests)
