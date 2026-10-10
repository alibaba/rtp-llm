"""Gather and restore MLA prefix caches stored in page-RR layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional, Sequence

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group

_INTEGER_DTYPES = (torch.int32, torch.int64)


def _validate_geometry(
    page_tokens: int,
    shard_size: int,
    shard_rank: int,
    kernel_page_tokens: Optional[int] = None,
) -> int:
    if page_tokens <= 0 or shard_size <= 0:
        raise ValueError("page_tokens and shard_size must be positive")
    if not 0 <= shard_rank < shard_size:
        raise ValueError(f"shard_rank must be in [0, {shard_size}), got {shard_rank}")
    kernel_page_tokens = (
        page_tokens if kernel_page_tokens is None else kernel_page_tokens
    )
    if kernel_page_tokens <= 0 or page_tokens % kernel_page_tokens:
        raise ValueError(
            "page_tokens must be a positive multiple of kernel_page_tokens"
        )
    return kernel_page_tokens


@dataclass(frozen=True)
class MlaPageRRChunkDescriptor:
    """Rank-independent layout for one launch's request-major prefix slices."""

    request_indices: tuple[int, ...]
    prefix_starts: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    local_page_offsets: tuple[int, ...]
    output_token_offsets: tuple[int, ...]
    page_tokens: int
    shard_size: int
    feature_width: int
    _cuda_metadata: Optional[object] = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def batch_size(self) -> int:
        return len(self.prefix_lens)

    @property
    def total_local_pages(self) -> int:
        return self.local_page_offsets[-1]

    @property
    def total_tokens(self) -> int:
        return self.output_token_offsets[-1]

    @property
    def global_page_starts(self) -> tuple[int, ...]:
        return tuple(start // self.page_tokens for start in self.prefix_starts)


# Keep the old private type name usable by tests and compatibility helpers.
MlaPageRRBatchDescriptor = MlaPageRRChunkDescriptor


def _host_int_values(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = []
    for value in values:
        if not isinstance(value, Integral):
            raise TypeError(f"{name} must contain integers, got {value!r}")
        result.append(int(value))
    return tuple(result)


def _chunk_descriptor(
    request_indices: tuple[int, ...],
    prefix_starts: tuple[int, ...],
    prefix_lens: tuple[int, ...],
    *,
    page_tokens: int,
    shard_size: int,
    feature_width: int,
    allow_empty: bool = False,
) -> MlaPageRRChunkDescriptor:
    if not prefix_lens:
        raise ValueError("prefix_lens must be non-empty")
    if not (len(request_indices) == len(prefix_starts) == len(prefix_lens)):
        raise ValueError(
            "request_indices, prefix_starts, and prefix_lens must have the same length"
        )
    if any(request_idx < 0 for request_idx in request_indices):
        raise ValueError("request_indices must be non-negative")
    if any(prefix_start < 0 for prefix_start in prefix_starts):
        raise ValueError("prefix_starts must be non-negative")
    if any(prefix_start % page_tokens for prefix_start in prefix_starts):
        raise ValueError("prefix_starts must be physical-page aligned")
    minimum = 0 if allow_empty else 1
    if any(prefix_len < minimum for prefix_len in prefix_lens):
        qualifier = "non-negative" if allow_empty else "positive"
        raise ValueError(f"prefix_lens must be {qualifier}, got {prefix_lens}")
    if feature_width <= 0:
        raise ValueError("feature_width must be positive")

    local_page_offsets = [0]
    output_token_offsets = [0]
    for prefix_len in prefix_lens:
        physical_pages = (prefix_len + page_tokens - 1) // page_tokens
        padded_pages_per_rank = (physical_pages + shard_size - 1) // shard_size
        local_page_offsets.append(local_page_offsets[-1] + padded_pages_per_rank)
        output_token_offsets.append(output_token_offsets[-1] + prefix_len)
    return MlaPageRRChunkDescriptor(
        request_indices=request_indices,
        prefix_starts=prefix_starts,
        prefix_lens=prefix_lens,
        local_page_offsets=tuple(local_page_offsets),
        output_token_offsets=tuple(output_token_offsets),
        page_tokens=page_tokens,
        shard_size=shard_size,
        feature_width=feature_width,
    )


def _prefix_descriptor(
    prefix_lens: tuple[int, ...],
    *,
    page_tokens: int,
    shard_size: int,
    feature_width: int,
) -> MlaPageRRChunkDescriptor:
    return _chunk_descriptor(
        tuple(range(len(prefix_lens))),
        (0,) * len(prefix_lens),
        prefix_lens,
        page_tokens=page_tokens,
        shard_size=shard_size,
        feature_width=feature_width,
        allow_empty=True,
    )


def _validate_raw_cache(raw_cache: torch.Tensor, page_tokens: int) -> None:
    if not isinstance(raw_cache, torch.Tensor) or raw_cache.dim() != 3:
        shape = tuple(raw_cache.shape) if isinstance(raw_cache, torch.Tensor) else None
        raise ValueError(f"raw cache must be [blocks, page, features], got {shape}")
    if (
        raw_cache.shape[0] <= 0
        or raw_cache.shape[1] != page_tokens
        or raw_cache.shape[2] <= 0
    ):
        raise ValueError(
            "raw cache must have blocks/features and match page_tokens, "
            f"got shape={tuple(raw_cache.shape)} page_tokens={page_tokens}"
        )


def _pack_mla_page_rr_prefix(
    raw_cache: torch.Tensor,
    local_block_table: torch.Tensor,
    descriptor: MlaPageRRChunkDescriptor,
    shard_rank: int,
) -> torch.Tensor:
    page_tokens = descriptor.page_tokens
    if not isinstance(local_block_table, torch.Tensor) or local_block_table.dim() != 2:
        shape = (
            tuple(local_block_table.shape)
            if isinstance(local_block_table, torch.Tensor)
            else None
        )
        raise ValueError(f"local block table must be 2D, got {shape}")
    if local_block_table.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            "local block table must use torch.int32 or torch.int64, got "
            f"{local_block_table.dtype}"
        )
    if local_block_table.device != raw_cache.device:
        raise ValueError("raw cache and local block table must be on the same device")

    required_requests = max(descriptor.request_indices, default=-1) + 1
    if int(local_block_table.shape[0]) < required_requests:
        raise ValueError(
            "local block table does not contain every descriptor request: "
            f"table={local_block_table.shape[0]} required={required_requests}"
        )

    output_shape = (
        descriptor.total_local_pages,
        page_tokens,
        descriptor.feature_width,
    )
    if descriptor.total_local_pages == 0:
        return raw_cache.new_empty(output_shape)

    from rtp_llm.models_py.triton_kernels.mla.page_rr import pack_prefix_cuda

    payload = raw_cache.new_empty(output_shape)
    pack_prefix_cuda(raw_cache, local_block_table, descriptor, shard_rank, payload)
    return payload


def _restore_mla_page_rr_prefix(
    gathered_payload: torch.Tensor,
    descriptor: MlaPageRRChunkDescriptor,
) -> torch.Tensor:
    """Remove rank/stripe padding into newly allocated request-major rows."""
    output = gathered_payload.new_empty(
        (descriptor.total_tokens, descriptor.feature_width)
    )
    if output.shape[0]:
        from rtp_llm.models_py.triton_kernels.mla.page_rr import restore_prefix_cuda

        restore_prefix_cuda(gathered_payload, descriptor, output)
    return output


@dataclass(frozen=True)
class MlaPageRRCacheGather:
    """Use physical pages for ownership and kernel pages for table/cache views."""

    page_tokens: int
    shard_size: int
    shard_rank: int
    kernel_page_tokens: Optional[int] = None
    _prefix_descriptor: Optional[MlaPageRRChunkDescriptor] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        kernel_page_tokens = _validate_geometry(
            self.page_tokens, self.shard_size, self.shard_rank, self.kernel_page_tokens
        )
        object.__setattr__(self, "kernel_page_tokens", kernel_page_tokens)

    def read_prefix(
        self,
        raw_cache: torch.Tensor,
        local_block_table: torch.Tensor,
        prefix_lens: Sequence[int],
    ) -> torch.Tensor:
        """Restore canonical prefixes; skip AllGather when all prefixes are empty."""

        payload = self._pack_prefix(raw_cache, local_block_table, prefix_lens)
        descriptor = self._prefix_descriptor
        gathered = payload.new_empty((self.shard_size, *payload.shape))
        if descriptor.total_local_pages:
            collective_torch.all_gather_into(payload, gathered.flatten(0, 1), Group.TP)
        return _restore_mla_page_rr_prefix(gathered, descriptor)

    def build_prefix_chunk_descriptor(
        self,
        request_indices: Sequence[int],
        prefix_starts: Sequence[int],
        prefix_lens: Sequence[int],
        *,
        feature_width: int,
    ) -> MlaPageRRChunkDescriptor:
        """Describe the exact request slices consumed by one KV-up launch."""

        return _chunk_descriptor(
            _host_int_values("request_indices", request_indices),
            _host_int_values("prefix_starts", prefix_starts),
            _host_int_values("prefix_lens", prefix_lens),
            page_tokens=self.page_tokens,
            shard_size=self.shard_size,
            feature_width=int(feature_width),
        )

    def read_prefix_chunk(
        self,
        raw_cache: torch.Tensor,
        local_block_table: torch.Tensor,
        descriptor: MlaPageRRChunkDescriptor,
    ) -> torch.Tensor:
        """Pack, AllGather, and restore one launch-local canonical chunk."""

        payload = self._pack_prefix_chunk(
            raw_cache,
            local_block_table,
            descriptor,
        )
        gathered = payload.new_empty((self.shard_size, *payload.shape))
        if descriptor.total_local_pages:
            collective_torch.all_gather_into(payload, gathered.flatten(0, 1), Group.TP)
        return _restore_mla_page_rr_prefix(gathered, descriptor)

    def _pack_prefix_chunk(
        self,
        raw_cache: torch.Tensor,
        local_block_table: torch.Tensor,
        descriptor: MlaPageRRChunkDescriptor,
    ) -> torch.Tensor:
        _validate_raw_cache(raw_cache, self.kernel_page_tokens)
        if (
            descriptor.page_tokens != self.page_tokens
            or descriptor.shard_size != self.shard_size
            or descriptor.feature_width != raw_cache.shape[2]
        ):
            raise ValueError("prefix chunk descriptor does not match cache geometry")
        return _pack_mla_page_rr_prefix(
            raw_cache,
            local_block_table,
            descriptor,
            self.shard_rank,
        )

    def _pack_prefix(
        self,
        raw_cache: torch.Tensor,
        local_block_table: torch.Tensor,
        prefix_lens: Sequence[int],
    ) -> torch.Tensor:
        # The request's attention wrapper reuses this adapter across MLA layers.
        # Retain only the latest batch, without any process-global metadata cache.
        _validate_raw_cache(raw_cache, self.kernel_page_tokens)
        prefix_values = _host_int_values("prefix_lens", prefix_lens)
        descriptor = self._prefix_descriptor
        if (
            descriptor is None
            or descriptor.prefix_lens != prefix_values
            or descriptor.feature_width != raw_cache.shape[2]
        ):
            descriptor = _prefix_descriptor(
                prefix_values,
                page_tokens=self.page_tokens,
                shard_size=self.shard_size,
                feature_width=raw_cache.shape[2],
            )
            object.__setattr__(self, "_prefix_descriptor", descriptor)
        return _pack_mla_page_rr_prefix(
            raw_cache, local_block_table, descriptor, self.shard_rank
        )


__all__ = [
    "MlaPageRRCacheGather",
    "MlaPageRRChunkDescriptor",
]
