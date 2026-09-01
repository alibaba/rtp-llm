"""Backend-neutral read/write adapter for MLA page-RR cache layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional, Sequence

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group

_INTEGER_DTYPES = (torch.int32, torch.int64)


def _validate_geometry(page_tokens: int, shard_size: int, shard_rank: int) -> None:
    if page_tokens <= 0 or shard_size <= 0:
        raise ValueError("page_tokens and shard_size must be positive")
    if not 0 <= shard_rank < shard_size:
        raise ValueError(f"shard_rank must be in [0, {shard_size}), got {shard_rank}")


def _validate_slot_mapping_inputs(
    positions: torch.Tensor,
    batch_indices: torch.Tensor,
    local_block_table: torch.Tensor,
) -> None:
    for name, tensor in (
        ("positions", positions),
        ("batch_indices", batch_indices),
        ("local_block_table", local_block_table),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if positions.dim() != 1:
        raise ValueError(f"positions must be 1D, got shape {tuple(positions.shape)}")
    if batch_indices.dim() != 1:
        raise ValueError(
            f"batch_indices must be 1D, got shape {tuple(batch_indices.shape)}"
        )
    if positions.numel() != batch_indices.numel():
        raise ValueError(
            "positions and batch_indices must have the same length, got "
            f"{positions.numel()} and {batch_indices.numel()}"
        )
    if local_block_table.dim() != 2:
        raise ValueError(
            "local_block_table must be 2D, got shape "
            f"{tuple(local_block_table.shape)}"
        )

    tensors = (positions, batch_indices, local_block_table)
    if any(tensor.dtype != positions.dtype for tensor in tensors[1:]):
        raise TypeError(
            "positions, batch_indices, and local_block_table must have the same dtype"
        )
    if positions.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            "page-RR metadata must use an integer dtype (torch.int32 or torch.int64), "
            f"got {positions.dtype}"
        )
    if any(tensor.device != positions.device for tensor in tensors[1:]):
        raise ValueError(
            "positions, batch_indices, and local_block_table must be on the same device"
        )


def build_mla_page_rr_slot_mapping(
    positions: torch.Tensor,
    batch_indices: torch.Tensor,
    local_block_table: torch.Tensor,
    page_tokens: int,
    shard_size: int,
    shard_rank: int,
) -> torch.Tensor:
    """Build owner-only MLA cache slots for a page-round-robin shard.

    ``positions`` are absolute positions within each request. Global page
    ``g`` belongs to rank ``g % shard_size`` and maps to local block-table
    column ``g // shard_size``. Non-owner rows receive ``-1``, which the
    existing MLA cache-write kernel treats as a no-op.

    The returned tensor is int64 because that is the cache-write kernel ABI.
    Runtime value checks use device-side asynchronous assertions so the valid
    CUDA path does not introduce host synchronization.
    """

    _validate_geometry(page_tokens, shard_size, shard_rank)
    _validate_slot_mapping_inputs(positions, batch_indices, local_block_table)

    positions_i64 = positions.to(torch.int64)
    batch_indices_i64 = batch_indices.to(torch.int64)
    skipped = torch.full_like(positions_i64, -1)
    if positions_i64.numel() == 0:
        return skipped

    torch._assert_async(torch.all(positions_i64 >= 0), "positions must be non-negative")

    request_count = int(local_block_table.shape[0])
    if request_count == 0:
        raise RuntimeError("batch index out of range for an empty local block table")
    batch_in_range = (batch_indices_i64 >= 0) & (batch_indices_i64 < request_count)
    torch._assert_async(torch.all(batch_in_range), "batch index out of range")
    safe_batch_indices = torch.clamp(batch_indices_i64, 0, request_count - 1)

    global_pages = torch.div(positions_i64, page_tokens, rounding_mode="floor")
    owner_rows = torch.remainder(global_pages, shard_size) == shard_rank
    local_pages = torch.div(global_pages, shard_size, rounding_mode="floor")

    table_width = int(local_block_table.shape[1])
    if table_width == 0:
        torch._assert_async(
            torch.all(~owner_rows),
            "local block table width is insufficient for an owner page",
        )
        return skipped

    page_in_range = (local_pages >= 0) & (local_pages < table_width)
    torch._assert_async(
        torch.all((~owner_rows) | page_in_range),
        "local block table width is insufficient for an owner page",
    )
    safe_local_pages = torch.clamp(local_pages, 0, table_width - 1)
    block_ids = local_block_table[safe_batch_indices, safe_local_pages].to(torch.int64)
    torch._assert_async(
        torch.all((~owner_rows) | (~page_in_range) | (block_ids > 0)),
        "owner page points to a null or unused local block",
    )

    valid_owner_rows = owner_rows & batch_in_range & page_in_range & (block_ids > 0)
    slots = block_ids * page_tokens + torch.remainder(positions_i64, page_tokens)
    return torch.where(valid_owner_rows, slots, skipped)


@dataclass(frozen=True)
class MlaPageRRBatchDescriptor:
    """Rank-independent layout of one invocation's raw prefix pages."""

    prefix_lens: tuple[int, ...]
    local_page_offsets: tuple[int, ...]
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


def _host_int_values(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = []
    for value in values:
        if not isinstance(value, Integral):
            raise TypeError(f"{name} must contain integers, got {value!r}")
        result.append(int(value))
    return tuple(result)


def _prefix_descriptor(
    prefix_lens: tuple[int, ...],
    *,
    page_tokens: int,
    shard_size: int,
    feature_width: int,
) -> MlaPageRRBatchDescriptor:
    if not prefix_lens:
        raise ValueError("prefix_lens must be non-empty")
    if any(prefix_len < 0 for prefix_len in prefix_lens):
        raise ValueError(f"prefix_lens must be non-negative, got {prefix_lens}")
    stripe_tokens = page_tokens * shard_size
    local_page_offsets = [0]
    for prefix_len in prefix_lens:
        count = (prefix_len + stripe_tokens - 1) // stripe_tokens
        local_page_offsets.append(local_page_offsets[-1] + count)
    return MlaPageRRBatchDescriptor(
        prefix_lens=prefix_lens,
        local_page_offsets=tuple(local_page_offsets),
        page_tokens=page_tokens,
        shard_size=shard_size,
        feature_width=feature_width,
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
    descriptor: MlaPageRRBatchDescriptor,
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

    if int(local_block_table.shape[0]) != descriptor.batch_size:
        raise ValueError(
            "local block table batch does not match prefix_lens: "
            f"table={local_block_table.shape[0]} prefix={descriptor.batch_size}"
        )

    output_shape = (
        descriptor.total_local_pages,
        page_tokens,
        descriptor.feature_width,
    )
    if descriptor.total_local_pages == 0:
        return raw_cache.new_empty(output_shape)

    from .mla_page_rr_prefix_kernels import pack_prefix_cuda

    payload = raw_cache.new_empty(output_shape)
    pack_prefix_cuda(raw_cache, local_block_table, descriptor, shard_rank, payload)
    return payload


def _restore_mla_page_rr_prefix(
    gathered_payload: torch.Tensor,
    descriptor: MlaPageRRBatchDescriptor,
) -> torch.Tensor:
    """Remove rank/stripe padding into newly allocated request-major rows."""
    output = gathered_payload.new_empty(
        (sum(descriptor.prefix_lens), descriptor.feature_width)
    )
    if output.shape[0]:
        from .mla_page_rr_prefix_kernels import restore_prefix_cuda

        restore_prefix_cuda(gathered_payload, descriptor, output)
    return output


@dataclass(frozen=True)
class MlaPageRRCacheAdapter:
    """Translate between rank-local page-RR storage and canonical MLA rows."""

    page_tokens: int
    shard_size: int
    shard_rank: int
    _prefix_descriptor: Optional[MlaPageRRBatchDescriptor] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_geometry(self.page_tokens, self.shard_size, self.shard_rank)

    def slot_mapping(
        self,
        positions: torch.Tensor,
        batch_indices: torch.Tensor,
        local_block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Return owner-only physical slots; non-owner rows map to ``-1``."""

        return build_mla_page_rr_slot_mapping(
            positions,
            batch_indices,
            local_block_table,
            self.page_tokens,
            self.shard_size,
            self.shard_rank,
        )

    def validate_block_table_capacity(
        self,
        local_block_table: torch.Tensor,
        sequence_lens: Sequence[int],
    ) -> None:
        """Fail before execution if this rank cannot store its owned pages."""

        sequence_values = _host_int_values("sequence_lens", sequence_lens)
        if local_block_table.dim() != 2:
            raise ValueError("local block table must be 2D")
        if int(local_block_table.shape[0]) != len(sequence_values):
            raise ValueError(
                "local block table batch does not match sequence lengths: "
                f"table={local_block_table.shape[0]} sequences={len(sequence_values)}"
            )
        if any(length < 0 for length in sequence_values):
            raise ValueError("sequence lengths must be non-negative")
        table_width = int(local_block_table.shape[1])
        max_pages = (
            max(sequence_values, default=0) + self.page_tokens - 1
        ) // self.page_tokens
        required_local_pages = (
            max_pages + self.shard_size - 1 - self.shard_rank
        ) // self.shard_size
        if required_local_pages > table_width:
            raise RuntimeError(
                "MLA page-RR sequence exceeds the rank-local block table: "
                f"required={required_local_pages} width={table_width} "
                f"shard_size={self.shard_size} shard_rank={self.shard_rank}"
            )

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

    def _pack_prefix(
        self,
        raw_cache: torch.Tensor,
        local_block_table: torch.Tensor,
        prefix_lens: Sequence[int],
    ) -> torch.Tensor:
        # The request's attention wrapper reuses this adapter across MLA layers.
        # Retain only the latest batch, without any process-global metadata cache.
        _validate_raw_cache(raw_cache, self.page_tokens)
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


__all__ = ["MlaPageRRCacheAdapter", "build_mla_page_rr_slot_mapping"]
