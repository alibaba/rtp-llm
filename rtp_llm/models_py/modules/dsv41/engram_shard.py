# SPDX-License-Identifier: Apache-2.0
"""Shard layout for distributed Engram storage over EGM.

The Engram table is split into ``shard_count`` contiguous row ranges, one per
holder process.  Every consumer maps all shards into **two** contiguous virtual
address ranges (weights and scales), so the existing lookup kernel keeps using
``weight + row * dim`` and ``scales + row * (dim // block)`` unchanged.

For that trick to work both the per-shard weight region and the per-shard scale
region must start at a CUDA allocation-granularity boundary.  ``shard_rows`` is
therefore rounded up so that

    shard_rows * dim            % granularity == 0
    shard_rows * (dim // block) % granularity == 0

which for dim=256, block=32, granularity=2 MiB means ``shard_rows`` must be a
multiple of 262144 rows.  The padding is at most one alignment unit per shard
(64 MiB of weights) and is never addressed by a lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

DEFAULT_GRANULARITY = 2 * 1024 * 1024
DEFAULT_BLOCK = 32
DEFAULT_SHARD_COUNT = 4


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def _round_up(value: int, unit: int) -> int:
    if unit <= 0:
        raise ValueError("alignment unit must be positive")
    return -(-value // unit) * unit


@dataclass(frozen=True)
class ShardPlan:
    """Immutable geometry of one sharded Engram tensor."""

    total_rows: int
    shard_count: int
    shard_rows: int
    dim: int
    block: int
    granularity: int

    @property
    def weight_bytes(self) -> int:
        return self.shard_rows * self.dim

    @property
    def scale_bytes(self) -> int:
        return self.shard_rows * (self.dim // self.block)

    @property
    def shard_bytes(self) -> int:
        return self.weight_bytes + self.scale_bytes

    @property
    def padded_rows(self) -> int:
        return self.shard_rows * self.shard_count

    @property
    def weight_span(self) -> int:
        """Size of the concatenated weight VA range."""
        return self.weight_bytes * self.shard_count

    @property
    def scale_span(self) -> int:
        """Size of the concatenated scale VA range."""
        return self.scale_bytes * self.shard_count

    def shard_of_row(self, row: int) -> int:
        if type(row) is not int or not 0 <= row < self.total_rows:
            raise IndexError(f"row {row} outside logical table of {self.total_rows}")
        return row // self.shard_rows

    def row_range(self, shard: int) -> tuple[int, int]:
        """Half-open *real* row range owned by ``shard`` (padding excluded)."""
        if type(shard) is not int or not 0 <= shard < self.shard_count:
            raise IndexError(f"shard {shard} outside 0..{self.shard_count - 1}")
        start = shard * self.shard_rows
        end = min(start + self.shard_rows, self.total_rows)
        return start, end

    def weight_offset(self, shard: int) -> int:
        """Byte offset of ``shard`` inside the concatenated weight VA."""
        if type(shard) is not int or not 0 <= shard < self.shard_count:
            raise IndexError(f"shard {shard} outside 0..{self.shard_count - 1}")
        return shard * self.weight_bytes

    def scale_offset(self, shard: int) -> int:
        """Byte offset of ``shard`` inside the concatenated scale VA."""
        if type(shard) is not int or not 0 <= shard < self.shard_count:
            raise IndexError(f"shard {shard} outside 0..{self.shard_count - 1}")
        return shard * self.scale_bytes

    def as_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "shard_count": self.shard_count,
            "shard_rows": self.shard_rows,
            "dim": self.dim,
            "block": self.block,
            "granularity": self.granularity,
            "weight_bytes": self.weight_bytes,
            "scale_bytes": self.scale_bytes,
            "shard_bytes": self.shard_bytes,
            "weight_span": self.weight_span,
            "scale_span": self.scale_span,
        }


def plan_shards(
    total_rows: int,
    dim: int,
    shard_count: int = DEFAULT_SHARD_COUNT,
    *,
    block: int = DEFAULT_BLOCK,
    granularity: int = DEFAULT_GRANULARITY,
) -> ShardPlan:
    """Compute the shard geometry, padding rows so every shard is mappable.

    Raises ``ValueError`` for geometries that cannot be mapped at a granularity
    boundary (for example a ``dim`` that is not a multiple of ``block``).
    """
    for name, value in (("total_rows", total_rows), ("dim", dim),
                        ("shard_count", shard_count), ("block", block),
                        ("granularity", granularity)):
        if type(value) is not int or not 0 < value < (1 << 63):
            raise ValueError(f"{name} must be a positive integer below 2**63")
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if dim <= 0 or dim % block:
        raise ValueError("dim must be a positive multiple of block")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if granularity <= 0 or granularity & (granularity - 1):
        raise ValueError("granularity must be a positive power of two")

    weight_unit = granularity // gcd(granularity, dim)
    scale_unit = granularity // gcd(granularity, dim // block)
    align_rows = _lcm(weight_unit, scale_unit)

    rows_per_shard = -(-total_rows // shard_count)
    shard_rows = _round_up(rows_per_shard, align_rows)
    if (shard_count - 1) * shard_rows >= total_rows:
        raise ValueError("shard_count too large: alignment would produce empty shards")
    if shard_rows * shard_count * dim >= (1 << 63):
        raise ValueError("shard address span exceeds 63-bit limit")
    plan = ShardPlan(
        total_rows=total_rows,
        shard_count=shard_count,
        shard_rows=shard_rows,
        dim=dim,
        block=block,
        granularity=granularity,
    )
    # Post-conditions the native mapping relies on.
    assert plan.weight_bytes % granularity == 0
    assert plan.scale_bytes % granularity == 0
    assert plan.shard_bytes % granularity == 0
    assert plan.padded_rows >= total_rows
    return plan


def split_rows(total_rows: int, plan: ShardPlan) -> list[tuple[int, int]]:
    """Real row ranges for every shard, in shard order."""
    if type(total_rows) is not int or total_rows != plan.total_rows:
        raise ValueError("total_rows does not match plan")
    return [plan.row_range(shard) for shard in range(plan.shard_count)]


def local_rows(plan: ShardPlan, shard_index: int) -> tuple[int, int]:
    """Rows the process at ``shard_index`` must load (``-1`` means none)."""
    if type(shard_index) is not int:
        raise ValueError("shard_index must be an integer")
    if shard_index == -1:
        return (0, 0)
    return plan.row_range(shard_index)


@dataclass(frozen=True)
class MappingRequest:
    """One ``cuMemMap`` call: map ``size`` bytes of a shard at ``va``."""

    va: int
    size: int
    handle_offset: int
    shard: int
    kind: str  # "weight" | "scale"


def mapping_requests(
    plan: ShardPlan, weight_base: int, scale_base: int
) -> list[MappingRequest]:
    """Map every shard's weight and scale region into the two VA spans.

    Weight and scale live in **separate** Owner allocations because
    ``cuMemMap`` on a fabric handle only accepts ``offset == 0`` (observed
    ``CUDA_ERROR_NOT_SUPPORTED`` for any non-zero offset).  ``weight_base`` and
    ``scale_base`` must be granularity aligned; the caller reserves
    ``plan.weight_span`` and ``plan.scale_span`` bytes respectively.
    """
    for name, base in (("weight_base", weight_base), ("scale_base", scale_base)):
        if base < 0 or base % plan.granularity:
            raise ValueError(f"{name} must be a non-negative granularity multiple")
    requests = []
    for shard in range(plan.shard_count):
        requests.append(
            MappingRequest(
                va=weight_base + plan.weight_offset(shard),
                size=plan.weight_bytes,
                handle_offset=0,
                shard=shard,
                kind="weight",
            )
        )
        requests.append(
            MappingRequest(
                va=scale_base + plan.scale_offset(shard),
                size=plan.scale_bytes,
                handle_offset=0,  # fabric handles cannot map a sub-range
                shard=shard,
                kind="scale",
            )
        )
    return requests
