"""Validated joint input-length/reuse sampling for synthetic prefill traffic.

Cell weighting and within-cell draws follow tools/request_length_distribution.py
from the source distribution generator. Bounds retain proportional cell mass.
"""

from __future__ import annotations

import bisect
import random
from dataclasses import dataclass
from typing import Any, List, Mapping

DISTRIBUTION_KIND = "conditional_request_lengths"
DISTRIBUTION_VERSION = 1


@dataclass(frozen=True)
class DistributionCell:
    input_bucket: int
    reuse_bin: int  # -1: no reuse; bin_count: complete reuse
    count: int
    input_min: int
    input_max: int


@dataclass(frozen=True)
class LengthDistribution:
    input_bucket_width: int
    reuse_ratio_bins: int
    source_rows: int
    incomplete_rows: int
    cells: List[DistributionCell]
    sources: List[dict]


def parse_length_distribution(payload: Mapping[str, Any]) -> LengthDistribution:
    """Validate decoded version-1 source data, raising ValueError on bad input."""
    if not isinstance(payload, Mapping) or (
        payload.get("kind") != DISTRIBUTION_KIND
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != DISTRIBUTION_VERSION
    ):
        raise ValueError("unsupported request length distribution")
    try:
        width = payload["input_bucket_width"]
        bins = payload["reuse_ratio_bins"]
        source_rows = payload["source_rows"]
        incomplete_rows = payload["incomplete_rows"]
        sources = payload.get("sources", [])
        cells = [DistributionCell(**cell) for cell in payload["cells"]]
    except (KeyError, TypeError) as error:
        raise ValueError("malformed request length distribution") from error
    if (
        type(width) is not int
        or width <= 0
        or type(bins) is not int
        or bins <= 0
        or type(source_rows) is not int
        or source_rows <= 0
        or type(incomplete_rows) is not int
        or incomplete_rows < 0
        or not cells
    ):
        raise ValueError("invalid request length distribution summary")
    for cell in cells:
        if any(type(value) is not int for value in vars(cell).values()) or (
            cell.count <= 0
            or cell.input_bucket < 0
            or not -1 <= cell.reuse_bin <= bins
            or not 0 <= cell.input_min <= cell.input_max
            or cell.input_min // width != cell.input_bucket
            or cell.input_max // width != cell.input_bucket
            or (cell.reuse_bin not in (-1, bins) and cell.input_min < 2)
        ):
            raise ValueError("invalid request length distribution cell")
    if sum(cell.count for cell in cells) != source_rows:
        raise ValueError("distribution cell counts do not match source_rows")
    if not isinstance(sources, list) or any(
        not isinstance(source, dict)
        or not isinstance(source.get("file"), str)
        or type(source.get("valid_rows")) is not int
        or type(source.get("incomplete_rows")) is not int
        or source["valid_rows"] < 0
        or source["incomplete_rows"] < 0
        for source in sources
    ):
        raise ValueError("invalid distribution sources")
    if sources and (
        sum(source["valid_rows"] for source in sources) != source_rows
        or sum(source["incomplete_rows"] for source in sources) != incomplete_rows
    ):
        raise ValueError("distribution source counts do not match summary")
    return LengthDistribution(width, bins, source_rows, incomplete_rows, cells, sources)


class LengthSampler:
    """Draw independent pairs with replacement from the bounded source mass."""

    def __init__(
        self, distribution: LengthDistribution, minimum: int, maximum: int, seed: int
    ):
        if (
            type(minimum) is not int
            or type(maximum) is not int
            or minimum < 0
            or maximum < minimum
        ):
            raise ValueError("invalid input length bounds")
        self._cells = []
        self._cumulative = []
        total = 0.0
        for cell in distribution.cells:
            lower = max(minimum, cell.input_min)
            upper = min(maximum, cell.input_max)
            if lower <= upper:
                fraction = (upper - lower + 1) / (cell.input_max - cell.input_min + 1)
                total += cell.count * fraction
                self._cells.append((cell, lower, upper))
                self._cumulative.append(total)
        if not self._cells:
            raise ValueError(
                "distribution has no samples inside the requested input length range"
            )
        self.eligible_source_fraction = total / distribution.source_rows
        self._total = total
        self._bins = distribution.reuse_ratio_bins
        self._rng = random.Random(seed)

    def draw(self) -> tuple[int, int]:
        index = min(
            bisect.bisect_right(self._cumulative, self._rng.random() * self._total),
            len(self._cells) - 1,
        )
        cell, lower, upper = self._cells[index]
        input_len = self._rng.randint(lower, upper)
        if cell.reuse_bin == -1:
            reuse_len = 0
        elif cell.reuse_bin == self._bins:
            reuse_len = input_len
        else:
            ratio = (cell.reuse_bin + self._rng.random()) / self._bins
            reuse_len = min(input_len - 1, max(1, round(input_len * ratio)))
        return input_len, reuse_len
