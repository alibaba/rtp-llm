"""Materialized decode distribution inputs for the unified planner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Tuple

from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchConfig,
    FixedBatchConfig,
    TargetTPOTConfig,
)
from rtp_llm.test.perf_test.errors import PlanningError


def _lengths(path: str, value: Any) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PlanningError(f"{path} must be a non-empty array")
    parsed = []
    for index, item in enumerate(value):
        if type(item) is not int or item <= 0:
            raise PlanningError(f"{path}[{index}] must be a positive integer")
        parsed.append(item)
    return tuple(parsed)


def parse_distribution_inputs(
    distribution: Any, batch: DecodeBatchConfig
) -> Tuple[Tuple[Tuple[int, ...], str], ...]:
    """Materialize policy-sized input prefixes or exact fixed-batch maps."""
    if isinstance(distribution, list):
        sequence_lengths = _lengths("distribution", distribution)
    elif isinstance(distribution, Mapping) and set(distribution) == {
        "sequence_lengths"
    }:
        sequence_lengths = _lengths(
            "distribution.sequence_lengths", distribution["sequence_lengths"]
        )
    else:
        sequence_lengths = ()
    if sequence_lengths:
        required = (
            max(batch.sizes) if isinstance(batch, FixedBatchConfig) else batch.max_size
        )
        if len(sequence_lengths) < required:
            raise PlanningError("distribution does not contain enough sequence lengths")
        sizes = (
            batch.sizes if isinstance(batch, FixedBatchConfig) else (batch.max_size,)
        )
        return tuple((sequence_lengths[:size], "sequence_lengths") for size in sizes)
    if not isinstance(distribution, Mapping) or set(distribution) != {
        "batch_seq_len_map"
    }:
        raise PlanningError(
            "distribution must contain sequence_lengths or batch_seq_len_map"
        )
    if isinstance(batch, TargetTPOTConfig):
        raise PlanningError("batch_seq_len_map cannot be used with target_tpot")
    raw_map = distribution["batch_seq_len_map"]
    expected_keys = {str(size) for size in batch.sizes}
    if not isinstance(raw_map, Mapping) or set(raw_map) != expected_keys:
        raise PlanningError(
            "distribution batch map keys must exactly match fixed batch sizes"
        )
    result = []
    for size in batch.sizes:
        lengths = _lengths(f"distribution.batch_seq_len_map.{size}", raw_map[str(size)])
        if len(lengths) != size:
            raise PlanningError(
                f"distribution batch {size} does not match its sequence count"
            )
        result.append((lengths, "batch_seq_len_map"))
    return tuple(result)
