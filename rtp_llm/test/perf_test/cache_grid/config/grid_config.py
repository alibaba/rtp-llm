"""Lightweight cache-grid loading and physical-geometry deduplication."""

import json
from typing import Any, Dict, List

from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
    CacheGridRunner,
    is_grouped_case,
    normalize_cache_case,
)


def resolve_cache_block_size(grid_payload: Any) -> int:
    """Read the single authoritative cache alignment in the grid."""
    if not isinstance(grid_payload, dict):
        raise ValueError("cache grid must be a JSON object")
    generator = grid_payload.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("grid requires generator.cache_alignment")
    sampling = generator.get("cache_sampling")
    if "cache_block_size" in grid_payload or (
        isinstance(sampling, dict) and "alignment" in sampling
    ):
        raise ValueError(
            "cache alignment must only be configured at generator.cache_alignment"
        )
    value = generator.get("cache_alignment")
    if type(value) is not int or value <= 0:
        raise ValueError("generator.cache_alignment must be a positive integer")
    return value


def load_cache_grid_cases(path: str) -> List[Dict[str, int]]:
    """Load current explicit grids without generating or filling case fields."""
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    resolve_cache_block_size(config)
    if type(config.get("schema_version")) is not int or config["schema_version"] != 2:
        raise ValueError("cache grid requires schema_version=2")
    if "cases" not in config:
        raise ValueError("cache grid requires explicit cases")
    raw_cases = config["cases"]
    if not isinstance(raw_cases, list):
        raise ValueError("cache grid cases must be a list")
    cases: List[Dict[str, int]] = []
    seen = set()
    seen_ids = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"cache grid case {index} must be an object")
        case = normalize_cache_case(raw)
        key = CacheGridRunner.case_key(case)
        if key in seen:
            raise ValueError(f"duplicate cache grid case: {case}")
        seen.add(key)
        if case["case_id"] in seen_ids:
            raise ValueError(f"duplicate cache grid case_id: {case['case_id']}")
        seen_ids.add(case["case_id"])
        cases.append(case)
    if not cases:
        raise ValueError(f"cache grid {path} contains no cases")
    return cases


def dedupe_cache_grid_cases(
    cases: List[Dict[str, int]], block_size: int
) -> List[Dict[str, int]]:
    """Collapse cases whose requested cache lengths share one physical bucket."""
    buckets: Dict[tuple, Dict[str, int]] = {}
    order: List[tuple] = []
    for case in cases:
        if is_grouped_case(case):
            key = (CacheGridRunner.case_key(case),)
            if key not in buckets:
                buckets[key] = case
                order.append(key)
            continue
        key = (case["batch_size"], case["input_len"], case["cache_len"] // block_size)
        kept = buckets.get(key)
        if kept is None:
            buckets[key] = case
            order.append(key)
        elif kept["cache_len"] % block_size and not case["cache_len"] % block_size:
            buckets[key] = case
        elif (
            kept["cache_len"] % block_size
            and case["cache_len"] % block_size
            and case["cache_len"] > kept["cache_len"]
        ):
            buckets[key] = case
    return [buckets[key] for key in order]


# Preserve the historical helper names used by batch_decode_test callers.
_load_cache_grid_cases = load_cache_grid_cases
_resolve_cache_block_size = resolve_cache_block_size
_dedupe_cache_grid_cases = dedupe_cache_grid_cases
