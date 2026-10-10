#!/usr/bin/env python3
"""Generate a reproducible, block-aligned prefix-cache performance grid.

Input lengths are aligned to ``--alignment`` (128 by default).  Cache lengths
should instead be aligned to the engine's physical reuse granularity
(``--cache-alignment``): the prefix cache only reuses whole blocks, so two
requested cache lengths that floor onto the same block bucket measure the
identical geometry and waste one seed plus ``--measure-runs`` full prefills.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import cache_grid_section
from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    fingerprint as profile_fingerprint,
)
from rtp_llm.test.perf_test.cache_grid.config.perf_profile import load_profile

DEFAULT_SEED = 104729  # A fixed prime, recorded in every generated plan.
DEFAULT_BOUNDARIES = (
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    786432,
    1048448,
    1048575,
)


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def comma_separated_ints(value: str) -> list[int]:
    """Parse one non-empty comma-separated integer argument."""
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers, for example 4096,8192"
        )
    try:
        return [int(part) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers, for example 4096,8192"
        ) from error


def _stratified_pick(values: list[int], count: int, seed: int) -> list[int]:
    if count >= len(values):
        return values
    rng = _stable_rng(seed, "input")
    picked = []
    for index in range(count):
        lo = index * len(values) // count
        hi = (index + 1) * len(values) // count
        picked.append(values[rng.randrange(lo, max(lo + 1, hi))])
    return sorted(set(picked))


def generate_input_lengths(
    minimum: int,
    maximum: int,
    alignment: int,
    count: int,
    mode: str,
    seed: int,
    boundaries: Iterable[int] = DEFAULT_BOUNDARIES,
) -> list[int]:
    if minimum <= 0 or maximum < minimum or alignment <= 0:
        raise ValueError("invalid input length bounds or alignment")
    first = ((minimum + alignment - 1) // alignment) * alignment
    aligned = list(range(first, maximum + 1, alignment))
    if not aligned:
        raise ValueError("input range contains no aligned value")
    forced = {x for x in boundaries if minimum <= x <= maximum}
    if mode == "stride":
        selected_set = set(aligned)
    else:
        if count < len(forced):
            raise ValueError(
                f"input point count {count} is smaller than {len(forced)} boundaries"
            )
        candidates = [x for x in aligned if x not in forced]
        selected_set = set(_stratified_pick(candidates, count - len(forced), seed))
    selected_set.update(forced)
    return sorted(selected_set)


def generate_cache_lengths(
    input_len: int,
    alignment: int,
    points: int,
    ratio_points: int,
    seed: int,
) -> list[int]:
    if points < 2 or ratio_points < 0 or ratio_points > points - 2:
        raise ValueError("cache points require two boundaries and valid ratio_points")
    max_cache = align_down(input_len - 1, alignment)
    if max_cache <= 0:
        return [0]
    result = {0, max_cache}
    interior_count = points - 2
    compute_points = interior_count - ratio_points
    rng = _stable_rng(seed, f"cache:{input_len}")

    def add_strata(count: int, reverse: bool) -> None:
        for index in range(count):
            # Pick an aligned index from the interior of each equal-width
            # stratum. Using indices avoids tokenizer/block-size assumptions.
            lo = 1 + index * max(1, max_cache // alignment - 1) // count
            hi = 1 + (index + 1) * max(1, max_cache // alignment - 1) // count
            cache_index = rng.randrange(lo, max(lo + 1, hi))
            value = min(cache_index * alignment, max_cache)
            result.add(max_cache - value if reverse else value)

    if ratio_points:
        add_strata(ratio_points, False)
    if compute_points:
        add_strata(compute_points, True)
    return sorted(x for x in result if 0 <= x < input_len)


def generate_fixed_cache_lengths(
    *,
    explicit: list[int],
    random_count: int | None,
    minimum: int,
    maximum: int,
    alignment: int,
    seed: int,
) -> list[int]:
    """Resolve explicit or reproducibly sampled cache lengths."""
    if alignment <= 0 or minimum < 0 or maximum < minimum:
        raise ValueError("invalid fixed-cache bounds or alignment")
    if explicit and random_count is not None:
        raise ValueError(
            "--fixed-cache-len and --random-cache-count are mutually exclusive"
        )
    if not explicit and random_count is None:
        raise ValueError(
            "fixed-cache-sweep requires --fixed-cache-len or --random-cache-count"
        )

    if explicit:
        if len(explicit) != len(set(explicit)):
            raise ValueError("--fixed-cache-len values must be unique")
        for value in explicit:
            if value < minimum or value > maximum:
                raise ValueError(
                    f"fixed cache length {value} is outside [{minimum}, {maximum}]"
                )
            if value % alignment:
                raise ValueError(
                    f"fixed cache length {value} is not aligned to {alignment}"
                )
        return list(explicit)

    assert random_count is not None
    if random_count <= 0:
        raise ValueError("--random-cache-count must be positive")
    first = ((minimum + alignment - 1) // alignment) * alignment
    candidates = list(range(first, maximum + 1, alignment))
    if random_count > len(candidates):
        raise ValueError(
            f"requested {random_count} random cache lengths, but only "
            f"{len(candidates)} aligned values exist in [{minimum}, {maximum}]"
        )
    rng = _stable_rng(seed, "fixed-cache-sweep")
    return sorted(rng.sample(candidates, random_count))


def build_fixed_cache_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Build fixed-cache slices while stepping the uncached compute length."""
    if args.batch_size <= 0 or args.measure_runs <= 0 or args.max_cases <= 0:
        raise ValueError("batch size, measure runs, and max cases must be positive")
    cache_alignment = getattr(args, "cache_alignment", None)
    if cache_alignment is None or cache_alignment <= 0:
        raise ValueError(
            "specify --cache-alignment or profile.cache_grid.cache_alignment explicitly"
        )
    raw_compute_steps = getattr(args, "compute_step", None)
    if isinstance(raw_compute_steps, int):
        compute_steps = [raw_compute_steps]
    else:
        compute_steps = list(raw_compute_steps or [])
    if not compute_steps or any(step <= 0 for step in compute_steps):
        raise ValueError("fixed-cache-sweep requires positive --compute-step values")
    if len(compute_steps) != len(set(compute_steps)):
        raise ValueError("--compute-step values must be unique")
    if compute_steps != sorted(compute_steps, reverse=True):
        raise ValueError("--compute-step values must be ordered coarse-to-fine")
    min_compute_len = getattr(args, "min_compute_len", None)
    if min_compute_len is None:
        min_compute_len = compute_steps[-1]
    if min_compute_len <= 0:
        raise ValueError("--min-compute-len must be positive")
    if args.max_input_len < min_compute_len:
        raise ValueError("--max-input-len is smaller than the minimum compute length")

    min_cache_len = getattr(args, "min_cache_len", 0)
    requested_max_cache = getattr(args, "max_cache_len", None)
    largest_usable_cache = args.max_input_len - min_compute_len
    max_cache_len = (
        largest_usable_cache if requested_max_cache is None else requested_max_cache
    )
    if max_cache_len > largest_usable_cache:
        raise ValueError(
            "--max-cache-len must leave room for --min-compute-len under "
            "--max-input-len"
        )
    cache_lengths = generate_fixed_cache_lengths(
        explicit=list(getattr(args, "fixed_cache_len", None) or []),
        random_count=getattr(args, "random_cache_count", None),
        minimum=min_cache_len,
        maximum=max_cache_len,
        alignment=cache_alignment,
        seed=args.seed,
    )

    cases = []
    seen = set()
    for cache_len in cache_lengths:
        max_compute_len = args.max_input_len - cache_len
        for refinement_level, compute_step in enumerate(compute_steps):
            for compute_len in range(
                min_compute_len, max_compute_len + 1, compute_step
            ):
                input_len = cache_len + compute_len
                geometry = (args.batch_size, input_len, cache_len)
                if geometry in seen:
                    continue
                seen.add(geometry)
                cases.append(
                    {
                        "case_id": len(cases),
                        "batch_size": args.batch_size,
                        "input_len": input_len,
                        "cache_len": cache_len,
                        "refinement_level": refinement_level,
                        "compute_step": compute_step,
                    }
                )
    for case in cases:
        if case["batch_size"] > 1:
            case["prefix_policy"] = "independent"
            case["request_groups"] = [
                {
                    "count": case["batch_size"],
                    "input_len": case["input_len"],
                    "cache_len": case["cache_len"],
                }
            ]
    if len(cases) > args.max_cases and not args.allow_large_grid:
        raise ValueError(
            f"generated {len(cases)} cases, exceeding --max-cases={args.max_cases}; "
            "reduce sampling or pass --allow-large-grid"
        )
    return {
        "schema_version": 2,
        "kind": "fixed_cache_compute_sweep",
        "generator": {
            "name": "fixed_cache_compute_sweep",
            "version": 2,
            "seed": args.seed,
            "cache_alignment": cache_alignment,
            "cache_sampling": {
                "mode": (
                    "explicit" if getattr(args, "fixed_cache_len", None) else "random"
                ),
                "values": cache_lengths,
                "requested_random_count": getattr(args, "random_cache_count", None),
                "min": min_cache_len,
                "max": max_cache_len,
            },
            "compute_sampling": {
                "min": min_compute_len,
                "steps": compute_steps,
                "max_input_len": args.max_input_len,
            },
        },
        "summary": {
            "cache_count": len(cache_lengths),
            "input_count": len({case["input_len"] for case in cases}),
            "case_count": len(cases),
            "estimated_measure_requests": len(cases) * args.measure_runs,
            "estimated_seed_requests": sum(c["cache_len"] > 0 for c in cases),
        },
        "cases": cases,
    }


def build_grid(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "grid_mode", "stratified") == "fixed-cache-sweep":
        return build_fixed_cache_sweep(args)
    if args.batch_size <= 0 or args.measure_runs <= 0 or args.max_cases <= 0:
        raise ValueError("batch size, measure runs, and max cases must be positive")
    inputs = generate_input_lengths(
        args.min_input_len,
        args.max_input_len,
        args.alignment,
        args.input_points,
        args.input_mode,
        args.seed,
    )
    cache_alignment = getattr(args, "cache_alignment", None)
    if cache_alignment is None or cache_alignment <= 0:
        raise ValueError(
            "specify --cache-alignment or profile.cache_grid.cache_alignment explicitly"
        )
    cases = []
    for input_len in inputs:
        cache_lengths = generate_cache_lengths(
            input_len,
            cache_alignment,
            args.cache_points_per_input,
            args.cache_ratio_points,
            args.seed,
        )
        for cache_len in cache_lengths:
            cases.append(
                {
                    "case_id": len(cases),
                    "batch_size": args.batch_size,
                    "input_len": input_len,
                    "cache_len": cache_len,
                }
            )
    for case in cases:
        if case["batch_size"] > 1:
            case["prefix_policy"] = "independent"
            case["request_groups"] = [
                {
                    "count": case["batch_size"],
                    "input_len": case["input_len"],
                    "cache_len": case["cache_len"],
                }
            ]
    if len(cases) > args.max_cases and not args.allow_large_grid:
        raise ValueError(
            f"generated {len(cases)} cases, exceeding --max-cases={args.max_cases}; "
            "reduce sampling or pass --allow-large-grid"
        )
    generator = {
        "name": "aligned_stratified_cache_grid",
        "cache_alignment": cache_alignment,
        "version": 1,
        "alignment": args.alignment,
        "seed": args.seed,
        "input_sampling": {
            "mode": args.input_mode,
            "min": args.min_input_len,
            "max": args.max_input_len,
            "requested_count": args.input_points,
        },
        "cache_sampling": {
            "points_per_input": args.cache_points_per_input,
            "ratio_points": args.cache_ratio_points,
            "compute_points": args.cache_points_per_input - 2 - args.cache_ratio_points,
            "include_cold": True,
            "include_near_full": True,
        },
        "unaligned_boundary_exceptions": [
            value for value in inputs if value % args.alignment
        ],
    }
    return {
        "schema_version": 2,
        "kind": "aligned_stratified_cache_grid",
        "generator": generator,
        "summary": {
            "input_count": len(inputs),
            "case_count": len(cases),
            "estimated_measure_requests": len(cases) * args.measure_runs,
            "estimated_seed_requests": sum(c["cache_len"] > 0 for c in cases),
        },
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--grid-mode",
        choices=("stratified", "fixed-cache-sweep"),
        default="stratified",
    )
    parser.add_argument("--min-input-len", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1048575)
    parser.add_argument(
        "--alignment",
        type=int,
        default=128,
        help="Alignment in tokens for sampled input lengths",
    )
    parser.add_argument(
        "--cache-alignment",
        type=int,
        default=None,
        help=(
            "Required cache alignment from CLI or profile. "
            "Set to the engine's physical reuse granularity "
            "(--seq_size_per_block, multiplied by CP size when "
            "PREFILL_CP_KV_CACHE_SHARDED=1) so every cache point is exactly "
            "reusable and no two points collapse onto the same block bucket."
        ),
    )
    parser.add_argument(
        "--input-mode", choices=("stratified", "stride"), default="stratified"
    )
    parser.add_argument("--input-points", type=int, default=1024)
    parser.add_argument("--cache-points-per-input", type=int, default=16)
    parser.add_argument("--cache-ratio-points", type=int, default=7)
    parser.add_argument(
        "--fixed-cache-len",
        type=comma_separated_ints,
        help=(
            "Comma-separated fixed cache lengths for fixed-cache-sweep. "
            "Mutually exclusive with --random-cache-count."
        ),
    )
    parser.add_argument(
        "--random-cache-count",
        type=int,
        default=None,
        help="Reproducibly sample this many aligned cache lengths without replacement",
    )
    parser.add_argument("--min-cache-len", type=int, default=0)
    parser.add_argument("--max-cache-len", type=int, default=None)
    parser.add_argument(
        "--compute-step",
        type=comma_separated_ints,
        default=None,
        help=(
            "Comma-separated compute-length steps ordered coarse-to-fine for "
            "fixed-cache-sweep"
        ),
    )
    parser.add_argument(
        "--min-compute-len",
        type=int,
        default=None,
        help="First compute length; defaults to the finest --compute-step",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-cases", type=int, default=20000)
    parser.add_argument("--allow-large-grid", action="store_true")
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help=(
            "JSON profile for parameter defaults and downstream embedding. "
            "explicit --cache-alignment overrides profile cache_grid.cache_alignment; "
            "engine.max_seq_len is informational only."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = None
    profile_sha256 = None
    if args.profile is not None:
        profile = load_profile(args.profile)
        profile_sha256 = profile_fingerprint(profile)
        cache_grid = cache_grid_section(profile)
        profile_cache_alignment = cache_grid.get("cache_alignment")
        if args.cache_alignment is None and profile_cache_alignment is not None:
            args.cache_alignment = int(profile_cache_alignment)
    if args.cache_alignment is None:
        args.cache_alignment = 0
    payload = build_grid(args)
    if profile is not None:
        payload["profile"] = profile
        payload["profile_sha256"] = profile_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
