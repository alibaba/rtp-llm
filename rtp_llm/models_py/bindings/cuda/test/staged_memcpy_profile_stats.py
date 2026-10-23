"""Validate the diagnostic subset and attribute host payload memcpy wall time."""

import collections
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from crc_copy_benchmark_stats import (
    DIRECTIONS,
    TILES,
    VARIANTS,
    exact,
    expected_geometry,
    require,
    summarize,
)

BLOCKS = (1, 8, 32)
PROFILED_VARIANTS = ("staged_no_crc", "main_staged")
FIELDS = ("direction", "layout", "blocks", "variant")


def key(row):
    return tuple(row[field] for field in FIELDS)


def read_run(path, settings, repeat):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    require(
        rows and exact(rows[-1], {"type": "complete", "success": True}),
        "Incomplete profile",
    )
    meta = rows[0]
    expected = dict(
        type="metadata",
        implementation="staged_memcpy_profile_v2",
        profiled_variants=list(PROFILED_VARIANTS),
        staged_host_payload_copy="direct_dma",
        main_staged_host_payload_copy="cpu_pack_unpack",
        profiled=True,
        block_counts=list(BLOCKS),
        variants=list(VARIANTS),
        repeat=repeat,
        seed=settings["seed"] + repeat,
        iterations=settings["iterations"],
        warmup=settings["warmup"],
        exclude_1d_h2d=False,
        correctness_only=False,
        measured_h2d_verified=True,
        timing="wall",
        regime="cold",
        evict_multiplier=8,
        seq_size_per_block=128,
        cp_size=4,
        tp_size=4,
        cp_mode="CP_RR",
        kv_cache_sharded=True,
        gen_num_per_cycle=0,
    )
    for field in ("source_commit", "base_commit", "binary_sha256"):
        expected[field] = settings[field]
    for field, value in expected.items():
        require(exact(meta.get(field), value), f"Metadata mismatch: {field}")
    matrix = set(itertools.product(DIRECTIONS, TILES, BLOCKS, VARIANTS))
    rounds = settings["iterations"]
    expected_counts = dict(
        metadata=1,
        layout=2,
        mixed_crc_selftest=1,
        correctness=len(matrix),
        crc_corruption=8,
        sample=len(matrix) * rounds,
        staged_memcpy_profile=len(matrix)
        * rounds
        * len(PROFILED_VARIANTS)
        // len(VARIANTS),
        complete=1,
    )
    require(
        collections.Counter(row["type"] for row in rows)
        == collections.Counter(expected_counts),
        "Profile record count mismatch",
    )
    layouts = {row["layout"]: row for row in rows if row["type"] == "layout"}
    require(set(layouts) == set(TILES), "Missing layout")
    for layout, tiles in TILES.items():
        shape = layouts[layout]
        payload = sum(tiles)
        require(
            shape["payload_bytes"] == payload
            and shape["host_stride"] == (payload + 19) // 16 * 16
            and shape["tile_bytes"] == tiles
            and shape["geometry"] == expected_geometry(layout),
            "Wrong layout",
        )
        require(
            shape["host_pinned_verified"] is True
            and shape["source_device_verified"] is True,
            "Unverified memory placement",
        )
    checks = [row for row in rows if row["type"] == "correctness"]
    require(
        collections.Counter(map(key, checks)) == collections.Counter(matrix),
        "Correctness matrix mismatch",
    )
    require(
        all(
            row["success"] is True and exact(row["rotations_checked"], [0, 1])
            for row in checks
        ),
        "Correctness failed",
    )
    require(
        all(
            row["success"] is True
            for row in rows
            if row["type"] in ("mixed_crc_selftest", "crc_corruption")
        ),
        "CRC self-test failed",
    )
    samples, profiles, pairs = {}, {}, collections.defaultdict(list)
    for row in rows:
        if row["type"] not in ("sample", "staged_memcpy_profile"):
            continue
        require(
            key(row) in matrix
            and type(row["round"]) is int
            and 0 <= row["round"] < rounds
            and row["repeat"] == repeat,
            "Unexpected sample",
        )
        identity = key(row) + (row["round"],)
        destination = samples if row["type"] == "sample" else profiles
        require(identity not in destination, "Duplicate sample/profile")
        destination[identity] = row
        if row["type"] == "sample":
            require(
                row["timing"] == "wall"
                and row["regime"] == "cold"
                and math.isfinite(row["us"])
                and row["us"] > 0,
                "Invalid timing",
            )
            require(
                type(row["source_plan"]) is int
                and 0
                <= row["source_plan"]
                < math.ceil(
                    (layouts[row["layout"]]["pool_blocks"] - 1) / row["blocks"]
                ),
                "Invalid rotation",
            )
            pairs[key(row)[:3] + (row["round"],)].append(row)
    require(
        set(samples) == {case + (r,) for case in matrix for r in range(rounds)},
        "Missing sample",
    )
    require(
        set(profiles)
        == {identity for identity in samples if identity[3] in PROFILED_VARIANTS},
        "Missing or unexpected memcpy profile",
    )
    for group in pairs.values():
        require(
            {row["position"] for row in group} == set(range(len(VARIANTS)))
            and len({row["source_plan"] for row in group}) == 1,
            "Unpaired samples",
        )
    for identity, profile in profiles.items():
        sample = samples[identity]
        payload, blocks = sum(TILES[profile["layout"]]), profile["blocks"]
        require(
            all(
                profile[field] == sample[field]
                for field in ("position", "source_plan", "repeat")
            )
            and profile["wall_us"] == sample["us"],
            "Profile/sample association mismatch",
        )
        expected_calls = blocks if profile["variant"] == "main_staged" else 0
        require(
            type(profile["memcpy_calls"]) is int
            and profile["memcpy_calls"] == expected_calls
            and type(profile["memcpy_bytes"]) is int
            and profile["memcpy_bytes"] == expected_calls * payload
            and profile["payload_bytes"] == payload,
            "Payload memcpy interception is incomplete",
        )
        require(
            type(profile["memcpy_us"]) in (int, float)
            and math.isfinite(profile["memcpy_us"])
            and 0 <= profile["memcpy_us"] <= profile["wall_us"]
            and (
                (profile["memcpy_us"] > 0)
                if expected_calls
                else profile["memcpy_us"] == 0
            ),
            "Invalid memcpy time",
        )
    return meta, samples, profiles


def analyze(paths, settings):
    require(len(paths) == 2, "Exactly two process repeats required")
    inputs, cases = [], collections.defaultdict(list)
    for repeat, path in zip((80, 81), paths):
        meta, samples, profiles = read_run(path, settings, repeat)
        inputs.append(
            dict(
                path=str(path),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                metadata=meta,
            )
        )
        for identity in profiles:
            if identity[3] != "staged_no_crc":
                continue
            case = identity[:3]
            values = {
                variant: samples[case + (variant, identity[-1])] for variant in VARIANTS
            }
            per_variant = {
                variant: profiles[case + (variant, identity[-1])]
                for variant in PROFILED_VARIANTS
            }
            cases[case].append((repeat, per_variant, values))
    for field in (
        "gpu_uuid",
        "source_commit",
        "base_commit",
        "binary_sha256",
        "runtime",
        "driver",
        "main_staged_source",
    ):
        require(
            inputs[0]["metadata"][field] == inputs[1]["metadata"][field],
            f"Repeat mismatch: {field}",
        )
    result, warnings = [], []
    for (direction, layout, blocks), rows in sorted(cases.items()):
        case = dict(
            direction=direction,
            layout=layout,
            blocks=blocks,
            variants={
                v: summarize([values[v]["us"] for _, _, values in rows])
                for v in VARIANTS
            },
        )
        for variant in VARIANTS:
            stats = case["variants"][variant]
            stats["by_repeat"] = {
                repeat: summarize(
                    [values[variant]["us"] for rep, _, values in rows if rep == repeat]
                )
                for repeat in (80, 81)
            }
            medians = [stats["by_repeat"][repeat]["p50_us"] for repeat in (80, 81)]
            if max(medians) / min(medians) > 1.1:
                warnings.append(
                    dict(
                        direction=direction,
                        layout=layout,
                        blocks=blocks,
                        variant=variant,
                        reason="repeat p50 difference exceeds 10%",
                    )
                )
        case["profiles"] = {}
        for variant in PROFILED_VARIANTS:
            profile_stats = {}
            for name, getter in (
                ("memcpy", lambda p: p["memcpy_us"]),
                ("remaining", lambda p: p["wall_us"] - p["memcpy_us"]),
            ):
                profile_stats[name] = summarize(
                    [getter(p[variant]) for _, p, _ in rows]
                )
                profile_stats[name]["by_repeat"] = {
                    repeat: summarize(
                        [getter(p[variant]) for rep, p, _ in rows if rep == repeat]
                    )
                    for repeat in (80, 81)
                }
            profile_stats["memcpy_fraction_p50"] = statistics.median(
                p[variant]["memcpy_us"] / p[variant]["wall_us"] for _, p, _ in rows
            )
            case["profiles"][variant] = profile_stats
        result.append(case)
    return dict(
        valid=True,
        diagnostic=True,
        settings=settings,
        inputs=inputs,
        cases=result,
        sample_count=len(cases) * settings["iterations"] * 2 * len(VARIANTS),
        paired_round_count=len(cases) * settings["iterations"] * 2,
        profile_count=len(cases) * settings["iterations"] * 2 * len(PROFILED_VARIANTS),
        warnings=warnings,
        notes=[
            "This is an instrumented BS 1/8/32 diagnostic, not the full uninstrumented performance matrix.",
            "Payload memcpy is host wall time measured around the original libc calls.",
            "Optimized staged must issue zero payload-sized memcpy calls; main_staged must issue BS calls as a positive control.",
            "Remaining wall time includes CPU preparation, CUDA submission and completion; it is not pure GPU time.",
            "Instrumentation overhead remains in total wall time; no samples are excluded.",
        ],
    )


def write_results(result, output):
    output = Path(output)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = result["notes"] + [
        "",
        "| Layout | Direction | BS | Variant | Total p50 µs | Payload memcpy p50 µs | Remaining p50 µs | Median memcpy share |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        for variant in PROFILED_VARIANTS:
            profile = case["profiles"][variant]
            lines.append(
                f"| {case['layout']} | {case['direction']} | {case['blocks']} | {variant} | "
                f"{case['variants'][variant]['p50_us']:.2f} | {profile['memcpy']['p50_us']:.2f} | "
                f"{profile['remaining']['p50_us']:.2f} | {100 * profile['memcpy_fraction_p50']:.1f}% |"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
