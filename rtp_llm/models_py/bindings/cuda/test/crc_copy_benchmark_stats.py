"""Validate the complete copy benchmark matrix before reporting any timing."""

import collections
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

VARIANTS = (
    "integrated_crc",
    "copy1d_batch",
    "copy3d_batch",
    "main_staged",
    "gather_control",
)
DIRECTIONS = ("d2h", "h2d")
BLOCKS = tuple(range(1, 33))
TILES = {
    "full": [1152] * 31 + [19008] * 30 + [4224] * 30,
    "prefill_cp8_no_spec_swa": [9360] * 61 + [2048] * 30 + [8192] * 30,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            exact(actual[k], v) for k, v in expected.items()
        )
    return actual == expected


def case_key(record):
    require(
        type(record["blocks"]) is int and record["blocks"] in BLOCKS,
        "Invalid backing count",
    )
    return record["direction"], record["layout"], record["blocks"], record["variant"]


def active_variants(direction, exclude_1d_h2d):
    return tuple(
        v
        for v in VARIANTS
        if not (exclude_1d_h2d and direction == "h2d" and v == "copy1d_batch")
    )


def read_run(path, settings, repeat):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    require(
        records
        and records[0].get("type") == "metadata"
        and exact(records[-1], {"type": "complete", "success": True}),
        f"Incomplete benchmark: {path}",
    )
    meta = records[0]
    exclude = settings["exclude_1d_h2d"]
    expected_meta = {
        "implementation": "crc_copy_benchmark_v1",
        "variants": list(VARIANTS),
        "block_counts": list(BLOCKS),
        "repeat": repeat,
        "seed": settings["seed"] + repeat,
        "iterations": settings["iterations"],
        "warmup": settings["warmup"],
        "correctness_only": settings["correctness_only"],
        "exclude_1d_h2d": exclude,
        "evict_multiplier": 8,
        "timing": "wall",
        "regime": "cold",
        "cpu_metadata": "warm",
        "layout_order": "production",
        "local_backing_count": True,
        "seq_size_per_block": 128,
        "cp_size": 8,
        "tp_size": 8,
        "cp_mode": "CP_RR",
        "gen_num_per_cycle": 0,
        "fallback_allowed": False,
    }
    for key, value in expected_meta.items():
        require(
            exact(meta.get(key), value),
            f"Metadata mismatch: {key}",
        )
    for key in ("l2_bytes", "driver", "runtime", "sm"):
        require(type(meta[key]) is int and meta[key] > 0, f"Invalid metadata: {key}")
    for key in (
        "boundary",
        "production_source",
        "gather_control",
        "copy3d_stream",
        "copy3d_submit_mutex",
        "host_input",
        "h2d_host_payload",
    ):
        require(
            isinstance(meta.get(key), str) and bool(meta[key]),
            f"Missing measurement contract: {key}",
        )
    require(meta["driver"] >= 13000 and meta["runtime"] >= 13000, "CUDA 13 required")
    require(bool(meta["gpu_uuid"]), "Missing GPU identity")
    exclusions = meta.get("excluded_cases", [])
    require(len(exclusions) == int(exclude), "Unexpected exclusion metadata")
    if exclude:
        require(
            exclusions[0]["direction"] == "h2d"
            and exclusions[0]["variant"] == "copy1d_batch"
            and bool(exclusions[0].get("reason")),
            "Exclusion must explicitly identify 1D H2D",
        )

    matrix = {
        (d, layout, n, v)
        for d in DIRECTIONS
        for layout in TILES
        for n in BLOCKS
        for v in active_variants(d, exclude)
    }
    rounds = 0 if settings["correctness_only"] else settings["iterations"]
    expected_counts = {
        "metadata": 1,
        "layout": 2,
        "mixed_crc_selftest": 1,
        "correctness": len(matrix),
        "crc_corruption": 8,
        "sample": len(matrix) * rounds,
        "complete": 1,
    }
    if exclude:
        expected_counts["excluded_correctness"] = 64
    counts = collections.Counter(r["type"] for r in records)
    require(counts == collections.Counter(expected_counts), f"Record counts: {counts}")
    layouts = {r["layout"]: r for r in records if r["type"] == "layout"}
    require(set(layouts) == set(TILES), "Layout coverage")
    for layout, tiles in TILES.items():
        shape = layouts[layout]
        payload = sum(tiles)
        encoded = (payload + 19) // 16 * 16
        expected = {
            "tile_bytes": tiles,
            "tiles": len(tiles),
            "payload_bytes": payload,
            "encoded_bytes": encoded,
            "host_stride": (encoded + 4095) // 4096 * 4096,
            "staging_stride": 878176,
            "host_pinned_verified": True,
            "source_device_verified": True,
            "copy3d_operations_per_backing": 3,
        }
        for key, value in expected.items():
            require(exact(shape.get(key), value), f"Invalid {layout} shape: {key}")
        for key in ("source_bytes", "destination_bytes", "pool_blocks", "evict_bytes"):
            require(
                type(shape[key]) is int and shape[key] > 0, f"Invalid pool field: {key}"
            )
        require(
            shape["pool_blocks"] >= 128 and shape["evict_bytes"] % 16 == 0,
            "Insufficient pool or unaligned eviction capacity",
        )
        require(
            shape["source_bytes"]
            == shape["destination_bytes"]
            == shape["pool_blocks"] * payload,
            "Pool capacity mismatch",
        )
        require(
            shape["source_bytes"] >= 8 * meta["l2_bytes"]
            and shape["evict_bytes"] >= 8 * meta["l2_bytes"],
            "Pool/eviction buffer smaller than 8x L2",
        )
    checks = [r for r in records if r["type"] == "correctness"]
    key = case_key
    require(
        collections.Counter(map(key, checks)) == collections.Counter(matrix),
        "Correctness matrix",
    )
    for record in checks:
        require(
            record["success"] is True and exact(record["rotations_checked"], [0, 1]),
            "Correctness check did not pass",
        )
    excluded = [r for r in records if r["type"] == "excluded_correctness"]
    excluded_matrix = (
        {("h2d", layout, n, "copy1d_batch") for layout in TILES for n in BLOCKS}
        if exclude
        else set()
    )
    require(
        collections.Counter(map(key, excluded)) == collections.Counter(excluded_matrix),
        "Exclusion matrix",
    )
    for record in excluded:
        require(
            bool(record.get("reason")) and "success" not in record,
            "Exclusion cannot claim success",
        )
    mixed = [r for r in records if r["type"] == "mixed_crc_selftest"]
    require(
        exact(
            mixed,
            [
                {
                    "type": "mixed_crc_selftest",
                    "variant": "integrated_crc",
                    "success": True,
                }
            ],
        ),
        "Mixed CRC self-test failed",
    )
    corruption = [r for r in records if r["type"] == "crc_corruption"]
    require(
        collections.Counter((r["layout"], r["blocks"]) for r in corruption)
        == collections.Counter(itertools.product(TILES, (1, 8, 16, 32))),
        "Corruption matrix",
    )
    for record in corruption:
        case_key(record)
        require(
            record["variant"] == "integrated_crc"
            and record["direction"] == "h2d"
            and exact(record["cases"], 6)
            and exact(record["corruption_cases"], 6),
            "Invalid corruption check",
        )
        for field in (
            "success",
            "all_targets_unchanged",
            "recovery_success",
            "good_bad_good",
            "entire_pool_guards",
        ):
            require(record[field] is True, f"CRC check failed: {field}")

    pairs = collections.defaultdict(dict)
    for record in records:
        if record["type"] != "sample":
            continue
        require(key(record) in matrix, "Unexpected or excluded sample")
        require(
            type(record["repeat"]) is int and record["repeat"] == repeat,
            "Sample from wrong repeat",
        )
        require(
            record["timing"] == "wall" and record["regime"] == "cold",
            "Wrong timing method",
        )
        require(
            type(record["round"]) is int and 0 <= record["round"] < rounds,
            "Invalid round",
        )
        require(
            type(record["position"]) is int
            and record["position"]
            in range(len(active_variants(record["direction"], exclude))),
            "Invalid execution position",
        )
        require(
            type(record["source_plan"]) is int
            and 0
            <= record["source_plan"]
            < math.ceil(layouts[record["layout"]]["pool_blocks"] / record["blocks"]),
            "Invalid source rotation",
        )
        require(
            type(record["us"]) in (float, int)
            and math.isfinite(record["us"])
            and record["us"] > 0,
            "Invalid duration",
        )
        pair = (
            record["direction"],
            record["layout"],
            record["blocks"],
            record["round"],
        )
        require(record["variant"] not in pairs[pair], "Duplicate sample")
        pairs[pair][record["variant"]] = record
    require(
        set(pairs) == set(itertools.product(DIRECTIONS, TILES, BLOCKS, range(rounds))),
        "Missing sample round",
    )
    for pair, values in pairs.items():
        require(
            set(values) == set(active_variants(pair[0], exclude)),
            "Incomplete paired round",
        )
        require(
            {r["position"] for r in values.values()} == set(range(len(values))),
            "Duplicate execution position",
        )
        require(
            len({r["source_plan"] for r in values.values()}) == 1,
            "Unpaired source blocks",
        )
    return meta, pairs


def summarize(values):
    values = sorted(values)
    return {
        "count": len(values),
        "p50_us": statistics.median(values),
        "p95_us": values[math.ceil(0.95 * len(values)) - 1],
        "min_us": values[0],
        "max_us": values[-1],
    }


def analyze(paths, settings):
    require(len(paths) == 2, "Exactly two process repeats required")
    inputs, cases, warnings = [], collections.defaultdict(list), []
    for repeat, path in zip((80, 81), paths):
        meta, pairs = read_run(path, settings, repeat)
        inputs.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "metadata": meta,
            }
        )
        for (direction, layout, n, round_id), values in pairs.items():
            cases[direction, layout, n].append((repeat, round_id, values))
    for key in ("gpu_uuid", "gpu", "sm", "runtime", "driver", "l2_bytes"):
        require(
            inputs[0]["metadata"][key] == inputs[1]["metadata"][key],
            f"Repeat mismatch: {key}",
        )
    output = []
    for (direction, layout, n), rows in sorted(cases.items()):
        case = {
            "direction": direction,
            "layout": layout,
            "blocks": n,
            "variants": {},
            "comparisons": {},
        }
        for variant in active_variants(direction, settings["exclude_1d_h2d"]):
            stats = summarize([values[variant]["us"] for _, _, values in rows])
            stats["by_repeat"] = {}
            for repeat in (80, 81):
                selected = [
                    (round_id, values)
                    for rep, round_id, values in rows
                    if rep == repeat
                ]
                stats["by_repeat"][repeat] = summarize(
                    [values[variant]["us"] for _, values in selected]
                )
                if settings["iterations"] >= 4:
                    quarters = [
                        statistics.median(
                            values[variant]["us"]
                            for round_id, values in selected
                            if round_id * 4 // settings["iterations"] == quarter
                        )
                        for quarter in range(4)
                    ]
                    stats["by_repeat"][repeat]["quarter_p50_us"] = quarters
                    if max(quarters) / min(quarters) > 1.10:
                        warnings.append(
                            {
                                "direction": direction,
                                "layout": layout,
                                "blocks": n,
                                "variant": variant,
                                "repeat": repeat,
                                "reason": "quarter p50 drift exceeds 10%",
                            }
                        )
            stats["by_position"] = {}
            for position in range(
                len(active_variants(direction, settings["exclude_1d_h2d"]))
            ):
                values = [
                    v[variant]["us"]
                    for _, _, v in rows
                    if v[variant]["position"] == position
                ]
                stats["by_position"][position] = (
                    summarize(values) if values else {"count": 0}
                )
            for label, groups in (
                ("repeat", stats["by_repeat"]),
                ("position", stats["by_position"]),
            ):
                medians = [
                    s["p50_us"]
                    for s in groups.values()
                    if s["count"] >= (10 if label == "position" else 1)
                ]
                if medians and max(medians) / min(medians) > 1.10:
                    warnings.append(
                        {
                            "direction": direction,
                            "layout": layout,
                            "blocks": n,
                            "variant": variant,
                            "reason": label + " p50 difference exceeds 10%",
                        }
                    )
            stats["payload_GBps_at_p50"] = (
                n * sum(TILES[layout]) / stats["p50_us"] / 1000
            )
            case["variants"][variant] = stats
        for baseline in case["variants"]:
            if baseline == "integrated_crc":
                continue
            deltas = [v["integrated_crc"]["us"] - v[baseline]["us"] for _, _, v in rows]
            ratios = [
                100 * (v["integrated_crc"]["us"] / v[baseline]["us"] - 1)
                for _, _, v in rows
            ]
            case["comparisons"][baseline] = {
                "paired_median_delta_us": statistics.median(deltas),
                "paired_median_delta_percent": statistics.median(ratios),
            }
        output.append(case)
    return {
        "valid": True,
        "settings": settings,
        "inputs": inputs,
        "sample_count": sum(len(v) for rows in cases.values() for _, _, v in rows),
        "paired_round_count": sum(map(len, cases.values())),
        "cases": output,
        "warnings": warnings,
        "notes": [
            "BS is local backing count per GPU, not request batch or an eight-rank sum.",
            "All samples retained. p50 is median; p95 is nearest rank.",
            "Negative paired CRC delta means CRC is faster than the named baseline.",
            "8x L2 eviction is outside timing; no hardware cache-miss guarantee.",
            "main_staged includes CPU packing; gather_control is a separate no-CRC diagnostic control.",
            "Explicit exclusions are unavailable, not passing tests or zero latency.",
        ],
    }


def write_results(result, output):
    output = Path(output)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "Complete synchronous call latency, p50 / p95 (us).",
        "",
        "| Direction | Layout | BS | CRC | 1D batch | 3D batch | main staged | gather control, no CRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    with (output / "latency.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "direction",
                "layout",
                "bs",
                "variant",
                "status",
                "samples",
                "p50_us",
                "p95_us",
                "payload_GBps_at_p50",
            ]
        )
        for case in result["cases"]:
            cells = []
            for variant in VARIANTS:
                stats = case["variants"].get(variant)
                cells.append(
                    f"{stats['p50_us']:.3f} / {stats['p95_us']:.3f}"
                    if stats
                    else "N/A (explicitly excluded)"
                )
                writer.writerow(
                    [
                        case["direction"],
                        case["layout"],
                        case["blocks"],
                        variant,
                        "measured" if stats else "explicitly_excluded",
                        *(
                            [
                                stats[k]
                                for k in (
                                    "count",
                                    "p50_us",
                                    "p95_us",
                                    "payload_GBps_at_p50",
                                )
                            ]
                            if stats
                            else [0, "", "", ""]
                        ),
                    ]
                )
            lines.append(
                "| "
                + " | ".join(
                    [case["direction"], case["layout"], str(case["blocks"]), *cells]
                )
                + " |"
            )
    lines.extend(
        ["", *result["notes"], "", f"Drift warnings: {len(result['warnings'])}"]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
