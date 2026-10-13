"""Reject incomplete CRC segmentation A/B evidence before calculating latency."""

import collections
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
from pathlib import Path

PROTOCOL = "crc_segmentation_ab_v1"
VARIANTS = ("crc_whole", "crc_segmented")
DIRECTIONS = ("d2h", "h2d")
BLOCKS = tuple(range(1, 33))
CHECK_BLOCKS = (1, 8, 16, 32)
REPEAT_IDS = (80, 81)
SEGMENT_BYTES = 16384
BOUNDARY = (
    "complete synchronous store/load: descriptors, checks, locks, metadata H2D, "
    "gather/scatter, payload DMA, nvCOMP, finish or combine, synchronization and verdict; "
    "policy setter and oracle excluded"
)
# Independent shape oracle for Flash-0731, TP4/CP4, without speculation.
RATIO4_LAYERS = list(range(2, 43, 2))
RATIO128_LAYERS = list(range(3, 43, 2))
GROUPS = {
    "full": [
        ("csa_kv", 19008, RATIO4_LAYERS),
        ("indexer_kv", 4224, RATIO4_LAYERS),
        ("hca_kv", 1152, RATIO128_LAYERS),
    ],
    "prefill_cp4_no_spec_swa": [
        ("swa_kv", 18720, list(range(43))),
        ("indexer_state", 4096, RATIO4_LAYERS),
        ("csa_state", 16384, RATIO4_LAYERS),
    ],
}
TILES = {
    layout: [width for _, width, layers in groups for _ in layers]
    for layout, groups in GROUPS.items()
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(actual, expected):
    """Do not let JSON booleans stand in for counts or success integers."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            exact(actual[k], value) for k, value in expected.items()
        )
    return actual == expected


def expected_geometry(layout):
    result, offset = [], 0
    for member, (tag, width, layers) in enumerate(GROUPS[layout]):
        for local_layer, model_layer in enumerate(layers):
            result.append(
                dict(
                    tag=tag,
                    member=member,
                    local_layer=local_layer,
                    model_layer=model_layer,
                    bytes=width,
                    offset_per_pool_block=offset,
                )
            )
            offset += width
    return result


def validate_settings(settings):
    for field, minimum in (("iterations", 1), ("warmup", 0)):
        value = settings.get(field)
        require(type(value) is int and value >= minimum, f"Invalid setting: {field}")
    seed = settings.get("seed")
    require(type(seed) is int and 0 <= seed <= 2**31 - 82, "Invalid setting: seed")
    require(type(settings.get("correctness_only")) is bool, "Invalid correctness_only")
    require(
        exact(settings.get("repeats", 2), 2), "Exactly two process repeats required"
    )


def block_key(record):
    require(
        isinstance(record.get("layout"), str) and record["layout"] in TILES,
        "Unknown layout",
    )
    require(
        type(record.get("blocks")) is int and record["blocks"] in BLOCKS,
        "Invalid backing count",
    )
    return record["layout"], record["blocks"]


def case_key(record):
    layout, blocks = block_key(record)
    require(record.get("direction") in DIRECTIONS, "Unknown direction")
    require(record.get("variant") in VARIANTS, "Unknown CRC variant")
    return record["direction"], layout, blocks, record["variant"]


def validate_crc_messages(record):
    messages = record["blocks"]
    if record["variant"] == "crc_segmented":
        messages *= math.ceil(sum(TILES[record["layout"]]) / SEGMENT_BYTES)
    require(
        exact(record.get("crc_messages"), messages),
        "CRC message count does not match the forced mode",
    )


def read_run(path, settings, repeat):
    validate_settings(settings)
    try:
        records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSONL: {path}: {error}") from error
    require(all(isinstance(r, dict) for r in records), "Records must be JSON objects")
    require(all(isinstance(r.get("type"), str) for r in records), "Missing record type")
    require(
        records
        and records[0].get("type") == "metadata"
        and exact(records[-1], {"type": "complete", "success": True}),
        f"Incomplete benchmark: {path}",
    )
    meta = records[0]
    expected_meta = {
        "protocol": PROTOCOL,
        "variants": list(VARIANTS),
        "repeat": repeat,
        "seed": settings["seed"] + repeat,
        "iterations": settings["iterations"],
        "warmup": settings["warmup"],
        "correctness_only": settings["correctness_only"],
        "max_backings": 32,
        "timing": "wall",
        "regime": "cold",
        "measured_h2d_verified": True,
        "shared_workspace": True,
        "forced_mode_outside_timer": True,
        "profiled": False,
        "fallback_allowed": False,
        "evict_multiplier": 8,
        "block_counts": list(BLOCKS),
        "segment_bytes": SEGMENT_BYTES,
        "boundary": BOUNDARY,
        "model": "DeepSeek-V4-Flash-0731",
        "target_layers": 43,
        "kv_cache_sharded": True,
        "seq_size_per_block": 128,
        "kernel_tokens_per_block": 128,
        "cp_size": 4,
        "tp_size": 4,
        "cp_mode": "CP_RR",
        "gen_num_per_cycle": 0,
        "local_backing_count": True,
        "shape_source": "CacheConfigCreator+DeviceBlockPoolConfigHelper",
    }
    for field, expected in expected_meta.items():
        require(exact(meta.get(field), expected), f"Metadata mismatch: {field}")
    for field, length in (
        ("source_commit", 40),
        ("base_commit", 40),
        ("binary_sha256", 64),
    ):
        require(
            isinstance(meta.get(field), str)
            and re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", meta[field])
            is not None,
            f"Missing or invalid provenance: {field}",
        )
        if field in settings:
            require(meta[field] == settings[field], f"Provenance mismatch: {field}")
    for field in ("l2_bytes", "driver", "runtime", "sm"):
        require(type(meta.get(field)) is int and meta[field] > 0, f"Invalid {field}")
    require(
        isinstance(meta.get("gpu_name"), str) and bool(meta["gpu_name"].strip()),
        "Missing GPU name",
    )
    require(
        isinstance(meta.get("gpu_uuid"), str) and bool(meta["gpu_uuid"].strip()),
        "Missing GPU UUID",
    )
    if "equal_workspace_bytes" in meta:
        require(
            meta["equal_workspace_bytes"] is True,
            "Metadata mismatch: equal_workspace_bytes",
        )

    matrix = set(itertools.product(DIRECTIONS, TILES, BLOCKS, VARIANTS))
    rounds = 0 if settings["correctness_only"] else settings["iterations"]
    expected_counts = {
        "metadata": 1,
        "layout": 2,
        "mixed_crc_selftest": 2,
        "correctness": len(matrix),
        "crc_corruption": len(TILES) * len(CHECK_BLOCKS) * len(VARIANTS),
        "cross_mode": len(TILES) * len(CHECK_BLOCKS),
        "sample": len(matrix) * rounds,
        "complete": 1,
    }
    counts = collections.Counter(r.get("type") for r in records)
    require(counts == collections.Counter(expected_counts), f"Record counts: {counts}")
    mixed = [r for r in records if r["type"] == "mixed_crc_selftest"]
    require(
        collections.Counter(r.get("variant") for r in mixed)
        == collections.Counter(VARIANTS),
        "Mixed CRC self-test matrix",
    )
    require(all(r.get("success") is True for r in mixed), "Mixed CRC self-test failed")
    layouts = {r.get("layout"): r for r in records if r["type"] == "layout"}
    require(set(layouts) == set(TILES), "Layout coverage")
    for layout, tiles in TILES.items():
        shape = layouts[layout]
        payload = sum(tiles)
        encoded = (payload + 19) // 16 * 16
        expected = {
            "geometry": expected_geometry(layout),
            "tiles": len(tiles),
            "payload_bytes": payload,
            "encoded_bytes": encoded,
            "host_stride": encoded,
        }
        for field, value in expected.items():
            require(exact(shape.get(field), value), f"Invalid {layout} shape: {field}")
        if "tile_bytes" in shape:
            require(exact(shape["tile_bytes"], tiles), "Invalid tile_bytes")
        for field in (
            "source_bytes",
            "pool_blocks",
            "evict_bytes",
            "staging_stride",
            "workspace_device_bytes",
        ):
            require(
                type(shape.get(field)) is int and shape[field] > 0, f"Invalid {field}"
            )
        require(
            shape["pool_blocks"] >= 129
            and shape["evict_bytes"] % 16 == 0
            and shape["staging_stride"] >= encoded
            and shape["staging_stride"] % 16 == 0,
            "Insufficient pool or invalid buffer alignment",
        )
        require(
            shape["source_bytes"] == shape["pool_blocks"] * payload,
            "Source pool capacity mismatch",
        )
        require(
            (shape["pool_blocks"] - 1) * payload >= 8 * meta["l2_bytes"]
            and shape["evict_bytes"] >= 8 * meta["l2_bytes"],
            "Pool/eviction buffer smaller than 8x L2",
        )
        require(
            shape["workspace_device_bytes"] >= 32 * shape["staging_stride"],
            "Workspace cannot hold the maximum staging batch",
        )
        if "equal_workspace_bytes" in shape:
            require(shape["equal_workspace_bytes"] is True, "Unequal workspace bytes")
        require(
            meta.get("equal_workspace_bytes") is True
            or shape.get("equal_workspace_bytes") is True,
            "Missing equal_workspace_bytes guarantee",
        )
        if "reserved_sentinel_blocks" in shape:
            require(
                exact(shape["reserved_sentinel_blocks"], 1), "Missing pool sentinel"
            )

    checks = [r for r in records if r["type"] == "correctness"]
    require(
        collections.Counter(map(case_key, checks)) == collections.Counter(matrix),
        "Correctness matrix",
    )
    for record in checks:
        validate_crc_messages(record)
        require(
            record.get("success") is True
            and exact(record.get("rotations_checked"), [0, 1]),
            "Correctness check did not pass",
        )
    corruption = [r for r in records if r["type"] == "crc_corruption"]
    require(
        collections.Counter(map(case_key, corruption))
        == collections.Counter(
            itertools.product(("h2d",), TILES, CHECK_BLOCKS, VARIANTS)
        ),
        "Corruption matrix",
    )
    for record in corruption:
        require(exact(record.get("cases"), 6), "Invalid corruption case count")
        for field in ("success", "all_targets_unchanged", "recovery_success"):
            require(record.get(field) is True, f"CRC check failed: {field}")
        for field in ("good_bad_good", "entire_pool_guards"):
            if field in record:
                require(record[field] is True, f"CRC check failed: {field}")
    cross_mode = [r for r in records if r["type"] == "cross_mode"]
    require(
        collections.Counter(map(block_key, cross_mode))
        == collections.Counter(itertools.product(TILES, CHECK_BLOCKS)),
        "Cross-mode matrix",
    )
    for record in cross_mode:
        for field in (
            "whole_store_segmented_load",
            "segmented_store_whole_load",
            "footer_equal",
        ):
            require(record.get(field) is True, f"Cross-mode check failed: {field}")
        if "success" in record:
            require(record["success"] is True, "Cross-mode check failed: success")

    pairs = collections.defaultdict(dict)
    for record in records:
        if record["type"] != "sample":
            continue
        require(case_key(record) in matrix, "Unexpected sample")
        validate_crc_messages(record)
        require(exact(record.get("repeat"), repeat), "Sample from wrong repeat")
        require(
            record.get("timing") == "wall" and record.get("regime") == "cold",
            "Wrong timing method",
        )
        require(
            type(record.get("round")) is int and 0 <= record["round"] < rounds,
            "Invalid round",
        )
        require(
            type(record.get("position")) is int and record["position"] in (0, 1),
            "Invalid execution position",
        )
        require(
            type(record.get("source_plan")) is int
            and 0
            <= record["source_plan"]
            < math.ceil(
                (layouts[record["layout"]]["pool_blocks"] - 1) / record["blocks"]
            ),
            "Invalid source rotation",
        )
        require(
            type(record.get("us")) in (float, int)
            and math.isfinite(record["us"])
            and record["us"] > 0,
            "Invalid duration",
        )
        pair = record["direction"], record["layout"], record["blocks"], record["round"]
        require(record["variant"] not in pairs[pair], "Duplicate sample")
        pairs[pair][record["variant"]] = record
    require(
        set(pairs) == set(itertools.product(DIRECTIONS, TILES, BLOCKS, range(rounds))),
        "Missing sample round",
    )
    for values in pairs.values():
        require(set(values) == set(VARIANTS), "Incomplete paired round")
        require(
            {r["position"] for r in values.values()} == {0, 1},
            "Duplicate execution position",
        )
        require(
            len({r["source_plan"] for r in values.values()}) == 1,
            "Unpaired source blocks",
        )
    return meta, layouts, pairs


def summarize(values):
    values = sorted(values)
    return {
        "count": len(values),
        "p50_us": statistics.median(values),
        "p95_us": values[math.ceil(0.95 * len(values)) - 1],
        "min_us": values[0],
        "max_us": values[-1],
    }


def paired_summary(rows):
    result = summarize([v["crc_whole"]["us"] - v["crc_segmented"]["us"] for v in rows])
    result["median_percent_of_whole"] = statistics.median(
        100 * (v["crc_whole"]["us"] - v["crc_segmented"]["us"]) / v["crc_whole"]["us"]
        for v in rows
    )
    return result


def analyze(paths, settings):
    validate_settings(settings)
    require(len(paths) == 2, "Exactly two process repeats required")
    require(len({Path(path).resolve() for path in paths}) == 2, "Repeated input file")
    inputs, cases, warnings = [], collections.defaultdict(list), []
    for repeat, path in zip(REPEAT_IDS, paths):
        meta, layouts, pairs = read_run(path, settings, repeat)
        inputs.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "metadata": meta,
                "layouts": layouts,
            }
        )
        for (direction, layout, blocks, round_id), values in pairs.items():
            cases[direction, layout, blocks].append((repeat, round_id, values))
    first, second = inputs
    for field in set(first["metadata"]) | set(second["metadata"]):
        if field not in ("seed", "repeat"):
            require(
                exact(first["metadata"].get(field), second["metadata"].get(field)),
                f"Repeat mismatch: {field}",
            )
    require(
        exact(first["layouts"], second["layouts"]), "Repeat mismatch: layout/workspace"
    )
    output = []
    for (direction, layout, blocks), rows in sorted(cases.items()):
        case = dict(direction=direction, layout=layout, blocks=blocks, variants={})
        for variant in VARIANTS:
            stats = summarize([values[variant]["us"] for _, _, values in rows])
            stats["by_repeat"] = {}
            for repeat in REPEAT_IDS:
                selected = [(rid, values) for rep, rid, values in rows if rep == repeat]
                current = summarize([values[variant]["us"] for _, values in selected])
                if settings["iterations"] >= 4:
                    quarters = [
                        statistics.median(
                            values[variant]["us"]
                            for rid, values in selected
                            if rid * 4 // settings["iterations"] == quarter
                        )
                        for quarter in range(4)
                    ]
                    current["quarter_p50_us"] = quarters
                    if max(quarters) / min(quarters) > 1.10:
                        warnings.append(
                            dict(
                                direction=direction,
                                layout=layout,
                                blocks=blocks,
                                variant=variant,
                                repeat=repeat,
                                reason="quarter p50 drift exceeds 10%",
                            )
                        )
                stats["by_repeat"][repeat] = current
            stats["by_position"] = {}
            for position in (0, 1):
                values = [
                    v[variant]["us"]
                    for _, _, v in rows
                    if v[variant]["position"] == position
                ]
                stats["by_position"][position] = (
                    summarize(values) if values else {"count": 0}
                )
            for label in ("repeat", "position"):
                medians = [
                    entry["p50_us"]
                    for entry in stats["by_" + label].values()
                    if entry["count"] >= (10 if label == "position" else 1)
                ]
                if medians and max(medians) / min(medians) > 1.10:
                    warnings.append(
                        dict(
                            direction=direction,
                            layout=layout,
                            blocks=blocks,
                            variant=variant,
                            reason=label + " p50 difference exceeds 10%",
                        )
                    )
            stats["payload_GBps_at_p50"] = (
                blocks * sum(TILES[layout]) / stats["p50_us"] / 1000
            )
            case["variants"][variant] = stats
        delta = paired_summary([values for _, _, values in rows])
        delta["by_repeat"] = {
            repeat: paired_summary([v for rep, _, v in rows if rep == repeat])
            for repeat in REPEAT_IDS
        }
        for repeat in REPEAT_IDS:
            delta["by_repeat"][repeat]["by_whole_position"] = {}
            for position in (0, 1):
                selected = [
                    v
                    for rep, _, v in rows
                    if rep == repeat and v["crc_whole"]["position"] == position
                ]
                delta["by_repeat"][repeat]["by_whole_position"][position] = (
                    paired_summary(selected) if selected else {"count": 0}
                )
        delta["by_whole_position"] = {}
        for position in (0, 1):
            selected = [v for _, _, v in rows if v["crc_whole"]["position"] == position]
            delta["by_whole_position"][position] = (
                paired_summary(selected) if selected else {"count": 0}
            )
        case["paired_whole_minus_segmented"] = delta
        output.append(case)
    return {
        "valid": True,
        "protocol": PROTOCOL,
        "settings": settings,
        "inputs": inputs,
        "sample_count": sum(len(v) for rows in cases.values() for _, _, v in rows),
        "paired_round_count": sum(map(len, cases.values())),
        "cases": output,
        "warnings": warnings,
        "notes": [
            "BS is the local backing count per GPU.",
            "Latency covers the synchronous call; the forced CRC mode is selected outside timing.",
            "Both modes share the same workspace allocation, source plan, and cold-cache procedure.",
            "All samples are retained. p50 is the median; p95 uses nearest rank.",
            "Paired delta is whole minus segmented for each round; positive values favor segmented CRC.",
            "The delta measures the two forced implementations. It does not measure a production threshold decision.",
            "8x L2 eviction is outside timing and does not guarantee a hardware cache miss.",
        ],
    }


def write_results(result, output):
    output = Path(output)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    meta = result["inputs"][0]["metadata"]
    lines = [
        f"Source: `{meta['source_commit']}`; base: `{meta['base_commit']}`.",
        f"GPU: {meta['gpu_name']} (SM{meta['sm']}); CUDA runtime: {meta['runtime']}.",
        f"Validated {result['sample_count']} samples and {result['paired_round_count']} paired rounds across two processes.",
        "",
    ]
    if result["settings"]["correctness_only"]:
        lines.extend(
            [
                "Correctness checks passed. This run contains no latency measurements.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Synchronous call p50 / p95 in microseconds. Paired delta is whole minus segmented.",
                "",
                "| Direction | Layout | BS | Whole p50 / p95 | Whole p50 R80 / R81 | Segmented p50 / p95 | Segmented p50 R80 / R81 | Paired delta p50 / p95 | Delta % of whole |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    with (output / "latency.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "direction",
                "layout",
                "blocks",
                "variant",
                "samples",
                "p50_us",
                "p95_us",
                "payload_GBps_at_p50",
                "repeat_80_p50_us",
                "repeat_80_p95_us",
                "repeat_81_p50_us",
                "repeat_81_p95_us",
            ]
        )
        for case in result["cases"]:
            cells = []
            for variant in VARIANTS:
                stats = case["variants"][variant]
                cells.append(f"{stats['p50_us']:.3f} / {stats['p95_us']:.3f}")
                cells.append(
                    " / ".join(
                        f"{stats['by_repeat'][repeat]['p50_us']:.3f}"
                        for repeat in REPEAT_IDS
                    )
                )
                writer.writerow(
                    [
                        case["direction"],
                        case["layout"],
                        case["blocks"],
                        variant,
                        *[
                            stats[key]
                            for key in (
                                "count",
                                "p50_us",
                                "p95_us",
                                "payload_GBps_at_p50",
                            )
                        ],
                        *[
                            stats["by_repeat"][repeat][key]
                            for repeat in REPEAT_IDS
                            for key in ("p50_us", "p95_us")
                        ],
                    ]
                )
            delta = case["paired_whole_minus_segmented"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        case["direction"],
                        case["layout"],
                        str(case["blocks"]),
                        *cells,
                        f"{delta['p50_us']:.3f} / {delta['p95_us']:.3f}",
                        f"{delta['median_percent_of_whole']:.2f}%",
                    ]
                )
                + " |"
            )
    with (output / "paired_delta.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "direction",
                "layout",
                "blocks",
                "paired_rounds",
                "p50_whole_minus_segmented_us",
                "p95_whole_minus_segmented_us",
                "median_percent_of_whole",
            ]
        )
        for case in result["cases"]:
            delta = case["paired_whole_minus_segmented"]
            writer.writerow(
                [
                    case["direction"],
                    case["layout"],
                    case["blocks"],
                    *[
                        delta[key]
                        for key in (
                            "count",
                            "p50_us",
                            "p95_us",
                            "median_percent_of_whole",
                        )
                    ],
                ]
            )
    lines.extend(
        ["", *result["notes"], "", f"Drift warnings: {len(result['warnings'])}."]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
