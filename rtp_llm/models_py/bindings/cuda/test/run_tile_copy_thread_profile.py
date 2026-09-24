"""Manual cold-cache thread sweep for the shared production tile-copy kernel."""

import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from run_crc_copy_benchmark import select_cpu

THREADS = (32, 64, 128, 256, 512, 1024)
BLOCKS = (1, 2, 4, 8, 16, 32)
LAYOUTS = {"full": (732672, 91), "prefill_cp8_no_spec_swa": (878160, 121)}
PACKINGS = ("crc_records", "main_staged")
DIRECTIONS = ("gather", "scatter")
FIELDS = ("layout", "packing", "direction", "blocks", "threads")
MATRIX = set(itertools.product(LAYOUTS, PACKINGS, DIRECTIONS, BLOCKS, THREADS))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(row):
    return tuple(row[field] for field in FIELDS)


def validate(records, settings, repeat):
    require(
        records and records[-1] == {"type": "complete", "success": True},
        "Incomplete profile",
    )
    meta = records[0]
    expected = {
        "type": "metadata",
        "implementation": "tile_copy_thread_profile_v1",
        "seed": settings["seed"] + repeat,
        "repeat": repeat,
        "iterations": settings["iterations"],
        "warmup": settings["warmup"],
        "correctness_only": settings["correctness_only"],
        "threads": list(THREADS),
        "blocks": list(BLOCKS),
        "evict_multiplier": 8,
        "timing": "cuda_event_stream_latency",
        "regime": "cold",
        "seq_size_per_block": 128,
        "cp_size": 8,
        "tp_size": 8,
        "cp_mode": "CP_RR",
        "profile_capture": False,
        "measured_copy_launches_per_round": 6,
        "scatter_setup_copy_launches_per_round": 1,
    }
    for field, value in expected.items():
        require(
            type(meta.get(field)) is type(value) and meta[field] == value,
            f"Metadata mismatch: {field}",
        )
    require(
        meta["l2_bytes"] > 0 and meta["production_default_threads"] in THREADS,
        "Invalid GPU/default metadata",
    )
    rounds = 0 if settings["correctness_only"] else settings["iterations"]
    counts = collections.Counter(row["type"] for row in records)
    require(
        counts
        == collections.Counter(
            metadata=1,
            layout=2,
            correctness=len(MATRIX),
            sample=len(MATRIX) * rounds,
            complete=1,
        ),
        "Record count mismatch",
    )
    shapes = [row for row in records if row["type"] == "layout"]
    require({row["layout"] for row in shapes} == set(LAYOUTS), "Missing layout")
    for row in shapes:
        require(
            (row["payload_bytes"], row["tiles"]) == LAYOUTS[row["layout"]],
            "Wrong DeepSeek shape",
        )
        require(
            row["crc_staging_stride"] == 878176,
            "CRC stride must use production maximum payload",
        )
        require(
            row["source_bytes"] >= 8 * meta["l2_bytes"]
            and row["evict_bytes"] >= 8 * meta["l2_bytes"],
            "Insufficient L2 eviction",
        )
    checks = [row for row in records if row["type"] == "correctness"]
    require({key(row) for row in checks} == MATRIX, "Correctness coverage mismatch")
    require(
        all(
            row["success"] is True
            and row["full_payload_and_guards"] is True
            and row["rotations_checked"] == [0, 1]
            for row in checks
        ),
        "Failed payload/guard verification",
    )
    samples = [row for row in records if row["type"] == "sample"]
    identities = {(key(row), row["round"]) for row in samples}
    require(
        identities == set(itertools.product(MATRIX, range(rounds))),
        "Duplicate/missing samples",
    )
    pairs = collections.defaultdict(list)
    for row in samples:
        require(
            row["repeat"] == repeat
            and type(row["us"]) in (float, int)
            and math.isfinite(row["us"])
            and row["us"] > 0,
            "Invalid sample",
        )
        pairs[(key(row)[:-1], row["round"])].append(row)
    for rows in pairs.values():
        require(
            {row["position"] for row in rows} == set(range(len(THREADS)))
            and len({row["source_plan"] for row in rows}) == 1,
            "Unpaired candidate round",
        )
    return meta, samples


def summarize(samples):
    grouped = collections.defaultdict(list)
    paired = {}
    for row in samples:
        grouped[key(row)].append(row)
        paired[(key(row), row["repeat"], row["round"])] = row["us"]
    summaries = []
    for case, rows in sorted(grouped.items()):
        values = sorted(row["us"] for row in rows)
        ratios = [
            row["us"] / paired[(case[:-1] + (256,), row["repeat"], row["round"])]
            for row in rows
        ]
        result = dict(zip(FIELDS, case))
        result.update(
            samples=len(values),
            median_us=statistics.median(values),
            p95_us=values[math.ceil(len(values) * 0.95) - 1],
            paired_ratio_to_256=statistics.median(ratios),
            repeat_80_median_us=statistics.median(
                row["us"] for row in rows if row["repeat"] == 80
            ),
            repeat_81_median_us=statistics.median(
                row["us"] for row in rows if row["repeat"] == 81
            ),
        )
        summaries.append(result)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--cpu", default="auto")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "TEST_UNDECLARED_OUTPUTS_DIR", "tile_copy_thread_profile_results"
            )
        ),
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or not 0 <= args.seed <= 2**31 - 82:
        parser.error(
            "iterations must be positive, warmup nonnegative, seed in [0, 2147483566]"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Output directory must be empty")
    settings = {
        field: getattr(args, field)
        for field in ("iterations", "warmup", "seed", "correctness_only")
    }
    manifest = {
        "success": False,
        "settings": settings,
        "command": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        binary = args.binary.resolve(strict=True)
        manifest.update(
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            cpu=select_cpu(args.cpu),
        )
        print(f"Shared tile profile artifacts: {output}", flush=True)
        samples, metadata = [], []
        for repeat in (80, 81):
            path = output / f"threads_{repeat}.jsonl"
            command = [
                str(binary),
                "--tile-thread-profile",
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(repeat),
                "--seed",
                str(args.seed + repeat),
                "--output",
                str(path),
            ]
            if args.correctness_only:
                command.append("--correctness-only")
            with (output / f"threads_{repeat}.stdout").open("w") as stdout, (
                output / f"threads_{repeat}.stderr"
            ).open("w") as stderr:
                completed = subprocess.run(
                    command, stdout=stdout, stderr=stderr, check=False
                )
            (output / f"threads_{repeat}.exitcode").write_text(
                str(completed.returncode) + "\n"
            )
            require(
                completed.returncode == 0,
                f"Profile repeat {repeat} failed; raw evidence retained",
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]
            meta, rows = validate(records, settings, repeat)
            metadata.append(meta)
            samples.extend(rows)
            print(f"repeat {repeat}: {len(rows)} validated samples", flush=True)
        summaries = summarize(samples)
        result = {
            "metadata": metadata,
            "sample_count": len(samples),
            "summary": summaries,
        }
        (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
        if summaries:
            with (output / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
        manifest.update(success=True, sample_count=len(samples), status="passed")
        print(
            f"Thread profile passed: {len(samples)} samples; {output / 'summary.csv'}",
            flush=True,
        )
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        IndexError,
        subprocess.SubprocessError,
    ) as error:
        manifest.update(status="failed", error=str(error))
        print(f"Shared tile profile FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
