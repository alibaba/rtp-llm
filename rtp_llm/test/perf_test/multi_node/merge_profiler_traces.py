#!/usr/bin/env python3
"""
Merge per-rank profiler timeline JSON files produced by RTP-LLM into a single
Chrome Trace JSON file for visualization in chrome://tracing or Perfetto.

Usage:
    python merge_profiler_traces.py <output_dir> [--suffix _1.json] [--output merged.json]

Example:
    python merge_profiler_traces.py TEST_OUTPUT_QWEN35_MOE_1_4_0_20260424_033740/
    python merge_profiler_traces.py TEST_OUTPUT_... --suffix _2.json --output my_trace.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def get_rank_of_path(p: Path) -> int:
    """Extract world-rank from filename like profiler_wr3_ts..._1.json"""
    m = re.search(r"profiler_wr(\d+)_", p.name)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract rank from filename: {p.name}")


def process_events(events, rank: int):
    """Prefix pid with [WR{rank:02d}] and adjust sort_index for ordering."""
    for e in events:
        if e.get("name") == "process_sort_index":
            pid = e.get("pid")
            try:
                pid_int = int(pid)
                if pid_int < 1000:
                    e["args"]["sort_index"] = 100 * rank + pid_int
            except (ValueError, TypeError):
                pass
        # Tag each event's pid with rank prefix so different ranks don't overlap
        e["pid"] = f"[WR{rank:02d}] {e['pid']}"
    return events


def merge_traces(json_files, output_path: Path):
    merged = {"traceEvents": []}

    for path in json_files:
        rank = get_rank_of_path(path)
        print(f"  Loading rank={rank}: {path.name} ({path.stat().st_size // 1024}KB)")
        with open(path, "r", encoding="utf-8") as f:
            trace = json.load(f)

        events = trace.get("traceEvents", [])
        print(f"    events: {len(events)}")
        processed = process_events(events, rank)
        merged["traceEvents"].extend(processed)

        # Copy top-level metadata (schemaVersion, deviceProperties, etc.) once
        for key, value in trace.items():
            if key != "traceEvents" and key not in merged:
                merged[key] = value

    print(f"\nTotal traceEvents: {len(merged['traceEvents'])}")
    print(f"Writing merged trace to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f)
    print(f"Done. File size: {output_path.stat().st_size // 1024}KB")


def main():
    parser = argparse.ArgumentParser(description="Merge RTP-LLM per-rank profiler JSON traces")
    parser.add_argument("output_dir", help="Directory containing profiler_wrN_ts..._*.json files")
    parser.add_argument(
        "--suffix",
        default="_1.json",
        help="File suffix to match (default: _1.json). Use _2.json for second session.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: <output_dir>/merged_trace<suffix>)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_dir():
        print(f"ERROR: {out_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    pattern = re.compile(r"^profiler_wr\d+_ts\d+.*" + re.escape(args.suffix) + r"$")
    json_files = sorted(
        [p for p in out_dir.iterdir() if pattern.match(p.name)],
        key=get_rank_of_path,
    )

    if not json_files:
        print(f"ERROR: No files matching profiler_wrN_ts*{args.suffix} in {out_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} rank file(s) with suffix '{args.suffix}':")
    for f in json_files:
        print(f"  {f.name}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = out_dir / f"merged_trace{args.suffix}"

    merge_traces(json_files, output_path)


if __name__ == "__main__":
    main()
